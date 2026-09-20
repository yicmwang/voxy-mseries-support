package me.cortex.voxy.client.core.metal;

import me.cortex.voxy.client.core.gpu.*;
import me.cortex.voxy.common.Logger;

import java.util.concurrent.atomic.AtomicLong;

import static org.lwjgl.opengl.GL11C.GL_TEXTURE_2D;

/**
 * Metal rendering backend for macOS Apple Silicon (M-series) support.
 *
 * This backend translates Voxy's rendering operations to Metal API calls
 * via the MetalNative JNI bridge. It manages the Metal device, command queue,
 * and shared event for fence synchronization.
 *
 * Metal architectural differences from OpenGL:
 *   - No global state machine; state is captured in pipeline state objects
 *   - No framebuffer objects; render targets are per-pass via render pass descriptors
 *   - No VAOs; vertex layout is part of the render pipeline state
 *   - Buffers in shared storage are always CPU-visible (unified memory)
 *   - Texture state (filtering, wrapping) is set via sampler objects, not on textures
 *   - Shaders are compiled to MTLLibrary from MSL (not GLSL)
 */
public class MetalRenderBackend implements RenderBackend {

    private final long device;

    /** Native MTLDevice handle. Exposed for ICB/IOSurface callers that bypass the backend's resource creators. */
    public long device() {
        return this.device;
    }
    private final long commandQueue;
    private final long sharedEvent;
    private final AtomicLong fenceCounter = new AtomicLong(1);
    private final long maxBufferLength;
    /** Lazily allocated MTLCommandBuffer for the current frame; 0 when no commands are queued. */
    private long activeCommandBuffer = 0;

    /**
     * Whether the guest branch of {@link #submit()} waits for the frame it just committed. ON by
     * default, because it is the fix for bug 3, not an experiment: the guest branch's own comment
     * claims the work is "committed and complete, making it CPU-visible for the draw path's
     * baseInstance read", and without this wait that claim is false.
     *
     * <p>The failure it fixes: {@code flushFrame()} commits without waiting, and Metallum's encoder
     * awaits three submits BACK from a counter that STARTS at 3, so the first three submits wait for
     * nothing. MetalRenderEncoder then reads {@code baseInstance} out of drawCallBuffer on the CPU,
     * per draw, and pushes it as the per-draw constant; a stale read gives the WRONG SECTION'S ORIGIN
     * while the draw keeps its own quads and their baked light -- bug 3's symptom, lighting included.
     *
     * <p>{@code VOXY_SUBMIT_ORDER=0} turns it off, for a controlled A/B or to measure its cost. Note
     * the polarity is deliberately the default-ON idiom, which is correct here and was wrong for the
     * debug readouts that once defaulted on by accident.
     */
    private static final boolean SUBMIT_ORDER = !"0".equals(System.getenv("VOXY_SUBMIT_ORDER"));
    /** False when the active buffer belongs to Metallum and must not be committed or released. */
    private boolean ownsActiveCommandBuffer = true;
    /** Lazily-opened MTLBlitCommandEncoder on the active buffer for stream copies; 0 when closed. */
    private long activeBlitEncoder = 0;
    /**
     * Last render encoder handed out. A non-zero handle() means a caller still
     * holds the pass open (bake-time MetalBudgetBufferRenderer holds its encoder
     * across UploadStream.commit), so copies/fences must not touch the active
     * buffer — Metal forbids a second encoder while one is open. Compute
     * encoders aren't tracked: every compute pass in the tree is a local
     * try-with-resources with no stream copy/fence calls inside it.
     */
    private MetalRenderEncoder lastRenderEncoder;
    /** True while the active buffer holds encoded-but-uncommitted stream copies. */
    private boolean activeBufferHasBlits = false;
    /**
     * Fence values whose event signal must ride the active buffer but couldn't
     * be encoded at creation time (caller pass open). Encoded in submit() so
     * the fence never signals ahead of the copies it guards.
     */
    private final java.util.ArrayList<Long> pendingFenceSignals = new java.util.ArrayList<>();

    public MetalRenderBackend() {
        if (!MetalNative.load()) {
            throw new RuntimeException("Metal native library is not available");
        }

        // Whole-frame Metal: Metallum owns the device and queue. Fall back to creating our own
        // only when Metallum is absent (the standalone smoke-test path).
        long metallumDevice = MetallumBridge.available() ? MetallumBridge.device() : 0L;
        if (metallumDevice != 0) {
            this.device = metallumDevice;
        } else {
            this.device = MetalNative.mtlCreateSystemDefaultDevice();
        }
        if (this.device == 0) {
            throw new RuntimeException("Failed to create Metal device");
        }

        String deviceName = MetalNative.mtlDeviceGetName(this.device);
        Logger.info("Metal device: " + deviceName);

        long metallumQueue = MetallumBridge.available() ? MetallumBridge.commandQueue() : 0L;
        this.commandQueue = metallumQueue != 0 ? metallumQueue
                : MetalNative.mtlDeviceNewCommandQueue(this.device);
        if (this.commandQueue == 0) {
            throw new RuntimeException("Failed to create Metal command queue");
        }

        this.sharedEvent = MetalNative.mtlDeviceNewSharedEvent(this.device);
        if (this.sharedEvent == 0) {
            throw new RuntimeException("Failed to create Metal shared event");
        }

        this.maxBufferLength = MetalNative.mtlDeviceMaxBufferLength(this.device);
    }

    @Override
    public BackendType getType() {
        return BackendType.METAL;
    }

    // --- Resource Creation ---

    @Override
    public IGpuBuffer createBuffer(long size) {
        return createBuffer(size, 0, true);
    }

    @Override
    public IGpuBuffer createBuffer(long size, int flags) {
        return createBuffer(size, flags, true);
    }

    @Override
    public IGpuBuffer createBuffer(long size, int flags, boolean zero) {
        return new MetalBuffer(this.device, size,
                MetalNative.MTLResourceStorageModeShared, zero);
    }

    @Override
    public IGpuTexture createTexture() {
        return new MetalTexture(this.device, GL_TEXTURE_2D);
    }

    @Override
    public IGpuTexture createTexture(int type) {
        return new MetalTexture(this.device, type);
    }

    @Override
    public IGpuFramebuffer createFramebuffer() {
        return new MetalFramebuffer();
    }

    @Override
    public IGpuRenderBuffer createRenderBuffer(int format, int width, int height) {
        return new MetalRenderBuffer(this.device, format, width, height);
    }

    @Override
    public IGpuVertexArray createVertexArray() {
        return new MetalVertexDescriptor();
    }

    @Override
    public IGpuFence createFence() {
        long targetValue = this.fenceCounter.getAndIncrement();

        if (this.activeCommandBuffer == 0) {
            // Queue idle — every commit path on this queue waits for
            // completion, so all previously requested work is done. CPU-signal
            // directly instead of burning a command buffer on the event.
            MetalNative.mtlSharedEventSetSignaledValue(this.sharedEvent, targetValue);
        } else if (this.callerPassOpen()) {
            // encodeSignalEvent is illegal while an encoder is open. Only the
            // stream-full emergency paths can land here (fence requested while
            // the bakery holds its render pass open).
            if (this.activeBufferHasBlits) {
                // Copies this fence may guard are parked, unexecuted, in the
                // active buffer. A commit-ahead dedicated buffer would signal
                // BEFORE them, letting UploadStream free + reuse staging whose
                // copies haven't run (silent corruption). Defer the signal to
                // submit() instead — the fence stays unsignaled until the
                // copies actually execute, so the emergency loop fails loudly
                // (IllegalStateException) rather than corrupting.
                this.pendingFenceSignals.add(targetValue);
            } else {
                // No parked copies — the dedicated signal-only buffer is safe.
                long cmdBuffer = MetalNative.mtlCommandQueueNewCommandBuffer(this.commandQueue);
                MetalNative.mtlCommandBufferEncodeSignalEvent(cmdBuffer, this.sharedEvent, targetValue);
                MetalNative.mtlCommandBufferCommit(cmdBuffer);
                MetalNative.mtlRelease(cmdBuffer);
            }
        } else {
            // Ride the active buffer: the signal executes at the next submit(),
            // strictly after everything encoded before it — including the
            // stream copies this fence guards (the old dedicated-buffer path
            // signalled AHEAD of the still-uncommitted frame work).
            this.endActiveBlitEncoder();
            MetalNative.mtlCommandBufferEncodeSignalEvent(this.activeCommandBuffer, this.sharedEvent, targetValue);
        }

        return new MetalFence(this.sharedEvent, targetValue);
    }

    @Override
    public IGpuPersistentBuffer createPersistentBuffer(long size, int flags) {
        return new MetalPersistentBuffer(this.device, size, flags);
    }

    // --- Texture Operations ---

    @Override
    public void bindTextureUnit(int unit, int texture) {
        // Metal binds textures per-encoder, not globally. Tracked for draw-time binding.
    }

    @Override
    public void bindTextureUnit(int unit, int target, int texture) {
        // Same — no global texture unit state in Metal
    }

    @Override
    public void textureParameteri(int texture, int pname, int param) {
        // Metal sets filtering/wrapping via MTLSamplerState, not on the texture
    }

    @Override
    public void textureParameteri(int texture, int target, int pname, int param) {
        // Same
    }

    @Override
    public void textureParameterf(int texture, int pname, float param) {
        // Sampler state
    }

    @Override
    public void textureParameterf(int texture, int target, int pname, float param) {
        // Same
    }

    @Override
    public void textureSubImage2D(int texture, int target, int level, int x, int y,
                                   int w, int h, int format, int type, long addr) {
        long texHandle = MetalHandleMap.getHandle(texture);
        int bpp = (int) MetalFormatUtil.bytesPerPixel(format);
        MetalNative.mtlTextureReplaceRegion(texHandle, level, x, y, w, h, addr, w * bpp);
    }

    @Override
    public void textureStorage2D(int texture, int target, int levels, int format, int width, int height) {
        // Metal allocates storage at texture creation time (MetalTexture.store())
    }

    // --- Framebuffer Operations ---

    @Override
    public void framebufferTexture(int fbo, int attachment, int texture, int level, int target) {
        // Handled by MetalFramebuffer.bind() when using the object API
    }

    @Override
    public void framebufferRenderbuffer(int fbo, int attachment, int renderbuffer) {
        // Handled by MetalFramebuffer.bind()
    }

    @Override
    public void framebufferDrawBuffers(int fbo, int... buffers) {
        // Metal configures this via render pass descriptor color attachments
    }

    @Override
    public int checkFramebufferStatus(int fbo) {
        return 0x8CD5; // GL_FRAMEBUFFER_COMPLETE — Metal validates at encoder creation
    }

    @Override
    public void clearDepthFramebuffer(int fbo, float depth) {
        // Deferred to render pass descriptor load action
    }

    @Override
    public void clearDepthStencilFramebuffer(int fbo, float depth, int stencil) {
        // Deferred to render pass descriptor load action
    }

    @Override
    public void blitFramebuffer(int readFbo, int drawFbo, int srcX0, int srcY0, int srcX1, int srcY1,
                                 int dstX0, int dstY0, int dstX1, int dstY1, int mask, int filter) {
        // TODO: Implement via blit command encoder or fullscreen render pass
    }

    // --- Renderbuffer Operations ---

    @Override
    public int createRenderbufferId() {
        return MetalHandleMap.register(1); // Placeholder — actual texture created in createRenderBuffer
    }

    @Override
    public void renderbufferStorage(int renderbuffer, int format, int width, int height) {
        // Metal renderbuffers are textures — storage is allocated at creation time
    }

    // --- Debug ---

    @Override
    public void objectLabel(int type, int id, String name) {
        try {
            long handle = MetalHandleMap.getHandle(id);
            MetalNative.mtlSetLabel(handle, name);
        } catch (IllegalArgumentException e) {
            // Unknown ID — silently ignore for debug labeling
        }
    }

    // --- Capabilities ---

    @Override
    public boolean hasCompute() {
        return true;
    }

    @Override
    public boolean hasIndirectCount() {
        return true;
    }

    @Override
    public boolean hasIndirectParameters() {
        return true;
    }

    @Override
    public boolean hasSparseBuffer() {
        return false;
    }

    @Override
    public long getMaxSSBOSize() {
        return this.maxBufferLength;
    }

    @Override
    public int getStaticVAO() {
        return 0; // Metal doesn't use VAOs
    }

    // --- Resource statistics ---

    @Override
    public int getBufferCount() {
        return MetalBuffer.getCount();
    }

    @Override
    public long getBufferTotalSize() {
        return MetalBuffer.getTotalSize();
    }

    @Override
    public int getTextureCount() {
        return MetalTexture.getCount();
    }

    @Override
    public long getTextureEstimatedTotalSize() {
        return MetalTexture.getEstimatedTotalSize();
    }

    @Override
    public void memoryBarrier(int flags) {
        // Metal performs automatic hazard tracking between encoders by default,
        // so most glMemoryBarrier bits are implicit. Explicit fences are needed
        // only for untracked resources, which we don't currently create.
    }

    @Override
    public void copyBufferSubData(IGpuBuffer src, IGpuBuffer dst, long srcOffset, long dstOffset, long size) {
        if (!(src instanceof MetalBuffer) || !(dst instanceof MetalBuffer)) {
            throw new IllegalArgumentException("copyBufferSubData on Metal backend requires MetalBuffer arguments");
        }
        enqueueBufferCopy(((MetalBuffer) src).getHandle(), ((MetalBuffer) dst).getHandle(),
                srcOffset, dstOffset, size);
    }

    @Override
    public void copyBufferSubData(IGpuPersistentBuffer src, IGpuBuffer dst, long srcOffset, long dstOffset, long size) {
        if (!(src instanceof MetalPersistentBuffer) || !(dst instanceof MetalBuffer)) {
            throw new IllegalArgumentException("copyBufferSubData on Metal backend requires Metal buffer arguments");
        }
        enqueueBufferCopy(((MetalPersistentBuffer) src).getHandle(), ((MetalBuffer) dst).getHandle(),
                srcOffset, dstOffset, size);
    }

    @Override
    public void copyBufferSubData(IGpuBuffer src, IGpuPersistentBuffer dst, long srcOffset, long dstOffset, long size) {
        if (!(src instanceof MetalBuffer) || !(dst instanceof MetalPersistentBuffer)) {
            throw new IllegalArgumentException("copyBufferSubData on Metal backend requires Metal buffer arguments");
        }
        enqueueBufferCopy(((MetalBuffer) src).getHandle(), ((MetalPersistentBuffer) dst).getHandle(),
                srcOffset, dstOffset, size);
    }

    private void enqueueBufferCopy(long srcHandle, long dstHandle, long srcOffset, long dstOffset, long size) {
        if (size <= 0) return;
        if (this.callerPassOpen()) {
            // A caller-held render pass is open on the active buffer (bake-time
            // MetalBudgetBufferRenderer.setup commits vertex uploads mid-pass);
            // a blit encoder can't coexist with it. Standalone buffer committed
            // ahead of the active one — queue order + hazard tracking run the
            // copy before the pass's reads of the destination.
            long cmdBuf = MetalNative.mtlCommandQueueNewCommandBuffer(this.commandQueue);
            long blit = MetalNative.mtlCommandBufferNewBlitEncoder(cmdBuf);
            MetalNative.mtlBlitEncoderCopyBuffer(blit, srcHandle, srcOffset, dstHandle, dstOffset, size);
            MetalNative.mtlEncoderEndEncoding(blit);
            MetalNative.mtlRelease(blit);
            MetalNative.mtlCommandBufferCommit(cmdBuf);
            MetalNative.mtlRelease(cmdBuf);
            return;
        }
        // Encode into the active command buffer so copies execute in request
        // order with the frame's passes: downloads run AFTER their producing
        // dispatches (the old commit-immediately path ran them BEFORE the
        // still-uncommitted compute work) and batched uploads share one
        // encoder instead of one throwaway command buffer per entry.
        this.ensureActiveCommandBuffer();
        if (this.activeBlitEncoder == 0) {
            this.endForeignEncoderIfNeeded();
            logEncoderOp("blit newBlitEncoder (active buffer)");
            this.activeBlitEncoder = MetalNative.mtlCommandBufferNewBlitEncoder(this.activeCommandBuffer);
            if (this.activeBlitEncoder == 0) {
                throw new RuntimeException("mtlCommandBufferNewBlitEncoder returned NULL");
            }
        }
        MetalNative.mtlBlitEncoderCopyBuffer(this.activeBlitEncoder, srcHandle, srcOffset, dstHandle, dstOffset, size);
        this.activeBufferHasBlits = true;
        this.closeBlitEncoderIfGuest();
    }

    /**
     * Encode a texture→buffer copy into the ACTIVE command buffer, preserving
     * frame encoding order (lands after already-encoded passes; the next
     * {@code beginRenderPass} closes the blit encoder, so passes encoded later
     * see the copy's result). Built for the Iris depth export: Metal silently
     * reads ZEROS when a depth-format texture is sampled through a
     * texture2d&lt;float&gt; declaration (SPIRV-Cross only emits depth2d for
     * shadow samplers), so depth crosses to the export shader as raw floats
     * in a plain buffer instead — buffer reads are format-blind, and
     * D32F→buffer blits are format-legal. Copies the full level-0 region
     * (width×height texels, 4 bytes each) tightly packed from offset 0.
     */
    public void copyTextureToBuffer(me.cortex.voxy.client.core.gpu.IGpuTexture src,
                                    IGpuBuffer dst, int width, int height) {
        this.copyTextureToBuffer(src, dst, width, height, 0);
    }

    public void copyTextureToBuffer(me.cortex.voxy.client.core.gpu.IGpuTexture src,
                                    IGpuBuffer dst, int width, int height, long dstOffset) {
        if (!(dst instanceof MetalBuffer dstBuf)) {
            throw new IllegalArgumentException("copyTextureToBuffer on Metal backend requires a MetalBuffer destination");
        }
        if (this.callerPassOpen()) {
            throw new IllegalStateException("copyTextureToBuffer while a caller render pass is open");
        }
        this.ensureActiveCommandBuffer();
        if (this.activeBlitEncoder == 0) {
            this.endForeignEncoderIfNeeded();
            logEncoderOp("blit newBlitEncoder (active buffer)");
            this.activeBlitEncoder = MetalNative.mtlCommandBufferNewBlitEncoder(this.activeCommandBuffer);
            if (this.activeBlitEncoder == 0) {
                throw new RuntimeException("mtlCommandBufferNewBlitEncoder returned NULL");
            }
        }
        long texHandle = MetalHandleMap.getHandle(src.id());
        int bytesPerRow = width * 4;
        MetalNative.mtlBlitEncoderCopyTextureToBuffer(this.activeBlitEncoder, texHandle, 0,
                0, 0, width, height,
                dstBuf.getHandle(), dstOffset, bytesPerRow, bytesPerRow * height);
        this.activeBufferHasBlits = true;
        this.closeBlitEncoderIfGuest();
    }

    /**
     * Ends the batched blit encoder immediately when Voxy is a guest on Metallum's command buffer.
     *
     * <p>Metal permits one open encoder per command buffer, and the command buffer is shared: leaving
     * a blit encoder open past the end of Voxy's work means Metallum's next render encoder — Sodium's
     * terrain pass — trips {@code A command encoder is already encoding to this command buffer}. Voxy
     * has no hook for "the host is about to draw", so the boundary has to be enforced from this side:
     * a guest holds an encoder only while it is actively encoding into it.
     *
     * <p>Costs an encoder per copy instead of one per batch. The expensive half of the old batching
     * is preserved — the copies still share the frame's command buffer rather than each getting a
     * throwaway one — and the owner path keeps batching, since committing the buffer is a clean
     * boundary that closes encoders anyway.
     */
    private void closeBlitEncoderIfGuest() {
        if (!this.ownsActiveCommandBuffer) {
            this.endActiveBlitEncoder();
        } else {
            logEncoderOp("closeBlitEncoderIfGuest SKIP (owner, keeping batch open)");
        }
    }

    /**
     * Whether Voxy must re-point its active command buffer at Metallum's current one.
     *
     * <p>A command buffer handle is only meaningful while its owner has not submitted it. Metallum
     * submits its own frame in addition to serving Voxy's {@code flushFrame()}, so a handle Voxy
     * cached can name a buffer that is already Committed — and encoding into that trips Metal's
     * {@code _status < MTLCommandBufferStatusCommitted} assertion inside
     * {@code setCurrentCommandEncoder:}, which arrives with no Java stack. Comparing against
     * Metallum's live handle every time is what keeps the two in step.
     *
     * <p>Pure and package-private so the truth table is pinned by a test rather than rediscovered
     * from a running game.
     */
    static boolean shouldAdoptMetallumBuffer(final long cachedBuffer, final long metallumBuffer) {
        return metallumBuffer != 0L && cachedBuffer != metallumBuffer;
    }

    private void ensureActiveCommandBuffer() {
        if (this.ownsActiveCommandBuffer && this.activeCommandBuffer != 0L) {
            return;
        }
        // Reaching here means Voxy has no uncommitted buffer of its own, so it is a guest again.
        this.ownsActiveCommandBuffer = false;
        long metallumBuffer = MetallumBridge.available() ? MetallumBridge.commandBuffer() : 0L;
        if (metallumBuffer != 0L) {
            if (shouldAdoptMetallumBuffer(this.activeCommandBuffer, metallumBuffer)) {
                // A blit batch cannot span a buffer change: its encoder belongs to the old buffer.
                this.endActiveBlitEncoder();
                this.activeCommandBuffer = metallumBuffer;
            }
            return;
        }
        if (this.activeCommandBuffer == 0L) {
            this.activeCommandBuffer = MetalNative.mtlCommandQueueNewCommandBuffer(this.commandQueue);
            this.ownsActiveCommandBuffer = true;
            if (this.activeCommandBuffer == 0L) {
                throw new RuntimeException("mtlCommandQueueNewCommandBuffer returned NULL");
            }
        }
    }

    /**
     * Metal forbids two open encoders on one command buffer, and Metallum's render encoder is
     * typically open when Voxy wants a blit. Close it first; Metallum reopens lazily on its next
     * draw, with load actions that preserve whatever Voxy writes. No-op when Voxy owns the buffer.
     */
    /**
     * Encoder tracing, off unless {@code -Dvoxy.encTrace=true} (or {@code VOXY_ENC_TRACE=1}).
     *
     * <p>This instrumentation is what located the three encoder-lifecycle bugs — Metal's "already
     * encoding" assertions carry no Java stack, so the trail was the only way in. It earned its
     * place, but it must not run by default: left on it emitted **428,118 lines / 82 MB in four
     * minutes** (~1,800 lines/second of string building plus log I/O on the render thread), which
     * silently inflates every performance number measured while it was on.
     */
    /** Per-pass attachment trace ({@code VOXY_ATTACH_TRACE=1}), independent of the verbose ENC_TRACE. */
    private static long passTraceCount = 0;

    /** Prefer a Metallum-untracked encoder for Voxy's passes; {@code VOXY_LOD_DETACHED_ENCODER=0} opts out. */
    /** Borrow Metallum's encoder instead of owning one; {@code VOXY_LOD_BORROW_ENCODER=1} for A/B. */
    static final boolean BORROW_ENCODER = "1".equals(System.getenv("VOXY_LOD_BORROW_ENCODER"));

    static final boolean ENC_TRACE = Boolean.getBoolean("voxy.encTrace")
            || "1".equals(System.getenv("VOXY_ENC_TRACE"));

    void logEncoderOp(String what) {
        if (!ENC_TRACE) {
            return;
        }
        Logger.info("[Metal-ENC] " + what
                + " ownsActive=" + this.ownsActiveCommandBuffer
                + " blitEnc=0x" + Long.toHexString(this.activeBlitEncoder)
                + " buf=0x" + Long.toHexString(this.activeCommandBuffer));
    }

    private void endForeignEncoderIfNeeded() {
        // Deliberately NOT gated on ownsActiveCommandBuffer: that flag is decided once, when the
        // active buffer is first created -- which can be before Metallum has a frame buffer at all.
        // Voxy would then believe it owns the buffer and skip this, while Metallum meanwhile opened
        // a render encoder on the same buffer. Asking Metallum to close is a no-op when it has
        // nothing open, so just always ask.
        if (!ENC_TRACE) {
            // Fast path: the reporting accessors are three reflective invocations, and this runs
            // ~150k times per session -- too expensive to keep for a log line that is off.
            MetallumBridge.endCurrentEncoder();
            return;
        }
        long openBefore = MetallumBridge.openEncoderHandle();
        long closed = MetallumBridge.endCurrentEncoderAndReport();
        long openAfter = MetallumBridge.openEncoderHandle();
        logEncoderOp("endForeignEncoder openBefore=0x" + Long.toHexString(openBefore)
                + " closed=0x" + Long.toHexString(closed)
                + " openAfter=0x" + Long.toHexString(openAfter)
                + " (buffers " + Long.toHexString(this.activeCommandBuffer)
                + "/" + Long.toHexString(MetallumBridge.commandBuffer()) + ")");
    }

    private void endActiveBlitEncoder() {
        if (this.activeBlitEncoder != 0) {
            if (ENC_TRACE) {
                logEncoderOp("endActiveBlitEncoder 0x" + Long.toHexString(this.activeBlitEncoder));
            }
            MetalNative.mtlEncoderEndEncoding(this.activeBlitEncoder);
            MetalNative.mtlRelease(this.activeBlitEncoder);
            this.activeBlitEncoder = 0;
        }
    }

    private boolean callerPassOpen() {
        MetalRenderEncoder enc = this.lastRenderEncoder;
        if (enc == null) return false;
        if (enc.handle() == 0) {
            this.lastRenderEncoder = null;
            return false;
        }
        return true;
    }

    /**
     * Commit + wait the active buffer so encoded fence signals execute; no-op
     * when a caller still holds a render pass open (committing then would be
     * invalid). Used by the streams' spin-wait/emergency paths, which would
     * otherwise wait forever on a signal parked in the uncommitted buffer.
     */
    public void flushForFenceProgress() {
        if (this.callerPassOpen()) return;
        this.submit();
    }

    /**
     * Block until every command buffer committed to this queue before this call has completed.
     *
     * <p>Needed because {@link #submit()} does <b>not</b> imply that. Under whole-frame Metal it
     * takes the guest branch: {@code MetallumBridge.flushFrame()} → {@code MetalCommandEncoder.submit()},
     * which commits with a completion block and then waits only for the submit
     * {@code MAX_SUBMITS_IN_FLIGHT} (= 3) earlier — never for the commit it just made — and clears
     * {@code activeCommandBuffer}, leaving nothing to wait on. The owned branch, by contrast, really
     * does call {@code mtlCommandBufferWaitUntilCompleted}. So "submit() waited for the GPU" is true
     * only when Voxy owns its own command buffer, and any CPU readback of GPU-written memory needs
     * this instead.
     *
     * <p>Metal executes command buffers on a queue in commit order, and {@link #newCommandBuffer()}
     * allocates on the same queue — which is Metallum's own when Metallum is present
     * ({@code MetalRenderBackend}'s constructor takes it from {@code MetallumBridge.commandQueue()}).
     * A fresh empty buffer committed here therefore cannot complete before a borrowed frame buffer
     * that was committed earlier, so waiting on it waits on our work too.
     */
    @Override
    public void waitForGpuIdle() {
        long cmdBuffer = this.newCommandBuffer();
        if (cmdBuffer == 0L) {
            throw new RuntimeException("mtlCommandQueueNewCommandBuffer returned NULL");
        }
        MetalNative.mtlCommandBufferCommit(cmdBuffer);
        MetalNative.mtlCommandBufferWaitUntilCompleted(cmdBuffer);
        MetalNative.mtlRelease(cmdBuffer);
    }

    // --- Metal-specific accessors ---

    public long getDevice() {
        return this.device;
    }

    public long getCommandQueue() {
        return this.commandQueue;
    }

    public long getSharedEvent() {
        return this.sharedEvent;
    }

    public long newCommandBuffer() {
        return MetalNative.mtlCommandQueueNewCommandBuffer(this.commandQueue);
    }

    public void shutdown() {
        MetalNative.mtlRelease(this.sharedEvent);
        MetalNative.mtlRelease(this.commandQueue);
        MetalNative.mtlRelease(this.device);
        Logger.info("Metal backend shut down");
    }

    // --- Render pass encoding ---

    @Override
    public RenderEncoder beginRenderPass(RenderPassDesc desc) {
        long passDescHandle = MetalNative.mtlNewRenderPassDescriptor();
        if (passDescHandle == 0) {
            throw new RuntimeException("mtlNewRenderPassDescriptor returned NULL");
        }
        long encoder;
        boolean borrowed = false;
        try {
            long colorHandle = 0L;
            long depthHandle = 0L;
            for (int i = 0; i < desc.colorAttachments().size(); i++) {
                RenderPassDesc.ColorAttachment c = desc.colorAttachments().get(i);
                long texHandle = MetalHandleMap.getHandle(c.texture().id());
                if (i == 0) colorHandle = texHandle;
                MetalNative.mtlRenderPassSetColorAttachment(passDescHandle, i,
                        texHandle, mapLoadAction(c.loadAction()),
                        mapStoreAction(c.storeAction()), c.level());
                if (c.loadAction() == RenderPassDesc.LoadAction.CLEAR) {
                    MetalNative.mtlRenderPassSetColorClearColor(passDescHandle, i,
                            c.clearR(), c.clearG(), c.clearB(), c.clearA());
                }
            }
            if (desc.depthAttachment() != null) {
                RenderPassDesc.DepthAttachment d = desc.depthAttachment();
                long texHandle = MetalHandleMap.getHandle(d.texture().id());
                depthHandle = texHandle;
                MetalNative.mtlRenderPassSetDepthAttachment(passDescHandle, texHandle,
                        mapLoadAction(d.loadAction()), mapStoreAction(d.storeAction()),
                        d.clearDepth(), d.level());
            }

            // VOXY_ATTACH_TRACE=1: the ENC trace logs EVERY beginRenderPass, including depth-only
            // passes (chunk-bound, depth export) that legitimately have no colour attachment, so a
            // bare "color=0x0" line says nothing about the LOD pass. Log the attachment list as
            // seen HERE, so the LOD pass (1 colour attachment, CLEAR or LOAD) is identifiable.
            // Key on the LOD pass's signature -- BOTH a colour and a depth attachment -- because a
            // "% 400" sample lands on depth-only helper passes and misses it entirely. The clear
            // colour is printed too: with VOXY_LOD_DEBUG_CLEAR=1 it is magenta (1,0,1,1), so a line
            // showing that proves the pass really is built against the frame's attachments.
            int nColors = desc.colorAttachments().size();
            if ("1".equals(System.getenv("VOXY_ATTACH_TRACE"))
                    && nColors > 0 && desc.depthAttachment() != null
                    && (passTraceCount++ % 60) == 1) {
                var c0 = desc.colorAttachments().get(0);
                // Report the attachment's REAL pixel formats (Metal-native values off the textures)
                // and what the pipeline declares. createGraphicsPipeline hardcodes the depth format
                // to Depth32Float; if the frame's depth attachment is anything else, the pipeline
                // state is incompatible with the pass and Metal drops the draws while every counter
                // still reports them. The P0 probe read both formats off the textures instead of
                // assuming, which is why it worked.
                // Read the formats straight off the live MTLTexture handles -- works for any
                // texture, and needs no access to the pipeline's attachment objects.
                int realColorFmt = colorHandle == 0 ? -1 : MetalNative.mtlTextureGetPixelFormat(colorHandle);
                int realDepthFmt = depthHandle == 0 ? -1 : MetalNative.mtlTextureGetPixelFormat(depthHandle);
                me.cortex.voxy.common.Logger.info("[Metal-PASS] colors=" + nColors
                        + " colorHandle=0x" + Long.toHexString(colorHandle)
                        + " depthHandle=0x" + Long.toHexString(depthHandle)
                        + " load0=" + c0.loadAction()
                        + " size=" + desc.viewportWidth() + "x" + desc.viewportHeight()
                        + " | realColorFmt=" + realColorFmt + " realDepthFmt=" + realDepthFmt
                        // Does Metallum still have a clear queued for the texture we are about to
                        // draw into? Our borrow path bypasses createRenderPass, which is where
                        // those are consumed -- so if one is queued, MC's next pass applies it and
                        // erases this frame's LOD. Measured, not assumed.
                        + " | pendingColorClear=" + MetallumBridge.hasPendingColorClear(colorHandle)
                        + " pendingDepthClear=" + MetallumBridge.hasPendingDepthClear(depthHandle)
                        + " pendingTotal=" + MetallumBridge.pendingColorClearCount());
            }

            // Pending stream copies must land before the pass's encoder opens
            // (one encoder at a time per buffer; copies feed the pass anyway).
            this.endActiveBlitEncoder();
            this.ensureActiveCommandBuffer();
            // Must be logged AFTER ensureActiveCommandBuffer, which is what decides whether Voxy
            // encodes into Metallum's live frame buffer or keeps one of its own. If these differ,
            // a perfectly-formed pass (correct attachments, correct clear) lands in a command
            // buffer that is never the one presented -- which is indistinguishable from "the pass
            // does nothing" everywhere else in the logs.
            if ("1".equals(System.getenv("VOXY_ATTACH_TRACE")) && nColors > 0
                    && desc.depthAttachment() != null && (passTraceCount % 60) == 1) {
                long metallumBuf = MetallumBridge.available() ? MetallumBridge.commandBuffer() : 0L;
                me.cortex.voxy.common.Logger.info("[Metal-PASSBUF] active=0x"
                        + Long.toHexString(this.activeCommandBuffer)
                        + " metallum=0x" + Long.toHexString(metallumBuf)
                        + " owns=" + this.ownsActiveCommandBuffer
                        + (metallumBuf != 0 && this.activeCommandBuffer != metallumBuf
                            ? "   <-- NOT the presented buffer" : ""));
            }

            // Prefer BORROWING Metallum's encoder for these attachments. Asking Metal for a second
            // encoder on the same command buffer is illegal ("A command encoder is already encoding
            // to this command buffer"), and closing Metallum's to open our own costs a pass split for
            // no reason. Borrowing also means Voxy's draws land in Metallum's pass directly, sharing
            // its depth. Falls back to owning an encoder when Metallum is absent.
            if (ENC_TRACE) {
                logEncoderOp("beginRenderPass borrowRequest color=0x" + Long.toHexString(colorHandle));
            }
            // Use Voxy's OWN encoder, created from the pass descriptor -- Metal's supported
            // pattern for sequential encoders on one command buffer. Do NOT borrow Metallum's.
            //
            // The borrow path registers its encoder as Metallum's currentEncoder but leaves
            // currentRenderPass pointing at whatever pass ran last, so the frame's teardown
            // (presentTextureToDrawable -> flushPendingClear -> submitRenderPass ->
            // materializePendingClear) can act on the wrong pass and end the borrowed encoder early
            // or clear into it. Bisected with VOXY_LOD_TRIANGLE: a magenta triangle through the
            // borrowed encoder produced 0.00%, through a directly-created one 17.13%.
            //
            // This is also why every other Voxy pass already works: the chunk-bound pass, the depth
            // export and the bakery all own their encoders and end them. Only the LOD pass borrowed.
            // A detached-encoder variant was tried and reverted: an encoder Metallum cannot see lets
            // it open a second one on the same command buffer, which asserts ("A command encoder is
            // already encoding to this command buffer"). Borrowing badly is worse than not
            // borrowing.
            // VOXY_LOD_BORROW_ENCODER=1 restores the borrow path for a controlled A/B.
            long shared = BORROW_ENCODER ? MetallumBridge.acquireRenderEncoder(
                    colorHandle, depthHandle, desc.viewportWidth(), desc.viewportHeight()) : 0L;
            if (ENC_TRACE) {
                logEncoderOp("beginRenderPass borrowResult=0x" + Long.toHexString(shared));
            }
            if (shared != 0L) {
                encoder = shared;
                borrowed = true;
            } else {
                this.endForeignEncoderIfNeeded();
                if (ENC_TRACE) {
                    logEncoderOp("beginRenderPass OWN newRenderEncoder");
                }
                encoder = MetalNative.mtlCommandBufferNewRenderEncoder(this.activeCommandBuffer, passDescHandle);
                if (encoder == 0) {
                    throw new RuntimeException("mtlCommandBufferNewRenderEncoder returned NULL");
                }
            }

            // The borrow result must be attributed to THIS pass. A depth-only pass (chunk-bound,
            // depth export) legitimately has colorHandle==0, and acquireRenderEncoder returns 0 for
            // it by design -- reading such a line as "the LOD pass failed to borrow" is wrong twice
            // over, and I made that mistake. Only a colour+depth pass is the LOD pass.
            if ("1".equals(System.getenv("VOXY_ATTACH_TRACE")) && nColors > 0
                    && desc.depthAttachment() != null && (passTraceCount % 60) == 1) {
                me.cortex.voxy.common.Logger.info("[Metal-BORROW] colors=" + nColors
                        + " borrowed=" + borrowed
                        + (borrowed
                            ? "  (Metallum's encoder reused -- its pending clear is already materialised)"
                            : "  <-- Voxy opened its OWN; Metallum's pending clear can wipe it"));
            }
        } finally {
            // The render encoder retains a reference to the descriptor; we can drop ours.
            MetalNative.mtlRelease(passDescHandle);
        }

        MetalRenderEncoder result = new MetalRenderEncoder(encoder, borrowed);
        this.lastRenderEncoder = result;
        return result;
    }

    @Override
    public IGpuPipeline createGraphicsPipeline(GraphicsPipelineDesc desc) {
        // If MSL isn't pre-baked, fall back to the runtime GLSL → SPIRV → MSL
        // pipeline. M9 migrated call sites supply GLSL only (one source, all
        // backends); only the M3-M8 smoke tests feed pre-baked MSL/SPIRV.
        String vertexMsl = desc.vertexMsl;
        String fragmentMsl = desc.fragmentMsl;
        if (vertexMsl == null || fragmentMsl == null) {
            if (desc.vertexGlsl == null || desc.fragmentGlsl == null) {
                throw new IllegalArgumentException(
                        "MetalRenderBackend.createGraphicsPipeline: need MSL or GLSL; got neither. label=" + desc.label);
            }
            try {
                var vertResult = me.cortex.voxy.client.core.gpu.shader.RuntimeShaderCompiler.compile(
                        desc.vertexGlsl,
                        me.cortex.voxy.client.core.gpu.shader.RuntimeShaderCompiler.Stage.VERTEX,
                        desc.defines,
                        me.cortex.voxy.client.core.gpu.shader.RuntimeShaderCompiler.Target.METAL_MSL);
                var fragResult = me.cortex.voxy.client.core.gpu.shader.RuntimeShaderCompiler.compile(
                        desc.fragmentGlsl,
                        me.cortex.voxy.client.core.gpu.shader.RuntimeShaderCompiler.Stage.FRAGMENT,
                        desc.defines,
                        me.cortex.voxy.client.core.gpu.shader.RuntimeShaderCompiler.Target.METAL_MSL);
                vertexMsl = vertResult.mslSource();
                fragmentMsl = fragResult.mslSource();
            } catch (Throwable t) {
                throw new RuntimeException(
                        "MetalRenderBackend.createGraphicsPipeline: GLSL → MSL transpile failed for label="
                                + desc.label, t);
            }
        }

        long vertexLib = MetalNative.mtlDeviceNewLibraryWithSource(this.device, vertexMsl);
        if (vertexLib == 0) {
            throw new RuntimeException("Vertex MSL compile failed: " + MetalNative.mtlGetLastCompileError());
        }
        long fragmentLib = 0;
        long vertexFn = 0;
        long fragmentFn = 0;
        long pipelineState = 0;
        long pipelineDesc = 0;
        long dssHandle = 0;
        try {
            fragmentLib = MetalNative.mtlDeviceNewLibraryWithSource(this.device, fragmentMsl);
            if (fragmentLib == 0) {
                throw new RuntimeException("Fragment MSL compile failed: " + MetalNative.mtlGetLastCompileError());
            }
            vertexFn = MetalNative.mtlLibraryNewFunction(vertexLib, "main0");
            if (vertexFn == 0) {
                // SPIRV-Cross emits the entry point as "main0" by default; fall back to "main".
                vertexFn = MetalNative.mtlLibraryNewFunction(vertexLib, "main");
                if (vertexFn == 0) {
                    throw new RuntimeException("Vertex function 'main0'/'main' not found in compiled library");
                }
            }
            fragmentFn = MetalNative.mtlLibraryNewFunction(fragmentLib, "main0");
            if (fragmentFn == 0) {
                fragmentFn = MetalNative.mtlLibraryNewFunction(fragmentLib, "main");
                if (fragmentFn == 0) {
                    throw new RuntimeException("Fragment function 'main0'/'main' not found in compiled library");
                }
            }

            // colorAttachmentFormat == 0 means "depth-only render pass — no
            // color target" (e.g. HiZBuffer's blit pipeline writes only
            // gl_FragDepth). Skip the color attachment format entirely in
            // that case; Metal accepts a pipeline with no color attachments
            // as long as the fragment shader doesn't write any.
            pipelineDesc = MetalNative.mtlNewRenderPipelineDescriptor();
            if (pipelineDesc == 0) {
                throw new RuntimeException("mtlNewRenderPipelineDescriptor returned NULL");
            }
            MetalNative.mtlRenderPipelineDescriptorSetVertexFunction(pipelineDesc, vertexFn);
            MetalNative.mtlRenderPipelineDescriptorSetFragmentFunction(pipelineDesc, fragmentFn);
            // Phase C: set the format for EACH MRT color attachment (the material
            // g-buffer uses 3). Single-attachment pipelines have a 1-element array.
            // A 0 entry means "no attachment at this index".
            for (int i = 0; i < desc.colorAttachmentFormats.length; i++) {
                int fmt = desc.colorAttachmentFormats[i];
                if (fmt != 0) {
                    MetalNative.mtlRenderPipelineDescriptorSetColorAttachmentFormat(
                            pipelineDesc, i, MetalFormatUtil.glFormatToMetal(fmt));
                }
            }
            // Blocker 1: enable ICB usage on every pipeline. The only Metal
            // features that conflict (vertex amplification, function constants
            // on stage-input) aren't used anywhere in Voxy. The runtime cost
            // is the validation overhead Metal adds at draw time; benchmarks
            // (M14) can re-evaluate gating this if it shows up as overhead.
            // ICB-incompatible shaders (notably fragment shaders that write
            // gl_FragDepth or use certain outputs) reject the flag with
            // "Fragment shader cannot be used with indirect command buffers".
            // Only the MDIC terrain pipelines actually need ICB execution, so
            // the flag is opt-in via the desc.
            if (desc.usedInIndirectCommandBuffer) {
                MetalNative.mtlRenderPipelineDescriptorSetSupportIndirectCommandBuffers(pipelineDesc, true);
            }
            // Always declare a depth attachment format. Some Voxy fragment
            // shaders (depth0.frag, depth_copy.frag, blit.fsh, hiz/blit.fsh,
            // and the patched terrain frags) write to gl_FragDepth — Metal
            // rejects the pipeline with "depthAttachmentPixelFormat is not
            // valid" unless the descriptor has a depth format set. Use
            // MTLPixelFormatDepth32Float (universally supported on Apple
            // Silicon; the equivalent of Voxy's most common depth target).
            // Pipelines that don't actually write depth pay no penalty here.
            MetalNative.mtlRenderPipelineDescriptorSetDepthAttachmentPixelFormat(
                    pipelineDesc, MetalFormatUtil.MTLPixelFormatDepth32Float);

            // Bake blend state into the pipeline (Metal stores it on the pipeline,
            // not on the encoder). Skip when there's no color attachment.
            if (desc.colorAttachmentFormat != 0) {
                PipelineState.BlendState blend = desc.state.blend;
                MetalNative.mtlRenderPipelineDescriptorSetColorAttachmentBlending(pipelineDesc, 0,
                        blend.enabled,
                        mapBlendOp(blend.colorOp), mapBlendOp(blend.alphaOp),
                        mapBlendFactor(blend.srcColor), mapBlendFactor(blend.dstColor),
                        mapBlendFactor(blend.srcAlpha), mapBlendFactor(blend.dstAlpha));
            }

            // Build + attach vertex descriptor if the pipeline declares vertex inputs.
            // Empty layout → no descriptor (gl_VertexIndex-driven shaders).
            long vertexDescHandle = 0;
            if (desc.vertexLayout.attributes.length > 0 || desc.vertexLayout.buffers.length > 0) {
                vertexDescHandle = MetalNative.mtlNewVertexDescriptor();
                if (vertexDescHandle == 0) {
                    throw new RuntimeException("mtlNewVertexDescriptor returned NULL");
                }
                for (VertexLayout.VertexAttribute attr : desc.vertexLayout.attributes) {
                    MetalNative.mtlVertexDescriptorSetAttribute(vertexDescHandle,
                            attr.location, attr.format.metalValue, attr.offset, attr.bufferSlot);
                }
                for (VertexLayout.VertexBufferBinding buf : desc.vertexLayout.buffers) {
                    int stepFunction = buf.stepRate == VertexLayout.StepRate.PER_INSTANCE
                            ? MetalNative.MTLVertexStepFunctionPerInstance
                            : MetalNative.MTLVertexStepFunctionPerVertex;
                    MetalNative.mtlVertexDescriptorSetLayout(vertexDescHandle,
                            buf.slot, buf.stride, stepFunction, 1);
                }
                MetalNative.mtlRenderPipelineDescriptorSetVertexDescriptor(pipelineDesc, vertexDescHandle);
            }

            pipelineState = MetalNative.mtlDeviceNewRenderPipelineState(this.device, pipelineDesc);
            if (vertexDescHandle != 0) {
                // Pipeline state retained the descriptor; drop our reference.
                MetalNative.mtlRelease(vertexDescHandle);
            }

            // Build the depth-stencil state. Skip if both test + write are off.
            if (desc.state.depth.testEnabled || desc.state.depth.writeEnabled) {
                long dssDesc = MetalNative.mtlNewDepthStencilDescriptor();
                if (dssDesc == 0) throw new RuntimeException("mtlNewDepthStencilDescriptor returned NULL");
                try {
                    int compareFn = desc.state.depth.testEnabled
                            ? mapCompareOp(desc.state.depth.compareOp)
                            : MetalNative.MTLCompareFunctionAlways;
                    MetalNative.mtlDepthStencilDescriptorSetCompareFunction(dssDesc, compareFn);
                    MetalNative.mtlDepthStencilDescriptorSetDepthWriteEnabled(dssDesc, desc.state.depth.writeEnabled);
                    dssHandle = MetalNative.mtlDeviceNewDepthStencilState(this.device, dssDesc);
                    if (dssHandle == 0) throw new RuntimeException("mtlDeviceNewDepthStencilState returned NULL");
                } finally {
                    MetalNative.mtlRelease(dssDesc);
                }
            }
            int cullModeI = mapCullMode(desc.state.raster.cullMode);
            int windingI = mapFrontFace(desc.state.raster.frontFace);
            int fillModeI = mapPolygonMode(desc.state.raster.polygonMode);
            if (pipelineState == 0) {
                throw new RuntimeException("Pipeline state link failed: " + MetalNative.mtlGetLastCompileError());
            }
            if (desc.label != null) {
                MetalNative.mtlSetLabel(pipelineState, desc.label);
            }

            MetalGraphicsPipeline result = new MetalGraphicsPipeline(
                    pipelineState, vertexLib, fragmentLib, vertexFn, fragmentFn,
                    dssHandle, cullModeI, windingI, fillModeI);
            // result owns the handles now; clear locals so the catch path doesn't double-release.
            pipelineState = 0;
            vertexFn = 0;
            fragmentFn = 0;
            vertexLib = 0;
            fragmentLib = 0;
            dssHandle = 0;
            return result;
        } finally {
            if (pipelineDesc != 0) MetalNative.mtlRelease(pipelineDesc);
            if (pipelineState != 0) MetalNative.mtlRelease(pipelineState);
            if (dssHandle != 0) MetalNative.mtlRelease(dssHandle);
            if (fragmentFn != 0) MetalNative.mtlRelease(fragmentFn);
            if (vertexFn != 0) MetalNative.mtlRelease(vertexFn);
            if (fragmentLib != 0) MetalNative.mtlRelease(fragmentLib);
            if (vertexLib != 0) MetalNative.mtlRelease(vertexLib);
        }
    }

    // --- PipelineState → Metal enum translators ---

    private static int mapCompareOp(PipelineState.CompareOp op) {
        return switch (op) {
            case NEVER -> MetalNative.MTLCompareFunctionNever;
            case LESS -> MetalNative.MTLCompareFunctionLess;
            case EQUAL -> MetalNative.MTLCompareFunctionEqual;
            case LESS_EQUAL -> MetalNative.MTLCompareFunctionLessEqual;
            case GREATER -> MetalNative.MTLCompareFunctionGreater;
            case NOT_EQUAL -> MetalNative.MTLCompareFunctionNotEqual;
            case GREATER_EQUAL -> MetalNative.MTLCompareFunctionGreaterEqual;
            case ALWAYS -> MetalNative.MTLCompareFunctionAlways;
        };
    }

    private static int mapCullMode(PipelineState.CullMode mode) {
        return switch (mode) {
            case NONE -> MetalNative.MTLCullModeNone;
            case FRONT -> MetalNative.MTLCullModeFront;
            case BACK -> MetalNative.MTLCullModeBack;
        };
    }

    private static int mapFrontFace(PipelineState.FrontFace face) {
        return switch (face) {
            case CLOCKWISE -> MetalNative.MTLWindingClockwise;
            case COUNTER_CLOCKWISE -> MetalNative.MTLWindingCounterClockwise;
        };
    }

    private static int mapPolygonMode(PipelineState.PolygonMode mode) {
        return switch (mode) {
            case FILL -> MetalNative.MTLTriangleFillModeFill;
            case LINE -> MetalNative.MTLTriangleFillModeLines;
        };
    }

    private static int mapBlendOp(PipelineState.BlendOp op) {
        return switch (op) {
            case ADD -> MetalNative.MTLBlendOperationAdd;
            case SUBTRACT -> MetalNative.MTLBlendOperationSubtract;
            case REVERSE_SUBTRACT -> MetalNative.MTLBlendOperationReverseSubtract;
            case MIN -> MetalNative.MTLBlendOperationMin;
            case MAX -> MetalNative.MTLBlendOperationMax;
        };
    }

    private static int mapSamplerFilter(SamplerDesc.Filter f) {
        return switch (f) {
            case NEAREST -> MetalNative.MTLSamplerFilterNearest;
            case LINEAR -> MetalNative.MTLSamplerFilterLinear;
        };
    }

    private static int mapSamplerMipFilter(SamplerDesc.MipFilter f) {
        return switch (f) {
            case NOT_MIPMAPPED -> MetalNative.MTLSamplerMipFilterNotMipmapped;
            case NEAREST -> MetalNative.MTLSamplerMipFilterNearest;
            case LINEAR -> MetalNative.MTLSamplerMipFilterLinear;
        };
    }

    private static int mapSamplerWrap(SamplerDesc.Wrap w) {
        return switch (w) {
            case CLAMP_TO_EDGE -> MetalNative.MTLSamplerAddressModeClampToEdge;
            case REPEAT -> MetalNative.MTLSamplerAddressModeRepeat;
            case MIRRORED_REPEAT -> MetalNative.MTLSamplerAddressModeMirrorRepeat;
            case CLAMP_TO_ZERO -> MetalNative.MTLSamplerAddressModeClampToZero;
        };
    }

    private static int mapBlendFactor(PipelineState.BlendFactor f) {
        return switch (f) {
            case ZERO -> MetalNative.MTLBlendFactorZero;
            case ONE -> MetalNative.MTLBlendFactorOne;
            case SRC_COLOR -> MetalNative.MTLBlendFactorSourceColor;
            case ONE_MINUS_SRC_COLOR -> MetalNative.MTLBlendFactorOneMinusSourceColor;
            case DST_COLOR -> MetalNative.MTLBlendFactorDestinationColor;
            case ONE_MINUS_DST_COLOR -> MetalNative.MTLBlendFactorOneMinusDestinationColor;
            case SRC_ALPHA -> MetalNative.MTLBlendFactorSourceAlpha;
            case ONE_MINUS_SRC_ALPHA -> MetalNative.MTLBlendFactorOneMinusSourceAlpha;
            case DST_ALPHA -> MetalNative.MTLBlendFactorDestinationAlpha;
            case ONE_MINUS_DST_ALPHA -> MetalNative.MTLBlendFactorOneMinusDestinationAlpha;
        };
    }

    @Override
    public void submit() {
        if (this.activeCommandBuffer == 0) return;
        this.endActiveBlitEncoder();
        // Fence signals deferred from createFence (caller pass was open while
        // copies sat parked in this buffer) — encode them now, after the copies,
        // so signaled() implies the guarded copies executed.
        if (!this.pendingFenceSignals.isEmpty()) {
            for (long v : this.pendingFenceSignals) {
                MetalNative.mtlCommandBufferEncodeSignalEvent(this.activeCommandBuffer, this.sharedEvent, v);
            }
            this.pendingFenceSignals.clear();
        }
        this.activeBufferHasBlits = false;

        if (!this.ownsActiveCommandBuffer) {
            // Metallum owns the frame buffer. Split the frame here so everything encoded so far --
            // including the compute prepasses that wrote drawCallBuffer -- is committed and
            // complete, making it CPU-visible for the draw path's baseInstance read. The next
            // ensureActiveCommandBuffer() picks up the fresh buffer Metallum opens afterwards.
            //
            // This costs a pipeline drain, which is what Voxy pays today when it owns its own
            // frames; correctness first. If Metallum predates the hook we cannot satisfy the read,
            // so drop our reference and let the caller see stale data rather than corrupting the
            // frame by committing a buffer we do not own.
            if (MetallumBridge.supportsFlushFrame()) {
                MetallumBridge.flushFrame();
                if (SUBMIT_ORDER) {
                    // flushFrame() COMMITS the frame; it does not wait for it, and the comment above
                    // claiming the work is "committed and complete, making it CPU-visible for the draw
                    // path's baseInstance read" is therefore false. Only this wait makes it true.
                    //
                    // The gap, traced: Metallum's MetalCommandEncoder commits with a completion block
                    // and then awaits `currentSubmitIndex - MAX_SUBMITS_IN_FLIGHT`, i.e. three submits
                    // BACK; and its counter STARTS at 3, so the first three submits wait for nothing at
                    // all. Meanwhile the owned branch immediately below really does waitUntilCompleted,
                    // which is exactly why this read looked safe for so long.
                    //
                    // What the un-ordered window exposes: MetalRenderEncoder reads `baseInstance` out
                    // of drawCallBuffer on the CPU, per draw, and pushes it as the per-draw constant. A
                    // stale read yields the WRONG SECTION'S ORIGIN while the draw keeps its own quads
                    // and their baked light -- which is bug 3's description almost word for word,
                    // including the preserved lighting. It also explains the timing dependence: how
                    // stale the read is depends on how far the CPU has run ahead of the GPU, so frame
                    // pacing moves the rate, and a frame-END GPU sync cannot help because these reads
                    // happen MID-frame.
                    this.waitForGpuIdle();
                }
            }
            this.activeCommandBuffer = 0;
            return;
        }

        MetalNative.mtlCommandBufferCommit(this.activeCommandBuffer);
        // Sync mode for M3: wait for completion so the smoke test can check status
        // before the buffer is released. M5+ will move to async + per-frame fences.
        MetalNative.mtlCommandBufferWaitUntilCompleted(this.activeCommandBuffer);
        int status = MetalNative.mtlCommandBufferGetStatus(this.activeCommandBuffer);
        MetalNative.mtlRelease(this.activeCommandBuffer);
        this.activeCommandBuffer = 0;
        if (status != MetalNative.MTLCommandBufferStatusCompleted) {
            throw new RuntimeException("Metal command buffer ended with status=" + status
                    + " (expected " + MetalNative.MTLCommandBufferStatusCompleted + " = Completed)");
        }
    }

    @Override
    public IGpuPipeline createComputePipeline(ComputePipelineDesc desc) {
        // The shader-declared local size is authoritative for Metal's
        // threadsPerThreadgroup — desc literals drifted from the shaders before
        // (cmdgen/prefixSum/translucentGen said 32 vs 128/256, dropping ~75% of
        // threads per dispatch = the 2026-05 LOD flicker). GL is immune since
        // glDispatchCompute always uses the compiled shader's size. MSL-only
        // descs (smoke tests) have no GLSL to parse; trust the desc there.
        int localSizeX = desc.localSizeX;
        int localSizeY = desc.localSizeY;
        int localSizeZ = desc.localSizeZ;
        if (desc.computeGlsl != null) {
            var parsed = ComputeLocalSizeParser.parse(desc.computeGlsl, desc.defines, desc.label);
            if (parsed.x() != desc.localSizeX || parsed.y() != desc.localSizeY || parsed.z() != desc.localSizeZ) {
                Logger.warn("createComputePipeline[" + desc.label + "]: desc localSize ("
                        + desc.localSizeX + "," + desc.localSizeY + "," + desc.localSizeZ
                        + ") != shader-declared (" + parsed.x() + "," + parsed.y() + "," + parsed.z()
                        + "); using shader-declared values for threadsPerThreadgroup");
            }
            localSizeX = parsed.x();
            localSizeY = parsed.y();
            localSizeZ = parsed.z();
        }
        String computeMsl = desc.computeMsl;
        if (computeMsl == null) {
            if (desc.computeGlsl == null) {
                throw new IllegalArgumentException(
                        "createComputePipeline: need MSL or GLSL; got neither. label=" + desc.label);
            }
            try {
                var result = me.cortex.voxy.client.core.gpu.shader.RuntimeShaderCompiler.compile(
                        desc.computeGlsl,
                        me.cortex.voxy.client.core.gpu.shader.RuntimeShaderCompiler.Stage.COMPUTE,
                        desc.defines,
                        me.cortex.voxy.client.core.gpu.shader.RuntimeShaderCompiler.Target.METAL_MSL);
                computeMsl = result.mslSource();
            } catch (Throwable t) {
                throw new RuntimeException(
                        "MetalRenderBackend.createComputePipeline: GLSL → MSL transpile failed for label="
                                + desc.label, t);
            }
        }
        long library = MetalNative.mtlDeviceNewLibraryWithSource(this.device, computeMsl);
        if (library == 0) {
            throw new RuntimeException("Compute MSL compile failed: " + MetalNative.mtlGetLastCompileError());
        }
        long function = 0;
        long pipelineState = 0;
        try {
            function = MetalNative.mtlLibraryNewFunction(library, "main0");
            if (function == 0) {
                function = MetalNative.mtlLibraryNewFunction(library, "main");
                if (function == 0) {
                    throw new RuntimeException("Compute function 'main0'/'main' not found in compiled library");
                }
            }
            pipelineState = MetalNative.mtlDeviceNewComputePipelineState(this.device, function,
                    localSizeX * localSizeY * localSizeZ);
            if (pipelineState == 0) {
                throw new RuntimeException("Compute pipeline state link failed: " + MetalNative.mtlGetLastCompileError());
            }
            if (desc.label != null) {
                MetalNative.mtlSetLabel(pipelineState, desc.label);
            }

            MetalComputePipeline result = new MetalComputePipeline(
                    pipelineState, library, function,
                    localSizeX, localSizeY, localSizeZ);
            pipelineState = 0;
            function = 0;
            library = 0;
            return result;
        } finally {
            if (pipelineState != 0) MetalNative.mtlRelease(pipelineState);
            if (function != 0) MetalNative.mtlRelease(function);
            if (library != 0) MetalNative.mtlRelease(library);
        }
    }

    @Override
    public IGpuSampler createSampler(SamplerDesc desc) {
        long descHandle = MetalNative.mtlNewSamplerDescriptor();
        if (descHandle == 0) throw new RuntimeException("mtlNewSamplerDescriptor returned NULL");
        try {
            MetalNative.mtlSamplerDescriptorSetMinFilter(descHandle, mapSamplerFilter(desc.minFilter));
            MetalNative.mtlSamplerDescriptorSetMagFilter(descHandle, mapSamplerFilter(desc.magFilter));
            MetalNative.mtlSamplerDescriptorSetMipFilter(descHandle, mapSamplerMipFilter(desc.mipFilter));
            MetalNative.mtlSamplerDescriptorSetSAddressMode(descHandle, mapSamplerWrap(desc.wrapS));
            MetalNative.mtlSamplerDescriptorSetTAddressMode(descHandle, mapSamplerWrap(desc.wrapT));
            MetalNative.mtlSamplerDescriptorSetRAddressMode(descHandle, mapSamplerWrap(desc.wrapR));
            MetalNative.mtlSamplerDescriptorSetLodMinClamp(descHandle, desc.lodMinClamp);
            MetalNative.mtlSamplerDescriptorSetLodMaxClamp(descHandle, desc.lodMaxClamp);
            int compareFn = desc.compareEnable
                    ? mapCompareOp(desc.compareOp)
                    : MetalNative.MTLCompareFunctionNever;
            MetalNative.mtlSamplerDescriptorSetCompareFunction(descHandle, compareFn);

            long stateHandle = MetalNative.mtlDeviceNewSamplerState(this.device, descHandle);
            if (stateHandle == 0) throw new RuntimeException("mtlDeviceNewSamplerState returned NULL");
            if (desc.label != null) MetalNative.mtlSetLabel(stateHandle, desc.label);
            return new MetalSampler(stateHandle);
        } finally {
            MetalNative.mtlRelease(descHandle);
        }
    }

    @Override
    public ComputeEncoder beginComputePass() {
        // Pending stream copies (e.g. UploadStream.commit feeding this pass's
        // SSBOs) must be encoded before the compute encoder opens.
        this.endActiveBlitEncoder();
        this.ensureActiveCommandBuffer();
        // One encoder at a time per command buffer. Voxy's compute prepasses run every frame while
        // Metallum may still have a render encoder open on this buffer, so close it first; Metallum
        // reopens lazily on its next draw with load actions that preserve what Voxy wrote.
        this.endForeignEncoderIfNeeded();
        logEncoderOp("beginComputePass newComputeEncoder");
        long encoder = MetalNative.mtlCommandBufferNewComputeEncoder(this.activeCommandBuffer);
        if (encoder == 0) {
            throw new RuntimeException("mtlCommandBufferNewComputeEncoder returned NULL");
        }
        return new MetalComputeEncoder(encoder);
    }

    @Override
    public me.cortex.voxy.client.core.gpu.IGpuIndirectCommandBuffer createIndirectCommandBuffer(int maxCommands) {
        if (maxCommands <= 0) {
            throw new IllegalArgumentException("maxCommands must be > 0, got " + maxCommands);
        }
        return new MetalIndirectCommandBuffer(this.device, maxCommands);
    }

    /**
     * Synchronously read RGBA8 pixels from a texture region into a byte array.
     * Allocates a transient Shared-storage MTLBuffer, runs a blit-encoder copy,
     * waits for completion, and copies the contents back into the JVM heap.
     *
     * Used by the M5 smoke test to verify a triangle actually rasterized over
     * the cleared background; will likely also serve M14 benchmark capture.
     * Not part of {@link RenderBackend} yet — backend-agnostic readback lands
     * in M9 with the migration of GLPixelDownload through the abstraction.
     */
    public byte[] readPixelsRGBA8(me.cortex.voxy.client.core.gpu.IGpuTexture texture,
                                   int x, int y, int width, int height) {
        final int bytesPerPixel = 4;
        final int bytesPerRow = width * bytesPerPixel;
        final long bufferSize = (long) bytesPerRow * height;

        long readbackBuf = MetalNative.mtlDeviceNewBuffer(this.device, bufferSize,
                MetalNative.MTLResourceStorageModeShared);
        if (readbackBuf == 0) {
            throw new RuntimeException("readPixelsRGBA8: failed to allocate readback buffer ("
                    + bufferSize + " bytes)");
        }
        long cmdBuf = 0;
        long blitEnc = 0;
        try {
            cmdBuf = MetalNative.mtlCommandQueueNewCommandBuffer(this.commandQueue);
            if (cmdBuf == 0) throw new RuntimeException("readPixelsRGBA8: failed to allocate command buffer");
            blitEnc = MetalNative.mtlCommandBufferNewBlitEncoder(cmdBuf);
            if (blitEnc == 0) throw new RuntimeException("readPixelsRGBA8: failed to allocate blit encoder");

            long texHandle = MetalHandleMap.getHandle(texture.id());
            MetalNative.mtlBlitEncoderCopyTextureToBuffer(blitEnc, texHandle, 0,
                    x, y, width, height,
                    readbackBuf, 0, bytesPerRow, (int) bufferSize);
            MetalNative.mtlEncoderEndEncoding(blitEnc);
            MetalNative.mtlRelease(blitEnc);
            blitEnc = 0;

            MetalNative.mtlCommandBufferCommit(cmdBuf);
            MetalNative.mtlCommandBufferWaitUntilCompleted(cmdBuf);
            int status = MetalNative.mtlCommandBufferGetStatus(cmdBuf);
            if (status != MetalNative.MTLCommandBufferStatusCompleted) {
                throw new RuntimeException("readPixelsRGBA8 blit ended with status=" + status);
            }

            long contentsPtr = MetalNative.mtlBufferContents(readbackBuf);
            if (contentsPtr == 0) throw new RuntimeException("readPixelsRGBA8: buffer contents pointer is null");
            byte[] result = new byte[(int) bufferSize];
            org.lwjgl.system.MemoryUtil.memByteBuffer(contentsPtr, result.length).get(result);
            return result;
        } finally {
            if (blitEnc != 0) MetalNative.mtlRelease(blitEnc);
            if (cmdBuf != 0) MetalNative.mtlRelease(cmdBuf);
            MetalNative.mtlRelease(readbackBuf);
        }
    }

    private static int mapLoadAction(RenderPassDesc.LoadAction action) {
        return switch (action) {
            case LOAD -> MetalNative.MTLLoadActionLoad;
            case CLEAR -> MetalNative.MTLLoadActionClear;
            case DONT_CARE -> MetalNative.MTLLoadActionDontCare;
        };
    }

    private static int mapStoreAction(RenderPassDesc.StoreAction action) {
        return switch (action) {
            case STORE -> MetalNative.MTLStoreActionStore;
            case DONT_CARE -> MetalNative.MTLStoreActionDontCare;
        };
    }
}
