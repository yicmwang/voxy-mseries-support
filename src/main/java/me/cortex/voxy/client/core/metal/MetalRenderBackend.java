package me.cortex.voxy.client.core.metal;

import me.cortex.voxy.client.core.gpu.*;
import me.cortex.voxy.common.Logger;

import java.util.concurrent.atomic.AtomicLong;

// GL_TEXTURE_2D is a texture-type TOKEN on the IGpuTexture API (MetalTexture maps it to the
// MTLTextureType), not a GL call.
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

    /** Set only for the duration of a {@link #submitDeferWait()} call, read by the guest branch. */
    private boolean deferOrderedWait = false;
    /** The handle {@link #submitDeferWait()} committed, awaiting {@link #awaitCommitted()}. */
    private volatile long pendingOrderedWait = 0;

    /**
     * GPU duration of the most recently completed command buffer, in milliseconds, or -1 when Metal
     * did not report timestamps.
     *
     * <p>This is the reading {@code [Metal-PERF]}'s {@code submit} field has been standing in for.
     * {@code submit} measures how long the CPU <em>waited</em> — {@code commit} plus
     * {@code waitUntilCompleted}, which on the guest branch also spans {@code MetallumBridge.flushFrame()}
     * and the deferred ordered index-wait in front of it. This measures how long the GPU <em>worked</em>.
     * When the two disagree, the gap is queueing and other work on the queue — and that gap is the
     * difference between "the GPU is the bottleneck" and "the CPU is waiting on someone else's GPU work".
     */
    private volatile double lastSubmitGpuMs = -1.0;

    /** See {@link #lastSubmitGpuMs}. -1 means unavailable, never "instantaneous". */
    public double lastSubmitGpuMs() {
        return this.lastSubmitGpuMs;
    }

    /**
     * Segment tags for {@link #captureGpuTime}. A frame commits more than one command buffer, and
     * WHICH one a duration belongs to is the difference between a usable budget and a single
     * unattributable number — the reading is only meaningful next to the name of its segment.
     *
     * <p>{@link #GPU_TAG_PRE_LOD} is the buffer committed by {@code submitDeferWait()} and waited on in
     * {@link #awaitCommitted()}: it carries the traversal, the five compute prepasses <em>and</em> the
     * Hi-Z pyramid build, everything encoded before the LOD pass begins.
     * {@link #GPU_TAG_POST_LOD} is the buffer the final {@code submit()} commits, which carries the
     * LOD render pass and nothing else.
     */
    public static final String GPU_TAG_PRE_LOD = "preLod";
    public static final String GPU_TAG_POST_LOD = "postLod";
    public static final String GPU_TAG_OWNED = "owned";

    /** Last GPU duration per segment tag, in ms; -1 when unavailable. */
    private final java.util.Map<String, Double> lastGpuByTag =
            new java.util.concurrent.ConcurrentHashMap<>();

    /** Last GPU duration recorded for {@code tag}, in ms, or -1 when there is none. */
    public double lastGpuMsFor(String tag) {
        Double v = this.lastGpuByTag.get(tag);
        return v == null ? -1.0 : v;
    }

    /**
     * Record the GPU duration of a command buffer that has just COMPLETED, in milliseconds.
     *
     * <p>Must be called after a wait and before release — Metal reports {@code GPUStartTime} /
     * {@code GPUEndTime} as 0.0 until the buffer completes, so calling this early records -1 rather
     * than failing, and a caller that ignored that would read "the GPU took no time".
     *
     * <p>{@code tag} names the segment the buffer carries; see the {@code GPU_TAG_*} constants. The
     * caller must pass the right one — a mislabelled duration is worse than no duration, because it
     * looks like an answer.
     */
    private void captureGpuTime(long cmdBuffer, String tag) {
        if (cmdBuffer == 0L) return;
        final double start = MetalNative.mtlCommandBufferGetGpuStartTime(cmdBuffer);
        final double end = MetalNative.mtlCommandBufferGetGpuEndTime(cmdBuffer);
        final double ms = (start > 0.0 && end > 0.0) ? (end - start) * 1000.0 : -1.0;
        this.lastGpuByTag.put(tag, ms);
        if (GPU_TAG_POST_LOD.equals(tag)) this.lastSubmitGpuMs = ms;
        if (GPU_TRACE) {
            me.cortex.voxy.common.Logger.info(String.format(
                    "[Metal-GPUTRACE] %-8s cb=%d start=%.3f end=%.3f span=%.2f ms",
                    tag, cmdBuffer, start * 1000.0, end * 1000.0, ms));
        }
    }

    /**
     * VOXY_GPU_TRACE=1: log every GPU-time capture with its segment tag, handle and window.
     *
     * <p>Off by default because it is one log line per commit — several per frame at 60 fps. It exists
     * because the untagged version of this instrument produced a reading of 9.81 ms for a segment that
     * contains only an empty render pass, and there was no way to tell from the output which segment
     * that number came from.
     */
    private static final boolean GPU_TRACE = "1".equals(System.getenv("VOXY_GPU_TRACE"));

    /**
     * Commit exactly as {@link #submit()} does, but leave the ordered wait for {@link #awaitCommitted()}.
     *
     * <p>Why this exists. The wait is load-bearing -- the guest branch commits without waiting, so the
     * CPU's per-draw read of {@code baseInstance} out of drawCallBuffer would otherwise be a frame
     * stale, which was bug 3. But WHERE it happens is free to move: the read it protects does not occur
     * until the draw, and the render pipeline does real CPU work in between (fog and clear setup, the
     * render-pass description, attachment resolution). Waiting immediately after the commit makes the
     * CPU idle through the GPU's prepasses and then do that work; deferring the wait lets the work
     * overlap them instead.
     *
     * <p>It cannot weaken the guarantee: the wait still happens before the first read, so the ordering
     * the fix establishes is unchanged. It only changes how much of it the CPU spends idle.
     */
    @Override
    public void submitDeferWait() {
        this.deferOrderedWait = true;
        try {
            this.submit();
        } finally {
            this.deferOrderedWait = false;
        }
    }

    /** Perform the wait {@link #submitDeferWait()} deferred. No-op when there is none. */
    @Override
    public void awaitCommitted() {
        final long h = this.pendingOrderedWait;
        if (h != 0L) {
            this.pendingOrderedWait = 0L;
            MetalNative.mtlCommandBufferWaitUntilCompleted(h);
            // The buffer submitDeferWait() committed: traversal + the five prepasses + the Hi-Z build.
            this.captureGpuTime(h, GPU_TAG_PRE_LOD);
        }
    }

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
            // NOT "queue idle". The old shortcut called mtlSharedEventSetSignaledValue here, on the
            // premise that "every commit path on this queue waits for completion" -- and the guest
            // branch of submit() does not (see its comment: it commits via flushFrame() and returns).
            // So after any guest submit(), activeCommandBuffer is 0 with up to MAX_SUBMITS_IN_FLIGHT
            // submits still executing, and this branch signalled the fence IMMEDIATELY. Worse, since
            // MTLSharedEvent's signaledValue is monotonic, one such jump-ahead permanently satisfies
            // every older outstanding fence -- not just the one being created.
            //
            // The functions that hit this are the ones whose entire job is to force a drain:
            // flushBackendFences() -> submit() -> activeCommandBuffer = 0 -> createFence() is exactly
            // the path that triggers it. DownloadStream.flushWaitClear (with production callers),
            // RawDownloadStream.free() and the stream-full emergency loops all spin on a fence that was
            // already signalled, so they do no synchronisation at all despite their names.
            //
            // A committed signal-only buffer orders correctly by queue order: it cannot execute before
            // work committed earlier, so the fence means what its callers assume. This is the same
            // buffer the callerPassOpen() branch below commits when no copies are parked.
            long cmdBuffer = MetalNative.mtlCommandQueueNewCommandBuffer(this.commandQueue);
            MetalNative.mtlCommandBufferEncodeSignalEvent(cmdBuffer, this.sharedEvent, targetValue);
            MetalNative.mtlCommandBufferCommit(cmdBuffer);
            MetalNative.mtlRelease(cmdBuffer);
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

    // --- Capabilities ---

    @Override
    public boolean hasCompute() {
        return true;
    }

    @Override
    public boolean hasIndirectParameters() {
        return true;
    }

    @Override
    public long getMaxSSBOSize() {
        return this.maxBufferLength;
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
     * Copy one texture's region into another, on the GPU. Same encoder discipline as
     * {@link #copyBufferSubData}: encode into the active command buffer so the copy executes in request
     * order with the frame's passes, and take the blit encoder slot rather than opening one while a
     * caller's render pass is still encoding -- Metal refuses that, which is what broke the borrowed
     * encoder arrangement.
     *
     * <p>Used to copy Minecraft's depth attachment into a Voxy-owned sampleable texture for the Hi-Z
     * pyramid. MC's attachment is Depth32Float but is missing {@code MTLTextureUsageShaderRead}, so a
     * sampler reads zeros from it however correct its format is, and a blit is the only way across.
     */
    public void copyTextureToTexture(final long srcHandle, final long dstHandle,
                                     final int width, final int height) {
        if (srcHandle == 0 || dstHandle == 0) return;
        this.ensureActiveCommandBuffer();
        if (this.activeBlitEncoder == 0) {
            this.endForeignEncoderIfNeeded();
            this.activeBlitEncoder = MetalNative.mtlCommandBufferNewBlitEncoder(this.activeCommandBuffer);
            if (this.activeBlitEncoder == 0) {
                throw new RuntimeException("mtlCommandBufferNewBlitEncoder returned NULL");
            }
        }
        MetalNative.mtlBlitEncoderCopyTextureToTexture(this.activeBlitEncoder,
                srcHandle, 0, 0, 0, width, height, dstHandle, 0, 0, 0);
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


    /** Prefer a Metallum-untracked encoder for Voxy's passes; {@code VOXY_LOD_DETACHED_ENCODER=0} opts out. */
    /** Borrow Metallum's encoder instead of owning one; {@code VOXY_LOD_BORROW_ENCODER=1} for A/B. */
    static final boolean BORROW_ENCODER = "1".equals(System.getenv("VOXY_LOD_BORROW_ENCODER"));




    private void endForeignEncoderIfNeeded() {
        // Deliberately NOT gated on ownsActiveCommandBuffer: that flag is decided once, when the
        // active buffer is first created -- which can be before Metallum has a frame buffer at all.
        // Voxy would then believe it owns the buffer and skip this, while Metallum meanwhile opened
        // a render encoder on the same buffer. Asking Metallum to close is a no-op when it has
        // nothing open, so just always ask.
        // The two-way branch that used to stand here existed only to pick between the cheap call and a
        // reporting variant that fed the encoder trace. The trace is gone, so this is just the cheap
        // path -- which is what ran in every real build anyway: the reporting accessors are three
        // reflective invocations and this runs ~150k times per session, so the trace was never viable
        // on for a normal run.
        MetallumBridge.endCurrentEncoder();
    }

    private void endActiveBlitEncoder() {
        if (this.activeBlitEncoder != 0) {
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
            // VOXY_PASS_SLOT_PROBE=1: read back what the descriptor ACTUALLY holds for slot 1, rather
            // than trusting the builder's list for it. "The pass carries two attachments" has only ever
            // been asserted from the Java side -- the builder's intent, not the descriptor's state. If
            // slot 1 is bound to a different texture than the one the probe reads, or its store action
            // is not STORE, every symptom of the missing [[color(1)]] write follows with no error
            // anywhere. Capped at three lines so it is cheap; the LOD pass is the one with BOTH a
            // second colour attachment and a depth attachment.

            // Pending stream copies must land before the pass's encoder opens
            // (one encoder at a time per buffer; copies feed the pass anyway).
            this.endActiveBlitEncoder();
            this.ensureActiveCommandBuffer();
            // Must be logged AFTER ensureActiveCommandBuffer, which is what decides whether Voxy
            // encodes into Metallum's live frame buffer or keeps one of its own. If these differ,
            // a perfectly-formed pass (correct attachments, correct clear) lands in a command
            // buffer that is never the one presented -- which is indistinguishable from "the pass
            // does nothing" everywhere else in the logs.

            // Prefer BORROWING Metallum's encoder for these attachments. Asking Metal for a second
            // encoder on the same command buffer is illegal ("A command encoder is already encoding
            // to this command buffer"), and closing Metallum's to open our own costs a pass split for
            // no reason. Borrowing also means Voxy's draws land in Metallum's pass directly, sharing
            // its depth. Falls back to owning an encoder when Metallum is absent.
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
            if (shared != 0L) {
                encoder = shared;
                borrowed = true;
            } else {
                this.endForeignEncoderIfNeeded();
                encoder = MetalNative.mtlCommandBufferNewRenderEncoder(this.activeCommandBuffer, passDescHandle);
                if (encoder == 0) {
                    throw new RuntimeException("mtlCommandBufferNewRenderEncoder returned NULL");
                }
            }

            // The borrow result must be attributed to THIS pass. A depth-only pass (chunk-bound,
            // depth export) legitimately has colorHandle==0, and acquireRenderEncoder returns 0 for
            // it by design -- reading such a line as "the LOD pass failed to borrow" is wrong twice
            // over, and I made that mistake. Only a colour+depth pass is the LOD pass.
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
            // Every declared colour attachment gets an EXPLICIT write mask. Metal documents the
            // default as MTLColorWriteMaskAll and this sets exactly that, so it should be a no-op --
            // but note the asymmetry it corrects: blending above is configured for index 0 ONLY, so
            // before this loop NOTHING ever touched attachment 1's pipeline descriptor at all. That
            // made "attachment 1's write mask is All" an assumption rather than something the pipeline
            // states, and attachment 1 is precisely the output whose write never appears. Stating it
            // costs one JNI call per pipeline and removes the last unfalsifiable link.
            for (int i = 0; i < desc.colorAttachmentFormats.length; i++) {
                if (desc.colorAttachmentFormats[i] != 0) {
                    MetalNative.mtlRenderPipelineDescriptorSetColorAttachmentWriteMask(
                            pipelineDesc, i, MetalNative.MTLColorWriteMaskAll);
                }
            }
            // VOXY_PIPELINE_SLOT_PROBE=1: read the formats back OFF THE DESCRIPTOR, right before the
            // pipeline state is built from it. The pass half is already measured ([Metal-PASSSLOT]);
            // this is the other half, and it is the last link in the MRT chain that had only ever been
            // established by reading code rather than by reading state. A 0 here means "no attachment
            // at this index" as far as Metal is concerned.

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
                // Capture the buffer Metallum is about to commit, then wait on THAT one. Waiting on it
                // directly rather than through waitForGpuIdle() saves a commit and a CPU<->GPU round
                // trip per frame: waitForGpuIdle() allocates a fresh buffer and waits on that, which
                // covers the same work (on a serial queue, waiting for the newest committed buffer
                // already implies every older one) at strictly higher cost.
                //
                // Do NOT release `committed`. It is Metallum's buffer -- Metallum holds it in its own
                // in-flight slot and closes it in a later submit(); releasing it here would be a
                // use-after-free. waitForGpuIdle() releases only because that buffer is its own.
                // waitUntilCompleted is safe to call as a second waiter on a buffer that also carries a
                // completion block; it neither consumes nor invalidates anything.
                final long committed = this.activeCommandBuffer;
                MetallumBridge.flushFrame();
                this.activeCommandBuffer = 0;
                if (SUBMIT_ORDER && committed != 0L) {
                    // Why this wait is load-bearing at all: flushFrame() COMMITS the frame but does not
                    // wait, and the comment above claiming the work is "committed and complete, making
                    // it CPU-visible for the draw path's baseInstance read" is false without this.
                    //
                    // Metallum's encoder commits with a completion block and then awaits
                    // `currentSubmitIndex - MAX_SUBMITS_IN_FLIGHT`, i.e. three submits BACK, from a
                    // counter that STARTS at 3 -- so the first three submits wait for nothing. The owned
                    // branch below really does waitUntilCompleted, which is why this looked safe.
                    //
                    // Without it, MetalRenderEncoder reads `baseInstance` out of drawCallBuffer on the
                    // CPU and pushes it as the per-draw constant, so a stale read gives the WRONG
                    // SECTION'S ORIGIN while the draw keeps its own quads and baked light -- bug 3's
                    // symptom exactly, including the preserved lighting. Confirmed fixed by eye.
                    if (this.deferOrderedWait) {
                        // The caller wants the commit now and the wait later, so it can spend the
                        // interval on CPU work that overlaps the GPU's prepasses. See submitDeferWait.
                        // The GPU time is captured in awaitCommitted(), where the wait actually lands.
                        this.pendingOrderedWait = committed;
                    } else {
                        MetalNative.mtlCommandBufferWaitUntilCompleted(committed);
                        // The buffer this submit() just committed: the LOD render pass alone.
                        this.captureGpuTime(committed, GPU_TAG_POST_LOD);
                    }
                }
                return;
            }
            this.activeCommandBuffer = 0;
            return;
        }

        MetalNative.mtlCommandBufferCommit(this.activeCommandBuffer);
        // Sync mode for M3: wait for completion so the smoke test can check status
        // before the buffer is released. M5+ will move to async + per-frame fences.
        MetalNative.mtlCommandBufferWaitUntilCompleted(this.activeCommandBuffer);
        // Capture BEFORE the release below -- the timestamps live on the command buffer.
        this.captureGpuTime(this.activeCommandBuffer, GPU_TAG_OWNED);
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
