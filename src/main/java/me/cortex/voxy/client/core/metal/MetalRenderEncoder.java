package me.cortex.voxy.client.core.metal;

import me.cortex.voxy.client.core.gpu.IGpuBuffer;
import me.cortex.voxy.client.core.gpu.IGpuPipeline;
import me.cortex.voxy.client.core.gpu.IGpuSampler;
import me.cortex.voxy.client.core.gpu.IGpuTexture;
import me.cortex.voxy.client.core.gpu.RenderEncoder;

import org.lwjgl.system.MemoryUtil;

/**
 * Metal-side {@link RenderEncoder}. Wraps a single MTLRenderCommandEncoder
 * for the duration of a render pass; close() ends encoding and releases.
 *
 * Operations are delegated through the JNI surface in {@link MetalNative};
 * this class only translates the backend-agnostic primitive constants and
 * downcasts {@link IGpuPipeline} to its Metal-specific implementation.
 */
public final class MetalRenderEncoder implements RenderEncoder {

    private long encoderHandle;
    /** Index buffer remembered between bindIndexBuffer() and drawIndexed(). */
    private long boundIndexBuffer;
    private long boundIndexBufferOffset;
    private int boundIndexType = MetalNative.MTLIndexTypeUInt32;

    /** Native MTLRenderCommandEncoder handle. Exposed for ICB callers that need to declare useResource. */
    public long handle() {
        return this.encoderHandle;
    }

    /**
     * True when this encoder belongs to Metallum and is merely borrowed for Voxy's draws. Closing a
     * borrowed encoder must NOT end it: Metallum keeps drawing on it afterwards. Instead it marks
     * Metallum's pass state stale so Metallum rebinds before its next draw.
     */
    private final boolean borrowed;
    MetalRenderEncoder(long encoderHandle) {
        this(encoderHandle, false);
    }

    MetalRenderEncoder(long encoderHandle, boolean borrowed) {
        this.encoderHandle = encoderHandle;
        this.borrowed = borrowed;
    }

    @Override
    public void setPipeline(IGpuPipeline pipeline) {
        if (!(pipeline instanceof MetalGraphicsPipeline mp)) {
            throw new IllegalArgumentException(
                    "MetalRenderEncoder.setPipeline expected MetalGraphicsPipeline, got "
                            + (pipeline == null ? "null" : pipeline.getClass().getName()));
        }
        MetalNative.mtlRenderEncoderSetRenderPipelineState(this.encoderHandle, mp.pipelineStateHandle());
        // Depth-stencil, cull, winding, and triangle fill mode live on the encoder
        // (not on the render pipeline state object) — apply them every time the
        // pipeline changes so the encoder picks up the per-pipeline raster state.
        if (mp.depthStencilStateHandle() != 0) {
            MetalNative.mtlRenderEncoderSetDepthStencilState(this.encoderHandle, mp.depthStencilStateHandle());
        }
        MetalNative.mtlRenderEncoderSetCullMode(this.encoderHandle, mp.cullMode);
        MetalNative.mtlRenderEncoderSetFrontFacingWinding(this.encoderHandle, mp.winding);
        MetalNative.mtlRenderEncoderSetTriangleFillMode(this.encoderHandle, mp.fillMode);
    }

    @Override
    public void setBuffer(int binding, IGpuBuffer buffer, long offset) {
        long handle = bufferHandle(buffer);
        // Voxy GLSL exposes the same binding to vertex + fragment, so bind both.
        // Cost is one redundant native call when only one stage reads the buffer;
        // on Metal that's cheap and the pattern matches Voxy's semantics today.
        MetalNative.mtlRenderEncoderSetVertexBuffer(this.encoderHandle, handle, offset, binding);
        MetalNative.mtlRenderEncoderSetFragmentBuffer(this.encoderHandle, handle, offset, binding);
    }

    @Override
    public void setTexture(int binding, IGpuTexture texture) {
        long handle = texture == null ? 0 : MetalHandleMap.getHandle(texture.id());
        MetalNative.mtlRenderEncoderSetVertexTexture(this.encoderHandle, handle, binding);
        MetalNative.mtlRenderEncoderSetFragmentTexture(this.encoderHandle, handle, binding);
    }

    @Override
    public void setSampler(int binding, IGpuSampler sampler) {
        long handle = 0;
        if (sampler != null) {
            if (!(sampler instanceof MetalSampler ms)) {
                throw new IllegalArgumentException("MetalRenderEncoder.setSampler expected MetalSampler, got "
                        + sampler.getClass().getName());
            }
            handle = ms.handle();
        }
        MetalNative.mtlRenderEncoderSetVertexSamplerState(this.encoderHandle, handle, binding);
        MetalNative.mtlRenderEncoderSetFragmentSamplerState(this.encoderHandle, handle, binding);
    }

    @Override
    public void setBytes(int binding, long dataAddr, int dataSize) {
        MetalNative.mtlRenderEncoderSetVertexBytes(this.encoderHandle, dataAddr, dataSize, binding);
        MetalNative.mtlRenderEncoderSetFragmentBytes(this.encoderHandle, dataAddr, dataSize, binding);
    }

    @Override
    public void bindVertexBuffer(int slot, IGpuBuffer buffer, long offset) {
        long handle = bufferHandle(buffer);
        MetalNative.mtlRenderEncoderSetVertexBuffer(this.encoderHandle, handle, offset, slot);
    }

    @Override
    public void bindIndexBuffer(IGpuBuffer buffer, int indexType, long offset) {
        this.boundIndexBuffer = bufferHandle(buffer);
        this.boundIndexBufferOffset = offset;
        this.boundIndexType = switch (indexType) {
            case INDEX_TYPE_UINT16 -> MetalNative.MTLIndexTypeUInt16;
            case INDEX_TYPE_UINT32 -> MetalNative.MTLIndexTypeUInt32;
            default -> throw new IllegalArgumentException("Unsupported index type: " + indexType);
        };
    }

    @Override
    public void setViewport(float x, float y, float width, float height,
                             float minDepth, float maxDepth) {
        MetalNative.mtlRenderEncoderSetViewport(this.encoderHandle,
                x, y, width, height, minDepth, maxDepth);
    }

    @Override
    public void setScissor(int x, int y, int width, int height) {
        MetalNative.mtlRenderEncoderSetScissorRect(this.encoderHandle, x, y, width, height);
    }

    @Override
    public void draw(int primitiveType, int firstVertex, int vertexCount,
                     int instanceCount, int baseInstance) {
        int metalPrimitive = mapPrimitiveType(primitiveType);
        MetalNative.mtlRenderEncoderDrawPrimitives(this.encoderHandle,
                metalPrimitive, firstVertex, vertexCount, instanceCount, baseInstance);
    }

    @Override
    public void drawIndexed(int primitiveType, int indexCount, int instanceCount,
                             int firstIndex, int vertexOffset, int firstInstance) {
        if (this.boundIndexBuffer == 0) {
            throw new IllegalStateException("drawIndexed() before bindIndexBuffer()");
        }
        int metalPrimitive = mapPrimitiveType(primitiveType);
        // firstIndex is fed into the index buffer offset because Metal's
        // drawIndexedPrimitives doesn't take a firstIndex argument; it's
        // baked into indexBufferOffset. indexType drives the multiplier.
        long indexBytes = this.boundIndexType == MetalNative.MTLIndexTypeUInt16 ? 2L : 4L;
        long offset = this.boundIndexBufferOffset + (long) firstIndex * indexBytes;
        MetalNative.mtlRenderEncoderDrawIndexedPrimitives(this.encoderHandle,
                metalPrimitive, indexCount, this.boundIndexType,
                this.boundIndexBuffer, offset,
                instanceCount, vertexOffset, firstInstance);
    }

    @Override
    public void drawIndirect(int primitiveType, IGpuBuffer buffer, long offset,
                              int drawCount, int stride) {
        long indirectBuf = bufferHandle(buffer);
        if (indirectBuf == 0) throw new IllegalArgumentException("drawIndirect: indirect buffer is null");
        int metalPrimitive = mapPrimitiveType(primitiveType);
        // Metal lacks native multi-draw-indirect; loop on the host. Each iteration
        // dispatches one indirect draw at offset + i*stride. For drawCount=1 this
        // is a single call; for larger counts we accept the per-call overhead
        // until M11+ wires up an MTLIndirectCommandBuffer cache.
        for (int i = 0; i < drawCount; i++) {
            MetalNative.mtlRenderEncoderDrawPrimitivesIndirect(this.encoderHandle,
                    metalPrimitive, indirectBuf, offset + (long) i * stride);
        }
    }

    @Override
    public void drawIndexedIndirect(int primitiveType, IGpuBuffer buffer, long offset,
                                     int drawCount, int stride) {
        if (this.boundIndexBuffer == 0) {
            throw new IllegalStateException("drawIndexedIndirect() before bindIndexBuffer()");
        }
        long indirectBuf = bufferHandle(buffer);
        if (indirectBuf == 0) throw new IllegalArgumentException("drawIndexedIndirect: indirect buffer is null");
        int metalPrimitive = mapPrimitiveType(primitiveType);

        // M13 2026-05-14 workaround: drawIndexedPrimitives:indirectBuffer: does
        // NOT propagate the indirect args' baseInstance to [[base_instance]]
        // in the vertex function on Metal. Diagnosed via shader probes —
        // gl_BaseInstance always reads 0. Push the per-draw baseInstance
        // value as inline constant bytes at vertex binding 6 via
        // setVertexBytes before each draw; the shader reads it from the
        // VoxyMetalPerDrawUBO uniform (gated by VOXY_METAL_BI_FIX).
        // Requires the indirect buffer to be Shared storage so CPU can
        // read it (Voxy's MetalRenderBackend.createBuffer uses Shared by
        // default). Caller must ensure the compute prepass that wrote
        // drawCallBuffer has been flushed before this draw — Voxy's
        // submit-and-wait between buildDrawCalls and renderTerrainMetal
        // handles that on Metal.
        long indirectContents = 0;
        if (buffer instanceof MetalBuffer mb) {
            indirectContents = mb.getContentsPtr();
        }
        long perDrawScratchAddr = MemoryUtil.memAddress(this.perDrawScratch);
        // VOXY_BI_TRACE=1: the push below is skipped entirely when indirectContents == 0, i.e.
        // when the indirect buffer is not CPU-visible. In that case no setVertexBytes happens for
        // ANY draw, every vertex reads the same stale voxyMetalDrawIndex, and the geometry
        // collapses to one quad -- thousands of valid indices rasterizing nothing. Log whether
        // the push actually happens and what it pushes.
        if (BI_TRACE && (biTraceCount++ % 600) == 1) {
        }
        for (int i = 0; i < drawCount; i++) {
            long cmdAddr = offset + (long) i * stride;
            if (BI_OFFSET && this.biOffsetBuffer != null && this.biOffsetBinding >= 0
                    && indirectContents != 0) {
                // The index selects WHICH entry of the position buffer is read, so binding the buffer
                // at that entry's offset makes the read correct with no push at all.
                int baseInstance = MemoryUtil.memGetInt(indirectContents + cmdAddr + 16);
                this.setBuffer(this.biOffsetBinding, this.biOffsetBuffer, (long) baseInstance * 8L);
            } else if (indirectContents != 0) {
                int baseInstance = MemoryUtil.memGetInt(indirectContents + cmdAddr + 16);
                MemoryUtil.memPutInt(perDrawScratchAddr, baseInstance);
                MetalNative.mtlRenderEncoderSetVertexBytes(this.encoderHandle,
                        perDrawScratchAddr, 16, VOXY_METAL_PER_DRAW_UBO_BINDING);
            }
            MetalNative.mtlRenderEncoderDrawIndexedPrimitivesIndirect(this.encoderHandle,
                    metalPrimitive, this.boundIndexType,
                    this.boundIndexBuffer, this.boundIndexBufferOffset,
                    indirectBuf, cmdAddr);
        }
    }

    /**
     * VOXY_BI_OFFSET=1: deliver the per-draw section index as a BUFFER OFFSET instead of a pushed
     * constant.
     *
     * The existing path pushes `cmd.baseInstance` per draw with setVertexBytes (see the workaround note
     * above). A per-draw constant is the one place a wrong section index can reach the vertex shader,
     * and it is the one link in this renderer no counter has ever verified. It is also UNTESTABLE by
     * observation: if the driver coalesces the constant, the shader receives a neighbouring draw's value
     * and then uses it consistently for both the position and everything derived from it, so the output
     * is internally consistent and no readout can tell it from correct behaviour.
     *
     * So the only test it admits is removing the mechanism. Binding the position buffer at an offset of
     * `baseInstance * 8` puts the index in core per-draw Metal state, which is not a constant and cannot
     * be coalesced the same way; the shader then reads index 0. One build, one env switch, so the two
     * arms are directly comparable.
     */
    private static final boolean BI_OFFSET = "1".equals(System.getenv("VOXY_BI_OFFSET"));
    private IGpuBuffer biOffsetBuffer;
    private int biOffsetBinding = -1;

    /** Register the buffer whose entry `baseInstance` selects, for the BI_OFFSET path. */
    public void setPerDrawIndexBuffer(int binding, IGpuBuffer buffer) {
        this.biOffsetBinding = binding;
        this.biOffsetBuffer = buffer;
    }

    /** Scratch buffer for per-draw setVertexBytes uniform (16 bytes std140). */
    private final java.nio.ByteBuffer perDrawScratch =
            org.lwjgl.system.MemoryUtil.memAlloc(16).order(java.nio.ByteOrder.nativeOrder());

    /** Vertex-buffer binding slot used by the per-draw baseInstance workaround. */
    private static final int VOXY_METAL_PER_DRAW_UBO_BINDING = 6;

    /** Experimental: end borrowed encoders too ({@code VOXY_END_BORROWED_ENCODER=1}). */
    private static final boolean END_BORROWED_ENCODER =
            "1".equals(System.getenv("VOXY_END_BORROWED_ENCODER"));

    /** Opt-in per-draw baseInstance trace ({@code VOXY_BI_TRACE=1}). */
    private static final boolean BI_TRACE = "1".equals(System.getenv("VOXY_BI_TRACE"));
    private static long biTraceCount = 0;

    @Override
    public void drawIndexedIndirectCount(int primitiveType,
                                          IGpuBuffer drawBuffer, long drawOffset,
                                          IGpuBuffer countBuffer, long countOffset,
                                          int maxDrawCount, int stride) {
        // Direct count-aware draw on Metal still requires the ICB indirection:
        // a compute prepass must translate the flat (drawBuf, countBuf) pair
        // into an MTLIndirectCommandBuffer + range. The encoder API exposes
        // executeCommandsInBuffer for that workflow; callers that need
        // count-aware draws on Metal go through there. Direct lowering of
        // this signature would need a runtime CPU readback of countBuffer,
        // which defeats the purpose (the whole point is GPU-resident count).
        throw new UnsupportedOperationException(
                "MetalRenderEncoder.drawIndexedIndirectCount: Metal has no direct count-aware MDI. "
                        + "Use createIndirectCommandBuffer + executeCommandsInBuffer instead "
                        + "(populate the ICB via a compute prepass). maxDrawCount=" + maxDrawCount);
    }

    @Override
    public void executeCommandsInBuffer(me.cortex.voxy.client.core.gpu.IGpuIndirectCommandBuffer icb,
                                         IGpuBuffer rangeBuffer, long rangeOffset) {
        if (!(icb instanceof MetalIndirectCommandBuffer m)) {
            throw new IllegalArgumentException(
                    "MetalRenderEncoder.executeCommandsInBuffer requires MetalIndirectCommandBuffer, got "
                            + (icb == null ? "null" : icb.getClass().getName()));
        }
        long rangeBufHandle = bufferHandle(rangeBuffer);
        if (rangeBufHandle == 0) {
            throw new IllegalArgumentException("executeCommandsInBuffer: range buffer is null");
        }
        MetalNative.mtlRenderEncoderExecuteCommandsInBuffer(
                this.encoderHandle, m.handle(), rangeBufHandle, rangeOffset);
    }

    /**
     * What {@link #close()} must do with the encoder, which depends entirely on who owns it.
     *
     * <p>Extracted so the rule can be unit-tested without a GPU: getting it wrong is what made
     * Voxy's LOD render nothing. An encoder Metallum hands out through its shared-encoder path is
     * tracked by Metallum, which may end it before the borrower is done with it; one Metallum knows
     * nothing about must be ended by us, or its work is discarded when the command buffer commits.
     */
    enum CloseAction {
        /** Borrowed: Metallum keeps drawing on it; only mark its cached state stale. */
        INVALIDATE_ONLY,
        /** Voxy's own: end it and release our reference. */
        END_AND_RELEASE
    }

    static CloseAction closeActionFor(final boolean borrowed) {
        return borrowed ? CloseAction.INVALIDATE_ONLY : CloseAction.END_AND_RELEASE;
    }

    @Override
    public void close() {
        if (this.encoderHandle == 0) return;
        if (this.borrowed) {
            // Shared with Metallum: leave the encoder open and let Metallum rebind its own state.
            //
            // VOXY_END_BORROWED_ENCODER=1 -- experiment. "Borrowed" covers two different things:
            // (a) an encoder Metallum had open for its own live pass, which must NOT be ended, and
            // (b) a FRESH encoder created for Voxy by renderCommandEncoderForHandles' slow path.
            // Case (b) happens whenever Metallum has no matching encoder open -- which is exactly
            // the state after the mid-frame submit. If that fresh encoder is never ended, Metal
            // discards everything encoded into it, which is what "no fragment ever rasterizes,
            // even a forced screen-covering triangle" looks like.
            if (END_BORROWED_ENCODER) {
                MetalNative.mtlEncoderEndEncoding(this.encoderHandle);
                MetalNative.mtlRelease(this.encoderHandle);
            } else {
                MetallumBridge.invalidateRenderPassState();
            }
        } else {
            MetalNative.mtlEncoderEndEncoding(this.encoderHandle);
            MetalNative.mtlRelease(this.encoderHandle);
        }
        this.encoderHandle = 0;
        // Free the per-instance off-heap scratch buffer (MemoryUtil.memAlloc in the
        // field initializer). It is native memory, NOT GC-tracked, so without this
        // every render pass leaks 16 bytes that never shows up in JVM heap stats.
        // The encoderHandle!=0 early-return above makes this run exactly once, so a
        // double close() cannot double-free.
        org.lwjgl.system.MemoryUtil.memFree(this.perDrawScratch);
    }

    private static long bufferHandle(IGpuBuffer buffer) {
        if (buffer == null) return 0;
        if (!(buffer instanceof MetalBuffer mb)) {
            throw new IllegalArgumentException(
                    "MetalRenderEncoder expected MetalBuffer, got " + buffer.getClass().getName());
        }
        return mb.handle();
    }

    private static int mapPrimitiveType(int abstractType) {
        return switch (abstractType) {
            case PRIMITIVE_TRIANGLES -> MetalNative.MTLPrimitiveTypeTriangle;
            case PRIMITIVE_TRIANGLE_STRIP -> MetalNative.MTLPrimitiveTypeTriangleStrip;
            case PRIMITIVE_LINES -> MetalNative.MTLPrimitiveTypeLine;
            case PRIMITIVE_POINTS -> MetalNative.MTLPrimitiveTypePoint;
            default -> throw new IllegalArgumentException("Unsupported primitive type: " + abstractType);
        };
    }
}
