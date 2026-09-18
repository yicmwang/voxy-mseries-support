package me.cortex.voxy.client.core.metal;

import me.cortex.voxy.client.core.gpu.GraphicsPipelineDesc;
import me.cortex.voxy.client.core.gpu.IGpuBuffer;
import me.cortex.voxy.client.core.gpu.IGpuIndirectCommandBuffer;
import me.cortex.voxy.client.core.gpu.IGpuPipeline;
import me.cortex.voxy.client.core.gpu.IGpuTexture;
import me.cortex.voxy.client.core.gpu.RenderEncoder;
import me.cortex.voxy.client.core.gpu.RenderPassDesc;
import me.cortex.voxy.client.core.gpu.VertexLayout;
import me.cortex.voxy.client.core.gpu.shader.RuntimeShaderCompiler;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.lwjgl.system.MemoryUtil;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Exercises {@code MTLIndirectCommandBuffer} + {@code executeCommandsInBuffer} on real hardware.
 *
 * <p>This is the foundation the P3 draw-path switch depends on. It was previously covered only by
 * {@code MetalIcbSmokeTest}, a {@code main()} program that had to be run by hand; that meant nothing
 * guarded the path during ordinary refactoring.
 *
 * <p>The second test answers the question that motivated the whole ICB plan: Metal's
 * {@code drawIndexedPrimitives:indirectBuffer:} does <b>not</b> propagate {@code baseInstance}
 * (Voxy works around it by CPU-reading every command, which is what forces a mid-frame GPU sync and
 * blocks the Metallum rewire). If ICB commands propagate it instead, the CPU loop and the sync can
 * both go away.
 */
class MetalIndirectCommandBufferTest {

    private static final int W = 256;
    private static final int H = 256;
    private static final int GL_RGBA8 = 0x8058;
    private static final int GL_TEXTURE_2D = 0x0DE1;
    private static final int VERTEX_STRIDE = 20; // vec2 pos + vec3 colour

    private static final int MTL_INDEX_UINT16 = 0;
    private static final int MTL_PRIMITIVE_TRIANGLE = 3;

    private static MetalRenderBackend backend;
    private static IGpuTexture target;

    @BeforeAll
    static void requireMetal() {
        assumeTrue(MetalNative.load(), "Metal native library unavailable");
        try {
            backend = new MetalRenderBackend();
        } catch (Throwable t) {
            assumeTrue(false, "Metal backend unavailable: " + t.getMessage());
        }
        target = backend.createTexture(GL_TEXTURE_2D);
        target.store(GL_RGBA8, 1, W, H);
        target.name("voxy-icb-test-target");
    }

    @AfterAll
    static void teardown() {
        if (target != null) {
            target.free();
        }
        if (backend != null) {
            backend.submit();
        }
    }

    // ---------------------------------------------------------------- helpers

    private static String vertexGlsl(String colorExpr) {
        return """
                #version 460 core
                #extension GL_ARB_shader_draw_parameters : require
                layout(location = 0) in vec2 inPos;
                layout(location = 1) in vec3 inColor;
                layout(location = 0) out vec3 vColor;
                void main() {
                    gl_Position = vec4(inPos, 0.0, 1.0);
                    vColor = %s;
                }
                """.formatted(colorExpr);
    }

    private static final String FRAG_GLSL = """
            #version 460 core
            layout(location = 0) in vec3 vColor;
            layout(location = 0) out vec4 fragColor;
            void main() { fragColor = vec4(vColor, 1.0); }
            """;

    private static VertexLayout layout() {
        return VertexLayout.builder()
                .buffer(0, VERTEX_STRIDE, VertexLayout.StepRate.PER_VERTEX)
                .attribute(0, VertexLayout.VertexFormat.FLOAT2, 0, 0)
                .attribute(1, VertexLayout.VertexFormat.FLOAT3, 8, 0)
                .build();
    }

    private static IGpuPipeline buildPipeline(String vertGlsl, String label) {
        RuntimeShaderCompiler.Result vert = RuntimeShaderCompiler.compile(vertGlsl,
                RuntimeShaderCompiler.Stage.VERTEX, Map.of(), RuntimeShaderCompiler.Target.METAL_MSL);
        RuntimeShaderCompiler.Result frag = RuntimeShaderCompiler.compile(FRAG_GLSL,
                RuntimeShaderCompiler.Stage.FRAGMENT, Map.of(), RuntimeShaderCompiler.Target.METAL_MSL);
        return backend.createGraphicsPipeline(new GraphicsPipelineDesc(
                vert.mslSource(), frag.mslSource(), vert.spirv(), frag.spirv(),
                GL_RGBA8, layout(), label).withIndirectCommandBufferUsage(true));
    }

    private static IGpuBuffer uploadFloats(float... values) {
        IGpuBuffer buf = backend.createBuffer((long) values.length * 4);
        ByteBuffer bb = MemoryUtil.memByteBuffer(((MetalBuffer) buf).getContentsPtr(), values.length * 4)
                .order(ByteOrder.LITTLE_ENDIAN);
        for (float v : values) {
            bb.putFloat(v);
        }
        return buf;
    }

    private static IGpuBuffer uploadTriangleIndices() {
        IGpuBuffer buf = backend.createBuffer(6);
        ByteBuffer bb = MemoryUtil.memByteBuffer(((MetalBuffer) buf).getContentsPtr(), 6)
                .order(ByteOrder.LITTLE_ENDIAN);
        bb.putShort((short) 0).putShort((short) 1).putShort((short) 2);
        return buf;
    }

    private static IGpuBuffer rangeBuffer(int location, int length) {
        // MTLIndirectCommandBufferExecutionRange { uint32 location; uint32 length; }, padded to 16.
        IGpuBuffer buf = backend.createBuffer(16);
        ByteBuffer bb = MemoryUtil.memByteBuffer(((MetalBuffer) buf).getContentsPtr(), 16)
                .order(ByteOrder.LITTLE_ENDIAN);
        bb.putInt(location).putInt(length).putInt(0).putInt(0);
        return buf;
    }

    private static int sampleRgba(byte[] px, int x, int y) {
        int i = (y * W + x) * 4;
        return ((px[i] & 0xFF) << 24) | ((px[i + 1] & 0xFF) << 16) | ((px[i + 2] & 0xFF) << 8) | (px[i + 3] & 0xFF);
    }

    private static int redOf(int rgba) {
        return (rgba >>> 24) & 0xFF;
    }

    private static final int CLEAR_R = 26; // 0.1 * 255

    // ------------------------------------------------------------------ tests

    @Test
    void executesCpuEncodedIndexedDrawThroughAnIcb() {
        IGpuBuffer vertexBuf = uploadFloats(
                -0.6f, -0.5f, 1.0f, 0.2f, 0.2f,
                 0.6f, -0.5f, 0.2f, 1.0f, 0.2f,
                 0.0f,  0.7f, 0.2f, 0.2f, 1.0f);
        IGpuBuffer indexBuf = uploadTriangleIndices();
        IGpuBuffer rangeBuf = rangeBuffer(0, 1);
        IGpuIndirectCommandBuffer icb = null;
        IGpuPipeline pipeline = null;
        try {
            pipeline = buildPipeline(vertexGlsl("inColor"), "voxy:test/icb-basic");

            icb = new MetalIndirectCommandBuffer(backend.device(), 2,
                    /*inheritBuffers=*/false, /*inheritPipelineState=*/false)
                    .name("voxy-test-icb");
            // Explicit encodings rather than inheritance: inheritance needs the PSO built with
            // supportIndirectCommandBuffers, and explicit is what MDIC's prepass path will do too.
            MetalIndirectCommandBuffer mIcb = (MetalIndirectCommandBuffer) icb;
            mIcb.reset(0, 2);
            assertTrue(mIcb.hasCommand(0), "ICB slot 0 must be reachable after reset");

            long psoHandle = ((MetalGraphicsPipeline) pipeline).pipelineStateHandle();
            long vbHandle = MetalHandleMap.getHandle(vertexBuf.id());
            long ibHandle = MetalHandleMap.getHandle(indexBuf.id());
            mIcb.encodeSetPipelineState(0, psoHandle);
            mIcb.encodeSetVertexBuffer(0, vbHandle, 0L, 0);
            mIcb.encodeDrawIndexedPrimitives(0, MTL_PRIMITIVE_TRIANGLE, 3, MTL_INDEX_UINT16,
                    ibHandle, 0L, 1, 0, 0);

            RenderPassDesc pass = RenderPassDesc.builder(W, H)
                    .clearColor(target, 0.1f, 0.1f, 0.15f, 1.0f)
                    .build();
            try (RenderEncoder enc = backend.beginRenderPass(pass)) {
                enc.setPipeline(pipeline);
                enc.setViewport(0, 0, W, H, 0, 1);
                enc.setScissor(0, 0, W, H);
                enc.bindVertexBuffer(0, vertexBuf, 0);
                // Metal cannot see resources an ICB uses indirectly; declare them.
                long encHandle = ((MetalRenderEncoder) enc).handle();
                MetalNative.mtlRenderEncoderUseResource(encHandle, ibHandle,
                        MetalNative.MTLResourceUsageRead, MetalNative.MTLRenderStageVertex);
                MetalNative.mtlRenderEncoderUseResource(encHandle, vbHandle,
                        MetalNative.MTLResourceUsageRead, MetalNative.MTLRenderStageVertex);
                enc.executeCommandsInBuffer(icb, rangeBuf, 0);
            }
            backend.submit();

            byte[] px = backend.readPixelsRGBA8(target, 0, 0, W, H);
            int topLeft = sampleRgba(px, 4, 4);
            int botRight = sampleRgba(px, W - 4, H - 4);
            int interior = sampleRgba(px, W / 2, H * 5 / 8);

            assertEquals(CLEAR_R, redOf(topLeft), "background corner must stay cleared");
            assertEquals(CLEAR_R, redOf(botRight), "background corner must stay cleared");
            assertNotEquals(CLEAR_R, redOf(interior),
                    "triangle interior unchanged — executeCommandsInBuffer never reached the rasterizer");
        } finally {
            if (icb != null) icb.close();
            vertexBuf.free();
            indexBuf.free();
            rangeBuf.free();
        }
    }

    @Test
    void icbCommandsPropagateBaseInstance() {
        // The motivating question. Voxy's workaround exists because
        // drawIndexedPrimitives:indirectBuffer: drops baseInstance (gl_BaseInstance reads 0),
        // forcing a CPU read of every command. If an ICB command carries it, that loop — and the
        // mid-frame sync it requires — can go.
        final int baseInstance = 200;

        IGpuBuffer vertexBuf = uploadFloats(
                -0.9f, -0.9f, 0f, 0f, 0f,
                 0.9f, -0.9f, 0f, 0f, 0f,
                 0.0f,  0.9f, 0f, 0f, 0f);
        IGpuBuffer indexBuf = uploadTriangleIndices();
        IGpuBuffer rangeBuf = rangeBuffer(0, 1);
        IGpuIndirectCommandBuffer icb = null;
        IGpuPipeline pipeline = null;
        try {
            // Colour is derived entirely from gl_BaseInstance, so the readback is the assertion.
            pipeline = buildPipeline(
                    vertexGlsl("vec3(float(gl_BaseInstance) / 255.0, 0.0, 0.0)"),
                    "voxy:test/icb-baseinstance");

            icb = new MetalIndirectCommandBuffer(backend.device(), 1,
                    /*inheritBuffers=*/false, /*inheritPipelineState=*/false)
                    .name("voxy-test-icb-bi");
            MetalIndirectCommandBuffer mIcb = (MetalIndirectCommandBuffer) icb;
            mIcb.reset(0, 1);

            long psoHandle = ((MetalGraphicsPipeline) pipeline).pipelineStateHandle();
            long vbHandle = MetalHandleMap.getHandle(vertexBuf.id());
            long ibHandle = MetalHandleMap.getHandle(indexBuf.id());
            mIcb.encodeSetPipelineState(0, psoHandle);
            mIcb.encodeSetVertexBuffer(0, vbHandle, 0L, 0);
            // baseVertex = 0, baseInstance = 200
            mIcb.encodeDrawIndexedPrimitives(0, MTL_PRIMITIVE_TRIANGLE, 3, MTL_INDEX_UINT16,
                    ibHandle, 0L, 1, 0, baseInstance);

            RenderPassDesc pass = RenderPassDesc.builder(W, H)
                    .clearColor(target, 0.1f, 0.1f, 0.15f, 1.0f)
                    .build();
            try (RenderEncoder enc = backend.beginRenderPass(pass)) {
                enc.setPipeline(pipeline);
                enc.setViewport(0, 0, W, H, 0, 1);
                enc.setScissor(0, 0, W, H);
                enc.bindVertexBuffer(0, vertexBuf, 0);
                long encHandle = ((MetalRenderEncoder) enc).handle();
                MetalNative.mtlRenderEncoderUseResource(encHandle, ibHandle,
                        MetalNative.MTLResourceUsageRead, MetalNative.MTLRenderStageVertex);
                MetalNative.mtlRenderEncoderUseResource(encHandle, vbHandle,
                        MetalNative.MTLResourceUsageRead, MetalNative.MTLRenderStageVertex);
                enc.executeCommandsInBuffer(icb, rangeBuf, 0);
            }
            backend.submit();

            byte[] px = backend.readPixelsRGBA8(target, 0, 0, W, H);
            int interior = redOf(sampleRgba(px, W / 2, H / 2));

            assertEquals(baseInstance, interior,
                    "ICB must propagate baseInstance to gl_BaseInstance; got red=" + interior
                            + " (0 would mean it is dropped exactly like the indirectBuffer path)");
        } finally {
            if (icb != null) icb.close();
            vertexBuf.free();
            indexBuf.free();
            rangeBuf.free();
        }
    }
}
