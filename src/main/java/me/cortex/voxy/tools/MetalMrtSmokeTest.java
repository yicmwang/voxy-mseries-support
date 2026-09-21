package me.cortex.voxy.tools;

import me.cortex.voxy.client.core.gpu.GraphicsPipelineDesc;
import me.cortex.voxy.client.core.gpu.IGpuPipeline;
import me.cortex.voxy.client.core.gpu.IGpuTexture;
import me.cortex.voxy.client.core.gpu.PipelineState;
import me.cortex.voxy.client.core.gpu.RenderEncoder;
import me.cortex.voxy.client.core.gpu.RenderPassDesc;
import me.cortex.voxy.client.core.gpu.VertexLayout;
import me.cortex.voxy.client.core.gpu.shader.RuntimeShaderCompiler;
import me.cortex.voxy.client.core.metal.MetalNative;
import me.cortex.voxy.client.core.metal.MetalRenderBackend;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

/**
 * Two-colour-attachment MRT harness, built to make the occlusion cull's last blocker reproducible in
 * seconds instead of six minutes.
 *
 * <p>What it answers, and why it is the right question. The pyramid's source is quads.frag's SECOND
 * colour output, and in the game that attachment reads empty while every structural link around it
 * checks out: the define reaches the shader, the compiled MSL declares {@code [[color(1)]]} and
 * assigns it, the pipeline declares two colour formats, the pass carries two attachments with STORE,
 * the wiring loops are indexed correctly, and a write mask is now set explicitly. So the fault is
 * either below all of that -- in {@code createGraphicsPipeline} or {@code beginRenderPass} -- or in
 * something about the game's configuration that this harness deliberately excludes.
 *
 * <p>It excludes Metallum, the LOD pass, indirect draws, depth attachments and the traversal. If
 * attachment 1 comes back with the shader's constant, MRT works end to end at the backend level and
 * the fault is in the game's configuration. If it comes back with its clear colour, the fault is in
 * the backend and this test now reproduces it in seconds.
 *
 * <p>Run with {@code ./gradlew testMetalMrt}.
 */
public final class MetalMrtSmokeTest {

    private MetalMrtSmokeTest() {}

    public static void main(String[] args) throws Exception {
        String os = System.getProperty("os.name", "").toLowerCase();
        String arch = System.getProperty("os.arch", "").toLowerCase();
        if (!(os.contains("mac") && arch.contains("aarch64"))) {
            System.err.println("MRT harness requires macOS aarch64 (got " + os + "/" + arch + ")");
            System.exit(2);
        }
        if (!MetalNative.load()) {
            System.err.println("Metal native library failed to load");
            System.exit(2);
        }

        Path shadersRoot = Path.of("src/main/resources/assets/voxy/shaders/tools").toAbsolutePath();
        String vertGlsl = Files.readString(shadersRoot.resolve("triangle.vert"), StandardCharsets.UTF_8);
        String fragGlsl = Files.readString(shadersRoot.resolve("mrt.frag"), StandardCharsets.UTF_8);

        RuntimeShaderCompiler.Result vertCompiled = RuntimeShaderCompiler.compile(
                vertGlsl, RuntimeShaderCompiler.Stage.VERTEX, Map.of(),
                RuntimeShaderCompiler.Target.METAL_MSL);
        RuntimeShaderCompiler.Result fragCompiled = RuntimeShaderCompiler.compile(
                fragGlsl, RuntimeShaderCompiler.Stage.FRAGMENT, Map.of(),
                RuntimeShaderCompiler.Target.METAL_MSL);

        // The MSL is the first thing worth asserting: SPIRV-Cross dropping or renaming the second
        // output would make every downstream check moot, and it is the one link this harness can
        // inspect directly.
        String fragMsl = fragCompiled.mslSource();
        boolean mslHasColor1 = fragMsl.contains("[[color(1)]]");
        System.out.println("mrt.frag MSL declares [[color(1)]]: " + (mslHasColor1 ? "YES" : "NO"));
        if (!mslHasColor1) {
            System.out.println(fragMsl);
            throw new RuntimeException("SPIRV-Cross did not emit a second colour output for mrt.frag");
        }

        // GL_RGBA8 = 0x8058, GL_TEXTURE_2D = 0x0DE1, GL_DEPTH_COMPONENT32F = 0x8CAC.
        final int GL_RGBA8 = 0x8058;
        final int GL_DEPTH_COMPONENT32F = 0x8CAC;
        final int GL_TEXTURE_2D = 0x0DE1;
        final int W = 256, H = 256;

        MetalRenderBackend backend = new MetalRenderBackend();
        IGpuPipeline plain = null;
        IGpuPipeline withDepth = null;
        try {
            // Phase 1 -- the minimal shape: two colour attachments, nothing else.
            plain = backend.createGraphicsPipeline(new GraphicsPipelineDesc(
                    vertGlsl, fragGlsl, Map.of(),
                    vertCompiled.mslSource(), fragCompiled.mslSource(),
                    vertCompiled.spirv(), fragCompiled.spirv(),
                    new int[]{GL_RGBA8, GL_RGBA8},
                    VertexLayout.EMPTY,
                    PipelineState.DEFAULT,
                    "voxy:tools/mrt"));
            System.out.println("=== phase 1: two colour attachments, no depth ===");
            boolean plainOk = phase(backend, plain, null, W, H, GL_RGBA8, GL_TEXTURE_2D);

            // Phase 2 -- the one shape this harness was built to add, because it is the largest
            // structural difference between it and the game's LOD pass. The game attaches MC's
            // Depth32Float depth target and depth-tests against it; phase 1 attaches nothing and its
            // pipeline has depth disabled. Everything else about the pass is identical.
            IGpuTexture depth = backend.createTexture(GL_TEXTURE_2D)
                    .store(GL_DEPTH_COMPONENT32F, 1, W, H).name("voxy-mrt-depth");
            withDepth = backend.createGraphicsPipeline(new GraphicsPipelineDesc(
                    vertGlsl, fragGlsl, Map.of(),
                    vertCompiled.mslSource(), fragCompiled.mslSource(),
                    vertCompiled.spirv(), fragCompiled.spirv(),
                    new int[]{GL_RGBA8, GL_RGBA8},
                    VertexLayout.EMPTY,
                    PipelineState.OPAQUE_MESH,
                    "voxy:tools/mrt-depth"));
            System.out.println();
            System.out.println("=== phase 2: two colour attachments PLUS a depth attachment ===");
            boolean depthOk = phase(backend, withDepth, depth, W, H, GL_RGBA8, GL_TEXTURE_2D);

            System.out.println();
            if (depthOk) {
                System.out.println("MRT OK in BOTH shapes — a second colour attachment receives its "
                        + "write with and without a depth attachment present.");
                System.out.println("The game's fault is therefore in something this harness still "
                        + "excludes: Metallum's own attachments in slots 0/depth, the LOD pass's "
                        + "specific depth state, or the traversal's frame ordering.");
            } else {
                System.out.println("REPRODUCED: attachment 1 lands without a depth attachment and is "
                        + "dropped with one. See optimisation.MD 8.6.");
            }
            if (!plainOk || !depthOk) {
                throw new RuntimeException("MRT harness FAILED — see the per-phase lines above");
            }
        } finally {
            if (plain != null) plain.close();
            if (withDepth != null) withDepth.close();
            backend.shutdown();
        }
    }

    /**
     * Renders the two-output triangle into a fresh pair of colour attachments and reports what landed
     * in each. {@code depth} may be null for a pass with no depth attachment at all.
     *
     * <p>Attachment 0 is cleared to a distinct colour and attachment 1 to black, so "the write landed",
     * "the attachment kept its clear colour" and "the value is wrong" are three distinguishable
     * outcomes rather than one zero.
     */
    private static boolean phase(MetalRenderBackend backend, IGpuPipeline pipeline, IGpuTexture depth,
                                 int W, int H, int GL_RGBA8, int GL_TEXTURE_2D) {
        IGpuTexture target0 = backend.createTexture(GL_TEXTURE_2D).store(GL_RGBA8, 1, W, H)
                .name("voxy-mrt-target0");
        IGpuTexture target1 = backend.createTexture(GL_TEXTURE_2D).store(GL_RGBA8, 1, W, H)
                .name("voxy-mrt-target1");

        RenderPassDesc.Builder b = RenderPassDesc.builder(W, H)
                .clearColor(target0, 0.1f, 0.1f, 0.15f, 1.0f)
                .addColorAttachment(target1, 0,
                        RenderPassDesc.LoadAction.CLEAR, RenderPassDesc.StoreAction.STORE,
                        0f, 0f, 0f, 0f);
        if (depth != null) {
            b.depthAttachment(depth, 0,
                    RenderPassDesc.LoadAction.CLEAR, RenderPassDesc.StoreAction.STORE, 1.0f);
        }
        RenderPassDesc pass = b.build();
        System.out.println("  pass colourAttachments=" + pass.colorAttachments().size()
                + " depthAttachment=" + (pass.depthAttachment() == null ? "none" : "present"));

        try (RenderEncoder enc = backend.beginRenderPass(pass)) {
            enc.setPipeline(pipeline);
            enc.draw(RenderEncoder.PRIMITIVE_TRIANGLES, 0, 3, 1, 0);
        }
        backend.submit();

        byte[] px0 = backend.readPixelsRGBA8(target0, 0, 0, W, H);
        byte[] px1 = backend.readPixelsRGBA8(target1, 0, 0, W, H);

        int clear0 = packRgba(0.1f, 0.1f, 0.15f, 1.0f);
        int clear1 = packRgba(0f, 0f, 0f, 0f);
        int want1 = packRgba(0.25f, 0.5f, 0.75f, 1.0f);

        // The triangle's interior; NDC y is up, pixel y is down.
        int interior0 = sample(px0, W, W / 2, H * 5 / 8);
        int interior1 = sample(px1, W, W / 2, H * 5 / 8);
        int corner1 = sample(px1, W, 4, 4);

        System.out.printf("  att0 interior: 0x%08X (clear=0x%08X) -> %s%n", interior0, clear0,
                interior0 != clear0 ? "TRIANGLE DREW" : "STILL CLEAR (no draw at all)");
        System.out.printf("  att1 interior: 0x%08X (want=0x%08X, clear=0x%08X) -> %s%n",
                interior1, want1, clear1,
                interior1 == want1 ? "SECOND OUTPUT LANDED"
                        : interior1 == clear1 ? "NEVER WRITTEN (still its clear colour)"
                                : "WRITTEN BUT WRONG VALUE");
        System.out.printf("  att1 corner:   0x%08X (clear=0x%08X) %s%n", corner1, clear1,
                corner1 == clear1 ? "OK (outside the triangle, untouched)" : "unexpected");

        if (interior0 == clear0) {
            throw new RuntimeException("attachment 0 never drew — the harness itself is broken, so "
                    + "nothing it says about attachment 1 means anything");
        }
        return interior1 == want1;
    }

    private static int packRgba(float r, float g, float b, float a) {
        return (clamp8(r) << 24) | (clamp8(g) << 16) | (clamp8(b) << 8) | clamp8(a);
    }

    private static int clamp8(float v) {
        return Math.max(0, Math.min(255, Math.round(v * 255.0f)));
    }

    private static int sample(byte[] pixels, int width, int x, int y) {
        int o = (y * width + x) * 4;
        return ((pixels[o] & 0xFF) << 24) | ((pixels[o + 1] & 0xFF) << 16)
                | ((pixels[o + 2] & 0xFF) << 8) | (pixels[o + 3] & 0xFF);
    }
}
