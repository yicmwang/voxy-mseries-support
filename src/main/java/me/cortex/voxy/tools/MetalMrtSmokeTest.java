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
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Two-colour-attachment MRT harness, built to make the occlusion cull's last blocker reproducible in
 * seconds instead of six minutes.
 *
 * <p>The pyramid's source is quads.frag's SECOND colour output, and in the game that attachment reads
 * empty while every measurable link around it is correct: the pass descriptor carries two colour
 * attachments with slot 1 bound to the right texture at RGBA8Unorm and a Store action
 * ({@code [Metal-PASSSLOT]}), the pipeline descriptor carries two RGBA8Unorm attachments with write
 * mask All ({@code [Metal-PIPESLOT]}), the compiled MSL declares and assigns {@code [[color(1)]]}, the
 * define reaches the shader, and this harness proves the backend delivers a second colour output.
 *
 * <p>What is left is the one input never varied: the SHADER'S OWN SHAPE. The game's quads.frag is a
 * large shader with {@code discard} paths and a {@code gl_FragCoord} input; this harness's fragment
 * shader had neither. So this runs the same pass four times over variants of that shader, which is the
 * cheapest way to find out whether the shader is the difference.
 *
 * <p>Run with {@code ./gradlew testMetalMrt}.
 */
public final class MetalMrtSmokeTest {

    private MetalMrtSmokeTest() {}

    /** The one varying the harness vertex shader supplies. */
    private static final String VERT = "tools/triangle.vert";

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

        Path shadersRoot = Path.of("src/main/resources/assets/voxy/shaders").toAbsolutePath();
        String vertGlsl = Files.readString(shadersRoot.resolve(VERT), StandardCharsets.UTF_8);
        String fragGlsl = Files.readString(shadersRoot.resolve("tools/mrt.frag"), StandardCharsets.UTF_8);

        RuntimeShaderCompiler.Result vertCompiled = RuntimeShaderCompiler.compile(
                vertGlsl, RuntimeShaderCompiler.Stage.VERTEX, Map.of(),
                RuntimeShaderCompiler.Target.METAL_MSL);

        // GL_RGBA8 = 0x8058, GL_DEPTH_COMPONENT32F = 0x8CAC, GL_TEXTURE_2D = 0x0DE1.
        final int GL_RGBA8 = 0x8058;
        final int GL_DEPTH_COMPONENT32F = 0x8CAC;
        final int GL_TEXTURE_2D = 0x0DE1;
        final int W = 256, H = 256;

        // The variants, in the order that isolates one feature at a time. The baseline has neither
        // feature; each step adds one.
        String[][] variants = {
                {"baseline (constant, no discard, no gl_FragCoord)", ""},
                {"+ gl_FragCoord.z (adds a [[position]] fragment input)", "MRT_FRAGCOORD"},
                {"+ a never-taken discard (mirrors quads.frag's cutout)", "MRT_DISCARD"},
                {"+ both", "MRT_FRAGCOORD,MRT_DISCARD"},
        };

        MetalRenderBackend backend = new MetalRenderBackend();
        try {
            int failures = 0;
            for (String[] variant : variants) {
                String label = variant[0];
                Map<String, String> defines = new LinkedHashMap<>();
                for (String d : variant[1].split(",")) {
                    if (!d.isEmpty()) defines.put(d, "");
                }

                RuntimeShaderCompiler.Result fragCompiled = RuntimeShaderCompiler.compile(
                        fragGlsl, RuntimeShaderCompiler.Stage.FRAGMENT, defines,
                        RuntimeShaderCompiler.Target.METAL_MSL);
                boolean mslHasColor1 = fragCompiled.mslSource().contains("[[color(1)]]");

                String pipeLabel = "voxy:tools/mrt[" + variant[1] + "]";
                IGpuPipeline pipeline = backend.createGraphicsPipeline(new GraphicsPipelineDesc(
                        vertGlsl, fragGlsl, defines,
                        vertCompiled.mslSource(), fragCompiled.mslSource(),
                        vertCompiled.spirv(), fragCompiled.spirv(),
                        new int[]{GL_RGBA8, GL_RGBA8},
                        VertexLayout.EMPTY,
                        PipelineState.OPAQUE_MESH,
                        pipeLabel));
                try {
                    System.out.println("=== " + label);
                    if (!mslHasColor1) {
                        System.out.println("  MSL declares [[color(1)]]: NO — the output is gone before "
                                + "Metal ever sees it");
                        failures++;
                        continue;
                    }
                    System.out.println("  MSL declares [[color(1)]]: YES");
                    if (!phase(backend, pipeline, W, H, GL_RGBA8, GL_DEPTH_COMPONENT32F, GL_TEXTURE_2D,
                            defines.containsKey("MRT_FRAGCOORD"))) {
                        failures++;
                    }
                } finally {
                    pipeline.close();
                }
            }
            System.out.println();
            if (failures == 0) {
                System.out.println("MRT OK in all " + variants.length + " shader variants — the shader's "
                        + "shape is NOT the difference. What remains excluded is the frame orchestration "
                        + "around the pass (Metallum's attachments in slots 0 and depth, the LOD pass's "
                        + "specific depth state).");
            } else {
                System.out.println(failures + " of " + variants.length + " variants FAILED — the shader "
                        + "shape IS the difference; bisect from the first failing variant.");
                throw new RuntimeException("MRT harness found a failing variant");
            }
        } finally {
            backend.shutdown();
        }
    }

    /**
     * Renders the two-output triangle into a fresh pair of colour attachments, with a depth attachment
     * present (the game's pass has one), and reports what landed in each.
     *
     * <p>Attachment 0 is cleared to a distinct colour and attachment 1 to black, so "the write landed",
     * "the attachment kept its clear colour" and "the value is wrong" are three distinguishable
     * outcomes rather than one zero.
     *
     * @param fragCoordVariant true when this variant writes {@code gl_FragCoord.z}, whose expected value
     *                         depends on the triangle's depth rather than being a known constant
     */
    private static boolean phase(MetalRenderBackend backend, IGpuPipeline pipeline, int W, int H,
                                 int GL_RGBA8, int GL_DEPTH_COMPONENT32F, int GL_TEXTURE_2D,
                                 boolean fragCoordVariant) {
        IGpuTexture target0 = backend.createTexture(GL_TEXTURE_2D).store(GL_RGBA8, 1, W, H)
                .name("voxy-mrt-target0");
        IGpuTexture target1 = backend.createTexture(GL_TEXTURE_2D).store(GL_RGBA8, 1, W, H)
                .name("voxy-mrt-target1");
        IGpuTexture depth = backend.createTexture(GL_TEXTURE_2D)
                .store(GL_DEPTH_COMPONENT32F, 1, W, H).name("voxy-mrt-depth");

        RenderPassDesc pass = RenderPassDesc.builder(W, H)
                .clearColor(target0, 0.1f, 0.1f, 0.15f, 1.0f)
                .addColorAttachment(target1, 0,
                        RenderPassDesc.LoadAction.CLEAR, RenderPassDesc.StoreAction.STORE,
                        0f, 0f, 0f, 0f)
                .depthAttachment(depth, 0,
                        RenderPassDesc.LoadAction.CLEAR, RenderPassDesc.StoreAction.STORE, 1.0f)
                .build();

        try (RenderEncoder enc = backend.beginRenderPass(pass)) {
            enc.setPipeline(pipeline);
            enc.draw(RenderEncoder.PRIMITIVE_TRIANGLES, 0, 3, 1, 0);
        }
        backend.submit();

        byte[] px0 = backend.readPixelsRGBA8(target0, 0, 0, W, H);
        byte[] px1 = backend.readPixelsRGBA8(target1, 0, 0, W, H);

        int clear0 = packRgba(0.1f, 0.1f, 0.15f, 1.0f);
        int clear1 = packRgba(0f, 0f, 0f, 0f);
        int want1 = fragCoordVariant ? packRgba(0f, 0f, 0f, 1f) : packRgba(0.25f, 0.5f, 0.75f, 1.0f);

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
