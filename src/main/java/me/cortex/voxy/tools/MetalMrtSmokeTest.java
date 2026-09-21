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

            icbCompatProbe(backend, shadersRoot, vertGlsl, vertCompiled);
        } finally {
            backend.shutdown();
        }
    }

    /**
     * Can a pipeline used inside an {@code MTLIndirectCommandBuffer} be built from these shaders?
     *
     * <p>This is the question that decides the largest item on the optimisation docket. An ICB would
     * remove both the per-draw {@code baseInstance} host read AND the synchronous drain it forces
     * (optimisation.MD 10.1) — together 79 % of the frame. Metal accepts {@code baseInstance} inside an
     * ICB, where it drops it on a plain indirect indexed draw.
     *
     * <p>The blocker is that a pipeline used in an ICB is rejected if its fragment shader is
     * ICB-incompatible, and this project's records disagree about what triggers that:
     * {@code MetalRenderBackend:800-810} names "fragment shaders that write gl_FragDepth or use certain
     * outputs" and does NOT name {@code discard}, while optimisation.MD 7.2 names {@code discard}.
     *
     * <p>Three cases, and the third is the one that matters:
     * <ol>
     *   <li>the toy shader with no discard — the control; if this fails the harness is wrong</li>
     *   <li>the toy shader with a never-taken {@code discard} — isolates discard alone</li>
     *   <li>the REAL terrain shaders — and case 2 can pass while this fails, which is exactly why it
     *       must be here rather than inferred</li>
     * </ol>
     * Nothing is drawn: Metal rejects the pipeline at creation, so acceptance is the whole answer, and
     * the rejection text (captured from {@code localizedDescription}) names the offending feature.
     */
    private static void icbCompatProbe(MetalRenderBackend backend, Path shadersRoot, String vertGlsl,
                                       RuntimeShaderCompiler.Result vertCompiled) {
        System.out.println();
        System.out.println("=== ICB compatibility probe (does Metal accept an ICB pipeline?) ===");

        // Case 3's shaders: the real terrain pair, with the defines ShaderCompilerSmokeTest uses for
        // quads.frag's Metal baseline. A copy rather than a reference on purpose -- if that harness's
        // case changes, this probe should be re-checked rather than silently following it.
        // Through ShaderLoader.parse, NOT Files.readString: these shaders carry `#import <voxy:...>`
        // directives and resolve them against the resource path. Reading them raw hands shaderc a bare
        // `#import`, which it rejects as an invalid directive -- an earlier version of this probe did
        // exactly that and its failure said nothing about the ICB. ShaderLoader is what the game uses.
        String terrainVert;
        String terrainFrag;
        try {
            terrainVert = me.cortex.voxy.client.core.gpu.shader.ShaderLoader.parse("voxy:lod/gl46/quads3.vert");
            terrainFrag = me.cortex.voxy.client.core.gpu.shader.ShaderLoader.parse("voxy:lod/gl46/quads.frag");
        } catch (Throwable e) {
            System.out.println("  case 3 SKIPPED: could not load the terrain shaders via ShaderLoader: " + e);
            return;
        }
        Map<String, String> terrainVertDefines = Map.of(
                "NO_SHADE_FACE_TINT", "1.0", "UP_FACE_TINT", "1.0", "DOWN_FACE_TINT", "0.5",
                "Z_AXIS_FACE_TINT", "0.8", "X_AXIS_FACE_TINT", "0.6",
                "VOXY_METAL_BI_FIX", "");
        Map<String, String> terrainFragDefines = Map.of(
                "VOXY_NO_DEPTH_BOUND", "", "VOXY_FORCE_OPAQUE_ALPHA", "");
        // Used whenever the ordinary define map would not compile
        Map<String, String> toyFragDefines = Map.of();

        icbCase(backend, 1, "tools/mrt.frag, no discard (CONTROL — must pass)",
                vertGlsl, vertCompiled, "tools/mrt.frag",
                Files0.frag(shadersRoot, "tools/mrt.frag"), toyFragDefines);
        icbCase(backend, 2, "tools/mrt.frag + a never-taken discard (isolates discard alone)",
                vertGlsl, vertCompiled, "tools/mrt.frag",
                Files0.frag(shadersRoot, "tools/mrt.frag"), Map.of("MRT_DISCARD", ""));
        icbCase(backend, 3, "the REAL terrain shaders (THE ANSWER)",
                terrainVert, null, "lod/gl46/quads3.vert",
                terrainFrag, terrainFragDefines);
        // Case 3 says the terrain fragment shader is ICB-incompatible, and case 2 says it is NOT the
        // discard. quads.frag writes no gl_FragDepth, so two candidates remain -- and these two cases
        // name which, so the fix targets the right one.
        icbCase(backend, 4, "toy + gl_FragCoord read (quads.frag reads it for the Hi-Z output)",
                vertGlsl, vertCompiled, "tools/mrt.frag",
                Files0.frag(shadersRoot, "tools/mrt.frag"), Map.of("MRT_FRAGCOORD", ""));
        icbCase(backend, 5, "toy + gl_HelperInvocation read (quads.frag reads it twice)",
                vertGlsl, vertCompiled, "tools/mrt.frag",
                Files0.frag(shadersRoot, "tools/mrt.frag"), Map.of("MRT_HELPER", ""));
        // Case 2's discard is guarded by a provably-false condition, so the compiler may have removed
        // it -- which would make that PASS meaningless. This one is REACHABLE, like the terrain's
        // alpha cutout. If this fails, discard is the trigger after all and case 2 was a false negative.
        icbCase(backend, 6, "toy + a REACHABLE discard (what the terrain cutout actually is)",
                vertGlsl, vertCompiled, "tools/mrt.frag",
                Files0.frag(shadersRoot, "tools/mrt.frag"), Map.of("MRT_DISCARD_LIVE", ""));
        // Now bisect the REAL shader with its own switches. quads.frag features the toy lacks:
        // atlas sampling (incl. textureGather), the chunk-cull SSBO read, and the imported helpers.
        // VOXY_NO_ATLAS removes the sampling path, so a PASS here names the atlas fetch as the cause.
        icbCase(backend, 7, "real terrain + VOXY_NO_ATLAS (removes the atlas sampling path)",
                terrainVert, null, "lod/gl46/quads3.vert", terrainFrag,
                Map.of("VOXY_NO_DEPTH_BOUND", "", "VOXY_FORCE_OPAQUE_ALPHA", "", "VOXY_NO_ATLAS", ""));
        // The inverse: keep sampling, remove everything else that reads a buffer or a derivative.
        icbCase(backend, 8, "real terrain + FLAT_FRAG (constant colour, no atlas, no tinting)",
                terrainVert, null, "lod/gl46/quads3.vert", terrainFrag,
                Map.of("VOXY_NO_DEPTH_BOUND", "", "VOXY_FORCE_OPAQUE_ALPHA", "",
                        "VOXY_NO_ATLAS", "", "VOXY_LOD_FLAT_FRAG", ""));
        // Case 9: THE DECISIVE ONE, and it retires -- or confirms -- §11.3.
        //
        // Cases 3, 7 and 8 all pass `terrainVert` with vertCompiled == null, so icbCase recompiles the
        // vertex from its OWN hardcoded define map (its fragDefines argument reaches only the fragment).
        // The terrain vertex MSL is therefore byte-identical in all three -- and case 8 PASSES with it.
        // So case 7's "Vertex shader cannot be used with indirect command buffers" cannot mean the
        // vertex is incompatible; the same vertex built a pipeline Metal accepted, in the same run.
        // Case 7's message named the wrong stage, and §11.3's "BOTH stages are implicated" is wrong.
        //
        // This case isolates the vertex completely: the real terrain VERTEX against the TOY fragment.
        //   PASS -> the vertex is ICB-compatible beyond doubt. Stop looking at it; the blocker is
        //           entirely in quads.frag, and cases 3/7/8's failures are all fragment-side.
        //   FAIL -> the vertex genuinely is implicated, and §11.3 is right for a reason case 8 hides.
        // Either way this is one pipeline build, not a bisection of quads3.vert's binding surface.
        // The fragment is tools/quads_iface.frag, NOT tools/mrt.frag: the terrain vertex writes
        // `flat uvec4 interData` at location 0 and mrt.frag declares `vec3 vColor` there, so that
        // pairing fails to LINK ("Fragment input(s) `user(locn0)` mismatching vertex shader output
        // type(s)") -- a failure that says nothing about ICBs and would have been misread as one.
        icbCase(backend, 9, "real terrain VERTEX + interface-matched toy fragment (isolates the vertex)",
                terrainVert, null, "lod/gl46/quads3.vert",
                Files0.frag(shadersRoot, "tools/quads_iface.frag"), toyFragDefines);
        // Cases 8 and 9 disagree: the SAME terrain vertex, and 8 PASSES while 9 FAILS with the vertex
        // message. Cases 10 and 11 separate the two explanations.
        //
        // Case 10 -- §11.4's candidate #1, and the highest-value answer in this probe. VOXY_METAL_BI_FIX
        // is the binding-6 per-draw UBO the Metal baseInstance workaround pushes (MetalRenderEncoder
        // :217-222). An ICB supplies baseInstance natively, so if a pipeline that ALSO reads a pushed
        // per-draw constant is what Metal refuses, then deleting the workaround removes the per-draw
        // CPU read, the drain, AND the ICB blocker in one move.
        icbCase(backend, 10, "real terrain VERTEX WITHOUT VOXY_METAL_BI_FIX (is binding 6 the trigger?)",
                terrainVert, null, "lod/gl46/quads3.vert",
                Files0.frag(shadersRoot, "tools/quads_iface.frag"), toyFragDefines,
                Map.of("NO_SHADE_FACE_TINT", "1.0", "UP_FACE_TINT", "1.0",
                        "DOWN_FACE_TINT", "0.5", "Z_AXIS_FACE_TINT", "0.8",
                        "X_AXIS_FACE_TINT", "0.6"));
        // Case 11 -- the same vertex again, against a fragment that declares NO inputs. If this passes
        // where case 9 fails, the trigger is the fragment's input interface (dropped by quads.frag's
        // FLAT early-out, which is why case 8 passes) and the vertex is exonerated.
        icbCase(backend, 11, "real terrain VERTEX + NO-INPUT fragment (is it the fragment's interface?)",
                terrainVert, null, "lod/gl46/quads3.vert",
                Files0.frag(shadersRoot, "tools/noinput.frag"), toyFragDefines);
        // Case 11 settled that the terrain VERTEX is ICB-compatible and case 10 that binding 6 is not
        // the trigger. What is left is the vertex->fragment interpolant interface: it fails with a
        // fragment that declares quads3.vert's two varyings and passes with one that declares none.
        // These two split that interface in half, to name the member rather than the whole.
        icbCase(backend, 12, "real terrain VERTEX + fragment declaring only `vec2 uv` (location 1)",
                terrainVert, null, "lod/gl46/quads3.vert",
                Files0.frag(shadersRoot, "tools/uv_only.frag"), toyFragDefines);
        icbCase(backend, 13, "real terrain VERTEX + fragment declaring only `flat uvec4 interData`",
                terrainVert, null, "lod/gl46/quads3.vert",
                Files0.frag(shadersRoot, "tools/idata_only.frag"), toyFragDefines);
        // Case 14 -- THE FIX, PROTOTYPED. Case 13 names `flat uvec4 interData` as the blocker; this
        // patches quads3.vert to carry the same bits in a float varying while KEEPING `flat`, and pairs
        // it with tools/idata_float.frag. PASS => the cause is the integer type, and the fix is a
        // two-shader bitcast. FAIL => the cause is the `flat` qualifier, and interData must leave the
        // varying interface altogether.
        final String FLAT_VAR_DECL = "layout(location = 0) out flat uvec4 interData;";
        final String FLOAT_VAR_DECL = "layout(location = 0) out flat vec4 interData;";
        final String VAR_ASSIGN = "interData = quad.attributeData;";
        final String FLOAT_ASSIGN = "interData = uintBitsToFloat(quad.attributeData);";
        String floatIfaceVert = terrainVert
                .replace(FLAT_VAR_DECL, FLOAT_VAR_DECL)
                .replace(VAR_ASSIGN, FLOAT_ASSIGN);
        // A `replace` that silently does not match would hand this case the UNPATCHED vertex, whose
        // fragment interface cannot link -- a FAIL that says nothing about the fix. Verify the patch
        // landed rather than trusting the result.
        if (floatIfaceVert.contains(FLAT_VAR_DECL) || !floatIfaceVert.contains(FLOAT_VAR_DECL)
                || !floatIfaceVert.contains(FLOAT_ASSIGN)) {
            System.out.println("  case 14 SKIPPED: the quads3.vert patch did not apply — the probe "
                    + "would have tested the unpatched shader and its FAIL would have meant nothing");
        } else {
            icbCase(backend, 14, "PATCHED vertex: interData as `flat vec4` + floatBitsToUint (THE FIX)",
                    floatIfaceVert, null, "lod/gl46/quads3.vert (patched)",
                    Files0.frag(shadersRoot, "tools/idata_float.frag"), toyFragDefines);
        }
    }

    /**
     * The vertex defines every case below compiles with: the terrain face-tint baseline plus
     * {@code VOXY_METAL_BI_FIX}, i.e. the binding-6 per-draw UBO that the Metal baseInstance
     * workaround pushes. Named rather than inlined so a bisection can vary it — which is the whole
     * point of the {@code vertDefines} overload underneath.
     */
    private static final Map<String, String> TERRAIN_VERT_DEFINES = Map.of(
            "NO_SHADE_FACE_TINT", "1.0", "UP_FACE_TINT", "1.0", "DOWN_FACE_TINT", "0.5",
            "Z_AXIS_FACE_TINT", "0.8", "X_AXIS_FACE_TINT", "0.6", "VOXY_METAL_BI_FIX", "");

    /** One ICB case, compiled with {@link #TERRAIN_VERT_DEFINES}. Never throws. */
    private static void icbCase(MetalRenderBackend backend, int n, String label,
                                String vertGlsl, RuntimeShaderCompiler.Result vertCompiled,
                                String fragPath, String fragGlsl, Map<String, String> fragDefines) {
        icbCase(backend, n, label, vertGlsl, vertCompiled, fragPath, fragGlsl, fragDefines,
                TERRAIN_VERT_DEFINES);
    }

    /** One ICB case with an explicit vertex define map, for bisecting the vertex stage. */
    private static void icbCase(MetalRenderBackend backend, int n, String label,
                                String vertGlsl, RuntimeShaderCompiler.Result vertCompiled,
                                String fragPath, String fragGlsl, Map<String, String> fragDefines,
                                Map<String, String> vertDefines) {
        IGpuPipeline pipeline = null;
        try {
            RuntimeShaderCompiler.Result v = vertCompiled != null ? vertCompiled
                    : RuntimeShaderCompiler.compile(vertGlsl, RuntimeShaderCompiler.Stage.VERTEX,
                            vertDefines, RuntimeShaderCompiler.Target.METAL_MSL);
            RuntimeShaderCompiler.Result f = RuntimeShaderCompiler.compile(fragGlsl,
                    RuntimeShaderCompiler.Stage.FRAGMENT, fragDefines,
                    RuntimeShaderCompiler.Target.METAL_MSL);

            GraphicsPipelineDesc desc = new GraphicsPipelineDesc(
                    vertGlsl, fragGlsl, Map.of(),
                    v.mslSource(), f.mslSource(), v.spirv(), f.spirv(),
                    new int[]{0x8058}, VertexLayout.EMPTY, PipelineState.OPAQUE_MESH,
                    "icb-probe-" + n).withIndirectCommandBufferUsage(true);

            pipeline = backend.createGraphicsPipeline(desc);
            System.out.println("  case " + n + " PASS  " + label);
            System.out.println("        -> an ICB pipeline IS available for this shader");
        } catch (Throwable t) {
            System.out.println("  case " + n + " FAIL  " + label);
            System.out.println("        -> Metal rejected it: " + t.getMessage());
        } finally {
            if (pipeline != null) pipeline.close();
        }
    }

    /** Tiny helper so the case table above reads as a table. */
    private static final class Files0 {
        static String frag(Path root, String rel) {
            try {
                return Files.readString(root.resolve(rel), StandardCharsets.UTF_8);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
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
