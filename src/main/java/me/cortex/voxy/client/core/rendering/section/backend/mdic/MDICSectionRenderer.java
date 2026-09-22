package me.cortex.voxy.client.core.rendering.section.backend.mdic;


import me.cortex.voxy.client.RenderStatistics;
import me.cortex.voxy.client.VoxyClient;
import me.cortex.voxy.client.core.AbstractRenderPipeline;
import me.cortex.voxy.client.core.gpu.Capabilities;
import me.cortex.voxy.client.core.gpu.ComputeEncoder;
import me.cortex.voxy.client.core.gpu.IGpuBuffer;
import me.cortex.voxy.client.core.gpu.RenderBackendFactory;
import me.cortex.voxy.client.core.gpu.shader.ShaderLoader;
import me.cortex.voxy.client.core.model.ModelStore;
import me.cortex.voxy.client.core.rendering.section.backend.AbstractSectionRenderer;
import me.cortex.voxy.client.core.rendering.section.geometry.BasicSectionGeometryData;
import me.cortex.voxy.client.core.rendering.util.DownloadStream;
import me.cortex.voxy.client.core.rendering.util.LightMapHelper;
import me.cortex.voxy.client.core.rendering.util.MetalMvpUtil;
import me.cortex.voxy.client.core.rendering.util.SharedIndexBuffer;
import me.cortex.voxy.client.core.rendering.util.UploadStream;
import me.cortex.voxy.common.Logger;
import me.cortex.voxy.common.world.WorldEngine;
import net.minecraft.client.Minecraft;
import net.minecraft.core.Direction;
import org.joml.Matrix4f;
import org.lwjgl.system.MemoryUtil;

import java.util.List;

// GL_RGBA8 survives as a plain format tag in GraphicsPipelineDesc (the backend maps it); it is the
// only OpenGL symbol left in this file now that the draw path is Metal-only.
import static org.lwjgl.opengl.GL11C.GL_RGBA8;

//Uses MDIC to render the sections
public class MDICSectionRenderer extends AbstractSectionRenderer<MDICViewport, BasicSectionGeometryData> {
    public static final Factory<MDICViewport, BasicSectionGeometryData> FACTORY = AbstractSectionRenderer.Factory.create(MDICSectionRenderer.class);

    private static final int TRANSLUCENT_OFFSET = 400_000;//in draw calls
    private static final int TEMPORAL_OFFSET = 500_000;//in draw calls
    private static final int STATISTICS_BUFFER_BINDING = 8;
    /** cmdgen binding for the built-section mask. 0-7 are the draw/metadata table, 8 is statistics. */
    private static final int BUILT_MASK_BINDING = 9;
    /**
     * Fragment-stage binding for the same mask, for the per-chunk-column cull in {@code quads.frag}.
     *
     * <p>9 is taken on the graphics path by the Metal chunk-bound depth buffer, so the mask rides at
     * 10. The two stages need it at different granularity: cmdgen decides for a whole LOD node,
     * which is 2x2 chunk columns at detail 0 and larger above, and the fragment decides for the one
     * column it is actually in.
     */
    private static final int BUILT_MASK_CHUNK_BINDING = 10;
    /**
     * Terrain pipelines, {@link me.cortex.voxy.client.core.gpu.IGpuPipeline} via
     * createGraphicsPipeline — the only path there is. (The Iris-patched
     * Shader.Builder pair that used to sit beside them is gone with the GL path:
     * patchOpaqueShader/patchTranslucentShader are abstract-pipeline hooks nothing
     * overrides any more, so the branch that built them could never be taken.)
     */
    private final me.cortex.voxy.client.core.gpu.IGpuPipeline terrainPipeline;
    private final me.cortex.voxy.client.core.gpu.IGpuPipeline translucentTerrainPipeline;

    private final me.cortex.voxy.client.core.gpu.RenderBackend backend = RenderBackendFactory.get();

    /** Sodium's built-section set, packed for cmdgen's cull. See BuiltSectionMask. */
    private final me.cortex.voxy.client.core.rendering.BuiltSectionMask builtSectionMask =
            new me.cortex.voxy.client.core.rendering.BuiltSectionMask();

    private final me.cortex.voxy.client.core.gpu.IGpuPipeline commandGenPipeline = this.backend.createComputePipeline(
            new me.cortex.voxy.client.core.gpu.ComputePipelineDesc(
                    ShaderLoader.parse("voxy:lod/gl46/cmdgen.comp"),
                    cmdgenDefines(),
                    null, null,
                    128, 1, 1, // matches cmdgen.comp's local_size_x=128 (and prep.comp's /128 dispatch math)
                    "MDICSectionRenderer.cmdgen"));
    // M12 chunk 3: commandGen prepass is dispatched via ComputeEncoder; no
    // cached glProgram id needed.

    private final me.cortex.voxy.client.core.gpu.IGpuPipeline prepPipeline = this.backend.createComputePipeline(
            new me.cortex.voxy.client.core.gpu.ComputePipelineDesc(
                    ShaderLoader.parse("voxy:lod/gl46/prep.comp"),
                    java.util.Map.of(),
                    null, null,
                    1, 1, 1,
                    "MDICSectionRenderer.prep"));
    // M12 chunk 2: prep prepass is dispatched via ComputeEncoder; no cached
    // glProgram id needed (encoder pulls it from GlComputePipeline on GL,
    // MTLComputePipelineState on Metal).

    // The cull graphics pipeline (lod/gl46/cull/raster.vert|frag) was built here and dispatched
    // ONLY by the GL occlusion-cull arm in buildDrawCalls; that arm is deleted, so the pipeline
    // and its two shader sources are gone. The live cull is sectionCullPipeline below (a compute
    // pass against the Hi-Z pyramid) plus quads.frag's per-section built-mask test.

    /**
     * The per-section occlusion cull (lod/gl46/section_cull.comp), and the replacement for the
     * M12-chunk-5 force-all-visible stub. It walks exactly the list the stub walked — the
     * traversal's render list, `indirectLookup` — but decides each section's
     * {@code visibilityData[sid]} with the same Hi-Z predicate the traversal uses, instead of
     * marking everything visible. Dispatched only when {@link #SECTION_CULL} is on and the
     * Hi-Z pyramid has a texture (see the dispatch block in buildDrawCalls). Allocated
     * unconditionally — negligible memory cost, and gating the allocation behind a backend check
     * would only make the class harder to read.
     */
    private final me.cortex.voxy.client.core.gpu.IGpuPipeline sectionCullPipeline = this.backend.createComputePipeline(
            new me.cortex.voxy.client.core.gpu.ComputePipelineDesc(
                    ShaderLoader.parse("voxy:lod/gl46/section_cull.comp"),
                    sectionCullDefines(),
                    null, null,
                    128, 1, 1,
                    "MDICSectionRenderer.sectionCull"));

    private final me.cortex.voxy.client.core.gpu.IGpuPipeline prefixSumPipeline = this.backend.createComputePipeline(
            new me.cortex.voxy.client.core.gpu.ComputePipelineDesc(
                    ShaderLoader.parse(Capabilities.INSTANCE.subgroup ? "voxy:util/prefixsum/inital3.comp" : "voxy:util/prefixsum/simple.comp"),
                    java.util.Map.of("IO_BUFFER", "0"),
                    null, null,
                    256, 1, 1, // matches WORK_SIZE 256 declared in both prefixsum variants
                    "MDICSectionRenderer.prefixSum"));
    // M12 chunk 1: prefixSum prepass is dispatched via ComputeEncoder, so it
    // does not need a cached glProgram id (the encoder pulls it from the
    // GlComputePipeline directly on GL; Metal uses the MTLComputePipelineState).

    private final me.cortex.voxy.client.core.gpu.IGpuPipeline translucentGenPipeline = this.backend.createComputePipeline(
            new me.cortex.voxy.client.core.gpu.ComputePipelineDesc(
                    ShaderLoader.parse("voxy:lod/gl46/buildtranslucents.comp"),
                    java.util.Map.of(
                            "TRANSLUCENT_WRITE_BASE", "1024",
                            "TRANSLUCENT_DISTANCE_BUFFER_BINDING", "5",
                            "TRANSLUCENT_OFFSET", Integer.toString(TRANSLUCENT_OFFSET),
                            // Exclusive upper bound of the translucent region
                            // [TRANSLUCENT_OFFSET, TEMPORAL_OFFSET): buildtranslucents.comp
                            // drops any command whose prefix-sum cursor would land in the
                            // temporal slice or off the buffer end. INT_MAX under the
                            // VOXY_CMDGEN_NOCLAMP A/B switch.
                            "MAX_TRANSLUCENT_DRAW_END",
                            Integer.toString(cmdgenNoClamp() ? Integer.MAX_VALUE : TEMPORAL_OFFSET)),
                    null, null,
                    128, 1, 1, // matches buildtranslucents.comp's local_size_x=128
                    "MDICSectionRenderer.translucentGen"));
    // M12 chunk 4: translucentGen prepass is dispatched via ComputeEncoder;
    // no cached glProgram id needed.

    /** {@code VOXY_CMDGEN_NOCLAMP=1} removes the cmdgen draw-command bound checks
     *  (caps injected as INT_MAX) so the pre-fix overflow can be A/B-reproduced
     *  on-device. Default: clamps ON. */
    private static boolean cmdgenNoClamp() {
        return "1".equals(System.getenv("VOXY_CMDGEN_NOCLAMP"));
    }

    /** True when the built-section cull runs at all; VOXY_LOD_BUILT_MASK=0 turns it off. */
    private static final boolean CULL_ENABLED = !"0".equals(System.getenv("VOXY_LOD_BUILT_MASK"));

    /**
     * VOXY_LOD_CHUNK_CULL=0 disables the fragment-stage per-section cull, leaving the mask built and
     * uploaded but unread. Exists for a three-way A/B: full cull, mask built but unread, nothing.
     */
    private static final boolean CHUNK_CULL = CULL_ENABLED
            && !"0".equals(System.getenv("VOXY_LOD_CHUNK_CULL"));

    /**
     * VOXY_LOD_SECCULL=0 skips the per-section occlusion cull pass, so nothing writes
     * {@code visibilityData} and cmdgen emits no sections at all (the LOD disappears). It exists to
     * prove the pass is what changed a measurement, not as a shipping configuration.
     *
     * <p>A DIFFERENT question from {@link #CHUNK_CULL}, deliberately kept separate: this one asks
     * whether a section's box is OCCLUDED (Hi-Z — it lies entirely behind the far side of what the
     * depth buffer already shows) and decides whether a draw command is emitted at all; CHUNK_CULL
     * asks whether vanilla has built the chunks the section overlaps, in the fragment stage, per
     * 16x16x16 section. Both are on; neither licenses removing the other.
     */
    private static final boolean SECTION_CULL = !"0".equals(System.getenv("VOXY_LOD_SECCULL"));

    /**
     * VOXY_LOD_SECCULL_BOX selects the box the cull pass tests: {@code node} (default) is the
     * section's whole cell, {@code aabb} is the occupied sub-box the mesher recorded for it.
     *
     * <p>{@code node} is Stage 1 of the cull work and is a PROVABLE NO-OP: the pass's input list IS
     * the traversal's render queue, filled only after each section's node passed that same Hi-Z test
     * with that same box, so Stage 1 removes zero draws by construction — any movement in
     * {@code rawOpaque} means the pass's coordinate frame is wrong. Only {@code aabb} can remove
     * draws. Its frame is ported from upstream's raster.vert rather than measured in this tree, so
     * confirm it by image diff, never by the derivation.
     */
    private static final boolean SECTION_CULL_AABB_BOX =
            "aabb".equals(System.getenv("VOXY_LOD_SECCULL_BOX"));

    /**
     * VOXY_LOD_SECCULL_BOX=force restores the deleted {@code force_all_visible.comp} stub's behaviour:
     * every section in the traversal's queue is marked visible-this-frame, so the pass culls nothing.
     *
     * <p><b>This is the A/B baseline, and it cannot be reached by disabling the pass.</b> Nothing else
     * writes {@code visibilityData}, so skipping the dispatch leaves it stale and cmdgen emits no
     * sections at all — the LOD disappears. A skipped pass is a no-LOD arm, not a no-cull one, and
     * comparing against it would contrast a working frame with an empty one. The box is still computed
     * in this mode, so the baseline and the cull arms differ in the write only.
     */
    private static final boolean SECTION_CULL_FORCE_VISIBLE =
            "force".equals(System.getenv("VOXY_LOD_SECCULL_BOX"));

    /**
     * Bindings and box variant for {@code lod/gl46/section_cull.comp}, mirroring the defaults the
     * shader declares for itself so neither side is the only place a binding is written down.
     *
     * <p>HIZ_BINDING is 0 and that is not a collision with the scene uniform at BUFFER 0: textures and
     * samplers are a different namespace from buffers (the traversal binds the pyramid to the same
     * slot while its SceneUniform sits at its own buffer binding). It names the same
     * NEAREST/NEAREST/CLAMP_TO_EDGE sampler the traversal uses, via HiZBuffer.getSampler().
     */
    private static java.util.Map<String, String> sectionCullDefines() {
        var m = new java.util.LinkedHashMap<String, String>();
        m.put("VISIBILITY_BUFFER_BINDING", "2");
        m.put("VISIBILITY_ACCESS", "writeonly");
        m.put("INDIRECT_SECTION_LOOKUP_BINDING", "3");
        m.put("SECTION_METADATA_BUFFER_BINDING", "1");
        m.put("HIZ_BINDING", "0");
        if (SECTION_CULL_AABB_BOX) m.put("SECCULL_BOX_AABB", "");
        if (SECTION_CULL_FORCE_VISIBLE) m.put("SECCULL_FORCE_VISIBLE", "");
        return m;
    }

    private static java.util.Map<String, String> cmdgenDefines() {
        var m = new java.util.LinkedHashMap<String, String>();
        // NOTE: no built-section mask define here any more. The cull is per 16x16x16 SECTION and is
        // taken in quads.frag; cmdgen.comp no longer declares the buffer, because a node-granular
        // pass cannot express a three-dimensional decision. See CULL_ENABLED / CHUNK_CULL.
        m.put("TRANSLUCENT_WRITE_BASE", "1024");
        m.put("TEMPORAL_OFFSET", Integer.toString(TEMPORAL_OFFSET));
        m.put("TRANSLUCENT_DISTANCE_BUFFER_BINDING", "7");
        // Capacities of the opaque and temporal draw-command regions, mirroring
        // MDICViewport.drawCallBuffer = 5*4*(400_000 + 100_000 + 100_000):
        //   opaque    [0, TRANSLUCENT_OFFSET)               -> 400_000 cmds
        //   temporal  [TEMPORAL_OFFSET, TEMPORAL_OFFSET+100k) -> 100_000 cmds (the tail)
        // cmdgen.comp clamps its unbounded atomic write cursors to these so a
        // large visible-section render list can never scatter draw commands past
        // a region into the next slice or off the end of the buffer. Past-the-end
        // SSBO writes are undefined on Metal (device fault / cmd-buffer abort) and
        // corrupt the adjacent slices on GL; the CPU read side already saturates
        // its draw counts at the same caps (maxDrawCount / metalDrawCount).
        boolean noClamp = cmdgenNoClamp();
        m.put("MAX_OPAQUE_DRAWS",   Integer.toString(noClamp ? Integer.MAX_VALUE : TRANSLUCENT_OFFSET));
        m.put("MAX_TEMPORAL_DRAWS", Integer.toString(noClamp ? Integer.MAX_VALUE : (TEMPORAL_OFFSET - TRANSLUCENT_OFFSET)));
        if (RenderStatistics.enabled) {
            m.put("HAS_STATISTICS", "");
            m.put("STATISTICS_BUFFER_BINDING", Integer.toString(STATISTICS_BUFFER_BINDING));
        }
        return m;
    }

    /**
     * The scene uniform, RING-BUFFERED one slot per frame in flight.
     *
     * <p>It used to be a single 1024-byte buffer rewritten every frame, and that is a race on Metal:
     * {@code MAX_SUBMITS_IN_FLIGHT} is 3, so frame N+1's upload can land before frame N's draws have
     * read theirs, and frame N then draws with frame N+1's {@code baseSectionPos} and MVP. GL would
     * rename the buffer behind the caller; Metal does not, which makes this port-specific.
     *
     * <p>It fits every symptom: pinning the camera makes {@code viewport.section} and the MVP identical
     * every frame, so a stale read is invisible -- which is exactly why the artifact needs movement;
     * and {@code baseSectionPos} enters the vertex path as
     * {@code (extractLoDPosition(sPos)*(1<<lodLevel)) - baseSectionPos}, so a one-frame-stale value
     * displaces a detail-0 section by 32 blocks and a detail-4 one by 512, i.e. scattered debris at
     * different scales rather than a uniform offset.
     *
     * <p>A ring of UNIFORM_RING slots, indexed by the frame id, keeps a frame's uniform alive until
     * well after its draws have retired.
     */
    // Sized for the CPU AHEAD of the GPU, not for MAX_SUBMITS_IN_FLIGHT. The ring's job is to keep a
    // frame's uniform alive until that frame's draws have retired, and the CPU does not block on the
    // GPU: it can be many frames ahead of what has executed, so 4 slots (just past the 3 submits in
    // flight) is not enough. 32 costs 32 KB and covers any lead this renderer produces.
    //
    // VOXY_UNIFORM_RING makes the slot count switchable so the A/B is ONE BUILD with one variable, and
    // DEFAULTS TO 1 -- i.e. upstream's single rewritten buffer. Two reasons it must default to 1:
    // `voxy/src/.../mdic/MDICSectionRenderer.java:89` is exactly `new GlBuffer(1024)` rewritten every
    // frame, so 1 is the faithful port; and my two previous "ring" runs were compared against a
    // different build running a different second flag, which is not an A/B. Defaulting to the port
    // means an unset run reproduces the baseline, and only an explicit value tests the ring.
    private static final int UNIFORM_RING = parseUniformRing();

    private static int parseUniformRing() {
        String v = System.getenv("VOXY_UNIFORM_RING");
        if (v == null || v.isEmpty()) return 1;
        try {
            int n = Integer.parseInt(v.trim());
            // A power of two keeps the modulus a mask; anything <=0 would be a divide by zero.
            return Integer.bitCount(n) == 1 && n >= 1 ? n : 1;
        } catch (NumberFormatException e) {
            return 1;
        }
    }

    private final IGpuBuffer[] uniformRing = new IGpuBuffer[UNIFORM_RING];
    {
        for (int i = 0; i < UNIFORM_RING; i++) {
            this.uniformRing[i] = RenderBackendFactory.get().createBuffer(1024).zero();
        }
    }

    /** This frame's uniform slot. */
    private IGpuBuffer uniformFor(MDICViewport viewport) {
        return this.uniformRing[(viewport.frameId & 0x7fffffff) % UNIFORM_RING];
    }

    // Far-water alpha ramp (2026-07-03, Metal translucent shader only —
    // see the VOXY_WATER_FAR_ALPHA injection + quads.frag). Target alpha at
    // the far end of the ramp; 0 disables. Ramp distances default to a
    // render-distance-relative window (uploadUniformBuffer) unless the
    // START/END envs pin them in blocks.
    private static final float WATER_FAR_ALPHA = parseEnvFloat("VOXY_WATER_FAR_ALPHA", 0.95f);
    private static final float WATER_FAR_ALPHA_START = parseEnvFloat("VOXY_WATER_FAR_ALPHA_START", 0.0f);
    private static final float WATER_FAR_ALPHA_END = parseEnvFloat("VOXY_WATER_FAR_ALPHA_END", 0.0f);

    // The translucent near-cull's tuning used to live here: TRANS_NEAR_CULL_XZ / _RADIAL / _MARGIN,
    // which switched the near-water cull between the Chebyshev square, the 3D slant distance and
    // Sodium's Euclidean cylinder (2026-07-03 rounds 3 and 5). All three are gone with the cull they
    // configured -- the near-cull was deleted when it was folded into the per-section built mask, the
    // VOXY_TRANS_NEAR_CULL* defines were never injected after that, and nothing read these constants.
    // The metric question they were answering is now settled by sodiumDrawsSection, once, in
    // BuiltSectionMask.

    private static float parseEnvFloat(String name, float def) {
        String v = System.getenv(name);
        if (v == null || v.isBlank()) return def;
        try {
            return Float.parseFloat(v.trim());
        } catch (NumberFormatException e) {
            return def;
        }
    }

    private static int parseEnvInt(String name, int def) {
        String v = System.getenv(name);
        if (v == null || v.isBlank()) return def;
        try {
            return Integer.parseInt(v.trim());
        } catch (NumberFormatException e) {
            return def;
        }
    }

    //TODO: needs to be in the viewport, since it contains the compute indirect call/values
    private final IGpuBuffer distanceCountBuffer = RenderBackendFactory.get().createBuffer(1024*4+100_000*4).zero();//TODO move to viewport?

    //Statistics
    private final IGpuBuffer statisticsBuffer = RenderBackendFactory.get().createBuffer(1024).zero();

    private final AbstractRenderPipeline pipeline;
    public MDICSectionRenderer(AbstractRenderPipeline pipeline, ModelStore modelStore, BasicSectionGeometryData geometryData) {
        super(modelStore, geometryData);
        this.pipeline = pipeline;
        //The pipeline can be used to transform the renderer in abstract ways

        // The bound-depth sampler used to be created here, for the depth mask at binding 2. Removed
        // with the binding: the mask is never rasterized on Metal, quads.frag's depth-bound sample is
        // compiled out by default anyway, and a sampler for a texture nothing samples is a GPU object
        // created to sit idle for the session.
        String vertex = ShaderLoader.parse("voxy:lod/gl46/quads3.vert");
        String taa = pipeline.taaFunction("taaShift");
        if (taa != null) {
            vertex += "\n"+taa;//inject it at the end
        }
        String frag = ShaderLoader.parse("voxy:lod/gl46/quads.frag");

        {
            // Defines mirror the face-tint floats from addDirectionalFaceTint plus
            // TAA_PATCH if a TAA function exists (see buildTerrainDefines).
            java.util.Map<String, String> commonDefines = buildTerrainDefines(taa);
            java.util.Map<String, String> opaqueDefines = new java.util.LinkedHashMap<>(commonDefines);
            java.util.Map<String, String> translucentDefines = new java.util.LinkedHashMap<>(commonDefines);
            translucentDefines.put("TRANSLUCENT", "");

            // Phase C material g-buffer mode (Metal + Iris vx-contract). Compile
            // quads.frag's PATCHED_SHADER path with the MRT emitter appended so it
            // writes the 3 material planes (albedo/tint/misc) instead of a final
            // colour; the pack's real voxy_opaque/voxy_translucent runs later GL-side
            // in MetalVxResolvePass. quads.frag's water/flat early-outs are guarded by
            // !defined(PATCHED_SHADER), so they are bypassed automatically here.
            boolean vxMaterial = pipeline.vxMaterialMode();
            // Opaque uses the material emitter (3 planes for the GL resolve) ONLY when
            // opaque-material is explicitly opted in. Default trans-only: opaque keeps the
            // proven base shader (single lit colour → bridge → normal composite, untouched
            // on dev); only the TRANSLUCENT (water) layer goes through the material g-buffer
            // + voxy_translucent resolve (issue #11). This is the mergeable shape.
            boolean vxOpaqueMat = pipeline.vxOpaqueMaterialMode();
            // P2: the Phase C material g-buffer emitter was Iris-only and is gone with Iris.
            String vxOpaqueFrag = frag;
            String vxTransFrag = frag;
            boolean gbufferDebug = "1".equals(System.getenv("VOXY_VX_GBUFFER_DEBUG"));
            if (vxOpaqueMat) {
                opaqueDefines.put("PATCHED_SHADER", "");
                opaqueDefines.put("VOXY_VX_GBUFFER", "");
                if (gbufferDebug) opaqueDefines.put("VOXY_VX_GBUFFER_DEBUG", "");
            }
            if (vxMaterial) {
                translucentDefines.put("PATCHED_SHADER", "");
                translucentDefines.put("VOXY_VX_GBUFFER", "");
                if (gbufferDebug) translucentDefines.put("VOXY_VX_GBUFFER_DEBUG", "");
            }
            {
                // NO depth-bound mask on the whole-frame Metal path. Voxy renders straight into MC's
                // own depth attachment, so the ordinary depth test already IS the occlusion, per
                // pixel, for free -- there is nothing for a mask to add. This is the Metal-native
                // equivalent of what upstream does with `initDepthStencil` (full-screen blit of MC's
                // depth + `glStencilFunc(GL_EQUAL, 1)` so Voxy draws only where MC has not); the
                // stencil half has no Metal analogue because MC's main depth target is
                // Depth32Float with no stencil, and the depth half is already in the buffer Voxy
                // draws into.
                //
                // What replaced it here was a chunk-AABB mask rasterized into a depth buffer and
                // read through an SSBO. That is chunk-granular by construction: a 16^3 box hides LOD
                // for the section's whole volume, including the empty air inside it, which showed as
                // sky-coloured rectangles ringing every vanilla chunk. Three attempts to fix it by
                // choosing which sections entered the mask could not have worked, because the box
                // was the problem, not its membership.
                //
                // The residual gap this leaves is COVERAGE, not granularity: the hook is Sodium's
                // CUTOUT pass, so SOLID and CUTOUT_MIPPED are in the depth buffer but CUTOUT and
                // TRANSLUCENT are not yet. LOD can therefore still land where water or cutout foliage
                // will draw. That is an ordering problem and belongs fixed as one.
                // VOXY_NO_DEPTH_BOUND is no longer injected: quads.frag's depth-bound mask test is
                // deleted, so the define had nothing left to guard. It was injected by DEFAULT, which
                // is why the mask was compiled out of every shipping build while a comment claimed it
                // was live -- see the note where the test used to be.

                // quads.frag's second colour output: this fragment's depth as colour, which is the Hi-Z
                // pyramid's source. Metal ONLY, and that is not tidiness -- a shader declaring an output
                // the pipeline has no attachment for is a Metal validation failure, and the GL pass has
                // one colour attachment where the Metal pass now has two.
                opaqueDefines.put("VOXY_LOD_DEPTH_COLOUR", "");
                translucentDefines.put("VOXY_LOD_DEPTH_COLOUR", "");
                // With the mask gone, vanilla and the LOD are compared purely by depth -- and since
                // the LOD approximates the surface vanilla draws, their depths agree to float
                // precision where they overlap and they z-fight. Bias the LOD behind so vanilla
                // always wins ties. See quads3.vert's VOXY_LOD_DEPTH_BIAS for the sign, which is a
                // property of the reverse-Z frame rather than a free choice.
                String lodBias = System.getenv("VOXY_LOD_DEPTH_BIAS");
                if (lodBias == null || lodBias.isBlank()) lodBias = "1e-5";
                opaqueDefines.put("VOXY_LOD_DEPTH_BIAS", lodBias + "f");
                translucentDefines.put("VOXY_LOD_DEPTH_BIAS", lodBias + "f");
                // `noDepthBound` and `boundDebug` used to sit here as constants feeding the
                // [Metal-DEFINES] prose below. Both were assigned once and never reassigned, so the
                // prose they fed described a branch that could not vary -- and the prose was
                // hand-maintained besides, which is why that log line misled this investigation twice.
                // It now prints the ACTUAL define key set instead of a sentence someone remembered.
                opaqueDefines.put("VOXY_FORCE_OPAQUE_ALPHA", "");
                // VOXY_LOD_FORCE_MAGENTA=1 -- bisection switch (see quads.frag). Solid magenta
                // emitted before every discard/early-out, so the frame shows whether LOD geometry
                // rasterizes at all. Diagnostic only.
                if ("1".equals(System.getenv("VOXY_LOD_FORCE_MAGENTA"))) {
                    opaqueDefines.put("VOXY_LOD_FORCE_MAGENTA", "");
                    translucentDefines.put("VOXY_LOD_FORCE_MAGENTA", "");
                }
                // Vertex-stage bisection (see quads3.vert). Defines reach both stages.
                if ("1".equals(System.getenv("VOXY_LOD_FORCE_VERTEX"))) {
                    opaqueDefines.put("VOXY_LOD_FORCE_VERTEX", "");
                    translucentDefines.put("VOXY_LOD_FORCE_VERTEX", "");
                }
                // VOXY_LOD_SHOW_LIGHT=1 -- reads the LOD's own light data back out as colour:
                // red = SKY light, green = BLOCK light, both /15, blue pinned at 0.5 for every
                // fragment the shader emits. Answers the one question the black-splotch
                // investigation kept failing to settle by argument -- whether a dark patch is
                // UNLIT geometry or ABSENT geometry -- because it bypasses every downstream term
                // (atlas, tint, fog, brightness) and paints the raw byte that
                // VoxelIngestService.getLightingSupplier produced for that voxel. Blue is what
                // separates the two: a dark patch with b == 128 is drawn geometry with a zero
                // light byte, and anything else means no fragment was emitted there at all.
                if ("1".equals(System.getenv("VOXY_LOD_SHOW_LIGHT"))) {
                    opaqueDefines.put("VOXY_LOD_SHOW_LIGHT", "");
                    translucentDefines.put("VOXY_LOD_SHOW_LIGHT", "");
                    // R and G were logged the wrong way round here. The shader unpacks
                    // `interData.w >> 24 & 0xF` as red and `>> 28 & 0xF` as green, and
                    // quad_util.glsl puts the light byte at bits 24-31 as `(lighting & 0xFF) << 24`
                    // where lighting is the byte the ingest wrote -- sky in the LOW nibble. So red
                    // is sky. The old text said the opposite, which is a wrong label on the one
                    // instrument whose whole job is to be read literally.
                }

                // VOXY_LOD_FIXED_MIP — sample atlas at LOD 0 instead of the
                //   derivative-based mip. DEFAULT ON for Metal (2026-06-09): the
                //   dFdx/dFdy-based mip collapses to the smallest mip on Metal,
                //   flattening every face to its texture's average colour (the
                //   "paper" look). Forcing mip 0 restored full texture detail at
                //   no measured FPS cost (user-verified ~111 fps).
                //   VOXY_LOD_FIXED_MIP=0 opts back into derivative mips.
                // VOXY_LOD_NO_DISCARD — skip the alpha discard (tests whether the
                //   discard is punching the flickering transparent holes).
                String fixedMipEnv = System.getenv("VOXY_LOD_FIXED_MIP");
                boolean lodFixedMip = fixedMipEnv == null || !"0".equals(fixedMipEnv.trim());
                boolean lodNoDiscard = "1".equals(System.getenv("VOXY_LOD_NO_DISCARD"));
                if (lodFixedMip) {
                    opaqueDefines.put("VOXY_LOD_FIXED_MIP", "");
                    translucentDefines.put("VOXY_LOD_FIXED_MIP", "");
                }
                if (lodNoDiscard) {
                    opaqueDefines.put("VOXY_LOD_NO_DISCARD", "");
                    translucentDefines.put("VOXY_LOD_NO_DISCARD", "");
                }
                if (lodFixedMip || lodNoDiscard) {
                }
                // VOXY_LOD_DIST_MIP — analytic distance-based atlas mip
                //   (2026-07-03), DEFAULT ON, takes precedence over the
                //   fixed-mip-0 diagnostic above (#ifdef order in quads.frag).
                //   Fixed mip 0 means NO minification: every distant pixel
                //   picks one arbitrary texel of its 16x16 face cell — the
                //   spyglass moire on LOD water and the pixel shimmer on
                //   distant terrain. Screen-space derivatives stay unusable
                //   (1-2 px quads -> noisy dFdx, the original "paper" collapse),
                //   so quads.frag computes the mip analytically from view
                //   distance, quad lodScale and the per-frame projection scale
                //   (voxyLodParams.x — tracks spyglass FOV). The mip chain has
                //   been in the atlas all along (MipGen bakes + uploads levels
                //   0..LAYERS-1 per cell; cell origins stay 2^lvl-aligned so
                //   NEAREST never crosses cells).
                //   VOXY_LOD_DIST_MIP=0 reverts to fixed mip 0.
                //   VOXY_LOD_MIP_BIAS=<f> biases the level (+0.5 = blurrier).
                String distMipEnv = System.getenv("VOXY_LOD_DIST_MIP");
                boolean lodDistMip = distMipEnv == null || !"0".equals(distMipEnv.trim());
                if (lodDistMip) {
                    float mipBias = 0.0f;
                    String mb = System.getenv("VOXY_LOD_MIP_BIAS");
                    if (mb != null && !mb.isBlank()) {
                        try {
                            mipBias = Float.parseFloat(mb.trim());
                        } catch (NumberFormatException e) {
                            mipBias = 0.0f;
                        }
                    }
                    // VOXY_ATLAS_MAX_LOD_OVERRIDE=<n>: cap the atlas mip the LOD may sample, so the
                    // level that goes black can be found by bisection instead of by argument. The
                    // CPU-side atlas is verified good at every level (VOXY_ATLAS_DUMP shows level 3
                    // holding the same colour and alpha as level 0 for Dirt/Stone/Grass Block/Sand),
                    // so if a level is black on screen the fault is between that buffer and the
                    // sampler, not in MipGen.
                    int maxLodLevel = me.cortex.voxy.client.core.model.ModelFactory.LAYERS - 1;
                    String maxLodOverride = System.getenv("VOXY_ATLAS_MAX_LOD_OVERRIDE");
                    if (maxLodOverride != null && !maxLodOverride.isBlank()) {
                        try {
                            maxLodLevel = Math.max(0, Integer.parseInt(maxLodOverride.trim()));
                        } catch (NumberFormatException e) {
                            Logger.warn("VOXY_ATLAS_MAX_LOD_OVERRIDE is not an integer: " + maxLodOverride);
                        }
                    }
                    String maxLod = String.format(java.util.Locale.ROOT, "%.1f", (float) maxLodLevel);
                    String biasStr = String.format(java.util.Locale.ROOT, "%.4f", mipBias);
                    opaqueDefines.put("VOXY_LOD_DIST_MIP", "");
                    opaqueDefines.put("VOXY_ATLAS_MAX_LOD", maxLod);
                    opaqueDefines.put("VOXY_LOD_DIST_MIP_BIAS", biasStr);
                    translucentDefines.put("VOXY_LOD_DIST_MIP", "");
                    translucentDefines.put("VOXY_ATLAS_MAX_LOD", maxLod);
                    translucentDefines.put("VOXY_LOD_DIST_MIP_BIAS", biasStr);
                }
                // VOXY_WATER_FAR_ALPHA — far-water opacity ramp (2026-07-03),
                //   DEFAULT ON, translucent only. Constant vanilla alpha 0.706
                //   out to the horizon lets seafloor/kelp ghost through LOD
                //   water and the fog-coloured bridge clear bleed up through
                //   it (the washed-out flat-blue sheet). Ramp start sits past
                //   the LOD<->MC seam so ring parity (alpha 0.706 exactly at
                //   neutral knobs) is untouched; params ride per frame in
                //   voxyLodParams.yzw (see uploadUniformBuffer).
                //   VOXY_WATER_FAR_ALPHA=0 kills it; =<f> sets the far target
                //   (default 0.95). VOXY_WATER_FAR_ALPHA_START/_END override
                //   the ramp distances in blocks.
                if (WATER_FAR_ALPHA > 0.0f) {
                    translucentDefines.put("VOXY_WATER_FAR_ALPHA", "");
                }
                // VOXY_LOD_ABS_INDENT — lodScale-invariant face indentation
                //   (2026-07-03), DEFAULT ON. quad_util scales the model-space
                //   indent by lodScale, so a level-L water plane sat
                //   0.109*2^L blocks below its cell top: parent planes floated
                //   ~0.9 blocks above child planes → stacked translucent
                //   blending (pale section-aligned veil squares), an exposed
                //   gap band at LOD ring transitions, and wrong mid/far water
                //   heights. VOXY_LOD_ABS_INDENT=0 restores upstream scaling.
                String absIndentEnv = System.getenv("VOXY_LOD_ABS_INDENT");
                boolean absIndent = absIndentEnv == null || !"0".equals(absIndentEnv.trim());
                if (absIndent) {
                    opaqueDefines.put("VOXY_LOD_ABS_INDENT", "");
                    translucentDefines.put("VOXY_LOD_ABS_INDENT", "");
                }
                // Seam-ring brightness parity: GL runs SSAO between opaque and
                // translucent; that pass is parked on Metal, so LOD terrain sits
                // ~10% brighter than AO-darkened Sodium terrain — the visible
                // brightness step at the render-distance boundary. Interim
                // compensation until the SSAO port: darken opaque LOD slightly.
                // VOXY_LOD_BRIGHTNESS=<f> tunes it; 1.0 disables.
                // vx contract mode: the pack's deferred applies ITS OWN AO
                // to LOD pixels (BSL deferred.glsl reads vxDepthTexOpaque),
                // so the interim darkening would double-darken — neutral.
                boolean vxContract = me.cortex.voxy.client.core.util.IrisUtil.vxContractActive();
                if (!vxContract) {
                    float brightness = 0.92f;
                    String b = System.getenv("VOXY_LOD_BRIGHTNESS");
                    if (b != null && !b.isBlank()) {
                        try {
                            brightness = Float.parseFloat(b.trim());
                        } catch (NumberFormatException e) {
                            brightness = 0.92f;
                        }
                    }
                    if (brightness != 1.0f) {
                        opaqueDefines.put("VOXY_LOD_BRIGHTNESS", String.format(java.util.Locale.ROOT, "%.4f", brightness));
                    }
                }
                // Water parity knobs — NEUTRAL by default since the chunk-bound
                // depth mask landed. The 0.90/0.85 interim defaults were tuned
                // for blending against the bright fog clear; with the bound the
                // seam background is real LOD seafloor and any non-neutral
                // value CREATES a tone step at the LOD<->MC water line (MC
                // water is alpha 0.706 exactly; LOD matches term-for-term at
                // neutral). Tunables kept for experiments:
                // VOXY_WATER_SHADE (1.0 = off), VOXY_WATER_MIN_ALPHA (0 = off).
                {
                    float waterShade = 1.0f;
                    float waterMinAlpha = 0.0f;
                    String ws = System.getenv("VOXY_WATER_SHADE");
                    if (ws != null && !ws.isBlank()) {
                        try {
                            waterShade = Float.parseFloat(ws.trim());
                        } catch (NumberFormatException e) {
                            waterShade = 1.0f;
                        }
                    }
                    String wa = System.getenv("VOXY_WATER_MIN_ALPHA");
                    if (wa != null && !wa.isBlank()) {
                        try {
                            waterMinAlpha = Float.parseFloat(wa.trim());
                        } catch (NumberFormatException e) {
                            waterMinAlpha = 0.0f;
                        }
                    }
                    if (waterShade != 1.0f) {
                        translucentDefines.put("VOXY_WATER_SHADE", String.format(java.util.Locale.ROOT, "%.4f", waterShade));
                    }
                    if (waterMinAlpha > 0.0f) {
                        translucentDefines.put("VOXY_WATER_MIN_ALPHA", String.format(java.util.Locale.ROOT, "%.4f", waterMinAlpha));
                    }
                    if (waterShade != 1.0f || waterMinAlpha > 0.0f) {
                    }
                }
                // Water diagnostic: paint translucent LOD water solid magenta so
                // a screenshot reveals exactly where water geometry rasterizes.
                if ("1".equals(System.getenv("VOXY_LOD_WATER_DEBUG"))) {
                    translucentDefines.put("VOXY_LOD_WATER_DEBUG", "");
                }
                // Depth bias for translucent LOD water (toward the camera).
                // DEFAULT 0 (off): testing on 2026-05-26 proved the water "holes"
                // are NOT z-fighting — VOXY_LOD_WATER_DEBUG (magenta + depth OFF)
                // still showed holes, i.e. the translucent water GEOMETRY itself
                // is missing in those patches (an LOD meshing/coverage gap, not a
                // depth-test loss). A non-zero bias here only caused artifacts
                // ("water bleeds through the ground"), so it stays off. Kept as a
                // tunable knob (VOXY_WATER_DEPTH_BIAS=<f>) for future experiments.
                {
                    String waterBias = System.getenv("VOXY_WATER_DEPTH_BIAS");
                    if (waterBias == null || waterBias.isBlank()) waterBias = "0";
                    try {
                        Float.parseFloat(waterBias.trim());
                    } catch (NumberFormatException e) {
                        waterBias = "0";
                    }
                    translucentDefines.put("VOXY_WATER_DEPTH_BIAS", waterBias.trim());
                    if (!"0".equals(waterBias.trim())) {
                    }
                }
                // Real translucent water is now the default (quads.frag falls
                // through to the atlas+tint+blend path). VOXY_LOD_FLAT_WATER=1
                // restores the interim flat ocean-blue via the VOXY_FLAT_WATER
                // define. Injected ONLY inside this non-GL guard — the old gate
                // (bare TRANSLUCENT) leaked the flat colour into plain-GL runs
                // because TRANSLUCENT is injected for every backend above (~:240).
                if ("1".equals(System.getenv("VOXY_LOD_FLAT_WATER"))) {
                    translucentDefines.put("VOXY_FLAT_WATER", "");
                }

                // M13 diagnostic — log the Metal shader define set ONCE at
                // construction so it's unambiguous in the runtime log
                // which terrain shader variant compiled. Catches "the
                // expected define wasn't injected" bugs that pure source
                // grep can't.
                boolean bakeryOff = "1".equals(System.getenv("VOXY_BAKERY_OFF"));
                boolean debugMissing = "1".equals(System.getenv("VOXY_BAKERY_DEBUG_MISSING"));
                // The depth-bound clause is gone from this prose with the mask itself; both the defining
                // branch and the red-tint variant it named are deleted from quads.frag.
                Logger.info("[Metal-DEFINES] terrain shader injections: " +
                        "VOXY_FORCE_OPAQUE_ALPHA" +
                        (bakeryOff
                                ? " + VOXY_NO_ATLAS (bakery disabled hash-colour fallback)"
                                : (debugMissing ? " + VOXY_DEBUG_MAGENTA_MISSING" : " + atlas bakery")) +
                        (pipeline.useEnvFog() ? " + USE_ENV_FOG" : "") +
                        // The list above is HAND-MAINTAINED, so it reports what someone remembered to
                        // write down, not what the shader got. Appending the real map makes the line
                        // evidence: a define missing here was never injected, whatever the prose says.
                        " | actual keys: " + opaqueDefines.keySet());

                // Default Metal now uses the real atlas path. VOXY_BAKERY_OFF
                // is retained as a runtime kill switch: ModelTextureBakery
                // writes synthetic face-visibility data and the shader skips
                // atlas sampling, restoring the old hash-colour fallback.
                if (bakeryOff) {
                    opaqueDefines.put("VOXY_NO_ATLAS", "");
                    translucentDefines.put("VOXY_NO_ATLAS", "");
                } else if (debugMissing) {
                    opaqueDefines.put("VOXY_DEBUG_MAGENTA_MISSING", "");
                    translucentDefines.put("VOXY_DEBUG_MAGENTA_MISSING", "");
                }

                // M13 chunk 5: on the Metal terrain path we apply fog
                // per-fragment inside quads.frag because the GL post-pass
                // (NormalRenderPipeline.finish) is skipped on this backend.
                // Gated on the pipeline's useEnvFog() so Iris / chunk-debug
                // pipelines don't pull fog in. The CPU-side uploadUniformBuffer
                // packs the fog params into SceneUniform regardless — at
                // zero alpha if the flag is off — but the shader only reads
                // them when this define is present.
                if (pipeline.useEnvFog()) {
                    opaqueDefines.put("USE_ENV_FOG", "");
                    translucentDefines.put("USE_ENV_FOG", "");
                }

                // M13 2026-05-14 baseInstance workaround. Metal's
                // drawIndexedPrimitives:indirectBuffer: doesn't propagate
                // the indirect args' baseInstance to [[base_instance]] in
                // the vertex function — diagnosed via shader probes
                // (gl_BaseInstance + gl_InstanceID both read 0). The
                // MetalRenderEncoder.drawIndexedIndirect loop pushes the
                // per-draw baseInstance via setVertexBytes at binding 6;
                // quads3.vert reads it from a small UBO when this define
                // is set, instead of gl_BaseInstance.
                opaqueDefines.put("VOXY_METAL_BI_FIX", "");
                translucentDefines.put("VOXY_METAL_BI_FIX", "");
            }

            // NOTE: MDIC terrain pipelines do NOT opt into supportIndirectCommandBuffers.
            // quads.frag uses gl_FragDepth writes + discard, both incompatible
            // with Metal's ICB linking ("Fragment shader cannot be used with
            // indirect command buffers"). For now MDIC uses the CPU-readback
            // path on Metal (read drawCountCallBuffer back, issue per-draw
            // glMultiDrawElementsIndirect — which MetalRenderEncoder.drawIndexedIndirect
            // implements as a CPU loop). The ICB infrastructure stays available
            // (smoke-tested independently) for future simpler-shader use cases.
            //
            // M12 chunk 6 polish: on non-GL backends the pipeline state uses
            // NO_CULL because the deleted GL draw path explicitly called
            // glDisable(GL_CULL_FACE) at draw time — that override doesn't
            // apply to Metal where the cull mode is baked into the pipeline.
            // Without this, ~half the LOD triangles disappear due to wrong-
            // winding back-face culling.
            me.cortex.voxy.client.core.gpu.PipelineState opaqueState
                    = me.cortex.voxy.client.core.gpu.PipelineState.OPAQUE_MESH;
            me.cortex.voxy.client.core.gpu.PipelineState translucentState
                    = me.cortex.voxy.client.core.gpu.PipelineState.TRANSLUCENT_MESH;
            {
                // DIAGNOSTIC (2026-05-25): VOXY_LOD_NO_DEPTH=1 disables the LOD
                // opaque depth test/write to check whether the view-dependent
                // flicker is z-fighting in the LOD's own depth buffer (overlapping
                // LOD geometry competing for depth; winner flips with tiny camera
                // angle changes). If the per-angle disappearing stops, depth/z-fight
                // is confirmed (the image may look unordered with depth off).
                boolean lodNoDepth = "1".equals(System.getenv("VOXY_LOD_NO_DEPTH"));
                if (lodNoDepth) {
                }
                boolean reverseZ = MetalMvpUtil.REVERSE_Z_REMAP;
                var opaqueDepth = lodDepthState(reverseZ, lodNoDepth, /*writeEnabled*/ true);
                if (reverseZ && !reverseZLogged) {
                    reverseZLogged = true;
                }
                opaqueState = new me.cortex.voxy.client.core.gpu.PipelineState(
                        opaqueDepth,
                        me.cortex.voxy.client.core.gpu.PipelineState.BlendState.OPAQUE,
                        me.cortex.voxy.client.core.gpu.PipelineState.RasterState.NO_CULL);
                // VOXY_LOD_WATER_DEBUG also DISABLES the depth test for the
                // translucent pass so the magenta shows ALL water geometry
                // regardless of depth — distinguishing "water missing" (coverage
                // / meshing) from "water depth-rejected" (z-fight vs seafloor).
                boolean waterDebugDepth = "1".equals(System.getenv("VOXY_LOD_WATER_DEBUG"));
                // Phase D (issue #11): in vx-contract mode the translucent
                // pass renders into its OWN depth attachment (seeded with
                // opaque depth) and must WRITE depth — the water surface
                // depth becomes vxDepthTexTrans, which the pack's deferred
                // uses to composite LOD water as water.
                // Same depth convention as the opaque pass -- the compare operator belongs to the
                // frame's depth buffer, not to this pass. Leaving this on the GL-convention presets
                // while the opaque pass was fixed is what put water in front of everything and made
                // it un-occludable.
                var transDepthState = lodDepthState(reverseZ, waterDebugDepth,
                        /*writeEnabled*/ me.cortex.voxy.client.core.util.IrisUtil.vxContractActive());
                // Material mode: the 3 g-buffer planes carry DATA (straight-alpha
                // albedo + nibble-packed light/face/id), not composited colour —
                // the pack's blender does the real compositing at resolve time.
                // Blending them is doubly wrong: MetalRenderBackend only wires
                // blending onto attachment 0 (planes 1/2 were silently
                // last-writer-wins anyway), and plane 0 would premultiplied-over-
                // accumulate straight-alpha samples under any overlapping draws
                // (RGB x1.294 / alpha 0.914 for water — "pale, more opaque").
                // OPAQUE writes + depth LEQUAL+write make all 3 planes agree on
                // nearest-fragment-wins. VOXY_VX_PLANE_BLEND=1 restores blending.
                var transBlend = vxMaterial && !"1".equals(System.getenv("VOXY_VX_PLANE_BLEND"))
                        ? me.cortex.voxy.client.core.gpu.PipelineState.BlendState.OPAQUE
                        : me.cortex.voxy.client.core.gpu.PipelineState.BlendState.PREMULTIPLIED_ALPHA;
                // transDepthState already carries the waterDebugDepth kill switch (lodDepthState
                // maps a disabled test to DISABLED).
                translucentState = new me.cortex.voxy.client.core.gpu.PipelineState(
                        transDepthState,
                        transBlend,
                        me.cortex.voxy.client.core.gpu.PipelineState.RasterState.NO_CULL);
            }
            // Material mode renders 3 BGRA8 planes (P0 albedo, P1 tint, P2 misc);
            // otherwise the single colour attachment as before.
            // 3-plane material g-buffer only for the layer(s) that go through the resolve:
            // opaque only when opted in, translucent under vxMaterial; else single bridge colour.
            int[] threePlane = new int[]{GL_RGBA8, GL_RGBA8, GL_RGBA8};
            int[] onePlane = new int[]{GL_RGBA8};
            int[] opaqueFormats = vxOpaqueMat ? threePlane : onePlane;
            int[] translucentFormats = vxMaterial ? threePlane : onePlane;
            // The Metal LOD pass carries a SECOND colour attachment -- quads.frag's depth-as-colour,
            // which is the Hi-Z pyramid's source (VOXY_LOD_DEPTH_COLOUR). The pipeline has to declare a
            // format for it, and this is the step that was missing: with the define injected and the
            // attachment on the pass but only one format declared, Metal DROPS the shader's second
            // output instead of erroring, so the attachment read empty and nothing anywhere said why.
            {
                // GL_R32F, NOT GL_RGBA8. The attachment carries reverse-Z depth, which for distant LOD
                // terrain is ~1e-4..1e-3; in an 8-bit-per-channel format that quantises to zero, so the
                // pyramid's textureGather -- which reads the RED channel -- got a buffer of zeros and the
                // cull had nothing to test. R32F holds it exactly. See AbstractRenderPipeline's
                // metalDepthTex allocation for the full account of how RGBA8 got here.
                opaqueFormats = java.util.Arrays.copyOf(opaqueFormats, opaqueFormats.length + 1);
                opaqueFormats[opaqueFormats.length - 1] = org.lwjgl.opengl.GL30C.GL_R32F;
                translucentFormats = java.util.Arrays.copyOf(translucentFormats,
                        translucentFormats.length + 1);
                translucentFormats[translucentFormats.length - 1] = org.lwjgl.opengl.GL30C.GL_R32F;
            }
            this.terrainPipeline = this.backend.createGraphicsPipeline(
                    new me.cortex.voxy.client.core.gpu.GraphicsPipelineDesc(
                            vertex, vxOpaqueFrag, opaqueDefines,
                            null, null, null, null,
                            opaqueFormats,
                            me.cortex.voxy.client.core.gpu.VertexLayout.EMPTY,
                            opaqueState,
                            "MDIC.terrain"));
            this.translucentTerrainPipeline = this.backend.createGraphicsPipeline(
                    new me.cortex.voxy.client.core.gpu.GraphicsPipelineDesc(
                            vertex, vxTransFrag, translucentDefines,
                            null, null, null, null,
                            translucentFormats,
                            me.cortex.voxy.client.core.gpu.VertexLayout.EMPTY,
                            translucentState,
                            "MDIC.translucentTerrain"));
        }
    }

    /** Mirror addDirectionalFaceTint + the TAA flag so the cross-backend pipeline desc gets the same defines. */
    private static java.util.Map<String, String> buildTerrainDefines(String taa) {
        var m = new java.util.LinkedHashMap<String, String>();
        net.minecraft.client.multiplayer.ClientLevel level = Minecraft.getInstance().level;
        if (level != null) {
            m.put("NO_SHADE_FACE_TINT", Float.toString(level.cardinalLighting().byFace(Direction.UP)) + "f");
            m.put("UP_FACE_TINT",       Float.toString(level.cardinalLighting().byFace(Direction.UP))  + "f");
            m.put("DOWN_FACE_TINT",     Float.toString(level.cardinalLighting().byFace(Direction.DOWN))+ "f");
            m.put("Z_AXIS_FACE_TINT",   Float.toString(level.cardinalLighting().byFace(Direction.NORTH))+ "f");
            m.put("X_AXIS_FACE_TINT",   Float.toString(level.cardinalLighting().byFace(Direction.EAST)) + "f");
        }
        if (taa != null) m.put("TAA_PATCH", "");
        // Per-chunk-column cull of the LOD against vanilla's built sections. Lives in the COMMON
        // defines so both LOD passes get it: the cull has to apply to the translucent surface too,
        // or LOD water survives over vanilla water and nowhere else, which is a far more visible
        // artifact than a terrain seam.
        if (CHUNK_CULL) {
            m.put("VOXY_LOD_CHUNK_CULL", "");
            m.put("VOXY_LOD_CHUNK_CULL_BINDING", Integer.toString(BUILT_MASK_CHUNK_BINDING));
        }
        // VOXY_LOD_CULL_DEBUG=1: paint every LOD fragment with the cull's own column index and bit
        // (R=cx, G=cz, B=bit). A readout of the shader's arithmetic, which no CPU-side counter can see.
        if ("1".equals(System.getenv("VOXY_LOD_CULL_DEBUG"))) {
            m.put("VOXY_LOD_CULL_DEBUG", "");
        }
        // VOXY_LOD_CULL_SHOW=1: paint culled fragments red instead of discarding them. A diagnostic,
        // and the only way to tell a cull boundary apart from the edge of Voxy's own LOD coverage --
        // both are "no magenta" in a forced-colour frame. Paired with VOXY_LOD_FORCE_MAGENTA the frame
        // reads MAGENTA = LOD drawn, RED = culled by the section mask, SKY = no LOD geometry.
        if ("1".equals(System.getenv("VOXY_LOD_CULL_SHOW"))) {
            m.put("VOXY_LOD_CULL_SHOW", "");
        }
        // VOXY_LOD_SHOW_DRAWID=1: paint the per-draw section index the vertex stage received. Injected
        // into the TERRAIN defines so quads3.vert/quads.frag see it, and into the common map so both LOD
        // passes get it -- they read the same pushed constant. Metal-specific concern: on GL the index
        // arrives via gl_BaseInstance.
        if ("1".equals(System.getenv("VOXY_LOD_SHOW_DRAWID"))) {
            m.put("VOXY_LOD_SHOW_DRAWID", "");
        }
        // VOXY_BI_OFFSET=1: take the per-draw section index from a buffer offset instead of the pushed
        // constant. The shader and the encoder must agree, so this define and the encoder's env read are
        // the same variable.
        if ("1".equals(System.getenv("VOXY_BI_OFFSET"))) {
            m.put("VOXY_BI_OFFSET", "");
        }
        // VOXY_LOD_FLAT_FRAG=1: the fill-vs-per-draw probe. Emits a constant colour from the fragment
        // stage as early as the cull allows, skipping the per-fragment mip computation, all three atlas
        // fetches, the tinting, the alpha cutout and the fog.
        //
        // The image it produces is meaningless -- every LOD surface becomes one flat colour -- and that
        // is the point. The frame costs ~50 ms, ~28 ms of which is the GPU drain, and nothing in the tree
        // can currently say whether that is per-pixel shading or the ~29k indirect draw commands
        // themselves. If `submit` collapses under this probe, the cost is per-pixel and the overdraw and
        // per-fragment work are the target; if it does not move, the cost is the command count and the
        // fragment stage is not worth optimising at all. See optimisation.MD section 7.
        if ("1".equals(System.getenv("VOXY_LOD_FLAT_FRAG"))) {
            m.put("VOXY_LOD_FLAT_FRAG", "");
        }
        return m;
    }

    /**
     * The exact matrix uploaded as {@code SceneUniform.MVP} — camera-relative, plus whichever
     * clip-space depth remap this backend needs. Kept as its own method so the geometry trace can
     * feed {@link me.cortex.voxy.client.core.rendering.util.LodVertexMath} the same matrix the
     * shader gets, rather than a reconstruction that could differ from it.
     */
    private Matrix4f lodMvp(MDICViewport viewport) {
        var mat = new Matrix4f(viewport.MVP);
        mat.translate(-viewport.innerTranslation.x, -viewport.innerTranslation.y, -viewport.innerTranslation.z);
        // No clip-space depth remap any more. The whole-frame path now runs vanilla's projection
        // unchanged (see VoxyRenderSystem.computeProjectionMat), so the LOD's depth values are
        // already in the frame's reverse-Z space; remapping here would push them out of it again and
        // reintroduce exactly the mismatch the depth test depends on not having. On GL neither remap
        // ever applied, so this block was non-GL-only and is simply gone.
        return mat;
    }

    private void uploadUniformBuffer(MDICViewport viewport) {
        long ptr = UploadStream.INSTANCE.upload(this.uniformFor(viewport), 0, 1024);
        long base = ptr;

        var mat = this.lodMvp(viewport);
        // VOXY_VP_TRACE=1: the VP actually uploaded, plus the same matrix WITHOUT the reverse-Z
        // remap for comparison. The LOD's quads are positioned entirely by this matrix, so a
        // degenerate one collapses every vertex to a point: no fragments, no error, and every
        // counter reading healthy -- which is the signature this whole investigation has had.
        if ("1".equals(System.getenv("VOXY_VP_TRACE")) && (vpTraceCount++ % 600) == 1) {
            var unremapped = new Matrix4f(viewport.MVP);
            unremapped.translate(-viewport.innerTranslation.x, -viewport.innerTranslation.y,
                    -viewport.innerTranslation.z);
        }
        mat.getToAddress(ptr); ptr += 4*4*4;

        viewport.section.getToAddress(ptr); ptr += 4*3;

        if (viewport.frameId<0) {
            Logger.error("Frame ID negative, this will cause things to break, wrapping around");
            viewport.frameId &= 0x7fffffff;
        }
        MemoryUtil.memPutInt(ptr, viewport.frameId&0x7fffffff); ptr += 4;
        viewport.innerTranslation.getToAddress(ptr); ptr += 4*3;

        // std140 padding: cameraSubPos (vec3 at offset 80) consumes 12B; the
        // next vec4 must be 16B-aligned, so skip the 4B trailing pad before
        // writing voxyFogEndParams + voxyFogColour. Using offsets from `base`
        // rather than chained `ptr +=` reads more locally about the std140
        // layout and is robust to future intermediate fields.
        // M13 chunk 5: pack the env-fog parameters that quads.frag reads when
        // USE_ENV_FOG is defined. Mirrors NormalRenderPipeline.finish so the
        // Metal terrain-shader fog matches what the GL post-pass would do.
        // When the pipeline doesn't want fog the whole 32-byte tail is zeroed
        // (fogColour.a == 0 makes the shader's mix a no-op anyway).
        long fogBase = base + 96; // matches SceneUniform's std140 layout
        if (this.pipeline.useEnvFog() && viewport.fogParameters != null) {
            float start = viewport.fogParameters.environmentalStart();
            float end   = viewport.fogParameters.environmentalEnd();
            if (Math.abs(end - start) > 1) {
                float invEndFogDelta = 1f / (end - start);
                float endDistance = Math.max(
                        (Minecraft.getInstance().options.renderDistance().get() * 16),
                        20 * 16);
                endDistance *= (float) Math.sqrt(3);
                float startDelta = -start * invEndFogDelta;
                MemoryUtil.memPutFloat(fogBase +  0, invEndFogDelta);
                MemoryUtil.memPutFloat(fogBase +  4, startDelta);
                MemoryUtil.memPutFloat(fogBase +  8,
                        Math.clamp(endDistance * invEndFogDelta + startDelta, 0f, 1f));
                MemoryUtil.memPutFloat(fogBase + 12, 0f);
                MemoryUtil.memPutFloat(fogBase + 16, viewport.fogParameters.red());
                MemoryUtil.memPutFloat(fogBase + 20, viewport.fogParameters.green());
                MemoryUtil.memPutFloat(fogBase + 24, viewport.fogParameters.blue());
                MemoryUtil.memPutFloat(fogBase + 28, viewport.fogParameters.alpha());
            } else {
                MemoryUtil.memSet(fogBase, 0, 32);
            }
        } else {
            MemoryUtil.memSet(fogBase, 0, 32);
        }

        // voxyLodParams (2026-07-03): distance-mip projection scale + the
        // far-water alpha ramp. Offset 128 = right after voxyFogColour in
        // SceneUniform's std140 layout (the buffer is 1024 B, so no growth).
        // Written on every backend — GL shaders declare the field but no GL
        // code path reads it (VOXY_LOD_DIST_MIP / VOXY_WATER_FAR_ALPHA are
        // Metal-only injections), so this is provably no-op on GL.
        {
            long lodBase = base + 128;
            // World units per pixel per unit view distance, from THIS frame's
            // projection: 2*tan(fovY/2)/viewportH == 2/(m11*viewportH). Tracks
            // spyglass zoom + window resizes. NaN/degenerate projection (the
            // known first-frames state, see HierarchicalOcclusionTraverser's
            // NaN guard) uploads 0, which quads.frag treats as "mip 0".
            float projK = 0.0f;
            float m11 = viewport.projection.m11();
            if (!Float.isNaN(m11) && m11 > 1e-6f && viewport.height > 0) {
                projK = 2.0f / (m11 * viewport.height);
            }
            // Ramp window: start past the LOD<->MC seam (seam parity keeps MC's
            // exact 0.706), reach the target alpha a few render distances out.
            float rdBlocks = Math.max((Minecraft.getInstance().options.renderDistance().get() * 16), 32f);
            float rampStart = WATER_FAR_ALPHA_START > 0f
                    ? WATER_FAR_ALPHA_START
                    : Math.max(rdBlocks * 1.5f, 384f);
            float rampEnd = WATER_FAR_ALPHA_END > rampStart
                    ? WATER_FAR_ALPHA_END
                    : Math.max(rdBlocks * 4f, rampStart + 768f);
            MemoryUtil.memPutFloat(lodBase,      projK);
            MemoryUtil.memPutFloat(lodBase +  4, rampStart);
            MemoryUtil.memPutFloat(lodBase +  8, 1.0f / (rampEnd - rampStart));
            MemoryUtil.memPutFloat(lodBase + 12, WATER_FAR_ALPHA);
            // voxyLodParams2.x: translucent near-cull distance (vx contract —
            // see VOXY_TRANS_NEAR_CULL). GL and no-pack sessions read 0.
            // voxyLodParams2.x: was the translucent near-cull distance, a camera-distance proxy for
            // "inside the MC render distance". The per-section cull covers translucent now (it rides in
            // the common defines), so this is 0 and the shader reads it as disabled.
            MemoryUtil.memPutFloat(lodBase + 16, 0f);
            MemoryUtil.memPutFloat(lodBase + 20, 0f);
            MemoryUtil.memPutFloat(lodBase + 24, 0f);
            MemoryUtil.memPutFloat(lodBase + 28, 0f);
        }

        UploadStream.INSTANCE.commit();
    }


    /**
     * The GL opaque/temporal/translucent draw entry points are gone: this renderer has no GL draw
     * path. The abstract superclass still declares them, and the live path is
     * {@link #renderOpaqueMetal}/{@link #renderTemporalMetal}/{@link #renderTranslucentMetal},
     * which {@code AbstractRenderPipeline.runPipelineMetal} calls directly with a {@link RenderEncoder}.
     */
    @Override
    public void renderOpaque(MDICViewport viewport) {
    }

    /**
     * M12 chunk 6 step 3 — Metal-only opaque draw via {@link RenderEncoder}.
     * Called from {@code AbstractRenderPipeline.runPipelineMetal} inside a
     * render pass that targets the IOSurface bridge color + Voxy's
     * Metal-side depth texture.
     *
     * <p>This is the only terrain draw there is; the raw-GL {@code renderTerrain}
     * it used to be contrasted with is deleted.
     * <ul>
     *   <li>All state flows through {@link RenderEncoder.setPipeline}
     *       / {@code setBuffer} — no raw program or buffer binding.</li>
     *   <li>No {@code setupAndBindOpaque} — the render pass already targets
     *       the bridge; there's no separate FBO bind step.</li>
     *   <li>Lightmap (binding 1 sampler) is bound via
     *       {@code LightMapHelper.bindMetal}, which binds MC's own lightmap
     *       texture — it is already a Metallum texture, so there is no
     *       mirror and no per-frame copy. Depth-bounding texture (binding 2
     *       sampler) stays unbound — it depends on M13 chunk 3's MC depth
     *       import.</li>
     *   <li>Model atlas texture + sampler (binding 0) also skipped —
     *       {@code ModelTextureBakery} is GL-only so the atlas is blank on
     *       Metal anyway; only the model + colour SSBOs feed shape data.</li>
     *   <li>{@code drawIndexedIndirect} with CPU-side {@code maxDrawCount}
     *       instead of {@code drawIndexedIndirectCount} — Metal has no
     *       count-aware MDI compatible with quads.frag (see gotcha #16);
     *       the bound is clamped to the GPU-written count via
     *       {@link #metalDrawCount} (CPU-coherent after runPipelineMetal's
     *       post-buildDrawCalls submit), so stale slots are never drawn.</li>
     * </ul>
     */
    public void renderOpaqueMetal(me.cortex.voxy.client.core.gpu.RenderEncoder encoder, MDICViewport viewport) {
        if (this.geometryManager.getSectionCount() == 0) return;
        pfDraws = pfLightZero = pfModelOob = pfCoarse = pfOrphan = pfStraddle = pfSampled = pfEmptyQuad
                = pfModelUnbaked = pfFaceZero = pfWrongSection = pfWrongChecked = pfStaleEntry = 0;
        pfWrongExamples = 0;
        // SceneUniform was already uploaded by buildDrawCalls this frame
        // (runPipelineMetal always pairs them); no re-upload.
        if (this.terrainPipeline == null) {
            // Iris-patched path — GL-only by construction (see 1e2a1190). Should
            // never hit on Metal because RenderPipelineFactory gates Iris pipeline.
            return;
        }
        int maxDrawCount = Math.min((int)(this.geometryManager.getSectionCount()*4.4+128), 400_000);
        maxDrawCount = metalDrawCount(viewport, OPAQUE_DRAW_COUNT_OFFSET, maxDrawCount);
        lodDrawDiag(viewport, this.geometryManager.getSectionCount(), maxDrawCount);
        if (maxDrawCount != 0) {
            this.renderTerrainMetal(encoder, this.terrainPipeline, viewport, 0L, maxDrawCount);
        }
        // One line per frame, aligned to the screenshot burst by wall clock: a captured frame's
        // filename (2026-09-20_01.02.50.png) carries the same timestamp as these log lines
        // ([01:02:50]), so the catcher's verdicts can be read against the renderer's state.
        if (DRAWCHK != 0) {
            Logger.info("[Metal-PF] draws=" + maxDrawCount + " sampled=" + pfSampled
                    + " lightZero=" + pfLightZero + " modelOob=" + pfModelOob
                    + " modelUnbaked=" + pfModelUnbaked + " faceZero=" + pfFaceZero
                    + " emptyQuad=" + pfEmptyQuad
                    + " WRONGSEC=" + pfWrongSection + "/" + pfWrongChecked + " staleTable=" + pfStaleEntry
                    + " coarseDetail=" + pfCoarse + " orphan=" + pfOrphan + " straddle=" + pfStraddle
                    + " allocs=" + drawchkTableSize);
        }
        // Encoder-sequencing test, drawn AFTER the LOD so the LOD cannot hide it. Deliberately
        // outside the maxDrawCount guard: the question is whether this encoder accepts a draw at
        // all, which must not depend on whether the LOD happened to issue any commands.
        this.drawDebugTriangle(encoder, viewport);
    }

    /**
     * M12 Metal-side temporal render — reuses the opaque terrain pipeline but
     * draws from the temporal slice of {@code drawCallBuffer}
     * ({@code TEMPORAL_OFFSET}+ slots, populated by commandGen.comp for sections
     * that were visible-this-frame-but-not-last).
     */
    public void renderTemporalMetal(me.cortex.voxy.client.core.gpu.RenderEncoder encoder, MDICViewport viewport) {
        if (this.geometryManager.getSectionCount() == 0) return;
        if (this.terrainPipeline == null) return;
        int maxDrawCount = Math.min(this.geometryManager.getSectionCount(), 100_000);
        maxDrawCount = metalDrawCount(viewport, TEMPORAL_DRAW_COUNT_OFFSET, maxDrawCount);
        if (maxDrawCount == 0) return;
        this.renderTerrainMetal(encoder, this.terrainPipeline, viewport,
                /*indirectOffset bytes*/ (long) TEMPORAL_OFFSET * 5L * 4L,
                maxDrawCount);
    }

    /**
     * M12 Metal-side translucent render — uses the dedicated translucent
     * pipeline ({@code TRANSLUCENT_MESH}-style state with depth-test-no-write
     * + premultiplied-alpha blend, both baked in at pipeline creation) and
     * draws from the translucent slice of {@code drawCallBuffer}
     * ({@code TRANSLUCENT_OFFSET}+ slots, populated by buildtranslucents.comp).
     * Same SSBO bindings as opaque since both share quads3.vert + quads.frag;
     * blend + depth state come from the pipeline state, no per-draw GL state
     * changes needed.
     */
    public void renderTranslucentMetal(me.cortex.voxy.client.core.gpu.RenderEncoder encoder, MDICViewport viewport) {
        if (this.geometryManager.getSectionCount() == 0) return;
        if (this.translucentTerrainPipeline == null) return;
        int translucentMax = Math.min(this.geometryManager.getSectionCount(), 100_000);
        translucentMax = metalDrawCount(viewport, TRANSLUCENT_DRAW_COUNT_OFFSET, translucentMax);
        if (translucentMax == 0) return;
        this.renderTerrainMetal(encoder, this.translucentTerrainPipeline, viewport,
                /*indirectOffset bytes*/ (long) TRANSLUCENT_OFFSET * 5L * 4L,
                translucentMax);
    }

    /**
     * Byte offsets of the GPU-written draw counts in {@code drawCountCallBuffer}
     * — fields 4/5/6 of bindings.glsl's DrawCommandCountBuffer (after the three
     * cmdGenDispatch uints). Same offsets the GL path feeds to
     * glMultiDrawElementsIndirectCountARB (4*3 / 4*4 / 4*5).
     */
    private static final long OPAQUE_DRAW_COUNT_OFFSET = 4 * 3;
    private static final long TRANSLUCENT_DRAW_COUNT_OFFSET = 4 * 4;
    private static final long TEMPORAL_DRAW_COUNT_OFFSET = 4 * 5;

    /**
     * Metal-only clamp of the CPU-side maxDrawCount to the count cmdgen.comp /
     * buildtranslucents.comp actually wrote. All three drawCallBuffer slices are
     * written compactly from their slice base (opaque: atomicAdd from slot 0;
     * temporal: atomicAdd from TEMPORAL_OFFSET; translucent: prefix-sum scatter
     * covering [TRANSLUCENT_OFFSET, +count) exactly once), so clamping skips
     * only never-written slots. Valid because runPipelineMetal submit()s (and
     * waits) between buildDrawCalls and the render pass — the same coherency
     * the baseInstance CPU-read in MetalRenderEncoder.drawIndexedIndirect
     * already requires. Cuts the per-draw JNI loop from the 150k-450k upper
     * bound to the real count. No-op on GL (drawCountCallBuffer is not a
     * MetalBuffer there).
     */
    /**
     * Diagnostics for "Voxy builds geometry but nothing appears". The render list being non-empty
     * (Metal-FLICKER) says nothing about whether any draw is ISSUED: renderOpaqueMetal returns early
     * when this count reads 0, so a healthy-looking render list and zero draws are entirely
     * compatible. Logs the raw value read from the GPU-written count buffer alongside the bound.
     */
    private static boolean reverseZLogged = false;
    private static long vpTraceCount = 0;
    private static long LOD_DIAG_FRAME = 0;
    private static void lodDrawDiag(MDICViewport viewport, int sectionCount, int maxDrawCount) {
        if ((LOD_DIAG_FRAME++ % 600) != 1) return;
        // rawOpaqueCount is read HERE rather than by the caller, and the placement is the point: it is
        // an instantaneous count, so sampling it once per 600 frames measures exactly what reading it
        // every frame then discarding 599 of the values did. That is unlike [Metal-FLICKER], whose
        // meaning is variance ACROSS frames and which therefore had to be gated rather than moved.
        //
        // The field names and their order, and the two spaces between them, are parsed by the
        // measurement harness (ab_cull_surface.sh, ab_occ.sh, tools/parse_perf.py). Do not reformat.
        final int rawOpaque = rawOpaqueCount(viewport, OPAQUE_DRAW_COUNT_OFFSET);
        me.cortex.voxy.common.Logger.info(String.format(
                "[Metal-LODDRAW f=%d] sections=%d  rawOpaqueCount=%d  maxDrawCount=%d  %s",
                LOD_DIAG_FRAME, sectionCount, rawOpaque, maxDrawCount,
                maxDrawCount == 0 ? "<-- NO DRAWS ISSUED" : "drawing"));
    }

    /** Raw (unclamped) value at the opaque draw-count offset, or -1 if unreadable. */
    private static int rawOpaqueCount(MDICViewport viewport, long countOffset) {
        // CONSUME slot: the counts belonging to the commands the draws will actually issue. Reading the
        // write slot here would describe frame N while frame N-1 is drawn.
        if (viewport.drawCountConsume(viewport.frameId) instanceof me.cortex.voxy.client.core.metal.MetalBuffer mb) {
            long p = mb.getContentsPtr();
            if (p != 0) {
                return MemoryUtil.memGetInt(p + countOffset);
            }
        }
        return -1;
    }

    private static int metalDrawCount(MDICViewport viewport, long countOffset, int upperBound) {
        // CONSUME slot, and this is the load-bearing one: the value returned here becomes
        // maxDrawCount, the CPU-side bound on how many draws are issued. Taking it from the write slot
        // would bound frame N-1's commands by frame N's count.
        if (viewport.drawCountConsume(viewport.frameId) instanceof me.cortex.voxy.client.core.metal.MetalBuffer mb) {
            long p = mb.getContentsPtr();
            if (p != 0) {
                int actual = MemoryUtil.memGetInt(p + countOffset);
                // unsigned compare: garbage >= 2^31 must not wrap negative past min()
                if (Integer.compareUnsigned(actual, upperBound) < 0) return actual;
            }
        }
        return upperBound;
    }

    /** Opt-in command dump ({@code VOXY_CMD_TRACE=1}). */
    static final boolean CMD_TRACE = "1".equals(System.getenv("VOXY_CMD_TRACE"));
    private static long cmdTraceCount = 0;

    /**
     * The LOD pass issues ~869 indirect draws per frame and contributes ZERO pixels (frames are
     * pixel-identical with Voxy enabled and with {@code VOXY_FORCE_METAL=0}). The draw COUNT being
     * non-zero only says how many commands were written, not that any of them rasterize: a command
     * whose indexCount is 0 is a no-op. Dump the first few so "empty commands" and "valid commands
     * that get clipped or written somewhere invisible" are distinguishable.
     */
    private static void traceCommands(String tag, MDICViewport viewport, long indirectOffset, int maxDrawCount) {
        if (!CMD_TRACE || (cmdTraceCount++ % 600) != 1) return;
        if (!(viewport.drawCallBuffer instanceof me.cortex.voxy.client.core.metal.MetalBuffer mb)) return;
        long p = mb.getContentsPtr();
        if (p == 0) return;
        int n = Math.min(Math.max(maxDrawCount, 0), 4);
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < n; i++) {
            long a = p + indirectOffset + (long) i * 20L;
            sb.append(String.format(java.util.Locale.ROOT,
                    " [#%d indexCount=%d instanceCount=%d firstIndex=%d baseVertex=%d baseInstance=%d]",
                    i, org.lwjgl.system.MemoryUtil.memGetInt(a),
                    org.lwjgl.system.MemoryUtil.memGetInt(a + 4),
                    org.lwjgl.system.MemoryUtil.memGetInt(a + 8),
                    org.lwjgl.system.MemoryUtil.memGetInt(a + 12),
                    org.lwjgl.system.MemoryUtil.memGetInt(a + 16)));
        }
    }

    // ---------------------------------------------------------------------------------------------
    // CPU validation of the draw commands about to be submitted (VOXY_DRAWCHK).
    // ---------------------------------------------------------------------------------------------

    /**
     * How thoroughly {@link #validateDrawCommands} walks the command list. {@code 0} off, {@code 1}
     * sampled on a rotating stride (the default — the full walk is ~450k commands a frame across the
     * three slices and the frame is only a few ms), {@code 2} every command.
     */
    // Default flipped to 0 for one experiment: does the per-frame CPU readback and its log line change
    // the artefact RATE? With the cull and every other file at the state that scores 22.1%, this is the
    // only per-frame work left that a normal run performs, so flipping it decides between two very
    // different readings of the 22.1%-vs-81% gap: either the readback's CPU cost masks the artefact
    // (making it timing-sensitive and every earlier baseline instrument-dependent), or it is irrelevant
    // and the gap is a defect in how the cull was re-applied by hand.
    private static final int DRAWCHK = parseEnvInt("VOXY_DRAWCHK", 0);
    private static final int DRAWCHK_STRIDE = Math.max(1, parseEnvInt("VOXY_DRAWCHK_STRIDE", 16));
    /**
     * Whether to build the live-allocation table and run the orphan/straddle checks against it.
     *
     * <p><b>Off by default, and it must stay that way for anything interactive.</b> The table is built
     * by scanning the whole metadata buffer -- {@code maxSectionCount} entries, 1,048,576 of them, at
     * three calls per frame. That was measured as "a few ms" when it was written and is nothing of the
     * kind: it is tens of megabytes of reads per frame, and with it on the client is visibly slow. The
     * per-quad counters below need no table and cost a handful of reads per sampled command, so this
     * is the only part worth gating.
     */
    private static final boolean DRAWCHK_TABLE =
            "1".equals(System.getenv("VOXY_DRAWCHK_TABLE"));

    /**
     * How often the allocation table is rebuilt from the metadata buffer, in validate calls, when it is
     * enabled. A longer interval lags the streaming state, and then `orphan` counts commands whose
     * allocation appeared after the table was built -- a counter moving for the wrong reason, which is
     * the failure mode this whole investigation keeps rediscovering. Sampling honestly is better than
     * sampling fast.
     */
    private static final int DRAWCHK_TABLE_EVERY = Math.max(1, parseEnvInt("VOXY_DRAWCHK_TABLE_EVERY", 30));

    /**
     * [0]=checked [1]=orphan [2]=straddle [3]=overlap [4]=sidOob [5]=noGroup [6]=posMismatch
     * [7]=listStale(depth) — 1 while the CPU's render-list read is not the frame cmdgen ran on.
     */
    private static final long[] DRAWCHK_TOT = new long[8];
    private static final long[] DRAWCHK_LAST = new long[8];
    private static final String[] DRAWCHK_NAMES =
            {"checked", "orphan", "straddle", "overlap", "sidOob", "noGroup", "posMismatch", "listStale"};
    private static long drawchkCalls = 0;
    private static int drawchkExamples = 0;

    /**
     * Per-frame counters, reset at the start of each opaque render and logged at the end of it.
     *
     * <p>Screenshot filenames carry the same wall clock as the log's own timestamps
     * ({@code 2026-09-20_01.02.50.png} against {@code [01:02:50]}), so a per-frame line is enough to
     * line a captured frame up with the renderer's state at that instant. That is the piece every
     * previous round was missing: the catcher can say *which* frames have the artefact, but without
     * this there is nothing to compare them against.
     */
    private static long pfDraws, pfLightZero, pfModelOob, pfCoarse, pfOrphan, pfStraddle;
    private static long pfSampled, pfEmptyQuad, pfModelUnbaked, pfFaceZero;
    private static long pfWrongSection, pfWrongChecked, pfStaleEntry;
    private static int pfExamples = 0;
    private static int pfWrongExamples = 0;

    /** Live geometry allocations, flat {@code [start, endExclusive, sectionId]} sorted by start. */
    private static long[] drawchkTable = null;
    private static int drawchkTableSize = 0;
    private static int drawchkTableAge = Integer.MAX_VALUE;
    /** Live section count when the table was built; the ownership check is gated on it not moving. */
    private static int drawchkTableSectionCount = -1;

    /**
     * Validate, on the CPU, every draw command this slice is about to submit against the geometry
     * allocations the sections it may draw actually own.
     *
     * <p>A section's 32-byte metadata is all that is needed to reconstruct its allocation. The manager
     * allocates {@code upsized = (itemCount + 1023) & ~1023} elements from a 1024-aligned arena and
     * stores {@code geometryPtr + offsets[0]} in the metadata, while {@code RenderDataFactory} always
     * starts {@code offsets[0]} at 0 — so {@code quadStart} <b>is</b> the allocation address, and the
     * eight packed group counts sum to {@code itemCount}. So the allocation is
     * {@code [quadStart, quadStart + ceil1024(sum(counts)))} and the table can be built by scanning
     * the metadata buffer, which is the same bytes the GPU reads.
     *
     * <p><b>Deliberately does not use the render list.</b> {@code HierarchicalOcclusionTraverser} zeroes
     * that buffer's count with a CPU {@code memset} at the start of the frame and the traversal then
     * writes it on the GPU; the CPU reading it at draw time sees the zeroed value while the entries
     * still hold the previous frame's ids. Measured here: {@code listCount=0} against
     * {@code cmdGenDispatch=(88,1,1)}, i.e. prep computed ~11k sections from that same uint on the
     * GPU. Using it would compare one frame's commands against another frame's section ids and
     * manufacture mismatches — the first version of this check did exactly that and reported
     * {@code posMismatch=35674}, which is that artifact and not a finding. The table below comes from
     * one buffer, written by one pass, so it is internally consistent.
     *
     * <p>What each counter means, none of which needs the render list:
     *
     * <ul>
     *   <li><b>orphan</b> — the command's quad range lies inside no live allocation. The draw reads
     *       geometry no section owns, i.e. arbitrary shared-buffer contents.</li>
     *   <li><b>straddle</b> — the range starts in one allocation and ends in another. The draw
     *       stitches two unrelated sections' quads into one mesh: geometry shaped nothing like the
     *       section it is drawn as. Cannot be reached by any shading fault.</li>
     *   <li><b>overlap</b> — two live sections were given overlapping allocations. Then one section's
     *       metadata points at memory holding another's quads, and which one the buffer holds changes
     *       as uploads land. This is the allocator-level version of the reported symptom, and the one
     *       a per-command check is otherwise structurally blind to.</li>
     *   <li><b>sidOob / noGroup / posMismatch</b> — need the render list, so they are only counted
     *       while {@code listStale == 0} and are meaningless otherwise.</li>
     * </ul>
     */
    private void validateDrawCommands(final String tag, MDICViewport viewport,
                                      long indirectOffset, int maxDrawCount) {
        if (DRAWCHK == 0 || maxDrawCount <= 0) return;
        if (!(viewport.drawCallBuffer instanceof me.cortex.voxy.client.core.metal.MetalBuffer cmds)) return;
        if (!(viewport.positionScratchBuffer instanceof me.cortex.voxy.client.core.metal.MetalBuffer pos)) return;
        if (!(viewport.indirectLookupBuffer instanceof me.cortex.voxy.client.core.metal.MetalBuffer list)) return;
        if (!(this.geometryManager.getMetadataBuffer() instanceof me.cortex.voxy.client.core.metal.MetalBuffer md)) return;
        if (!(this.geometryManager.getGeometryBuffer() instanceof me.cortex.voxy.client.core.metal.MetalBuffer geo)) return;
        long cp = cmds.getContentsPtr();
        long pp = pos.getContentsPtr();
        long lp = list.getContentsPtr();
        long mp = md.getContentsPtr();
        final long gpp = geo.getContentsPtr();
        final long modelPtr = this.modelStore.getModelBuffer()
                instanceof me.cortex.voxy.client.core.metal.MetalBuffer mmb ? mmb.getContentsPtr() : 0;
        if (cp == 0 || pp == 0 || lp == 0 || mp == 0) return;

        final long posEntries = pos.size() / 8L;//uvec2 per entry
        final long heapElements = geo.size() / 8L;//8 bytes per quad
        final int maxSections = this.geometryManager.getMaxSectionCount();
        long avail = Math.max(0, (cmds.size() - indirectOffset) / 20L);
        int n = (int) Math.min(maxDrawCount, avail);

        // Rebuild the allocation table periodically rather than every call, and also whenever the live
        // section count moves. The count gate matters for correctness, not freshness: the ownership
        // check below asks "which section owns this address", and a table built before a section was
        // freed and its address reused answers with the PREVIOUS owner -- a false positive that would
        // be worse than no reading at all. Skipped entirely unless opted into -- see DRAWCHK_TABLE.
        final int sectionCount = this.geometryManager.getSectionCount();
        if (DRAWCHK_TABLE) {
            if (drawchkTableAge >= DRAWCHK_TABLE_EVERY || sectionCount != drawchkTableSectionCount) {
                drawchkTableAge = 0;
                rebuildAllocationTable(mp, maxSections, heapElements);
                drawchkTableSectionCount = sectionCount;
            }
            drawchkTableAge += 3;// opaque + temporal + translucent per frame
        }
        final long[] table = DRAWCHK_TABLE ? drawchkTable : null;
        final int tableSize = DRAWCHK_TABLE ? drawchkTableSize : 0;

        // Is the CPU's render list the frame cmdgen actually ran on? prep.comp derives the same
        // dispatch size from the same uint, so a count that cannot produce this dispatch is stale.
        // NOTE the pointer must stay a long: an earlier version of this line narrowed it with
        // `(int) dcb.getContentsPtr()`, which truncates a 64-bit address to 32 bits and sends
        // memGetInt to an unmapped page -- a SIGSEGV in the render thread, not a wrong reading.
        final long listCount = Integer.toUnsignedLong(MemoryUtil.memGetInt(lp));
        long dispatchPtr = 0;
        if (viewport.drawCountCallBuffer instanceof me.cortex.voxy.client.core.metal.MetalBuffer dcb) {
            dispatchPtr = dcb.getContentsPtr();
        }
        final int cmdGenDispatchX = dispatchPtr == 0 ? 0 : MemoryUtil.memGetInt(dispatchPtr);
        final boolean listFresh = listCount > 0 && cmdGenDispatchX > 0
                && listCount <= (long) cmdGenDispatchX * 128L;

        int d0 = 0, d1 = 0, d2 = 0, d3 = 0, d4 = 0, d5 = 0, d6 = 0, d7 = listFresh ? 0 : 1;

        for (int i = 0; i < n; i++) {
            if (DRAWCHK == 1 && DRAWCHK_STRIDE > 1 && (i % DRAWCHK_STRIDE) != 0) continue;
            long a = cp + indirectOffset + (long) i * 20L;
            int indexCount = MemoryUtil.memGetInt(a);
            int baseVertex = MemoryUtil.memGetInt(a + 12);
            long baseInstance = Integer.toUnsignedLong(MemoryUtil.memGetInt(a + 16));
            d0++;
            if (indexCount <= 0) continue;

            // The range this draw reads, in geometry-heap elements (one element is one quad).
            final long qs = Integer.toUnsignedLong(baseVertex) >> 2;
            final long qe = qs + indexCount / 6L;

            final int slot = (table == null || tableSize == 0) ? -1 : findContaining(table, tableSize, qs);
            if (table != null && slot < 0) {
                d1++;
                pfOrphan++;
                drawchkExample(tag, i, "orphan", "baseVertex>>2=" + qs + " quads=" + (indexCount / 6L)
                        + " end=" + qe + " table=" + tableSize + " heapElems=" + heapElements);
                continue;
            }
            if (table != null) {
                final long allocEnd = table[slot * 3 + 1];
                if (qe > allocEnd) {
                    d2++;
                    pfStraddle++;
                    drawchkExample(tag, i, "straddle", "baseVertex>>2=" + qs + " quads=" + (indexCount / 6L)
                            + " end=" + qe + " allocEnd=" + allocEnd + " sid=" + table[slot * 3 + 2]);
                    continue;
                }

                // THE OWNERSHIP CHECK -- "is this geometry in the right place?"
                //
                // Each section's quads live in one contiguous allocation, so the allocation table
                // answers "which section owns the quads at baseVertex" without the render list. The
                // vertex shader places that geometry using positionBuffer[baseInstance], which cmdgen
                // wrote from the metadata of whatever section it built the command for. So if the
                // owning section and the position the draw uses disagree, the draw is rendering one
                // section's quads at another section's position -- trees with their trunks in open
                // sky, kelp over dry ground, water at the wrong height, terrain slices detached.
                //
                // This is the one form of the fault that a per-command check CAN see, and it is
                // deliberately independent of the render list: both sides of the comparison come from
                // buffers written by the same pass, so a stale CPU read of a GPU-written list cannot
                // manufacture a mismatch the way it did for posMismatch.
                final long ownerSid = table[slot * 3 + 2];
                // SELF-VALIDATION, which is what makes this check trustworthy rather than merely
                // suggestive. The table can be stale: a section replaced since it was built leaves
                // the live COUNT unchanged (a replace frees one id and allocates another), so no
                // count-based gate catches it, and a stale entry would name the previous owner --
                // a false positive that would look exactly like the bug.
                //
                // But a stale entry is detectable for free: if the section the table names no longer
                // owns that address, its CURRENT metadata will not claim it either. So compare the
                // table's recorded start against the section's live quadStart and skip the entry when
                // they disagree. A removed section's record is zeroed, so that case is covered too.
                // The cost is one read per checked command, against three million-entry scans a frame
                // for the alternative (rebuilding every call).
                final long ownerStart = ownerSid < maxSections
                        ? Integer.toUnsignedLong(MemoryUtil.memGetInt(mp + ownerSid * 32L + 12L)) : -1;
                if (ownerStart != table[slot * 3]) {
                    pfStaleEntry++;
                    continue;
                }
                if (ownerSid < maxSections && baseInstance < posEntries) {
                    final long om = mp + ownerSid * 32L;
                    final int ownerHi = MemoryUtil.memGetInt(om);
                    final int ownerLo = MemoryUtil.memGetInt(om + 4);
                    final int useHi = MemoryUtil.memGetInt(pp + baseInstance * 8L);
                    final int useLo = MemoryUtil.memGetInt(pp + baseInstance * 8L + 4);
                    if (ownerHi != useHi || ownerLo != useLo) {
                        pfWrongSection++;
                        if (pfWrongExamples++ < 8) {
                        }
                    } else {
                        pfWrongChecked++;
                    }
                }
            }

            // The quad the vertex shader will actually read for this command (baseVertex>>2 is the
            // first quad's index). Two properties of it decide whether the quad can draw as a flat
            // black rectangle: a fully dark light byte, and a model id past the 65536-entry model
            // table, which reads out of bounds and yields a garbage face -- and therefore a garbage
            // UV, a garbage tint and no texture. Both are cheap to decode here (quad_format.glsl:
            // face = q0&7, light = (q1>>>23)&0xFF with sky low, modelId = ((q0>>>26)&0x3F)|((q1&0x3FFF)<<6)).
            if (gpp != 0 && qs * 8L + 8L <= geo.size()) {
                final long qa = gpp + qs * 8L;
                final int q0 = MemoryUtil.memGetInt(qa);
                final int q1 = MemoryUtil.memGetInt(qa + 4);
                final int light = (q1 >>> 23) & 0xFF;
                final int modelId = ((q0 >>> 26) & 0x3F) | ((q1 & 0x3FFF) << 6);
                if (light == 0) pfLightZero++;
                // A zero quad inside a drawn range can never be legitimate: the mesher writes real
                // quads, and cmdgen only emits a command for a group whose count is non-zero. So a
                // zero quad here means the geometry at this range is not what the section uploaded --
                // either the upload has not landed yet or the range belongs to someone else. This is
                // the one signature that separates "wrong metadata" from "right metadata, wrong
                // geometry", and nothing so far has tested the latter.
                if (q0 == 0 && q1 == 0) pfEmptyQuad++;
                // The model this quad points at: is it actually populated? `ModelFactory` allocates
                // `modelId = modelTexture2id.size()` and maps the state to it, but the model is only
                // written into this buffer later, after an async GPU bake and readback. A quad that
                // carries an id allocated but not yet uploaded reads a zeroed BlockModel, whose
                // `faceData[face]` of 0 means a zero UV rect, tint 0, no cutout and no indentation --
                // a flat, textureless, unlit-looking rectangle the size of the LOD cell. Nothing is
                // corrupt here: every index and every byte the other checks look at is valid.
                if (modelPtr != 0) {
                    final long modelAt = modelPtr + (long) modelId * 64L;
                    final int face = q0 & 0x7;
                    int allZero = 0;
                    for (int f = 0; f < 6; f++) {
                        allZero |= MemoryUtil.memGetInt(modelAt + f * 4L);
                    }
                    if (allZero == 0) {
                        pfModelUnbaked++;
                        if (pfExamples++ < 6) {
                            drawchkExample(tag, i, "modelUnbaked", "modelId=" + modelId
                                    + " face=" + face + " light=" + light);
                        }
                    } else if (face < 6 && MemoryUtil.memGetInt(modelAt + face * 4L) == 0) {
                        pfFaceZero++;
                    }
                }
                if (modelId >= 65536) {
                    pfModelOob++;
                    if (pfExamples++ < 6) {
                        drawchkExample(tag, i, "modelOob", "modelId=" + modelId + " light=" + light
                                + " quad=[" + Integer.toUnsignedString(q0) + ","
                                + Integer.toUnsignedString(q1) + "] quads=" + (indexCount / 6L));
                    }
                }
                // Detail lives in the top nibble of the positionBuffer entry the vertex shader reads.
                final long rawPosHi = Integer.toUnsignedLong(MemoryUtil.memGetInt(pp + baseInstance * 8L));
                if ((rawPosHi >>> 28) >= 3) pfCoarse++;
                pfSampled++;
            }

            if (!listFresh || baseInstance >= posEntries) continue;
            long sid = Integer.toUnsignedLong(MemoryUtil.memGetInt(lp + 4 + baseInstance * 4L));
            if (sid >= maxSections) { d4++; continue; }
            long m = mp + sid * 32L;
            int metaPosHi = MemoryUtil.memGetInt(m);
            int metaPosLo = MemoryUtil.memGetInt(m + 4);
            if (metaPosHi != MemoryUtil.memGetInt(pp + baseInstance * 8L)
                    || metaPosLo != MemoryUtil.memGetInt(pp + baseInstance * 8L + 4)) {
                d6++;
                continue;
            }
            long quadStart = Integer.toUnsignedLong(MemoryUtil.memGetInt(m + 12));
            long[] cnt = unpackQuadGroups(
                    Integer.toUnsignedLong(MemoryUtil.memGetInt(m + 16)),
                    Integer.toUnsignedLong(MemoryUtil.memGetInt(m + 20)),
                    Integer.toUnsignedLong(MemoryUtil.memGetInt(m + 24)),
                    Integer.toUnsignedLong(MemoryUtil.memGetInt(m + 28)));
            if (!matchesQuadGroup(quadStart, cnt, qs, indexCount / 6L)) d5++;
        }

        DRAWCHK_TOT[0] += d0; DRAWCHK_TOT[1] += d1; DRAWCHK_TOT[2] += d2;
        DRAWCHK_TOT[4] += d4; DRAWCHK_TOT[5] += d5; DRAWCHK_TOT[6] += d6;
        DRAWCHK_TOT[7] = d7;

        if ((drawchkCalls++ % 600) == 1) {
            StringBuilder sb = new StringBuilder();
            for (int k = 0; k < DRAWCHK_NAMES.length; k++) {
                sb.append(' ').append(DRAWCHK_NAMES[k]).append('=').append(DRAWCHK_TOT[k])
                  .append("(+").append(DRAWCHK_TOT[k] - DRAWCHK_LAST[k]).append(')');
                DRAWCHK_LAST[k] = DRAWCHK_TOT[k];
            }
        }
    }

    /**
     * Rebuild the live-allocation table from the metadata buffer, and count overlaps while it is
     * sorted.
     *
     * <p>An entry is live if it claims any geometry at all. A freed slot is written as 32 zero bytes
     * by {@code BasicAsyncGeometryManager.writeMetadata}, so {@code quadStart == 0 && no counts} is
     * the empty marker — with the one exception of a genuinely live section allocated at address 0,
     * which is kept by testing the counts as well.
     *
     * <p>Sorting by start makes the overlap scan a single linear pass over neighbours: the arena
     * hands out 1024-aligned, non-overlapping ranges, so any two live entries whose ranges intersect
     * means the allocator gave the same memory to two sections.
     */
    private static void rebuildAllocationTable(final long mp, final int maxSections, final long heapElements) {
        long[] scratch = new long[Math.max(1024, maxSections / 4) * 3];
        int count = 0;
        for (int sid = 0; sid < maxSections; sid++) {
            long m = mp + (long) sid * 32L;
            long quadStart = Integer.toUnsignedLong(MemoryUtil.memGetInt(m + 12));
            long[] cnt = unpackQuadGroups(
                    Integer.toUnsignedLong(MemoryUtil.memGetInt(m + 16)),
                    Integer.toUnsignedLong(MemoryUtil.memGetInt(m + 20)),
                    Integer.toUnsignedLong(MemoryUtil.memGetInt(m + 24)),
                    Integer.toUnsignedLong(MemoryUtil.memGetInt(m + 28)));
            long len = quadEnd(0L, cnt);
            if (len == 0) continue;
            if (quadStart + len > heapElements) continue;// not a usable allocation; orphan catches it
            if (count * 3 + 3 > scratch.length) {
                long[] bigger = new long[scratch.length * 2];
                System.arraycopy(scratch, 0, bigger, 0, scratch.length);
                scratch = bigger;
            }
            scratch[count * 3] = quadStart;
            scratch[count * 3 + 1] = quadStart + ((len + 1023L) & ~1023L);
            scratch[count * 3 + 2] = sid;
            count++;
        }
        // Sort the triples by start (insertion-free: sort an index-free copy via a simple merge on
        // the flat array, which is small enough that a boxed sort would dominate the scan).
        sortAllocations(scratch, count);

        long overlaps = countOverlaps(scratch, count);
        DRAWCHK_TOT[3] = overlaps;
        drawchkTable = scratch;
        drawchkTableSize = count;
    }

    /**
     * Count pairs of adjacent (post-sort) live allocations that intersect. The arena hands out
     * 1024-aligned, non-overlapping ranges, so a non-zero result means it gave the same memory to
     * two sections — and whichever section's geometry was uploaded last is what both draws read.
     */
    static long countOverlaps(final long[] table, final int count) {
        long overlaps = 0;
        for (int i = 1; i < count; i++) {
            if (table[i * 3] < table[(i - 1) * 3 + 1]) {
                overlaps++;
                if (overlaps <= 4) {
                    Logger.warn("[Metal-DRAWCHK! alloc] OVERLAP sid=" + table[(i - 1) * 3 + 2]
                            + " [" + table[(i - 1) * 3] + "," + table[(i - 1) * 3 + 1] + ") and sid="
                            + table[i * 3 + 2] + " [" + table[i * 3] + "," + table[i * 3 + 1] + ")");
                }
            }
        }
        return overlaps;
    }

    /** In-place merge sort of flat {@code [start, end, sid]} triples by start. */
    private static void sortAllocations(final long[] a, final int count) {
        if (count < 2) return;
        long[] tmp = new long[count * 3];
        for (int width = 1; width < count; width *= 2) {
            for (int lo = 0; lo < count; lo += 2 * width) {
                final int mid = Math.min(lo + width, count);
                final int hi = Math.min(lo + 2 * width, count);
                int l = lo, r = mid, o = lo;
                while (l < mid && r < hi) {
                    final int pick = a[l * 3] <= a[r * 3] ? l++ : r++;
                    tmp[o * 3] = a[pick * 3]; tmp[o * 3 + 1] = a[pick * 3 + 1]; tmp[o * 3 + 2] = a[pick * 3 + 2];
                    o++;
                }
                while (l < mid) { tmp[o * 3] = a[l * 3]; tmp[o * 3 + 1] = a[l * 3 + 1]; tmp[o * 3 + 2] = a[l * 3 + 2]; l++; o++; }
                while (r < hi) { tmp[o * 3] = a[r * 3]; tmp[o * 3 + 1] = a[r * 3 + 1]; tmp[o * 3 + 2] = a[r * 3 + 2]; r++; o++; }
            }
            System.arraycopy(tmp, 0, a, 0, count * 3);
        }
    }

    /**
     * Index of the allocation containing element {@code q}, or {@code -1}. Binary search over the
     * sorted table; {@code -1} means the quad belongs to no live section.
     */
    static int findContaining(final long[] table, final int tableSize, final long q) {
        int lo = 0, hi = tableSize - 1, best = -1;
        while (lo <= hi) {
            final int mid = (lo + hi) >>> 1;
            if (table[mid * 3] <= q) { best = mid; lo = mid + 1; } else { hi = mid - 1; }
        }
        if (best < 0) return -1;
        return q < table[best * 3 + 1] ? best : -1;
    }

    /** Rate-limited detail for the first few offenders, so the counters come with a concrete case. */
    private static void drawchkExample(String tag, int i, String kind, String detail) {
        if (drawchkExamples++ >= 8) return;
        Logger.warn("[Metal-DRAWCHK! " + tag + "] cmd#" + i + " " + kind + " " + detail);
    }

    /**
     * A section's raw position as {@code "detail L<x,y,z>"} -- the top nibble is the LOD level and the
     * rest decodes exactly as {@code extractLoDPosition} does in quad_util.glsl. Printed instead of hex
     * so a WRONGSEC line can be read without hand-decoding the packing, which is error-prone enough
     * that doing it by hand is how a same-detail mismatch and a level mismatch get confused.
     */
    private static String pprintRawPos(final int hi, final int lo) {
        final int d = hi >>> 28;
        final int y = (hi >> 20) & 0xFF;
        final int x = (lo >> 4) & 0xFFFFFF;
        final int z = (((hi & 0xFFFFF) << 4) | (lo >>> 28)) & 0xFFFFFF;
        // Sign-extend x and z from 24 bits, y from 8, matching bitfieldExtract on a signed int.
        final int sx = (x << 8) >> 8;
        final int sz = (z << 8) >> 8;
        final int sy = (byte) y;
        return "d" + d + " L" + sx + "," + sy + "," + sz;
    }

    /**
     * Unpack a section's 32-byte metadata into the eight quad-group counts, in the order
     * {@code cmdgen.comp} walks them: translucent, double-sided, down, up, north, south, west, east.
     * Mirrors {@code BasicAsyncGeometryManager.SectionMeta.writeMetadataSplitParts}, which packs
     * {@code (offsets[b+1]-offsets[b])} into the low half of each uint and the next delta into the
     * high half, with the final group's high half being {@code itemCount - offsets[7]}.
     *
     * @param c0 first packed uint (metadata byte 16), through {@code c3} (byte 28)
     */
    static long[] unpackQuadGroups(final long c0, final long c1, final long c2, final long c3) {
        return new long[] {c0 & 0xFFFFL, c0 >>> 16, c1 & 0xFFFFL, c1 >>> 16,
                           c2 & 0xFFFFL, c2 >>> 16, c3 & 0xFFFFL, c3 >>> 16};
    }

    /**
     * One past the last quad index this section's metadata claims, i.e. where the geometry heap
     * allocation must extend to. {@code quadStart} is the metadata's {@code geometryPtr +
     * offsets[0]}, so this is {@code geometryPtr + itemCount}.
     *
     * <p>This is the number the existing {@code verifyBuiltSectionOffsets} guard does NOT cover: it
     * checks the deltas <i>between</i> offsets, but the final segment runs to {@code itemCount},
     * which {@code BuiltSection} does not know (it is the geometry buffer's element count). The last
     * group's count is packed as {@code itemCount - offsets[7]}, so if that difference is ever
     * negative it wraps to a count near 65535 and the draw reads far past its own allocation.
     */
    static long quadEnd(final long quadStart, final long[] groups) {
        long end = quadStart;
        for (long c : groups) end += c;
        return end;
    }

    /**
     * Is {@code (baseVertex>>2, indexCount/6)} one of the eight {@code (ptr, count)} groups this
     * section's metadata encodes? A draw command is only legitimate if it is — this is the check
     * that a command's quad range lies inside its own section's allocation, since the groups tile
     * exactly {@code [quadStart, quadEnd)}.
     *
     * <p>Empty groups are skipped rather than matched, because {@code cmdgen.comp} emits nothing for
     * them ({@code if (count != 0)}): a command claiming a zero-quad group is as wrong as one
     * claiming a range that does not exist.
     */
    static boolean matchesQuadGroup(final long quadStart, final long[] groups,
                                    final long baseVertexQuads, final long quads) {
        long ptr = quadStart;
        for (int g = 0; g < 8; g++) {
            if (groups[g] != 0 && baseVertexQuads == ptr && quads == groups[g]) return true;
            ptr += groups[g];
        }
        return false;
    }

    /**
     * The depth state both LOD passes must use — the compare OPERATOR is a property of the frame's
     * depth buffer, not of the pass, so opaque and translucent have to agree on it.
     *
     * <p>Extracted so it can be pinned by a unit test rather than only by running the game. This is
     * the decision that was wrong: the frame's depth buffer is reverse-Z, and comparing LessEqual
     * against it rejects every fragment -- over sky the buffer holds 0 (far) while a Voxy fragment's
     * depth is a large positive number, so {@code fragDepth <= 0} is false everywhere. The LOD drew
     * and contributed nothing, which no draw counter or command dump can show.
     *
     * <p>The opaque pass was fixed first and the translucent one missed, which inverted the water
     * pass in both directions at once: it drew in front of terrain it should have been hidden by,
     * and was hidden by terrain it should have drawn in front of. Both passes now come through here.
     *
     * @param reverseZFrame true when the render target's depth buffer is reverse-Z (Metallum's is)
     * @param depthDisabled true for {@code VOXY_LOD_NO_DEPTH=1} / {@code VOXY_LOD_WATER_DEBUG=1},
     *                      which force the test off entirely
     * @param writeEnabled  true for the opaque pass; false for translucent water, which must test
     *                      against the opaque depth but not write, or it occludes itself
     */
    static me.cortex.voxy.client.core.gpu.PipelineState.DepthState lodDepthState(
            final boolean reverseZFrame, final boolean depthDisabled, final boolean writeEnabled) {
        if (depthDisabled) {
            return me.cortex.voxy.client.core.gpu.PipelineState.DepthState.DISABLED;
        }
        if (reverseZFrame) {
            return new me.cortex.voxy.client.core.gpu.PipelineState.DepthState(
                    true, writeEnabled, me.cortex.voxy.client.core.gpu.PipelineState.CompareOp.GREATER_EQUAL);
        }
        return writeEnabled
                ? me.cortex.voxy.client.core.gpu.PipelineState.DepthState.DEFAULT
                : me.cortex.voxy.client.core.gpu.PipelineState.DepthState.TEST_NO_WRITE;
    }

    /** Opt-in encoder test ({@code VOXY_LOD_TRIANGLE=1}); see {@link #drawDebugTriangle}. */
    private static final String DEBUG_TRIANGLE_MODE = System.getenv("VOXY_LOD_TRIANGLE");
    /** Voxy's own createGraphicsPipeline pipeline, drawn through Voxy's (detached) encoder. */
    private static final boolean DEBUG_TRIANGLE =
            "1".equals(DEBUG_TRIANGLE_MODE) || "both".equals(DEBUG_TRIANGLE_MODE);
    /** The P0 probe's MSL pipeline, drawn through an encoder made directly from the command buffer. */
    private static final boolean DEBUG_TRIANGLE_PROBE =
            "probe".equals(DEBUG_TRIANGLE_MODE) || "both".equals(DEBUG_TRIANGLE_MODE);
    /**
     * Draw the debug triangle through the TERRAIN pipeline instead of the debug one. Combined
     * with VOXY_LOD_FORCE_VERTEX=1 (which makes the vertex shader emit a fixed clip-space
     * triangle, needing no inputs) this separates two explanations that look identical:
     * the terrain pipeline itself is broken, versus its bindings/vertex data being broken.
     * Draws => the pipeline and this encoder are fine and the fault is the LOD's inputs.
     */
    private static final boolean DEBUG_TRIANGLE_TERRAIN = "terrain".equals(DEBUG_TRIANGLE_MODE);
    private static int probeTriLogged;
    private static int terrainTriLogged;
    private me.cortex.voxy.client.core.gpu.IGpuPipeline debugTrianglePipeline;

    private static final String DEBUG_TRI_VERT = """
            #version 430 core
            void main() {
                // Triangle covering the lower-left half of clip space, generated from
                // gl_VertexID so it needs no vertex buffers and no uniforms at all.
                vec2 p = vec2(float(gl_VertexID & 1) * 1.7 - 0.9, float(gl_VertexID >> 1) * 1.7 - 0.9);
                gl_Position = vec4(p, 0.5, 1.0);
            }
            """;
    private static final String DEBUG_TRI_FRAG = """
            #version 430 core
            layout(location = 0) out vec4 outColour;
            void main() { outColour = vec4(1.0, 0.0, 1.0, 1.0); }
            """;

    /**
     * Encoder-sequencing test: draw a magenta triangle through the SAME encoder, at the SAME point
     * in the frame, that Voxy's LOD pass just used -- i.e. after the HOT traversal, the five cmdgen
     * compute prepasses, the mid-frame {@code backend.submit()} (which commits the command buffer
     * and nulls Metallum's encoder), the chunk-bound render pass and the depth-mask blit.
     *
     * <p>The P0 probe already proved a foreign renderer can draw into this frame, but it draws at
     * Sodium's CUTOUT tail, before any of that Voxy-specific machinery runs. This test draws after
     * all of it, so it separates two explanations that look identical from every other diagnostic:
     *
     * <ul>
     *   <li><b>Magenta appears</b> -- the encoder state at Voxy's draw point accepts draws, so the
     *       sequencing is sound and the fault is Voxy's own pipeline/draw state (depth convention,
     *       pipeline object, or the LOD's own commands).</li>
     *   <li><b>No magenta</b> -- draws at that point do not reach the framebuffer at all, whatever
     *       the pipeline is, and the cause is the sequencing itself.</li>
     * </ul>
     *
     * <p>Depth testing is DISABLED for the triangle on purpose: the point is to test whether the
     * encoder accepts a draw, so nothing about depth conventions may be allowed to reject it and
     * muddy the result. Drawn AFTER the LOD so the LOD cannot cover it.
     */
    private void drawDebugTriangle(me.cortex.voxy.client.core.gpu.RenderEncoder encoder, MDICViewport viewport) {
        // VOXY_LOD_TRIANGLE=probe -- draw the same magenta triangle at the same point, but via the
        // P0 probe's own path: end whatever encoder is open, make a fresh one straight from the
        // command buffer, draw, end it. That bypasses Metallum's renderCommandEncoderForHandles
        // bookkeeping, which is the one structural difference between the probe (works) and Voxy's
        // LOD pass (draws nothing). If this appears where VOXY_LOD_TRIANGLE=1 did not, the fault is
        // in that bookkeeping rather than in the moment in the frame.
        if (DEBUG_TRIANGLE_PROBE) {
            long colorHandle = me.cortex.voxy.client.core.metal.MetallumBridge.colorAttachment();
            long depthHandle = me.cortex.voxy.client.core.metal.MetallumBridge.depthAttachment();
            boolean drew = me.cortex.voxy.client.core.metal.MetallumBridge.drawProbeStyleTriangle(
                    colorHandle, depthHandle, viewport.width, viewport.height, "voxy-lod-pass");
            if (probeTriLogged++ == 0) {
            }
            if (!DEBUG_TRIANGLE) {
                return;   // "both": fall through and also draw with Voxy's own pipeline
            }
        }
        if (DEBUG_TRIANGLE_TERRAIN) {
            if (this.terrainPipeline == null) return;
            encoder.setPipeline(this.terrainPipeline);
            encoder.draw(me.cortex.voxy.client.core.gpu.RenderEncoder.PRIMITIVE_TRIANGLES, 0, 3, 1, 0);
            if (terrainTriLogged++ == 0) {
            }
            return;
        }
        if (!DEBUG_TRIANGLE) {
            return;
        }
        if (this.debugTrianglePipeline == null) {
            this.debugTrianglePipeline = this.backend.createGraphicsPipeline(
                    new me.cortex.voxy.client.core.gpu.GraphicsPipelineDesc(
                            DEBUG_TRI_VERT, DEBUG_TRI_FRAG, java.util.Map.of(),
                            null, null, null, null,
                            GL_RGBA8,
                            me.cortex.voxy.client.core.gpu.VertexLayout.EMPTY,
                            // PipelineState.DEFAULT == (DepthState.DISABLED, BlendState.OPAQUE,
                            // RasterState.NO_CULL) -- depth test and write both off, which is what
                            // this test needs so no depth convention can reject the triangle.
                            me.cortex.voxy.client.core.gpu.PipelineState.DEFAULT,
                            "VoxyDebugTriangle"));
        }
        encoder.setPipeline(this.debugTrianglePipeline);
        encoder.draw(me.cortex.voxy.client.core.gpu.RenderEncoder.PRIMITIVE_TRIANGLES, 0, 3, 1, 0);
    }

    /** Opt-in geometry dump ({@code VOXY_GEOM_TRACE=1}). */
    private static final boolean GEOM_TRACE = "1".equals(System.getenv("VOXY_GEOM_TRACE"));
    private static long geomTraceCount = 0;

    /**
     * Dump the two per-quad inputs the vertex shader positions geometry with:
     * {@code positionBuffer[baseInstance]} (the section/quad base point) and {@code quadData[...]}.
     *
     * <p>Every other layer has been cleared by bisection -- encoder, pipeline, rasterizer, target,
     * viewport, bindings, indirect draw and the scene uniform's VP all measured sound, and the LOD
     * draws only when the vertex shader is made to ignore THESE. So if the base points are garbage
     * or point somewhere unrelated to the camera, every quad lands off-screen: no fragments, no
     * error, and every counter still healthy.
     */
    private void traceGeometry(MDICViewport viewport, long indirectOffset, int maxDrawCount,
                               me.cortex.voxy.client.core.metal.MetalBuffer geo) {
        if (!GEOM_TRACE || (geomTraceCount++ % 600) != 1) return;
        if (!(viewport.drawCallBuffer instanceof me.cortex.voxy.client.core.metal.MetalBuffer cmds)) return;
        if (!(viewport.positionScratchBuffer instanceof me.cortex.voxy.client.core.metal.MetalBuffer pos)) return;
        long cp = cmds.getContentsPtr();
        long pp = pos.getContentsPtr();
        if (cp == 0 || pp == 0) return;
        StringBuilder sb = new StringBuilder();
        int n = Math.min(Math.max(maxDrawCount, 0), 4);
        for (int i = 0; i < n; i++) {
            long a = cp + indirectOffset + (long) i * 20L;
            int baseInstance = org.lwjgl.system.MemoryUtil.memGetInt(a + 16);
            int indexCount = org.lwjgl.system.MemoryUtil.memGetInt(a);
            // positionBuffer is uvec2 per entry.
            long pa = pp + (long) baseInstance * 8L;
            long u0 = org.lwjgl.system.MemoryUtil.memGetInt(pa) & 0xFFFFFFFFL;
            long u1 = org.lwjgl.system.MemoryUtil.memGetInt(pa + 4) & 0xFFFFFFFFL;
            // The quad metadata the vertex shader actually reads: quadData[gl_VertexID>>2],
            // where gl_VertexID includes baseVertex. Decoded per quad_format.glsl:
            // face = bit 3, size = bits 4..6 (+1), modelId = bits 6..31 | (bits 14..45 << 6).
            int baseVertex = org.lwjgl.system.MemoryUtil.memGetInt(a + 12);
            long quadIdx = Integer.toUnsignedLong(baseVertex) >> 2;
            long q0 = -1, q1 = -1;
            int face = -1, sizeX = -1, modelId = -1;
            if (geo != null) {
                long gp = geo.getContentsPtr();
                if (gp != 0) {
                    long qa = gp + quadIdx * 8L;
                    q0 = org.lwjgl.system.MemoryUtil.memGetInt(qa) & 0xFFFFFFFFL;
                    q1 = org.lwjgl.system.MemoryUtil.memGetInt(qa + 4) & 0xFFFFFFFFL;
                    face   = (int) ((q0 >>> 3) & 1L);
                    sizeX  = (int) ((q0 >>> 4) & 7L) + 1;
                    modelId = (int) (((q0 >>> 6) & 0x3FFFFFFL) | (((q0 >>> 14) | (q1 << 18)) << 6));
                }
            }
            sb.append(String.format(java.util.Locale.ROOT,
                    " [#%d idx=%d bi=%d sPos=[%d,%d] baseVtx=%d quadIdx=%d quad=[%d,%d] face=%d sizeX=%d modelId=%d]",
                    i, indexCount, baseInstance, u0, u1, baseVertex, quadIdx, q0, q1, face, sizeX, modelId));
        }
        traceCorners(viewport, indirectOffset, Math.min(Math.max(maxDrawCount, 0), 4));
    }

    /**
     * The decisive half of the geometry trace: run the SAME transform the vertex shader runs
     * ({@link me.cortex.voxy.client.core.rendering.util.LodVertexMath}) over the same inputs and
     * report where the four corners land in clip space.
     *
     * <p>This separates the two remaining explanations, which every existing diagnostic renders
     * identically: the quads are geometrically fine and something downstream eats them (in which
     * case the corners are inside the clip volume with non-zero area), or the quads have always
     * been landing outside it or collapsed (in which case no pipeline state could ever have made
     * them visible, and the fault is in the matrix or the decoders).
     */
    private void traceCorners(MDICViewport viewport, long indirectOffset, int n) {
        if (n <= 0) return;
        if (!(viewport.drawCallBuffer instanceof me.cortex.voxy.client.core.metal.MetalBuffer cmds)) return;
        if (!(viewport.positionScratchBuffer instanceof me.cortex.voxy.client.core.metal.MetalBuffer pos)) return;
        if (!(this.geometryManager.getGeometryBuffer()
                instanceof me.cortex.voxy.client.core.metal.MetalBuffer geo)) return;
        if (!(this.modelStore.getModelBuffer()
                instanceof me.cortex.voxy.client.core.metal.MetalBuffer models)) return;
        long cp = cmds.getContentsPtr();
        long pp = pos.getContentsPtr();
        long gp = geo.getContentsPtr();
        long mp = models.getContentsPtr();
        if (cp == 0 || pp == 0 || gp == 0 || mp == 0) return;

        float[] mvp = new float[16];
        this.lodMvp(viewport).get(mvp);
        int[] baseSectionPos = {viewport.section.x, viewport.section.y, viewport.section.z};

        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < n; i++) {
            long a = cp + indirectOffset + (long) i * 20L;
            int baseInstance = org.lwjgl.system.MemoryUtil.memGetInt(a + 16);
            int baseVertex = org.lwjgl.system.MemoryUtil.memGetInt(a + 12);
            long pa = pp + (long) baseInstance * 8L;
            int sPosX = org.lwjgl.system.MemoryUtil.memGetInt(pa);
            int sPosY = org.lwjgl.system.MemoryUtil.memGetInt(pa + 4);
            long qa = gp + (Integer.toUnsignedLong(baseVertex) >> 2) * 8L;
            int qx = org.lwjgl.system.MemoryUtil.memGetInt(qa);
            int qy = org.lwjgl.system.MemoryUtil.memGetInt(qa + 4);

            int modelId = me.cortex.voxy.client.core.rendering.util.LodVertexMath.stateId(qx, qy);
            int face = me.cortex.voxy.client.core.rendering.util.LodVertexMath.face(qx, qy);
            // BlockModel = { uint faceData[6]; ... } at MODEL_SIZE bytes per entry.
            int faceData = (modelId < (1 << 16) && face < 6)
                    ? org.lwjgl.system.MemoryUtil.memGetInt(mp + (long) modelId * 64L + face * 4L)
                    : -1;

            float[] c = me.cortex.voxy.client.core.rendering.util.LodVertexMath.corners(
                    mvp, baseSectionPos, sPosX, sPosY, qx, qy, faceData, true);
            int inside = 0;
            float minW = Float.MAX_VALUE, maxZ = -Float.MAX_VALUE, minZ = Float.MAX_VALUE;
            float minNdcX = Float.MAX_VALUE, maxNdcX = -Float.MAX_VALUE;
            float minNdcY = Float.MAX_VALUE, maxNdcY = -Float.MAX_VALUE;
            for (int k = 0; k < 4; k++) {
                float x = c[k * 4], y = c[k * 4 + 1], z = c[k * 4 + 2], w = c[k * 4 + 3];
                if (me.cortex.voxy.client.core.rendering.util.LodVertexMath
                        .insideClipVolume(x, y, z, w)) inside++;
                minW = Math.min(minW, w);
                float iw = w == 0 ? 1 : 1 / w;
                minZ = Math.min(minZ, z * iw);
                maxZ = Math.max(maxZ, z * iw);
                minNdcX = Math.min(minNdcX, x * iw);
                maxNdcX = Math.max(maxNdcX, x * iw);
                minNdcY = Math.min(minNdcY, y * iw);
                maxNdcY = Math.max(maxNdcY, y * iw);
            }
            // NDC y is reported as the shader produces it (y up); the viewport has a negative
            // height, so it reaches the screen flipped.
            sb.append(String.format(java.util.Locale.ROOT,
                    " [#%d model=%d face=%d faceData=%d inside=%d/4 minW=%.3f ndcZ=[%.3f,%.3f] "
                            + "ndcX=[%.3f,%.3f] ndcY=[%.3f,%.3f]]",
                    i, modelId, face, faceData, inside, minW, minZ, maxZ,
                    minNdcX, maxNdcX, minNdcY, maxNdcY));
        }
    }

    private void renderTerrainMetal(me.cortex.voxy.client.core.gpu.RenderEncoder encoder,
                                    me.cortex.voxy.client.core.gpu.IGpuPipeline pipeline,
                                    MDICViewport viewport, long indirectOffset, int maxDrawCount) {
        traceCommands("off=" + indirectOffset, viewport, indirectOffset, maxDrawCount);
        var gb = this.geometryManager.getGeometryBuffer();
        traceGeometry(viewport, indirectOffset, maxDrawCount,
                gb instanceof me.cortex.voxy.client.core.metal.MetalBuffer mb ? mb : null);
        validateDrawCommands(
                indirectOffset == 0L ? "opaque"
                        : indirectOffset == (long) TEMPORAL_OFFSET * 5L * 4L ? "temporal" : "translucent",
                viewport, indirectOffset, maxDrawCount);
        encoder.setPipeline(pipeline);
        // SSBO bindings 0..5; SceneUniform is an SSBO, not a UBO.
        encoder.setBuffer(0, this.uniformFor(viewport), 0);
        encoder.setBuffer(1, this.geometryManager.getGeometryBuffer(), 0);
        encoder.setBuffer(2, this.geometryManager.getMetadataBuffer(), 0);
        // M13 chunk 1: bindBuffers now also wires up the model atlas texture +
        // cross-backend sampler at binding 0 (blockModelAtlas in quads.frag).
        this.modelStore.bindBuffers(encoder, 3, 4, 0);
        // CONSUME slot, NOT the current frame -- this is the coupling that would otherwise re-create
        // bug 3. The vertex shader indexes this buffer by `baseInstance`, and that value comes from the
        // command in the indirect buffer bound at the draw below, which is the consume slot. Give the
        // shader frame N's positions and frame N-1's index and every draw is placed at another
        // section's origin, keeping its own quads and baked light -- the exact symptom just fixed.
        encoder.setBuffer(5, viewport.positionScratch(viewport.frameId - 1), 0);
        // Texture / sampler binding 1 — MC's 16×16 RGBA8 lightmap, mirrored
        // into a Shared-storage Metal texture each frame (M13 chunk 2).
        LightMapHelper.bindMetal(encoder, 1);
        // Texture/sampler binding 2 and buffer binding 9 used to carry the chunk-bound depth mask here.
        // Both are gone, along with the mask's rasterizer and the Viewport depth buffer it wrote, and
        // they were carried for nothing on this path:
        //   - nothing ever wrote that depth buffer on Metal, so the mask was never rasterized;
        //   - quads.frag's depth-bound test is compiled out anyway, because VOXY_NO_DEPTH_BOUND is
        //     injected unless the env var is exactly "0" (see the terrain defines above);
        //   - and the binding-9 comment already conceded "the shader no longer samples it" -- a
        //     depth-format texture reads zeros through a texture2d<float> declaration, which is the
        //     same usage-flag wall that blocks the Hi-Z pyramid from reading MC's depth directly.
        // So this was a per-frame bind of an unwritten, unsampled texture plus an SSBO nothing read.
        // Kept as a note rather than silently deleted: if the mask is ever revived on Metal it needs
        // all three of those facts revisited, not just the bind restored.
        // Buffer binding 10 — the built-section mask again, for the per-chunk-column cull. The same
        // upload cmdgen read this frame; the compute pass that maintains it runs before the draws.
        // Bound for all three draws (this method is shared), because the cull has to apply to the
        // translucent surface too or water is culled over vanilla water and nowhere else.
        if (CHUNK_CULL && this.builtSectionMask.buffer() != null) {
            encoder.setBuffer(BUILT_MASK_CHUNK_BINDING, this.builtSectionMask.buffer(), 0);
        }
        // VOXY_BI_OFFSET=1: hand the encoder the buffer its per-draw reads come from, so it can bind it
        // at `baseInstance * 8` instead of pushing the index as a constant.
        if ("1".equals(System.getenv("VOXY_BI_OFFSET"))
                && encoder instanceof me.cortex.voxy.client.core.metal.MetalRenderEncoder mre) {
            // CONSUME slot, matching the draw's indirect buffer below: the encoder derives the index
            // from a command in THAT buffer, so it has to index the same frame's positions.
            mre.setPerDrawIndexBuffer(5, viewport.positionScratch(viewport.frameId - 1));
        }

        encoder.bindIndexBuffer(me.cortex.voxy.client.core.rendering.util.SharedIndexBuffer.INSTANCE.getBuffer(),
                me.cortex.voxy.client.core.gpu.RenderEncoder.INDEX_TYPE_UINT16, 0);
        encoder.drawIndexedIndirect(
                me.cortex.voxy.client.core.gpu.RenderEncoder.PRIMITIVE_TRIANGLES,
                viewport.drawCallConsume(viewport.frameId), indirectOffset,
                maxDrawCount,
                /*stride*/ 5 * 4); // DrawElementsIndirectCommand = 5 uint32
    }

    @Override
    public void renderTranslucent(MDICViewport viewport) {
        // GL draw path deleted — see renderOpaque above; renderTranslucentMetal is the live one.
    }

    private static boolean COMPUTE_SERIALIZE_LOGGED = false;

    /** Fallback: re-enable the per-frame drawCallBuffer zero on Metal. */
    private static final boolean METAL_ZERO_DRAWBUF =
            "1".equals(System.getenv("VOXY_LOD_ZERO_DRAWBUF"));

    @Override
    public void buildDrawCalls(MDICViewport viewport) {
        if (this.geometryManager.getSectionCount() == 0) return;
        this.uploadUniformBuffer(viewport);

        // DIAGNOSTIC (2026-05-25): VOXY_COMPUTE_SERIALIZE=1 forces submit()+wait
        // after each Metal compute prepass (prep, cull, commandGen) so they run
        // strictly serially with memory coherency — the guarantee the GL path
        // gets from glMemoryBarrier. If the LOD flicker stops with this on, the
        // prepasses were racing across encoder boundaries and the real fix is
        // inter-encoder fences/barriers. Metal-only; GL unaffected.
        boolean computeSerialize = "1".equals(System.getenv("VOXY_COMPUTE_SERIALIZE"));
        if (computeSerialize && !COMPUTE_SERIALIZE_LOGGED) {
            COMPUTE_SERIALIZE_LOGGED = true;
            Logger.info("[Metal-SERIALIZE] VOXY_COMPUTE_SERIALIZE active: submit()+wait after each compute prepass");
        }

        // On non-GL backends the renderer issues `drawIndexedIndirect` against
        // viewport.drawCallBuffer with a CPU-side `maxDrawCount` upper bound
        // (Metal has no count-aware MDI without an ICB, and MDIC's terrain
        // pipeline opts out of ICB — see gotcha #16). The render*Metal calls
        // clamp that bound to the GPU-written counts (see metalDrawCount), and
        // all three slices are compact, so stale slots beyond the counts are
        // never drawn — the previous per-frame ~12 MB zero of the whole
        // cmdBuffer is unnecessary (it's zeroed once at allocation in
        // MDICViewport). VOXY_LOD_ZERO_DRAWBUF=1 restores it as a fallback.
        if (METAL_ZERO_DRAWBUF) {
            viewport.drawCallWrite(viewport.frameId).zero();
        }

        //Can do a sneeky trick, since the sectionRenderList is a list to things to render, it invokes the culler
        // which only marks visible sections


        {//Dispatch prep
            // M12 chunk 2: prep prepass migrated to ComputeEncoder. prep.comp
            // does not reference SceneUniform fields (only writes to the
            // DrawCommandCountBuffer at binding 1 from sectionCount at
            // binding 2), so the encoder skips binding 0 entirely — that
            // keeps the SceneUniform UBO→SSBO decision deferred to chunks
            // 3 and 4 (which DO use SceneUniform).
            try (var encoder = this.backend.beginComputePass()) {
                encoder.setPipeline(this.prepPipeline);
                encoder.setBuffer(1, viewport.drawCountWrite(viewport.frameId), 0);
                encoder.setBuffer(2, viewport.getRenderList(), 0);
                encoder.barrier(ComputeEncoder.BARRIER_SHADER, ComputeEncoder.BARRIER_SHADER);
                encoder.dispatch(1, 1, 1);
                encoder.barrier(ComputeEncoder.BARRIER_SHADER, ComputeEncoder.BARRIER_SHADER);
            }
        }
        if (computeSerialize) this.backend.submit(); // serialize: prep complete

        {//Test occlusion
            // Per-section Hi-Z cull (lod/gl46/section_cull.comp), which replaced the M12 chunk 5
            // force-all-visible stub. It walks the same list the stub walked — `indirectLookup`, the
            // traversal's own render queue — but decides each section's visibilityData with the same
            // Hi-Z predicate the traversal used, instead of marking everything visible. (The GL arm
            // that rasterized each section's AABB against MC's depth buffer, with optional
            // NV_representative_fragment_test, is deleted along with lod/gl46/cull/raster.vert|frag
            // and the cullPipeline they compiled.)
            //
            // FAIL OPEN when the pyramid has no texture: there is nothing to test against, and
            // dispatching anyway would sample an unbound texture, i.e. cull on undefined values.
            // Skipping the pass leaves visibilityData stale, so cmdgen emits NO sections — visibly
            // wrong, and deliberately so, because a silent wrong answer is what this tree keeps
            // paying for. It is a guard, not a live path: AbstractRenderPipeline always allocates the
            // pyramid for the LOD pass before it calls buildDrawCalls.
            final var hizTexture = viewport.hiZBuffer.getHizTexture();
            if (SECTION_CULL && hizTexture != null) {
                try (var encoder = this.backend.beginComputePass()) {
                    encoder.setPipeline(this.sectionCullPipeline);
                    encoder.setBuffer(0, this.uniformFor(viewport), 0);
                    // The section metadata: the pass decodes each section's box from it, with the
                    // vertex path's decoders (quad_util.glsl) — see the shader.
                    encoder.setBuffer(1, this.geometryManager.getMetadataBuffer(), 0);
                    encoder.setBuffer(2, viewport.visibilityBuffer, 0);
                    encoder.setBuffer(3, viewport.indirectLookupBuffer, 0);
                    // Texture/sampler are a DIFFERENT namespace from buffers, so slot 0 here does
                    // not collide with the scene uniform at buffer 0 — the same slot the traversal
                    // binds the pyramid to. One sampler, shared from HiZBuffer, so a second copy of
                    // a filtering mode cannot start disagreeing with the pyramid's texels.
                    encoder.setTexture(0, hizTexture);
                    encoder.setSampler(0, viewport.hiZBuffer.getSampler());
                    encoder.barrier(ComputeEncoder.BARRIER_SHADER | ComputeEncoder.BARRIER_INDIRECT,
                                    ComputeEncoder.BARRIER_SHADER | ComputeEncoder.BARRIER_INDIRECT);
                    // Reuses prep's dispatch sizing — cmdGenDispatchX/Y/Z at
                    // offset 0 of drawCountCallBuffer holds ceil(sectionCount/128),
                    // matching this shader's local_size_x=128.
                    encoder.dispatchIndirect(viewport.drawCountWrite(viewport.frameId), 0);
                    encoder.barrier(ComputeEncoder.BARRIER_SHADER, ComputeEncoder.BARRIER_SHADER);
                }
            }
        }
        if (computeSerialize) this.backend.submit(); // serialize: cull/visibility complete before commandGen


        {//Generate the commands
            this.distanceCountBuffer.zeroRange(0, 1024*4);
            // M12 chunk 3: commandGen migrated to ComputeEncoder. SceneUniform
            // is now an SSBO (see bindings.glsl), so binding 0 flows through
            // setBuffer just like the other SSBOs. Read count comes from
            // drawCountCallBuffer at offset 0 via dispatchIndirect.
            if (RenderStatistics.enabled) {
                this.statisticsBuffer.zero();
            }
            try (var encoder = this.backend.beginComputePass()) {
                // Back to gating update() itself, not just the bind. This had been made
                // unconditional so the CPU-side set stayed inspectable in both arms of the
                // VOXY_LOD_BUILT_MASK A/B, and that was a real behaviour change on the arm where
                // the cull is OFF: update() allocates a buffer and does an UploadStream upload,
                // and it runs from inside an open compute encoder. With the cull off it did none
                // of that before. One run on that arm hung during load having emitted no render
                // diagnostics at all — a single sample, cause unproven, but the mask-off arm is
                // the only path this touched, so it goes back to doing nothing there rather than
                // being left as an unexplained behaviour change for a diagnostic that is moot.
                // Maintained here because the compute encoder is already open, but READ by
                // quads.frag now, not by cmdgen: the cull moved to a per-16x16x16-section test in
                // the fragment stage, so cmdgen.comp no longer declares the buffer and the
                // BUILT_MASK_BINDING bind that used to follow this is gone with it.
                if (CULL_ENABLED) {
                    this.builtSectionMask.update(viewport, this.backend);
                }
                encoder.setPipeline(this.commandGenPipeline);
                encoder.setBuffer(0, this.uniformFor(viewport), 0);
                encoder.setBuffer(1, viewport.drawCallWrite(viewport.frameId), 0);
                encoder.setBuffer(2, viewport.drawCountWrite(viewport.frameId), 0);
                encoder.setBuffer(3, this.geometryManager.getMetadataBuffer(), 0);
                encoder.setBuffer(4, viewport.visibilityBuffer, 0);
                encoder.setBuffer(5, viewport.indirectLookupBuffer, 0);
                encoder.setBuffer(6, viewport.positionScratch(viewport.frameId), 0);
                encoder.setBuffer(7, this.distanceCountBuffer, 0);
                if (RenderStatistics.enabled) {
                    encoder.setBuffer(STATISTICS_BUFFER_BINDING, this.statisticsBuffer, 0);
                }
                encoder.barrier(ComputeEncoder.BARRIER_SHADER, ComputeEncoder.BARRIER_SHADER);
                encoder.dispatchIndirect(viewport.drawCountWrite(viewport.frameId), 0);
                encoder.barrier(ComputeEncoder.BARRIER_SHADER | ComputeEncoder.BARRIER_INDIRECT,
                                ComputeEncoder.BARRIER_SHADER | ComputeEncoder.BARRIER_INDIRECT);
            }

            if (RenderStatistics.enabled) {
                DownloadStream.INSTANCE.download(this.statisticsBuffer, down->{
                    final int LAYERS = WorldEngine.MAX_LOD_LAYER+1;
                    for (int i = 0; i < LAYERS; i++) {
                        RenderStatistics.visibleSections[i] = MemoryUtil.memGetInt(down.address+i*4L);
                    }

                    for (int i = 0; i < LAYERS; i++) {
                        RenderStatistics.quadCount[i] = MemoryUtil.memGetInt(down.address+LAYERS*4L+i*4L);
                    }
                });
            }
        }
        if (computeSerialize) this.backend.submit(); // serialize: commandGen (draw commands) complete

        {//Do translucency sorting
            // M12 chunk 1: prefixSum dispatched through ComputeEncoder (Metal
            // opens an MTLComputeCommandEncoder).
            try (var encoder = this.backend.beginComputePass()) {
                encoder.setPipeline(this.prefixSumPipeline);
                encoder.setBuffer(0, this.distanceCountBuffer, 0);
                encoder.barrier(ComputeEncoder.BARRIER_SHADER, ComputeEncoder.BARRIER_SHADER);
                encoder.dispatch(1, 1, 1);
                encoder.barrier(ComputeEncoder.BARRIER_SHADER, ComputeEncoder.BARRIER_SHADER);
            }

            // M12 chunk 4: translucentGen migrated to ComputeEncoder. SceneUniform
            // is bound as an SSBO at binding 0 (post-flip in chunk 3); the
            // dispatch count comes from drawCountCallBuffer at offset 0 via
            // dispatchIndirect — the same buffer doubles as the indirect arg
            // and as one of the shader's SSBO inputs at binding 2 (an unusual
            // but pre-existing read-then-dispatch pattern).
            try (var encoder = this.backend.beginComputePass()) {
                encoder.setPipeline(this.translucentGenPipeline);
                encoder.setBuffer(0, this.uniformFor(viewport), 0);
                encoder.setBuffer(1, viewport.drawCallWrite(viewport.frameId), 0);
                encoder.setBuffer(2, viewport.drawCountWrite(viewport.frameId), 0);
                encoder.setBuffer(3, this.geometryManager.getMetadataBuffer(), 0);
                encoder.setBuffer(4, viewport.indirectLookupBuffer, 0);
                encoder.setBuffer(5, this.distanceCountBuffer, 0);
                encoder.barrier(ComputeEncoder.BARRIER_SHADER | ComputeEncoder.BARRIER_INDIRECT,
                                ComputeEncoder.BARRIER_SHADER | ComputeEncoder.BARRIER_INDIRECT);
                encoder.dispatchIndirect(viewport.drawCountWrite(viewport.frameId), 0);
                encoder.barrier(ComputeEncoder.BARRIER_SHADER | ComputeEncoder.BARRIER_INDIRECT,
                                ComputeEncoder.BARRIER_SHADER | ComputeEncoder.BARRIER_INDIRECT);
            }
        }

    }

    @Override
    public void renderTemporal(MDICViewport viewport) {
        // GL draw path deleted — see renderOpaque above; renderTemporalMetal is the live one.
    }

    @Override
    public void addDebug(List<String> lines) {
        super.addDebug(lines);
        //lines.add("SC/GS: " + this.geometryManager.getSectionCount() + "/" + (this.geometryManager.getGeometryUsed()/(1024*1024)));//section count/geometry size (MB)
    }

    @Override
    public MDICViewport createViewport() {
        return new MDICViewport(this.geometryManager.getMaxSectionCount());
    }

    @Override
    public void free() {
        for (IGpuBuffer u : this.uniformRing) {
            u.free();
        }
        this.distanceCountBuffer.free();
        if (this.translucentTerrainPipeline != null) this.translucentTerrainPipeline.close();
        if (this.terrainPipeline != null) this.terrainPipeline.close();
        this.commandGenPipeline.close();
        this.sectionCullPipeline.close();
        this.prepPipeline.close();
        this.translucentGenPipeline.close();
        this.prefixSumPipeline.close();
        this.statisticsBuffer.free();
    }
}
