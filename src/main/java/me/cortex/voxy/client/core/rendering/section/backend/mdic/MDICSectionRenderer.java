package me.cortex.voxy.client.core.rendering.section.backend.mdic;


import me.cortex.voxy.client.RenderStatistics;
import me.cortex.voxy.client.VoxyClient;
import me.cortex.voxy.client.core.AbstractRenderPipeline;
import me.cortex.voxy.client.core.gl.Capabilities;
import me.cortex.voxy.client.core.gpu.BackendType;
import me.cortex.voxy.client.core.gpu.ComputeEncoder;
import me.cortex.voxy.client.core.gpu.IGpuBuffer;
import me.cortex.voxy.client.core.gpu.RenderBackendFactory;
import me.cortex.voxy.client.core.gl.shader.Shader;
import me.cortex.voxy.client.core.gl.shader.ShaderLoader;
import me.cortex.voxy.client.core.gl.shader.ShaderType;
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

import static org.lwjgl.opengl.ARBIndirectParameters.GL_PARAMETER_BUFFER_ARB;
import static org.lwjgl.opengl.ARBIndirectParameters.glMultiDrawElementsIndirectCountARB;
import static org.lwjgl.opengl.GL11.*;
import static org.lwjgl.opengl.GL11C.GL_RGBA8;
import static org.lwjgl.opengl.GL11C.GL_TEXTURE_2D;
import static org.lwjgl.opengl.GL15C.GL_ELEMENT_ARRAY_BUFFER;
import static org.lwjgl.opengl.GL15C.glBindBuffer;
import static org.lwjgl.opengl.GL15C.glGetBufferSubData;
import static org.lwjgl.opengl.GL30.glBindBufferBase;
import static org.lwjgl.opengl.GL30.glBindVertexArray;
import static org.lwjgl.opengl.GL31C.GL_COPY_READ_BUFFER;
import static org.lwjgl.opengl.GL31.GL_UNIFORM_BUFFER;
import static org.lwjgl.opengl.GL33.glBindSampler;
import static org.lwjgl.opengl.GL40C.GL_DRAW_INDIRECT_BUFFER;
import static org.lwjgl.opengl.GL42.glMemoryBarrier;
import static org.lwjgl.opengl.GL43.*;
import static org.lwjgl.opengl.GL43.glMultiDrawElementsIndirect;
import static me.cortex.voxy.client.core.gl.GLCompat.bindTextureUnit;
import static org.lwjgl.opengl.NVRepresentativeFragmentTest.GL_REPRESENTATIVE_FRAGMENT_TEST_NV;

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
     * Terrain shaders. Two paths:
     *   - Iris-patched (legacy {@link Shader.Builder}, GL-only by definition since
     *     the Iris pipeline is GL-gated in RenderPipelineFactory).
     *   - Unpatched ({@link me.cortex.voxy.client.core.gpu.IGpuPipeline} via
     *     createGraphicsPipeline). On Metal this is the only path that ever
     *     runs; on GL it's used when no Iris pack is active.
     * Exactly one of each pair is non-null per opaque/translucent slot.
     */
    private final Shader terrainShader;
    private final Shader translucentTerrainShader;
    private final me.cortex.voxy.client.core.gpu.IGpuPipeline terrainPipeline;
    private final me.cortex.voxy.client.core.gpu.IGpuPipeline translucentTerrainPipeline;
    private final int terrainProgram;
    private final int translucentTerrainProgram;
    /**
     * NEAREST/CLAMP sampler for the chunk-bound depth mask at texture
     * binding 2 (quads.frag's {@code depthTex}; pattern: HiZBuffer's blit
     * sampler). Metal-only — the GL path binds the raw texture unit in
     * {@link #bindRenderingBuffers}; null on OpenGL.
     */
    private final me.cortex.voxy.client.core.gpu.IGpuSampler boundDepthSampler;

    // M9 migration: MDIC's 5 non-Iris-patched shaders (4 compute + 1 graphics)
    // now flow through RenderBackend.create*Pipeline so they compile cleanly on
    // Metal/Vulkan. terrainShader + translucentTerrainShader stay on the legacy
    // Shader.Builder path because they thread Iris's patchOpaqueShader /
    // patchTranslucentShader callbacks; that path is GL-only after the
    // RenderPipelineFactory gate (commit 1e2a1190). Bind/draw stays raw GL —
    // MDIC operates inside AbstractRenderPipeline's FBO context, not a
    // RenderEncoder.

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

    private final me.cortex.voxy.client.core.gpu.IGpuPipeline cullPipeline = this.backend.createGraphicsPipeline(
            new me.cortex.voxy.client.core.gpu.GraphicsPipelineDesc(
                    ShaderLoader.parse("voxy:lod/gl46/cull/raster.vert"),
                    ShaderLoader.parse("voxy:lod/gl46/cull/raster.frag"),
                    java.util.Map.of(),
                    null, null, null, null,
                    GL_RGBA8,
                    me.cortex.voxy.client.core.gpu.VertexLayout.EMPTY,
                    me.cortex.voxy.client.core.gpu.PipelineState.DEFAULT,
                    "MDICSectionRenderer.cull"));
    private final int cullProgram = mdicProgramId(this.cullPipeline);

    /**
     * M12 chunk 5 Metal stub: substitutes for the depth-test-based cull pass
     * on backends that can't currently open a depth-only render pass against
     * MC's depth buffer. Writes `visibilityData[sid] = frameId | (1<<31)` for
     * every section in `indirectLookup`, so cmdgen queues all frustum-visible
     * sections for rendering (slower than real depth occlusion but
     * functionally correct). Allocated unconditionally — only dispatched
     * when the backend isn't OpenGL. Negligible memory cost; the alternative
     * (gating allocation behind a backend check) makes the class harder to
     * read for no real benefit.
     */
    private final me.cortex.voxy.client.core.gpu.IGpuPipeline forceAllVisiblePipeline = this.backend.createComputePipeline(
            new me.cortex.voxy.client.core.gpu.ComputePipelineDesc(
                    ShaderLoader.parse("voxy:lod/gl46/force_all_visible.comp"),
                    java.util.Map.of(),
                    null, null,
                    128, 1, 1,
                    "MDICSectionRenderer.forceAllVisible"));

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

    private static int mdicProgramId(me.cortex.voxy.client.core.gpu.IGpuPipeline p) {
        if (p instanceof me.cortex.voxy.client.core.gl.GlGraphicsPipeline gg) return gg.program();
        if (p instanceof me.cortex.voxy.client.core.gl.GlComputePipeline gc) return gc.program();
        return 0;
    }

    private final IGpuBuffer uniform = RenderBackendFactory.get().createBuffer(1024).zero();//TODO move to viewport?

    // Far-water alpha ramp (2026-07-03, Metal translucent shader only —
    // see the VOXY_WATER_FAR_ALPHA injection + quads.frag). Target alpha at
    // the far end of the ramp; 0 disables. Ramp distances default to a
    // render-distance-relative window (uploadUniformBuffer) unless the
    // START/END envs pin them in blocks.
    private static final float WATER_FAR_ALPHA = parseEnvFloat("VOXY_WATER_FAR_ALPHA", 0.95f);
    private static final float WATER_FAR_ALPHA_START = parseEnvFloat("VOXY_WATER_FAR_ALPHA_START", 0.0f);
    private static final float WATER_FAR_ALPHA_END = parseEnvFloat("VOXY_WATER_FAR_ALPHA_END", 0.0f);

    // Near-cull metric (2026-07-03 round 3). XZ mode compares the horizontal
    // Chebyshev distance max(|dx|,|dz|) against the threshold — the metric MC
    // renders chunks in — instead of the 3D slant distance, which from a high
    // camera / toward the square's diagonals let LOD water survive INSIDE the
    // MC ring and double-composite with BSL/Sodium water (the flickering pale
    // squares). VOXY_TRANS_NEAR_CULL_XZ=0 restores the slant metric. The
    // margin shrinks from 48 to 16 in XZ mode because Chebyshev matches the
    // loaded-chunk square exactly (48 only papered over the slant mismatch);
    // VOXY_TRANS_NEAR_CULL_MARGIN overrides in blocks.
    private static final boolean TRANS_NEAR_CULL_XZ =
            !"0".equals(System.getenv("VOXY_TRANS_NEAR_CULL_XZ"));
    // 2026-07-03 round 5: Sodium renders sections in a Euclidean XZ CYLINDER
    // (OcclusionCuller fx*fx+fz*fz <= r*r), so the Chebyshev SQUARE cull left
    // a ring toward the render square's diagonals (Euclid RD..RD*sqrt(2))
    // with NEITHER MC water NOR LOD water — the naked kelp/seafloor band the
    // colortex16 clear fix exposed. Radial matches Sodium's real coverage and
    // turns the diagonal gap into the same ~margin-wide overlap ring the axes
    // already have (handled by the chunk-bound mask).
    // VOXY_TRANS_NEAR_CULL_RADIAL=0 falls back to the Chebyshev square.
    private static final boolean TRANS_NEAR_CULL_RADIAL =
            TRANS_NEAR_CULL_XZ && !"0".equals(System.getenv("VOXY_TRANS_NEAR_CULL_RADIAL"));
    private static final float TRANS_NEAR_CULL_MARGIN =
            parseEnvFloat("VOXY_TRANS_NEAR_CULL_MARGIN", TRANS_NEAR_CULL_XZ ? 16f : 48f);

    private static float parseEnvFloat(String name, float def) {
        String v = System.getenv(name);
        if (v == null || v.isBlank()) return def;
        try {
            return Float.parseFloat(v.trim());
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

        // M13 chunk 3: sampler for the bound-depth texture at binding 2 —
        // quads.frag only texelFetches it, but the MSL signature still wants
        // a sampler bound alongside the texture.
        this.boundDepthSampler = this.backend.getType() != BackendType.OPENGL
                ? this.backend.createSampler(me.cortex.voxy.client.core.gpu.SamplerDesc.builder()
                        .filter(me.cortex.voxy.client.core.gpu.SamplerDesc.Filter.NEAREST,
                                me.cortex.voxy.client.core.gpu.SamplerDesc.Filter.NEAREST)
                        .mipFilter(me.cortex.voxy.client.core.gpu.SamplerDesc.MipFilter.NEAREST)
                        .wrap(me.cortex.voxy.client.core.gpu.SamplerDesc.Wrap.CLAMP_TO_EDGE,
                                me.cortex.voxy.client.core.gpu.SamplerDesc.Wrap.CLAMP_TO_EDGE)
                        .label("MDIC.boundDepthSampler")
                        .build())
                : null;

        String vertex = ShaderLoader.parse("voxy:lod/gl46/quads3.vert");
        String taa = pipeline.taaFunction("taaShift");
        if (taa != null) {
            vertex += "\n"+taa;//inject it at the end
        }
        var builder = Shader.make()
                .defineIf("TAA_PATCH", taa != null)
                .defineIf("DEBUG_RENDER", false)

                //.defineIf("USE_NV_BARRY", Capabilities.INSTANCE.nvBarryCoords)

                .addSource(ShaderType.VERTEX, vertex);

        //Apply per face tinting
        addDirectionalFaceTint(builder, Minecraft.getInstance().level);

        String frag = ShaderLoader.parse("voxy:lod/gl46/quads.frag");

        String opaqueFrag = pipeline.patchOpaqueShader(this, frag);
        boolean opaquePatched = opaqueFrag != null;
        if (!opaquePatched) opaqueFrag = frag;

        String translucentFrag = pipeline.patchTranslucentShader(this, frag);
        boolean translucentPatched = translucentFrag != null;
        if (!translucentPatched) translucentFrag = frag;

        if (opaquePatched || translucentPatched) {
            // Iris-patched path stays on the legacy Shader.Builder. It's GL-only
            // because the Iris pipeline itself is now gated to OpenGL in
            // RenderPipelineFactory (commit 1e2a1190).
            this.terrainShader = tryCompilePatchedOrNormal(builder, opaqueFrag, frag);
            this.translucentTerrainShader = tryCompilePatchedOrNormal(
                    builder.define("TRANSLUCENT"), translucentFrag, frag);
            this.terrainPipeline = null;
            this.translucentTerrainPipeline = null;
            this.terrainProgram = 0;
            this.translucentTerrainProgram = 0;
        } else {
            // Unpatched path — runs on every backend including Metal. Build the
            // two pipelines via the cross-backend abstraction. Defines mirror
            // what Shader.Builder collected above (face-tint floats from
            // addDirectionalFaceTint + TAA_PATCH if a TAA function exists).
            this.terrainShader = null;
            this.translucentTerrainShader = null;
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
            if (this.backend.getType() != BackendType.OPENGL) {
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
                // VOXY_NO_DEPTH_BOUND=0 restores the old mask path for an A/B.
                if (!"0".equals(System.getenv("VOXY_NO_DEPTH_BOUND"))) {
                    opaqueDefines.put("VOXY_NO_DEPTH_BOUND", "");
                    translucentDefines.put("VOXY_NO_DEPTH_BOUND", "");
                }
                // With the mask gone, vanilla and the LOD are compared purely by depth -- and since
                // the LOD approximates the surface vanilla draws, their depths agree to float
                // precision where they overlap and they z-fight. Bias the LOD behind so vanilla
                // always wins ties. See quads3.vert's VOXY_LOD_DEPTH_BIAS for the sign, which is a
                // property of the reverse-Z frame rather than a free choice.
                String lodBias = System.getenv("VOXY_LOD_DEPTH_BIAS");
                if (lodBias == null || lodBias.isBlank()) lodBias = "1e-5";
                opaqueDefines.put("VOXY_LOD_DEPTH_BIAS", lodBias + "f");
                translucentDefines.put("VOXY_LOD_DEPTH_BIAS", lodBias + "f");
                boolean noDepthBound = true;
                boolean boundDebug = false;
                opaqueDefines.put("VOXY_FORCE_OPAQUE_ALPHA", "");
                // VOXY_LOD_FORCE_MAGENTA=1 -- bisection switch (see quads.frag). Solid magenta
                // emitted before every discard/early-out, so the frame shows whether LOD geometry
                // rasterizes at all. Diagnostic only.
                if ("1".equals(System.getenv("VOXY_LOD_FORCE_MAGENTA"))) {
                    opaqueDefines.put("VOXY_LOD_FORCE_MAGENTA", "");
                    translucentDefines.put("VOXY_LOD_FORCE_MAGENTA", "");
                    Logger.info("[Metal-LODTEST] VOXY_LOD_FORCE_MAGENTA active: LOD emits solid magenta before any discard");
                }
                // Vertex-stage bisection (see quads3.vert). Defines reach both stages.
                if ("1".equals(System.getenv("VOXY_LOD_FORCE_VERTEX"))) {
                    opaqueDefines.put("VOXY_LOD_FORCE_VERTEX", "");
                    translucentDefines.put("VOXY_LOD_FORCE_VERTEX", "");
                    Logger.info("[Metal-LODTEST] VOXY_LOD_FORCE_VERTEX active: LOD emits a fixed clip-space triangle");
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
                    Logger.info("[Metal-LODTEST] VOXY_LOD_SHOW_LIGHT active: LOD emits raw light as colour (R=sky, G=block, B=128 iff a fragment was drawn)");
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
                    Logger.info("[Metal-LODTEST] fixedMip=" + lodFixedMip + " noDiscard=" + lodNoDiscard);
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
                    Logger.info("[Metal-LODTEST] distance-based atlas mip ON (maxLod=" + maxLod
                            + ", bias=" + biasStr + "); VOXY_LOD_DIST_MIP=0 reverts to fixed mip 0");
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
                    Logger.info("[Metal-LODTEST] far-water alpha ramp ON (target=" + WATER_FAR_ALPHA
                            + "); VOXY_WATER_FAR_ALPHA=0 disables");
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
                    Logger.info("[Metal-LODTEST] absolute face indent ON (water plane height "
                            + "lodScale-invariant); VOXY_LOD_ABS_INDENT=0 reverts");
                }
                // VOXY_TRANS_NEAR_CULL — vx contract only (2026-07-03),
                //   DEFAULT ON. BSL composites the injected LOD water AND
                //   draws MC's own water inside render distance; LOD water
                //   that survives the chunk-bound mask there (the depth
                //   compare flips with camera pitch at grazing angles)
                //   double-blends into pale veil squares on near/mid water.
                //   Hard-cull translucent LOD fragments inside the MC ring;
                //   the cull distance rides in voxyLodParams2.x per frame.
                //   VOXY_TRANS_NEAR_CULL=0 disables.
                String nearCullEnv = System.getenv("VOXY_TRANS_NEAR_CULL");
                boolean transNearCull = (nearCullEnv == null || !"0".equals(nearCullEnv.trim()))
                        && me.cortex.voxy.client.core.util.IrisUtil.vxContractActive();
                if (transNearCull) {
                    translucentDefines.put("VOXY_TRANS_NEAR_CULL", "");
                    if (TRANS_NEAR_CULL_XZ) {
                        translucentDefines.put("VOXY_TRANS_NEAR_CULL_XZ", "");
                        if (TRANS_NEAR_CULL_RADIAL) {
                            translucentDefines.put("VOXY_TRANS_NEAR_CULL_RADIAL", "");
                        }
                    }
                    Logger.info("[Metal-LODTEST] translucent near-cull ON (vx contract: no LOD water "
                            + "inside MC render distance; metric="
                            + (TRANS_NEAR_CULL_RADIAL ? "xz-radial" : TRANS_NEAR_CULL_XZ ? "xz-chebyshev" : "3d-slant")
                            + ", margin=" + TRANS_NEAR_CULL_MARGIN + "); VOXY_TRANS_NEAR_CULL=0 disables, "
                            + "VOXY_TRANS_NEAR_CULL_RADIAL=0 restores the Chebyshev square, "
                            + "VOXY_TRANS_NEAR_CULL_XZ=0 restores the slant metric");
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
                        Logger.info("[Metal-LODTEST] LOD brightness compensation = " + brightness + " (SSAO parity interim)");
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
                        Logger.info("[Metal-LODTEST] water parity: shade=" + waterShade + " minAlpha=" + waterMinAlpha);
                    }
                }
                // Water diagnostic: paint translucent LOD water solid magenta so
                // a screenshot reveals exactly where water geometry rasterizes.
                if ("1".equals(System.getenv("VOXY_LOD_WATER_DEBUG"))) {
                    translucentDefines.put("VOXY_LOD_WATER_DEBUG", "");
                    Logger.info("[Metal-LODTEST] VOXY_LOD_WATER_DEBUG: translucent LOD water = solid magenta + depth test OFF");
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
                        Logger.info("[Metal-LODTEST] translucent water depth bias = " + waterBias.trim());
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
                    Logger.info("[Metal-LODTEST] VOXY_LOD_FLAT_WATER: translucent LOD water = flat ocean blue (interim path)");
                }

                // M13 diagnostic — log the Metal shader define set ONCE at
                // construction so it's unambiguous in the runtime log
                // which terrain shader variant compiled. Catches "the
                // expected define wasn't injected" bugs that pure source
                // grep can't.
                boolean bakeryOff = "1".equals(System.getenv("VOXY_BAKERY_OFF"));
                boolean debugMissing = "1".equals(System.getenv("VOXY_BAKERY_DEBUG_MISSING"));
                Logger.info("[Metal-DEFINES] terrain shader injections: " +
                        (noDepthBound
                                ? "VOXY_NO_DEPTH_BOUND (depth-bound kill switch)"
                                : "depth-bound ON" + (boundDebug ? " + VOXY_BOUND_DEBUG (red tint)" : "")) +
                        " + VOXY_FORCE_OPAQUE_ALPHA" +
                        (bakeryOff
                                ? " + VOXY_NO_ATLAS (bakery disabled hash-colour fallback)"
                                : (debugMissing ? " + VOXY_DEBUG_MAGENTA_MISSING" : " + atlas bakery")) +
                        (pipeline.useEnvFog() ? " + USE_ENV_FOG" : ""));

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
            // NO_CULL because the GL renderTerrain path explicitly calls
            // glDisable(GL_CULL_FACE) at draw time — that override doesn't
            // apply to Metal where the cull mode is baked into the pipeline.
            // Without this, ~half the LOD triangles disappear due to wrong-
            // winding back-face culling.
            me.cortex.voxy.client.core.gpu.PipelineState opaqueState
                    = me.cortex.voxy.client.core.gpu.PipelineState.OPAQUE_MESH;
            me.cortex.voxy.client.core.gpu.PipelineState translucentState
                    = me.cortex.voxy.client.core.gpu.PipelineState.TRANSLUCENT_MESH;
            if (this.backend.getType() != BackendType.OPENGL) {
                // DIAGNOSTIC (2026-05-25): VOXY_LOD_NO_DEPTH=1 disables the LOD
                // opaque depth test/write to check whether the view-dependent
                // flicker is z-fighting in the LOD's own depth buffer (overlapping
                // LOD geometry competing for depth; winner flips with tiny camera
                // angle changes). If the per-angle disappearing stops, depth/z-fight
                // is confirmed (the image may look unordered with depth off).
                boolean lodNoDepth = "1".equals(System.getenv("VOXY_LOD_NO_DEPTH"));
                if (lodNoDepth) {
                    Logger.info("[Metal-LODTEST] VOXY_LOD_NO_DEPTH active: opaque LOD depth test/write DISABLED");
                }
                boolean reverseZ = MetalMvpUtil.REVERSE_Z_REMAP;
                var opaqueDepth = lodDepthState(reverseZ, lodNoDepth, /*writeEnabled*/ true);
                if (reverseZ && !reverseZLogged) {
                    reverseZLogged = true;
                    Logger.info("[Metal-LODTEST] VOXY_LOD_REVERSE_Z active: LOD depth compare = GreaterEqual");
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
            this.terrainProgram = mdicProgramId(this.terrainPipeline);
            this.translucentTerrainProgram = mdicProgramId(this.translucentTerrainPipeline);
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
        long ptr = UploadStream.INSTANCE.upload(this.uniform, 0, 1024);
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
            Logger.info(String.format(java.util.Locale.ROOT,
                    "[Metal-VP] remap=%s | m00=%.6f m11=%.6f m22=%.6f m23=%.6f m32=%.6f m33=%.6f "
                            + "| t=(%.2f,%.2f,%.2f) | det=%.6e | withoutRemap m00=%.6f m11=%.6f m22=%.6f",
                    MetalMvpUtil.REVERSE_Z_REMAP,
                    mat.m00(), mat.m11(), mat.m22(), mat.m23(), mat.m32(), mat.m33(),
                    mat.m30(), mat.m31(), mat.m32(),
                    mat.determinant(),
                    unremapped.m00(), unremapped.m11(), unremapped.m22()));
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
            float nearCull = 0.0f;
            if (me.cortex.voxy.client.core.util.IrisUtil.vxContractActive()) {
                nearCull = Math.max(rdBlocks - TRANS_NEAR_CULL_MARGIN, 64f);
            }
            MemoryUtil.memPutFloat(lodBase + 16, nearCull);
            MemoryUtil.memPutFloat(lodBase + 20, 0f);
            MemoryUtil.memPutFloat(lodBase + 24, 0f);
            MemoryUtil.memPutFloat(lodBase + 28, 0f);
        }

        UploadStream.INSTANCE.commit();
    }


    private void bindRenderingBuffers(MDICViewport viewport) {
        // SceneUniform is now an SSBO (see bindings.glsl); bind it to the
        // GL_SHADER_STORAGE_BUFFER target so the in-shader binding=0 matches.
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, this.uniform.id());
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, this.geometryManager.getGeometryBuffer().id());
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, this.geometryManager.getMetadataBuffer().id());
        this.modelStore.bind(3, 4, 0);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, viewport.positionScratchBuffer.id());
        LightMapHelper.bind(1);
        bindTextureUnit(2, GL_TEXTURE_2D, viewport.depthBoundingBuffer.getDepthTex().id());

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, SharedIndexBuffer.INSTANCE.id());
        glBindBuffer(GL_DRAW_INDIRECT_BUFFER, viewport.drawCallBuffer.id());
        glBindBuffer(GL_PARAMETER_BUFFER_ARB, viewport.drawCountCallBuffer.id());
    }

    private void renderTerrain(MDICViewport viewport, long indirectOffset, long drawCountOffset, int maxDrawCount) {
        //RenderLayer.getCutoutMipped().startDrawing();


        glDisable(GL_CULL_FACE);
        glEnable(GL_DEPTH_TEST);
        if (this.terrainShader != null) {
            this.terrainShader.bind();
        } else if (this.terrainProgram != 0) {
            org.lwjgl.opengl.GL20C.glUseProgram(this.terrainProgram);
        }
        glBindVertexArray(RenderBackendFactory.get().getStaticVAO());//Needs to be before binding
        this.pipeline.setupAndBindOpaque(viewport);
        this.bindRenderingBuffers(viewport);

        glMemoryBarrier(GL_COMMAND_BARRIER_BIT|GL_SHADER_STORAGE_BARRIER_BIT);//Barrier everything is needed
        glProvokingVertex(GL_FIRST_VERTEX_CONVENTION);

        if (VoxyClient.getOcclusionDebugState()==3) {
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        }
        if (Capabilities.INSTANCE.indirectCount) {
            glMultiDrawElementsIndirectCountARB(GL_TRIANGLES, GL_UNSIGNED_SHORT, indirectOffset, drawCountOffset, maxDrawCount, 0);
        } else {
            int drawCount = Math.min(readDrawCount(viewport.drawCountCallBuffer.id(), drawCountOffset), maxDrawCount);
            glMultiDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_SHORT, indirectOffset, drawCount, 0);
        }
        if (VoxyClient.getOcclusionDebugState()==3) {
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        }

        glEnable(GL_CULL_FACE);
        glBindVertexArray(0);
        glBindSampler(0, 0);
        bindTextureUnit(0, GL_TEXTURE_2D, 0);
        glBindSampler(1, 0);
        bindTextureUnit(1, GL_TEXTURE_2D, 0);

        //RenderLayer.getCutoutMipped().endDrawing();
    }

    @Override
    public void renderOpaque(MDICViewport viewport) {
        if (this.geometryManager.getSectionCount() == 0) return;

        this.uploadUniformBuffer(viewport);

        this.renderTerrain(viewport, 0, 4*3, Math.min((int)(this.geometryManager.getSectionCount()*4.4+128), 400_000));
    }

    /**
     * M12 chunk 6 step 3 — Metal-only opaque draw via {@link RenderEncoder}.
     * Called from {@code AbstractRenderPipeline.runPipelineMetal} inside a
     * render pass that targets the IOSurface bridge color + Voxy's
     * Metal-side depth texture.
     *
     * Differences from the GL {@link #renderTerrain}:
     * <ul>
     *   <li>No raw {@code glUseProgram} / {@code glBindBufferBase} / vertex
     *       array binding — all flows through {@link RenderEncoder.setPipeline}
     *       / {@code setBuffer}.</li>
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
        // SceneUniform was already uploaded by buildDrawCalls this frame
        // (runPipelineMetal always pairs them); no re-upload.
        if (this.terrainPipeline == null) {
            // Iris-patched path — GL-only by construction (see 1e2a1190). Should
            // never hit on Metal because RenderPipelineFactory gates Iris pipeline.
            return;
        }
        int maxDrawCount = Math.min((int)(this.geometryManager.getSectionCount()*4.4+128), 400_000);
        int rawCount = rawOpaqueCount(viewport, OPAQUE_DRAW_COUNT_OFFSET);
        maxDrawCount = metalDrawCount(viewport, OPAQUE_DRAW_COUNT_OFFSET, maxDrawCount);
        lodDrawDiag(this.geometryManager.getSectionCount(), rawCount, maxDrawCount);
        if (maxDrawCount != 0) {
            this.renderTerrainMetal(encoder, this.terrainPipeline, viewport, 0L, maxDrawCount);
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
     * that were visible-this-frame-but-not-last). On GL the equivalent path
     * is {@link #renderTemporal}, which forwards to {@link #renderTerrain}
     * with the temporal offsets.
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
    private static void lodDrawDiag(int sectionCount, int rawOpaque, int maxDrawCount) {
        if ((LOD_DIAG_FRAME++ % 600) != 1) return;
        me.cortex.voxy.common.Logger.info(String.format(
                "[Metal-LODDRAW f=%d] sections=%d  rawOpaqueCount=%d  maxDrawCount=%d  %s",
                LOD_DIAG_FRAME, sectionCount, rawOpaque, maxDrawCount,
                maxDrawCount == 0 ? "<-- NO DRAWS ISSUED" : "drawing"));
    }

    /** Raw (unclamped) value at the opaque draw-count offset, or -1 if unreadable. */
    private static int rawOpaqueCount(MDICViewport viewport, long countOffset) {
        if (viewport.drawCountCallBuffer instanceof me.cortex.voxy.client.core.metal.MetalBuffer mb) {
            long p = mb.getContentsPtr();
            if (p != 0) {
                return MemoryUtil.memGetInt(p + countOffset);
            }
        }
        return -1;
    }

    private static int metalDrawCount(MDICViewport viewport, long countOffset, int upperBound) {
        if (viewport.drawCountCallBuffer instanceof me.cortex.voxy.client.core.metal.MetalBuffer mb) {
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
        me.cortex.voxy.common.Logger.info("[Metal-CMD " + tag + "] maxDrawCount=" + maxDrawCount + sb);
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
                Logger.info("[Metal-LODTEST] VOXY_LOD_TRIANGLE=probe at the LOD pass: drew=" + drew
                        + " color=0x" + Long.toHexString(colorHandle)
                        + " depth=0x" + Long.toHexString(depthHandle));
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
                Logger.info("[Metal-LODTEST] VOXY_LOD_TRIANGLE=terrain: debug triangle drawn through the TERRAIN pipeline");
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
            Logger.info("[Metal-LODTEST] VOXY_LOD_TRIANGLE active: magenta triangle drawn through the LOD encoder, depth test off");
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
        Logger.info("[Metal-GEOM] " + sb);
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
        Logger.info("[Metal-CORNERS] " + sb);
    }

    private void renderTerrainMetal(me.cortex.voxy.client.core.gpu.RenderEncoder encoder,
                                    me.cortex.voxy.client.core.gpu.IGpuPipeline pipeline,
                                    MDICViewport viewport, long indirectOffset, int maxDrawCount) {
        traceCommands("off=" + indirectOffset, viewport, indirectOffset, maxDrawCount);
        var gb = this.geometryManager.getGeometryBuffer();
        traceGeometry(viewport, indirectOffset, maxDrawCount,
                gb instanceof me.cortex.voxy.client.core.metal.MetalBuffer mb ? mb : null);
        encoder.setPipeline(pipeline);
        // SSBO bindings 0..5 — mirror bindRenderingBuffers; SceneUniform is an
        // SSBO post-chunk-3 SceneUniform flip.
        encoder.setBuffer(0, this.uniform, 0);
        encoder.setBuffer(1, this.geometryManager.getGeometryBuffer(), 0);
        encoder.setBuffer(2, this.geometryManager.getMetadataBuffer(), 0);
        // M13 chunk 1: bindBuffers now also wires up the model atlas texture +
        // cross-backend sampler at binding 0 (blockModelAtlas in quads.frag).
        this.modelStore.bindBuffers(encoder, 3, 4, 0);
        encoder.setBuffer(5, viewport.positionScratchBuffer, 0);
        // Texture / sampler binding 1 — MC's 16×16 RGBA8 lightmap, mirrored
        // into a Shared-storage Metal texture each frame (M13 chunk 2).
        LightMapHelper.bindMetal(encoder, 1);
        // Texture / sampler binding 2 — the chunk-bound depth mask
        // (ChunkBoundRenderer.renderMetal rasterized it into
        // viewport.depthBoundingBuffer earlier this frame; same command
        // queue, so ordering is guaranteed). quads.frag's depth-bound test
        // texelFetches it to discard LOD fragments inside MC's loaded-chunk
        // volume (M13 chunk 3). Bound for all three Metal draws (opaque /
        // temporal / translucent share this method). Harmlessly unused when
        // the VOXY_NO_DEPTH_BOUND kill switch removed the sample.
        if (this.boundDepthSampler != null) {
            encoder.setTexture(2, viewport.depthBoundingBuffer.getDepthTex());
            encoder.setSampler(2, this.boundDepthSampler);
        }
        // Buffer binding 9 — the bound mask as raw floats (round 20,
        // VOXY_METAL_BOUND_SSBO; 6 is quads3.vert's per-draw UBO slot, 7/8
        // are cmdgen compute bindings). The texture binding above is kept
        // for compatibility but the shader no longer samples it (depth-
        // format textures read zeros through texture2d<float> declarations).
        if (viewport.metalBoundReadBuffer != null) {
            encoder.setBuffer(9, viewport.metalBoundReadBuffer, 0);
        }
        // Buffer binding 10 — the built-section mask again, for the per-chunk-column cull. The same
        // upload cmdgen read this frame; the compute pass that maintains it runs before the draws.
        // Bound for all three draws (this method is shared), because the cull has to apply to the
        // translucent surface too or water is culled over vanilla water and nowhere else.
        if (CHUNK_CULL && this.builtSectionMask.buffer() != null) {
            encoder.setBuffer(BUILT_MASK_CHUNK_BINDING, this.builtSectionMask.buffer(), 0);
        }

        encoder.bindIndexBuffer(me.cortex.voxy.client.core.rendering.util.SharedIndexBuffer.INSTANCE.getBuffer(),
                me.cortex.voxy.client.core.gpu.RenderEncoder.INDEX_TYPE_UINT16, 0);
        encoder.drawIndexedIndirect(
                me.cortex.voxy.client.core.gpu.RenderEncoder.PRIMITIVE_TRIANGLES,
                viewport.drawCallBuffer, indirectOffset,
                maxDrawCount,
                /*stride*/ 5 * 4); // DrawElementsIndirectCommand = 5 uint32
    }

    @Override
    public void renderTranslucent(MDICViewport viewport) {
        if (this.geometryManager.getSectionCount() == 0) return;

        glEnable(GL_BLEND);
        glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

        glDisable(GL_CULL_FACE);
        glEnable(GL_DEPTH_TEST);
        if (this.translucentTerrainShader != null) {
            this.translucentTerrainShader.bind();
        } else if (this.translucentTerrainProgram != 0) {
            org.lwjgl.opengl.GL20C.glUseProgram(this.translucentTerrainProgram);
        }
        glBindVertexArray(RenderBackendFactory.get().getStaticVAO());//Needs to be before binding
        this.pipeline.setupAndBindTranslucent(viewport);
        this.bindRenderingBuffers(viewport);

        glMemoryBarrier(GL_COMMAND_BARRIER_BIT|GL_SHADER_STORAGE_BARRIER_BIT);//Barrier everything is needed
        glProvokingVertex(GL_FIRST_VERTEX_CONVENTION);
        int translucentMax = Math.min(this.geometryManager.getSectionCount(), 100_000);
        if (Capabilities.INSTANCE.indirectCount) {
            glMultiDrawElementsIndirectCountARB(GL_TRIANGLES, GL_UNSIGNED_SHORT, TRANSLUCENT_OFFSET*5*4, 4*4, translucentMax, 0);
        } else {
            int drawCount = Math.min(readDrawCount(viewport.drawCountCallBuffer.id(), 4*4L), translucentMax);
            glMultiDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_SHORT, TRANSLUCENT_OFFSET*5*4, drawCount, 0);
        }

        glEnable(GL_CULL_FACE);
        glBindVertexArray(0);
        glBindSampler(0, 0);
        bindTextureUnit(0, GL_TEXTURE_2D, 0);
        glBindSampler(1, 0);
        bindTextureUnit(1, GL_TEXTURE_2D, 0);

        glDisable(GL_BLEND);
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
        boolean computeSerialize = "1".equals(System.getenv("VOXY_COMPUTE_SERIALIZE"))
                && this.backend.getType() != BackendType.OPENGL;
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
        if (this.backend.getType() != BackendType.OPENGL && METAL_ZERO_DRAWBUF) {
            viewport.drawCallBuffer.zero();
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
                encoder.setBuffer(1, viewport.drawCountCallBuffer, 0);
                encoder.setBuffer(2, viewport.getRenderList(), 0);
                encoder.barrier(ComputeEncoder.BARRIER_SHADER, ComputeEncoder.BARRIER_SHADER);
                encoder.dispatch(1, 1, 1);
                encoder.barrier(ComputeEncoder.BARRIER_SHADER, ComputeEncoder.BARRIER_SHADER);
            }
        }
        if (computeSerialize) this.backend.submit(); // serialize: prep complete

        {//Test occlusion
            if (this.backend.getType() == BackendType.OPENGL) {
                // GL path — depth-test-based occlusion cull. Rasterizes each
                // section's AABB against MC's depth buffer with color/depth
                // masks off; raster.frag writes visibilityData for sections
                // whose AABBs survive depth test (with optional
                // NV_representative_fragment_test for perf).
                if (this.cullProgram != 0) org.lwjgl.opengl.GL20C.glUseProgram(this.cullProgram);
                if (Capabilities.INSTANCE.repFragTest) {
                    glEnable(GL_REPRESENTATIVE_FRAGMENT_TEST_NV);
                }
                glBindVertexArray(RenderBackendFactory.get().getStaticVAO());
                // SceneUniform is an SSBO now (see bindings.glsl).
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, this.uniform.id());
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, this.geometryManager.getMetadataBuffer().id());
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, viewport.visibilityBuffer.id());
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, viewport.indirectLookupBuffer.id());
                glBindBuffer(GL_DRAW_INDIRECT_BUFFER, viewport.drawCountCallBuffer.id());
                glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, SharedIndexBuffer.INSTANCE.id());
                glEnable(GL_DEPTH_TEST);
                glColorMask(false, false, false, false);
                glDepthMask(false);
                glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT|GL_COMMAND_BARRIER_BIT);
                glDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_BYTE, 6*4);
                glDepthMask(true);
                glColorMask(true, true, true, true);
                glDisable(GL_DEPTH_TEST);
                if (Capabilities.INSTANCE.repFragTest) {
                    glDisable(GL_REPRESENTATIVE_FRAGMENT_TEST_NV);
                }
            } else {
                // Non-GL path (Metal today) — compute stub that skips
                // occlusion and marks every frustum-visible section as
                // visible-this-frame + visible-last-frame. Slower than real
                // depth occlusion but functionally correct; the real cull
                // depends on cross-context MC-depth access which is part of
                // chunk 6's IGpuRenderTarget work.
                try (var encoder = this.backend.beginComputePass()) {
                    encoder.setPipeline(this.forceAllVisiblePipeline);
                    encoder.setBuffer(0, this.uniform, 0);
                    encoder.setBuffer(2, viewport.visibilityBuffer, 0);
                    encoder.setBuffer(3, viewport.indirectLookupBuffer, 0);
                    encoder.barrier(ComputeEncoder.BARRIER_SHADER | ComputeEncoder.BARRIER_INDIRECT,
                                    ComputeEncoder.BARRIER_SHADER | ComputeEncoder.BARRIER_INDIRECT);
                    // Reuses prep's dispatch sizing — cmdGenDispatchX/Y/Z at
                    // offset 0 of drawCountCallBuffer holds ceil(sectionCount/128),
                    // matching this shader's local_size_x=128.
                    encoder.dispatchIndirect(viewport.drawCountCallBuffer, 0);
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
                encoder.setBuffer(0, this.uniform, 0);
                encoder.setBuffer(1, viewport.drawCallBuffer, 0);
                encoder.setBuffer(2, viewport.drawCountCallBuffer, 0);
                encoder.setBuffer(3, this.geometryManager.getMetadataBuffer(), 0);
                encoder.setBuffer(4, viewport.visibilityBuffer, 0);
                encoder.setBuffer(5, viewport.indirectLookupBuffer, 0);
                encoder.setBuffer(6, viewport.positionScratchBuffer, 0);
                encoder.setBuffer(7, this.distanceCountBuffer, 0);
                if (RenderStatistics.enabled) {
                    encoder.setBuffer(STATISTICS_BUFFER_BINDING, this.statisticsBuffer, 0);
                }
                encoder.barrier(ComputeEncoder.BARRIER_SHADER, ComputeEncoder.BARRIER_SHADER);
                encoder.dispatchIndirect(viewport.drawCountCallBuffer, 0);
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
            // M12 chunk 1: prefixSum migrated to ComputeEncoder. Runs on every
            // backend (GL lowers to glUseProgram + glBindBufferBase +
            // glDispatchCompute; Metal opens an MTLComputeCommandEncoder). The
            // previous raw-GL pattern no-opped on Metal because
            // mdicProgramId(prefixSumPipeline) returns 0.
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
                encoder.setBuffer(0, this.uniform, 0);
                encoder.setBuffer(1, viewport.drawCallBuffer, 0);
                encoder.setBuffer(2, viewport.drawCountCallBuffer, 0);
                encoder.setBuffer(3, this.geometryManager.getMetadataBuffer(), 0);
                encoder.setBuffer(4, viewport.indirectLookupBuffer, 0);
                encoder.setBuffer(5, this.distanceCountBuffer, 0);
                encoder.barrier(ComputeEncoder.BARRIER_SHADER | ComputeEncoder.BARRIER_INDIRECT,
                                ComputeEncoder.BARRIER_SHADER | ComputeEncoder.BARRIER_INDIRECT);
                encoder.dispatchIndirect(viewport.drawCountCallBuffer, 0);
                encoder.barrier(ComputeEncoder.BARRIER_SHADER | ComputeEncoder.BARRIER_INDIRECT,
                                ComputeEncoder.BARRIER_SHADER | ComputeEncoder.BARRIER_INDIRECT);
            }
        }

    }

    @Override
    public void renderTemporal(MDICViewport viewport) {
        if (this.geometryManager.getSectionCount() == 0) return;
        //Render temporal
        this.renderTerrain(viewport, TEMPORAL_OFFSET*5*4, 4*5, Math.min(this.geometryManager.getSectionCount(), 100_000));
    }

    private int readDrawCount(int bufferId, long offsetBytes) {
        var tmp = MemoryUtil.memAllocInt(1);
        glBindBuffer(GL_COPY_READ_BUFFER, bufferId);
        glGetBufferSubData(GL_COPY_READ_BUFFER, offsetBytes, tmp);
        glBindBuffer(GL_COPY_READ_BUFFER, 0);
        int count = tmp.get(0);
        MemoryUtil.memFree(tmp);
        return Math.max(count, 0);
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
        this.uniform.free();
        this.distanceCountBuffer.free();
        if (this.translucentTerrainShader != null) this.translucentTerrainShader.free();
        if (this.terrainShader != null) this.terrainShader.free();
        if (this.translucentTerrainPipeline != null) this.translucentTerrainPipeline.close();
        if (this.terrainPipeline != null) this.terrainPipeline.close();
        this.commandGenPipeline.close();
        this.cullPipeline.close();
        this.forceAllVisiblePipeline.close();
        this.prepPipeline.close();
        this.translucentGenPipeline.close();
        this.prefixSumPipeline.close();
        this.statisticsBuffer.free();
        if (this.boundDepthSampler != null) this.boundDepthSampler.close();
    }
}
