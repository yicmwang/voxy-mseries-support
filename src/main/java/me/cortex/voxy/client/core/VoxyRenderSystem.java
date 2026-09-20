package me.cortex.voxy.client.core;

import me.cortex.voxy.client.core.gpu.GlCompat;

import com.mojang.blaze3d.opengl.GlConst;
import com.mojang.blaze3d.opengl.GlStateManager;
import me.cortex.voxy.client.TimingStatistics;
import me.cortex.voxy.client.VoxyClient;
import me.cortex.voxy.client.config.VoxyConfig;
import me.cortex.voxy.client.core.gl.Capabilities;
import me.cortex.voxy.client.core.model.ModelBakerySubsystem;
import me.cortex.voxy.client.core.model.ModelStore;
import me.cortex.voxy.client.core.rendering.ChunkBoundRenderer;
import me.cortex.voxy.client.core.rendering.RenderDistanceTracker;
import me.cortex.voxy.client.core.rendering.Viewport;
import me.cortex.voxy.client.core.rendering.ViewportSelector;
import me.cortex.voxy.client.core.rendering.building.RenderGenerationService;
import me.cortex.voxy.client.core.rendering.hierachical.AsyncNodeManager;
import me.cortex.voxy.client.core.rendering.hierachical.HierarchicalOcclusionTraverser;
import me.cortex.voxy.client.core.rendering.hierachical.NodeCleaner;
import me.cortex.voxy.client.core.rendering.section.IUsesMeshlets;
import me.cortex.voxy.client.core.rendering.section.backend.AbstractSectionRenderer;
import me.cortex.voxy.client.core.rendering.section.backend.mdic.MDICSectionRenderer;
import me.cortex.voxy.client.core.rendering.section.geometry.BasicSectionGeometryData;
import me.cortex.voxy.client.core.rendering.section.geometry.IGeometryData;
import me.cortex.voxy.client.core.rendering.util.DownloadStream;
import me.cortex.voxy.client.core.rendering.util.PrintfDebugUtil;
import me.cortex.voxy.client.core.rendering.util.UploadStream;
import me.cortex.voxy.client.core.util.GPUTiming;
import me.cortex.voxy.client.core.util.IrisUtil;
import me.cortex.voxy.common.Logger;
import me.cortex.voxy.common.thread.ServiceManager;
import me.cortex.voxy.common.world.WorldEngine;
import me.cortex.voxy.commonImpl.VoxyCommon;
import net.caffeinemc.mods.sodium.client.render.chunk.ChunkRenderMatrices;
import net.caffeinemc.mods.sodium.client.util.FogParameters;
import net.minecraft.client.Minecraft;
import org.joml.Matrix4f;
import org.joml.Matrix4fc;
import org.lwjgl.opengl.GL11;

import java.util.Arrays;
import java.util.List;

import static org.lwjgl.opengl.GL11.GL_VIEWPORT;
import static org.lwjgl.opengl.GL11.glGetIntegerv;
import static org.lwjgl.opengl.GL11C.*;
import static org.lwjgl.opengl.GL30C.*;
import static org.lwjgl.opengl.GL33.glBindSampler;
import static org.lwjgl.opengl.GL43.GL_SHADER_STORAGE_BUFFER;
import static org.lwjgl.opengl.GL43C.GL_SHADER_STORAGE_BUFFER_BINDING;

public class VoxyRenderSystem {
    /**
     * Saving and restoring MC's GL SSBO bindings is meaningless -- and, worse, fatal -- without a GL
     * backend: under whole-frame Metal there is no context and glGetIntegeri aborts the JVM
     * ("FATAL ERROR in native method: No context is current"). Nothing Voxy does on Metal touches GL
     * state, so there is nothing to preserve.
     */
    private static final boolean SAVE_GL_STATE =
            me.cortex.voxy.client.core.gpu.RenderBackendFactory.get().getType()
                    == me.cortex.voxy.client.core.gpu.BackendType.OPENGL;

    /**
     * VOXY_GPU_SYNC=1: block the render thread until every command buffer committed before this point
     * has completed, once per frame.
     *
     * <p>A DIAGNOSTIC, not a fix -- it serialises CPU against GPU and costs real frame time. It exists
     * to answer one question that a one-line measurement raised and nothing else can settle.
     *
     * <p>Flipping the {@code VOXY_DRAWCHK} default from 1 to 0 -- i.e. skipping a per-frame CPU readback
     * of GPU buffers and its log line, and changing NOTHING else -- moves the artefact from 22.1% of
     * frames over 10,000 near-black px to 85.1%. The cull, the shaders and every other file were
     * identical between those two runs. So the artefact's RATE depends on how much CPU work the render
     * thread does: it is a race, and the render thread wins it more often when it is fast. That also
     * means every baseline this investigation has quoted was measured in a regime whose instrumentation
     * was itself suppressing the bug.
     *
     * <p>If waiting for the GPU each frame removes the artefact, the race is "the render thread draws
     * before the uploads it depends on have landed", and the fix is a synchronisation point that uses
     * the existing abstraction ({@code IGpuFence} / {@code RenderBackend.waitForGpuIdle}) rather than a
     * new mechanism. If it does not, the race is elsewhere -- between the async thread's staging and
     * the render thread -- and this narrows that too, since the GPU side will have been excluded.
     */
    private static final boolean GPU_SYNC = "1".equals(System.getenv("VOXY_GPU_SYNC"));

    private final WorldEngine worldIn;


    private final ModelBakerySubsystem modelService;
    private final RenderGenerationService renderGen;
    private final IGeometryData geometryData;
    private final AsyncNodeManager nodeManager;
    private final NodeCleaner nodeCleaner;
    private final HierarchicalOcclusionTraverser traversal;


    private final RenderDistanceTracker renderDistanceTracker;
    public final ChunkBoundRenderer chunkBoundRenderer;

    private final ViewportSelector<?> viewportSelector;

    private final AbstractRenderPipeline pipeline;

    /** Diagnostic frame counter for the Metal LOD-ring log in {@link #renderOpaque}. */
    private int metalRingDiagFrame;

    /** Sampling counter for the {@code VOXY_VP_TRACE} projection dump. */
    private static long baseProjTraceCount = 0;

    // Bakery warmup burst (2026-07-03, Metal branch only). Root cause of the
    // "gigantic untextured LOD blocks for minutes after world join": mesh
    // builds throw IdNotYetComputedException for any unbaked block (the
    // 2026-05-25 log showed >1M throws vs 54k sections), the mesh queue
    // saturates past HierarchicalOcclusionTraverser's 4000-task request
    // throttle, and the bakery drains the whole backlog at ~5 bakes/tick
    // under the steady-state 0.9 ms budget. While the bake backlog is large
    // the frame is startup-degraded anyway, so spend real frame time
    // draining it: full burst above BAKE_BURST_HIGH pending bakes, half
    // burst above BAKE_BURST_LOW, steady 0.9 ms once warm.
    // VOXY_BAKE_WARMUP_MS tunes the burst budget in ms (default 8;
    // 0 disables the burst entirely — the pre-2026-07 behaviour).
    private static final long BAKE_BUDGET_STEADY_NS = 900_000L;
    private static final long BAKE_WARMUP_NS = parseBakeWarmupNs();
    // 2026-07-03 tier retune from the first BSL session: backlog peaked at 68
    // (full tier at 256 never engaged) and then hovered at 29-31 — just under
    // the old LOW=32 release — leaving a ~30-bake tail draining at 0.9 ms for
    // the whole session. Full burst from 128, and hold the burst until the
    // backlog is basically empty (8).
    private static final int BAKE_BURST_HIGH = 128;
    private static final int BAKE_BURST_LOW = 8;
    private boolean bakeWarmupActive;
    private int bakeWarmupTransitions;

    private static long parseBakeWarmupNs() {
        String v = System.getenv("VOXY_BAKE_WARMUP_MS");
        if (v == null || v.isBlank()) return 8_000_000L;
        try {
            return (long) (Float.parseFloat(v.trim()) * 1_000_000L);
        } catch (NumberFormatException e) {
            return 8_000_000L;
        }
    }

    private long computeBakeBudgetNs() {
        if (BAKE_WARMUP_NS <= 0) return BAKE_BUDGET_STEADY_NS;
        int backlog = this.modelService.getProcessingCount();
        long budget = BAKE_BUDGET_STEADY_NS;
        if (backlog > BAKE_BURST_HIGH) {
            budget = Math.max(BAKE_BUDGET_STEADY_NS, BAKE_WARMUP_NS);
        } else if (backlog > BAKE_BURST_LOW) {
            budget = Math.max(BAKE_BUDGET_STEADY_NS, BAKE_WARMUP_NS / 2);
        }
        boolean active = budget > BAKE_BUDGET_STEADY_NS;
        if (active != this.bakeWarmupActive) {
            this.bakeWarmupActive = active;
            // The low release threshold makes engage/release flap frame-to-
            // frame while bakes trickle in; log the first few transitions
            // (the interesting ones at world join) then sample.
            this.bakeWarmupTransitions++;
            if (this.bakeWarmupTransitions <= 4 || (this.bakeWarmupTransitions % 200) == 0) {
                me.cortex.voxy.common.Logger.info("[Metal-BAKE] warmup burst "
                        + (active ? ("ENGAGED (backlog=" + backlog + ", budget=" + (budget / 1_000_000L) + "ms)")
                                  : ("released (backlog=" + backlog + ", back to 0.9ms)"))
                        + " [transition " + this.bakeWarmupTransitions + "]");
            }
        }
        return budget;
    }

    /**
     * Iris pack-inject state the renderer (and its pipeline) was constructed
     * with. NormalRenderPipeline bakes {@code useEnvFog} from this at
     * construction (it's a compile-time shader define), so a pack
     * enable/disable at runtime needs a full renderer recreation — the same
     * shutdownRenderer()/createRenderer() path Sodium's
     * REQUIRES_RENDERER_RELOAD config flag (e.g. the env-fog toggle) drives.
     * {@link #renderOpaque} watches for the flip on Metal.
     */
    private final boolean constructedIrisGbufferInject;
    /** One-shot guard so the reload is scheduled exactly once per flip. */
    private boolean irisReloadScheduled;

    /** Accessor exposed for the Metal compositing mixin so it can read the IOSurface bridge. */
    public AbstractRenderPipeline getPipeline() {
        return this.pipeline;
    }

    private static AbstractSectionRenderer.Factory<?,? extends IGeometryData> getRenderBackendFactory() {
        //TODO: need todo a thing where selects optimal section render based on if supports the pipeline and geometry data type
        return MDICSectionRenderer.FACTORY;
    }

    public VoxyRenderSystem(WorldEngine world, ServiceManager sm) {
        //Keep the world loaded, NOTE: this is done FIRST, to keep and ensure that even if the rest of loading takes more
        // than timeout, we keep the world acquired
        world.acquireRef();
        System.gc();

        if (Minecraft.getInstance().options.getEffectiveRenderDistance()<3) {
            Logger.warn("Having a vanilla render distance of 2 can cause rare culling near the edge of your screen issues, please use 3 or more");
        }

        this.constructedIrisGbufferInject = IrisUtil.irisGbufferInjectMode();

        // 2026-07-03 (Metal): migrate sub_division_size drift. The FPS-based
        // autoBalanceSubDivSize loop (call site commented out below in
        // renderOpaque) used to RAISE this value up to 256 whenever FPS<55
        // and persist it via VoxyConfig.save() — but nothing ever lowers it
        // again. A drifted value (user config had 229.18) makes LOD leaves
        // stop subdividing at a huge screen footprint: 16-block voxels at
        // only ~2000 blocks = the "gigantic shapeless distant blocks"
        // report. Values above 128 can only come from that disabled loop
        // (the config UI stays well below it), so reset them to the
        // default 64. Metal-only guard keeps GL behaviour untouched.
        if (me.cortex.voxy.client.core.gpu.RenderBackendFactory.get().getType()
                != me.cortex.voxy.client.core.gpu.BackendType.OPENGL
                && VoxyConfig.CONFIG.subDivisionSize > 128f) {
            Logger.warn("[Metal] sub_division_size drifted to " + VoxyConfig.CONFIG.subDivisionSize
                    + " (residue of the disabled FPS auto-balancer) — resetting to 64 for full LOD detail");
            VoxyConfig.CONFIG.subDivisionSize = 64f;
            VoxyConfig.CONFIG.save();
        }

        //Fking HATE EVERYTHING AAAAAAAAAAAAAAAA
        int[] oldBufferBindings = new int[10];
        if (SAVE_GL_STATE) {
            for (int i = 0; i < oldBufferBindings.length; i++) {
                oldBufferBindings[i] = glGetIntegeri(GL_SHADER_STORAGE_BUFFER_BINDING, i);
            }
        }

        try {
            //wait for opengl to be finished, this should hopefully ensure all memory allocations are free
            if (SAVE_GL_STATE) {
                GlCompat.finish();
                GlCompat.finish();
            }

            this.worldIn = world;

            long geometryCapacity = getGeometryBufferSize();
            var backendFactory = getRenderBackendFactory();

            {
                this.modelService = new ModelBakerySubsystem(world.getMapper());
                this.renderGen = new RenderGenerationService(world, this.modelService, sm, IUsesMeshlets.class.isAssignableFrom(backendFactory.clz()));

                this.geometryData = new BasicSectionGeometryData(1 << 20, geometryCapacity);

                this.nodeManager = new AsyncNodeManager(1 << 21, this.geometryData, this.renderGen);
                this.nodeCleaner = new NodeCleaner(this.nodeManager);
                this.traversal = new HierarchicalOcclusionTraverser(this.nodeManager, this.nodeCleaner, this.renderGen);

                world.setDirtyCallback(this.nodeManager::worldEvent);

                Arrays.stream(world.getMapper().getBiomeEntries()).forEach(this.modelService::addBiome);
                world.getMapper().setBiomeCallback(this.modelService::addBiome);

                this.nodeManager.start();
            }

            this.pipeline = RenderPipelineFactory.createPipeline(this.nodeManager, this.nodeCleaner, this.traversal, this::frexStillHasWork);
            this.pipeline.setupExtraModelBakeryData(this.modelService);//Configure the model service
            var sectionRenderer = backendFactory.create(this.pipeline, this.modelService.getStore(), this.geometryData);
            this.pipeline.setSectionRenderer(sectionRenderer);
            this.viewportSelector = new ViewportSelector<>(sectionRenderer::createViewport);

            {
                int minSec = Minecraft.getInstance().level.getMinSectionY() >> 5;
                int maxSec = (Minecraft.getInstance().level.getMaxSectionY() - 1) >> 5;

                //Do some very cheeky stuff for MiB
                if (VoxyCommon.IS_MINE_IN_ABYSS) {//TODO: make this somehow configurable
                    minSec = -8;
                    maxSec = 7;
                }

                this.renderDistanceTracker = new RenderDistanceTracker(20,
                        minSec,
                        maxSec,
                        this.nodeManager::addTopLevel,
                        this.nodeManager::removeTopLevel);

                this.setRenderDistance(VoxyConfig.CONFIG.sectionRenderDistance);
            }

            this.chunkBoundRenderer = new ChunkBoundRenderer(this.pipeline);

            Logger.info("Voxy render system created with " + geometryCapacity + " geometry capacity, using pipeline '" + this.pipeline.getClass().getSimpleName() + "' with renderer '" + sectionRenderer.getClass().getSimpleName() + "'");
        } catch (RuntimeException e) {
            world.releaseRef();//If something goes wrong, we must release the world first
            throw e;
        }

        if (SAVE_GL_STATE) {
            for (int i = 0; i < oldBufferBindings.length; i++) {
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, i, oldBufferBindings[i]);
            }
        }

        if (SAVE_GL_STATE) {
            for (int i = 0; i < 12; i++) {
                GlStateManager._activeTexture(GlConst.GL_TEXTURE0+i);
                GlStateManager._bindTexture(0);
                glBindSampler(i, 0);
            }
        }
    }


    /**
     * Temporal smoothing of the captured fog. MC's eye-in-fluid test is a
     * binary per-frame flip (no hysteresis), so bobbing across the water
     * surface alternates the captured FogParameters between water-fog (dark,
     * env end ~24-96) and air-fog (light, env end ~render distance) every
     * frame. Voxy paints the ENTIRE far field from this one record (bridge
     * clear + LOD fog mix), so the raw flip strobes the whole horizon.
     * Exponentially lerp all components toward the current value (~200 ms
     * time constant) so a crossing becomes a brief fade instead.
     * VOXY_FOG_SMOOTH_MS overrides the time constant; 0 disables.
     */
    private static final float FOG_SMOOTH_MS = parseFogSmoothMs();
    private static float parseFogSmoothMs() {
        String v = System.getenv("VOXY_FOG_SMOOTH_MS");
        if (v == null || v.isBlank()) return 200.0f;
        try {
            return Float.parseFloat(v.trim());
        } catch (NumberFormatException e) {
            return 200.0f;
        }
    }
    private FogParameters smoothedFog;
    private long lastFogSmoothNs;
    private boolean fogClassWater;
    private int fogClassStreak;
    private long fogClassStreakStartNs;
    private static boolean loggedViewportLeak;

    /**
     * Submersion-type fog records (water/lava/powder-snow/blindness) carry a
     * short environmental end; atmospheric fog is hundreds of blocks. The
     * class of the RAW captured record tracks MC's binary eye-in-fluid
     * verdict without querying the camera.
     */
    private static boolean isSubmersionClassFog(FogParameters p) {
        return p.environmentalEnd() < 128.0f;
    }

    private FogParameters smoothFogParameters(FogParameters target) {
        if (FOG_SMOOTH_MS <= 0 || target == null) return target;
        long now = System.nanoTime();
        if (this.smoothedFog == null) {
            this.smoothedFog = target;
            this.lastFogSmoothNs = now;
            this.fogClassWater = isSubmersionClassFog(target);
            this.fogClassStreak = 0;
            return target;
        }
        // ASYMMETRIC DEBOUNCED SNAP on fog-class change. MC's eye-in-fluid
        // verdict is binary per frame and OSCILLATES while swimming at the
        // surface (flowing-water fractional fluid heights + swim bob), with
        // run lengths of 100-300 ms — long enough to defeat a symmetric
        // 4-frame filter (each bob produced two full-screen snaps). The
        // failure modes are asymmetric, so the filter is too:
        //  - AIR→WATER (densify) adopts after 2 agreeing frames — murk hides
        //    everything, divers get instant response, and a spurious densify
        //    is visually harmless.
        //  - WATER→AIR (thin/REVEAL) adopts only after 400 ms of consecutive
        //    air verdicts — bobbing never thins the fog, so the far field
        //    stays murky and stable through any splash pattern; a real
        //    surfacing pays 0.4 s of lingering haze.
        // While a flip is pending, hold the DISTANCE fields stable and keep
        // lerping the colour toward the target (no colour strobe either).
        boolean targetClass = isSubmersionClassFog(target);
        if (targetClass != this.fogClassWater) {
            if (this.fogClassStreak == 0) {
                this.fogClassStreakStartNs = now;
            }
            this.fogClassStreak++;
            boolean adopt = targetClass
                    ? this.fogClassStreak >= 2
                    : (now - this.fogClassStreakStartNs) >= 400_000_000L;
            if (adopt) {
                this.fogClassWater = targetClass;
                this.fogClassStreak = 0;
                this.smoothedFog = target;
                this.lastFogSmoothNs = now;
                return target;
            }
            float dtHold = (now - this.lastFogSmoothNs) / 1.0e9f;
            this.lastFogSmoothNs = now;
            float kHold = 1.0f - (float) Math.exp(-dtHold / (FOG_SMOOTH_MS / 1000.0f));
            FogParameters h = this.smoothedFog;
            this.smoothedFog = new FogParameters(
                    h.red()   + (target.red()   - h.red())   * kHold,
                    h.green() + (target.green() - h.green()) * kHold,
                    h.blue()  + (target.blue()  - h.blue())  * kHold,
                    h.alpha() + (target.alpha() - h.alpha()) * kHold,
                    h.environmentalStart(), h.environmentalEnd(),
                    h.renderStart(), h.renderEnd());
            return this.smoothedFog;
        }
        this.fogClassStreak = 0;
        float dt = (now - this.lastFogSmoothNs) / 1.0e9f;
        this.lastFogSmoothNs = now;
        // Colour uses the full time constant (kills the eye-crossing colour
        // strobe); the DISTANCE fields use a fast constant (<=250 ms) so fog
        // density tracks promptly within a fog type (e.g. waterVision ramp) —
        // slow distance smoothing dilutes underwater murk and reveals the far
        // field that vanilla hides.
        float kCol = 1.0f - (float) Math.exp(-dt / (FOG_SMOOTH_MS / 1000.0f));
        float kDist = 1.0f - (float) Math.exp(-dt / (Math.min(FOG_SMOOTH_MS, 250.0f) / 1000.0f));
        FogParameters p = this.smoothedFog;
        this.smoothedFog = new FogParameters(
                p.red()   + (target.red()   - p.red())   * kCol,
                p.green() + (target.green() - p.green()) * kCol,
                p.blue()  + (target.blue()  - p.blue())  * kCol,
                p.alpha() + (target.alpha() - p.alpha()) * kCol,
                p.environmentalStart() + (target.environmentalStart() - p.environmentalStart()) * kDist,
                p.environmentalEnd()   + (target.environmentalEnd()   - p.environmentalEnd())   * kDist,
                p.renderStart() + (target.renderStart() - p.renderStart()) * kDist,
                p.renderEnd()   + (target.renderEnd()   - p.renderEnd())   * kDist);
        return this.smoothedFog;
    }

    public Viewport<?> setupViewport(ChunkRenderMatrices matrices, FogParameters fogParameters, double cameraX, double cameraY, double cameraZ) {
        var viewport = this.getViewport();
        if (viewport == null) {
            return null;
        }
        fogParameters = this.smoothFogParameters(fogParameters);

        //Do some very cheeky stuff for MiB
        if (VoxyCommon.IS_MINE_IN_ABYSS) {
            int sector = (((int)Math.floor(cameraX)>>4)+512)>>10;
            cameraX -= sector<<14;//10+4
            cameraY += (16+(256-32-sector*30))*16;
        }

        //cameraY += 100;
        // VOXY_VP_TRACE=1: computeProjectionMat substitutes Voxy's near/far by composing
        // `base . P(0.05, rd*16)^-1 . P(nearVoxy, 48000)`. That cancellation is only exact when
        // `base` really is the GL-convention P(0.05, rd*16) it assumes -- for any other near or far
        // the z row survives the composition scaled, which pushes every LOD vertex out of the clip
        // volume while w and xy still look right. Print what `base` actually is, because the whole
        // construction hinges on it and nothing else in the frame does.
        if ("1".equals(System.getenv("VOXY_VP_TRACE")) && (baseProjTraceCount++ % 600) == 0) {
            Matrix4fc base = matrices.projection();
            // base = P(near, far) in some convention; recover (near, far) for each.
            float glN = base.m23() / (base.m22() + 1.0f);
            float glF = base.m23() / (base.m22() - 1.0f);
            float rzN = base.m23() / base.m22();
            float rzF = base.m23() / (base.m22() - 1.0f);
            Logger.info(String.format(java.util.Locale.ROOT,
                    "[Metal-BASEPROJ] m00=%.6f m11=%.6f m22=%.6f m23=%.6f m32=%.6f m33=%.6f "
                            + "| ifGL near=%.5f far=%.3f | ifReverseZ near=%.5f far=%.3f",
                    base.m00(), base.m11(), base.m22(), base.m23(), base.m32(), base.m33(),
                    glN, glF, rzN, rzF));

            float[] b = new float[16];
            new Matrix4f(base).get(b);
            float[] v = new float[16];
            computeProjectionMat(matrices.projection()).get(v);
            Logger.info(String.format(java.util.Locale.ROOT,
                    "[Metal-BASEPROJ16] base=[%.5f %.5f %.5f %.5f | %.5f %.5f %.5f %.5f | "
                            + "%.5f %.5f %.5f %.5f | %.5f %.5f %.5f %.5f]",
                    b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11],
                    b[12], b[13], b[14], b[15]));
            Logger.info(String.format(java.util.Locale.ROOT,
                    "[Metal-BASEPROJ16] voxyProj=[%.5f %.5f %.5f %.5f | %.5f %.5f %.5f %.5f | "
                            + "%.5f %.5f %.5f %.5f | %.5f %.5f %.5f %.5f]",
                    v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11],
                    v[12], v[13], v[14], v[15]));
            Logger.info(String.format(java.util.Locale.ROOT,
                    "[Metal-BASEPROJ16] modelView=[%.5f %.5f %.5f %.5f | %.5f %.5f %.5f %.5f | "
                            + "%.5f %.5f %.5f %.5f | %.5f %.5f %.5f %.5f]",
                    matrices.modelView().m00(), matrices.modelView().m10(),
                    matrices.modelView().m20(), matrices.modelView().m30(),
                    matrices.modelView().m01(), matrices.modelView().m11(),
                    matrices.modelView().m21(), matrices.modelView().m31(),
                    matrices.modelView().m02(), matrices.modelView().m12(),
                    matrices.modelView().m22(), matrices.modelView().m32(),
                    matrices.modelView().m03(), matrices.modelView().m13(),
                    matrices.modelView().m23(), matrices.modelView().m33()));
        }
        var projection = computeProjectionMat(matrices.projection());//RenderSystem.getProjectionMatrix();
        //var projection = ShadowMatrices.createOrthoMatrix(160, -16*300, 16*300);
        //var projection = new Matrix4f(matrices.projection());

        // GL_VIEWPORT is only readable with a GL context; on Metal the non-GL branch below
        // derives the size from MC's main render target instead (and does so deliberately --
        // see its note on why GL_VIEWPORT is untrustworthy there anyway).
        int width = 0;
        int height = 0;
        if (me.cortex.voxy.client.core.gpu.RenderBackendFactory.get().getType()
                == me.cortex.voxy.client.core.gpu.BackendType.OPENGL) {
            int[] dims = new int[4];
            glGetIntegerv(GL_VIEWPORT, dims);
            width = dims[2];
            height = dims[3];
        }
        if (me.cortex.voxy.client.core.gpu.RenderBackendFactory.get().getType()
                != me.cortex.voxy.client.core.gpu.BackendType.OPENGL) {
            // GL_VIEWPORT is NOT trustworthy here: MC re-renders the 16x16
            // lightmap every game tick and blaze3d's createRenderPass sets the
            // GL viewport eagerly without restoring; above water the fullscreen
            // sky pass resets it before Sodium's terrain hook, but UNDERWATER
            // Sodium skips the sky pass — so GL_VIEWPORT reads 16x16 on every
            // tick frame (~20 Hz). That inflated minSSS 6400x (the octree walk
            // stopped at the top level: renderList collapsed to ~16) and
            // reallocated the IOSurface bridge to 16x16 (broken blit) — the
            // underwater strobe. MC's main RT is the authoritative frame size
            // (same source the compositor uses).
            var rt = Minecraft.getInstance().gameRenderer.mainRenderTarget();
            if (rt != null && rt.width > 0 && rt.height > 0) {
                if ((width != rt.width || height != rt.height) && !loggedViewportLeak) {
                    loggedViewportLeak = true;
                    Logger.warn("[Metal-VIEWPORT] GL_VIEWPORT " + width + "x" + height
                            + " != mainRT " + rt.width + "x" + rt.height
                            + " (leaked pass viewport; using mainRT size)");
                }
                width = rt.width;
                height = rt.height;
            }
        }

        {//Apply render scaling factor
            var factor = this.pipeline.getRenderScalingFactor();
            if (factor != null) {
                width = (int) (width*factor[0]);
                height = (int) (height*factor[1]);
            }
        }

        viewport
                .setVanillaProjection(matrices.projection())
                .setProjection(projection)
                .setModelView(new Matrix4f(matrices.modelView()))
                .setCamera(cameraX, cameraY, cameraZ)
                .setScreenSize(width, height)
                .setFogParameters(fogParameters)
                .update();

        if (VoxyClient.getOcclusionDebugState()==0) {
            viewport.frameId++;
        }

        return viewport;
    }

    /**
     * VOXY_ATLAS_SYNC=0 disables the per-frame wait before the model-atlas upload. ON by default: it is
     * a correctness fix, not an experiment.
     *
     * <p>The hazard: the bakery writes freshly baked tiles into the block model atlas inside
     * {@code modelService.tick()}, which runs AFTER this frame's LOD draws have been submitted, with
     * frames N-1 and N-2 still executing on the GPU. The write lowers to a bare
     * {@code mtlTextureReplaceRegion} (MetalTexture:163-172) -- a host memcpy that Metal does not order
     * against command buffers already committed on the queue. A tile caught mid-copy reads alpha 0,
     * which for a {@code useDiscard()} quad hits the {@code discard} and gives a hole, and otherwise
     * gives rgb 0: an opaque black fragment with a correct silhouette and a healthy draw count.
     *
     * <p>Why it is worth fixing even though it costs a stall: the atlas changes ONLY when a bake lands,
     * so it is a frame-to-frame difference at a moment when nothing about the camera or the world has
     * changed -- which is exactly the signature the captures have been showing. And the asymmetry is
     * telling: the model SSBO and the biome colour uploads in the same method already go through
     * UploadStream, which is fence-tracked. Only the texture path was left unguarded.
     *
     * <p>Cost, stated plainly: this is a second full drain per frame, on top of the one the submit
     * ordering already takes, so it will make the frame slower. It is correctness-first, and the
     * stall-free version is double-buffering the atlas -- writing to the tile the draws are not
     * sampling -- which is invasive because the atlas is bound at a shader binding and indexed by the
     * model SSBO.
     */
    private static final boolean ATLAS_SYNC = !"0".equals(System.getenv("VOXY_ATLAS_SYNC"));

    /**
     * VOXY_FRAME_SLEEP_MS=<n>: idle the render thread for n milliseconds at the top of each frame.
     *
     * <p>A PROBE, not a fix, and it exists to break one specific confound. Every timing measurement
     * this investigation has is ambiguous, because the instrument that produced the strongest effect
     * does two things at once: {@code VOXY_DRAWCHK} slows the render thread AND reads GPU-written
     * buffers from the CPU, and it moved the artefact rate from 5.6% to 75.9%. So "the rate depends on
     * frame pacing" and "the rate depends on the CPU touching those buffers" are indistinguishable in
     * that data. A sleep adds idle time and touches no buffer, so it separates them.
     *
     * <p>The discriminator is this against {@code VOXY_GPU_SYNC}, which bounds only the GPU boundary
     * and recovered 75.9% -> 57.4%:
     *
     * <ul>
     *   <li>sleep helps a lot, GPU sync little -> the race is against the CPU-side producer (async
     *       staging and metadata), not the GPU;</li>
     *   <li>sleep helps about as much as GPU sync -> the hazard is the GPU boundary itself.</li>
     * </ul>
     *
     * <p>Sweeping several levels (say 16 ms for 60 fps, 33 for 30, 100 for 10) adds a monotonicity
     * check: a smooth curve is strong evidence for a timing effect, whereas a step or a cliff would
     * point somewhere else.
     *
     * <p>Two confounds to be honest about. A sleep is not a pure timing knob -- fewer frames per second
     * means more streaming work queued per frame, so per-frame load rises even as pacing improves, and
     * those two push the rate in OPPOSITE directions. That means a null result would NOT refute the
     * timing story; only a clear reduction is informative, and the absence of one is not evidence. And
     * a dedicated sleep is used rather than MC's maxFps/vsync limiter because the latter also affects
     * vanilla and interacts with the fifo present mode visible in the captures.
     *
     * <p>It cannot say WHICH buffer or thread, and it must never be mistaken for a fix: suppressing the
     * rate is exactly the trap this investigation fell into three times, where an instrument made the
     * number look better while the bug was untouched.
     */
    private static final long FRAME_SLEEP_MS = parseFrameSleepMs();

    private static long parseFrameSleepMs() {
        String v = System.getenv("VOXY_FRAME_SLEEP_MS");
        if (v == null || v.isBlank()) return 0L;
        try {
            long ms = Long.parseLong(v.trim());
            if (ms > 0) {
                me.cortex.voxy.common.Logger.info("[Metal-SLEEP] VOXY_FRAME_SLEEP_MS=" + ms
                        + " -- idling the render thread each frame (probe, not a fix)");
            }
            return Math.max(0L, ms);
        } catch (NumberFormatException e) {
            return 0L;
        }
    }

    public void renderOpaque(Viewport<?> viewport) {
        if (viewport == null) {
            return;
        }
        // See FRAME_SLEEP_MS. Ahead of the frame's work rather than after it: the gap between frames is
        // what the async producer gets, and placing it here keeps it off the path of anything that
        // measures or submits.
        if (FRAME_SLEEP_MS > 0) {
            try {
                Thread.sleep(FRAME_SLEEP_MS);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }

        if (me.cortex.voxy.client.core.gpu.RenderBackendFactory.get().getType()
                != me.cortex.voxy.client.core.gpu.BackendType.OPENGL) {
            // Iris pack toggled since construction? The pipeline's env-fog
            // define (and the inject mode it pairs with) is baked at
            // construction, so recreate the renderer through the same
            // shutdown/create path the config screen's renderer-reload flag
            // uses. Deferred via execute(): the task runs on the render
            // thread BETWEEN frames — tearing this renderer down from inside
            // its own renderOpaque would free GPU objects mid-render.
            if (!this.irisReloadScheduled
                    && IrisUtil.irisGbufferInjectMode() != this.constructedIrisGbufferInject) {
                this.irisReloadScheduled = true;
                Logger.info("Iris pack state changed (gbufferInject "
                        + this.constructedIrisGbufferInject + " -> "
                        + IrisUtil.irisGbufferInjectMode()
                        + ") — scheduling Voxy renderer reload to rebake fog/inject mode");
                Minecraft.getInstance().execute(() -> {
                    if (Minecraft.getInstance().levelRenderer instanceof IGetVoxyRenderSystem holder
                            && holder.getVoxyRenderSystem() == this) {
                        holder.shutdownRenderer();
                        holder.createRenderer();
                    }
                });
            }
            // Metal path — skip all the GL state save/restore and drive the pipeline's Metal render.
            // The compositing mixin runs separately at renderLevel RETURN.
            this.pipeline.preSetup(viewport);
            // No chunk-bound mask pass here any more. Voxy draws into MC's own depth attachment, so
            // the depth test occludes LOD against real terrain per pixel with no help; the mask that
            // used to run here was a chunk-AABB approximation of exactly that, and was the source of
            // the rectangles and the mirrored-discard artifacts. See MDICSectionRenderer's define
            // block for why the AABB mask could not have been fixed by choosing sections better, and
            // for the one real gap this leaves (CUTOUT/TRANSLUCENT are not in MC's depth yet when
            // Voxy draws).
            this.pipeline.runPipeline(viewport, 0, viewport.width, viewport.height);

            // M13 chunk 2 follow-up: drive the per-frame dynamic-runtime
            // block on Metal too. Without these, the section tree never
            // advances with the player — `setCenterAndProcess` is what
            // adds/removes top-level LOD nodes as the player moves
            // (more than CHECK_DISTANCE_BLOCKS = 128), loading their
            // subtrees from LMDB into AsyncNodeManager. Without it, the
            // initial nodes are the ONLY LOD that ever renders, so the
            // distant horizon disappears once the player walks out of
            // the spawn ring. `UploadStream.tick` commits the geometry
            // upload buffer copies to the GPU and rotates fenced frames.
            // M13 chunk 1 follow-up: `modelService.tick` runs on Metal
            // too now. The bakery's resources are all raw GL bound to
            // MC's GL context (which is always current on the render
            // thread regardless of Voxy's backend) and the CPU readback
            // path in GlViewCapture works on Apple's GL 4.1 cap. Without
            // this tick the bakery queue stalls at 1 invocation and
            // every mesher call throws IdNotYetComputedException →
            // no LOD geometry ever materializes.
            UploadStream.INSTANCE.tick();
            // See GPU_SYNC. Placed immediately after the frame's uploads are committed, so the stall
            // covers exactly the copies this frame's draws depend on.
            if (GPU_SYNC) {
                me.cortex.voxy.client.core.gpu.RenderBackendFactory.get().waitForGpuIdle();
            }
            boolean processedThisFrame = this.renderDistanceTracker.setCenterAndProcess(
                    viewport.cameraX, viewport.cameraZ);
            while (processedThisFrame && VoxyClient.isFrexActive()) {
                processedThisFrame = this.renderDistanceTracker.setCenterAndProcess(
                        viewport.cameraX, viewport.cameraZ);
            }
            // 2026-07-03: adaptive budget — burst through the startup bake
            // backlog instead of the flat 0.9 ms (see computeBakeBudgetNs).
            //
            // ATLAS_SYNC: see the field. The bakery writes freshly baked tiles into the block model
            // atlas HERE, and that atlas is sampled by the LOD draws submitted earlier this frame --
            // with frames N-1 and N-2 still executing. The write is a host memcpy through
            // mtlTextureReplaceRegion, which Metal does not order against committed command buffers,
            // so a tile caught mid-copy reads alpha 0: a discard hole for a useDiscard() quad, or an
            // opaque black fragment with a correct silhouette otherwise. One wait per frame rather
            // than one per bake, because the bakery drains several tiles per tick and a wait per tile
            // would stall on each of them.
            if (ATLAS_SYNC) {
                me.cortex.voxy.client.core.gpu.RenderBackendFactory.get().waitForGpuIdle();
            }
            do { this.modelService.tick(this.computeBakeBudgetNs()); } while (VoxyClient.isFrexActive() && !this.modelService.areQueuesEmpty());
            // Diagnostic: log every ~10s (600 frames) whether the LOD ring is
            // still adding/removing cells. After the ring converges this
            // should mostly read `processedThisFrame=false` until the player
            // moves >128 blocks.
            this.metalRingDiagFrame++;
            if (this.metalRingDiagFrame % 600 == 1) {
                me.cortex.voxy.common.Logger.info(String.format(
                        "[Metal-RING f=%d] processedThisFrame=%s  cam=(%.0f, %.0f)  cfgRD=%d",
                        this.metalRingDiagFrame, processedThisFrame,
                        viewport.cameraX, viewport.cameraZ,
                        me.cortex.voxy.client.config.VoxyConfig.CONFIG.sectionRenderDistance));
            }
            return;
        }

        TimingStatistics.resetSamplers();

        long startTime = System.nanoTime();
        TimingStatistics.all.start();
        GPUTiming.INSTANCE.marker();//Start marker
        TimingStatistics.main.start();

        //TODO: optimize
        int[] oldBufferBindings = new int[10];
        if (SAVE_GL_STATE) {
            for (int i = 0; i < oldBufferBindings.length; i++) {
                oldBufferBindings[i] = glGetIntegeri(GL_SHADER_STORAGE_BUFFER_BINDING, i);
            }
        }


        int oldFB = GL11.glGetInteger(GL_DRAW_FRAMEBUFFER_BINDING);
        int boundFB = oldFB;

        int[] dims = new int[4];
        glGetIntegerv(GL_VIEWPORT, dims);

        glViewport(0,0, viewport.width, viewport.height);

        //var target = DefaultTerrainRenderPasses.CUTOUT.getTarget();
        //boundFB = ((net.minecraft.client.texture.GlTexture) target.getColorAttachment()).getOrCreateFramebuffer(((GlBackend) RenderSystem.getDevice()).getFramebufferManager(), target.getDepthAttachment());
        if (boundFB == 0) {
            throw new IllegalStateException("Cannot use the default framebuffer as cannot source from it");
        }

        //this.autoBalanceSubDivSize();

        this.pipeline.preSetup(viewport);

        TimingStatistics.E.start();
        if ((!VoxyClient.disableSodiumChunkRender())&&!IrisUtil.irisShadowActive()) {
            this.chunkBoundRenderer.render(viewport);
        } else {
            viewport.depthBoundingBuffer.clear(0);
        }
        TimingStatistics.E.stop();


        //The entire rendering pipeline (excluding the chunkbound thing)
        this.pipeline.runPipeline(viewport, boundFB, dims[2], dims[3]);


        TimingStatistics.main.stop();
        TimingStatistics.postDynamic.start();

        PrintfDebugUtil.tick();

        //As much dynamic runtime stuff here
        {
            //Tick upload stream (this is ok to do here as upload ticking is just memory management)
            UploadStream.INSTANCE.tick();

            while (this.renderDistanceTracker.setCenterAndProcess(viewport.cameraX, viewport.cameraZ) && VoxyClient.isFrexActive());//While FF is active, run until everything is processed
            TimingStatistics.H.start();
            //Done here as is allows less gl state resetup
            do { this.modelService.tick(900_000); } while (VoxyClient.isFrexActive() && !this.modelService.areQueuesEmpty());
            TimingStatistics.H.stop();
        }
        GPUTiming.INSTANCE.marker();
        TimingStatistics.postDynamic.stop();

        GPUTiming.INSTANCE.tick();

        glBindFramebuffer(GlConst.GL_FRAMEBUFFER, oldFB);
        glViewport(dims[0], dims[1], dims[2], dims[3]);

        {//Reset state manager stuffs
            glUseProgram(0);
            glEnable(GL_DEPTH_TEST);

            GlStateManager._glBindVertexArray(0);//Clear binding

            GlStateManager._activeTexture(GlConst.GL_TEXTURE1);
            for (int i = 0; i < 12; i++) {
                GlStateManager._activeTexture(GlConst.GL_TEXTURE0+i);
                GlStateManager._bindTexture(0);
                glBindSampler(i, 0);
            }

            IrisUtil.clearIrisSamplers();//Thanks iris (sigh)

            //TODO: should/needto actually restore all of these, not just clear them
            //Clear all the bindings
            if (SAVE_GL_STATE) {
                for (int i = 0; i < oldBufferBindings.length; i++) {
                    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, i, oldBufferBindings[i]);
                }
            }

            //((SodiumShader) Iris.getPipelineManager().getPipelineNullable().getSodiumPrograms().getProgram(DefaultTerrainRenderPasses.CUTOUT).getInterface()).setupState(DefaultTerrainRenderPasses.CUTOUT, fogParameters);
        }

        TimingStatistics.all.stop();

        //TimingStatistics.I.start();
        //glFlush();
        //TimingStatistics.I.stop();

        /*
        TimingStatistics.F.start();
        this.postProcessing.setup(viewport.width, viewport.height, boundFB);
        TimingStatistics.F.stop();

        this.renderer.renderFarAwayOpaque(viewport, this.chunkBoundRenderer.getDepthBoundTexture());


        TimingStatistics.F.start();
        //Compute the SSAO of the rendered terrain, TODO: fix it breaking depth or breaking _something_ am not sure what
        this.postProcessing.computeSSAO(viewport.MVP);
        TimingStatistics.F.stop();

        TimingStatistics.G.start();
        //We can render the translucent directly after as it is the furthest translucent objects
        this.renderer.renderFarAwayTranslucent(viewport, this.chunkBoundRenderer.getDepthBoundTexture());
        TimingStatistics.G.stop();


        TimingStatistics.F.start();
        this.postProcessing.renderPost(viewport, matrices.projection(), boundFB);
        TimingStatistics.F.stop();
         */
    }



    private void autoBalanceSubDivSize() {
        //only increase quality while there are very few mesh queues, this stops,
        // e.g. while flying and is rendering alot of low quality chunks
        boolean canDecreaseSize = this.renderGen.getTaskCount() < 300;
        int MIN_FPS = 55;
        int MAX_FPS = 65;
        float INCREASE_PER_SECOND = 60;
        float DECREASE_PER_SECOND = 30;
        //Auto fps targeting
        if (Minecraft.getInstance().getFps() < MIN_FPS) {
            VoxyConfig.CONFIG.subDivisionSize = Math.min(VoxyConfig.CONFIG.subDivisionSize + INCREASE_PER_SECOND / Math.max(1f, Minecraft.getInstance().getFps()), 256);
        }

        if (MAX_FPS < Minecraft.getInstance().getFps() && canDecreaseSize) {
            VoxyConfig.CONFIG.subDivisionSize = Math.max(VoxyConfig.CONFIG.subDivisionSize - DECREASE_PER_SECOND / Math.max(1f, Minecraft.getInstance().getFps()), 28);
        }
    }

    //TODO: Make a reverse z buffer
    private static Matrix4f computeProjectionMat(Matrix4fc base) {
        //THis is a wild and insane problem to have
        // at short render distances the vanilla terrain doesnt end up covering the 16f near plane voxy uses
        // meaning that it explodes (due to near plane clipping).. _badly_ with the rastered culling being wrong in rare cases for the immediate
        // sections rendered after the vanilla render distance
        // Whole-frame Metal: use vanilla's projection UNCHANGED.
        //
        // Voxy normally substitutes its own near/far so the LOD ring starts where the loaded chunks
        // end. That changes the depth MAPPING -- and on this path Voxy draws into MC's own depth
        // attachment, so a LOD fragment's gl_FragCoord.z and the depth MC's terrain already wrote
        // are values in two different spaces. A depth test cannot occlude across a mismatched
        // mapping: LOD passes GREATER_EQUAL almost everywhere and paints over vanilla terrain
        // ("renders over everything"), while everything nearer than Voxy's 16-block near plane is
        // clipped outright -- which is the plane the LOD refuses to appear in front of.
        //
        // Vanilla's own projection is reverse-Z with an infinite far plane, where depth = near/d and
        // precision is near-uniform at ANY distance. The 16-block near plane's rationale was GL's
        // [-1,1] depth against a 48000-block far, which does not apply here. Sharing vanilla's
        // mapping is simpler AND the only way the depth test means anything.
        //
        // (This is also why upstream's depth path needs transformBlitDepth: it reprojects MC's depth
        // through inverse(viewport.MVP) and a target transform precisely because the two projections
        // differ. Sharing one projection removes the need for the transform entirely.)
        if (me.cortex.voxy.client.core.gpu.RenderBackendFactory.get().getType()
                != me.cortex.voxy.client.core.gpu.BackendType.OPENGL) {
            return new Matrix4f(base);
        }

        float nearVoxy = me.cortex.voxy.client.core.rendering.util.LodProjection.nearVoxy(
                Minecraft.getInstance().options.renderDistance().get(),
                VoxyClient.disableSodiumChunkRender());
        return me.cortex.voxy.client.core.rendering.util.LodProjection.compute(base, nearVoxy, 16 * 3000);
    }

    private boolean frexStillHasWork() {
        if (!VoxyClient.isFrexActive()) {
            return false;
        }
        //If frex is running we must tick everything to ensure correctness
        UploadStream.INSTANCE.tick();
        //Done here as is allows less gl state resetup
        this.modelService.tick(100_000_000);
        GlCompat.finish();
        return this.nodeManager.hasWork() || this.renderGen.getTaskCount()!=0 || !this.modelService.areQueuesEmpty();
    }

    public void setRenderDistance(int renderDistance) {
        this.renderDistanceTracker.setRenderDistance(renderDistance);
    }

    public Viewport<?> getViewport() {
        if (IrisUtil.irisShadowActive()) {
            return null;
        }
        return this.viewportSelector.getViewport();
    }





    public void addDebugInfo(List<String> debug) {
        var backend = me.cortex.voxy.client.core.gpu.RenderBackendFactory.get();
        debug.add("Buf/Tex [#/Mb]: [" + backend.getBufferCount() + "/" + (backend.getBufferTotalSize()/1_000_000) + "],[" + backend.getTextureCount() + "/" + (backend.getTextureEstimatedTotalSize()/1_000_000)+"]");
        {
            this.modelService.addDebugData(debug);
            this.renderGen.addDebugData(debug);
            this.nodeManager.addDebug(debug);
            this.pipeline.addDebug(debug);
        }
        {
            TimingStatistics.update();
            debug.add("Voxy frame runtime (millis): " + TimingStatistics.dynamic.pVal() + ", " + TimingStatistics.main.pVal()+ ", " + TimingStatistics.postDynamic.pVal()+ ", " + TimingStatistics.all.pVal());
            debug.add("Extra time: " + TimingStatistics.A.pVal() + ", " + TimingStatistics.B.pVal() + ", " + TimingStatistics.C.pVal() + ", " + TimingStatistics.D.pVal());
            debug.add("Extra 2 time: " + TimingStatistics.E.pVal() + ", " + TimingStatistics.F.pVal() + ", " + TimingStatistics.G.pVal() + ", " + TimingStatistics.H.pVal() + ", " + TimingStatistics.I.pVal());
        }
        PrintfDebugUtil.addToOut(debug);
    }

    public void shutdown() {
        Logger.info("Flushing download stream");
        DownloadStream.INSTANCE.flushWaitClear();
        Logger.info("Shutting down rendering");
        try {
            //Cleanup callbacks
            this.worldIn.setDirtyCallback(null);
            this.worldIn.getMapper().setBiomeCallback(null);
            this.worldIn.getMapper().setStateCallback(null);

            this.nodeManager.stop();

            this.modelService.shutdown();
            this.renderGen.shutdown();
            this.traversal.free();
            this.nodeCleaner.free();

            this.geometryData.free();
            this.chunkBoundRenderer.free();

            this.viewportSelector.free();
        } catch (Exception e) {Logger.error("Error shutting down renderer components", e);}
        Logger.info("Shutting down render pipeline");
        try {this.pipeline.free();} catch (Exception e){Logger.error("Error releasing render pipeline", e);}



        Logger.info("Flushing download stream");
        DownloadStream.INSTANCE.flushWaitClear();

        //Release hold on the world
        this.worldIn.releaseRef();
        Logger.info("Render shutdown completed");
    }

    private static long getGeometryBufferSize() {
        // M9 transitional: on Mac with Metal/Vulkan backend, Apple's frozen GL 4.1
        // doesn't expose GL_MAX_SHADER_STORAGE_BLOCK_SIZE, so Capabilities.INSTANCE.ssboMaxSize
        // is 0 — the bit-magic computation below produces -1024, which is invalid
        // and BasicSectionGeometryData rejects it (must be %8==0). Use the
        // backend-agnostic getter when GL capabilities aren't available.
        long ssboMaxSize = Capabilities.INSTANCE.ssboMaxSize;
        if (ssboMaxSize <= 0) {
            ssboMaxSize = me.cortex.voxy.client.core.gpu.RenderBackendFactory.get().getMaxSSBOSize();
        }
        if (ssboMaxSize <= 0) {
            // Final fallback: a sane 1GB default. Caller may further clamp by
            // available GPU memory below.
            ssboMaxSize = 1L << 30;
        }
        long geometryCapacity = Math.min((1L<<(64-Long.numberOfLeadingZeros(ssboMaxSize-1)))<<1, 1L<<32)-1024/*(1L<<32)-1024*/;
        if (Capabilities.INSTANCE.isIntel) {
            geometryCapacity = Math.max(geometryCapacity, 1L<<30);//intel moment, force min 1gb
        }

        //Limit to available dedicated memory if possible
        if (Capabilities.INSTANCE.canQueryGpuMemory) {
            //512mb less than avalible,
            long limit = Capabilities.INSTANCE.getFreeDedicatedGpuMemory() - (long)(1.5*1024*1024*1024);//1.5gb vram buffer
            // Give a minimum of 512 mb requirement
            limit = Math.max(512*1024*1024, limit);

            geometryCapacity = Math.min(geometryCapacity, limit);
        }
        //geometryCapacity = 1<<28;
        //geometryCapacity = 1<<30;//1GB test
        var override = System.getProperty("voxy.geometryBufferSizeOverrideMB", "");
        if (!override.isEmpty()) {
            geometryCapacity = Long.parseLong(override)*1024L*1024L;
        }
        return geometryCapacity;
    }

    public WorldEngine getEngine() {
        return this.worldIn;
    }
}
