package me.cortex.voxy.client.core.rendering.hierachical;

import it.unimi.dsi.fastutil.ints.Int2IntOpenHashMap;
import me.cortex.voxy.client.RenderStatistics;
import me.cortex.voxy.client.config.VoxyConfig;
import me.cortex.voxy.client.core.gpu.shader.ShaderLoader;
import me.cortex.voxy.client.core.gpu.ComputeEncoder;
import me.cortex.voxy.client.core.gpu.ComputePipelineDesc;
import me.cortex.voxy.client.core.gpu.IGpuBuffer;
import me.cortex.voxy.client.core.gpu.IGpuPipeline;
import me.cortex.voxy.client.core.gpu.IGpuSampler;
import me.cortex.voxy.client.core.gpu.RenderBackend;
import me.cortex.voxy.client.core.gpu.RenderBackendFactory;
import me.cortex.voxy.client.core.gpu.SamplerDesc;
import me.cortex.voxy.client.core.rendering.Viewport;
import me.cortex.voxy.client.core.rendering.building.RenderGenerationService;
import me.cortex.voxy.client.core.rendering.util.DownloadStream;
import me.cortex.voxy.client.core.rendering.util.UploadStream;
import me.cortex.voxy.common.Logger;
import me.cortex.voxy.common.util.MemoryBuffer;
import me.cortex.voxy.common.world.WorldEngine;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Hierarchical occlusion traverser. Walks the LOD octree on the GPU via a
 * compute shader, iterating per LOD level — each iteration's dispatch shape
 * is read indirectly from a metadata buffer the shader itself writes.
 *
 * M9 status: fully migrated onto the {@link RenderBackend} encoder
 * abstraction. Bind once per pass, dispatch indirect from
 * {@link #queueMetaBuffer}, flip-flop source/sink between iterations. The
 * single-int CPU→GPU writes ({@link #addTLN}/{@link #remTLN}) flow through
 * {@link UploadStream}, and counter resets ({@link #doTraversal},
 * {@link #downloadResetRequestQueue}) use {@code IGpuBuffer.zeroRange} —
 * so the class is fully backend-agnostic (M12 chunk 6 prep).
 */
public class HierarchicalOcclusionTraverser {
    public static final boolean HIERARCHICAL_SHADER_DEBUG = System.getProperty("voxy.hierarchicalShaderDebug", "false").equals("true");

    public static final int MAX_REQUEST_QUEUE_SIZE = 50;
    public static final int MAX_QUEUE_SIZE = 200_000;


    private static final int MAX_ITERATIONS = WorldEngine.MAX_LOD_LAYER + 1;

    /**
     * {@code VOXY_TRAV_STATS=1} logs the traversal's per-frame decision counters.
     *
     * <p><b>Why every frame, not sampled.</b> The symptom this exists for is a ONE-FRAME flicker. A
     * once-a-second readout cannot see a one-frame event at all -- it would sample 60 frames, miss the
     * bad one, and report that nothing happened, which is exactly the false-negative this project keeps
     * re-learning to avoid. So this logs the whole series and the ANALYSIS looks for a step in it.
     *
     * <p><b>How to read it.</b> Four decisions can remove a node, and all four are now counted per LOD
     * level: culled by frustum, culled by occlusion, handed off to children (descend), and drawn
     * (enqueued). Geometry that is missing can only come from one of two places:
     * <ul>
     *   <li>A counter STEPS for a frame while the others hold -- a gating flicker, and the counter that
     *       stepped names the gate. Correlate it with the frame the user reports the flicker on.</li>
     *   <li>All four hold steady while geometry is visibly absent -- nothing is being culled, so the
     *       nodes are being DRAWN SOMEWHERE ELSE. That is a placement bug, not a cull bug, and this
     *       series is what separates the two. Every "explanation" offered for these symptoms so far has
     *       assumed a cull; this is the measurement that would refuse that assumption.</li>
     * </ul>
     *
     * <p>Buffer layout is five consecutive {@code uint[MAX_ITERATIONS]} arrays, in the order the shader
     * declares them: visited, enqueued, frustumCulled, hizCulled, descended.
     */
    private static final boolean TRAV_STATS = "1".equals(System.getenv("VOXY_TRAV_STATS"));

    /**
     * {@code VOXY_LOD_CHILD_READY=0} restores the old behaviour: a node descends into children that
     * cannot draw yet, leaving a hole until their meshes arrive.
     *
     * <p>ON by default. The user-reported symptom it fixes is a chunk blinking out as it crosses from
     * the second-finest detail level to the finest -- the last subdivision, and the one where the wait
     * for a mesh is longest (level-0 nodes are the most numerous and most expensive) at the closest
     * range, so it is the most visible. See {@code childrenCanDraw} in traversal_dev.comp.
     *
     * <p>The escape hatch exists because the previous attempt at this branch broke the LOD entirely and
     * this one touches the same lines: if the LOD sticks at its coarsest level, set this to 0.
     */
    private static final boolean CHILD_READY = !"0".equals(System.getenv("VOXY_LOD_CHILD_READY"));


    private static long travStatsFrame = 0;

    /** Reads the six arrays back and logs one line. Called from the download callback. */
    private static void logTraversalStats(long addr) {
        final String[] names = {"visited", "enqueued", "frustumCulled", "hizCulled", "descended",
                "notReady"};
        StringBuilder sb = new StringBuilder("[Metal-TRAV f=").append(travStatsFrame++).append("] ");
        for (int a = 0; a < names.length; a++) {
            int total = 0;
            for (int i = 0; i < MAX_ITERATIONS; i++) {
                total += MemoryUtil.memGetInt(addr + (a * MAX_ITERATIONS + i) * 4L);
            }
            sb.append(names[a]).append('=').append(total).append("  ");
        }
        // Per-level breakdown for the two CULLS only: "the far sections vanish" is a statement about
        // LOD level, and the totals alone cannot confirm or refute it.
        // visitedByLevel too, not just the totals: a cull count is meaningless without its denominator.
        // The traversal only visits a node whose parent passed, so framumCulled/visited is a LOCAL rate
        // and the useful question -- "is the frustum rejecting as much of the outermost ring as it
        // should?" -- cannot be answered from totals at all. Absolute counts actively mislead here:
        // frustumCulledByLevel=[210,103,49,26,30] makes the coarsest level look least-culled, when it is
        // simply the least-visited.
        for (int a : new int[]{0, 2, 3, 5}) {
            sb.append("| ").append(names[a]).append("ByLevel=[");
            for (int i = 0; i < MAX_ITERATIONS; i++) {
                if (i > 0) sb.append(',');
                sb.append(MemoryUtil.memGetInt(addr + (a * MAX_ITERATIONS + i) * 4L));
            }
            sb.append("] ");
        }
        // The request budget, on the same line as `notReady`, because they are the same question.
        // `requestSize` is what the traversal was ALLOWED to ask for this frame; `tasks` is the mesh
        // queue depth that decided it. When tasks >= TARGET_COUNT (4000) the quadratic fillness term
        // goes to zero and the budget collapses to REQUEST_FLOOR -- 8, which is exactly ONE
        // level-1 -> level-0 refinement. So a persistently saturated queue pins LOD refinement at one
        // node per frame no matter how fast the camera moves, and the terrain that is not refined yet
        // is terrain that is missing from the screen. Read this next to `notReady`.
        sb.append("| reqBudget=").append(lastRequestSize)
          .append(" tasks=").append(lastTaskCount);
        me.cortex.voxy.common.Logger.info(sb.toString());
    }

    /** Published by {@link #uploadUniform} for {@link #logTraversalStats}; diagnostic only. */
    private static volatile int lastRequestSize = -1;
    private static volatile int lastTaskCount = -1;
    private static final int LOCAL_WORK_SIZE_BITS = 5;
    private static final int LOCAL_WORK_SIZE = 1 << LOCAL_WORK_SIZE_BITS;

    /** UBO binding used to push queueIdx (see queue.glsl). */
    private static final int PUSH_BINDING = 14;

    private final AsyncNodeManager nodeManager;
    private final NodeCleaner nodeCleaner;
    private final RenderGenerationService meshGen;

    private final IGpuBuffer requestBuffer;

    private final IGpuBuffer nodeBuffer;
    private final IGpuBuffer uniformBuffer = RenderBackendFactory.get().createBuffer(1024).zero();
    private final IGpuBuffer statisticsBuffer = RenderBackendFactory.get().createBuffer(1024).zero();


    private int topNodeCount;

    /** Layer-B diag: read by AbstractRenderPipeline to log per-frame topNodeCount + firstDispatchSize. */
    public int getTopNodeCount() {
        return this.topNodeCount;
    }
    private final Int2IntOpenHashMap topNode2idxMapping = new Int2IntOpenHashMap();//Used to store mapping from TLN to array index
    private final int[] idx2topNodeMapping = new int[MAX_QUEUE_SIZE];//Used to map idx to TLN id
    private final IGpuBuffer topNodeIds = RenderBackendFactory.get().createBuffer(MAX_QUEUE_SIZE * 4).zero();
    private final IGpuBuffer queueMetaBuffer = RenderBackendFactory.get().createBuffer(4 * 4 * MAX_ITERATIONS).zero();
    private final IGpuBuffer scratchQueueA = RenderBackendFactory.get().createBuffer(MAX_QUEUE_SIZE * 4).zero();
    private final IGpuBuffer scratchQueueB = RenderBackendFactory.get().createBuffer(MAX_QUEUE_SIZE * 4).zero();

    private static int BINDING_COUNTER = 1;
    private static final int SCENE_UNIFORM_BINDING = BINDING_COUNTER++;
    private static final int REQUEST_QUEUE_BINDING = BINDING_COUNTER++;
    private static final int RENDER_QUEUE_BINDING = BINDING_COUNTER++;
    private static final int NODE_DATA_BINDING = BINDING_COUNTER++;
    /** Reserved — old GL build used this as a uniform location for queueIdx. queue.glsl now pushes via PUSH_BINDING UBO. */
    private static final int NODE_QUEUE_INDEX_BINDING_RESERVED = BINDING_COUNTER++;
    private static final int NODE_QUEUE_META_BINDING = BINDING_COUNTER++;
    private static final int NODE_QUEUE_SOURCE_BINDING = BINDING_COUNTER++;
    private static final int NODE_QUEUE_SINK_BINDING = BINDING_COUNTER++;
    private static final int RENDER_TRACKER_BINDING = BINDING_COUNTER++;
    private static final int STATISTICS_BUFFER_BINDING = BINDING_COUNTER++;

    /** HiZ sampled-texture slot (texture-unit equivalent). */
    private static final int HIZ_BINDING = 0;

    private final RenderBackend backend = RenderBackendFactory.get();
    private final IGpuSampler hizSampler = this.backend.createSampler(SamplerDesc.builder()
            .filter(SamplerDesc.Filter.NEAREST, SamplerDesc.Filter.NEAREST)
            .mipFilter(SamplerDesc.MipFilter.NEAREST)
            .wrap(SamplerDesc.Wrap.CLAMP_TO_EDGE, SamplerDesc.Wrap.CLAMP_TO_EDGE)
            .label("hizSampler")
            .build());

    private final IGpuPipeline traversal;



    public HierarchicalOcclusionTraverser(AsyncNodeManager nodeManager, NodeCleaner nodeCleaner, RenderGenerationService meshGen) {
        this.nodeCleaner = nodeCleaner;
        this.nodeManager = nodeManager;
        this.meshGen = meshGen;
        this.requestBuffer = RenderBackendFactory.get().createBuffer(MAX_REQUEST_QUEUE_SIZE * 8L + 8).zero();
        this.nodeBuffer = RenderBackendFactory.get().createBuffer(nodeManager.maxNodeCount * 16L).fill(-1);

        this.traversal = this.backend.createComputePipeline(new ComputePipelineDesc(
                ShaderLoader.parse("voxy:lod/hierarchical/traversal_dev.comp"),
                traversalDefines(),
                null, null,
                LOCAL_WORK_SIZE, 1, 1,
                "HierarchicalOcclusionTraverser.traversal"));

        this.topNode2idxMapping.defaultReturnValue(-1);
        this.nodeManager.setTLNAddRemoveCallbacks(this::addTLN, this::remTLN);
    }

    private static Map<String, String> traversalDefines() {
        var m = new LinkedHashMap<String, String>();
        if (HIERARCHICAL_SHADER_DEBUG) m.put("DEBUG", "");
        m.put("MAX_ITERATIONS", Integer.toString(MAX_ITERATIONS));
        m.put("LOCAL_SIZE_BITS", Integer.toString(LOCAL_WORK_SIZE_BITS));
        m.put("MAX_REQUEST_QUEUE_SIZE", Integer.toString(MAX_REQUEST_QUEUE_SIZE));
        // The occlusion test is enabled, and screenspace.glsl is written for ONE depth convention:
        // this frame's, which is reverse-Z (near 1, far 0). The pyramid is therefore min-reduced
        // (hiz/blit.fsh) and the box test is `tileFarthest > boxNearest`. There is no define to inject
        // for that any more -- the other convention's branches are deleted, because this project has
        // one backend and it is not going to grow a second one.
        m.put("HIZ_BINDING", Integer.toString(HIZ_BINDING));
        m.put("SCENE_UNIFORM_BINDING", Integer.toString(SCENE_UNIFORM_BINDING));
        m.put("REQUEST_QUEUE_BINDING", Integer.toString(REQUEST_QUEUE_BINDING));
        m.put("RENDER_QUEUE_BINDING", Integer.toString(RENDER_QUEUE_BINDING));
        m.put("NODE_DATA_BINDING", Integer.toString(NODE_DATA_BINDING));
        m.put("NODE_QUEUE_META_BINDING", Integer.toString(NODE_QUEUE_META_BINDING));
        m.put("NODE_QUEUE_SOURCE_BINDING", Integer.toString(NODE_QUEUE_SOURCE_BINDING));
        m.put("NODE_QUEUE_SINK_BINDING", Integer.toString(NODE_QUEUE_SINK_BINDING));
        m.put("RENDER_TRACKER_BINDING", Integer.toString(RENDER_TRACKER_BINDING));
        m.put("PUSH_BINDING", Integer.toString(PUSH_BINDING));
        // THREE gates, all of which must be open together: this define, the setBuffer in the dispatch
        // loop, and the readback. Only opening some of them is how a diagnostic returns a clean,
        // confident, entirely empty series -- which is exactly what happened here: the first run of
        // these counters produced 12 368 frames of zeros, which reads as "nothing is being culled"
        // rather than "the instrument is not connected".
        if (RenderStatistics.enabled || TRAV_STATS) {
            m.put("HAS_STATISTICS", "");
            m.put("STATISTICS_BUFFER_BINDING", Integer.toString(STATISTICS_BUFFER_BINDING));
        }
        // VOXY_LOD_CHILD_READY=0 restores the old descend behaviour, where a node hands off to
        // children that have no meshes yet and leaves a hole until they arrive. The escape hatch.
        if (!CHILD_READY) {
            m.put("VOXY_LOD_NO_CHILD_READY", "");
        }
        return m;
    }

    private void addTLN(int id) {
        int aid = this.topNodeCount++;//Increment buffer
        if (this.topNodeCount > this.topNodeIds.size() / 4) {
            throw new IllegalStateException("Top level node count greater than capacity");
        }

        // M12 chunk 6 prep: route through UploadStream (cross-backend) instead
        // of raw glBindBuffer + nglBufferSubData. On Metal the previous raw GL
        // pattern would have written into MC's GL context against a meaningless
        // ID (MetalBuffer.id() is a Metal-internal handle, not a GL buffer name).
        long ptr = UploadStream.INSTANCE.upload(this.topNodeIds, aid * 4L, 4);
        MemoryUtil.memPutInt(ptr, id);
        UploadStream.INSTANCE.commit();

        if (this.topNode2idxMapping.put(id, aid) != -1) {
            throw new IllegalStateException();
        }
        this.idx2topNodeMapping[aid] = id;
    }

    private void remTLN(int id) {
        int idx = this.topNode2idxMapping.remove(id);
        this.topNodeCount--;
        if (idx == -1) {
            throw new IllegalStateException();
        }
        if (idx == this.topNodeCount) return;

        int endTLNId = this.idx2topNodeMapping[this.topNodeCount];
        this.idx2topNodeMapping[idx] = endTLNId;
        if (this.topNode2idxMapping.put(endTLNId, idx) == -1)
            throw new IllegalStateException();

        long ptr = UploadStream.INSTANCE.upload(this.topNodeIds, idx * 4L, 4);
        MemoryUtil.memPutInt(ptr, endTLNId);
        UploadStream.INSTANCE.commit();
    }

    // Frustum cull on Metal — DEFAULT ON with a world-space MARGIN (2026-05-26
    // round 7). The cull recovers ~2x sections (60-80 fps vs 15-25 with it off)
    // and, with a margin, keeps the OPAQUE TERRAIN solid: testing showed margin
    // 96 = solid terrain at 60-80 fps (the user only saw flicker at margin 0).
    // The TRANSLUCENT WATER still flickers under the cull — that is deferred per
    // the user (fps first). So: cull ON by default, margin keeps terrain solid.
    //   default (Metal)        -> cull ON, margin 96, terrain solid, ~60-80 fps
    //                             (water still flickers — deferred)
    //   VOXY_LOD_FRUSTUM_MARGIN -> tune: RAISE if any terrain flicker appears,
    //                              LOWER (64/48/32) for even more fps while the
    //                              terrain stays solid
    //   VOXY_LOD_NO_CULL=1      -> disable the cull (pass-all): zero flicker on
    //                              everything incl. water, but ~15-25 fps
    // TODO: the no-flicker-AND-culled-water fix needs a STABLE cull frustum
    // (jitter-free planes) — then the water margin could shrink without flicker.
    private static final boolean CULL_DISABLED =
            "1".equals(System.getenv("VOXY_LOD_NO_CULL"));
    private static final float FRUSTUM_MARGIN = parseFrustumMargin();
    /** Metal-only floor for the per-frame child-request budget (see
     *  uploadUniform). VOXY_HOT_REQUEST_FLOOR tunes; 0 restores the old
     *  "0 requests while the mesh queue is saturated" cliff. */
    private static final int REQUEST_FLOOR = parseRequestFloor();

    private static int parseRequestFloor() {
        String v = System.getenv("VOXY_HOT_REQUEST_FLOOR");
        if (v == null || v.isBlank()) return 8;
        try {
            return Math.max(0, Math.min(MAX_REQUEST_QUEUE_SIZE, Integer.parseInt(v.trim())));
        } catch (NumberFormatException e) {
            return 8;
        }
    }
    private static boolean frustumModeLogged = false;
    private static long frustumFrameCount = 0;
    private static long frustumNanCount = 0;

    private static float parseFrustumMargin() {
        // Margin that keeps the opaque terrain solid under the cull. 96 is the
        // tested terrain-solid value; lower for more fps, raise if terrain flickers.
        String v = System.getenv("VOXY_LOD_FRUSTUM_MARGIN");
        if (v == null) return 96.0f;
        try {
            return Math.max(0.0f, Float.parseFloat(v.trim()));
        } catch (NumberFormatException e) {
            return 96.0f;
        }
    }

    private static void setFrustum(Viewport<?> viewport, long ptr) {
        // On Metal, cull by default (with margin) unless the user disables it
        // (VOXY_LOD_NO_CULL=1 -> pass-all, zero flicker, low fps).
        boolean doCull = !CULL_DISABLED;
        if (!doCull) {
            if (!frustumModeLogged) {
                frustumModeLogged = true;
                me.cortex.voxy.common.Logger.info(
                        "[Metal] HOT frustum cull OFF (VOXY_LOD_NO_CULL=1): zero flicker everywhere, ~15-25 fps");
            }
            for (int i = 0; i < 6; i++) {
                MemoryUtil.memPutFloat(ptr,      0.0f);    // nx
                MemoryUtil.memPutFloat(ptr + 4,  0.0f);    // ny
                MemoryUtil.memPutFloat(ptr + 8,  0.0f);    // nz
                MemoryUtil.memPutFloat(ptr + 12, 1.0e30f); // w (n=0, w>=0 -> testPlane always true)
                ptr += 4 * 4;
            }
            return;
        }
        float margin = FRUSTUM_MARGIN;
        if (!frustumModeLogged) {
            frustumModeLogged = true;
            me.cortex.voxy.common.Logger.info(
                    "[Metal] HOT frustum cull ON, margin=" + margin + " blocks (terrain solid; water still flickers — deferred; VOXY_LOD_FRUSTUM_MARGIN to tune, VOXY_LOD_NO_CULL=1 to disable)");
        }
        // [Metal-FRUSTUM] diagnostic + NaN GUARD (2026-05-26). The diag run
        // showed `proj m00=NaN` → NaN frustum planes. NaN planes under Metal's
        // shader fast-math (which assumes no NaN) make the cull test UNDEFINED →
        // erratic / direction-dependent edge dropping (and likely the original
        // flicker). GUARD: if the projection is NaN this frame, upload pass-all
        // (don't cull) so the cull can never go undefined. Also count NaN frames
        // and log m00/m11 every 600 frames to learn whether the NaN is a
        // transient first-frame thing or persistent, and — when valid — whether
        // the frustum is genuinely too narrow (impliedAspect vs screen aspect).
        frustumFrameCount++;
        boolean projNaN = Float.isNaN(viewport.projection.m00())
                || Float.isNaN(viewport.frustumPlanes[0].x);
        if (projNaN) frustumNanCount++;
        if ((frustumFrameCount % 600) == 1) {
            float m00 = viewport.projection.m00();
            float m11 = viewport.projection.m11();
            var win = net.minecraft.client.Minecraft.getInstance().getWindow();
            var L = viewport.frustumPlanes[0];
            var R = viewport.frustumPlanes[1];
            me.cortex.voxy.common.Logger.info(String.format(
                    "[Metal-FRUSTUM f=%d] projNaN=%s nanFrames=%d | m00=%.4f m11=%.4f impliedAspect=%.3f | viewport=%dx%d(%.3f) window=%dx%d(%.3f) | L=(%.3f,%.3f,%.3f,%.1f) R=(%.3f,%.3f,%.3f,%.1f)",
                    frustumFrameCount, projNaN, frustumNanCount,
                    m00, m11, java.lang.Math.abs(m11 / m00),
                    viewport.width, viewport.height,
                    viewport.height == 0 ? 0.0 : (double) viewport.width / viewport.height,
                    win.getWidth(), win.getHeight(),
                    win.getHeight() == 0 ? 0.0 : (double) win.getWidth() / win.getHeight(),
                    L.x, L.y, L.z, L.w, R.x, R.y, R.z, R.w));
        }
        if (projNaN) {
            for (int i = 0; i < 6; i++) {
                MemoryUtil.memPutFloat(ptr,      0.0f);
                MemoryUtil.memPutFloat(ptr + 4,  0.0f);
                MemoryUtil.memPutFloat(ptr + 8,  0.0f);
                MemoryUtil.memPutFloat(ptr + 12, 1.0e30f);
                ptr += 4 * 4;
            }
            return;
        }
        for (int i = 0; i < 6; i++) {
            var plane = viewport.frustumPlanes[i];
            float nx = plane.x, ny = plane.y, nz = plane.z, w = plane.w;
            if (margin != 0.0f) {
                float len = (float) Math.sqrt(nx * nx + ny * ny + nz * nz);
                w += margin * len;
            }
            MemoryUtil.memPutFloat(ptr,      nx);
            MemoryUtil.memPutFloat(ptr + 4,  ny);
            MemoryUtil.memPutFloat(ptr + 8,  nz);
            MemoryUtil.memPutFloat(ptr + 12, w);
            ptr += 4 * 4;
        }
    }

    private void uploadUniform(Viewport<?> viewport) {
        long ptr = UploadStream.INSTANCE.upload(this.uniformBuffer, 0, 1024);

        viewport.MVP.getToAddress(ptr); ptr += 4 * 4 * 4;
        viewport.section.getToAddress(ptr); ptr += 4 * 3;
        MemoryUtil.memPutInt(ptr, viewport.hiZBuffer.getPackedLevels()); ptr += 4;
        viewport.innerTranslation.getToAddress(ptr); ptr += 4 * 3;

        final float screenspaceAreaDecreasingSize = VoxyConfig.CONFIG.subDivisionSize * VoxyConfig.CONFIG.subDivisionSize;
        MemoryUtil.memPutFloat(ptr, (float) (screenspaceAreaDecreasingSize) / (viewport.width * viewport.height)); ptr += 4;
        setFrustum(viewport, ptr); ptr += 4 * 4 * 6;
        MemoryUtil.memPutInt(ptr, (int) (viewport.getRenderList().size() / 4 - 1)); ptr += 4;
        MemoryUtil.memPutInt(ptr, this.nodeCleaner.visibilityId); ptr += 4;

        {
            final double TARGET_COUNT = 4000;
            double iFillness = Math.max(0, (TARGET_COUNT - this.meshGen.getTaskCount()) / TARGET_COUNT);
            iFillness = Math.pow(iFillness, 2);
            int requestSize = (int) Math.ceil(iFillness * MAX_REQUEST_QUEUE_SIZE);
            // 2026-07-03 (Metal): FLOOR the per-frame child-request budget
            // instead of letting it hit 0 while the mesh queue holds >4000
            // tasks. At world join the queue saturates with tasks BLOCKED on
            // model bakes (IdNotYetComputedException retries, not real
            // meshing throughput), so the quadratic throttle shut off LOD
            // refinement exactly while the giant coarse parents were on
            // screen — and each new request is also what seeds the bakery
            // with the block ids it still needs (RenderGenerationService
            // computeAndRequestRequiredModels). A small floor keeps
            // discovery trickling; the bakery warmup burst
            // (VoxyRenderSystem.computeBakeBudgetNs) drains the resulting
            // bake demand. VOXY_HOT_REQUEST_FLOOR tunes it (0 restores the
            // old cliff).
            requestSize = Math.max(REQUEST_FLOOR, requestSize);
            final int budget = Math.max(0, Math.min(MAX_REQUEST_QUEUE_SIZE, requestSize));
            lastRequestSize = budget;
            lastTaskCount = this.meshGen.getTaskCount();
            MemoryUtil.memPutInt(ptr, budget); ptr += 4;
        }
    }

    public void doTraversal(Viewport<?> viewport) {
        this.uploadUniform(viewport);

        // PrintfDebugUtil.bind() (raw glBindBufferBase for the printf debug SSBO,
        // gated on -Dvoxy.enableShaderDebugPrintf=true, default off) is gone with the
        // GL path, so the printf feature can no longer bind its output buffer.
        // tick()/addToOut() and the shader-source injection are unaffected; reviving
        // it would mean a ComputeEncoder.setBuffer inside this pass.

        if (RenderStatistics.enabled || TRAV_STATS) {
            this.statisticsBuffer.zero();
        }

        //Clear the render output counter
        viewport.getRenderList().zeroRange(0, 4);

        this.traverseInternal(viewport);
        this.downloadResetRequestQueue();

        if (RenderStatistics.enabled || TRAV_STATS) {
            DownloadStream.INSTANCE.download(this.statisticsBuffer, down -> {
                for (int i = 0; i < MAX_ITERATIONS; i++) {
                    RenderStatistics.hierarchicalTraversalCounts[i] = MemoryUtil.memGetInt(down.address + i * 4L);
                }
                for (int i = 0; i < MAX_ITERATIONS; i++) {
                    RenderStatistics.hierarchicalRenderSections[i] = MemoryUtil.memGetInt(down.address + MAX_ITERATIONS * 4L + i * 4L);
                }
                if (TRAV_STATS) {
                    logTraversalStats(down.address);
                }
            });
        }
    }

    private void traverseInternal(Viewport<?> viewport) {
        // (The Mesa glPixelStorei unpack-state reset that used to run here was GL-only: Metal
        // has no such global unpack state.)

        int firstDispatchSize = (this.topNodeCount + LOCAL_WORK_SIZE - 1) >> LOCAL_WORK_SIZE_BITS;

        {
            //TODO:FIXME: THIS IS BULLSHIT BY INTEL need to fix the clearing
            long ptr = UploadStream.INSTANCE.upload(this.queueMetaBuffer, 0, 16 * MAX_ITERATIONS);
            MemoryUtil.memPutInt(ptr +  0, firstDispatchSize);
            MemoryUtil.memPutInt(ptr +  4, 1);
            MemoryUtil.memPutInt(ptr +  8, 1);
            MemoryUtil.memPutInt(ptr + 12, this.topNodeCount);
            for (int i = 1; i < MAX_ITERATIONS; i++) {
                MemoryUtil.memPutInt(ptr + (i * 16) +  0, 0);
                MemoryUtil.memPutInt(ptr + (i * 16) +  4, 1);
                MemoryUtil.memPutInt(ptr + (i * 16) +  8, 1);
                MemoryUtil.memPutInt(ptr + (i * 16) + 12, 0);
            }
            UploadStream.INSTANCE.commit();
        }

        {
            // METAL: one compute encoder PER ITERATION. Inside a single Metal
            // compute encoder, memoryBarrier(scope:) orders shader memory
            // access between dispatches but does NOT reliably fence the
            // command processor's INDIRECT-ARGUMENT fetch for the next
            // dispatchIndirect — iteration N+1's group count can be read
            // before iteration N finished writing it, truncating the octree
            // walk at a random depth. That is the static-camera renderList
            // collapse (min=14 / max=2827 in the [Metal-FLICKER] logs) = the
            // persistent underwater strobe. Encoder boundaries ARE full
            // hazard-tracked barriers on Metal, covering indirect args.
            // VOXY_HOT_SERIALIZE=1 additionally submits+waits per iteration
            // (diagnostic only — costs up to 7 waits/frame).
            try (MemoryStack stack = MemoryStack.stackPush()) {
                long pushAddr = stack.nmalloc(4);
                for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
                    try (ComputeEncoder encoder = this.backend.beginComputePass()) {
                        encoder.setPipeline(this.traversal);
                        encoder.setBuffer(SCENE_UNIFORM_BINDING, this.uniformBuffer, 0);
                        encoder.setBuffer(REQUEST_QUEUE_BINDING, this.requestBuffer, 0);
                        encoder.setBuffer(RENDER_QUEUE_BINDING, viewport.getRenderList(), 0);
                        encoder.setBuffer(NODE_DATA_BINDING, this.nodeBuffer, 0);
                        encoder.setBuffer(NODE_QUEUE_META_BINDING, this.queueMetaBuffer, 0);
                        encoder.setBuffer(RENDER_TRACKER_BINDING, this.nodeCleaner.visibilityBuffer, 0);
                        // TRAV_STATS must be here as well as at the readback. Gating only the readback
                        // left the buffer bound to nothing, so the shader's atomicAdds went to an
                        // unbound slot and the readback returned a buffer that was zeroed every frame
                        // and never written -- 12 368 frames of zeros that looked like "no culling is
                        // happening" rather than "the instrument is not connected". A diagnostic with
                        // two gates must have both of them opened by the same switch.
                        //
                        // (They are exclusive, incidentally: the traversal's own bindings run 1-9 and
                        // this one is 10, so an unbound statistics slot writes nowhere rather than over
                        // something that matters. The two classes define the constant separately --
                        // MDICSectionRenderer's is 8 for cmdgen -- which is fine because they are
                        // separate shader compiles, but it is not obvious and is worth knowing before
                        // changing either.)
                        if (RenderStatistics.enabled || TRAV_STATS) {
                            encoder.setBuffer(STATISTICS_BUFFER_BINDING, this.statisticsBuffer, 0);
                        }
                        encoder.setTexture(HIZ_BINDING, viewport.hiZBuffer.getHizTexture());
                        encoder.setSampler(HIZ_BINDING, this.hizSampler);

                        MemoryUtil.memPutInt(pushAddr, iter);
                        encoder.setBytes(PUSH_BINDING, pushAddr, 4);
                        IGpuBuffer source = iter == 0 ? this.topNodeIds
                                : ((iter & 1) == 0 ? this.scratchQueueA : this.scratchQueueB);
                        IGpuBuffer sink = ((iter & 1) == 0 ? this.scratchQueueB : this.scratchQueueA);
                        encoder.setBuffer(NODE_QUEUE_SOURCE_BINDING, source, 0);
                        encoder.setBuffer(NODE_QUEUE_SINK_BINDING, sink, 0);

                        if (iter == 0) {
                            encoder.dispatch(firstDispatchSize, 1, 1);
                        } else {
                            encoder.dispatchIndirect(this.queueMetaBuffer, iter * 4L * 4);
                        }
                    }
                    if (HOT_SERIALIZE) {
                        this.backend.submit();
                    }
                }
            }
        }
    }

    /** Diagnostic: submit+wait after every HOT iteration (Metal). */
    private static final boolean HOT_SERIALIZE = "1".equals(System.getenv("VOXY_HOT_SERIALIZE"));


    private void downloadResetRequestQueue() {
        if (this.requestBuffer instanceof me.cortex.voxy.client.core.metal.MetalBuffer metalBuffer) {
            // Metal encodes traversal into the backend's active command buffer,
            // while DownloadStream uses a separate blit command buffer. If we
            // schedule a DownloadStream read before submitting traversal, the
            // readback sees the previous frame's zeroed queue and no child LOD
            // requests are ever created. Shared-storage Metal buffers are CPU
            // readable after submit(), so read the queue directly and then
            // clear it for the next frame.
            this.backend.submit();
            this.forwardDownloadResult(metalBuffer.getContentsPtr(), this.requestBuffer.size());
            this.requestBuffer.zeroRange(0, 4);
            return;
        }

        DownloadStream.INSTANCE.download(this.requestBuffer, this::forwardDownloadResult);
        // M12 chunk 6 prep: cross-backend zero (was raw glBindBuffer +
        // nglBufferSubData(null) which is UB-on-strict-drivers and outright
        // broken on Metal where the buffer id isn't a GL name).
        this.requestBuffer.zeroRange(0, 4);
    }

    private void forwardDownloadResult(long ptr, long size) {
        int count = MemoryUtil.memGetInt(ptr); ptr += 8;
        if (count < 0 || count > 50000) {
            Logger.error(new IllegalStateException("Count unexpected extreme value: " + count + " things may get weird"));
            return;
        }
        if (count > (this.requestBuffer.size() >> 3) - 1) {
            count = (int) ((this.requestBuffer.size() >> 3) - 1);
            MemoryUtil.memPutInt(ptr - 8, count);
        }
        if (count != 0) {
            this.nodeManager.submitRequestBatch(new MemoryBuffer(count * 8L + 8).cpyFrom(ptr - 8));
        }
    }

    public IGpuBuffer getNodeBuffer() {
        return this.nodeBuffer;
    }

    public void free() {
        this.traversal.close();
        this.requestBuffer.free();
        this.nodeBuffer.free();
        this.uniformBuffer.free();
        this.statisticsBuffer.free();
        this.queueMetaBuffer.free();
        this.topNodeIds.free();
        this.scratchQueueA.free();
        this.scratchQueueB.free();
        this.hizSampler.close();
    }
}
