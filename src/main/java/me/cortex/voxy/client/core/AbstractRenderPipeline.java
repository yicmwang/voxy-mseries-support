package me.cortex.voxy.client.core;

import me.cortex.voxy.client.RenderStatistics;
import me.cortex.voxy.client.TimingStatistics;
import me.cortex.voxy.client.VoxyClient;
import me.cortex.voxy.client.core.model.ModelBakerySubsystem;
import me.cortex.voxy.client.core.rendering.Viewport;
import me.cortex.voxy.client.core.rendering.hierachical.AsyncNodeManager;
import me.cortex.voxy.client.core.rendering.hierachical.HierarchicalOcclusionTraverser;
import me.cortex.voxy.client.core.rendering.hierachical.NodeCleaner;
import me.cortex.voxy.client.core.rendering.section.backend.AbstractSectionRenderer;
import me.cortex.voxy.client.core.rendering.util.DownloadStream;
import me.cortex.voxy.common.Logger;
import me.cortex.voxy.common.util.TrackedObject;
import org.joml.Matrix4f;
import org.lwjgl.system.MemoryUtil;

import java.util.List;
import java.util.function.BooleanSupplier;


public abstract class AbstractRenderPipeline extends TrackedObject {
    private final BooleanSupplier frexStillHasWork;

    private final AsyncNodeManager nodeManager;
    private final NodeCleaner nodeCleaner;
    private final HierarchicalOcclusionTraverser traversal;

    protected AbstractSectionRenderer<?,?> sectionRenderer;



    protected final boolean deferTranslucency;

    /**
     * Whether this pipeline wants environmental fog mixed into LOD terrain.
     * On GL the fog pass lives in {@link #finish(Viewport, int, int, int)};
     * on Metal it's applied per-fragment by quads.frag's USE_ENV_FOG branch
     * (M13 chunk 5) using fog params packed into the SceneUniform SSBO.
     * Subclasses that drive an env-fog config flag override this. Defaults
     * to false so unrelated pipelines (e.g. Iris) don't inject the define.
     */
    public boolean useEnvFog() {
        return false;
    }


    protected AbstractRenderPipeline(AsyncNodeManager nodeManager, NodeCleaner nodeCleaner, HierarchicalOcclusionTraverser traversal, BooleanSupplier frexSupplier, boolean deferTranslucency) {
        this.frexStillHasWork = frexSupplier;
        this.nodeManager = nodeManager;
        this.nodeCleaner = nodeCleaner;
        this.traversal = traversal;
        this.deferTranslucency = deferTranslucency;
    }

    //Allows pipelines to configure model baking system
    public void setupExtraModelBakeryData(ModelBakerySubsystem modelService) {}

    public final void setSectionRenderer(AbstractSectionRenderer<?,?> sectionRenderer) {//Stupid java ordering not allowing something pre super
        if (this.sectionRenderer != null) throw new IllegalStateException();
        this.sectionRenderer = sectionRenderer;
    }

    //Called before the pipeline starts running, used to update uniforms etc
    public void preSetup(Viewport<?> viewport) {

    }

    // The four GL pipeline hooks (setup / postOpaquePreTranslucent / setupAndBindOpaque /
    // setupAndBindTranslucent) stood here. They existed to drive the GL render pipeline's
    // framebuffers, SSAO compute pass and final blit, all of which are deleted; the Metal
    // path builds its pass in runPipelineMetal and needs none of them.

    /** IOSurface bridge for the Metal render path. Lazy-allocated on first non-GL frame. */
    /**
     * Whole-frame Metal: the frame's own attachments, adapted to IGpuTexture. When these are bound
     * Voxy renders straight into Metallum's pass and there is no bridge and no composite.
     */
    private me.cortex.voxy.client.core.metal.MetallumAttachmentTexture metallumColor;
    private me.cortex.voxy.client.core.metal.MetallumAttachmentTexture metallumDepth;
    /** One-shot guard for the [Metal-DEPTHFMT] report; see where the depth attachment is refreshed. */
    private boolean depthFormatLogged;
    /** One-shot guard for [Metal-HIZBUILD]: which branch the pyramid build took, and why. */
    private boolean hizBuildLogged;
    /** One-shot guard for [Metal-PASSSHAPE]. */
    private boolean passShapeLogged;

    /**
     * The Hi-Z pyramid build. <b>OFF by default as of 2026-09-22</b>: the cull built on it removes
     * terrain it should keep, so the pyramid is left zero-filled — whose every box answers "not
     * occluded" — and the traversal's Hi-Z test becomes a no-op.
     *
     * <p><b>Why the default flipped, and the exact evidence.</b> The owner reported "entire LOD
     * sections randomly disappearing depending on camera angle": large regions of DISTANT LOD missing
     * with sky through them, and detached slabs of terrain floating where no terrain should be.
     * Measured against `VOXY_HIZ_BUILD=0` on the same build, same camera and same drift, judged from
     * the screenshots at an honest frame rate with no instrumentation in either arm: with the pyramid
     * built the far field is missing and fragmented; with it left zero-filled <b>the run "works
     * great"</b> (owner's words). Both Hi-Z callers are no-ops in that arm — the traversal's test and,
     * until it was deleted the same day, the per-section cull's — so this is the Hi-Z family as a
     * whole, not one caller.
     *
     * <p>It also fits the onset the owner described: "it looked good for the first 30 seconds, then
     * exploded". At world join the depth buffer is nearly empty, so the pyramid holds no occluders and
     * the cull removes nothing; the failure needs the pyramid to fill in first.
     *
     * <p>{@code VOXY_HIZ_BUILD=1} re-enables it, and that is the arm to use while fixing it — the
     * defect is in the test or in the pyramid's content, not in the idea of occlusion culling. The
     * history is worth keeping: the pyramid was once attached as a DEPTH attachment while its blit
     * pipeline declares a disabled depth test and an R32F COLOUR format, so the blit's output was
     * dropped and nothing ever wrote it; on top of that the source texture was RGBA8, and reverse-Z
     * LOD depth (~1e-4) quantises to zero in 8 bits. Both were fixed, and the cull then measured
     * 24 793 -> 14 387 draws (-42 %), `submit` 21.38 -> 18.08 ms — which is why it is worth repairing
     * rather than abandoning. See optimisation.MD 8.8.
     */
    private static final boolean HIZ_BUILD = !"0".equals(System.getenv("VOXY_HIZ_BUILD"));

    /**
     * One-shot gate for the R32Float-depth-view probe. That probe decides whether Bug A's real fix is
     * reachable at all, so it runs once per run and says so in the log; see the call site for the full
     * reasoning and {@code lod-bugs.MD} §17.11.
     */
    private static boolean DEPTH_VIEW_PROBED = false;

    // Hoisted out of the per-frame body. These were `System.getenv` calls on the render path — three
    // per render pass for ATTACH_TRACE alone — and an env lookup is a native call, not a field read.
    // Reading them once at class-init is the same semantics for a variable that cannot change at
    // runtime, minus the per-pass cost.
    private static final boolean HIZ_SOURCE_COLOUR = "1".equals(System.getenv("VOXY_HIZ_SOURCE_COLOUR"));
    private static final boolean BRIDGE_SOLID_TEST = "1".equals(System.getenv("VOXY_BRIDGE_SOLID_TEST"));
    private static final boolean ATTACH_TRACE = "1".equals(System.getenv("VOXY_ATTACH_TRACE"));
    private static final boolean LOD_DEBUG_CLEAR = "1".equals(System.getenv("VOXY_LOD_DEBUG_CLEAR"));
    private static final boolean LOD_FORCE_MAGENTA = "1".equals(System.getenv("VOXY_LOD_FORCE_MAGENTA"));
    /** True for the current frame when rendering into Metallum's attachments rather than the bridge. */
    private boolean useMetallumTarget;
    private boolean loggedNoMetallumTarget;
    private int diagCount;

    /**
     * Flip Voxy's viewport vertically so it matches MC's bottom-up framebuffer. MC
     * compensates in its projection; Voxy's draws do not, so they rendered mirrored.
     * {@code VOXY_LOD_FLIP_Y=0} opts out for a controlled A/B.
     */
    private static final boolean FLIP_Y = !"0".equals(System.getenv("VOXY_LOD_FLIP_Y"));
    /** Opt-in attachment-identity trace ({@code VOXY_ATTACH_TRACE=1}). */
    private static long attachTraceCount = 0;

    /**
     * {@code VOXY_LOD_EMPTY_SCISSOR=1} draws the LOD pass into a 1x1 scissor rect: the draws all still
     * issue and the vertex stage still runs, but nothing is rasterised or shaded.
     *
     * <p>It answers one question that nothing else can, and the answer bounds every per-draw
     * optimisation on the docket. The LOD pass costs ~1.0 us/draw with no fixed term (optimisation.MD
     * 25), which is 5-10x a well-optimised indirect draw. If that time is per-draw overhead and vertex
     * work, an ICB can reach it. If it is rasterisation and fragment shading, an ICB cannot touch it at
     * all, and neither can any other change to how commands are issued.
     */
    private static final boolean EMPTY_SCISSOR = "1".equals(System.getenv("VOXY_LOD_EMPTY_SCISSOR"));
    private static boolean EMPTY_SCISSOR_LOGGED = false;

    /**
     * Blit destination for {@link #metalDepthTex} (w×h raw D32F floats) and
     * read source of the export pass. Exists because Metal silently reads
     * zeros when a depth-format texture is sampled through the
     * texture2d&lt;float&gt; declaration SPIRV-Cross emits for sampler2D;
     * buffer reads are format-blind. Lazy like the bridge.
     */
    private me.cortex.voxy.client.core.gpu.IGpuBuffer metalDepthReadBuffer;
    /**
     * Depth texture for {@code runPipelineMetal}'s render pass. Lazy-allocated
     * to match the bridge size so depth-tested LOD terrain self-occludes correctly.
     * Lives in Metal-side memory (the bridge's color is shared with GL via
     * IOSurface; the depth has no GL consumer so it stays Metal-private).
     */
    private me.cortex.voxy.client.core.gpu.IGpuTexture metalDepthTex;
    private int metalDepthWidth;
    private int metalDepthHeight;
    /**
     * M13 chunk 3: Shared-storage mirror of MC's main-FBO depth, refreshed
     * per frame via {@code glGetTexImage}. Sourced by
     * {@code HiZBuffer.buildMipChain(IGpuTexture, ...)} so the HiZ pyramid
     * carries real occlusion data instead of the zero-init stub from M12.
     * Lazy — allocated on the first Metal frame that has a non-zero sized
     * source framebuffer.
     */
    /** Animation counter for the placeholder Metal render — replaced by real Voxy output incrementally. */
    private int metalFrame;
    /** VOXY_UNDERWATER_LOD=1 forces LOD draws even when submerged-fog saturates the far field. */
    private static final boolean UNDERWATER_LOD_FORCE = "1".equals(System.getenv("VOXY_UNDERWATER_LOD"));
    /**
     * 2026-07-03 round 3: submersionSkip used to skip the ENTIRE vx-contract
     * translucent block — including the material-plane clears — so going
     * underwater froze metalVxTrans0-2 (+ the trans depth bridge) at the last
     * above-water frame while the GL resolve kept re-compositing the stale
     * planes into the pack's colortex every frame: ghost water squares that
     * toggle with the skip at the waterline. Keep the clear-only maintenance
     * running every contract frame (cleared planes make the resolve discard
     * everything) and gate only the DRAWS on the skip.
     * VOXY_TRANS_SUBMERSION_CLEAR=0 restores the old skip-everything gating.
     */
    private static final boolean TRANS_SUBMERSION_CLEAR = !"0".equals(System.getenv("VOXY_TRANS_SUBMERSION_CLEAR"));

    /**
     * Run one frame. This used to branch on the backend and carry a second, complete OpenGL pipeline
     * after the Metal early-return: Voxy's own depth-stencil framebuffer, a depth copy and a
     * stencil-mask dance, an SSAO compute pass, a fog blit, and the section renderer's GL draw calls.
     * All of it is deleted, along with the GL backend it drove.
     */
    public void runPipeline(Viewport<?> viewport, int sourceFrameBuffer, int srcWidth, int srcHeight) {
        this.runPipelineMetal(viewport, sourceFrameBuffer);
    }

    @Override
    protected void free0() {
        this.sectionRenderer.free();
        if (this.metalDepthReadBuffer != null) {
            this.metalDepthReadBuffer.free();
            this.metalDepthReadBuffer = null;
        }
        if (this.metalDepthTex != null) {
            this.metalDepthTex.free();
            this.metalDepthTex = null;
        }
        super.free0();
    }

    /**
     * Metal-path runPipeline (M12 chunk 6, evolving). Today runs the migrated
     * compute side end-to-end on Metal — every stage in this method is either
     * already encoder-backed or skipped with a Metal-aware substitute — and
     * still clears the IOSurface bridge as visual confirmation that the
     * pipeline executed. Real LOD draws land in subsequent chunk 6 steps when
     * renderTerrain / renderTranslucent migrate to RenderEncoder.
     * <p>
     * Compared to the GL {@link #runPipeline}, the Metal path currently
     * skips: {@link #setup} (depth-stencil FBO copy uses raw GL),
     * {@link me.cortex.voxy.client.core.rendering.util.HiZBuffer#buildMipChain}
     * (partial GL — see chunk 6 step 2), the FrEx work loop wrapper, the
     * raw {@code glMemoryBarrier} inside {@link #innerPrimaryWork}, the
     * post-opaque SSAO compute, and {@link #finish}. The compute pipeline
     * (HOT traversal + buildDrawCalls' 5 prepasses) runs in full.
     */
    /**
     * Per-phase frame timings, in nanoseconds, reported every 600 frames.
     *
     * <p>Added because the frame rate told us nothing about *where* the time goes, and the obvious
     * suspects — draw count, the per-draw CPU loop, the mid-frame submit — need separating before
     * any of them is worth optimising. Splits the frame into: preparation (traversal + the draw-call
     * prepasses), the split submit (which blocks on GPU completion), the encode of the LOD pass, and
     * the final submit.
     */
    private long perfPrep, perfSplit, perfEncode, perfSubmit;
    private int perfFrames;
    /**
     * Sum and count of Metal's OWN GPU durations for the frames in this report window, in ms.
     *
     * <p>Separate from {@code perfSubmit} on purpose. {@code submit} is the CPU's wait — {@code commit}
     * plus {@code waitUntilCompleted} — which on the guest branch also spans Metallum's
     * {@code flushFrame()} and the deferred ordered index-wait. {@code gpu} is how long the GPU
     * actually worked. They answer different questions and the gap between them is itself a
     * measurement: a large gap means the CPU is blocked on someone else's queued work, not on Voxy's.
     */
    private double perfGpuSum;
    private int perfGpuSamples;
    /** Same, for the PRE-LOD segment: traversal + the five prepasses + the Hi-Z pyramid build. */
    private double perfGpuPreSum;
    private int perfGpuPreSamples;

    private void perfReport() {
        if (++this.perfFrames < 600) return;
        double n = this.perfFrames;
        // `gpu` is APPENDED, not inserted: tools/parse_perf.py's regex matches the named fields in
        // order and is not anchored at the end, so a trailing field is safe and reordering is not.
        // -1 means Metal reported no timestamps for any frame in the window.
        // `gpu` is the LOD pass segment; `gpuPre` is everything encoded before it (traversal, the five
        // prepasses, the Hi-Z build). They are reported separately because a single untagged number
        // could not say which segment a duration belonged to, and an empty LOD pass once appeared to
        // cost 9.81 ms with nothing able to confirm or deny it.
        me.cortex.voxy.common.Logger.info(String.format(
                "[Metal-PERF] ms/frame  prep=%.2f  split=%.2f  encode=%.2f  submit=%.2f  total=%.2f  "
                        + "gpu=%.2f  gpuPre=%.2f",
                this.perfPrep / n / 1e6, this.perfSplit / n / 1e6,
                this.perfEncode / n / 1e6, this.perfSubmit / n / 1e6,
                (this.perfPrep + this.perfSplit + this.perfEncode + this.perfSubmit) / n / 1e6,
                this.perfGpuSamples > 0 ? this.perfGpuSum / this.perfGpuSamples : -1.0,
                this.perfGpuPreSamples > 0 ? this.perfGpuPreSum / this.perfGpuPreSamples : -1.0));
        this.perfPrep = this.perfSplit = this.perfEncode = this.perfSubmit = 0;
        this.perfGpuSum = 0.0;
        this.perfGpuSamples = 0;
        this.perfGpuPreSum = 0.0;
        this.perfGpuPreSamples = 0;
        this.perfFrames = 0;
    }

    /**
     * Sample the backend's most recent GPU duration, once per frame, for {@link #perfReport()}.
     *
     * <p>Negative values are Metal's "no timestamps" answer and are skipped rather than averaged in,
     * so a window where the instrument is unavailable reports -1 instead of silently reading as 0 ms.
     */
    private void sampleGpuTime(me.cortex.voxy.client.core.gpu.RenderBackend backend) {
        if (backend instanceof me.cortex.voxy.client.core.metal.MetalRenderBackend mb) {
            double gpu = mb.lastGpuMsFor(me.cortex.voxy.client.core.metal.MetalRenderBackend.GPU_TAG_POST_LOD);
            if (gpu >= 0.0) {
                this.perfGpuSum += gpu;
                this.perfGpuSamples++;
            }
            double gpuPre = mb.lastGpuMsFor(me.cortex.voxy.client.core.metal.MetalRenderBackend.GPU_TAG_PRE_LOD);
            if (gpuPre >= 0.0) {
                this.perfGpuPreSum += gpuPre;
                this.perfGpuPreSamples++;
            }
        }
    }

    private void runPipelineMetal(Viewport<?> viewport, int sourceFrameBuffer) {
        int fbw = viewport.width;
        int fbh = viewport.height;
        if (fbw <= 0 || fbh <= 0) return;
        var backend = me.cortex.voxy.client.core.gpu.RenderBackendFactory.get();
        if (!(backend instanceof me.cortex.voxy.client.core.metal.MetalRenderBackend voxyMb)) return;
        final long perfT0 = System.nanoTime();

        // 1) Allocate the IOSurface bridge sized to MC's framebuffer. The
        //    bridge is the cross-context handle: Metal renders into the
        //    backing MTLTexture, IOSurfaceBridgeCompositor blits it into MC's
        //    main RT via a CGL-bound GL_TEXTURE_RECTANGLE source FBO.
        boolean metallumTarget = me.cortex.voxy.client.core.metal.MetallumBridge.available();

        // 2) Ensure the HiZ texture is allocated so HOT can bind it. The
        //    encoder-driven mip-chain build is wired up but parked: an
        //    initial integration test (2026-05-13) showed LOD chunks
        //    disappearing at the horizon when HOT samples the populated
        //    pyramid — likely a Depth32Float_Stencil8 sampling-as-sampler2D
        //    mismatch versus the GL path's pre-processed depth (see
        //    initDepthStencil's stencil-mask dance that zeros sky regions).
        //    Revert to M12's zero-init pyramid until that's debugged so
        //    HOT trivially passes every frustum-visible section. The
        //    DepthMirror class + MetalNative.mtlTextureNewSubresourceView
        //    JNI + IGpuTexture.createView(level, count) all stay committed
        //    for the follow-up. ensureAllocated zero-fills every mip on
        //    Metal at (re)allocation — MTLTexture contents are otherwise
        //    UNDEFINED and the screenspace.glsl "pointSample <= 0.0" guard
        //    needs real zeros, not luck.
        // 2) Build the Hi-Z pyramid from THIS frame's depth, so the traversal below can cull occluded
        //    subtrees -- the thing upstream has and this build has never had on Metal.
        //
        //    Two facts make it possible now, and neither held when the build was parked in 2026-05:
        //      - MC's depth attachment is Depth32Float (see the [Metal-DEPTHFMT] report below), i.e. a
        //        format a sampler can read. The parked attempt blamed a packed Depth32Float_Stencil8,
        //        which genuinely cannot be sampled as texture2d<float>. That is not what this path has.
        //      - ensureAllocated zero-fills every mip, so an unbuilt or partly-built pyramid reads as
        //        "nothing occludes" instead of the garbage that made the original attempt cull subtrees
        //        on noise. buildMipChain calls it itself.
        //
        //    ORDERING, which is the only real constraint: the hook fires at the TAIL of Sodium's SOLID
        //    pass (MixinDefaultChunkRenderer), so vanilla's terrain for this frame is already in the
        //    attachment -- this is NOT last frame's depth. Build after that is populated and before
        //    doTraversal, and nothing else matters. The refresh therefore has to happen HERE rather than
        //    in the passBuilder block below, which runs after the traversal.
        //
        //    If the attachment is unavailable (no Metallum target this frame) the pyramid stays
        //    zero-filled and the traversal's guard makes the cull a no-op -- i.e. exactly the previous
        //    behaviour, so this cannot regress that case.
        if (metallumTarget) {
            if (this.metallumDepth == null) {
                this.metallumDepth = me.cortex.voxy.client.core.metal.MetallumAttachmentTexture.depth();
            }
            this.metallumDepth.refresh();
        }

        // ---------------------------------------------------------------------------------------
        // THE ONE PROBE THAT DECIDES BUG A'S REAL FIX — and it runs before anything is built on it.
        //
        // See lod-bugs.MD 17.11. The Hi-Z pyramid is built from THIS pass's depth-as-colour attachment,
        // which is last frame's by construction, so the traversal culls frame N against frame N-1's
        // depth and a moving camera slides each node's footprint onto texels describing where it used to
        // be. MC's own depth is the fix: it is already ShaderRead, and at this point in the frame -- the
        // tail of Sodium's SOLID pass -- it already holds THIS frame's vanilla terrain. The port even
        // has the handle here and never uses it as the pyramid's source.
        //
        // It reads zeros for one reason: a Depth32Float sampled through a `sampler2D` becomes MSL
        // `texture2d<float>`, which Metal does not allow (it wants `depth2d`). The way round it is an
        // R32Float VIEW of the same texture, and `newTextureViewWithPixelFormat:` REFUSES to make a view
        // whose parent lacks MTLTextureUsagePixelFormatView -- which is why metallum's
        // MetalGpuTexture.toMtlTextureUsage now grants it to every render attachment.
        //
        // Whether Metal permits a Depth32Float -> R32Float view AT ALL is unverified and is the whole
        // question: if it refuses, the fallback is a hand-written MSL `depth2d` compute pass, and none
        // of the other three steps of the fix are worth writing. So this runs first, logs the answer,
        // and costs one texture view per run.
        // ---------------------------------------------------------------------------------------
        if (this.metallumDepth != null && this.metallumDepth.metalHandle() != 0 && !DEPTH_VIEW_PROBED) {
            DEPTH_VIEW_PROBED = true;
            final long dh = this.metallumDepth.metalHandle();
            final long view = me.cortex.voxy.client.core.metal.MetalNative.mtlTextureNewSubresourceView(
                    dh, 55 /* MTLPixelFormatR32Float */,
                    me.cortex.voxy.client.core.metal.MetalNative.mtlTextureGetTextureType(dh),
                    0, 1, 0, 1);
            if (view == 0L) {
                me.cortex.voxy.common.Logger.info("[Metal-DEPTHVIEW] depth fmt="
                        + this.metallumDepth.mtlPixelFormat() + " (" + this.metallumDepth.getWidth() + "x"
                        + this.metallumDepth.getHeight() + ") -> R32Float view REFUSED (null). Metal does"
                        + " not allow this format view, so the pyramid cannot read the frame's depth"
                        + " this way and lod-bugs.MD 17.11's fallback (a hand-written MSL depth2d pass)"
                        + " is the only route left.");
            } else {
                me.cortex.voxy.common.Logger.info("[Metal-DEPTHVIEW] depth fmt="
                        + this.metallumDepth.mtlPixelFormat() + " (" + this.metallumDepth.getWidth() + "x"
                        + this.metallumDepth.getHeight() + ") -> R32Float view OK, handle=" + view
                        + " fmt=" + me.cortex.voxy.client.core.metal.MetalNative
                                .mtlTextureGetPixelFormat(view)
                        + ". The frame's own depth is readable as R32Float, so the pyramid can be built"
                        + " from THIS frame instead of the last one.");
                me.cortex.voxy.client.core.metal.MetalNative.mtlRelease(view);
            }
        }

        // 2b) The Voxy-owned depth-texture, allocated BEFORE the build because it is the build's source.
        //
        //     R32F, and that is load-bearing rather than a preference. An earlier version of this used
        //     RGBA8, chosen because it is the format the depth readback probe had been PROVEN on -- and
        //     that choice silently destroyed the data. On this frame's reverse-Z convention the LOD's
        //     depth is near/d, so distant terrain sits around 1e-4..1e-3; quantised into 8 bits that is
        //     ZERO. The shader still wrote vec4(gl_FragCoord.z, 0, 0, 1), so the texel's alpha byte made
        //     the attachment read as "100% non-zero" on a raw byte probe while its RED channel -- the
        //     only channel the pyramid's textureGather samples -- was empty, and the pyramid min-reduced
        //     to all zeros. R32F holds the depth exactly.
        //
        //     The probe that "proved" RGBA8 was measuring a frame in which the LOD drew nothing at all
        //     (see the camera note on the harness), so its verdict was about the vantage, not the format.
        if (this.metalDepthTex == null || this.metalDepthWidth != fbw || this.metalDepthHeight != fbh) {
            if (this.metalDepthTex != null) this.metalDepthTex.free();
            this.metalDepthTex = backend.createTexture()
                    .store(org.lwjgl.opengl.GL30C.GL_R32F, 1, fbw, fbh)
                    .name("VoxyLodDepthColour");
            this.metalDepthWidth = fbw;
            this.metalDepthHeight = fbh;
            this.hizBuildLogged = false;   // a resize invalidates whatever the pyramid held
        }
        //    The source is metalDepthTex -- the LOD pass's SECOND COLOUR attachment, which carries each
        //    fragment's depth as colour. It is NOT MC's depth attachment, and that is deliberate: a
        //    depth-format texture sampled through a `sampler2D` becomes MSL `texture2d<float>`, from
        //    which Metal silently reads zeros. See quads.frag's VOXY_LOD_DEPTH_COLOUR block.
        //
        //    ONE FRAME BEHIND, by construction. This call sits before doTraversal() below and before
        //    the LOD pass that CLEARs and rewrites metalDepthTex, so the traversal culls frame N against
        //    frame N-1's depth. That is the ordinary Hi-Z lag and it is fine. An earlier comment here
        //    claimed the pyramid "reads the very texture the geometry was drawn into" -- true of the
        //    texture, false of the timing, and the sort of claim that sends the next reader looking in
        //    the wrong place.
        //
        //    ON BY DEFAULT AGAIN, 2026-09-22, and the reason it was off is worth keeping. It used to be
        //    pure cost: the pyramid it built was attached as a depth attachment while its pipeline
        //    declares an R32F colour format, so the blit's output was dropped and nothing ever wrote it
        //    -- +3.7 to +7 ms of `submit` for zero culled sections. That attachment bug was fixed (see
        //    HiZBuffer's class doc) but the switch stayed off, because turning it on removed the distant
        //    LOD: the pyramid's vertical orientation was inverted (HiZBuffer.buildMipChain), so the
        //    traversal tested every node against the screen region opposite it. With the orientation
        //    fixed the cull is safe to ship on, which is what the owner asked for -- "we cannot just
        //    turn off occlusion cull, there's something broken with the occlusion cull and you should
        //    fix it". VOXY_HIZ_BUILD=0 remains the ceiling arm for an A/B and is the only way to run
        //    with the pyramid deliberately zero-filled.
        if (HIZ_BUILD && this.metalDepthTex != null && this.metalDepthTex.id() != -1) {
            // VOXY_HIZ_SOURCE_COLOUR=1 -- EXPERIMENTAL BISECTION, not a feature. Build the pyramid from
            // the LOD pass's ALBEDO attachment (metallumColor) instead of the depth-as-colour one.
            //
            // It exists because the two defects in this cull cannot be told apart by the pyramid probe
            // alone: metalDepthTex (attachment 1) reads empty, so the pyramid is zero whether or not its
            // build is correct, and "the build works" is unfalsifiable. metallumColor is a colour
            // attachment this tree has already read back at 100% nonzero, so pointing the build at it
            // separates the two cleanly:
            //   pyramid non-zero with this switch -> the colour-attachment BUILD works (Defect A fixed),
            //                                        and the remaining fault is the source (Defect B);
            //   pyramid still zero with a source that has data -> the build is still broken.
            final boolean sourceIsColour = HIZ_SOURCE_COLOUR;
            final me.cortex.voxy.client.core.gpu.IGpuTexture hizSource =
                    sourceIsColour && this.metallumColor != null ? this.metallumColor : this.metalDepthTex;
            viewport.hiZBuffer.buildMipChain(hizSource, viewport.width, viewport.height);
            if (!this.hizBuildLogged) {
                this.hizBuildLogged = true;
                me.cortex.voxy.common.Logger.info("[Metal-HIZBUILD] built pyramid from the LOD pass's"
                        + " depth-as-colour attachment, " + viewport.width + "x" + viewport.height
                        + " (attachment=" + this.metalDepthTex.id()
                        + ", depth=" + (this.metallumDepth == null ? "null"
                                : Long.toString(this.metallumDepth.metalHandle())) + ")");
            }
        } else {
            viewport.hiZBuffer.ensureAllocated(viewport.width, viewport.height);
            if (!this.hizBuildLogged) {
                this.hizBuildLogged = true;
                // Distinguish the two reasons for not building, because they mean opposite things: the
                // switch being off is a deliberate choice, while a missing attachment is a real fault.
                // Conflating them in one message is how a diagnostic ends up lying about the state it
                // was written to report.
                final String why = !HIZ_BUILD
                        ? "VOXY_HIZ_BUILD=0 -- the cull is deliberately disabled, and the pyramid stays"
                          + " zero-filled so it is a no-op"
                        : "the build is on but no depth attachment is available";
                me.cortex.voxy.common.Logger.info("[Metal-HIZBUILD] not building: " + why
                        + " (target=" + metallumTarget
                        + ", depthHandle=" + (this.metallumDepth == null ? "null"
                                : Long.toString(this.metallumDepth.metalHandle()))
                        + ", depthId=" + (this.metallumDepth == null ? "n/a"
                                : Integer.toString(this.metallumDepth.id())) + ")");
            }
        }

        // 2b) The Metal-side depth texture used to be allocated here, lazily, and its comment claimed
        //     "the encoder pass clears it to 1.0 (far plane) each frame" -- which was never true of this
        //     pass (it LOADs MC's attachment, so a clear value is inert) and would have been the wrong
        //     value under reverse-Z anyway, where 1.0 is the NEAR plane. It is now allocated in step 2b
        //     above, because the Hi-Z build reads it and the build has to run before the traversal.

        // 3) Compute side — copy of innerPrimaryWork's body minus the GL bits
        //    (HiZBuffer.buildMipChain, raw glMemoryBarrier, FrEx loop). Each
        //    sub-stage is already encoder-backed (commits 0b963825, 68734b78,
        //    5dcbc645, 89b35814 for HOT's last raw-GL gaps).
        me.cortex.voxy.client.core.rendering.util.DownloadStream.INSTANCE.tick();
        this.nodeManager.tick(this.traversal.getNodeBuffer(), this.nodeCleaner);
        this.nodeCleaner.tick(this.traversal.getNodeBuffer());
        this.traversal.doTraversal(viewport);

        // 4) Per-frame draw-command generation — all 5 MDIC compute prepasses
        //    (prep / cull-stub / commandGen / prefixSum / translucentGen) now
        //    flow through ComputeEncoder (chunks 1–5). Raw cast matches the
        //    GL path (line 119) — the renderer's viewport generic is set at
        //    construction by RenderPipelineFactory and we trust the pairing.
        @SuppressWarnings({"rawtypes", "unchecked"})
        AbstractSectionRenderer rs = (AbstractSectionRenderer) this.sectionRenderer;
        rs.buildDrawCalls(viewport);
        final long perfT1 = System.nanoTime();

        // M13 2026-05-14 baseInstance workaround: flush + wait so the compute
        // prepasses (commandGen writes drawCallBuffer's baseInstance field)
        // complete before the render pass starts. MetalRenderEncoder.
        // drawIndexedIndirect needs to CPU-read drawCallBuffer per draw to
        // push baseInstance via setVertexBytes (drawIndexedPrimitives:
        // indirectBuffer: doesn't propagate it natively). Without this
        // submit() the CPU sees stale data from the prior frame.
        backend.submitDeferWait();
        final long perfT2 = System.nanoTime();
        this.perfPrep += perfT1 - perfT0;   // traversal + draw-call prepasses
        this.perfSplit += perfT2 - perfT1;  // split submit, commit only -- the ordered wait moved to the draw

        // The Layer-B diagnostic that read the renderList and drawCountCallBuffer back on a 600-frame
        // cadence is GONE with its [Metal-LayerB] line. It was eight memGetInt reads of GPU-written
        // memory every ten seconds to compute seven locals that nothing consumed once the log went.

        // 5) Render pass against bridge color + Voxy-owned depth. Clears both
        //    each frame (no MC-depth import on Metal yet, so we render every
        //    LOD chunk against a fresh depth buffer — they self-occlude but
        //    don't z-test against MC's foreground terrain). Inside the pass
        //    we call MDIC's Metal-aware renderOpaque equivalent to issue
        //    the actual LOD draws via the RenderEncoder API. 2026-05-14
        //    revert: alpha back to 1.0 (M12-stable) since the alpha-composite
        //    shader path made LOD invisible in-game. With the blit compositor
        //    the clear colour shows through in non-LOD areas (sky no longer
        //    visible through them); Sodium overdraws its near terrain on top.
        // 2026-05-14 diagnostic finding: with magenta clear, user reports
        // "Veo magenta en todo (excepto terreno MC cercano)" — confirming
        // the IOSurface bridge + blit-to-MC-mainRT path works end-to-end.
        // The reason LOD chunks are invisible is the MDIC opaque/temporal/
        // translucent draws are NOT producing visible pixels in the bridge.
        // Likely causes: drawIndexedIndirect counts are zero (HOT culling /
        // commandGen prepass), or vertex shader clips all geometry. Restored
        // dark clear so day-to-day play isn't magenta-flooded; the rendering
        // pipeline diagnosis continues in MDIC + buildDrawCalls.
        float clearR = 0.02f;
        float clearG = 0.02f;
        float clearB = 0.04f;
        if (viewport.fogParameters != null) {
            clearR = viewport.fogParameters.red();
            clearG = viewport.fogParameters.green();
            clearB = viewport.fogParameters.blue();
        }
        // DIAGNOSTIC (2026-05-25): VOXY_BRIDGE_SOLID_TEST=1 fills the bridge
        // with a static bright-green clear and SKIPS all LOD draws below. If
        // the green is rock-stable on screen, the IOSurface bridge + composite
        // + sync path is sound and the flicker lives in the LOD draws/content;
        // if the green itself flickers, the bridge/sync is the culprit.
        boolean bridgeSolidTest = BRIDGE_SOLID_TEST;
        if (bridgeSolidTest) {
            clearR = 0.0f; clearG = 1.0f; clearB = 0.0f;
            if ((this.metalFrame % 600) == 1) {
            }
        }
        // Clear alpha 0.0: the alpha-discard composite drops undrawn bridge
        // pixels so MC's own sky/fog shows behind the LODs (kills the
        // whole-far-field fog flash when the eye crosses the water surface).
        // The blit fallback (VOXY_COMPOSITE_BLIT=1) copies raw pixels and
        // needs the M12-stable opaque clear; the solid test must stay visible.
        // Iris-pack mode injects the bridge into Iris's terrain gbuffer with
        // an alpha-discard + depth-write shader — undrawn pixels must carry
        // alpha 0 so only Voxy-drawn pixels write into the pack's colortex.
        // lodExport: TRUE for both LOD-export consumers — the legacy gbuffer
        // injection AND the native vx contract (issue #9); both need the
        // colour bridge with alpha-as-coverage plus the packed depth bridge.
        var passBuilder = me.cortex.voxy.client.core.gpu.RenderPassDesc.builder(fbw, fbh);
        if (metallumTarget) {
            if (this.metallumColor == null) {
                this.metallumColor = me.cortex.voxy.client.core.metal.MetallumAttachmentTexture.color();
            }
            if (this.metallumDepth == null) {
                this.metallumDepth = me.cortex.voxy.client.core.metal.MetallumAttachmentTexture.depth();
            }
            this.metallumColor.refresh();
            this.metallumDepth.refresh();
            // One-shot: what FORMAT is MC's Metal depth attachment? This decides whether Voxy's
            // hierarchical occlusion pyramid can be built straight from it or needs a sampleable copy
            // first. A *Stencil8 format cannot be reliably sampled as texture2d<float> -- that is the
            // documented failure behind both the Iris depth export reading zeros and the parked Hi-Z
            // build -- while a pure Depth32Float can. Numeric values are MTLPixelFormat: 252 =
            // Depth32Float, 260 = Depth32Float_Stencil8. Logged once: it is a property of the frame,
            // not of any particular frame, and the answer is not deducible from source because
            // Metallum maps the format through from whatever MC requested.
            if (!this.depthFormatLogged && this.metallumDepth.mtlPixelFormat() != 0) {
                this.depthFormatLogged = true;
                final int fmt = this.metallumDepth.mtlPixelFormat();
            }
        }
        // Both attachments are only ASSIGNED inside the `if` above, so on a frame where the target is
        // unavailable they are still null -- and the diag line below dereferences them. Guarding here
        // rather than relying on the short-circuit: with `metallumTarget` false the `&&` never
        // evaluates `metallumColor.id()`, so a null would slip through this line and only surface in
        // the diag block, on the first frames, which is exactly when the target can be missing.
        this.useMetallumTarget = metallumTarget
                && this.metallumColor != null && this.metallumDepth != null
                && this.metallumColor.id() != -1 && this.metallumDepth.id() != -1;
        if (this.diagCount < 5) {
            this.diagCount++;
            me.cortex.voxy.client.core.metal.MetallumBridge.logRenderPassCounters();
        }
        // VOXY_ATTACH_TRACE=1: compare the colour attachment Voxy renders the LOD into against
        // Minecraft's actual main render target. The LOD pass provably runs (no "skipping the LOD
        // pass" warning, useMetallumTarget=true) and provably carries valid geometry, yet a
        // debug-CLEAR of that attachment to magenta never reaches the screen. If these two handles
        // differ, the pass is drawing into a texture that is never composited -- which explains
        // valid draws producing zero pixels without anything else being wrong.
        if (ATTACH_TRACE && (attachTraceCount++ % 600) == 1) {
            long voxyColor = me.cortex.voxy.client.core.metal.MetallumBridge.colorAttachment();
            long mcColor = 0;
            try {
                // mainRenderTarget() returns a RenderTarget, not a texture -- textureHandle needs
                // the GpuTexture inside it, or it returns 0 and the comparison says nothing.
                com.mojang.blaze3d.pipeline.RenderTarget rt =
                        net.minecraft.client.Minecraft.getInstance().gameRenderer.mainRenderTarget();
                if (rt != null && rt.getColorTexture() != null) {
                    mcColor = me.cortex.voxy.client.core.metal.MetallumBridge.textureHandle(rt.getColorTexture());
                }
            } catch (Throwable ignored) {
                // diagnostic only
            }
        }

        if (this.useMetallumTarget) {
            // Whole-frame Metal: draw straight into the frame's own attachments. LOAD on both,
            // not CLEAR -- at SOLID-head the frame already holds the sky and MC's cleared depth,
            // and Sodium's near terrain draws over the LODs afterwards. Voxy owns neither buffer,
            // so it must not clear them.
            // VOXY_LOD_DEBUG_CLEAR=1 -- render-target test that does not depend on geometry
            // at all. The LOD pass normally LOADs the frame's attachments (Voxy owns neither
            // buffer). Clearing to magenta instead proves in one frame whether this pass writes
            // to the attachments the screen is actually made of: a magenta frame means the
            // target and pass are sound and the fault is downstream in the draws; an unchanged
            // frame means the pass is writing somewhere that never reaches the screen.
            boolean debugClear = LOD_DEBUG_CLEAR;
            passBuilder.addColorAttachment(this.metallumColor, 0,
                            debugClear
                                    ? me.cortex.voxy.client.core.gpu.RenderPassDesc.LoadAction.CLEAR
                                    : me.cortex.voxy.client.core.gpu.RenderPassDesc.LoadAction.LOAD,
                            me.cortex.voxy.client.core.gpu.RenderPassDesc.StoreAction.STORE,
                            debugClear ? 1f : 0f, 0f, debugClear ? 1f : 0f, 1f)
                    // Depth: MC's OWN attachment, restored.
                    //
                    // An earlier version of this used a Voxy-owned depth texture, because MC's depth
                    // cannot be SAMPLED -- a depth-format texture read through `sampler2D` becomes MSL
                    // `texture2d<float>`, from which Metal silently reads zeros (it needs `depth2d`; this
                    // file records the measurement above). But that constrains the PYRAMID'S SOURCE, not
                    // the depth attachment, and conflating the two cost the LOD its depth test against
                    // vanilla terrain for no reason. The pyramid takes its source from a colour
                    // attachment now (see below), so the depth attachment goes back to what it should be.
                    //
                    // LOAD, not CLEAR, and that is the whole point: it is what makes LOD fragments behind
                    // vanilla terrain fail the test.
                    .depthAttachment(this.metallumDepth, 0,
                            me.cortex.voxy.client.core.gpu.RenderPassDesc.LoadAction.LOAD,
                            me.cortex.voxy.client.core.gpu.RenderPassDesc.StoreAction.STORE,
                            0f);
            if (this.metalDepthTex != null && this.metallumDepth != null) {
                // Colour attachment 1: this fragment's depth as COLOUR, written by quads.frag under
                // VOXY_LOD_DEPTH_COLOUR. It is the Hi-Z pyramid's source, and colour is the point: an
                // R32F texture is an ordinary `texture2d<float>` read on Metal, where a depth-format
                // texture silently reads zeros. No blit, no buffer transfer, no encoder transition --
                // the pyramid is fed by the same pass that draws the geometry.
                passBuilder.addColorAttachment(this.metalDepthTex, 0,
                        me.cortex.voxy.client.core.gpu.RenderPassDesc.LoadAction.CLEAR,
                        me.cortex.voxy.client.core.gpu.RenderPassDesc.StoreAction.STORE,
                        0f, 0f, 0f, 0f);
            }
        }
        var pass = passBuilder.build();
        // One-shot: how many colour attachments did the pass actually end up with? Every other link was
        // verified correct (the define reaches the shader, the compiled MSL declares [[color(1)]], the
        // pipeline declares an R32F format) and the attachment still reads zero -- so the question is
        // whether the builder kept the second one at all, which nothing reported.
        if (!this.passShapeLogged) {
            this.passShapeLogged = true;
        }
        // Submersion far-field skip: with the eye in water/lava the env fog
        // saturates at 24-96 blocks while every LOD fragment sits far beyond
        // it — the whole LOD field is 100% fog colour by construction. Drawing
        // it anyway only exposes artifacts: Sodium's fog-occlusion culling
        // de-renders near seafloor whose pixels then fall through the opaque
        // blit to the LOD field ("sand turns transparent", flooded caverns),
        // and any residual draw nondeterminism strobes. Skip the LOD draws and
        // let the fog-coloured clear stand — visually identical murk, stable
        // by construction. Guarded so tiny render distances (where LOD could
        // outrange the fog) keep drawing. VOXY_UNDERWATER_LOD=1 forces draws.
        boolean submersionSkip = false;
        // useEnvFog() gate: with Voxy fog disabled there is no murk to hide
        // behind — the skip only applies when the far field is provably
        // fog-saturated. (Previously fog-off avoided the skip only by the
        // accident of MixinFogRenderer inflating envEnd to 999999999.)
        // Round 23: inject mode forces useEnvFog() false, which made this
        // skip DEAD CODE under packs — the underwater far-field protection
        // (added specifically against "sand turns transparent / flooded
        // caverns") never engaged while swimming with a pack active. In
        // inject mode the pack's own underwater fog saturates the far field
        // (BSL composites it by depth), so the skip's premise holds there.
        boolean submersionEligible = this.useEnvFog()
                || me.cortex.voxy.client.core.util.IrisUtil.irisGbufferInjectMode();
        if (!UNDERWATER_LOD_FORCE && submersionEligible && viewport.fogParameters != null) {
            float envEnd = viewport.fogParameters.environmentalEnd();
            int rdBlocks = net.minecraft.client.Minecraft.getInstance().options.getEffectiveRenderDistance() * 16;
            submersionSkip = envEnd < 128.0f && rdBlocks > envEnd * 2.0f;
        }
        if (!this.useMetallumTarget) {
            // No attachments to render into: the pass would be empty and Metal returns a NULL
            // encoder. Metallum has not opened a render encoder for this pass yet, so there is
            // nothing to borrow (verified: colorHandle=0x0 encoder=0x0 at this point, even at the
            // Sodium pass tail).
            //
            // Skipping keeps the client running on Metal. Rendering LODs needs the attachments to
            // actually be bound by the time Voxy's hook fires.
            if (this.diagCount <= 5) {
                this.diagCount++;
                me.cortex.voxy.common.Logger.warn(
                        "[Metal] no Metallum attachments to render into; skipping the LOD pass this frame");
            }
            return;
        }
        // VOXY_LOD_ICB_OPTIMIZE=1: populate and optimize the ICBs BEFORE the render pass opens.
        //
        // This is the only place it can happen. optimizeIndirectCommandBuffer is a blit-encoder call
        // and a command buffer has one encoder at a time, so it cannot be issued from inside the pass
        // the ICB is executed in -- and populating up front is what lets the sequence be
        // populate -> optimize -> execute, which is the order the API requires. No-op unless the
        // switch is on. See MDICSectionRenderer.prepareIcb.
        if (me.cortex.voxy.client.core.rendering.section.backend.mdic.MDICViewport.ICB_OPTIMIZE
                && this.sectionRenderer instanceof me.cortex.voxy.client.core.rendering.section.backend.mdic.MDICSectionRenderer mdic0
                && viewport instanceof me.cortex.voxy.client.core.rendering.section.backend.mdic.MDICViewport mv0) {
            // THE DRAIN MUST COME FIRST, and this is the whole reason the optimized path is arranged
            // this way. prepareIcb reads the draw list and the draw counts on the CPU; on Metal those
            // are GPU-written, and until the prepasses are waited for the read is a frame stale -- the
            // geometry then lands at another section's origin, keeping its own baked light, which is
            // bug 3 exactly. The ordinary drain lives further down, INSIDE the pass, where it is too
            // late for a read that has already happened.
            //
            // It is idempotent, so the later call finds nothing pending and costs nothing.
            backend.awaitCommitted();
            mdic0.prepareIcb(mv0);
        }
        try (var enc = backend.beginRenderPass(pass)) {
            // Y orientation. Metal's setViewport maps NDC y=+1 to originY, i.e. the TOP of the
            // target, which is the opposite of the GL-style bottom-up convention MC's framebuffer
            // is written in. MC compensates inside its own projection, so its scene is correct --
            // but every Voxy draw is NOT: the debug triangles take their position straight from
            // gl_VertexID with no matrix at all, and the LOD's computeProjectionMat cancels MC's
            // projection (base * inverse(base) * voxyProj), so neither inherits the compensation.
            // Both therefore rendered vertically mirrored, which is visible in the triangle tests:
            // an apex-up triangle drew apex-down, and a bottom-left right angle drew top-left.
            // Measured by eye, not by bounding box -- a bbox is identical under a flip, which is
            // why every orientation claim made from one was unfounded.
            //
            // A negative height with originY at the bottom edge flips the mapping, so the viewport
            // corrects both paths at once. VOXY_LOD_FLIP_Y=0 restores the unflipped viewport.
            if (FLIP_Y) {
                enc.setViewport(0, fbh, fbw, -fbh, 0.0f, 1.0f);
            } else {
                enc.setViewport(0, 0, fbw, fbh, 0.0f, 1.0f);
            }
            // VOXY_LOD_EMPTY_SCISSOR=1 gives the LOD pass a 1x1 scissor rect, so every draw still issues,
            // still carries its index count, and still runs its vertex shader -- but no fragment is
            // ever shaded. It exists to split the LOD pass's ~1.0 us/draw into the two halves that an
            // ICB can and cannot reach: per-draw/vertex work on one side, rasterisation and fragment
            // shading on the other. The ICB removes the first and nothing whatsoever of the second, so
            // if this arm collapses `gpu` the ICB is not the lever (optimisation.MD 25.3).
            //
            // The scissor is set HERE, in the LOD pass's own setup, and never restored: Voxy closes its
            // borrowed encoder with the state marked stale, so Metallum rebinds before its next draw.
            // It is a single-rect state, deliberately 1x1 and not 0x0 -- an empty rect is legal but
            // there is nothing to gain from arguing with a driver about it, and 1 pixel of 8.3M is
            // 0.00001% of the rasterisation this arm is removing.
            //
            // Deliberately set ONLY in the probe arm. Writing the full-viewport rect in the default
            // arm would be tidier -- a defined scissor instead of an inherited one -- but it would also
            // perturb the shipped path, and the reference screenshot set was captured without it. The
            // baseline arm has to stay byte-identical to today's build or the diff harness starts
            // reporting a change that the probe did not make.
            if (EMPTY_SCISSOR) {
                enc.setScissor(0, 0, 1, 1);
                if (!EMPTY_SCISSOR_LOGGED) {
                    EMPTY_SCISSOR_LOGGED = true;
                    me.cortex.voxy.common.Logger.info("[Metal-SCISSOR] VOXY_LOD_EMPTY_SCISSOR active: LOD"
                            + " pass rasterises into 1x1, so every draw still issues and every vertex still"
                            + " shades but no fragment is covered");
                }
            }
            // M12 close — invoke MDIC's Metal-aware draws in the same order
            // GL runPipeline uses (opaque → temporal → translucent). Iris is
            // GL-gated upstream so on non-GL the section renderer is always
            // an MDICSectionRenderer (and its viewport an MDICViewport —
            // typing follows from the RenderPipelineFactory pairing).
            // postOpaquePreTranslucent (SSAO) is skipped on Metal — SSAO
            // is M13 polish; the LOD result is intelligible without it.
            // The deferred wait, taken as late as possible: everything above this line is CPU work that
            // has been overlapping the GPU's prepasses. This is the first point that reads anything the
            // prepasses wrote (MetalRenderEncoder pulls baseInstance out of drawCallBuffer per draw), so
            // the ordering the fix establishes is unchanged -- only the idle time moved.
            backend.awaitCommitted();
            if (!bridgeSolidTest && !submersionSkip
                    && this.sectionRenderer instanceof me.cortex.voxy.client.core.rendering.section.backend.mdic.MDICSectionRenderer mdic
                    && viewport instanceof me.cortex.voxy.client.core.rendering.section.backend.mdic.MDICViewport mv) {
                // Age the geometry-free queue once per frame, on the render thread, so addresses whose
                // section was removed return to the arena only after no in-flight command buffer can
                // still be drawing from them. Before the draws rather than after, so the queue advances
                // even on frames that draw nothing. See IGeometryManager.releaseRetiredFrees.
                this.nodeManager.releaseRetiredFrees(mv.frameId);
                mdic.renderOpaqueMetal(enc, mv);
                mdic.renderTemporalMetal(enc, mv);
                // Phase D (issue #11): in vx-contract mode translucent LOD
                // renders in its OWN pass below — separate colour bridge
                // (premultiplied accumulation -> the pack's colortex16
                // layer) + separate depth (-> vxDepthTexTrans) so the
                // pack's deferred composites LOD water as WATER instead of
                // shading it as opaque land.
                if (!this.deferTranslucency
                        && !me.cortex.voxy.client.core.util.IrisUtil.vxContractActive()) {
                    mdic.renderTranslucentMetal(enc, mv);
                }
            }
        }
        // Iris gbuffer injection: export the LOD pass's depth into the R32F
        // depth bridge so the GL-side injector can unproject it back into
        // MC clip space and write pack-visible gl_FragDepth. Encoded into the
        // SAME command buffer as the LOD pass (encoder order = the barrier),
        // so the submit() below covers it — no extra waits. Bridge alloc
        // mirrors metalBridge's resize discipline above.
        final long perfT3 = System.nanoTime();
        this.perfEncode += perfT3 - perfT2;  // encoding the LOD pass (CPU side)

        // The Hi-Z readback probes that used to be encoded here are GONE: three GPU->CPU blits with
        // their stall (the depth-as-colour attachment twice, the pyramid's mip 0 once, and the albedo
        // attachment under FORCE_MAGENTA), each followed by a full-screen memGet loop on the CPU, to
        // answer a question that is now answered -- the pyramid holds real depth and the cull culls.
        // Recoverable from git if the cull ever misbehaves again; the commit that removed them says
        // what each one measured.
        backend.submit();
        this.perfSubmit += System.nanoTime() - perfT3;
        this.sampleGpuTime(backend);
        this.perfReport();
        this.metalFrame++;


        // Two orphaned comment blocks stood here, describing [Metal-VXPLANES] and [Metal-DEPTHDIAG]
        // read-backs whose code had already been removed before this cleanup began -- a reminder that
        // a comment outlives the code it explains unless someone deletes it.
        //
        // The [Metal-FLICKER] renderList-variance probe that read GPU-written memory HERE, on every
        // frame, is GONE -- and it is worth recording why it could not simply be sampled less often,
        // because that constraint is not obvious: its entire meaning is variance ACROSS frames
        // (`changes` counts frames where the count moved), so reading it once per 600 frames would
        // have made `changes` structurally near-zero while still printing under the same label. The
        // only honest options were to pay for it per frame or to remove it, and the flicker it
        // diagnosed is fixed. Recoverable from git.
    }


    // Phase C material g-buffer planes (null unless vxMaterialMode rendered a frame).

    public void addDebug(List<String> debug) {
        this.sectionRenderer.addDebug(debug);
        RenderStatistics.addDebug(debug);
    }

    // The four GL pipeline hooks (setup / postOpaquePreTranslucent / setupAndBindOpaque /
    // setupAndBindTranslucent) stood here. They existed to drive the GL render pipeline's
    // framebuffers, SSAO compute pass and final blit, all of which are deleted; the Metal
    // path builds its pass in runPipelineMetal and needs none of them.


    public void bindUniforms() {
        this.bindUniforms(-1);
    }

    public void bindUniforms(int index) {
    }

    //null means no function, otherwise return the taa injection function
    public String taaFunction(String functionName) {
        return this.taaFunction(-1, functionName);
    }

    public String taaFunction(int uboBindingPoint, String functionName) {
        return null;
    }

    /**
     * Phase C (issue #11): when true, the Metal LOD terrain pipelines render a
     * material g-buffer (quads.frag under PATCHED_SHADER + VOXY_VX_GBUFFER + the
     * {@link me.cortex.voxy.client.core.util.MetalVxGbufferEmitter} into 3 BGRA8
     * planes) instead of compositing a final colour, for the GL-side
     * {@link me.cortex.voxy.client.core.util.MetalVxResolvePass} to shade. Only
     * {@code MetalVxRenderPipeline} (Metal + Iris vx-contract) enables it.
     */
    public boolean vxMaterialMode() {
        return false;
    }

    /**
     * Whether the OPAQUE LOD layer also goes through the material g-buffer + BSL resolve.
     * Default FALSE: opaque LODs render on the proven base path (lit colour → bridge →
     * normal composite), which is what's stable on dev — the resolve runs ONLY on the
     * translucent (water) layer (issue #11). The full opaque-material path darkened far
     * terrain (BSL deferred shading of grazing LOD); kept behind VOXY_VX_MATERIAL_OPAQUE=1
     * for A/B only. Trans-only is the mergeable shape: water BSL-shaded, opaque untouched.
     */
    public static final boolean VX_MATERIAL_OPAQUE = "1".equals(System.getenv("VOXY_VX_MATERIAL_OPAQUE"));

    /** Opaque LOD uses the material g-buffer + resolve only when explicitly opted in. */
    public boolean vxOpaqueMaterialMode() {
        return vxMaterialMode() && VX_MATERIAL_OPAQUE;
    }

    //null means dont transform the shader
    public String patchOpaqueShader(AbstractSectionRenderer<?,?> renderer, String input) {
        return null;
    }

    //Returning null means apply the same patch as the opaque
    public String patchTranslucentShader(AbstractSectionRenderer<?,?> renderer, String input) {
        return null;
    }

    //Null means no scaling factor
    public float[] getRenderScalingFactor() {return null;}

}
