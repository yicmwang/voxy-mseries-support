package me.cortex.voxy.client.core;

import me.cortex.voxy.client.RenderStatistics;
import me.cortex.voxy.client.TimingStatistics;
import me.cortex.voxy.client.VoxyClient;
import me.cortex.voxy.client.core.model.ModelBakerySubsystem;
import me.cortex.voxy.client.core.rendering.Viewport;
import me.cortex.voxy.client.core.rendering.hierachical.AsyncNodeManager;
import me.cortex.voxy.client.core.rendering.hierachical.HierarchicalOcclusionTraverser;
import me.cortex.voxy.client.core.rendering.hierachical.NodeCleaner;
import me.cortex.voxy.client.core.rendering.post.FullscreenBlit;
import me.cortex.voxy.client.core.rendering.section.backend.AbstractSectionRenderer;
import me.cortex.voxy.client.core.rendering.util.DepthFramebuffer;
import me.cortex.voxy.client.core.rendering.util.DownloadStream;
import me.cortex.voxy.common.Logger;
import me.cortex.voxy.common.util.TrackedObject;
import org.joml.Matrix4f;
import org.lwjgl.opengl.GL30;
import org.lwjgl.system.MemoryUtil;

import java.util.List;
import java.util.function.BooleanSupplier;

import static org.lwjgl.opengl.GL11C.GL_ALWAYS;
import static org.lwjgl.opengl.GL11C.GL_DEPTH_TEST;
import static org.lwjgl.opengl.GL11C.GL_EQUAL;
import static org.lwjgl.opengl.GL11C.GL_KEEP;
import static org.lwjgl.opengl.GL11C.GL_REPLACE;
import static org.lwjgl.opengl.GL11C.GL_STENCIL_TEST;
import static org.lwjgl.opengl.GL11C.glColorMask;
import static org.lwjgl.opengl.GL11C.glDisable;
import static org.lwjgl.opengl.GL11C.glEnable;
import static org.lwjgl.opengl.GL11C.glStencilFunc;
import static org.lwjgl.opengl.GL11C.glStencilMask;
import static org.lwjgl.opengl.GL11C.glStencilOp;
import static org.lwjgl.opengl.GL30C.GL_DEPTH24_STENCIL8;
import static org.lwjgl.opengl.GL30C.GL_FRAMEBUFFER;
import static org.lwjgl.opengl.GL30C.glBindFramebuffer;
import static org.lwjgl.opengl.GL42.GL_LEQUAL;
import static org.lwjgl.opengl.GL42.GL_NOTEQUAL;
import static org.lwjgl.opengl.GL42.glDepthFunc;
import static org.lwjgl.opengl.GL42.*;
import static org.lwjgl.opengl.GL45.glClearNamedFramebufferfi;
import static org.lwjgl.opengl.GL45.glGetNamedFramebufferAttachmentParameteri;
import static me.cortex.voxy.client.core.gl.GLCompat.bindTextureUnit;

public abstract class AbstractRenderPipeline extends TrackedObject {
    private final BooleanSupplier frexStillHasWork;

    private final AsyncNodeManager nodeManager;
    private final NodeCleaner nodeCleaner;
    private final HierarchicalOcclusionTraverser traversal;

    protected AbstractSectionRenderer<?,?> sectionRenderer;

    private final FullscreenBlit depthMaskBlit = new FullscreenBlit("voxy:post/fullscreen2.vert", "voxy:post/noop.frag");
    private final FullscreenBlit depthSetBlit = new FullscreenBlit("voxy:post/fullscreen2.vert", "voxy:post/depth0.frag");
    private final FullscreenBlit depthCopy = new FullscreenBlit("voxy:post/fullscreen2.vert", "voxy:post/depth_copy.frag");

    public final DepthFramebuffer fb = new DepthFramebuffer(GL_DEPTH24_STENCIL8);

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

    private static final int DEPTH_SAMPLER = initDepthSampler();

    /**
     * GL sampler used by {@link #initDepthStencil}, which is itself GL-only.
     *
     * <p>Must not be created on a non-GL backend: this is a <em>static initializer</em>, so
     * glGenSamplers runs on class load -- with no GL context it aborts the whole JVM before any
     * exception can be caught. Zero on Metal, where nothing reads it.
     */
    private static int initDepthSampler() {
        if (me.cortex.voxy.client.core.gpu.RenderBackendFactory.get().getType()
                != me.cortex.voxy.client.core.gpu.BackendType.OPENGL) {
            return 0;
        }
        int sampler = glGenSamplers();
        glSamplerParameteri(sampler, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glSamplerParameteri(sampler, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        return sampler;
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

    protected abstract int setup(Viewport<?> viewport, int sourceFramebuffer, int srcWidth, int srcHeight);
    protected abstract void postOpaquePreTranslucent(Viewport<?> viewport);
    protected void finish(Viewport<?> viewport, int sourceFrameBuffer, int srcWidth, int srcHeight) {
        glDisable(GL_STENCIL_TEST);
        glBindFramebuffer(GL_FRAMEBUFFER, sourceFrameBuffer);
    }

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

    /**
     * VOXY_HIZ_BUILD=1 enables the Hi-Z pyramid build on the Metal path.
     *
     * <p>Off by default because the build currently cannot cull: the depth attachment it samples reads
     * as zeros, so the pyramid is all-zero, the traversal's guard fires for every box, and the only
     * measurable effect is the cost of building it (+3.7 to +7 ms of `submit`, ~11 mip-chain passes).
     * The wiring is kept so that the missing piece -- a texture-to-texture copy of MC's depth into a
     * sampleable Voxy-owned texture -- can be added and A/B'd with this one switch.
     */
    private static final boolean HIZ_BUILD = "1".equals(System.getenv("VOXY_HIZ_BUILD"));
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

    // [Metal-FLICKER] diagnostic (2026-05-26): track whether the rendered
    // section set (renderList count) varies frame-to-frame. With a perfectly
    // static camera, a varying count proves NON-DETERMINISTIC section selection
    // (a GPU race in the HOT traversal) — vs a stable count meaning the flicker
    // is view-jitter at the frustum boundary. Logged every 600 frames.
    private int rlCountLast = -1;
    private int rlCountMin = Integer.MAX_VALUE;
    private int rlCountMax = 0;
    private int rlChanges = 0;

    public void runPipeline(Viewport<?> viewport, int sourceFrameBuffer, int srcWidth, int srcHeight) {
        if (me.cortex.voxy.client.core.gpu.RenderBackendFactory.get().getType()
                != me.cortex.voxy.client.core.gpu.BackendType.OPENGL) {
            // Metal path — M12 chunk 6 step 1 runs the migrated compute side
            // of the pipeline (DownloadStream + AsyncNodeManager.tick +
            // NodeCleaner.tick + HOT.doTraversal + MDIC.buildDrawCalls) but
            // still clears the bridge for visual feedback instead of issuing
            // real LOD draws. Subsequent steps migrate renderTerrain /
            // renderTranslucent and replace the clear with encoder draws so
            // Voxy's actual LOD content reaches the IOSurface bridge.
            this.runPipelineMetal(viewport, sourceFrameBuffer);
            return;
        }
        int depthTexture = this.setup(viewport, sourceFrameBuffer, srcWidth, srcHeight);

        var rs = ((AbstractSectionRenderer)this.sectionRenderer);
        rs.renderOpaque(viewport);
        var occlusionDebug = VoxyClient.getOcclusionDebugState();
        if (occlusionDebug==0) {
            this.innerPrimaryWork(viewport, depthTexture);
        }
        if (occlusionDebug<=1) {
            rs.buildDrawCalls(viewport);
        }
        rs.renderTemporal(viewport);

        this.postOpaquePreTranslucent(viewport);

        if (!this.deferTranslucency) {
            rs.renderTranslucent(viewport);
        }

        this.finish(viewport, sourceFrameBuffer, srcWidth, srcHeight);
        glBindFramebuffer(GL_FRAMEBUFFER, sourceFrameBuffer);
    }

    /** Push-block binding for depth_copy.frag's scaleFactor (see Push struct in the shader). */
    private static final int DEPTH_COPY_PUSH_BINDING = 14;
    /** Push-block binding for blit_texture_depth_cutout.frag's PushMats (invProj + proj). */
    private static final int BLIT_DEPTH_MATS_PUSH_BINDING = 14;
    private static final int BLIT_DEPTH_MATS_PUSH_SIZE = 4 * 4 * 4 * 2; // two mat4s

    protected void initDepthStencil(int sourceFrameBuffer, int targetFb, int srcWidth, int srcHeight, int width, int height) {
        glClearNamedFramebufferfi(targetFb, GL_DEPTH_STENCIL, 0, 1.0f, 1);
        // using blit to copy depth from mismatched depth formats is not portable so instead a full screen pass is performed for a depth copy
        // the mismatched formats in this case is the d32 to d24s8
        glBindFramebuffer(GL30.GL_FRAMEBUFFER, targetFb);

        this.depthCopy.bind();
        int depthTexture = glGetNamedFramebufferAttachmentParameteri(sourceFrameBuffer, GL_DEPTH_ATTACHMENT, GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME);
        bindTextureUnit(0, depthTexture);
        glBindSampler(0, DEPTH_SAMPLER);
        // Push scaleFactor (vec2) into the Push UBO declared at DEPTH_COPY_PUSH_BINDING.
        try (var stack = org.lwjgl.system.MemoryStack.stackPush()) {
            long addr = stack.nmalloc(8);
            MemoryUtil.memPutFloat(addr,     ((float) width) / srcWidth);
            MemoryUtil.memPutFloat(addr + 4, ((float) height) / srcHeight);
            this.depthCopy.setBytes(DEPTH_COPY_PUSH_BINDING, addr, 8);
        }
        glColorMask(false,false,false,false);
        this.depthCopy.blit();

        /*
        if (Capabilities.INSTANCE.isMesa){
            glClearStencil(1);
            glClear(GL_STENCIL_BUFFER_BIT);
        }*/

        //This whole thing is hell, we basicly want to create a mask stenicel/depth mask specificiclly
        // in theory we could do this in a single pass by passing in the depth buffer from the sourceFrambuffer
        // but the current implmentation does a 2 pass system
        glEnable(GL_STENCIL_TEST);
        glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
        glStencilFunc(GL_ALWAYS, 0, 0xFF);
        glStencilMask(0xFF);

        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_NOTEQUAL);//If != 1 pass
        //We do here
        this.depthMaskBlit.blit();
        glDisable(GL_DEPTH_TEST);

        //Blit depth 0 where stencil is 0
        glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);
        glStencilFunc(GL_EQUAL, 0, 0xFF);

        this.depthSetBlit.blit();

        glDepthFunc(GL_LEQUAL);
        glColorMask(true,true,true,true);

        //Make voxy terrain render only where there isnt mc terrain
        glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);
        glStencilFunc(GL_EQUAL, 1, 0xFF);
    }

    protected static void transformBlitDepth(FullscreenBlit blitShader, int srcDepthTex, int dstFB, Viewport<?> viewport, Matrix4f targetTransform) {
        // at this point the dst frame buffer doesn't have a stencil attachment so we don't need to keep the stencil test on for the blit
        // in the worst case the dstFB does have a stencil attachment causing this pass to become 'corrupted'
        glDisable(GL_STENCIL_TEST);
        glBindFramebuffer(GL30.GL_FRAMEBUFFER, dstFB);

        blitShader.bind();
        bindTextureUnit(0, srcDepthTex);

        // Push PushMats { mat4 invProjMat; mat4 projMat; } into the UBO at BLIT_DEPTH_MATS_PUSH_BINDING.
        try (var stack = org.lwjgl.system.MemoryStack.stackPush()) {
            long addr = stack.nmalloc(BLIT_DEPTH_MATS_PUSH_SIZE);
            new Matrix4f(viewport.MVP).invert().getToAddress(addr);                 // invProjMat
            targetTransform.getToAddress(addr + 4 * 4 * 4);                          // projMat
            blitShader.setBytes(BLIT_DEPTH_MATS_PUSH_BINDING, addr, BLIT_DEPTH_MATS_PUSH_SIZE);
        }

        glEnable(GL_DEPTH_TEST);
        blitShader.blit();
        glDisable(GL_STENCIL_TEST);
        glDisable(GL_DEPTH_TEST);
    }

    protected void innerPrimaryWork(Viewport<?> viewport, int depthBuffer) {

        //Compute the mip chain
        viewport.hiZBuffer.buildMipChain(depthBuffer, viewport.width, viewport.height);

        do {
            TimingStatistics.main.stop();
            TimingStatistics.dynamic.start();

            TimingStatistics.D.start();
            //Tick download stream
            DownloadStream.INSTANCE.tick();
            TimingStatistics.D.stop();

            this.nodeManager.tick(this.traversal.getNodeBuffer(), this.nodeCleaner);
            //glFlush();

            this.nodeCleaner.tick(this.traversal.getNodeBuffer());//Probably do this here??

            TimingStatistics.dynamic.stop();
            TimingStatistics.main.start();

            glMemoryBarrier(GL_FRAMEBUFFER_BARRIER_BIT | GL_PIXEL_BUFFER_BARRIER_BIT);

            TimingStatistics.F.start();
            this.traversal.doTraversal(viewport);
            TimingStatistics.F.stop();
        } while (this.frexStillHasWork.getAsBoolean());
    }

    @Override
    protected void free0() {
        this.fb.free();
        this.sectionRenderer.free();
        this.depthMaskBlit.delete();
        this.depthSetBlit.delete();
        this.depthCopy.delete();
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

    private void perfReport() {
        if (++this.perfFrames < 600) return;
        double n = this.perfFrames;
        me.cortex.voxy.common.Logger.info(String.format(
                "[Metal-PERF] ms/frame  prep=%.2f  split=%.2f  encode=%.2f  submit=%.2f  total=%.2f",
                this.perfPrep / n / 1e6, this.perfSplit / n / 1e6,
                this.perfEncode / n / 1e6, this.perfSubmit / n / 1e6,
                (this.perfPrep + this.perfSplit + this.perfEncode + this.perfSubmit) / n / 1e6));
        this.perfPrep = this.perfSplit = this.perfEncode = this.perfSubmit = 0;
        this.perfFrames = 0;
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

        // 2b) The Voxy-owned depth texture, allocated BEFORE the build because it is the build's source.
        //     PURE D32F, not packed D24S8: Depth32Float_Stencil8 cannot be sampled as texture2d<float>
        //     (the Iris-inject depth export read zeros from it and every injected LOD pixel discarded),
        //     whereas pure D32F is the sampleable format. MetalTexture.store() creates it with
        //     ShaderRead -- precisely the flag MC's own attachment lacks -- so this is the texture the
        //     pyramid can actually read.
        if (this.metalDepthTex == null || this.metalDepthWidth != fbw || this.metalDepthHeight != fbh) {
            if (this.metalDepthTex != null) this.metalDepthTex.free();
            this.metalDepthTex = backend.createTexture()
                    .store(org.lwjgl.opengl.GL30C.GL_DEPTH_COMPONENT32F, 1, fbw, fbh)
                    .name("VoxyMetalDepth");
            this.metalDepthWidth = fbw;
            this.metalDepthHeight = fbh;
            this.hizBuildLogged = false;   // a resize invalidates whatever the pyramid held
        }
        //    OFF BY DEFAULT, and that is a measured decision rather than caution. The build works --
        //    it runs clean, no Metal validation errors -- but it culls nothing, because the depth
        //    attachment it samples reads as zeros: MC's attachment is Depth32Float (fmt 252) yet cannot
        //    be sampled, which is a USAGE-flag problem, not a format one, and is exactly why the
        //    orphaned metalDepthTex was created as a copy target. Until that copy exists the pyramid is
        //    pure cost: measured +3.7 to +7 ms of `submit` across two runs, for zero culled sections
        //    (`maxDrawCount` unchanged at ~27k). Roughly eleven mip-chain render passes per frame.
        //
        //    So the default path is ensureAllocated -- a zero-filled pyramid, whose guard makes the cull
        //    a no-op, which is the behaviour verified before this was written. VOXY_HIZ_BUILD=1 turns
        //    the build on, which is what makes the eventual fix A/B-able with one switch.
        if (HIZ_BUILD && this.metalDepthTex != null && this.metalDepthTex.id() != -1) {
            viewport.hiZBuffer.buildMipChain(this.metalDepthTex, viewport.width, viewport.height);
            if (!this.hizBuildLogged) {
                this.hizBuildLogged = true;
                me.cortex.voxy.common.Logger.info("[Metal-HIZBUILD] built pyramid from the Voxy-owned"
                        + " sampleable depth copy " + viewport.width + "x" + viewport.height
                        + " (source handle=" + (this.metallumDepth == null ? "null"
                                : Long.toString(this.metallumDepth.metalHandle())) + ")");
            }
        } else {
            viewport.hiZBuffer.ensureAllocated(viewport.width, viewport.height);
            if (!this.hizBuildLogged) {
                this.hizBuildLogged = true;
                // Distinguish the two reasons for not building, because they mean opposite things: the
                // switch being off is the intended default, while a missing attachment is a real fault.
                // Conflating them in one message is how a diagnostic ends up lying about the state it
                // was written to report.
                final String why = !HIZ_BUILD
                        ? "VOXY_HIZ_BUILD is not set -- this is the default, and the pyramid stays"
                          + " zero-filled so the cull is a no-op"
                        : "VOXY_HIZ_BUILD is set but no depth attachment is available";
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

        // M13 2026-05-15 Layer B diagnostic — read back the renderList and
        // drawCountCallBuffer values so we can see exactly how many sections
        // HOT enqueued for rendering, and what dispatch parameters prep.comp
        // wrote for cmdgen. If sectionCount is small, the upstream traversal
        // is the bottleneck. If sectionCount is big but cmdGenDispatchX is
        // small, prep.comp's read of sectionCount is racing or stale.
        // (2026-05-26: logged every 600 frames ≈ 10s, offset from the other
        // 600-frame diag block below, so the translucent draw count — which
        // tells us how much water LOD is actually being drawn — is visible
        // regularly while diagnosing the "no colour" water.)
        if (this.metalFrame % 600 == 300
                && viewport instanceof me.cortex.voxy.client.core.rendering.section.backend.mdic.MDICViewport mv) {
            int renderListSectionCount = -1;
            int cmdGenDispatchX = -1;
            int cmdGenDispatchY = -1;
            int cmdGenDispatchZ = -1;
            int opaqueDrawCount = -1;
            int translucentDrawCount = -1;
            int temporalOpaqueDrawCount = -1;
            if (mv.getRenderList() instanceof me.cortex.voxy.client.core.metal.MetalBuffer rl) {
                renderListSectionCount = org.lwjgl.system.MemoryUtil.memGetInt(rl.getContentsPtr());
            }
            if (mv.drawCountCallBuffer instanceof me.cortex.voxy.client.core.metal.MetalBuffer dc) {
                long p = dc.getContentsPtr();
                cmdGenDispatchX        = org.lwjgl.system.MemoryUtil.memGetInt(p +  0);
                cmdGenDispatchY        = org.lwjgl.system.MemoryUtil.memGetInt(p +  4);
                cmdGenDispatchZ        = org.lwjgl.system.MemoryUtil.memGetInt(p +  8);
                opaqueDrawCount        = org.lwjgl.system.MemoryUtil.memGetInt(p + 12);
                translucentDrawCount   = org.lwjgl.system.MemoryUtil.memGetInt(p + 16);
                temporalOpaqueDrawCount = org.lwjgl.system.MemoryUtil.memGetInt(p + 20);
            }
            int topNodeCount = this.traversal.getTopNodeCount();
            int firstDispatchSize = (topNodeCount + 127) >> 7;
            Logger.info(String.format(
                    "[Metal-LayerB f=%d] topNodeCount=%d firstDispatchSize=%d renderList.sectionCount=%d cmdGenDispatch=(%d,%d,%d) draws opaque=%d translucent=%d temporal=%d",
                    this.metalFrame, topNodeCount, firstDispatchSize,
                    renderListSectionCount,
                    cmdGenDispatchX, cmdGenDispatchY, cmdGenDispatchZ,
                    opaqueDrawCount, translucentDrawCount, temporalOpaqueDrawCount));
        }

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
        boolean bridgeSolidTest = "1".equals(System.getenv("VOXY_BRIDGE_SOLID_TEST"));
        if (bridgeSolidTest) {
            clearR = 0.0f; clearG = 1.0f; clearB = 0.0f;
            if ((this.metalFrame % 600) == 1) {
                Logger.info("[Metal-SOLID-TEST] VOXY_BRIDGE_SOLID_TEST active: bridge=green, LOD draws skipped");
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
                me.cortex.voxy.common.Logger.info("[Metal-DEPTHFMT] MC depth attachment pixelFormat=" + fmt
                        + (fmt == 252 ? " Depth32Float -- sampleable, the Hi-Z can read it directly"
                        : fmt == 260 ? " Depth32Float_Stencil8 -- NOT reliably sampleable as texture2d<float>,"
                                     + " so the Hi-Z needs a pure-D32F copy (metalDepthTex is exactly that)"
                        : " unrecognised -- check MTLPixelFormat"));
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
            me.cortex.voxy.common.Logger.info(
                    "[Metal-DIAG] metallumTarget=" + metallumTarget
                    + " colorHandle=0x" + Long.toHexString(me.cortex.voxy.client.core.metal.MetallumBridge.colorAttachment())
                    + " depthHandle=0x" + Long.toHexString(me.cortex.voxy.client.core.metal.MetallumBridge.depthAttachment())
                    + " encoder=0x" + Long.toHexString(me.cortex.voxy.client.core.metal.MetallumBridge.renderEncoder())
                    + " colorId=" + (this.metallumColor == null ? "null" : this.metallumColor.id())
                    + " depthId=" + (this.metallumDepth == null ? "null" : this.metallumDepth.id())
                    + " useMetallumTarget=" + this.useMetallumTarget);
            me.cortex.voxy.client.core.metal.MetallumBridge.logRenderPassCounters();
        }
        // VOXY_ATTACH_TRACE=1: compare the colour attachment Voxy renders the LOD into against
        // Minecraft's actual main render target. The LOD pass provably runs (no "skipping the LOD
        // pass" warning, useMetallumTarget=true) and provably carries valid geometry, yet a
        // debug-CLEAR of that attachment to magenta never reaches the screen. If these two handles
        // differ, the pass is drawing into a texture that is never composited -- which explains
        // valid draws producing zero pixels without anything else being wrong.
        if ("1".equals(System.getenv("VOXY_ATTACH_TRACE")) && (attachTraceCount++ % 600) == 1) {
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
            me.cortex.voxy.common.Logger.info("[Metal-ATTACH] voxyColor=0x" + Long.toHexString(voxyColor)
                    + "  mcMainTarget=0x" + Long.toHexString(mcColor)
                    + (mcColor != 0 && voxyColor != mcColor ? "   <-- MISMATCH: LOD draws into a texture that is never composited" : ""));
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
            boolean debugClear = "1".equals(System.getenv("VOXY_LOD_DEBUG_CLEAR"));
            passBuilder.addColorAttachment(this.metallumColor, 0,
                            debugClear
                                    ? me.cortex.voxy.client.core.gpu.RenderPassDesc.LoadAction.CLEAR
                                    : me.cortex.voxy.client.core.gpu.RenderPassDesc.LoadAction.LOAD,
                            me.cortex.voxy.client.core.gpu.RenderPassDesc.StoreAction.STORE,
                            debugClear ? 1f : 0f, 0f, debugClear ? 1f : 0f, 1f)
                    .depthAttachment(this.metallumDepth, 0,
                            me.cortex.voxy.client.core.gpu.RenderPassDesc.LoadAction.LOAD,
                            me.cortex.voxy.client.core.gpu.RenderPassDesc.StoreAction.STORE,
                            0f);
        }
        var pass = passBuilder.build();
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

        // Copy this frame's combined depth into the Voxy-owned sampleable texture, for next frame's
        // Hi-Z build. Taken HERE -- after the LOD passes, before submit() -- because at this point MC's
        // attachment holds vanilla terrain AND the LOD just drawn, so one copy carries both the terrain
        // occlusion and the LOD self-occlusion the pyramid needs. Encoded before submit() so it rides
        // the same command buffer rather than adding a wait.
        if (HIZ_BUILD && metallumTarget && this.metalDepthTex != null
                && this.metallumDepth != null && this.metallumDepth.id() != -1) {
            // The destination handle comes from MetalHandleMap, NOT from MetallumBridge.textureHandle:
            // that bridge resolves a BLAZE3D texture via reflection (it exists so Voxy can sample MC's
            // own atlas) and throws "argument type mismatch" on a Voxy IGpuTexture. MetalTexture
            // registers its raw handle here in store(), which is the lookup that matches the type.
            final long dstHandle = me.cortex.voxy.client.core.metal.MetalHandleMap
                    .getHandle(this.metalDepthTex.id());
            if (dstHandle != 0) {
                voxyMb.copyTextureToTexture(this.metallumDepth.metalHandle(), dstHandle, fbw, fbh);
            }
        }
        backend.submit();
        this.perfSubmit += System.nanoTime() - perfT3;
        this.perfReport();
        this.metalFrame++;

        // [Metal-VXPLANES] one-shot CPU read-back of the material g-buffer planes
        // (VOXY_VX_DUMP_PLANES=1). submit() waited, so the IOSurface holds the exact
        // rendered bytes — definitive (no tonemap / overdraw confound). Center-row px.

        // [Metal-DEPTHDIAG] chain-bisect instrumentation: the blit buffer is
        // Shared storage and submit() waited, so its contents are the exact
        // floats the export pass read. Histogram of the center rows tells
        // which chain segment is broken: all-zero = blit/LOD-depth broken
        // (Metal side), real spread = Metal side fine, break is in the
        // export pass or the IOSurface→GL hop. Periodic so world-load
        // progression is visible.

        // [Metal-FLICKER] per-frame: read the renderList section count (shared
        // storage, valid after submit) and track its variance over the window.
        if (viewport instanceof me.cortex.voxy.client.core.rendering.section.backend.mdic.MDICViewport mvf
                && mvf.getRenderList() instanceof me.cortex.voxy.client.core.metal.MetalBuffer rlb) {
            int c = MemoryUtil.memGetInt(rlb.getContentsPtr());
            if (this.rlCountLast != -1 && c != this.rlCountLast) this.rlChanges++;
            this.rlCountLast = c;
            if (c < this.rlCountMin) this.rlCountMin = c;
            if (c > this.rlCountMax) this.rlCountMax = c;
        }

        // M13 diagnostic logging: every ~10s (600 frames at 60fps) report what
        // the Metal render path is actually doing — section count loaded into
        // the GPU geometry buffer, MB used, whether AsyncNodeManager has
        // pending work, and the camera position the LOD ring follows. This
        // is the equivalent of the F3 voxy panel for users who can't easily
        // capture it. Drops to silent once the data lines up cleanly.
        if (this.metalFrame % 600 == 1) {
            int sectionCount = -1;
            var geomData = this.sectionRenderer.getGeometryManager();
            if (geomData instanceof me.cortex.voxy.client.core.rendering.section.geometry.BasicSectionGeometryData bgd) {
                sectionCount = bgd.getSectionCount();
            }
            long usedMb = this.nodeManager.getUsedGeometryCapacity() / (1L << 20);
            long capMb  = this.nodeManager.getGeometryCapacity()    / (1L << 20);
            boolean hasWork = this.nodeManager.hasWork();
            Logger.info(String.format(
                    "[Metal-FLICKER f=%d] renderList count over last ~600 frames: min=%d max=%d changes=%d  (STATIC camera: changes>0 / min!=max => NON-DETERMINISTIC section selection = GPU traversal race; stable => flicker is frustum-edge view-jitter)",
                    this.metalFrame,
                    this.rlCountMin == Integer.MAX_VALUE ? -1 : this.rlCountMin,
                    this.rlCountMax, this.rlChanges));
            this.rlCountMin = Integer.MAX_VALUE; this.rlCountMax = 0; this.rlChanges = 0;
            Logger.info(String.format(
                    "[Metal-DIAG f=%d] sections=%d  geom=%d/%d MB  nodeMgr.hasWork=%s  cam=(%.0f, %.0f, %.0f)",
                    this.metalFrame, sectionCount, usedMb, capMb, hasWork,
                    viewport.cameraX, viewport.cameraY, viewport.cameraZ));
            Logger.info(String.format(
                    "[Metal-CHAIN f=%d] ingestCall=%d  ingestNoLight=%d  ingestQ=%d  ingestProc=%d  rawIngest=%d  worldEvt=%d  topLvlAdd=%d  geomResult=%d",
                    this.metalFrame,
                    me.cortex.voxy.common.world.service.VoxelIngestService.DIAG_ENQUEUE_CALL_COUNT.get(),
                    me.cortex.voxy.common.world.service.VoxelIngestService.DIAG_ENQUEUE_NO_LIGHTING_COUNT.get(),
                    me.cortex.voxy.common.world.service.VoxelIngestService.DIAG_ENQUEUE_COUNT.get(),
                    me.cortex.voxy.common.world.service.VoxelIngestService.DIAG_PROCESS_COUNT.get(),
                    me.cortex.voxy.common.world.service.VoxelIngestService.DIAG_RAW_INGEST_COUNT.get(),
                    me.cortex.voxy.client.core.rendering.hierachical.AsyncNodeManager.DIAG_WORLD_EVENT_COUNT.get(),
                    me.cortex.voxy.client.core.rendering.hierachical.AsyncNodeManager.DIAG_TOP_LEVEL_ADD_COUNT.get(),
                    me.cortex.voxy.client.core.rendering.hierachical.AsyncNodeManager.DIAG_GEOMETRY_RESULT_COUNT.get()));
            Logger.info(String.format(
                    "[Metal-LIGHT f=%d] noSkyLayer=%d  bothNull_chunk=%d  blockOnly=%d  (ingestQ=%d)",
                    this.metalFrame,
                    me.cortex.voxy.common.world.service.VoxelIngestService.DIAG_LIGHT_NO_SKY.get(),
                    me.cortex.voxy.common.world.service.VoxelIngestService.DIAG_LIGHT_NONE_CHUNK.get(),
                    me.cortex.voxy.common.world.service.VoxelIngestService.DIAG_LIGHT_HALF.get(),
                    me.cortex.voxy.common.world.service.VoxelIngestService.DIAG_ENQUEUE_COUNT.get()));
            if (me.cortex.voxy.common.world.service.VoxelIngestService.DIAG_CMP_SAMPLES.get() > 0) {
                Logger.info(String.format(
                        "[Metal-CMP   f=%d] samples=%d  agree=%.2f%%  MCbrighter=%.2f%%  voxyBrighter=%.2f%%",
                        this.metalFrame,
                        me.cortex.voxy.common.world.service.VoxelIngestService.DIAG_CMP_SAMPLES.get(),
                        100.0 * me.cortex.voxy.common.world.service.VoxelIngestService.DIAG_CMP_AGREE.get()
                                / Math.max(1, me.cortex.voxy.common.world.service.VoxelIngestService.DIAG_CMP_SAMPLES.get()),
                        100.0 * me.cortex.voxy.common.world.service.VoxelIngestService.DIAG_CMP_MC_BRIGHTER.get()
                                / Math.max(1, me.cortex.voxy.common.world.service.VoxelIngestService.DIAG_CMP_SAMPLES.get()),
                        100.0 * me.cortex.voxy.common.world.service.VoxelIngestService.DIAG_CMP_VOXY_BRIGHTER.get()
                                / Math.max(1, me.cortex.voxy.common.world.service.VoxelIngestService.DIAG_CMP_SAMPLES.get())));
            }
            if (me.cortex.voxy.common.world.service.VoxelIngestService.DIAG_VOXEL_SOLID.get() > 0) {
                Logger.info(String.format(
                        "[Metal-VOXEL f=%d] solid=%d dark=%.2f%%  |  exposedSurface=%d dark=%.2f%%",
                        this.metalFrame,
                        me.cortex.voxy.common.world.service.VoxelIngestService.DIAG_VOXEL_SOLID.get(),
                        100.0 * me.cortex.voxy.common.world.service.VoxelIngestService.DIAG_VOXEL_SOLID_DARK.get()
                                / Math.max(1, me.cortex.voxy.common.world.service.VoxelIngestService.DIAG_VOXEL_SOLID.get()),
                        me.cortex.voxy.common.world.service.VoxelIngestService.DIAG_VOXEL_TOP.get(),
                        100.0 * me.cortex.voxy.common.world.service.VoxelIngestService.DIAG_VOXEL_TOP_DARK.get()
                                / Math.max(1, me.cortex.voxy.common.world.service.VoxelIngestService.DIAG_VOXEL_TOP.get())));
                Logger.info(String.format(
                        "[Metal-FACE  f=%d] selfLit=%d dark=%.2f%%   neighLit=%d dark=%.2f%%",
                        this.metalFrame,
                        me.cortex.voxy.client.core.rendering.building.RenderDataFactory.DIAG_FACE_SELF.get(),
                        100.0 * me.cortex.voxy.client.core.rendering.building.RenderDataFactory.DIAG_FACE_SELF_DARK.get()
                                / Math.max(1, me.cortex.voxy.client.core.rendering.building.RenderDataFactory.DIAG_FACE_SELF.get()),
                        me.cortex.voxy.client.core.rendering.building.RenderDataFactory.DIAG_FACE_NEIGH.get(),
                        100.0 * me.cortex.voxy.client.core.rendering.building.RenderDataFactory.DIAG_FACE_NEIGH_DARK.get()
                                / Math.max(1, me.cortex.voxy.client.core.rendering.building.RenderDataFactory.DIAG_FACE_NEIGH.get())));
            }
            if (me.cortex.voxy.client.core.rendering.building.RenderDataFactory.DIAG_ALL_FACES.get() > 0) {
                Logger.info(String.format(
                        "[Metal-LITAUD f=%d] meshedFaces=%d  zeroLight=%.2f%%   <- camera-independent; compare THIS across builds",
                        this.metalFrame,
                        me.cortex.voxy.client.core.rendering.building.RenderDataFactory.DIAG_ALL_FACES.get(),
                        100.0 * me.cortex.voxy.client.core.rendering.building.RenderDataFactory.DIAG_ALL_DARK.get()
                                / Math.max(1, me.cortex.voxy.client.core.rendering.building.RenderDataFactory.DIAG_ALL_FACES.get())));
                Logger.info(String.format(
                        "[Metal-DARKSRC f=%d] darkFaces fromAIR=%d  fromSOLID=%d  (both meshing paths)",
                        this.metalFrame,
                        me.cortex.voxy.client.core.rendering.building.RenderDataFactory.DIAG_ALL_DARK_FROM_AIR.get(),
                        me.cortex.voxy.client.core.rendering.building.RenderDataFactory.DIAG_ALL_DARK_FROM_SOLID.get()));
                Logger.info(String.format(
                        "[Metal-CULL  f=%d] byOccludes_fullCube=%d  byOccludes_NONcube=%d  bySameModel=%d  byFullyOpaque=%d",
                        this.metalFrame,
                        me.cortex.voxy.client.core.rendering.building.RenderDataFactory.DIAG_CULL_OCCLUDES_FULL.get(),
                        me.cortex.voxy.client.core.rendering.building.RenderDataFactory.DIAG_CULL_OCCLUDES_PARTIAL.get(),
                        me.cortex.voxy.client.core.rendering.building.RenderDataFactory.DIAG_CULL_SAME.get(),
                        me.cortex.voxy.client.core.rendering.building.RenderDataFactory.DIAG_CULL_FULLY_OPAQUE.get()));
            }
            if (me.cortex.voxy.client.core.rendering.building.RenderDataFactory.DIAG_FACE_NEIGH_DARK.get() > 0) {
                Logger.info(String.format(
                        "[Metal-NDARK f=%d] darkNeigh lightFromAir=%d  lightFromSolid=%d",
                        this.metalFrame,
                        me.cortex.voxy.client.core.rendering.building.RenderDataFactory.DIAG_NEIGH_DARK_FROM_AIR.get(),
                        me.cortex.voxy.client.core.rendering.building.RenderDataFactory.DIAG_NEIGH_DARK_FROM_SOLID.get()));
                Logger.info(String.format(
                        "[Metal-VOXEL2 f=%d] groundAir=%d  dark=%.2f%%   <- the cells a surface face takes its light from",
                        this.metalFrame,
                        me.cortex.voxy.common.world.service.VoxelIngestService.DIAG_VOXEL_GROUND.get(),
                        100.0 * me.cortex.voxy.common.world.service.VoxelIngestService.DIAG_VOXEL_GROUND_DARK.get()
                                / Math.max(1, me.cortex.voxy.common.world.service.VoxelIngestService.DIAG_VOXEL_GROUND.get())));
            }
            Logger.info(String.format(
                    "[Metal-TICK  f=%d] tickWithResults=%d  tickWithUploads=%d  lastResultSectionCount=%d  basicSectionCount=%d",
                    this.metalFrame,
                    me.cortex.voxy.client.core.rendering.hierachical.AsyncNodeManager.DIAG_TICK_WITH_RESULTS_COUNT.get(),
                    me.cortex.voxy.client.core.rendering.hierachical.AsyncNodeManager.DIAG_TICK_WITH_UPLOADS_COUNT.get(),
                    me.cortex.voxy.client.core.rendering.hierachical.AsyncNodeManager.DIAG_LAST_TICK_SECTION_COUNT.get(),
                    sectionCount));
            Logger.info(String.format(
                    "[Metal-PGR   f=%d] notInMap=%d  reqSingle=%d  reqChild=%d  innerLeaf=%d  notWatched=%d  uploadEmpty=%d  emptyKids=%d  emptyNoKids=%d  uploadReal=%d  topNoDataDeferred=%d",
                    this.metalFrame,
                    me.cortex.voxy.client.core.rendering.hierachical.NodeManager.DIAG_PGR_NOT_IN_MAP.get(),
                    me.cortex.voxy.client.core.rendering.hierachical.NodeManager.DIAG_PGR_REQUEST_SINGLE.get(),
                    me.cortex.voxy.client.core.rendering.hierachical.NodeManager.DIAG_PGR_REQUEST_CHILD.get(),
                    me.cortex.voxy.client.core.rendering.hierachical.NodeManager.DIAG_PGR_INNER_LEAF.get(),
                    me.cortex.voxy.client.core.rendering.hierachical.NodeManager.DIAG_PGR_NOT_WATCHED.get(),
                    me.cortex.voxy.client.core.rendering.hierachical.NodeManager.DIAG_UPLOAD_EMPTY.get(),
                    me.cortex.voxy.client.core.rendering.hierachical.NodeManager.DIAG_UPLOAD_EMPTY_WITH_CHILDREN.get(),
                    me.cortex.voxy.client.core.rendering.hierachical.NodeManager.DIAG_UPLOAD_EMPTY_NO_CHILDREN.get(),
                    me.cortex.voxy.client.core.rendering.hierachical.NodeManager.DIAG_UPLOAD_REAL.get(),
                    me.cortex.voxy.client.core.rendering.hierachical.NodeManager.DIAG_TOP_LEVEL_NO_DATA_DEFER.get()));
            Logger.info(String.format(
                    "[Metal-GEN   f=%d] called=%d  prepThrow=%d  faceThrow=%d  zeroQ=%d  realQ=%d  lastQ=%d",
                    this.metalFrame,
                    me.cortex.voxy.client.core.rendering.building.RenderDataFactory.DIAG_GEN_CALLED.get(),
                    me.cortex.voxy.client.core.rendering.building.RenderDataFactory.DIAG_GEN_PREPARE_THROW.get(),
                    me.cortex.voxy.client.core.rendering.building.RenderDataFactory.DIAG_GEN_FACE_THROW.get(),
                    me.cortex.voxy.client.core.rendering.building.RenderDataFactory.DIAG_GEN_ZERO_QUADS.get(),
                    me.cortex.voxy.client.core.rendering.building.RenderDataFactory.DIAG_GEN_REAL_QUADS.get(),
                    me.cortex.voxy.client.core.rendering.building.RenderDataFactory.DIAG_GEN_LAST_QUADCOUNT.get()));
            Logger.info(String.format(
                    "[Metal-BAKE  f=%d] invocations=%d  nonzeroPixels=%d  fullAlpha=%d  zeroAlpha=%d  dilateRuns=%d  dilateFilled=%d",
                    this.metalFrame,
                    me.cortex.voxy.client.core.model.bakery.GlViewCapture.DIAG_BAKE_INVOCATIONS.get(),
                    me.cortex.voxy.client.core.model.bakery.GlViewCapture.DIAG_BAKE_NONZERO_PIXEL_INVOCATIONS.get(),
                    me.cortex.voxy.client.core.model.bakery.GlViewCapture.DIAG_BAKE_FULL_ALPHA_INVOCATIONS.get(),
                    me.cortex.voxy.client.core.model.bakery.GlViewCapture.DIAG_BAKE_ZERO_ALPHA_INVOCATIONS.get(),
                    me.cortex.voxy.client.core.model.bakery.GlViewCapture.DIAG_BAKE_DILATE_RUNS.get(),
                    me.cortex.voxy.client.core.model.bakery.GlViewCapture.DIAG_BAKE_DILATE_PIXELS_FILLED.get()));
            Logger.info(String.format(
                    "[Metal-PIPE  f=%d] addEntry=%d  cpyBuf=%d  procModel=%d  atlasUpload=%d",
                    this.metalFrame,
                    me.cortex.voxy.client.core.model.ModelFactory.DIAG_ADDENTRY_CALLS.get(),
                    me.cortex.voxy.client.core.model.ModelFactory.DIAG_CPYBUF_CALLBACKS.get(),
                    me.cortex.voxy.client.core.model.ModelFactory.DIAG_PROCESS_MODEL_RESULTS.get(),
                    me.cortex.voxy.client.core.model.ModelFactory.DIAG_ATLAS_UPLOADS.get()));
            Logger.info(String.format(
                    "[Metal-REQ   f=%d] last=%d  total=%d  directRead=%d  downloadRead=%d",
                    this.metalFrame,
                    me.cortex.voxy.client.core.rendering.hierachical.HierarchicalOcclusionTraverser.DIAG_LAST_REQUEST_COUNT.get(),
                    me.cortex.voxy.client.core.rendering.hierachical.HierarchicalOcclusionTraverser.DIAG_TOTAL_REQUEST_COUNT.get(),
                    me.cortex.voxy.client.core.rendering.hierachical.HierarchicalOcclusionTraverser.DIAG_REQUEST_DIRECT_READ_COUNT.get(),
                    me.cortex.voxy.client.core.rendering.hierachical.HierarchicalOcclusionTraverser.DIAG_REQUEST_DOWNLOAD_COUNT.get()));
        }
    }


    // Phase C material g-buffer planes (null unless vxMaterialMode rendered a frame).

    public void addDebug(List<String> debug) {
        this.sectionRenderer.addDebug(debug);
        RenderStatistics.addDebug(debug);
    }

    //Binds the framebuffer and any other bindings needed for rendering
    public abstract void setupAndBindOpaque(Viewport<?> viewport);
    public abstract void setupAndBindTranslucent(Viewport<?> viewport);


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
