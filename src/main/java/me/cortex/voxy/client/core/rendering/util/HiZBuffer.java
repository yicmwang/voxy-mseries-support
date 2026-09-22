package me.cortex.voxy.client.core.rendering.util;

import me.cortex.voxy.client.core.gpu.shader.ShaderLoader;
import me.cortex.voxy.client.core.gpu.GraphicsPipelineDesc;
import me.cortex.voxy.client.core.gpu.IGpuPipeline;
import me.cortex.voxy.client.core.gpu.IGpuSampler;
import me.cortex.voxy.client.core.gpu.IGpuTexture;
import me.cortex.voxy.client.core.gpu.PipelineState;
import me.cortex.voxy.client.core.gpu.RenderBackend;
import me.cortex.voxy.client.core.gpu.RenderBackendFactory;
import me.cortex.voxy.client.core.gpu.RenderEncoder;
import me.cortex.voxy.client.core.gpu.RenderPassDesc;
import me.cortex.voxy.client.core.gpu.SamplerDesc;
import me.cortex.voxy.client.core.gpu.VertexLayout;

import static org.lwjgl.opengl.GL30C.GL_R32F;

/**
 * Hierarchical-Z buffer — power-of-two depth pyramid sampled by the
 * traversal compute pass for occlusion culling.
 *
 * Build strategy (one per frame): for each mip level i in [0, levels), draw
 * a fullscreen quad with depth-test=DISABLED into the COLOUR attachment at
 * level i, sampling from level (i-1). For i=0 the source is the LOD pass's
 * depth-as-colour attachment; for i&gt;0 the source is this texture's own
 * previous mip, wrapped in a single-mip
 * {@code IGpuTexture.createView(level, 1)} so that {@code textureGather} in
 * {@code hiz/blit.fsh} sees it as LOD 0.
 *
 * <h2>Why this is R32F colour and not a depth texture</h2>
 * A depth-format texture sampled through the {@code sampler2D} that both this
 * blit and the traversal declare becomes MSL {@code texture2d&lt;float&gt;}, and
 * Metal silently reads ZEROS from a depth texture read that way — it needs
 * {@code depth2d}, which SPIRV-Cross only emits for a shadow sampler. This
 * tree measured that once already and wrote it down ({@code quads.frag}'s
 * depth-bound note). So the pyramid is an R32F COLOUR texture, the blit writes
 * {@code colour} instead of {@code gl_FragDepth}, and the frame's reverse-Z
 * convention makes the tile reduction a {@code min} (near is 1, far is 0, and
 * the pyramid keeps the tile's FARTHEST occluder — the smallest value).
 *
 * <h2>The attachment must be a colour attachment, and that was the bug</h2>
 * {@link #zeroFillPyramid} and {@link #buildMipChain(IGpuTexture, int, int)}
 * used to attach this texture with {@code depthAttachment(...)}. The pipeline
 * declares depth DISABLED and an R32F COLOUR format, so the blit's
 * {@code colour} output had no target and was dropped — silently, like every
 * other member of this failure class in this codebase. Nothing wrote the
 * pyramid at any point, which made the cull a no-op no matter what fed it.
 * A colour-only pass with a pipeline that declares a depth format is fine:
 * {@code MetalBudgetBufferRenderer} does exactly that and bakes the atlas.
 */
public class HiZBuffer {

    /** R32F is the only pyramid format — see the class doc for why it is not a depth format. */
    private static final int FORMAT = GL_R32F;

    private final RenderBackend backend = RenderBackendFactory.get();
    private final IGpuPipeline blitPipeline;
    private final IGpuSampler sampler = this.backend.createSampler(SamplerDesc.builder()
            .filter(SamplerDesc.Filter.NEAREST, SamplerDesc.Filter.NEAREST)
            .mipFilter(SamplerDesc.MipFilter.NEAREST)
            .wrap(SamplerDesc.Wrap.CLAMP_TO_EDGE, SamplerDesc.Wrap.CLAMP_TO_EDGE)
            .label("hizBlitSampler")
            .build());

    private IGpuTexture texture;
    private int levels;
    private int width;
    private int height;

    public HiZBuffer() {
        // Depth-test DISABLED: the blit renders a fullscreen quad and writes
        // every texel; there is no depth attachment to test against.
        PipelineState state = new PipelineState(
                PipelineState.DepthState.DISABLED,
                PipelineState.BlendState.OPAQUE,
                PipelineState.RasterState.NO_CULL);
        this.blitPipeline = this.backend.createGraphicsPipeline(new GraphicsPipelineDesc(
                ShaderLoader.parse("voxy:hiz/blit.vsh"),
                ShaderLoader.parse("voxy:hiz/blit.fsh"),
                java.util.Map.of(),
                null, null,                  // no MSL
                null, null,                  // no SPIRV
                FORMAT,
                VertexLayout.EMPTY,
                state,
                "HiZBuffer.blit"));
    }

    private void alloc(int width, int height) {
        this.levels = (int) Math.ceil(Math.log(Math.max(width, height)) / Math.log(2));

        // A resize re-creates the parent texture, so any cached per-mip
        // self-views are now orphans — drop them before allocating fresh.
        freeSelfViews();

        this.texture = this.backend.createTexture()
                .store(FORMAT, this.levels, width, height)
                .name("HiZ");

        this.width = width;
        this.height = height;
    }

    /**
     * Ensure the HiZ texture is allocated for the given viewport size, but
     * do not populate it. Lets callers that can't (yet) build the mip chain
     * still bind the HiZ slot in HOT without dereferencing a null texture.
     * A zero-filled pyramid makes the traversal's test answer "not occluded"
     * for every section — slower than real occlusion, but correct.
     */
    /**
     * {@code VOXY_HIZ_EXACT=1} allocates the pyramid at the EXACT source size instead of rounding down
     * to a power of two. Diagnostic arm for the over-culling bug; see {@link #ensureAllocated}.
     */
    private static final boolean HIZ_EXACT =
            "1".equals(System.getenv("VOXY_HIZ_EXACT"));

    /**
     * {@code VOXY_HIZ_BLIT_YFLIP=0} restores the UNFLIPPED blit viewport, which is the state this
     * pyramid was in until 2026-09-22 and is the controlled A/B arm for the orientation fix below.
     *
     * <p>Default ON, because the unflipped form is wrong on Metal and was removing the whole distant
     * LOD. The long derivation is on {@link #buildMipChain}; the short version is that the blit pairs
     * {@code uv.y = 0} with clip {@code y = -1}, and Metal's viewport puts clip {@code y = -1} at the
     * BOTTOM of the target where GL puts it at the top, so an unflipped blit mirrors every level it
     * writes. The traversal reads the pyramid at the same v the LOD pass wrote, so a mirrored pyramid
     * makes the cull test a distant node against the screen region diametrically opposite it — which
     * at a downward camera is the near ground, whose depth is large, so the node reads "occluded".
     */
    private static final boolean BLIT_YFLIP =
            !"0".equals(System.getenv("VOXY_HIZ_BLIT_YFLIP"));

    /** One-shot, so the orientation in force is readable from the log rather than inferred from the
     *  command line. The project has been bitten twice by a switch that silently never applied, and an
     *  A/B whose two arms are the same build state is indistinguishable from a null result. */
    private static boolean blitYFlipLogged = false;

    public void ensureAllocated(int width, int height) {
        // -----------------------------------------------------------------------------------------
        // THE POWER-OF-TWO ROUNDING, AND WHY IT IS A SUSPECT.
        //
        // `Integer.highestOneBit` rounds the pyramid's mip 0 DOWN to a power of two: for this port's
        // 1708x960 viewport that is 1024x512. The per-mip blit then draws a full-screen quad into each
        // level, so mip 0 is a resample of the depth buffer at a ratio of 1024/1708 = 0.5995 in x and
        // 512/960 = 0.5333 in y -- NOT a halving.
        //
        // The blit's reduce is `textureGather`, which returns the 2x2 block around the sample point.
        // For an exact halving that block is precisely the four children of the destination texel. At a
        // 0.6 ratio it is not: the sample point lands at a non-integer source coordinate, so the 2x2
        // blocks step unevenly across the source and roughly 40% of source columns and 47% of source
        // rows are never gathered at all.
        //
        // A MIN-reduce over a subset is >= the true min, and on reverse-Z larger means NEARER. So the
        // first mip is biased toward NEARER depth -- i.e. toward reporting occlusion that is not there
        // -- and `pointSample > maxBB.z` then culls boxes it should keep. Every coarser level halves
        // exactly (1024 -> 512 -> ...), so the error is confined to mip 0 -- which is the level a
        // SMALL, DISTANT node samples, because the traversal picks its mip from the box's screen
        // footprint. That is the shape of the report: distant LOD sections, at every camera direction,
        // needing the depth buffer to fill in first.
        //
        // Upstream has the same rounding (`voxy/.../HiZBuffer.java` is structurally identical here), so
        // this is not a port divergence -- which is exactly why it needs MEASURING rather than
        // reasoning. One env var, one run: if VOXY_HIZ_EXACT=1 fixes the far field, the resample was
        // the defect.
        // -----------------------------------------------------------------------------------------
        int targetW = HIZ_EXACT ? width : Integer.highestOneBit(width);
        int targetH = HIZ_EXACT ? height : Integer.highestOneBit(height);
        if (this.texture == null || this.width != targetW || this.height != targetH) {
            if (this.texture != null) {
                this.texture.free();
                this.texture = null;
            }
            this.alloc(targetW, targetH);
            // A fresh MTLTexture's contents are UNDEFINED, and the traversal
            // treats "pyramid says 0.0" as "nothing occludes here". Garbage
            // would nondeterministically cull sections, so clear every mip
            // once per allocation.
            this.zeroFillPyramid();
        }
    }

    /** Clears every mip of the pyramid to 0.0 via load-action-only passes. */
    private void zeroFillPyramid() {
        int cw = this.width;
        int ch = this.height;
        for (int i = 0; i < this.levels; i++) {
            try (RenderEncoder ignored = this.backend.beginRenderPass(
                    RenderPassDesc.builder(cw, ch)
                            .addColorAttachment(this.texture, i,
                                    RenderPassDesc.LoadAction.CLEAR,
                                    RenderPassDesc.StoreAction.STORE, 0f, 0f, 0f, 0f)
                            .build())) {
                // no draws — the CLEAR load action does the fill
            }
            cw = Math.max(cw / 2, 1);
            ch = Math.max(ch / 2, 1);
        }
    }

    /**
     * Cross-backend mip-chain build that takes the source depth-as-colour as
     * an {@link IGpuTexture} and binds it via the encoder's
     * {@link RenderEncoder#setTexture}. The pyramid is the same {@link #texture}
     * the traversal samples, so HOT reads a populated pyramid.
     *
     * <p>Per-mip view dance: {@code textureGather} in {@code hiz/blit.fsh}
     * samples at LOD 0, so for mips 1+ we wrap the pyramid texture in a
     * single-mip view targeting level i-1 before binding. Views are cached by
     * level so we don't burn a JNI allocation per frame. The external source
     * (mip 0 input) is bound as-is.
     */
    public void buildMipChain(IGpuTexture srcDepth, int width, int height) {
        this.ensureAllocated(width, height);

        // Lazy-cache one self-view per level so the per-frame allocator stays
        // quiet. The array is sized to `levels` so view[i] feeds mip i+1.
        if (this.selfViews == null || this.selfViews.length != this.levels) {
            freeSelfViews();
            this.selfViews = new IGpuTexture[this.levels];
        }

        int cw = this.width;
        int ch = this.height;
        IGpuTexture currentSource = srcDepth;
        for (int i = 0; i < this.levels; i++) {
            try (RenderEncoder encoder = this.backend.beginRenderPass(
                    RenderPassDesc.builder(cw, ch)
                            .addColorAttachment(this.texture, i,
                                    RenderPassDesc.LoadAction.DONT_CARE,
                                    RenderPassDesc.StoreAction.STORE, 0f, 0f, 0f, 0f)
                            .build())) {
                encoder.setPipeline(this.blitPipeline);
                encoder.setTexture(0, currentSource);
                encoder.setSampler(0, this.sampler);
                // ---------------------------------------------------------------------------------
                // THE BLIT MUST FLIP, OR EVERY LEVEL IT WRITES IS UPSIDE DOWN.
                //
                // blit.vsh emits `gl_Position = vec4(corner*2-1, 0, 1)` with `uv = corner`, i.e. it
                // pairs uv.y=0 with clip y=-1. Metal's viewport maps clip y=-1 to the BOTTOM of the
                // target, where GL maps it to the top; so with a positive-height viewport here, uv.y=0
                // is written to the bottom row while it SAMPLES the source's v=0 — and v=0 is the
                // source's FIRST row in memory, which on Metal is the image's TOP. Output bottom,
                // input top: this pass mirrors, and every mip is therefore mirrored against the level
                // above it, alternating: mip0 is mirrored w.r.t. the source, mip1 w.r.t. mip0 (so
                // upright again), mip2 mirrored again, and so on. Upstream's GL blit has neither
                // problem because GL's v=0 is the bottom, which is exactly where clip y=-1 goes; this
                // is a Metal-convention port bug, not an algorithm difference, and it is invisible in
                // every other fullscreen blit in this tree because those are orientation-symmetric.
                //
                // WHY IT LOOKS LIKE "DISTANT LOD VANISHES". The traversal reads the pyramid at
                // `v = 0.5 + 0.5*ndc.y` (hiz.glsl's toScreenspace), which is the SAME v the LOD pass
                // wrote that fragment to — the traversal is calibrated to the source, and correctly so:
                // the LOD pass renders with a flipped viewport (AbstractRenderPipeline's FLIP_Y), so
                // its attachment's v=0 row really is the top of the screen. A mirrored pyramid breaks
                // that agreement, and the cull then tests each node against the screen region
                // diametrically opposite it. At a downward camera that region is the near ground, whose
                // reverse-Z depth is LARGE, so `pointSample > maxBB.z` is true for any box whose
                // mirrored tile is nearer than it — which is every distant box and no near one. That is
                // the report exactly: the far field goes, the near field stays, at every heading, and
                // only once the near field has been meshed and drawn enough to fill the mirrored tiles,
                // which is why the first ~30 s of a run look correct.
                //
                // A negative height with originY at the target's bottom edge inverts the mapping, so
                // output row and input row agree and the parity problem disappears for ALL levels at
                // once — which is the reason to fix it here rather than by mirroring the lookup in
                // toScreenspace, since that would correct the even mips and break the odd ones.
                // ---------------------------------------------------------------------------------
                if (BLIT_YFLIP) {
                    encoder.setViewport(0, ch, cw, -ch, 0, 1);
                } else {
                    encoder.setViewport(0, 0, cw, ch, 0, 1);
                }
                if (!blitYFlipLogged) {
                    blitYFlipLogged = true;
                    me.cortex.voxy.common.Logger.info("[Metal-HIZORIENT] blit viewport "
                            + (BLIT_YFLIP ? "FLIPPED (0,ch,cw,-ch) -- the fix" : "unflipped (0,0,cw,ch)"
                              + " -- VOXY_HIZ_BLIT_YFLIP=0, the pre-fix arm")
                            + "; pyramid " + this.width + "x" + this.height + " x" + this.levels + " levels");
                }
                encoder.draw(RenderEncoder.PRIMITIVE_TRIANGLE_STRIP, 0, 4, 1, 0);
            }

            cw = Math.max(cw / 2, 1);
            ch = Math.max(ch / 2, 1);

            // Next pass reads from this iteration's just-written mip. Use a
            // cached per-mip view so textureGather samples LOD 0 of that
            // view — i.e. mip i of the parent pyramid.
            if (i + 1 < this.levels) {
                if (this.selfViews[i] == null) {
                    this.selfViews[i] = this.texture.createView(i, 1);
                }
                currentSource = this.selfViews[i];
            }
        }
    }

    /**
     * Cached single-mip views of the HiZ pyramid texture, indexed by parent
     * mip level. Populated lazily by the mip-chain build; held until
     * {@link #free} or a re-alloc clears them. Sized to {@code levels} but
     * only entries 0..levels-2 are ever populated (the last mip needs no
     * self-source view because the build stops).
     */
    private IGpuTexture[] selfViews;

    private void freeSelfViews() {
        if (this.selfViews == null) return;
        for (int i = 0; i < this.selfViews.length; i++) {
            if (this.selfViews[i] != null) {
                this.selfViews[i].free();
                this.selfViews[i] = null;
            }
        }
        this.selfViews = null;
    }

    public void free() {
        freeSelfViews();
        if (this.texture != null) {
            this.texture.free();
            this.texture = null;
        }
        this.sampler.close();
        this.blitPipeline.close();
    }

    public int getHizTextureId() {
        return this.texture.id();
    }

    /** Backend-agnostic accessor used by the traversal to bind the pyramid. */
    public IGpuTexture getHizTexture() {
        return this.texture;
    }

    /**
     * The pyramid's NEAREST/NEAREST/CLAMP_TO_EDGE sampler — the same configuration the traversal
     * builds its own copy of, because the pyramid's texels must be read exactly as written: the
     * occlusion test compares integer mip levels' texels against a box depth, so any filtering
     * invents occluders. Exposed so a second consumer (the per-section cull pass) binds this one
     * instead of a third copy.
     *
     * <p>Ownership stays here: {@link #free()} closes it, so a sharer must NOT close it, and must
     * not outlive the owning {@code HiZBuffer}.
     */
    public IGpuSampler getSampler() {
        return this.sampler;
    }

    public int getPackedLevels() {
        return (this.width << 16) | this.height;
    }
}
