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
    public void ensureAllocated(int width, int height) {
        int targetW = Integer.highestOneBit(width);
        int targetH = Integer.highestOneBit(height);
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
                encoder.setViewport(0, 0, cw, ch, 0, 1);
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
