package me.cortex.voxy.client.core.model.bakery;

import me.cortex.voxy.client.core.gpu.BackendType;
import me.cortex.voxy.client.core.gpu.IGpuSampler;
import me.cortex.voxy.client.core.gpu.IGpuTexture;
import me.cortex.voxy.client.core.gpu.RenderBackend;
import me.cortex.voxy.client.core.gpu.RenderBackendFactory;
import me.cortex.voxy.client.core.metal.MetalTexture;
import me.cortex.voxy.client.core.rendering.util.AtlasMirror;
import org.joml.Matrix4f;
import org.lwjgl.system.MemoryUtil;

import static org.lwjgl.opengl.GL11C.GL_RGBA8;
import static org.lwjgl.opengl.GL11C.GL_TEXTURE_2D;

/**
 * M13 chunk 1 (2026-05-13): Metal-side counterpart to {@link GlViewCapture}.
 * Owns the 48×32 Shared-storage Metal bake target, the
 * {@link MetalBudgetBufferRenderer}, and the {@link AtlasMirror} that lifts
 * MC's GL atlas onto Metal. Exposes the same API shape
 * ({@link #emitToStream}) so {@link ModelTextureBakery} can dispatch to
 * either GL or Metal at run-time without per-pixel format conversion in
 * the caller.
 *
 * <p>Per-bake flow used by {@code renderToStreamMetal}:
 * <pre>
 *   capture.beginBake(mcAtlasGlId, meshAddr, quadCount);
 *   for (int face = 0; face &lt; 6; face++) {
 *       capture.renderFace(face%3, face/3, matrix);
 *   }
 *   capture.endBake();           // ends pass + submit + GPU wait
 *   capture.emitToStream(dest);  // CPU read + pack into uvec2-per-pixel
 * </pre>
 *
 * <p>Output format matches {@link GlViewCapture#emitToStream} pixel-for-pixel
 * so {@code ModelStore} consumers don't need to branch on backend:
 * <pre>
 *   outA[0..3] = packed RGBA (little-endian: R at lowest byte)
 *   outA[4..7] = (depthBits &lt;&lt; 8) | (tintBit &lt;&lt; 7)
 * </pre>
 * There are no real depth or stencil buffers on the Metal path (single RGBA8
 * colour attachment), so the low byte of the second word carries a synthetic
 * "the model drew this pixel" marker instead. That marker is the channel the
 * model metadata is read from — see {@link BakeCoverage}, which owns the
 * snapshot/dilate/pack rules and is unit-tested.
 *
 * <p>The single-attachment shape sidesteps a Metal-side
 * {@code RenderPassDesc.colorAttachment(N, ...)} multi-binding investigation
 * — the bakery's MVP can ship without it, and a follow-up can swap to a
 * two-attachment variant once we verify the descriptor supports it on Metal.
 */
public final class MetalViewCapture {
    private final int width;        // per-face cell width
    private final int height;       // per-face cell height
    private final int totalW;       // 3 * width
    private final int totalH;       // 2 * height

    private final RenderBackend backend;
    private final MetalTexture bakeTarget;
    private final AtlasMirror atlasMirror;
    /** Atlas sampler, created on first bake and shared by both atlas sources. */
    private IGpuSampler sampler;
    private final MetalBudgetBufferRenderer renderer;
    private final long readbackBuffer;
    private final long readbackBytes;
    private boolean activeBake;

    /**
     * Opt-out for the pre-readback GPU sync ({@code VOXY_BAKE_NO_SYNC=1}). Only exists so the fix can
     * be A/B'd against the unsynchronised readback; leaving it on reproduces the black splotches.
     */
    private static final boolean NO_SYNC = "1".equals(System.getenv("VOXY_BAKE_NO_SYNC"));

    /** Diagnostic only ({@code VOXY_BAKE_NO_DILATE=1}): skip the atlas dilation. Not a fix. */
    private static final boolean NO_DILATE = "1".equals(System.getenv("VOXY_BAKE_NO_DILATE"));

    public MetalViewCapture(int width, int height) {
        this.width = width;
        this.height = height;
        this.totalW = width * 3;
        this.totalH = height * 2;
        this.backend = RenderBackendFactory.get();
        if (this.backend.getType() == BackendType.OPENGL) {
            throw new IllegalStateException(
                    "MetalViewCapture is Metal-only — GL goes through GlViewCapture.");
        }

        MetalTexture t = (MetalTexture) this.backend.createTexture(GL_TEXTURE_2D);
        t.storeRenderTargetUploadable(GL_RGBA8, 1, this.totalW, this.totalH);
        t.name("Voxy.MetalBakeTarget");
        this.bakeTarget = t;

        this.atlasMirror = new AtlasMirror();
        this.renderer = new MetalBudgetBufferRenderer();

        this.readbackBytes = (long) this.totalW * this.totalH * 4L;
        this.readbackBuffer = MemoryUtil.nmemAllocChecked(this.readbackBytes);
    }

    /**
     * Open the render pass with a clear-to-zero load and stage the mesh +
     * atlas + sampler bindings. {@code meshAddr} must point at packed
     * quad vertex data in the same {@link BudgetBufferRenderer#VERTEX_FORMAT_SIZE}-stride
     * layout the GL path uses; {@code mcAtlasGlId} is MC's
     * {@code textures/atlas/blocks.png} GL texture id (the
     * {@link AtlasMirror} CPU-reads it via {@code nglGetTexImage} and uploads
     * to the Metal mirror only when the id changes).
     */
    /**
     * Clear the bake target to transparent black. Opens and closes a tiny
     * render pass that only does a CLEAR load action — no draws. Lets the
     * caller separate "clear once at start of bake" from "draw each face",
     * which is essential for the fluid path where the face draws happen in
     * per-face passes (each with LOAD) to allow per-face mesh re-uploads
     * outside the active render encoder.
     */
    public void clear() {
        if (this.activeBake) {
            throw new IllegalStateException("clear() while a bake pass is active");
        }
        // Open + immediately close a CLEAR pass. The Metal driver collapses
        // this into a single clearColor command on the bake target.
        try (var enc = this.backend.beginRenderPass(
                me.cortex.voxy.client.core.gpu.RenderPassDesc.builder(this.totalW, this.totalH)
                        .clearColor(this.bakeTarget, 0f, 0f, 0f, 0f)
                        .build())) {
            enc.setViewport(0, 0, this.totalW, this.totalH, 0, 1);
        }
        this.backend.submit();

        // PROBE (VOXY_BAKE_CLEARCHK): confirm the clear actually landed, by settling the GPU and
        // reading the target back. A settled read of a bake that was never cleared still shows the
        // previous bake's pixels, which is indistinguishable from a bake that genuinely covered the
        // cell -- and since dilation makes every bake's output full coverage, one missing clear
        // makes every later bake look like a full occluding cube.
        if ("1".equals(System.getenv("VOXY_BAKE_CLEARCHK"))) {
            this.backend.waitForGpuIdle();
            this.bakeTarget.getBytes(0, 0, 0, this.totalW, this.totalH, this.readbackBuffer);
            me.cortex.voxy.common.Logger.info("[Metal-BAKE-CLEARCHK] residual after clear = "
                    + java.util.Arrays.toString(perCellWrittenAlpha(this.readbackBuffer)));
        }
    }

    /**
     * GL path: MC's atlas is a GL texture, so it has to be mirrored onto the Metal device first.
     * {@code mcAtlasGlId} is the id of {@code textures/atlas/blocks.png}.
     */
    public void beginBake(int mcAtlasGlId, long meshAddr, int quadCount, boolean clear) {
        beginBake(this.atlasMirror.syncMetal(mcAtlasGlId), meshAddr, quadCount, clear);
    }

    /**
     * Whole-frame Metal: MC's atlas is <b>already</b> a Metal texture on this device, so it is
     * sampled directly.
     *
     * <p>This is strictly better than the mirror path rather than merely a port of it. The mirror
     * exists to cross a context boundary — bind the GL texture, read every texel back through
     * {@code nglGetTexImage}, upload it to a Metal texture, and repeat whenever the atlas id changes.
     * Under whole-frame Metal there is no boundary to cross: the source is already samplable, so the
     * readback, the staging buffer, the id-change cache and its warm-up heuristic all drop away, and
     * the real mip chain comes along instead of being truncated to level 0.
     */
    public void beginBake(IGpuTexture atlas, long meshAddr, int quadCount, boolean clear) {
        if (this.activeBake) {
            throw new IllegalStateException("beginBake while a previous bake is active");
        }
        if (atlas == null) {
            // MC's atlas not yet ready — leave the bake target zeroed. The
            // pack loop below will emit transparent black pixels which the
            // shader's alpha-discard will skip naturally on the next draw.
            return;
        }
        this.renderer.beginPass(this.bakeTarget, this.totalW, this.totalH, clear);
        this.renderer.setup(meshAddr, quadCount, atlas, this.bakeSampler());
        this.activeBake = true;
    }

    /**
     * Sampler for the atlas, created once. Both paths share {@link AtlasMirror}'s conventions so a
     * block cannot bake differently depending on which one supplied its atlas.
     */
    private IGpuSampler bakeSampler() {
        if (this.sampler == null) {
            this.sampler = AtlasMirror.createAtlasSampler(this.backend);
        }
        return this.sampler;
    }

    /**
     * Render the per-block mesh once into the 16×16 cell at
     * {@code (faceX, faceY)} of the 3×2 grid, using {@code matrix} as the
     * cube-projection transform. Mirrors the per-face draw the GL bakery
     * does at {@link ModelTextureBakery#renderToStream}.
     */
    public void renderFace(int faceX, int faceY, Matrix4f matrix) {
        if (!this.activeBake) return; // beginBake bailed (atlas not ready)
        this.renderer.setViewport(faceX * this.width, faceY * this.height,
                this.width, this.height);
        this.renderer.render(matrix);
    }

    /**
     * Close the render pass and submit. On Metal,
     * {@code RenderBackend.submit()} waits for the command buffer to
     * complete (verified at {@code MetalRenderBackend.submit():691}), so the
     * Shared bake target is CPU-readable as soon as this returns.
     */
    public void endBake() {
        if (!this.activeBake) return;
        this.renderer.endPass();
        this.activeBake = false;
    }

    /**
     * Read the bake target via {@link MetalTexture#getBytes} and pack into the uvec2-per-pixel layout
     * {@link GlViewCapture#emitToStream} produces, with the metadata word carrying true coverage.
     *
     * <p>Two things here are load-bearing and were each the source of the missing-surface-face
     * symptom. The GPU must be synchronised before the readback (see the sync block below), and the
     * coverage marker must come from PRE-dilation data while the colour carries the dilated tile —
     * see {@link BakeCoverage} for why, and {@code BakeCoverageTest} for the pinned version.
     */
    public void emitToStream(long destAddr) {
        // Make the bake's GPU work CPU-visible before reading it back.
        //
        // The readback below is a plain CPU memcpy out of the bake target's Shared storage, and until
        // this call existed nothing ordered it against the render pass that fills it. submit() does
        // NOT imply completion here: under whole-frame Metal Metallum owns the frame command buffer,
        // so MetalRenderBackend.submit() takes its guest branch — MetallumBridge.flushFrame() →
        // MetalCommandEncoder.submit(), which commits with a completion block and then waits only for
        // the submit MAX_SUBMITS_IN_FLIGHT (=3) earlier (MetalCommandEncoder:213), never for the commit
        // it just made, and clears activeCommandBuffer so there is nothing left to wait on.
        // MetalRenderBackend.waitForGpuIdle() is the primitive that does order it.
        //
        // Measured, VOXY_BAKE_RECHECK=1 over 675 bakes: 531 of them (79%) read all six face cells as
        // EMPTY on the immediate read and as fully written 40 ms later. Because the target is cleared
        // at the start of every bake, a read that lands early sees the clear — every face
        // non-existent, so the mesher emits no geometry for that block at all and you see through the
        // terrain into its dark interior — while a read that lands late sees the PREVIOUS block's
        // post-dilation bake, every face 256/256, i.e. a false fully-occluding cube. Both were
        // observed, which is why the same build reported different blocks as full cubes on different
        // runs. After the fix: 0 unsettled reads out of 115, and every block reports its true shape.
        if (!NO_SYNC) {
            this.backend.waitForGpuIdle();
        }

        this.bakeTarget.getBytes(0, 0, 0, this.totalW, this.totalH, this.readbackBuffer);

        // Snapshot true coverage BEFORE dilating. Everything ModelFactory derives — occludesFace,
        // faceCoversFullBlock, the packed quad bounds and depths, fullyOpaque — is computed from these
        // markers, and dilation fills every empty texel of any cell that had one opaque pixel, so
        // reading them afterwards makes each model report itself a full occluding cube.
        final int pixelCount = this.totalW * this.totalH;
        final int[] rgba = new int[pixelCount];
        for (int i = 0; i < pixelCount; i++) {
            rgba[i] = MemoryUtil.memGetInt(this.readbackBuffer + (long) i * 4L);
        }
        final boolean[] drawn = BakeCoverage.snapshotDrawn(rgba);

        // Diagnostic: alpha distribution BEFORE dilation. The [Metal-BAKE]
        // log reads these counters and steady-state shows ~1.5% fullAlpha,
        // ~50% nonzero, ~50% zero — the bakery itself is sparse, which is why
        // dilation matters for the LOD horizon visual.
        int nonzeroAlphaPixels = 0;
        for (int p : rgba) {
            if ((p & 0xFF000000) != 0) nonzeroAlphaPixels++;
        }
        if (nonzeroAlphaPixels > 0) {
            GlViewCapture.DIAG_BAKE_NONZERO_PIXEL_INVOCATIONS.incrementAndGet();
            if (nonzeroAlphaPixels * 2L > pixelCount) {
                GlViewCapture.DIAG_BAKE_FULL_ALPHA_INVOCATIONS.incrementAndGet();
            }
        } else {
            GlViewCapture.DIAG_BAKE_ZERO_ALPHA_INVOCATIONS.incrementAndGet();
        }

        // M13 chunk 1 polish (2026-05-16): bake-fill via per-cell average. The bakery's per-face model
        // rendering only fills the cell where the model has geometry — for non-cube blocks (leaves,
        // fences, slabs, ~98% of baked states) that's 5-30% of pixels, leaving the rest at the
        // RGBA(0,0,0,0) clear value. Without dilation, distant LOD chunks mip-average to alpha≈0 and
        // the shader's discard kicks in, leaving see-through patches on the LOD horizon.
        //
        // This feeds the ATLAS only. The markers packed below deliberately do not see it.
        // VOXY_BAKE_NO_DILATE=1 skips it, which leaves the atlas holey and is a diagnostic only.
        if (nonzeroAlphaPixels > 0 && !NO_DILATE) {
            GlViewCapture.DIAG_BAKE_DILATE_RUNS.incrementAndGet();
            int filled = BakeCoverage.dilateOpaqueIntoGaps(rgba, this.totalW, this.width, this.height);
            GlViewCapture.DIAG_BAKE_DILATE_PIXELS_FILLED.addAndGet(filled);
        }

        BakeCoverage.pack(destAddr, rgba, drawn, this.totalW, this.width, this.height);
    }

    /**
     * Per-face-cell count of pixels with nonzero alpha, in the 3x2 cell order the bake renders into.
     * Counts the raw readback, so it is meaningful whatever the cell layout is; used by the
     * VOXY_BAKE_RECHECK probe to compare two reads of the same texture.
     */
    private int[] perCellWrittenAlpha(long buffer) {
        int[] counts = new int[6];
        for (int cellRow = 0; cellRow < 2; cellRow++) {
            for (int cellCol = 0; cellCol < 3; cellCol++) {
                int n = 0;
                for (int y = cellRow * this.height; y < (cellRow + 1) * this.height; y++) {
                    for (int x = cellCol * this.width; x < (cellCol + 1) * this.width; x++) {
                        long off = ((long) y * this.totalW + x) * 4L;
                        if ((MemoryUtil.memGetInt(buffer + off) & 0xFF000000) != 0) n++;
                    }
                }
                counts[cellRow * 3 + cellCol] = n;
            }
        }
        return counts;
    }


    public void free() {
        this.renderer.shutdown();
        this.atlasMirror.free();
        if (this.sampler != null) { this.sampler.close(); this.sampler = null; }
        if (this.bakeTarget != null) this.bakeTarget.free();
        if (this.readbackBuffer != 0L) MemoryUtil.nmemFree(this.readbackBuffer);
    }
}
