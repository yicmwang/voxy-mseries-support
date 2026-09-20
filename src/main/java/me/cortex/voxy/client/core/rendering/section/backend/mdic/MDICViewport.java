package me.cortex.voxy.client.core.rendering.section.backend.mdic;

import me.cortex.voxy.client.core.gpu.IGpuBuffer;
import me.cortex.voxy.client.core.gpu.RenderBackendFactory;
import me.cortex.voxy.client.core.rendering.Viewport;
import me.cortex.voxy.client.core.rendering.hierachical.HierarchicalOcclusionTraverser;

public class MDICViewport extends Viewport<MDICViewport> {
    /**
     * VOXY_VIEWPORT_NOZERO=1 skips the zeroing of {@link #indirectLookupBuffer} and
     * {@link #visibilityBuffer} below, restoring the state they were in before those two calls were
     * added. It exists to test one question that no amount of reading can settle: whether that change
     * is what stopped the misplaced-geometry artefact from reproducing.
     *
     * <p>The evidence for asking: a real-coloured camera-drift run at 01:59 (logged as `sha=37e4ddc9`,
     * though the tree carried uncommitted drift work, so that sha is HEAD and not necessarily the code
     * that ran) reproduced the artefact heavily -- max 144,262 near-black px, 6.6% of frames over
     * 10,000, against a clean steady state of 714. The same real-coloured drift configuration at 04:02
     * on `c4ad29a8` reproduced nothing at all (median 19, and its largest frame was the world-load
     * overlay). The only commit in between that changes what the renderer draws is the zeroing: the
     * others are debug flags at their defaults, guards, and a check that reports rather than acts.
     *
     * <p>Default OFF, i.e. the zeroing stays on -- this is a switch for reproducing an old build, not a
     * behaviour change.
     */
    private static final boolean SKIP_ZERO = "1".equals(System.getenv("VOXY_VIEWPORT_NOZERO"));

    private static IGpuBuffer maybeZero(IGpuBuffer b) {
        return SKIP_ZERO ? b : b.zero();
    }

    public final IGpuBuffer drawCountCallBuffer = RenderBackendFactory.get().createBuffer(1024).zero();
    public final IGpuBuffer drawCallBuffer = RenderBackendFactory.get().createBuffer(5*4*(400_000+100_000+100_000)).zero();//400k draw calls
    public final IGpuBuffer positionScratchBuffer  = RenderBackendFactory.get().createBuffer(8*400000).zero();//400k positions
    /**
     * Zeroed like its siblings above. It was not, and its FIRST UINT IS NOT DATA -- it is the render
     * list's length, which `prep.comp` turns into a dispatch size and every consumer of the list uses
     * as an array bound. An un-zeroed allocation therefore hands the first frames a garbage bound and
     * the draws walk a garbage range of section ids. The sibling buffers got `.zero()` and this one
     * was missed, which is the kind of asymmetry that reads as intentional when it is not.
     */
    public final IGpuBuffer indirectLookupBuffer = maybeZero(RenderBackendFactory.get().createBuffer(HierarchicalOcclusionTraverser.MAX_QUEUE_SIZE *4+4));//In theory, this could be global/not unique to the viewport
    public final IGpuBuffer visibilityBuffer;

    public MDICViewport(int maxSectionCount) {
        // Zeroed for the same reason as indirectLookupBuffer: `cmdgen` gates every draw on
        // `(visibilityData[sid] & 0x7fffffff) == frameId`, so an un-zeroed entry that happens to
        // equal the current frame id makes a section render on evidence that was never written.
        this.visibilityBuffer = maybeZero(RenderBackendFactory.get().createBuffer(maxSectionCount*4L));
    }

    @Override
    protected void delete0() {
        super.delete0();
        this.visibilityBuffer.free();
        this.indirectLookupBuffer.free();
        this.drawCountCallBuffer.free();
        this.drawCallBuffer.free();
        this.positionScratchBuffer.free();
    }

    @Override
    public IGpuBuffer getRenderList() {
        return this.indirectLookupBuffer;
    }
}
