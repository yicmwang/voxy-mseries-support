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
     * VOXY_RING_FRAME_BUFFERS=1 gives the per-frame position buffer one slot per command buffer that
     * can be in flight, instead of the single copy it has always been.
     *
     * <p>Why this is the prime suspect for bug 3. cmdgen writes {@code positionBuffer[drawId]} for every
     * draw in the frame, and the vertex stage reads it as that draw's BASE POINT. So a draw that reads a
     * different frame's entry gets another section's origin -- while keeping its OWN quads and their own
     * baked light. That is the artefact's description almost word for word: LOD geometry drawn in a
     * section it does not belong to, still carrying the lighting it was baked with. A partial overwrite
     * mid-pass would give scattered debris rather than one coherent wrong image, which is also what is
     * seen.
     *
     * <p>In GL a single copy is safe: cmdgen's write and the draws' read are commands in ONE ordered
     * stream, so frame N+1's write necessarily lands after frame N's draws have read. This port runs
     * whole-frame Metal with {@code MAX_SUBMITS_IN_FLIGHT} command buffers in flight, so frame N+1's
     * cmdgen can overwrite the buffer while frame N's draws are still reading it.
     *
     * <p>Slot 0 IS {@link #positionScratchBuffer}, so with the flag off every caller keeps using exactly
     * the buffer it used before, and the GL path and the CPU-side validation keep reading slot 0.
     */
    private static final boolean RING_FRAME_BUFFERS = "1".equals(System.getenv("VOXY_RING_FRAME_BUFFERS"));
    private static final int FRAME_SLOTS = 3;

    private final IGpuBuffer[] positionRing = new IGpuBuffer[FRAME_SLOTS];
    {
        this.positionRing[0] = this.positionScratchBuffer;
        for (int i = 1; i < FRAME_SLOTS; i++) {
            this.positionRing[i] = RenderBackendFactory.get().createBuffer(8*400000).zero();
        }
    }

    /**
     * The position buffer THIS frame's cmdgen writes and THIS frame's draws read. Callers that write
     * it and read it for the same frame must resolve it through here so they land on the same slot.
     */
    public IGpuBuffer positionScratch(long frameId) {
        return RING_FRAME_BUFFERS
                ? this.positionRing[(int) ((frameId & 0x7fffffff) % FRAME_SLOTS)]
                : this.positionScratchBuffer;
    }

    // ---------------------------------------------------------------------------------------------
    // Draw-command slots: one frame deep, so the CPU read needs no wait.
    //
    // These are the buffers the bug was actually about. MetalRenderEncoder reads `baseInstance` out of
    // the draw commands on the CPU and pushes it as a per-draw constant, because Apple Silicon does not
    // propagate it to [[base_instance]] for drawIndexedPrimitives:indirectBuffer:. In GL that read is
    // safe by construction -- cmdgen's write and the draw's read are commands in one ordered stream --
    // but on Metal with three command buffers in flight it is a frame stale unless either the CPU waits
    // for the prepasses (what the shipped fix does, and what costs the stall) or the read is taken from
    // a slot that is already complete.
    //
    // Hence: cmdgen writes slot N; the DRAWS consume slot N-1. Frame N-1's prepasses have certainly
    // completed, so the CPU read needs no wait at all -- and because the indirect args and the pushed
    // constant come from the SAME slot, the two agents agree, which is exactly the property whose
    // violation was bug 3.
    //
    // Ringing positionScratchBuffer instead was tried and did nothing: its lifetime was never the
    // problem, the INDEX was. These two are the ones the CPU reads.
    // ---------------------------------------------------------------------------------------------

    private final IGpuBuffer[] drawCallRing = new IGpuBuffer[FRAME_SLOTS];
    private final IGpuBuffer[] drawCountRing = new IGpuBuffer[FRAME_SLOTS];
    {
        this.drawCallRing[0] = this.drawCallBuffer;
        this.drawCountRing[0] = this.drawCountCallBuffer;
        for (int i = 1; i < FRAME_SLOTS; i++) {
            this.drawCallRing[i] = RenderBackendFactory.get()
                    .createBuffer(5*4*(400_000+100_000+100_000)).zero();
            this.drawCountRing[i] = RenderBackendFactory.get().createBuffer(1024).zero();
        }
    }

    /** The slot a frame's cmdgen writes. */
    public IGpuBuffer drawCallWrite(long frameId) {
        return RING_FRAME_BUFFERS
                ? this.drawCallRing[(int) ((frameId & 0x7fffffff) % FRAME_SLOTS)] : this.drawCallBuffer;
    }

    /** The slot the draws consume: one frame behind the writer, so its contents are complete. */
    public IGpuBuffer drawCallConsume(long frameId) {
        return RING_FRAME_BUFFERS
                ? this.drawCallRing[(int) (((frameId - 1) & 0x7fffffff) % FRAME_SLOTS)] : this.drawCallBuffer;
    }

    /** The slot a frame's cmdgen writes its counts into. */
    public IGpuBuffer drawCountWrite(long frameId) {
        return RING_FRAME_BUFFERS
                ? this.drawCountRing[(int) ((frameId & 0x7fffffff) % FRAME_SLOTS)] : this.drawCountCallBuffer;
    }

    /** The counts slot the draws consume, one frame behind the writer. */
    public IGpuBuffer drawCountConsume(long frameId) {
        return RING_FRAME_BUFFERS
                ? this.drawCountRing[(int) (((frameId - 1) & 0x7fffffff) % FRAME_SLOTS)] : this.drawCountCallBuffer;
    }

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
        for (int i = 1; i < FRAME_SLOTS; i++) {
            this.positionRing[i].free();
            this.drawCallRing[i].free();
            this.drawCountRing[i].free();
        }
    }

    @Override
    public IGpuBuffer getRenderList() {
        return this.indirectLookupBuffer;
    }
}
