package me.cortex.voxy.client.core.rendering.section.backend.mdic;

import me.cortex.voxy.client.core.gpu.IGpuBuffer;
import me.cortex.voxy.client.core.gpu.IGpuIndirectCommandBuffer;
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

    // ---------------------------------------------------------------------------------------------
    // Draw execution through an MTLIndirectCommandBuffer (VOXY_LOD_ICB=1).
    //
    // The draws used to go out as a host loop of drawIndexedPrimitives:indirectBuffer: calls, one per
    // command, each preceded by a setVertexBytes that pushed that command's baseInstance -- because
    // Apple Silicon does not propagate baseInstance to [[base_instance]] for that draw form. An ICB
    // command carries baseInstance natively, so the push, and the per-draw host read that forces the
    // drain, both disappear.
    //
    // RUNG, like the draw-command slots above, and for the same reason. Metal keeps MAX_SUBMITS_IN_FLIGHT
    // command buffers in flight, and an ICB's own command slots are NOT documented as hazard-tracked:
    // Apple documents hazard tracking for resources "you directly bind to an encoder", and an ICB's
    // internal slots are reached only through executeCommandsInBuffer. So a CPU resetWithRange: plus
    // re-populate for frame N+1 while frame N's execute is still running is exactly the unasserted-sync
    // shape that caused bug 3 -- nothing crashes, the wrong commands run. Hence one ICB per frame slot,
    // rotating, so a slot is only rewritten three frames after it was last executed.
    //
    // Allocated lazily and only when the switch is on: 400k commands is not free, and the default build
    // should not pay it.
    // ---------------------------------------------------------------------------------------------

    private static final boolean LOD_ICB = "1".equals(System.getenv("VOXY_LOD_ICB"));

    /**
     * {@code VOXY_LOD_ICB_OPTIMIZE=1} additionally runs {@code optimizeIndirectCommandBuffer:withRange:}
     * over each pass's region, and therefore executes through the CPU-ranged form.
     *
     * <p>The two are inseparable. An optimized range may only be run whole and from its start, and the
     * buffer-driven execute lets the GPU choose the end — so optimizing without also switching the
     * execute is undefined behaviour, not just a missed optimization. This flag switches both, together,
     * on purpose. It implies {@link #LOD_ICB}.
     *
     * <p>It also forces the populate-and-optimize to happen BEFORE the LOD render pass opens, because
     * {@code optimize} is a blit-encoder call and a command buffer has only one encoder at a time. See
     * {@code MDICSectionRenderer.prepareIcb}.
     */
    public static final boolean ICB_OPTIMIZE = "1".equals(System.getenv("VOXY_LOD_ICB_OPTIMIZE"));

    /**
     * One ICB command per slot of {@link #drawCallBuffer}, laid out the SAME way: opaque at 0,
     * translucent at 400 000, temporal at 500 000.
     *
     * <p><b>The three passes must not share a region, and this is the whole reason the layout is
     * mirrored rather than "one pass at a time".</b> All three passes encode into ONE command buffer,
     * and Metal reads both the ICB's commands and the execution range at GPU execution time -- which is
     * after every pass has been encoded. Resetting and repopulating a single shared ICB per pass leaves
     * the GPU executing the LAST pass's commands three times, and the last range, for all three passes.
     * Measured: the LOD rendered as a band torn with holes, because the opaque pass was running the
     * translucent pass's commands. Mirroring the draw buffer's slices means each pass owns its indices
     * and no pass can overwrite another's.
     */
    public static final int ICB_CAPACITY = 400_000 + 100_000 + 100_000;

    /**
     * Range slots per pass: {@code ceil(400_000 / 0x4000)}, the number of execute chunks the largest
     * pass can need. Each chunk needs its own slot because the range is read at GPU execution time --
     * see {@code MetalRenderEncoder.drawIndexedIndirectIcb}.
     */
    public static final int ICB_MAX_CHUNKS = (400_000 + 0x3fff) / 0x4000;

    /** Byte stride between the per-pass execution-range regions inside {@link #icbRangeBuffer}. */
    public static final long ICB_RANGE_STRIDE = ICB_MAX_CHUNKS * 16L;

    /** Which range region a pass owns: opaque, temporal, translucent. Distinct, never shared. */
    public static final int ICB_PASS_OPAQUE = 0;
    public static final int ICB_PASS_TEMPORAL = 1;
    public static final int ICB_PASS_TRANSLUCENT = 2;

    private final IGpuIndirectCommandBuffer[] icbRing;

    /**
     * The {@code {uint32 location; uint32 length;}} the encoder reads to decide how much of the ICB to
     * run. 8 bytes live (MTLIndirectCommandBufferExecutionRange), padded to 16 so it does not share a
     * cache line with anything else.
     *
     * <p>Metal reads this <b>on the GPU at execution time</b> -- MTLRenderCommandEncoder.h: "an indirect
     * buffer from which the device reads the execution range parameter" -- which is what would let a
     * GPU-written draw count drive execution in a later phase. Today the CPU still writes it, because
     * the CPU is what decides how many commands to populate.
     */
    public final IGpuBuffer icbRangeBuffer;

    {
        if (LOD_ICB) {
            this.icbRing = new IGpuIndirectCommandBuffer[FRAME_SLOTS];
            for (int i = 0; i < FRAME_SLOTS; i++) {
                this.icbRing[i] = RenderBackendFactory.get()
                        .createIndirectCommandBuffer(ICB_CAPACITY)
                        .name("voxy-lod-icb-" + i);
            }
            // One 16-byte range per pass. They must be SEPARATE slots for the same reason the ICB
            // regions must be: the range is read at GPU execution time, so a single shared slot would
            // have all three passes reading whichever pass encoded last.
            this.icbRangeBuffer = RenderBackendFactory.get().createBuffer(3 * ICB_RANGE_STRIDE).zero();
            me.cortex.voxy.common.Logger.info("[Metal-ICB] range buffer: 3 passes x "
                    + ICB_MAX_CHUNKS + " chunks x 16 B = " + (3 * ICB_RANGE_STRIDE) + " bytes");
        } else {
            this.icbRing = null;
            this.icbRangeBuffer = null;
        }
    }

    /** The ICB this frame's draws execute, or null when {@code VOXY_LOD_ICB} is off. */
    public IGpuIndirectCommandBuffer drawIcb(long frameId) {
        return this.icbRing == null
                ? null
                : this.icbRing[(int) (((frameId - 1) & 0x7fffffff) % FRAME_SLOTS)];
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
        // The ICB ring and its range buffer exist only under VOXY_LOD_ICB=1, so unlike the rings above
        // there is no slot 0 -- every slot is ours to free, and all of them are null when it is off.
        if (this.icbRing != null) {
            for (IGpuIndirectCommandBuffer icb : this.icbRing) {
                icb.close();
            }
            this.icbRangeBuffer.free();
        }
    }

    @Override
    public IGpuBuffer getRenderList() {
        return this.indirectLookupBuffer;
    }
}
