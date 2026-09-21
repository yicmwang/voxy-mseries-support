package me.cortex.voxy.client.core.rendering;

import it.unimi.dsi.fastutil.longs.LongOpenHashSet;
import me.cortex.voxy.client.core.gpu.IGpuBuffer;
import me.cortex.voxy.client.core.gpu.RenderBackend;
import me.cortex.voxy.client.core.gpu.RenderBackendFactory;
import me.cortex.voxy.client.core.rendering.util.UploadStream;
import net.minecraft.core.SectionPos;
import org.lwjgl.system.MemoryUtil;

import java.util.Arrays;

/**
 * The set of 16³ sections Sodium has actually built geometry for, packed into a bitmask over the
 * vanilla render-distance square and handed to {@code cmdgen.comp} so the LOD can cull exactly
 * where vanilla drew.
 *
 * <p><b>Why this exists.</b> The LOD must not draw where vanilla already did. Approximating that
 * with "inside the render-distance box" is wrong in both directions: it culls LOD over chunks
 * Sodium has not built yet, which leaves holes at the seam, and it keeps LOD over chunks vanilla
 * did build, which z-fights. The set of built sections is deterministic and Sodium will tell us —
 * so this asks it, and the cull becomes a membership test rather than a guess.
 *
 * <p>The mask is one 64-bit vertical bitmask per chunk COLUMN, indexed relative to the camera''s
 * column, and the query is a single 16x16x16 SECTION: bit {`(secY - camSecY) + 32`} of the column''s
 * mask. Three dimensions, because Sodium enforces a vertical render distance as well as a
 * horizontal one — a column being built says nothing about which of its sections are, so a cull
 * keyed on the column removes LOD sections vanilla never drew. Measured: with the cull off the
 * missing chunks disappear entirely; with it on their positions track the camera, which is what a
 * camera-relative square of column bits produces.
 *
 * <p>Two earlier versions of this query are worth not repeating. The first tested a LOD node''s
 * CENTRE column and removed the whole node — but a node is 2x2 columns at detail 0, so one bit
 * decided four, and a node whose centre was covered while its other columns were not was removed
 * whole: a hole up to 32 blocks across, ringing the rim of vanilla''s coverage. The second tested
 * every column of the node, which fixed that and nothing else, because it was still two-dimensional.
 * A node-level rule cannot be right in any case: the decision is finer than the node.
 *
 * <p>Cost is a bit test in the existing cmdgen pass — no rasterization, no depth blit, no
 * readback, no depth precision to reason about.
 */
public final class BuiltSectionMask {

    /** Section positions (16-block units) Sodium has geometry for. Written from the mixin, read here. */
    private static final LongOpenHashSet BUILT = new LongOpenHashSet();

    public static synchronized void add(final long sectionPos) {
        added++;
        BUILT.add(sectionPos);
    }

    public static synchronized void remove(final long sectionPos) {
        removed++;
        BUILT.remove(sectionPos);
    }

    /**
     * Feeder call counts, split by which way {@code isBuilt()} sent them.
     *
     * <p>These exist because the set size alone cannot say which half of the feeder is broken, and a
     * pinned run measured {@code built=2 maxBuilt=2} over its whole length -- a loaded, rendering
     * world that reported two sections. If {@code removed} runs far ahead of {@code added}, the
     * feeder is being handed sections whose build has not been marked yet at the injection point
     * (HEAD of the region manager's {@code uploadResults}, which does not itself touch {@code
     * isBuilt}), and every section is being retracted instead of claimed. If both are near zero, the
     * funnel itself is not being reached.
     */
    private static long added, removed;

    /** How many sections the set holds. Package-private: the tests and the diagnostics both read it. */
    static synchronized int builtCount() {
        return BUILT.size();
    }

    /** Sodium rebuilds its section manager on a level or render-distance change; the set starts over. */
    public static synchronized void reset() {
        BUILT.clear();
    }

    /**
     * The level whose sections this set holds, so a render-distance change can be told from a level
     * change. Held strongly and deliberately: it is only ever replaced by the next level, so the
     * most it retains is one already-unloaded level until the next world loads -- and comparing
     * through a weak reference instead would let a collection turn into a spurious clear, which is
     * the exact failure this method exists to remove.
     */
    private static Object maskLevel;
    private static long resets, resetsSkipped;

    /**
     * Clear the set only when the LEVEL changed, not when Sodium merely rebuilt its section manager.
     *
     * <p>{@code RenderSectionManager} is constructed on a level change <i>and</i> on every
     * render-distance change, and the mixin cleared the mask on both. Clearing on a distance change
     * left the mask permanently empty, and that is measured rather than argued: with {@code
     * VOXY_VMASK=1} in a run whose render distance went 3 to 2, {@code [Metal-VMASK]} reported
     * {@code built=0 columns=0/81 sections=0 uploads=7} at frame 1202 and then the same numbers,
     * unchanged, at every report through frame 4802 -- three thousand six hundred frames with the
     * mask empty, in a loaded world that was rendering.
     *
     * <p>The reason it never came back is the feed: {@link #BUILT} only ever refills from mesh-upload
     * deltas, and Sodium does not re-mesh a chunk that is already built, so a clear is permanent for
     * every section that stays loaded. An empty mask means the fragment-stage discard never fires,
     * which is the LOD drawn over vanilla everywhere -- the reported near-field overlap. The cull
     * was not mis-shaped during that time; it was OFF.
     *
     * <p>Clearing is not needed for a distance change. The per-frame distance filter already refuses
     * to claim any section outside the render region, so a section the new distance no longer covers
     * cannot set a bit whatever the set holds, and {@link #prune} bounds what is kept. A level change
     * is genuinely different -- the set would otherwise hold another dimension's sections -- so that
     * still clears.
     *
     * @return whether the set was cleared
     */
    public static synchronized boolean resetForLevel(final Object level) {
        if (maskLevel == level) {
            resetsSkipped++;
            return false;
        }
        maskLevel = level;
        resets++;
        BUILT.clear();
        return true;
    }


    /**
     * side, camSecX, camSecY, camSecZ, camBlockX, camBlockY, camBlockZ, pad.
     *
     * <p>Eight, not six, so the {@code uvec2[]} that follows starts at byte 32 and is 8-byte
     * aligned as std430 requires for a vec2 array. Getting that wrong does not fail to compile; it
     * silently shifts every read by four bytes, which is the kind of bug this file has already paid
     * for once.
     *
     * <p>The block-coordinate camera position is here for the fragment-stage cull: a fragment knows
     * its camera-relative offset in blocks (quads3.vert interpolates it), so camera block position
     * plus that offset gives the fragment's own section in all three axes. Without it the shader
     * would have only the camera's section and could not tell where inside it the camera sits.
     */
    private static final int HEADER_UINTS = 8;

    /**
     * How many section-Y values either side of the camera's section one column's bitmask covers:
     * bit {@code (secY - camSecY) + Y_BIAS} for {@code 0 <= bit < 64}, i.e. sections from 32 below
     * the camera to 31 above, or +/-512 blocks. The overworld is 24 sections tall, so this is ample;
     * anything outside it reads as "not covered", which keeps the LOD rather than removing it.
     */
    private static final int Y_BIAS = 32;

    /**
     * How much slack the square carries beyond the render distance, in chunks, so it can be
     * anchored to a WORLD grid instead of being re-centred on the camera every time the camera
     * crosses a chunk.
     *
     * <p>This is the user's observation, and it is the right way round: with the square centred on
     * the camera, its edge moves at exactly camera speed, so the culled region's boundary slides
     * with the player and the LOD edge appears to be dragged along behind the terrain edge — worst
     * when moving fast, which is exactly when the LOD's own streaming is furthest behind. Anchoring
     * to a world grid makes the boundary static in world space; it steps once per ANCHOR_STEP chunks
     * of travel instead of continuously, and a step is a one-frame discontinuity rather than a drag.
     *
     * <p>Set equal to the render distance, so the square is (4*rd+1) columns on a side — 33 at RD 8,
     * which is 8.7 KB of column bitmasks. Any camera position leaves at least rd columns of margin
     * on every side, and the anchor only moves when the camera crosses a multiple of the step.
     */
    private static final int ANCHOR_SLACK = -1;   // -1 = "same as the render distance", resolved in update()

    /** Largest multiple of {@code step} that is <= {@code v}, for negative {@code v} too. */
    static int floorToStep(final int v, final int step) {
        return Math.floorDiv(v, step) * step;
    }

    /**
     * How far outside the render distance an entry is kept, in chunks, before pruning is allowed to
     * drop it. Kept entries are the ones that could re-enter range without Sodium rebuilding them.
     */
    private static final int PRUNE_MARGIN = 4;

    /**
     * Whether an entry is close enough to the camera to be worth keeping.
     *
     * <p>{@link #BUILT} accumulates one entry per section the player has ever been near, because the
     * mixin only ever adds (a section that leaves Sodium's storage without appearing in an upload
     * batch is never removed). Measured across one session: built 675 -> 779 -> 1632 -> 3423 as the
     * player travelled, and the per-frame rebuild of the column masks walks the whole thing.
     *
     * <p>Pruning is safe only because of the distance filter: an entry outside the render region
     * cannot set a bit, so dropping it cannot change what the mask claims today. The margin exists
     * for tomorrow — a section just outside the render distance may come back into range without a
     * rebuild, and if it had been dropped the mask would under-claim until the next one, which shows
     * as an LOD drawn over vanilla (a z-fight) rather than a hole. Anything further out is reached
     * again only by travelling there, which reloads the chunk and re-meshes it, and the mixin adds
     * it back.
     */
    static boolean worthKeeping(final int dx, final int dy, final int dz, final int rd) {
        if (Math.abs(dy) > Y_BIAS) return false;                  // outside the mask's own bit span
        if (withinRenderDistance(dx, dy, dz, rd)) {
            // Claimable now -- but the union claims at any horizontal distance, so bound it by what the
            // square can address, or the set grows without limit along a journey.
            final int addressable = 4 * rd;
            return Math.abs(dx) <= addressable && Math.abs(dz) <= addressable;
        }
        final long reach = (long) rd + PRUNE_MARGIN;
        return (long) dx * dx + (long) dz * dz <= reach * reach;  // or it comes back as the player moves
    }

    /**
     * Drop entries the player has left far behind. Amortised, not per-frame: it walks the whole set,
     * so it runs at most once per {@link #PRUNE_INTERVAL} mask rebuilds.
     */
    private void prune(final int camSecX, final int camSecY, final int camSecZ, final int rd) {
        synchronized (BuiltSectionMask.class) {
            BUILT.removeIf(pos -> !worthKeeping(
                    SectionPos.x(pos) - camSecX,
                    SectionPos.y(pos) - camSecY,
                    SectionPos.z(pos) - camSecZ, rd));
        }
    }

    /** Mask rebuilds between prunes. The set is small, so this only needs to bound growth. */
    private static final long PRUNE_INTERVAL = 600;
    private long updatesSincePrune = 0;

    /**
     * Whether a section at this chunk offset from the camera is inside the region Sodium RENDERS.
     *
     * <p>Sodium renders a Euclidean cylinder of radius {@code renderDistance} centred on the
     * camera's chunk — the same metric the translucent near-cull in quads.frag uses, and for the
     * same reason: it is the metric Sodium actually draws in. It MESHES a square, because it builds
     * every section of every loaded chunk, plus a margin beyond the render distance while chunks
     * load and unload. Measured at a moved camera: built=779 with 45 sections (5.8%) past the render
     * distance, out to Chebyshev 10 against rd=8, concentrated in the square's corners — which are
     * Euclidean up to 11.3 chunks out.
     *
     * <p>Every one of those is meshed and never drawn. A mask that believed {@code isBuilt()} alone
     * culled the LOD there, which is a hole in a ring just outside vanilla's render distance — the
     * "smaller holes around the edges" the user reported. A section outside this cylinder is not
     * drawn, whatever its mesh says.
     */
    static boolean withinRenderCylinder(final int chunkDx, final int chunkDz, final int rd) {
        // Strict, because Sodium's is: OcclusionCuller.testDistance is `a < c*c`, so a section exactly
        // at the radius is NOT drawn. Inclusive here would claim a one-section ring around the whole
        // circle that vanilla leaves empty, which at RD 8 is a ~50-column band of culled LOD with
        // nothing behind it -- small next to the slab term this replaced, but the same kind of error.
        return (long) chunkDx * chunkDx + (long) chunkDz * chunkDz < (long) rd * rd;
    }

    /**
     * Claim a section when its COLUMN is inside the radius of the camera's section. Nothing vertical.
     *
     * <p>This is the vanilla render area, section-quantised: a disc of {@code rd} chunks around the
     * section the player is standing in, which sits still while the player is inside that section and
     * snaps when they enter the next one. Vanilla renders whole columns of the chunks it draws, so the
     * vertical extent is the whole column and the only bound needed is the mask's own bit window.
     *
     * <p><b>Why the vertical term was removed, measured.</b> The rule was Sodium's own
     * {@code OcclusionCuller.testDistance}, {@code (dx²+dz² < rd²) || (|dy| < rd)}, on the reasoning
     * that Sodium's rule must be right. It is right for Sodium, which applies it as a TRAVERSAL bound
     * and then clips the result by the frustum and the occlusion tree — so its slab term never
     * manifests as claimed ground. Applied as the CLAIM rule over the meshed set it does exactly that,
     * and on flat terrain it is vacuous: every section is at the camera's own section Y, so
     * {@code |dy| = 0 < rd} holds everywhere and the horizontal radius stops mattering. From
     * {@code VOXY_VMASK=1} on the superflat fixture at RD 8:
     *
     * <pre>
     *   built=754  outsideChebyshev=497  outsideRenderDistance3D=0 (0.0%)
     *   maxChebyshev=31 (rd=8)  secY -4..-4
     * </pre>
     *
     * Half the set is beyond Chebyshev 8 and it reaches **31 chunks**, with nothing refused. So the mask
     * culled the LOD out to 31 chunks while vanilla drew to 8, and everything in between was culled LOD
     * with nothing behind it. That is the void edge at the LOD/vanilla intersection, and because its
     * boundary is the MESHED set's edge, it drifts as chunks load and unload.
     *
     * <p>Dropping the slab term also retires the altitude report it was added for: a section directly
     * below the camera IS in a rendered column, so claiming it is correct, and the earlier "LODs close
     * to me are absent" came from claiming meshed sections outside the disc, which this no longer does.
     */
    static boolean withinRenderDistance(final int dx, final int dy, final int dz, final int rd) {
        return withinRenderCylinder(dx, dz, rd);
    }

    /**
     * The fragment's section on one axis, reconstructed the way {@code quads.frag} does it: the
     * camera's world position plus the camera-relative offset, floored ONCE.
     *
     * <p>The single floor is the whole point. {@code floor(cam) + floor(rel)} is not
     * {@code floor(cam + rel)}:
     *
     * <pre>
     *   floor(camX) + floor(fragX - camX)  ==  floor(fragX) - [frac(fragX) &lt; frac(camX)]
     * </pre>
     *
     * so the two-part form is off by one for every fragment whose in-block fraction is below the
     * camera's, and that band is positioned by {@code frac(camX)} — where the camera sits inside its
     * own block. Which means the culled region's edges track the player's sub-block position, i.e.
     * they crawl with the player instead of stepping at chunk boundaries. A one-block band out of
     * every sixteen is invisible in a still frame, which is why this survived both the tests and the
     * screenshots and was reported from play as "the edges seem to follow me as I move".
     */
    static int shaderSectionAxis(final float camWorld, final float camRel) {
        return (int) Math.floor(camWorld + camRel) >> 4;
    }

    private IGpuBuffer buffer;
    private int side = -1;
    private int camSecX = Integer.MIN_VALUE;
    private int camSecY = Integer.MIN_VALUE;
    private int camSecZ = Integer.MIN_VALUE;
    /** The uploaded column/Y-mask set, kept to skip re-uploads when nothing moved. */
    private long[] shadow;

    /**
     * Diagnostics for "LOD chunks near the player are completely empty".
     *
     * <p>The cull is only as good as the set behind it, and the set has one assumption that has
     * never been checked: that Sodium reports a section whose mesh was <i>cleared</i> through
     * {@code uploadResults}, so the {@code else} branch in {@code MixinRenderRegionManager} can
     * retract it. If that is false, {@link #BUILT} only ever grows and accumulates columns
     * vanilla has long since unloaded — and because {@link #update} re-indexes the whole set
     * relative to the <i>current</i> camera, a stale section from where the player used to be
     * re-enters the square and culls LOD over a column vanilla is no longer drawing. That is a
     * hole with vanilla's name on it, and it is why the holes would cluster near the player.
     *
     * <p>{@code maxBuilt} is the high-water mark that exposes it: a maintained set oscillates,
     * an accumulating one climbs. {@code bits/side^2} is the other half — if the square saturates
     * at 289/289 the mask has degenerated into exactly the render-distance-box proxy this class
     * exists to replace, and every column where vanilla is not drawing becomes a hole.
     */
    /** Opt-in: the grid is 17 lines per report, which is noise in a log nobody is reading for it. */
    private static final boolean VMASK_LOG = "1".equals(System.getenv("VOXY_VMASK"));
    private static long VMASK_FRAME = 0;
    private static int maxBuilt = 0;
    private long uploads = 0;

    public static void logPopulation(int side, int anchorSecX, int camSecY, int anchorSecZ,
                                     int camSecX, int camSecZ, long[] columnY, long uploads) {
        if (!VMASK_LOG) return;
        int builtSize;
        synchronized (BuiltSectionMask.class) {
            builtSize = BUILT.size();
        }
        if (builtSize > maxBuilt) maxBuilt = builtSize;
        if ((VMASK_FRAME++ % 600) != 1) return;

        // What the cull will actually do, in the three numbers that can be wrong independently.
        //
        // columns is the horizontal extent: how many of the side^2 columns claim ANY built section.
        // sections is the true size of the set the shader can see, which is what a 3D cull keys on
        // -- a saturated column count is no longer alarming on its own, because the vertical bits
        // are what decide. If sections is far below builtSize, most of what Sodium reported is
        // outside the square or outside the representable vertical span and the cull is blind to
        // it; if builtSize climbs without bound the set is accumulating and reset() needs wiring to
        // Sodium's section-manager rebuild.
        int columns = 0;
        int sections = 0;
        int minBit = 64, maxBit = -1;
        for (final long m : columnY) {
            if (m == 0) continue;
            columns++;
            sections += Long.bitCount(m);
            minBit = Math.min(minBit, Long.numberOfTrailingZeros(m));
            maxBit = Math.max(maxBit, 63 - Long.numberOfLeadingZeros(m));
        }

        final StringBuilder grid = new StringBuilder();
        // 'O' marks the CAMERA's column, which is the only column whose position says anything about
        // centring. This used to be drawn at a fixed (side/2, side/2), which is the camera's column
        // only when the anchor happens to land so -- one time in anchorStep -- so reading centring off
        // the grid gave a wrong answer most of the time. The camera is the thing to compare against.
        final int camDx = camSecX - anchorSecX;
        final int camDz = camSecZ - anchorSecZ;
        for (int dz = 0; dz < side; dz++) {
            grid.append("\n  ");
            for (int dx = 0; dx < side; dx++) {
                final long m = columnY[dz * side + dx];
                final char c = m == 0 ? '.' : (Long.bitCount(m) >= 8 ? '#' : '+');
                grid.append(dx == camDx && dz == camDz ? (m == 0 ? 'O' : 'X') : c);
            }
        }
        me.cortex.voxy.common.Logger.info(String.format(
                "[Metal-VMASK f=%d] built=%d maxBuilt=%d columns=%d/%d sections=%d  anchor=%d,%d cam=%d,%d camSecY=%d uploads=%d resets=%d kept=%d added=%d removed=%d%s",
                VMASK_FRAME, builtSize, maxBuilt, columns, side * side, sections,
                anchorSecX, anchorSecZ, camSecX, camSecZ, camSecY, uploads, resets, resetsSkipped,
                added, removed, grid));
        me.cortex.voxy.common.Logger.info(String.format(
                "[Metal-VMASK2 f=%d] vertical span: bits %d..%d of 64 (bias %d => sections %d..%d relative to the camera's)",
                VMASK_FRAME, minBit < 64 ? minBit : -1, maxBit, Y_BIAS,
                minBit < 64 ? minBit - Y_BIAS : 0, maxBit - Y_BIAS));
    }

    /**
     * Enumerate what the mask CLAIMS vanilla renders, and hold it against the region Sodium is
     * documented to render: a cylinder of radius {@code renderDistance} chunks around the camera.
     *
     * <p>This exists because {@code isBuilt()} answers "Sodium finished a mesh for this section",
     * which is not the same question as "Sodium draws this section". The mask has always treated
     * them as the same, and if they differ the mask claims coverage where vanilla draws nothing --
     * which is a hole, and specifically an EDGE hole, because that is where the two sets diverge.
     *
     * <p>Reports the horizontal Chebyshev distance distribution of the set in chunk units. Anything
     * past the render distance is a section the mask believes is covered and Sodium will not draw.
     */
    public static void logBuiltExtent(final int camSecX, final int camSecY, final int camSecZ, final int rd) {
        if (!VMASK_LOG) return;
        final long[] snapshot;
        synchronized (BuiltSectionMask.class) {
            snapshot = BUILT.toLongArray();
        }
        if (snapshot.length == 0) return;

        // Bands of one chunk, plus a tail bucket; enough to see the shape without 400 log lines.
        final int BANDS = 24;
        final int[] hist = new int[BANDS + 1];
        int beyondRd = 0;
        int outsideCylinder = 0;
        int maxD = 0;
        int minY = Integer.MAX_VALUE, maxY = Integer.MIN_VALUE;
        for (final long pos : snapshot) {
            final int dx = Math.abs(SectionPos.x(pos) - camSecX);
            final int dz = Math.abs(SectionPos.z(pos) - camSecZ);
            final int d = Math.max(dx, dz);
            hist[Math.min(d, BANDS)]++;
            if (d > maxD) maxD = d;
            if (d > rd) beyondRd++;
            // The count that matters is the one the FILTER applies: a section outside Sodium's drawn
            // region is meshed and never rendered, and everything the mask claims there is a hole.
            // Reported in three dimensions, because at altitude it is the vertical component that
            // decides -- a section directly below the camera is zero chunks away horizontally.
            final int dy = SectionPos.y(pos) - camSecY;
            if (!withinRenderDistance(dx, dy, dz, rd)) outsideCylinder++;
            final int y = SectionPos.y(pos);
            if (y < minY) minY = y;
            if (y > maxY) maxY = y;
        }

        final StringBuilder b = new StringBuilder();
        for (int d = 0; d <= BANDS; d++) {
            if (hist[d] == 0) continue;
            b.append(String.format("  d%02d%s=%d", d, d == BANDS ? "+" : " ", hist[d]));
        }
        me.cortex.voxy.common.Logger.info(String.format(
                "[Metal-VMASK3 f=%d] built=%d  outsideChebyshev=%d  outsideRenderDistance3D=%d (%.1f%%)  maxChebyshev=%d (rd=%d)  secY %d..%d%s",
                VMASK_FRAME, snapshot.length, beyondRd, outsideCylinder,
                100.0 * outsideCylinder / Math.max(1, snapshot.length), maxD, rd, minY, maxY, b));
    }

    /** The mask buffer, or null before the first {@link #update}. */
    public IGpuBuffer buffer() {
        return this.buffer;
    }

    /**
     * Whether the 16x16x16 section containing a given world block is one vanilla has drawn — the
     * rule the fragment stage applies, and the only granularity that can be right.
     *
     * <p><b>Three dimensions, not two.</b> Sodium enforces a VERTICAL render distance as well as a
     * horizontal one: it renders the sections it has, which is not every section in a column it has
     * something in. A cull keyed on the column therefore removes LOD sections vanilla never drew
     * wherever the column is only partly built vertically — a hole above or below vanilla's range,
     * and the LOD was the only thing that would have filled it. Measured: turning the cull off makes
     * the missing chunks disappear entirely, and with it on their positions track the camera, which
     * is what a camera-relative square of column bits would do.
     *
     * <p>It is also finer horizontally than a LOD node can express: a node is 2x2 chunk columns at
     * detail 0 and larger above, so a node-level answer decides for columns that were never asked
     * about. A fragment knows its own position, so it can ask about its own section exactly.
     *
     * <p>Outside the square, or outside the representable vertical span, reads as NOT covered, which
     * keeps the LOD. That is the safe direction: an over-drawn LOD z-fights, an under-drawn one
     * shows the void.
     *
     * <p>Mirrored by {@code quads.frag}; change both together.
     *
     * @param columnY   one Y bitmask per column, {@code bit = (secY - camSecY) + Y_BIAS}
     * @param side      columns per axis of the mask square
     * @param camSecX   the camera's section X, which is column 0 of the square shifted by side/2
     * @param camSecY   the camera's section Y, the origin of each column's bitmask
     * @param camSecZ   the camera's section Z
     * @param secX      the section being asked about, in world section coordinates
     * @param secY      likewise
     * @param secZ      likewise
     */
    public static boolean sectionCovered(final long[] columnY, final int side,
                                         final int anchorSecX, final int camSecY, final int anchorSecZ,
                                         final int secX, final int secY, final int secZ) {
        // Indices are 0-based from the WORLD-anchored origin, so a section's index does not change
        // as the camera moves within an anchor cell. That is the whole point: nothing about the
        // answer moves with the player.
        final int cx = secX - anchorSecX;
        final int cz = secZ - anchorSecZ;
        if (cx < 0 || cz < 0 || cx >= side || cz >= side) {
            return false;
        }
        final int bit = (secY - camSecY) + Y_BIAS;
        if (bit < 0 || bit > 63) {
            return false;
        }
        return (columnY[cz * side + cx] & (1L << bit)) != 0;
    }

    /**
     * The same answer for a world BLOCK position, which is what a fragment actually has: it knows
     * where it is in blocks, not which section it is in. Split out so the block-to-section step is
     * covered by the same tests as the lookup, since an arithmetic shift on a negative coordinate
     * floors and a cast truncates, and the difference is the whole negative half of the world.
     */
    public static boolean blockCovered(final long[] columnY, final int side,
                                       final int anchorSecX, final int camSecY, final int anchorSecZ,
                                       final int blockX, final int blockY, final int blockZ) {
        return sectionCovered(columnY, side, anchorSecX, camSecY, anchorSecZ,
                blockX >> 4, blockY >> 4, blockZ >> 4);
    }

    /**
     * Repack and re-upload if anything changed. Cheap enough to call every frame: the set holds a
     * few thousand entries and the compare-and-skip means the upload only happens when the camera
     * crosses a chunk or Sodium builds something.
     */
    public void update(final Viewport<?> viewport, final RenderBackend backend) {
        final int rd = Math.max(2, net.minecraft.client.Minecraft.getInstance().options.renderDistance().get());
        final int slack = ANCHOR_SLACK < 0 ? rd : ANCHOR_SLACK;
        final int newSide = rd * 2 + 1 + slack * 2;
        // The anchor step is what keeps the camera inside the square with at least rd of margin:
        // the valid anchor positions are exactly one step apart, so the anchor is a deterministic
        // function of the camera and does not depend on where it has been.
        final int anchorStep = newSide - rd * 2;
        final int camBlockX = net.minecraft.util.Mth.floor(viewport.cameraX);
        final int camBlockY = net.minecraft.util.Mth.floor(viewport.cameraY);
        final int camBlockZ = net.minecraft.util.Mth.floor(viewport.cameraZ);
        // Section coords derived from the same floored camera the block coords came from, so a
        // camera sitting exactly on a section border cannot disagree with itself.
        final int newCamX = camBlockX >> 4;
        final int newCamY = camBlockY >> 4;
        final int newCamZ = camBlockZ >> 4;
        // A WORLD-anchored origin, not the camera's column. The camera keeps its own section Y,
        // because the vertical window is a window and not an edge.
        final int anchorX = floorToStep(newCamX - rd, anchorStep);
        final int anchorZ = floorToStep(newCamZ - rd, anchorStep);

        // One 64-bit mask per COLUMN, bit (secY - camSecY + Y_BIAS). The column is the horizontal
        // index and the mask is the vertical extent -- which is the whole difference from the
        // bit-per-column version this replaces. Sodium enforces a VERTICAL render distance, so a
        // column being built says nothing about which of its sections are built, and a cull that
        // believed otherwise removed LOD sections vanilla never drew. That is a hole above or below
        // vanilla's vertical range, and because the square is camera-relative those holes followed
        // the player.
        // Sodium MESHES more than it RENDERS, and the mask has to answer the second question.
        //
        // Measured, at a camera that had been moved: built=779 with 45 sections (5.8%) beyond the
        // render distance, out to Chebyshev 10 against rd=8, and a square-corner distribution --
        // while Sodium renders a EUCLIDEAN CYLINDER of radius renderDistance centred on the camera's
        // chunk (the same metric quads.frag's translucent near-cull already uses, and the reason it
        // uses it). So "isBuilt()" includes the square's corners, which are Euclidean up to 11.3
        // chunks out, and a margin beyond the render distance while chunks load and unload. Every
        // one of those is meshed and never drawn, and the mask culled the LOD there -- a hole in a
        // ring just outside vanilla's render distance, which is where the user sees them.
        //
        // The filter is the metric Sodium renders in, not a margin: a section outside the cylinder
        // is not drawn, whatever its mesh says.
        final long[] columnY = new long[newSide * newSide];
        synchronized (BuiltSectionMask.class) {
            for (final long pos : BUILT) {
                // Distance from the CAMERA's column, because the cylinder is centred there, while
                // the square's origin is world-anchored.
                // Three-dimensional, not the horizontal cylinder: at altitude the sections below the
                // camera are the ones this must exclude, and they are zero chunks away horizontally.
                if (!withinRenderDistance(SectionPos.x(pos) - newCamX, SectionPos.y(pos) - newCamY,
                                          SectionPos.z(pos) - newCamZ, rd)) {
                    continue;
                }
                final int dx = SectionPos.x(pos) - anchorX;
                final int dz = SectionPos.z(pos) - anchorZ;
                if (dx < 0 || dz < 0 || dx >= newSide || dz >= newSide) continue;
                final int bit = SectionPos.y(pos) - newCamY + Y_BIAS;
                if (bit < 0 || bit > 63) continue;   // outside the representable span: not covered
                columnY[dz * newSide + dx] |= 1L << bit;
            }
        }

        // Amortised: the set accumulates one entry per section the player has been near, and every
        // rebuild walks all of it. Pruning cannot change what the mask claims (the distance filter
        // already excludes anything this drops), so it only bounds the cost.
        if (++this.updatesSincePrune >= PRUNE_INTERVAL) {
            this.updatesSincePrune = 0;
            prune(newCamX, newCamY, newCamZ, rd);
        }

        logPopulation(newSide, anchorX, newCamY, anchorZ, newCamX, newCamZ, columnY, this.uploads);
        logBuiltExtent(newCamX, newCamY, newCamZ, rd);

        if (this.buffer != null && this.side == newSide
                && this.camSecX == anchorX && this.camSecY == newCamY && this.camSecZ == anchorZ
                && Arrays.equals(this.shadow, columnY)) {
            return;   // nothing moved
        }

        final int size = (HEADER_UINTS + columnY.length * 2) * 4;
        if (this.buffer == null || this.buffer.size() != size) {
            if (this.buffer != null) this.buffer.free();
            this.buffer = backend.createBuffer(size);
        }
        this.side = newSide;
        this.camSecX = anchorX;
        this.camSecY = newCamY;
        this.camSecZ = anchorZ;
        this.shadow = columnY;
        this.uploads++;

        final long ptr = UploadStream.INSTANCE.upload(this.buffer, 0, size);
        MemoryUtil.memPutInt(ptr, newSide);
        MemoryUtil.memPutInt(ptr + 4, anchorX);
        MemoryUtil.memPutInt(ptr + 8, newCamY);
        MemoryUtil.memPutInt(ptr + 12, anchorZ);
        MemoryUtil.memPutFloat(ptr + 16, (float) viewport.cameraX);
        MemoryUtil.memPutFloat(ptr + 20, (float) viewport.cameraY);
        MemoryUtil.memPutFloat(ptr + 24, (float) viewport.cameraZ);
        MemoryUtil.memPutInt(ptr + 28, 0);
        // uvec2 per column, lo then hi — written as two ints so no 64-bit integer type is needed in
        // the shader, where MSL translation of a GLSL uint64_t is the risk this avoids.
        for (int i = 0; i < columnY.length; i++) {
            MemoryUtil.memPutInt(ptr + 32L + i * 8L, (int) columnY[i]);
            MemoryUtil.memPutInt(ptr + 36L + i * 8L, (int) (columnY[i] >>> 32));
        }
    }

    public void free() {
        if (this.buffer != null) {
            this.buffer.free();
            this.buffer = null;
        }
    }
}
