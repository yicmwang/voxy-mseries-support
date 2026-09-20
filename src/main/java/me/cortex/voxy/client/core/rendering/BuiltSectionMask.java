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
        BUILT.add(sectionPos);
    }

    public static synchronized void remove(final long sectionPos) {
        BUILT.remove(sectionPos);
    }

    /** Sodium rebuilds its section manager on a level or render-distance change; the set starts over. */
    public static synchronized void reset() {
        BUILT.clear();
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

    public static void logPopulation(int side, int camSecX, int camSecY, int camSecZ,
                                     long[] columnY, long uploads) {
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
        for (int dz = 0; dz < side; dz++) {
            grid.append("\n  ");
            for (int dx = 0; dx < side; dx++) {
                final long m = columnY[dz * side + dx];
                final char c = m == 0 ? '.' : (Long.bitCount(m) >= 8 ? '#' : '+');
                grid.append(dx == side / 2 && dz == side / 2 ? (m == 0 ? 'O' : 'X') : c);
            }
        }
        me.cortex.voxy.common.Logger.info(String.format(
                "[Metal-VMASK f=%d] built=%d maxBuilt=%d columns=%d/%d sections=%d  cam=%d,%d,%d uploads=%d%s",
                VMASK_FRAME, builtSize, maxBuilt, columns, side * side, sections,
                camSecX, camSecY, camSecZ, uploads, grid));
        me.cortex.voxy.common.Logger.info(String.format(
                "[Metal-VMASK2 f=%d] vertical span: bits %d..%d of 64 (bias %d => sections %d..%d relative to the camera's)",
                VMASK_FRAME, minBit < 64 ? minBit : -1, maxBit, Y_BIAS,
                minBit < 64 ? minBit - Y_BIAS : 0, maxBit - Y_BIAS));
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
                                         final int camSecX, final int camSecY, final int camSecZ,
                                         final int secX, final int secY, final int secZ) {
        final int cx = (secX - camSecX) + (side >> 1);
        final int cz = (secZ - camSecZ) + (side >> 1);
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
                                       final int camSecX, final int camSecY, final int camSecZ,
                                       final int blockX, final int blockY, final int blockZ) {
        return sectionCovered(columnY, side, camSecX, camSecY, camSecZ,
                blockX >> 4, blockY >> 4, blockZ >> 4);
    }

    /**
     * Repack and re-upload if anything changed. Cheap enough to call every frame: the set holds a
     * few thousand entries and the compare-and-skip means the upload only happens when the camera
     * crosses a chunk or Sodium builds something.
     */
    public void update(final Viewport<?> viewport, final RenderBackend backend) {
        final int rd = Math.max(2, net.minecraft.client.Minecraft.getInstance().options.renderDistance().get());
        final int newSide = rd * 2 + 1;
        final int camBlockX = net.minecraft.util.Mth.floor(viewport.cameraX);
        final int camBlockY = net.minecraft.util.Mth.floor(viewport.cameraY);
        final int camBlockZ = net.minecraft.util.Mth.floor(viewport.cameraZ);
        // Section coords derived from the same floored camera the block coords came from, so a
        // camera sitting exactly on a section border cannot disagree with itself.
        final int newCamX = camBlockX >> 4;
        final int newCamY = camBlockY >> 4;
        final int newCamZ = camBlockZ >> 4;

        // One 64-bit mask per COLUMN, bit (secY - camSecY + Y_BIAS). The column is the horizontal
        // index and the mask is the vertical extent -- which is the whole difference from the
        // bit-per-column version this replaces. Sodium enforces a VERTICAL render distance, so a
        // column being built says nothing about which of its sections are built, and a cull that
        // believed otherwise removed LOD sections vanilla never drew. That is a hole above or below
        // vanilla's vertical range, and because the square is camera-relative those holes followed
        // the player.
        final long[] columnY = new long[newSide * newSide];
        synchronized (BuiltSectionMask.class) {
            for (final long pos : BUILT) {
                final int dx = SectionPos.x(pos) - newCamX + rd;
                final int dz = SectionPos.z(pos) - newCamZ + rd;
                if (dx < 0 || dz < 0 || dx >= newSide || dz >= newSide) continue;
                final int bit = SectionPos.y(pos) - newCamY + Y_BIAS;
                if (bit < 0 || bit > 63) continue;   // outside the representable span: not covered
                columnY[dz * newSide + dx] |= 1L << bit;
            }
        }

        logPopulation(newSide, newCamX, newCamY, newCamZ, columnY, this.uploads);

        if (this.buffer != null && this.side == newSide
                && this.camSecX == newCamX && this.camSecY == newCamY && this.camSecZ == newCamZ
                && Arrays.equals(this.shadow, columnY)) {
            return;   // nothing moved
        }

        final int size = (HEADER_UINTS + columnY.length * 2) * 4;
        if (this.buffer == null || this.buffer.size() != size) {
            if (this.buffer != null) this.buffer.free();
            this.buffer = backend.createBuffer(size);
        }
        this.side = newSide;
        this.camSecX = newCamX;
        this.camSecY = newCamY;
        this.camSecZ = newCamZ;
        this.shadow = columnY;
        this.uploads++;

        final long ptr = UploadStream.INSTANCE.upload(this.buffer, 0, size);
        MemoryUtil.memPutInt(ptr, newSide);
        MemoryUtil.memPutInt(ptr + 4, newCamX);
        MemoryUtil.memPutInt(ptr + 8, newCamY);
        MemoryUtil.memPutInt(ptr + 12, newCamZ);
        MemoryUtil.memPutInt(ptr + 16, camBlockX);
        MemoryUtil.memPutInt(ptr + 20, camBlockY);
        MemoryUtil.memPutInt(ptr + 24, camBlockZ);
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
