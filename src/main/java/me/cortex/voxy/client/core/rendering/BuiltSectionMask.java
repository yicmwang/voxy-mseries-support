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
 * <p>The mask is a bit per chunk COLUMN, indexed relative to the camera's column. A column is set
 * if any section in it has geometry. The query in {@code cmdgen.comp} removes a node only when
 * <b>every column the node covers</b> is set — see {@link #nodeFullyCovered}, which is mirrored
 * there and unit-tested. It used to test the node's centre column alone, and that is the defect
 * this file's history should not lose: at detail 0 a node is 2x2 columns, so one bit decided four,
 * and a node whose centre was covered but whose other columns were not was removed whole — a hole
 * up to 32 blocks across, repeated around the rim of vanilla's coverage, which is "LOD chunks near
 * the player are completely empty". The earlier claim that the centre is "exact for the fine LOD
 * levels" was simply wrong: the fine levels are exactly where nodes straddle the rim.
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
     * side, camSecX, camSecZ, camBlockX, camBlockZ, pad.
     *
     * <p>The block-coordinate camera position is here for the fragment-stage cull: a fragment knows
     * its camera-relative offset in blocks (quads3.vert already interpolates it for the near-cull),
     * so camera block X plus that offset gives the fragment's own chunk column. Without it the shader
     * would have only the camera's chunk column and could not tell where inside that column the
     * camera sits, which is the difference between the right chunk and its neighbour.
     */
    private static final int HEADER_UINTS = 6;

    private IGpuBuffer buffer;
    private int side = -1;
    private int camSecX = Integer.MIN_VALUE;
    private int camSecZ = Integer.MIN_VALUE;
    private int[] shadow;

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

    public static void logPopulation(int side, int camSecX, int camSecZ, int[] bits, long uploads) {
        if (!VMASK_LOG) return;
        int builtSize;
        synchronized (BuiltSectionMask.class) {
            builtSize = BUILT.size();
        }
        if (builtSize > maxBuilt) maxBuilt = builtSize;
        if ((VMASK_FRAME++ % 600) != 1) return;
        int setBits = 0;
        for (final int w : bits) setBits += Integer.bitCount(w);

        // Detail-0 nodes are aligned to EVEN world section coordinates, and the mask's column 0 is
        // the camera's column, so the node grid starts at an offset that depends on the camera's
        // parity. Getting that wrong would count the wrong 2x2 blocks, so derive it rather than
        // assuming the camera is on an even column.
        final int parity = ((camSecX % 2) + 2) % 2;
        int partial = 0, wronglyRemoved = 0, wronglyKept = 0;
        for (int oz = -parity; oz + 2 <= side; oz += 2) {
            for (int ox = -parity; ox + 2 <= side; ox += 2) {
                final boolean centre = nodeCentreCovered(bits, side, ox, oz, 2);
                final boolean full = nodeFullyCovered(bits, side, ox, oz, 2);
                if (centre == full) continue;
                if (centre) wronglyRemoved++; else wronglyKept++;
                partial++;
            }
        }

        // The grid is the mask itself, drawn in the frame the shader reads it in: rows are dz,
        // columns are dx, both relative to the camera column ('C'). A node is culled exactly when
        // its CENTRE section falls on a '#', so '#' is "LOD is removed here and vanilla draws it
        // instead". A gap in an otherwise solid square is a column where the mask under-claims;
        // a '#' with no vanilla geometry under it is where a hole can appear. Printing the shape
        // rather than the count is the difference between knowing the mask is 239/289 and knowing
        // WHICH 239 — and the two failures look identical in the count.
        final StringBuilder grid = new StringBuilder();
        for (int dz = 0; dz < side; dz++) {
            grid.append("\n  ");
            for (int dx = 0; dx < side; dx++) {
                final int bit = dz * side + dx;
                final boolean on = (bits[bit >> 5] & (1 << (bit & 31))) != 0;
                grid.append(dx == side / 2 && dz == side / 2 ? (on ? 'X' : 'O') : (on ? '#' : '.'));
            }
        }
        me.cortex.voxy.common.Logger.info(String.format(
                "[Metal-VMASK f=%d] built=%d maxBuilt=%d bits=%d/%d side=%d cam=%d,%d uploads=%d%s",
                VMASK_FRAME, builtSize, maxBuilt, setBits, side * side, side, camSecX, camSecZ, uploads,
                grid));
        // The count the grid cannot give: how many detail-0 nodes the centre-only rule decides
        // differently from "every column covered". wronglyRemoved is the hole ring -- each of those
        // nodes had LOD removed while up to three of its four columns were never drawn by vanilla,
        // and nothing else draws them. wronglyKept is the mirror image, LOD left on top of vanilla.
        me.cortex.voxy.common.Logger.info(String.format(
                "[Metal-VMASK2 f=%d] detail0Nodes partial=%d  wronglyRemoved=%d  wronglyKept=%d  <- centre-only rule vs every-column rule",
                VMASK_FRAME, partial, wronglyRemoved, wronglyKept));
    }

    /** The mask buffer, or null before the first {@link #update}. */
    public IGpuBuffer buffer() {
        return this.buffer;
    }

    /**
     * Whether every chunk column a LOD node covers is drawn by vanilla, which is the only condition
     * under which removing the node cannot leave a hole.
     *
     * <p>A node at LOD detail {@code d} spans {@code n = 2 << d} chunk columns per axis. The cull
     * used to test the node's CENTRE column and remove the whole node on that one bit. At detail 0
     * a node is 2x2 columns, so one bit decided four: a node straddling the edge of vanilla's
     * coverage with its centre inside was removed even though up to three of its columns were never
     * drawn by vanilla, and nothing else draws there — the LOD was the only thing that would have.
     * That is a hole up to 32 blocks across, repeated around the whole seam, which is what "LOD
     * chunks near the player are completely empty" is. The same rule errs the other way when the
     * centre falls outside and the node's other columns are vanilla-drawn, which keeps LOD on top
     * of vanilla.
     *
     * <p>A column outside the mask square is not covered by definition, so a node that overhangs
     * the square can never be proven covered and is kept. That also bounds the shader's work: the
     * per-column loop only runs for nodes that fit entirely inside the square, which for the coarse
     * levels means almost none, since the square is 2*renderDistance+1 columns wide.
     *
     * <p>Mirrored exactly by {@code cmdgen.comp}; this copy exists so the rule can be unit-tested
     * and counted rather than reasoned about, and the two must be changed together.
     *
     * @param bits      the packed mask, one bit per column, {@code bit = dz * side + dx}
     * @param side      columns per axis of the mask square
     * @param ox        the node's first column, relative to the mask's origin column (may be negative)
     * @param oz        the node's first row, likewise
     * @param n         columns per axis the node spans, {@code 2 << detail}
     */
    public static boolean nodeFullyCovered(final int[] bits, final int side,
                                           final int ox, final int oz, final int n) {
        if (ox < 0 || oz < 0 || ox + n > side || oz + n > side) {
            return false;
        }
        for (int z = 0; z < n; z++) {
            for (int x = 0; x < n; x++) {
                final int bit = (oz + z) * side + (ox + x);
                if ((bits[bit >> 5] & (1 << (bit & 31))) == 0) {
                    return false;
                }
            }
        }
        return true;
    }

    /**
     * Whether the chunk column containing a given world block is one vanilla has drawn — the test
     * the fragment stage applies per fragment, and the granularity the cull is required to have.
     *
     * <p>This is the whole point of moving the cull out of {@code cmdgen}: a node is 2x2 chunk
     * columns at detail 0 and larger above it, so a node-level decision is coarser than the thing
     * being decided. A fragment knows its own block position, so it can ask the question at exactly
     * chunk granularity, at every detail level, and neither remove a column vanilla did not draw
     * (a hole) nor keep one it did (a doubled surface).
     *
     * <p>{@code worldBlock} is floored world coordinates; the shift is arithmetic so that negative
     * coordinates floor rather than truncate, which matters because the world extends either side
     * of the origin. Mirrored by {@code quads.frag}; change both together.
     */
    public static boolean columnCovered(final int[] bits, final int side,
                                        final int camBlockX, final int camBlockZ,
                                        final int worldBlockX, final int worldBlockZ) {
        // The mask's origin column is the camera's column, and it is centred, so the camera column
        // sits at index side>>1. Chunk coords are recovered from the block coords rather than taken
        // from the header's camSecX so that both ends of the comparison floor identically.
        final int cx = ((worldBlockX >> 4) - (camBlockX >> 4)) + (side >> 1);
        final int cz = ((worldBlockZ >> 4) - (camBlockZ >> 4)) + (side >> 1);
        if (cx < 0 || cz < 0 || cx >= side || cz >= side) {
            return false;
        }
        final int bit = cz * side + cx;
        return (bits[bit >> 5] & (1 << (bit & 31))) != 0;
    }

    /**
     * The centre-column rule the cull used to apply, kept only so the two can be counted against
     * each other. A node is "wrongly removed" when this says yes and {@link #nodeFullyCovered}
     * says no, and that count is the size of the hole ring.
     */
    public static boolean nodeCentreCovered(final int[] bits, final int side,
                                            final int ox, final int oz, final int n) {
        final int cx = ox + (n >> 1);
        final int cz = oz + (n >> 1);
        if (cx < 0 || cz < 0 || cx >= side || cz >= side) {
            return false;
        }
        final int bit = cz * side + cx;
        return (bits[bit >> 5] & (1 << (bit & 31))) != 0;
    }

    /**
     * Repack and re-upload if anything changed. Cheap enough to call every frame: the set holds a
     * few thousand entries and the compare-and-skip means the upload only happens when the camera
     * crosses a chunk or Sodium builds something.
     */
    public void update(final Viewport<?> viewport, final RenderBackend backend) {
        final int rd = Math.max(2, net.minecraft.client.Minecraft.getInstance().options.renderDistance().get());
        final int newSide = rd * 2 + 1;
        final int newCamX = net.minecraft.util.Mth.floor(viewport.cameraX) >> 4;
        final int newCamZ = net.minecraft.util.Mth.floor(viewport.cameraZ) >> 4;

        final int[] bits = new int[(newSide * newSide + 31) / 32];
        synchronized (BuiltSectionMask.class) {
            for (final long pos : BUILT) {
                final int dx = SectionPos.x(pos) - newCamX + rd;
                final int dz = SectionPos.z(pos) - newCamZ + rd;
                if (dx < 0 || dz < 0 || dx >= newSide || dz >= newSide) continue;
                final int bit = dz * newSide + dx;
                bits[bit >> 5] |= 1 << (bit & 31);
            }
        }

        logPopulation(newSide, newCamX, newCamZ, bits, this.uploads);

        if (this.buffer != null && this.side == newSide
                && this.camSecX == newCamX && this.camSecZ == newCamZ
                && Arrays.equals(this.shadow, bits)) {
            return;   // nothing moved
        }

        final int size = (HEADER_UINTS + bits.length) * 4;
        if (this.buffer == null || this.buffer.size() != size) {
            if (this.buffer != null) this.buffer.free();
            this.buffer = backend.createBuffer(size);
        }
        this.side = newSide;
        this.camSecX = newCamX;
        this.camSecZ = newCamZ;
        this.shadow = bits;
        this.uploads++;

        final long ptr = UploadStream.INSTANCE.upload(this.buffer, 0, size);
        MemoryUtil.memPutInt(ptr, newSide);
        MemoryUtil.memPutInt(ptr + 4, newCamX);
        MemoryUtil.memPutInt(ptr + 8, newCamZ);
        // Block coords, floored — the fragment stage adds its own camera-relative offset to these
        // to find its chunk column. Must use the same floored camera the chunk coords came from,
        // or a camera sitting exactly on a chunk border would disagree with itself.
        MemoryUtil.memPutInt(ptr + 12, net.minecraft.util.Mth.floor(viewport.cameraX));
        MemoryUtil.memPutInt(ptr + 16, net.minecraft.util.Mth.floor(viewport.cameraZ));
        MemoryUtil.memPutInt(ptr + 20, 0);
        for (int i = 0; i < bits.length; i++) {
            MemoryUtil.memPutInt(ptr + 24L + i * 4L, bits[i]);
        }
    }

    public void free() {
        if (this.buffer != null) {
            this.buffer.free();
            this.buffer = null;
        }
    }
}
