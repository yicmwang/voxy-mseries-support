package me.cortex.voxy.client.core.rendering;

import it.unimi.dsi.fastutil.longs.LongOpenHashSet;
import me.cortex.voxy.client.core.gpu.IGpuBuffer;
import me.cortex.voxy.client.core.gpu.RenderBackend;
import me.cortex.voxy.client.core.rendering.util.UploadStream;
import net.minecraft.core.SectionPos;
import org.lwjgl.system.MemoryUtil;

import java.util.Arrays;

/**
 * The set of 16³ sections vanilla is drawing THIS FRAME, packed into a bitmask over a chunk-column
 * square and handed to the LOD's fragment stage so the LOD is culled exactly where vanilla drew.
 *
 * <p><b>The feed is Sodium's render list, and that is the whole point of this class.</b> Earlier
 * versions fed it from {@code RenderRegionManager.uploadResults} — what Sodium had MESHED — and then
 * tried to turn that into what Sodium RENDERS with a distance rule. A distance is not a frustum, so
 * every version of that rule was wrong in some direction, and each fix moved the error instead of
 * removing it: a 3-D sphere under-claimed, a bare cylinder over-claimed at altitude (a hole through
 * the world), a conjunction of the two refused 90% of what Sodium had built, and Sodium's own union
 * over-claimed again. The sections Sodium draws are not something to be calculated — they are handed
 * to the render hook, and {@code MixinDefaultChunkRenderer} passes them here each frame.
 *
 * <p>What that removes: the accumulated built set, its pruning, its reset-on-rebuild policy, and
 * every distance term. There is nothing left to keep in sync with Sodium, and nothing that can be
 * stale, because the mask is rebuilt from live data every frame.
 *
 * <p>The mask itself is one 64-bit vertical bitmask per chunk COLUMN, addressed by
 * {@code (secX - anchorX, secZ - anchorZ)}, with the query a single 16³ section at bit
 * {@code (secY - camSecY) + 32}. The square is world-anchored on a step, which only sizes the
 * address space — with an exact set the boundary is vanilla's own boundary, so nothing about it
 * drags or snaps beyond what vanilla's geometry does.
 *
 * <p>Cost is a bit test in the existing fragment stage: no rasterization, no depth blit, no
 * readback, no depth precision to reason about.
 */
public final class BuiltSectionMask {

    /**
     * Section positions vanilla is drawing this frame. Cleared at the start of each frame's feed and
     * filled from the render list, so it is a snapshot and never accumulates.
     */
    private static final LongOpenHashSet DRAWN = new LongOpenHashSet();

    /** Start a frame's feed. Anything not offered again this frame is gone. */
    public static synchronized void beginFrame() {
        DRAWN.clear();
    }

    /** Offer one section vanilla is drawing this frame. */
    public static synchronized void addDrawn(final long sectionPos) {
        DRAWN.add(sectionPos);
    }

    /** How many sections this frame's feed offered. Package-private: the tests and the log read it. */
    static synchronized int drawnCount() {
        return DRAWN.size();
    }

    /**
     * side, anchorX, camSecY, anchorZ, camBlockX, camBlockY, camBlockZ, pad.
     *
     * <p>Eight, not six, so the {@code uvec2[]} that follows starts at byte 32 and is 8-byte aligned
     * as std430 requires for a vec2 array. Getting that wrong does not fail to compile; it silently
     * shifts every read by four bytes, which is the kind of bug this file has already paid for once.
     */
    private static final int HEADER_UINTS = 8;

    /**
     * How many section-Y values either side of the camera's section one column's bitmask covers: bit
     * {@code (secY - camSecY) + Y_BIAS} for {@code 0 <= bit < 64}, i.e. sections from 32 below the
     * camera to 31 above, or ±512 blocks. The overworld is 24 sections tall, so this is ample;
     * anything outside it reads as "not covered", which keeps the LOD rather than removing it.
     */
    private static final int Y_BIAS = 32;

    /**
     * How much slack the square carries beyond the render distance, in chunks, so it can be anchored
     * to a WORLD grid rather than re-centred on the camera every frame. Set equal to the render
     * distance, so the square is {@code (4*rd+1)} columns on a side — 33 at RD 8 — which is 8.7 KB of
     * column bitmasks. Any camera position leaves at least rd columns of margin on every side.
     */
    private static final int ANCHOR_SLACK = -1;   // -1 = "same as the render distance", resolved in update()

    private IGpuBuffer buffer;
    private int side = -1;
    /** The anchor the uploaded mask is addressed from, and the camera section Y its bits are biased by. */
    private int anchorSecX = Integer.MIN_VALUE;
    private int camSecY = Integer.MIN_VALUE;
    private int anchorSecZ = Integer.MIN_VALUE;
    /** The uploaded column/Y-mask set, kept to skip re-uploads when nothing moved. */
    private long[] shadow;
    private long uploads;

    /** Opt-in: the grid is one line per report, which is noise in a log nobody is reading for it. */
    private static final boolean VMASK_LOG = "1".equals(System.getenv("VOXY_VMASK"));
    private static long VMASK_FRAME = 0;
    private static int maxDrawn = 0;

    /** The mask buffer, or null before the first {@link #update}. */
    public IGpuBuffer buffer() {
        return this.buffer;
    }

    static int floorToStep(final int v, final int step) {
        return Math.floorDiv(v, step) * step;
    }

    /**
     * Pack the frame's drawn sections into one 64-bit vertical mask per column.
     *
     * <p>Separate from {@link #update} only so the packing can be tested without a {@code Viewport} or
     * a GPU backend. The two bounds are the address space and nothing else: a column outside the
     * square, or a Y outside the bit span, is left unclaimed — which keeps the LOD rather than
     * removing it, the safe direction. There is deliberately NO distance test here; see the class
     * comment for the four versions of one that were tried and why none could work.
     */
    static long[] buildColumns(final int side, final int anchorX, final int camSecY, final int anchorZ) {
        final long[] columnY = new long[side * side];
        synchronized (BuiltSectionMask.class) {
            for (final long pos : DRAWN) {
                final int dx = SectionPos.x(pos) - anchorX;
                final int dz = SectionPos.z(pos) - anchorZ;
                if (dx < 0 || dz < 0 || dx >= side || dz >= side) continue;
                final int bit = SectionPos.y(pos) - camSecY + Y_BIAS;
                if (bit < 0 || bit > 63) continue;
                columnY[dz * side + dx] |= 1L << bit;
            }
        }
        return columnY;
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

    /**
     * The shader's own lookup, on the CPU: {@code quads.frag} takes the fragment's section, subtracts
     * the anchor for a column, reads that column's {@code uvec2} and tests
     * {@code bit = (secY - camSecY) + 32}. Reproduced here so the producer and the consumer are
     * checked against each other rather than each against its own idea of the layout — the mirror is
     * the seam that had no test through every version of the distance rule, and a packing mismatch
     * reads on screen as a hole or an over-draw with no other symptom.
     */
    static boolean shaderSaysCovered(final long[] columnY, final int side, final int anchorX,
                                     final int camSecY, final int anchorZ,
                                     final int secX, final int secY, final int secZ) {
        final int cx = secX - anchorX;
        final int cz = secZ - anchorZ;
        if (cx < 0 || cz < 0 || cx >= side || cz >= side) return false;
        final int bit = (secY - camSecY) + Y_BIAS;
        if (bit < 0 || bit >= 64) return false;
        final long col = columnY[cz * side + cx];
        final long word = bit < 32 ? (col & 0xFFFFFFFFL) : (col >>> 32);
        return (word & (1L << (bit & 31))) != 0L;
    }

    /**
     * Repack and re-upload from this frame's feed. Cheap enough to call every frame: the compare-and-skip
     * means the upload only happens when the camera crosses a chunk or vanilla's drawn set changes.
     */
    public void update(final Viewport<?> viewport, final RenderBackend backend) {
        final int rd = Math.max(2, net.minecraft.client.Minecraft.getInstance().options.renderDistance().get());
        final int slack = ANCHOR_SLACK < 0 ? rd : ANCHOR_SLACK;
        final int newSide = rd * 2 + 1 + slack * 2;
        final int anchorStep = newSide - rd * 2;
        final int camBlockX = net.minecraft.util.Mth.floor(viewport.cameraX);
        final int camBlockY = net.minecraft.util.Mth.floor(viewport.cameraY);
        final int camBlockZ = net.minecraft.util.Mth.floor(viewport.cameraZ);
        final int newCamX = camBlockX >> 4;
        final int newCamY = camBlockY >> 4;
        final int newCamZ = camBlockZ >> 4;
        final int anchorX = floorToStep(newCamX - rd, anchorStep);
        final int anchorZ = floorToStep(newCamZ - rd, anchorStep);

        // One 64-bit mask per COLUMN, bit (secY - camSecY + Y_BIAS). No distance test: every section
        // offered this frame is one Sodium is about to draw, whatever its distance, height or angle.
        // The only bounds are the address space -- the square's columns and the bit span.
        final long[] columnY = buildColumns(newSide, anchorX, newCamY, anchorZ);

        logPopulation(newSide, anchorX, newCamY, anchorZ, newCamX, newCamZ, columnY, this.uploads);

        if (this.buffer != null && this.side == newSide
                && this.anchorSecX == anchorX && this.camSecY == newCamY && this.anchorSecZ == anchorZ
                && Arrays.equals(this.shadow, columnY)) {
            return;   // nothing moved
        }

        final int size = (HEADER_UINTS + columnY.length * 2) * 4;
        if (this.buffer == null || this.buffer.size() != size) {
            if (this.buffer != null) this.buffer.free();
            this.buffer = backend.createBuffer(size);
        }
        this.side = newSide;
        this.anchorSecX = anchorX;
        this.camSecY = newCamY;
        this.anchorSecZ = anchorZ;
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

    /**
     * What the cull will actually do, in the numbers that can be wrong independently.
     *
     * <p>{@code drawn} is what this frame's render list offered; {@code columns} is how many of the
     * square's columns claim any of it, and {@code sections} how many bits are set. If {@code drawn}
     * is large and {@code sections} is near zero, the feed and the mask disagree about the address
     * space (a decode or anchor fault); if {@code drawn} itself is zero while terrain is on screen,
     * the feed is not being reached.
     */
    public static void logPopulation(int side, int anchorSecX, int camSecY, int anchorSecZ,
                                     int camSecX, int camSecZ, long[] columnY, long uploads) {
        if (!VMASK_LOG) return;
        int drawn;
        synchronized (BuiltSectionMask.class) {
            drawn = DRAWN.size();
        }
        if (drawn > maxDrawn) maxDrawn = drawn;
        if ((VMASK_FRAME++ % 600) != 1) return;

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
        // only when the anchor happens to land so, so reading centring off the grid gave a wrong
        // answer most of the time.
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
                "[Metal-VMASK f=%d] drawn=%d maxDrawn=%d columns=%d/%d sections=%d bits %d..%d"
                        + "  anchor=%d,%d cam=%d,%d camSecY=%d uploads=%d%s",
                VMASK_FRAME, drawn, maxDrawn, columns, side * side, sections,
                minBit < 64 ? minBit : -1, maxBit,
                anchorSecX, anchorSecZ, camSecX, camSecZ, camSecY, uploads, grid));
    }

    public void free() {
        if (this.buffer != null) {
            this.buffer.free();
            this.buffer = null;
        }
    }
}
