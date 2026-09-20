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
 * if any section in it has geometry. The query in {@code cmdgen.comp} uses a LOD node's centre
 * column, which is exact for the fine LOD levels that exist near the camera; coarse levels only
 * exist far away, where their centre falls outside the square and they are simply not culled.
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

    private static final int HEADER_UINTS = 4;

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
    }

    /** The mask buffer, or null before the first {@link #update}. */
    public IGpuBuffer buffer() {
        return this.buffer;
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
        MemoryUtil.memPutInt(ptr + 12, 0);
        for (int i = 0; i < bits.length; i++) {
            MemoryUtil.memPutInt(ptr + 16L + i * 4L, bits[i]);
        }
    }

    public void free() {
        if (this.buffer != null) {
            this.buffer.free();
            this.buffer = null;
        }
    }
}
