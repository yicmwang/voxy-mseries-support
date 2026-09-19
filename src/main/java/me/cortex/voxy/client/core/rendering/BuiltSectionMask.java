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
