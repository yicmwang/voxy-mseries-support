package me.cortex.voxy.client.core.rendering;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * The section position is packed once on the CPU and then used for two things that must agree
 * exactly: it is the key of {@code ChunkBoundRenderer.chunk2idx}, and it is the value the mask pass
 * rasterizes and {@code outline.vsh} decodes with {@code unpackPos}. A mismatch would put mask boxes
 * at the wrong coordinates and quietly discard the wrong LOD fragments — no crash, no counter
 * anomaly, just terrain disappearing in the wrong places.
 *
 * <p>{@link #unpackPosMatchesTheShader} is a transcription of the shader's decode, so this test
 * fails if either side drifts:
 *
 * <pre>
 * ivec3 unpackPos(ivec2 pos) {
 *     return ivec3(pos.y&gt;&gt;10, (pos.x&lt;&lt;12)&gt;&gt;12, ((pos.y&lt;&lt;22)|int(uint(pos.x)&gt;&gt;10))&gt;&gt;10);
 * }
 * </pre>
 */
class ChunkBoundPosPackingTest {

    /** Port of outline.vsh's {@code unpackPos}, including GLSL's arithmetic-shift semantics. */
    private static int[] unpackPos(long packed) {
        int lo = (int) (packed & 0xFFFFFFFFL);
        int hi = (int) ((packed >>> 32) & 0xFFFFFFFFL);
        int x = hi >> 10;
        int y = (lo << 12) >> 12;
        int z = ((hi << 22) | (lo >>> 10)) >> 10;
        return new int[] {x, y, z};
    }

    private static void roundTrip(int x, int y, int z) {
        int[] back = unpackPos(ChunkBoundRenderer.packSectionPos(x, y, z));
        assertEquals(x, back[0], "x round-trip for (" + x + "," + y + "," + z + ")");
        assertEquals(y, back[1], "y round-trip for (" + x + "," + y + "," + z + ")");
        assertEquals(z, back[2], "z round-trip for (" + x + "," + y + "," + z + ")");
    }

    @Test
    void roundTripsOrdinarySections() {
        roundTrip(0, 0, 0);
        roundTrip(1, 2, 3);
        roundTrip(-1, -1, -1);
        roundTrip(100, 20, -300);
        roundTrip(-1234, 4, 5678);
    }

    /**
     * The vertical axis is only 20 bits wide while the horizontal pair get 22 each — so the limits
     * are asymmetric and are exactly where an off-by-one in the masks would show up. Voxy's world
     * height is far inside the y limit; the horizontal limits are the ones that actually matter.
     */
    @Test
    void roundTripsAtEachFieldsLimit() {
        roundTrip((1 << 21) - 1, (1 << 19) - 1, (1 << 21) - 1);     // max positive
        roundTrip(-(1 << 21), -(1 << 19), -(1 << 21));              // min negative
    }

    /** Adjacent sections must not collide — the fields are packed, not hashed. */
    @Test
    void neighbouringSectionsPackToDifferentKeys() {
        long base = ChunkBoundRenderer.packSectionPos(10, 4, 10);
        org.junit.jupiter.api.Assertions.assertNotEquals(base, ChunkBoundRenderer.packSectionPos(11, 4, 10));
        org.junit.jupiter.api.Assertions.assertNotEquals(base, ChunkBoundRenderer.packSectionPos(10, 5, 10));
        org.junit.jupiter.api.Assertions.assertNotEquals(base, ChunkBoundRenderer.packSectionPos(10, 4, 11));
    }
}
