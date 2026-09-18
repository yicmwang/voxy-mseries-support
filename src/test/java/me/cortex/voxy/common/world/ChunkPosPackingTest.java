package me.cortex.voxy.common.world;

import net.minecraft.world.level.ChunkPos;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Pins {@code ChunkPos.pack}/{@code getX}/{@code getZ} as an exact round trip.
 *
 * <p>Written because the 26.2 port replaced a call to the (renamed) {@code ChunkPos.asLong} with a
 * hand-rolled bit layout:
 * <pre>(((long) x) & 0xFFFFFFFFL) | ((((long) z) & 0xFFFFFFFFL) << 32)</pre>
 * Nothing would have thrown if that layout were wrong — the chunk-tracker cache in
 * {@code MixinRenderSectionManager} would simply have mis-keyed, silently mis-gating ingest. This
 * test makes the layout an assertion rather than an assumption.
 */
class ChunkPosPackingTest {

    private static final int[] COORDS = {
            0, 1, -1, 2, -2, 31, -32, 511, -512, 12345, -12345,
            Integer.MAX_VALUE, Integer.MIN_VALUE
    };

    @Test
    void packRoundTripsForAllCoordinatePairs() {
        for (int x : COORDS) {
            for (int z : COORDS) {
                long packed = ChunkPos.pack(x, z);
                assertEquals(x, ChunkPos.getX(packed), "x round trip for (" + x + "," + z + ")");
                assertEquals(z, ChunkPos.getZ(packed), "z round trip for (" + x + "," + z + ")");
            }
        }
    }

    @Test
    void distinctCoordinatesProduceDistinctKeys() {
        // The cache keys on this value; collisions would alias unrelated chunks.
        assertEquals(ChunkPos.pack(1, 0) != ChunkPos.pack(0, 1), true,
                "swapping x and z must not alias");
    }
}
