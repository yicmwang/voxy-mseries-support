package me.cortex.voxy.client.core.rendering;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Pins the rule that decides whether a LOD node may be removed because vanilla Minecraft is already
 * drawing there.
 *
 * <p>The rule is a safety property, not a preference: removing a node whose columns are not all
 * vanilla-drawn leaves a hole, because the LOD was the only thing that would have drawn them. The
 * previous rule tested the node's centre column and removed the whole node on that one bit, which
 * at detail 0 meant one bit deciding four columns — a hole up to 32 blocks across, ringing the
 * seam wherever vanilla's coverage edge cuts through a node. The tests below are the two directions
 * of that error, and they are written so that reverting to the centre test fails them.
 */
class BuiltSectionMaskTest {

    /** A square mask with the listed columns (dx, dz pairs) set. */
    private static int[] mask(int side, int[][] set) {
        int[] bits = new int[(side * side + 31) / 32];
        for (int[] p : set) {
            int bit = p[1] * side + p[0];
            bits[bit >> 5] |= 1 << (bit & 31);
        }
        return bits;
    }

    private static int[][] block(int ox, int oz, int n) {
        int[][] out = new int[n * n][];
        int i = 0;
        for (int z = 0; z < n; z++) {
            for (int x = 0; x < n; x++) {
                out[i++] = new int[] {ox + x, oz + z};
            }
        }
        return out;
    }

    @Test
    void fullyCoveredNodeIsRemoved() {
        int[] bits = mask(9, block(2, 2, 2));
        assertTrue(BuiltSectionMask.nodeFullyCovered(bits, 9, 2, 2, 2));
    }

    @Test
    void nodeWithOnlyItsCentreCoveredIsKept() {
        // The regression. A 2x2 node whose centre column (its bottom-right at detail 0, since the
        // centre is origin + n/2) is covered but whose other three columns are not. The old rule
        // removed it and left three columns of nothing.
        int[] bits = mask(9, new int[][] {{3, 3}});   // node origin (2,2): columns 2..3
        assertTrue(BuiltSectionMask.nodeCentreCovered(bits, 9, 2, 2, 2),
                "precondition: the centre column is covered, which is what the old rule keyed on");
        assertFalse(BuiltSectionMask.nodeFullyCovered(bits, 9, 2, 2, 2),
                "removing this node would leave three columns nothing draws");
    }

    @Test
    void nodeWithThreeOfFourColumnsCoveredIsKept() {
        int[] bits = mask(9, new int[][] {{3, 3}, {2, 2}, {3, 2}});
        assertFalse(BuiltSectionMask.nodeFullyCovered(bits, 9, 2, 2, 2));
    }

    @Test
    void nodeOverhangingTheSquareIsKept() {
        // Columns outside the square are not covered by definition, so full coverage cannot be
        // proven. This is also what bounds the shader's per-column loop.
        int[] bits = mask(9, block(0, 0, 4));
        assertFalse(BuiltSectionMask.nodeFullyCovered(bits, 9, -1, 0, 2), "negative origin");
        assertFalse(BuiltSectionMask.nodeFullyCovered(bits, 9, 8, 0, 2), "runs past +x");
        assertFalse(BuiltSectionMask.nodeFullyCovered(bits, 9, 0, 8, 2), "runs past +z");
        assertTrue(BuiltSectionMask.nodeFullyCovered(bits, 9, 0, 0, 2), "the fully inside corner");
    }

    @Test
    void coverageIsRequiredAtEveryDetailLevel() {
        // The old comment claimed this was only a fine-level concern because coarse nodes fall
        // outside the square. That is not so: a coarse node can sit entirely inside when it is
        // near the camera, and then the same one-bit-decides-many error applies with n = 4, 8, 16.
        int side = 17;
        int[] full = mask(side, block(0, 0, 16));
        assertTrue(BuiltSectionMask.nodeFullyCovered(full, side, 0, 0, 16));
        assertFalse(BuiltSectionMask.nodeFullyCovered(full, side, 0, 0, 17), "17 columns do not fit");

        // poke one hole in an otherwise complete 4x4 node
        int[] holed = mask(side, block(0, 0, 16));
        int bit = 2 * side + 2;
        holed[bit >> 5] &= ~(1 << (bit & 31));
        assertFalse(BuiltSectionMask.nodeFullyCovered(holed, side, 0, 0, 4));
        assertTrue(BuiltSectionMask.nodeFullyCovered(holed, side, 4, 0, 4), "a neighbouring node is unaffected");
    }

    @Test
    void aFragmentFindsItsOwnColumn() {
        // The fragment-stage test, which is the one that decides at chunk granularity. Column 0 of
        // the square is the camera's column and the camera sits at index side>>1, so a world block
        // in the camera's own chunk must land there whatever the camera's offset within the chunk
        // is -- the mask header carries block coordinates precisely so this holds.
        int side = 17;
        int camBlockX = 5 * 16 + 7;    // chunk 5, seven blocks in
        int camBlockZ = -3 * 16 + 15;  // chunk -3, fifteen blocks in (negative-to-positive edge)

        int[] bits = mask(side, new int[][] {{side >> 1, side >> 1}});
        assertTrue(BuiltSectionMask.columnCovered(bits, side, camBlockX, camBlockZ, camBlockX, camBlockZ),
                "the camera's own block is in the camera's own column");
        // The camera is 7 blocks into chunk 5, so 8 further blocks is still chunk 5 and 9 is chunk 6.
        // This is the boundary the mask header's block coordinates exist to get right: with only the
        // camera's chunk column the shader could not tell that 87 + 8 stays put.
        assertTrue(BuiltSectionMask.columnCovered(bits, side, camBlockX, camBlockZ, camBlockX + 8, camBlockZ),
                "8 blocks on, still chunk 5");
        assertFalse(BuiltSectionMask.columnCovered(bits, side, camBlockX, camBlockZ, camBlockX + 9, camBlockZ),
                "9 blocks on is chunk 6, one column across");
    }

    @Test
    void neighboursAreDistinguishedAtChunkBoundaries() {
        // The defect this cull exists to remove: a decision taken at node granularity cannot tell
        // these apart, because both are in the same 2x2 node at detail 0.
        int side = 17;
        int camBlockX = 0, camBlockZ = 0;
        int c = side >> 1;

        int[] onlyCameraColumn = mask(side, new int[][] {{c, c}});
        assertTrue(BuiltSectionMask.columnCovered(onlyCameraColumn, side, camBlockX, camBlockZ, 0, 0));
        assertFalse(BuiltSectionMask.columnCovered(onlyCameraColumn, side, camBlockX, camBlockZ, 16, 0),
                "one chunk over is a different column and must not be culled with it");
        assertFalse(BuiltSectionMask.columnCovered(onlyCameraColumn, side, camBlockX, camBlockZ, -1, -1),
                "and so is the chunk on the negative side of the origin");
    }

    @Test
    void negativeCoordinatesFloorRatherThanTruncate() {
        // Block -1 is in chunk -1, not chunk 0: a truncating cast would put it in chunk 0 and cull
        // the wrong column across the whole negative half of the world.
        int side = 17;
        int[] bits = mask(side, new int[][] {{(side >> 1) - 1, side >> 1}});   // chunk -1 in x
        assertTrue(BuiltSectionMask.columnCovered(bits, side, 0, 0, -1, 0));
        assertTrue(BuiltSectionMask.columnCovered(bits, side, 0, 0, -16, 0));
        assertFalse(BuiltSectionMask.columnCovered(bits, side, 0, 0, 0, 0));
    }

    @Test
    void outsideTheSquareIsNeverCulled() {
        int side = 17;
        int[] full = mask(side, block(0, 0, side));
        assertFalse(BuiltSectionMask.columnCovered(full, side, 0, 0, 1000 * 16, 0),
                "a column beyond the square is not covered, whatever the bits say");
    }

    @Test
    void anEmptyMaskRemovesNothing() {
        int[] bits = mask(17, new int[0][]);
        for (int n = 2; n <= 16; n <<= 1) {
            assertFalse(BuiltSectionMask.nodeFullyCovered(bits, 17, 0, 0, n));
        }
    }
}
