package me.cortex.voxy.client.core.rendering;

import net.minecraft.core.SectionPos;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Pins the packing that turns "the sections vanilla is drawing this frame" into the column bitmask
 * the LOD's fragment stage tests.
 *
 * <p>What this class used to test is worth recording, because the tests were green through every
 * version of a broken cull. They tested a DISTANCE RULE — sphere, then cylinder, then a cylinder with
 * a vertical cap, then Sodium's own union — as a pure function. Each was green while the feature was
 * wrong, because the rule was never the whole problem: the rule existed to approximate a set that
 * Sodium hands over directly, and no pure-function test can see an approximation being wrong.
 *
 * <p>So the properties pinned here are the ones on the seam instead: that the producer's packing and
 * the consumer's lookup agree ({@link #theShaderLookupAgreesWithThePacking}), that a frame's feed does
 * not leak into the next ({@link #aSectionDrawnLastFrameIsNotCoveredThisFrame}), and that the address
 * space's edges fail toward keeping the LOD rather than removing it.
 */
class BuiltSectionMaskTest {

    private static final int SIDE = 33;          // 4*rd+1 at rd 8
    private static final int ANCHOR_X = -100;
    private static final int ANCHOR_Z = -100;
    private static final int CAM_SEC_Y = 8;

    private static long at(int x, int y, int z) {
        return SectionPos.asLong(x, y, z);
    }

    private static long[] columnsOf(long... sections) {
        BuiltSectionMask.beginFrame();
        for (long s : sections) {
            BuiltSectionMask.addDrawn(s);
        }
        return BuiltSectionMask.buildColumns(SIDE, ANCHOR_X, CAM_SEC_Y, ANCHOR_Z);
    }

    private static boolean covered(long[] columnY, int x, int y, int z) {
        return BuiltSectionMask.shaderSaysCovered(columnY, SIDE, ANCHOR_X, CAM_SEC_Y, ANCHOR_Z, x, y, z);
    }

    @Test
    void aDrawnSectionIsClaimed() {
        long[] m = columnsOf(at(-92, 8, -92));       // the camera's own section
        assertTrue(covered(m, -92, 8, -92));
        assertEquals(1, BuiltSectionMask.drawnCount());
    }

    @Test
    void nothingIsClaimedBeforeAnyFeed() {
        BuiltSectionMask.beginFrame();
        long[] m = BuiltSectionMask.buildColumns(SIDE, ANCHOR_X, CAM_SEC_Y, ANCHOR_Z);
        for (long c : m) {
            assertEquals(0L, c, "an empty feed must claim nothing");
        }
    }

    @Test
    void aSectionDrawnLastFrameIsNotCoveredThisFrame() {
        // The property that replaces the whole reset/prune story. The mask is not a set that
        // accumulates and has to be invalidated when Sodium rebuilds; it is a snapshot of one frame's
        // render list, so a section that stops being drawn stops being claimed with no policy at all.
        // This is also why a render-distance change needs no reset: the old version cleared the mask
        // there and could never refill it, and the cull was silently off for the rest of the session.
        columnsOf(at(-92, 8, -92), at(-91, 8, -92));

        long[] next = BuiltSectionMask.buildColumns(SIDE, ANCHOR_X, CAM_SEC_Y, ANCHOR_Z);
        assertTrue(covered(next, -92, 8, -92), "still drawn, so still claimed");

        // A new frame offers only one of the two.
        long[] after = columnsOf(at(-91, 8, -92));
        assertFalse(covered(after, -92, 8, -92), "no longer drawn, so no longer claimed");
        assertTrue(covered(after, -91, 8, -92));
    }

    @Test
    void theShaderLookupAgreesWithThePacking() {
        // The seam that had no test: the producer writes columnY[dz*side+dx] with bit (secY-camSecY)+32,
        // and quads.frag reads columnY[cz*sd+cx] with bit (secY-camSecY)+32. If the two ever disagreed,
        // the symptom would be a hole or an over-draw in the world with nothing else to go on.
        long[] fed = { at(-92, 8, -92), at(-88, 4, -95), at(-95, 12, -88) };
        long[] m = columnsOf(fed);
        for (long s : fed) {
            assertTrue(covered(m, SectionPos.x(s), SectionPos.y(s), SectionPos.z(s)),
                    "every section fed must be covered by the shader's own lookup");
        }
        // A control that was never fed, and is inside the square and the bit span.
        assertFalse(covered(m, -93, 8, -93));
        assertFalse(covered(m, -92, 9, -92));
        assertFalse(covered(m, -92, 7, -92));
    }

    @Test
    void theTwoHalvesOfTheColumnAreAddressedCorrectly() {
        // bit < 32 lives in the low word and bit >= 32 in the high one, and the shader picks between
        // them by the same comparison. A section 32 or more above the camera's own is the only way to
        // reach the high word, so it is the only case that can catch a swapped pair.
        long[] m = columnsOf(at(-92, 8 + 31, -92), at(-92, 8 - 31, -92));
        assertTrue(covered(m, -92, 8 + 31, -92), "bit 63, the last bit of the high word");
        assertTrue(covered(m, -92, 8 - 31, -92), "bit 1, the low word");
    }

    @Test
    void theVerticalWindowEndsAtTheBitSpanAndKeepsTheLodOutsideIt() {
        // 64 bits, biased by 32: sections from 32 below the camera to 31 above are addressable, and
        // outside that the answer is "not covered", which KEEPS the LOD. An over-drawn LOD z-fights,
        // an under-drawn one shows the void, and this is the direction that only z-fights.
        long[] m = columnsOf(at(-92, 8 + 31, -92), at(-92, 8 - 32, -92), at(-92, 8 + 32, -92));
        assertTrue(covered(m, -92, 8 + 31, -92), "the last addressable bit");
        assertTrue(covered(m, -92, 8 - 32, -92), "the first addressable bit");
        assertFalse(covered(m, -92, 8 + 32, -92), "one past the span is not covered");
    }

    @Test
    void columnsOutsideTheSquareAreNotClaimed() {
        // The square is the address space, not a claim about visibility. A drawn section outside it is
        // simply not claimed, which keeps the LOD there -- the safe direction, and the reason the
        // square is sized with slack on every side of the camera.
        long[] m = columnsOf(at(ANCHOR_X - 1, 8, -92), at(-92, 8, ANCHOR_Z + SIDE));
        assertFalse(covered(m, ANCHOR_X - 1, 8, -92), "left of the square");
        assertFalse(covered(m, -92, 8, ANCHOR_Z + SIDE), "behind the square");
    }

    @Test
    void neighbouringColumnsAreDistinguished() {
        // Row-major in (z, x) with the anchor as origin; a transpose would put every discard one
        // column over and read as a cull that is right in one diagonal direction and wrong in the
        // other.
        long[] m = columnsOf(at(-92, 8, -92));
        assertTrue(covered(m, -92, 8, -92));
        assertFalse(covered(m, -91, 8, -92), "one column in x");
        assertFalse(covered(m, -92, 8, -91), "one column in z");
    }
}
