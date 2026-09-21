package me.cortex.voxy.client.core.rendering;

import net.minecraft.core.SectionPos;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
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
    void theSnapshotIsQuantisedToTheCameraSection() {
        // The culled region's edges must step at chunk boundaries, not slide with the player. Measured
        // cause: Sodium's render list is the exact frustum and occlusion result, both continuous in
        // the camera's position, so over 600 frames in which the camera crossed TWO section
        // boundaries the mask was rebuilt 17 times. So the feed is gated on the camera's section.
        assertTrue(BuiltSectionMask.beginFrameIfSectionChanged(4, 5, 6), "a new section restarts it");
        BuiltSectionMask.addDrawn(at(-92, 8, -92));
        assertEquals(1, BuiltSectionMask.drawnCount());

        // The camera moved, but not out of its section: the snapshot must be left alone.
        assertFalse(BuiltSectionMask.beginFrameIfSectionChanged(4, 5, 6), "same section holds");
        assertEquals(1, BuiltSectionMask.drawnCount(), "and nothing was cleared");

        // Any axis crossing a boundary restarts it, because the bit window is biased by camSecY.
        assertTrue(BuiltSectionMask.beginFrameIfSectionChanged(5, 5, 6), "x crossed");
        assertTrue(BuiltSectionMask.beginFrameIfSectionChanged(5, 5, 7), "z crossed");
        assertTrue(BuiltSectionMask.beginFrameIfSectionChanged(5, 6, 7), "y crossed");
        assertEquals(0, BuiltSectionMask.drawnCount(), "and each restart clears");
    }

    @Test
    void theSectionIsReconstructedByAddingBeforeFlooring() {
        // The defect nobody caught: quads.frag floored the camera-RELATIVE offset and added it to a
        // FLOORED camera, and floor(cam) + floor(rel) is not floor(cam + rel):
        //
        //     floor(camX) + floor(fragX - camX)  ==  floor(fragX) - [frac(fragX) < frac(camX)]
        //
        // The stray -1 lands on every fragment whose in-block fraction is below the camera's, so the
        // culled region's edges are positioned by frac(camX) -- where the camera sits in its own
        // block -- and crawl along with the player instead of stepping at chunk boundaries. Reported
        // from play, and reported precisely: "the culling region should only step as I cross chunk
        // boundaries. But now, the edges seem to follow me as I move."
        //
        // A one-block band out of every sixteen, so it is invisible in a still frame and in every
        // screenshot taken from one. This walks the camera through every eighth of a block and the
        // fragment across a section boundary, which is the only place the two forms differ.
        for (double fragX : new double[]{ 16.0, 16.25, 16.5, 16.75, 17.0, 31.9, 32.0, -16.5, -15.75 }) {
            for (int eighth = 0; eighth < 8; eighth++) {
                final double camFrac = eighth / 8.0;
                // A camera well away from the fragment, so only the fractional part can matter.
                final float camWorld = (float) (Math.floor(fragX) - 40.0 + camFrac);
                final float rel = (float) (fragX - camWorld);
                assertEquals((int) Math.floor(fragX) >> 4,
                        BuiltSectionMask.shaderSectionAxis(camWorld, rel),
                        "block " + fragX + " with the camera " + camFrac + " into its own block");
            }
        }

        // And the form this replaced, on one case it gets wrong, so a future reader can see the bug
        // rather than take the paragraph above on trust.
        final double fragX = 16.25, camX = 8.5;
        final int old = ((int) Math.floor(camX) + (int) Math.floor(fragX - camX)) >> 4;
        assertNotEquals((int) Math.floor(fragX) >> 4, old, "the two-part form was wrong here");
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
