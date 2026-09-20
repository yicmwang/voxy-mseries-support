package me.cortex.voxy.client.core.rendering;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Pins the rule that decides whether a LOD section may be removed because vanilla Minecraft is
 * already drawing it.
 *
 * <p>The rule is a safety property, not a preference: removing a section vanilla does not draw
 * leaves a hole, because the LOD was the only thing that would have drawn it. Measured on device:
 * with the cull off the missing chunks disappear; with it on their positions follow the camera.
 *
 * <p>Two things about the rule are load-bearing and were each wrong in an earlier version.
 *
 * <p><b>It is per SECTION, not per LOD node.</b> A node is 2x2 chunk columns at detail 0 and more
 * above, so a node-level answer decides for columns nobody asked about.
 *
 * <p><b>It is three-dimensional, not two.</b> Sodium enforces a vertical render distance as well as
 * a horizontal one, so a column having geometry says nothing about whether the section above it
 * does. A column-keyed cull removes LOD sections vanilla never drew — a hole in the air — and
 * because the square is camera-relative those holes travel with the player.
 */
class BuiltSectionMaskTest {

    private static final int SIDE = 17;
    private static final int C = SIDE / 2;   // the camera's own column in the square's indices

    /** The square is anchored so that world sections -C..C land on indices 0..2C. */
    private static final int ANCHOR = -C;
    private static long[] empty() {
        return new long[SIDE * SIDE];
    }

    private static void mark(long[] columnY, int secX, int secY, int secZ) {
        int cx = secX - ANCHOR, cz = secZ - ANCHOR;
        if (cx < 0 || cz < 0 || cx >= SIDE || cz >= SIDE) return;
        int bit = secY + 32;
        if (bit < 0 || bit > 63) return;
        columnY[cz * SIDE + cx] |= 1L << bit;
    }

    private static boolean covered(long[] columnY, int secX, int secY, int secZ) {
        return BuiltSectionMask.sectionCovered(columnY, SIDE, ANCHOR, 0, ANCHOR, secX, secY, secZ);
    }

    @Test
    void aBuiltSectionIsCovered() {
        long[] m = empty();
        mark(m, 3, 4, 5);
        assertTrue(covered(m, 3, 4, 5));
    }

    @Test
    void aColumnBeingBuiltDoesNotCoverItsOtherHeights() {
        // THE regression, and the one the user found by eye. Vanilla has geometry at section
        // (2, 0, 2). The section above it is not built -- vanilla renders nothing there, at any
        // altitude -- so the LOD must not be removed there. A column-keyed cull removed it, which
        // is a hole in the air directly above terrain vanilla draws, and the hole follows the camera
        // because the square does.
        long[] m = empty();
        mark(m, 2, 0, 2);

        assertTrue(covered(m, 2, 0, 2), "the section vanilla built");
        assertFalse(covered(m, 2, 1, 2), "one section up: vanilla draws nothing here");
        assertFalse(covered(m, 2, 8, 2), "far above");
        assertFalse(covered(m, 2, -1, 2), "and below");
    }

    @Test
    void everyHeightVanillaBuiltIsCoveredIndependently() {
        long[] m = empty();
        mark(m, 0, -4, 0);
        mark(m, 0, 5, 0);
        assertTrue(covered(m, 0, -4, 0));
        assertTrue(covered(m, 0, 5, 0));
        assertFalse(covered(m, 0, 0, 0), "the gap between them is not covered");
        assertFalse(covered(m, 0, 6, 0), "nor the section above the top one");
    }

    @Test
    void neighbouringColumnsAreDistinguished() {
        // A node-level rule cannot tell these apart: both are in the same 2x2 node at detail 0.
        long[] m = empty();
        mark(m, 0, 0, 0);
        assertTrue(covered(m, 0, 0, 0));
        assertFalse(covered(m, 1, 0, 0), "one column across");
        assertFalse(covered(m, -1, 0, 0), "and on the negative side of the origin");
        assertFalse(covered(m, 0, 0, 1));
    }

    @Test
    void blockCoordinatesFloorRatherThanTruncate() {
        // The fragment has block coordinates, not section coordinates, so the shift is what puts it
        // in a section. An arithmetic shift floors a negative; a cast truncates toward zero, which
        // would put block -1 in section 0 and cull the wrong section across the whole negative half
        // of the world.
        long[] m = empty();
        mark(m, -1, 0, 0);   // section -1 covers blocks -16..-1
        assertTrue(BuiltSectionMask.blockCovered(m, SIDE, ANCHOR, 0, ANCHOR, -1, 0, 0));
        assertTrue(BuiltSectionMask.blockCovered(m, SIDE, ANCHOR, 0, ANCHOR, -16, 0, 0));
        assertFalse(BuiltSectionMask.blockCovered(m, SIDE, ANCHOR, 0, ANCHOR, 0, 0, 0));

        long[] v = empty();
        mark(v, 0, -1, 0);   // section -1 covers blocks -16..-1 vertically
        assertTrue(BuiltSectionMask.blockCovered(v, SIDE, ANCHOR, 0, ANCHOR, 0, -1, 0));
        assertFalse(BuiltSectionMask.blockCovered(v, SIDE, ANCHOR, 0, ANCHOR, 0, 0, 0));
    }

    @Test
    void outsideTheSquareIsNeverCovered() {
        long[] m = empty();
        for (int x = -C; x <= C; x++) {
            for (int z = -C; z <= C; z++) {
                for (int y = -20; y <= 20; y++) mark(m, x, y, z);
            }
        }
        assertFalse(covered(m, C + 1, 0, 0), "one column past +x");
        assertFalse(covered(m, 0, 0, C + 1), "one column past +z");
        assertFalse(covered(m, -(C + 1), 0, 0), "one column past -x");
    }

    @Test
    void outsideTheRepresentableVerticalSpanIsNeverCovered() {
        // The per-column bitmask spans 64 sections, +/-32 around the camera's. Outside that reads as
        // not covered, which keeps the LOD -- the safe direction, since an over-drawn LOD z-fights
        // and an under-drawn one shows the void.
        long[] m = empty();
        assertFalse(covered(m, 0, 32, 0), "bit 64 does not exist");
        assertFalse(covered(m, 0, -33, 0), "bit -1 does not exist");
        assertFalse(BuiltSectionMask.sectionCovered(new long[] {0}, 1, 0, 0, 0, 0, 31, 0),
                "and a 1x1 square with no bits covers nothing at all");
    }

    @Test
    void theVerticalBitsDoNotLeakIntoEachOther() {
        // Bit packing: 32 sections below the camera through 31 above, so bit 32 is the camera's own
        // section. Off-by-one here would cull one section too high or too low everywhere.
        long[] m = empty();
        mark(m, 0, 0, 0);
        assertTrue(covered(m, 0, 0, 0), "the camera's own section");
        assertFalse(covered(m, 0, 1, 0));
        assertFalse(covered(m, 0, -1, 0));
        for (int y = -32; y <= 31; y++) {
            long[] one = empty();
            mark(one, 0, y, 0);
            for (int probe = -32; probe <= 31; probe++) {
                if (probe == y) {
                    assertTrue(covered(one, 0, probe, 0), "height " + y + " must cover itself");
                } else {
                    assertFalse(covered(one, 0, probe, 0), "height " + y + " must not cover " + probe);
                }
            }
        }
    }

    @Test
    void theAnchorStepKeepsTheCameraInsideWithMargin() {
        // The anchor must be a deterministic function of the camera and leave at least the render
        // distance of margin on every side, or the square would not cover the region the cull is
        // asked about and the far edge would read as "not covered" everywhere.
        int rd = 8;
        int side = rd * 2 + 1 + rd * 2;          // slack == rd, as ANCHOR_SLACK resolves to
        int step = side - rd * 2;
        for (int cam = -100; cam <= 100; cam++) {
            int anchor = BuiltSectionMask.floorToStep(cam - rd, step);
            assertTrue(anchor <= cam - rd, "anchor must be at or before cam-rd");
            assertTrue(cam + rd < anchor + side, "and the far margin must fit inside the square");
        }
    }

    @Test
    void floorToStepFloorsNegativesRatherThanTruncating() {
        // floorDiv, not integer division: -17 / 16 truncates to -1 in Java but floors to -2, and the
        // difference anchors the square on the wrong side of the camera for half the world.
        assertEquals(-32, BuiltSectionMask.floorToStep(-17, 16));
        assertEquals(-16, BuiltSectionMask.floorToStep(-16, 16));
        assertEquals(-16, BuiltSectionMask.floorToStep(-1, 16));
        assertEquals(0, BuiltSectionMask.floorToStep(0, 16));
        assertEquals(16, BuiltSectionMask.floorToStep(16, 16));
        assertEquals(16, BuiltSectionMask.floorToStep(31, 16));
    }

    @Test
    void theAnswerDoesNotDependOnTheCamera() {
        // The point of a world-anchored origin: the predicate takes no camera argument at all, so a
        // world section's answer cannot change as the player moves within an anchor cell. The lag
        // the user saw -- "I feel like I'm dragging the LOD edge with me" -- is impossible by
        // construction rather than merely made smaller by re-uploading faster.
        long[] m = empty();
        mark(m, 4, 3, -2);
        assertTrue(covered(m, 4, 3, -2));
        assertTrue(BuiltSectionMask.sectionCovered(m, SIDE, ANCHOR, 0, ANCHOR, 4, 3, -2),
                "the same question from any caller gives the same answer");
    }


    @Test
    void onlySectionsInsideSodiumsRenderCylinderAreClaimed() {
        // Sodium MESHES a square but RENDERS a Euclidean cylinder. Measured at a moved camera:
        // built=779 with 45 sections (5.8%) past the render distance, out to Chebyshev 10 against
        // rd=8, concentrated in the square corners. Every one is meshed and never drawn, so a mask
        // that counted them culled the LOD in a ring just outside vanilla render distance -- the
        // edge holes. This is the rule that excludes them, in the metric Sodium draws in.
        int rd = 8;
        assertTrue(BuiltSectionMask.withinRenderCylinder(0, 0, rd));
        assertTrue(BuiltSectionMask.withinRenderCylinder(8, 0, rd), "on the axis at the radius");
        assertTrue(BuiltSectionMask.withinRenderCylinder(0, -8, rd));
        assertFalse(BuiltSectionMask.withinRenderCylinder(9, 0, rd), "one past the radius on the axis");
        // The corners of the Chebyshev square are the case that matters: distance sqrt(8^2+8^2) = 11.3.
        assertFalse(BuiltSectionMask.withinRenderCylinder(8, 8, rd), "the square corner is outside the circle");
        assertFalse(BuiltSectionMask.withinRenderCylinder(-8, 8, rd));
        assertFalse(BuiltSectionMask.withinRenderCylinder(6, 6, rd), "sqrt(72)=8.49 > 8");
        assertTrue(BuiltSectionMask.withinRenderCylinder(5, 6, rd), "sqrt(61)=7.81 <= 8");
        // Chebyshev distance alone would have accepted every one of those corners.
        assertEquals(8, Math.max(Math.abs(8), Math.abs(8)));
    }


    @Test
    void sectionsFarBelowTheCameraAreNotClaimed() {
        // The user's altitude report: flying high, the top surface of NEAR LODs is absent and you
        // see straight through, while FAR LODs are fine. Near LODs sit directly under the camera, so
        // they are zero chunks away horizontally -- the old horizontal-only test claimed them
        // whatever their height, while vanilla (a frustum traversal limited by its build distance)
        // was not drawing them. Far LODs are outside the horizontal radius anyway, so they were
        // never claimed and never broke, which is why the symptom split that way.
        int rd = 8;
        assertTrue(BuiltSectionMask.withinRenderDistance(0, 0, 0, rd));
        assertTrue(BuiltSectionMask.withinRenderDistance(0, 4, 0, rd), "directly below, near ground");
        assertTrue(BuiltSectionMask.withinRenderDistance(0, -8, 0, rd), "directly above, at the radius");
        assertFalse(BuiltSectionMask.withinRenderDistance(0, 14, 0, rd),
                "224 blocks straight down is outside 128, and vanilla does not draw it");
        assertFalse(BuiltSectionMask.withinRenderDistance(0, -9, 0, rd));
        // The corner case in three dimensions: inside the horizontal cylinder but far in Y.
        assertFalse(BuiltSectionMask.withinRenderDistance(7, 7, 0, rd), "sqrt(49+49)=9.9 > 8");
        assertTrue(BuiltSectionMask.withinRenderDistance(4, 4, 4, rd), "sqrt(48)=6.9 <= 8");
    }

    @Test
    void anEmptyMaskCoversNothing() {
        long[] m = empty();
        for (int x = -C; x <= C; x += 4) {
            for (int y = -20; y <= 20; y += 4) {
                for (int z = -C; z <= C; z += 4) {
                    assertFalse(covered(m, x, y, z));
                }
            }
        }
    }
}
