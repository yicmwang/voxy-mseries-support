package me.cortex.voxy.client.core.rendering;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
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
        // WAS assertTrue. The boundary is STRICT, because Sodium's is: OcclusionCuller.testDistance is
        // `a < c*c`, so a section exactly at the radius is not drawn. An inclusive test here claims a
        // one-section ring around the whole circle that vanilla leaves empty -- at RD 8, ~50 columns of
        // culled LOD with nothing behind it. Same class of error as the slab term, in miniature.
        assertFalse(BuiltSectionMask.withinRenderCylinder(8, 0, rd), "exactly at the radius is not drawn");
        assertTrue(BuiltSectionMask.withinRenderCylinder(7, 0, rd), "one inside it is");
        assertFalse(BuiltSectionMask.withinRenderCylinder(0, -8, rd));
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
    void sectionsInARenderedColumnAreClaimedHoweverDeep() {
        // THIS TEST USED TO ASSERT THE OPPOSITE, and the change is the point: every assertion below was
        // assertFalse for deep sections, holding a CONJUNCTION (`horizontal < rd && |dy| < rd`).
        //
        // That conjunction refuses 90% of what Sodium builds, measured at the users's render distance:
        // with VOXY_VMASK=1 on an unpinned ground-level camera at RD 2,
        //   [Metal-VMASK3] built=192 outsideRenderDistance3D=173 (90.1%) maxChebyshev=6 (rd=2) secY 2..7
        // and the mask claimed 6 columns out of the 81 addressable while Sodium's mesh set is a 5x5
        // square of columns, d01=33 d02=52 sections. Columns at Chebyshev 2 were refused unless their
        // dy was within 1, so most of the terrain at the camera's OWN LEVEL stayed LOD-covered on top
        // of vanilla. That is the near-field overlap, and it is the cull's own operator, not its feed.
        //
        // Sodium's rule is a UNION -- OcclusionCuller.testDistance, called as
        // testDistance(dx*dx+dz*dz, |dy|, searchDistance) from SectionTree.traverse:
        //     (a < c*c) || (b < c)   with a = dx^2+dz^2, b = |dy|
        // so a section passes on EITHER term. The vertical slab term is what claims the corners at the
        // camera's own level; the horizontal term is what claims a column at any depth.
        //
        // The cost is the altitude report this conjunction was written for, and it is knowingly
        // accepted here: a section far below the camera now IS claimed, so if the frustum is not
        // looking at it the LOD is culled over nothing. No distance rule can tell -- the fix is to feed
        // the mask from the sections Sodium actually RENDERS (cull.MD 6.1), not to pick a different
        // operator. Do not "tighten" this back without fixing that first.
        int rd = 8;
        assertTrue(BuiltSectionMask.withinRenderDistance(0, 0, 0, rd));
        assertTrue(BuiltSectionMask.withinRenderDistance(0, 4, 0, rd), "directly below, near ground");
        assertTrue(BuiltSectionMask.withinRenderDistance(0, 8, 0, rd),
                "the horizontal term alone claims the whole column, and vanilla draws the column");
        assertTrue(BuiltSectionMask.withinRenderDistance(0, 14, 0, rd),
                "224 blocks straight down is still in a rendered column -- previously refused here");
        assertTrue(BuiltSectionMask.withinRenderDistance(0, -9, 0, rd));
        assertTrue(BuiltSectionMask.withinRenderDistance(4, 9, 4, rd),
                "inside the horizontal radius, so claimed whatever its height");
    }


    @Test
    void sectionsInsideTheHorizontalRadiusWithModerateDepthAreClaimed() {
        // The reported over-draw, and the case the sphere got wrong. These are sections vanilla DRAWS
        // -- they are inside the horizontal render radius -- that the old 3-D sphere dropped, so the
        // LOD was drawn on top of them. The band is `dx^2+dz^2 < rd^2 <= dx^2+dy^2+dz^2`.
        int rd = 8;
        // 7 chunks out, 4 sections down: vanilla draws it (7 < 8), the sphere scored 49+16 = 65 > 64.
        assertTrue(BuiltSectionMask.withinRenderDistance(7, 4, 0, rd), "the sphere dropped this");
        // The two-axis twin: horizontal 5^2+5^2 = 50 < 64, and the sphere scored 50+16 = 66 > 64.
        assertTrue(BuiltSectionMask.withinRenderDistance(5, 4, 5, rd), "and its two-axis twin");
        assertTrue(BuiltSectionMask.withinRenderDistance(0, 7, 0, rd), "straight down, inside the cap");
        // THESE TWO WERE assertTrue, AND THAT WAS THE BUG THEY PINNED. They asserted that Sodium's
        // union -- (dx^2+dz^2 < rd^2) || (|dy| < rd) -- claims them via the vertical slab term, on the
        // reasoning that Sodium's own rule must be the right one for a cull that mirrors Sodium.
        //
        // It is the right rule for Sodium, which applies it as a TRAVERSAL bound and then clips the
        // result by the frustum and the occlusion tree, so its slab term never becomes claimed ground.
        // Applied as the CLAIM rule over the meshed set it does: on flat terrain every section shares
        // the camera's section Y, so |dy| = 0 < rd holds everywhere and the horizontal radius stops
        // mattering at all. Measured on the superflat fixture at RD 8:
        //
        //   built=754  outsideChebyshev=497  outsideRenderDistance3D=0 (0.0%)
        //   maxChebyshev=31 (rd=8)  secY -4..-4
        //
        // Half the set beyond Chebyshev 8, reaching 31 chunks against a render distance of 8, and
        // nothing refused. The mask culled the LOD out to 31 chunks where vanilla drew to 8, so
        // everything between was culled LOD with nothing behind it -- the void edge at the LOD/vanilla
        // intersection, with its boundary at the MESHED set's edge, which is why it drifted as chunks
        // loaded and unloaded.
        assertFalse(BuiltSectionMask.withinRenderDistance(8, 0, 0, rd),
                "outside the radius: the slab term that used to claim this was the bug");
        assertFalse(BuiltSectionMask.withinRenderDistance(6, 4, 6, rd), "sqrt(72) = 8.49 > 8");
        assertTrue(BuiltSectionMask.withinRenderDistance(4, 4, 4, rd));
    }


    @Test
    void pruningKeepsWhatCouldComeBackAndDropsWhatCannot() {
        // BUILT accumulates one entry per section the player has ever been near -- measured 675 ->
        // 779 -> 1632 -> 3423 across one session -- and rebuilds walk all of it. Pruning is safe
        // ONLY because the distance filter already excludes these from the mask, so dropping one
        // cannot change what is claimed today. The margin keeps anything that could re-enter range
        // without a rebuild; dropping those would under-claim and show as LOD over vanilla.
        int rd = 8;
        assertTrue(BuiltSectionMask.worthKeeping(0, 0, 0, rd));
        assertTrue(BuiltSectionMask.worthKeeping(rd, 0, 0, rd), "at the radius");
        assertTrue(BuiltSectionMask.worthKeeping(rd + 3, 0, 0, rd), "just outside, could come back");
        // WAS assertTrue, and it was there only to keep step with the slab term: while the claim rule
        // was the union it claimed |dy| < rd at any horizontal distance, so this had to be kept or
        // pruning would have under-claimed. With the slab term gone the claim rule is the radius
        // alone, so this is outside it again and only the margin saves it -- which is what the line
        // above tests. See sectionsInsideTheHorizontalRadiusWithModerateDepthAreClaimed.
        assertFalse(BuiltSectionMask.worthKeeping(rd + 5, 0, 0, rd), "outside the radius and the margin");
        assertFalse(BuiltSectionMask.worthKeeping(0, 40, 0, rd), "left far below, past the bit span");
        assertFalse(BuiltSectionMask.worthKeeping(100, 0, 0, rd), "left far behind, outside the square");
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

    @Test
    void aRenderDistanceChangeDoesNotEmptyTheMask() {
        // The measured defect, and the reason bug A was permanent rather than occasional.
        //
        // Sodium constructs a new RenderSectionManager on a render-distance change as well as on a
        // level change, and the mixin cleared the mask on both. The set only refills from mesh-upload
        // deltas, and Sodium does not re-mesh a chunk that is already built, so the clear never came
        // back: a live run with VOXY_VMASK=1 held built=0, columns=0/81, sections=0 for 3600
        // consecutive frames after its render distance went 3 to 2. An empty mask means the
        // fragment-stage discard never fires, so during all of that the LOD was drawn over vanilla
        // everywhere -- not a mis-shaped cull, no cull.
        //
        // Nothing is lost by keeping it: a section the smaller distance no longer covers cannot set a
        // bit, because the per-frame distance filter refuses it whatever the set holds.
        final Object level = new Object();
        assertTrue(BuiltSectionMask.resetForLevel(level), "a level the mask has not seen clears it");
        BuiltSectionMask.add(12345L);
        assertEquals(1, BuiltSectionMask.builtCount());

        assertFalse(BuiltSectionMask.resetForLevel(level), "the same level must not clear");
        assertEquals(1, BuiltSectionMask.builtCount(), "a section survives a render-distance change");

        assertTrue(BuiltSectionMask.resetForLevel(new Object()), "a different level does clear");
        assertEquals(0, BuiltSectionMask.builtCount(), "another dimension's sections go with it");
    }

    @Test
    void theSectionIsReconstructedByAddingBeforeFlooring() {
        // quads.frag floored the camera-RELATIVE offset and added it to a FLOORED camera, and
        // floor(cam) + floor(rel) is not floor(cam + rel):
        //
        //     floor(camX) + floor(fragX - camX)  ==  floor(fragX) - [frac(fragX) < frac(camX)]
        //
        // The stray -1 lands on every fragment whose in-block fraction is below the camera's, so the
        // culled region's edges were positioned by frac(camX) -- where the camera sits in its own
        // block -- and crawled along with the player instead of stepping at chunk boundaries.
        // Reported from play: "the culling region should only step as I cross chunk boundaries. But
        // now, the edges seem to follow me as I move."
        //
        // A one-block band out of every sixteen, so it is invisible in a still frame and in every
        // screenshot taken from one. This walks the camera through every eighth of a block and the
        // fragment across a section boundary, which is the only place the two forms differ.
        for (double fragX : new double[]{ 16.0, 16.25, 16.5, 16.75, 17.0, 31.9, 32.0, -16.5, -15.75 }) {
            for (int eighth = 0; eighth < 8; eighth++) {
                final double camFrac = eighth / 8.0;
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
}
