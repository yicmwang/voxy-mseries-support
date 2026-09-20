package me.cortex.voxy.client.core.rendering.section.backend.mdic;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Pins the arithmetic behind {@code [Metal-DRAWCHK]}, the CPU check that a draw command's quad range
 * lies inside the allocation of the section it names.
 *
 * <p>The chain being tested is a packing and an unpacking written in two different languages, with a
 * shader walking the result in a third place. {@code BasicAsyncGeometryManager.SectionMeta}
 * {@code .writeMetadataSplitParts} packs eight cumulative offsets into four uints --
 * {@code (offsets[b+1]-offsets[b])} in the low half, the next delta in the high half, and
 * {@code itemCount - offsets[7]} in the last high half. {@code cmdgen.comp} then walks them, emitting
 * one draw command per non-zero group with {@code baseVertex = ptr<<2} and
 * {@code count = quads*6}. So a command is only legitimate if its {@code (baseVertex>>2, count/6)}
 * pair is one of the eight {@code (ptr, count)} groups, and the last group must land inside the
 * geometry heap.
 *
 * <p>The case worth holding still is the last group. {@code BuiltSection}'s own
 * {@code verifyBuiltSectionOffsets} checks the deltas <i>between</i> offsets and never the final
 * segment, because {@code itemCount} is the geometry buffer's element count and {@code BuiltSection}
 * does not carry it. If {@code offsets[7]} ever exceeded {@code itemCount}, that delta is negative,
 * packs into the high half as a near-{@code 0xFFFF} count, and the draw reads tens of thousands of
 * quads past its own allocation -- one section's geometry drawn as another's, which is the shape of
 * the reported splotches and cannot be reached by any shading fault.
 */
class QuadGroupValidationTest {

    /** The packing, transcribed from {@code writeMetadataSplitParts}, so the two cannot drift apart. */
    private static long[] pack(final int[] offsets, final int itemCount) {
        final long[] c = new long[4];
        for (int i = 0; i < 4; i++) {
            final long lo = Integer.toUnsignedLong(offsets[i * 2 + 1] - offsets[i * 2]);
            final long hi = (i == 3)
                    ? Integer.toUnsignedLong(itemCount - offsets[7])
                    : Integer.toUnsignedLong(offsets[i * 2 + 2] - offsets[i * 2 + 1]);
            c[i] = (lo & 0xFFFFL) | ((hi & 0xFFFFL) << 16);
        }
        return c;
    }

    @Test
    void unpacksTheEightGroupsInCmdgensOrder() {
        // translucent, double-sided, down, up, north, south, west, east
        final int[] offsets = {0, 10, 20, 30, 40, 50, 60, 70};
        final long[] groups = MDICSectionRenderer.unpackQuadGroups(
                pack(offsets, 80)[0], pack(offsets, 80)[1], pack(offsets, 80)[2], pack(offsets, 80)[3]);
        assertArrayEqualsLongs(new long[] {10, 10, 10, 10, 10, 10, 10, 10}, groups);
    }

    @Test
    void quadEndIsTheAllocationsEnd() {
        // quadStart is geometryPtr + offsets[0]; the groups sum to itemCount - offsets[0], so quadEnd
        // is geometryPtr + itemCount -- the boundary the geometry heap allocation must cover.
        final int[] offsets = {0, 4, 9, 9, 12, 12, 12, 20};
        final int itemCount = 25;
        final long geometryPtr = 100L;
        final long[] groups = unpack(offsets, itemCount);
        assertEquals(geometryPtr + itemCount, MDICSectionRenderer.quadEnd(geometryPtr, groups));
        assertEquals(25L, MDICSectionRenderer.quadEnd(geometryPtr, groups) - geometryPtr,
                "with offsets[0] == 0 the groups tile exactly [geometryPtr, geometryPtr + itemCount)");
    }

    @Test
    void aCommandIsOneOfTheEightGroups() {
        final int[] offsets = {0, 4, 9, 9, 12, 12, 12, 20};
        final long[] groups = unpack(offsets, 25);
        final long quadStart = 1000L;

        // Every group cmdgen.comp would emit for this section, at its real baseVertex and count.
        long ptr = quadStart;
        int emitted = 0;
        for (int g = 0; g < 8; g++) {
            if (groups[g] != 0) {
                assertTrue(MDICSectionRenderer.matchesQuadGroup(quadStart, groups, ptr, groups[g]),
                        "group " + g + " at ptr " + ptr + " count " + groups[g]);
                emitted++;
            }
            ptr += groups[g];
        }
        // Groups 2, 4 and 5 (down, north, south) are empty here, so cmdgen emits five commands and
        // the three empty groups must match nothing.
        assertEquals(5, emitted, "only the non-empty groups emit a command");
    }

    @Test
    void anEmptyGroupDoesNotMatchACommandClaimingIt() {
        final int[] offsets = {0, 4, 9, 9, 12, 12, 12, 20};
        final long[] groups = unpack(offsets, 25);
        // Group 2 (down) is empty at ptr quadStart+4+5. A command claiming it is not legitimate.
        assertFalse(MDICSectionRenderer.matchesQuadGroup(1000L, groups, 1009L, 0));
    }

    @Test
    void aCommandFromADifferentSectionDoesNotMatch() {
        final long[] groups = unpack(new int[] {0, 4, 9, 9, 12, 12, 12, 20}, 25);
        assertFalse(MDICSectionRenderer.matchesQuadGroup(1000L, groups, 1000L + 21L, 4L));
        assertFalse(MDICSectionRenderer.matchesQuadGroup(1000L, groups, 1000L, 5L));
    }

    /**
     * The failure this check exists for: a final offset past {@code itemCount} packs as a negative
     * delta, the unsigned unpack turns it into a huge count, and the section's claimed range runs far
     * past the heap. Without the range check the draw would read whatever the allocator put after it.
     */
    @Test
    void aFinalOffsetPastItemCountIsCaughtByTheRangeCheck() {
        final int[] offsets = {0, 4, 9, 9, 12, 12, 12, 900};
        final int itemCount = 25;// geometryBuffer.size/8, i.e. what BuiltSection's own guard cannot see
        final long[] groups = unpack(offsets, itemCount);

        assertEquals(65536L - 900L + 25L, groups[7], "the negative delta wraps into the high half");
        final long heapElements = 4096L;
        final long quadEnd = MDICSectionRenderer.quadEnd(1000L, groups);
        assertTrue(quadEnd > heapElements,
                "quadEnd=" + quadEnd + " must exceed a " + heapElements + "-element heap so rangeOob fires");
    }

    private static long[] unpack(final int[] offsets, final int itemCount) {
        final long[] c = pack(offsets, itemCount);
        return MDICSectionRenderer.unpackQuadGroups(c[0], c[1], c[2], c[3]);
    }

    private static void assertArrayEqualsLongs(final long[] expected, final long[] actual) {
        assertEquals(expected.length, actual.length);
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], actual[i], "index " + i);
        }
    }
}
