package me.cortex.voxy.client.core.rendering.section.backend.mdic;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Pins the allocation table {@code [Metal-DRAWCHK]} builds from the section metadata buffer: how a
 * draw command's quad range is matched to the allocation that owns it, and how two sections sharing
 * memory is detected.
 *
 * <p>The table is reconstructed from metadata alone because the manager's arena is not reachable from
 * the render thread. That is exact rather than approximate: the manager allocates
 * {@code upsized = (itemCount + 1023) & ~1023} from a 1024-aligned arena, and
 * {@code RenderDataFactory} sets {@code offsets[0] = 0}, so the metadata's {@code quadStart} is the
 * allocation address and {@code sum(counts)} is {@code itemCount} — the allocation is
 * {@code [quadStart, quadStart + ceil1024(sum(counts)))}.
 *
 * <p>Two failure modes matter, and neither can be produced by a shading fault:
 * <ul>
 *   <li>a command whose range is in no allocation (<b>orphan</b>) or spans two of them
 *       (<b>straddle</b>) — the draw reads arbitrary geometry-buffer contents;</li>
 *   <li>two live sections given overlapping ranges (<b>overlap</b>) — one section's metadata points
 *       at memory holding another's quads, so the shape drawn is not the shape of the section.</li>
 * </ul>
 */
class AllocationTableTest {

    /** Flat {@code [start, end, sid]} triples, as {@code rebuildAllocationTable} produces. */
    private static long[] table(final long[]... allocs) {
        final long[] t = new long[allocs.length * 3];
        for (int i = 0; i < allocs.length; i++) {
            t[i * 3] = allocs[i][0];
            t[i * 3 + 1] = allocs[i][1];
            t[i * 3 + 2] = allocs[i][2];
        }
        return t;
    }

    @Test
    void findsTheAllocationOwningAQuad() {
        final long[] t = table(new long[] {0, 1024, 7}, new long[] {1024, 2048, 9}, new long[] {4096, 5120, 11});
        assertEquals(0, MDICSectionRenderer.findContaining(t, 3, 0));
        assertEquals(0, MDICSectionRenderer.findContaining(t, 3, 1023));
        assertEquals(1, MDICSectionRenderer.findContaining(t, 3, 1024));
        assertEquals(1, MDICSectionRenderer.findContaining(t, 3, 2047));
        assertEquals(2, MDICSectionRenderer.findContaining(t, 3, 4096));
        assertEquals(2, MDICSectionRenderer.findContaining(t, 3, 5119));
    }

    @Test
    void aQuadInNoAllocationIsAnOrphan() {
        final long[] t = table(new long[] {0, 1024, 7}, new long[] {4096, 5120, 11});
        assertEquals(-1, MDICSectionRenderer.findContaining(t, 2, 1024), "first past the end");
        assertEquals(-1, MDICSectionRenderer.findContaining(t, 2, 3000), "in the gap");
        assertEquals(-1, MDICSectionRenderer.findContaining(t, 2, 5120), "past the last end");
    }

    @Test
    void anEmptyTableContainsNothing() {
        assertEquals(-1, MDICSectionRenderer.findContaining(new long[0], 0, 0));
        assertEquals(-1, MDICSectionRenderer.findContaining(new long[3], 0, 0));
    }

    /**
     * The straddle case: a command starting in one allocation and running past its end. The draw
     * would stitch together quads from two unrelated sections.
     */
    @Test
    void aRangeRunningPastItsAllocationIsAStraddle() {
        final long[] t = table(new long[] {0, 1024, 7}, new long[] {1024, 2048, 9});
        final int slot = MDICSectionRenderer.findContaining(t, 2, 1000);
        assertEquals(0, slot);
        assertEquals(1024L, t[slot * 3 + 1], "allocEnd -- a range ending past this is a straddle");
    }

    @Test
    void adjacentAllocationsDoNotCountAsOverlapping() {
        final long[] t = table(new long[] {0, 1024, 7}, new long[] {1024, 2048, 9}, new long[] {2048, 3072, 11});
        assertEquals(0L, MDICSectionRenderer.countOverlaps(t, 3));
    }

    @Test
    void twoSectionsSharingMemoryIsAnOverlap() {
        // What a bad arena free/merge would produce: sid 9's range starts inside sid 7's.
        final long[] t = table(new long[] {0, 2048, 7}, new long[] {1024, 3072, 9});
        assertEquals(1L, MDICSectionRenderer.countOverlaps(t, 2));
    }

    @Test
    void anIdenticalRangeIsAnOverlap() {
        final long[] t = table(new long[] {0, 1024, 7}, new long[] {0, 1024, 9});
        assertEquals(1L, MDICSectionRenderer.countOverlaps(t, 2));
    }
}
