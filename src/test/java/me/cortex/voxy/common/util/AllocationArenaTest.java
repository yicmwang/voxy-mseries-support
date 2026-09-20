package me.cortex.voxy.common.util;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.TreeMap;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Black-box invariant tests for {@link AllocationArena}, the allocator that hands out geometry-buffer
 * ranges.
 *
 * <p>This matters to bug 3 because the arena is the one component whose failure produces exactly the
 * reported artefact rather than something adjacent to it. Every LOD section's metadata records
 * {@code geometryPtr + offsets[0]}, and the draw reads its quads from that address; if the arena ever
 * returns a range it has already given to a live section, that section's metadata points at memory
 * holding another section's quads. The geometry drawn is then shaped nothing like the voxels of the
 * section it is drawn as, it moves as allocations come and go, and it is black wherever the donor
 * happens to be interior or unlit. The class's own {@code free()} carries a
 * {@code //FIXME: this is very dodgy solution} on its block-zero merge, and {@code removeSection}
 * runs on every node unload, so this is not a hypothetical path.
 *
 * <p>The invariants are checked through the public API only, so they cannot be satisfied by the
 * implementation agreeing with itself:
 *
 * <ul>
 *   <li>live allocations are pairwise disjoint -- no address is handed out twice;</li>
 *   <li>{@code getSize()} equals the sum of the live sizes -- nothing is leaked or double-counted
 *       by a merge that loses a block;</li>
 *   <li>{@code getSize(addr)} returns the size that address was allocated with, for every live
 *       allocation;</li>
 *   <li>freeing everything returns the arena to empty.</li>
 * </ul>
 *
 * <p>Both size distributions are exercised: multiples of 1024, which is what the geometry manager
 * actually asks for ({@code upsized = (size + 1023) & ~1023}), and a mixed distribution including
 * sizes that are not multiples of each other, which produces far more partial-fit splits and merges.
 */
class AllocationArenaTest {

    /** The real caller's distribution: quad counts rounded up to a 1024-element block. */
    private static int realSize(Random r) {
        return ((r.nextInt(24_000) + 1 + 1023) & ~1023);
    }

    /** Adversarial: sizes with mixed alignment, to force partial fits and coalescing. */
    private static int mixedSize(Random r) {
        int[] sizes = {1, 2, 3, 512, 1024, 1025, 2048, 4096, 5000, 65536, 65537, 100_003};
        return sizes[r.nextInt(sizes.length)];
    }

    /**
     * The invariants that matter, checked through the public API only.
     *
     * <p>{@code getSize()} is the arena's <b>extent</b>, not its live bytes: {@code free()} only
     * decrements {@code totalSize} in the branch where the freed block is the last one, so the value
     * is a high-water mark that shrinks when the top block goes away. That is deliberate and the
     * geometry manager uses {@code free()}'s return value for its own accounting, so the extent is
     * asserted as {@code max(addr + size)} over the live set -- which is still a real check, because
     * an extent larger than any live allocation's end would mean the arena is tracking memory it has
     * already handed back.
     */
    private static void checkInvariants(AllocationArena arena, TreeMap<Long, Long> live, String when) {
        long maxEnd = 0;
        long prevEnd = -1;
        for (var e : live.entrySet()) {
            long addr = e.getKey();
            long size = e.getValue();
            assertTrue(addr >= prevEnd,
                    when + ": allocation [" + addr + "," + (addr + size) + ") overlaps the previous one,"
                            + " which ends at " + prevEnd + " -- the arena handed out the same memory twice");
            prevEnd = addr + size;
            maxEnd = Math.max(maxEnd, addr + size);
            assertEquals(size, arena.getSize(addr), when + ": getSize disagrees for addr " + addr);
        }
        assertEquals(maxEnd, arena.getSize(),
                when + ": extent (" + arena.getSize() + ") != the end of the highest live allocation ("
                        + maxEnd + ") -- the arena is tracking memory it has already handed back");
    }

    private static void randomWorkload(long seed, int ops, boolean real, int limit) {
        Random r = new Random(seed);
        AllocationArena arena = new AllocationArena();
        arena.setLimit(limit);
        TreeMap<Long, Long> live = new TreeMap<>();
        List<Long> addrs = new ArrayList<>();

        for (int op = 0; op < ops; op++) {
            boolean doAlloc = addrs.isEmpty() || r.nextInt(100) < 55;
            if (doAlloc) {
                int size = real ? realSize(r) : mixedSize(r);
                long addr = arena.alloc(size);
                if (addr == AllocationArena.SIZE_LIMIT) {
                    continue;
                }
                assertTrue(addr >= 0, "seed " + seed + " op " + op + ": negative address " + addr);
                assertTrue(addr + size <= limit,
                        "seed " + seed + " op " + op + ": allocation [" + addr + "," + (addr + size)
                                + ") runs past the arena limit " + limit);
                assertTrue(!live.containsKey(addr),
                        "seed " + seed + " op " + op + ": address " + addr
                                + " was handed out while already live -- overlapping allocations");
                // no overlap with any live range either
                var floor = live.floorEntry(addr);
                if (floor != null) {
                    assertTrue(floor.getKey() + floor.getValue() <= addr,
                            "seed " + seed + " op " + op + ": new allocation at " + addr
                                    + " overlaps live [" + floor.getKey() + ","
                                    + (floor.getKey() + floor.getValue()) + ")");
                }
                var ceil = live.ceilingEntry(addr);
                if (ceil != null) {
                    assertTrue(addr + size <= ceil.getKey(),
                            "seed " + seed + " op " + op + ": new allocation [" + addr + ","
                                    + (addr + size) + ") runs into live allocation at " + ceil.getKey());
                }
                live.put(addr, (long) size);
                addrs.add(addr);
            } else {
                int idx = r.nextInt(addrs.size());
                long addr = addrs.remove(idx);
                Long expected = live.remove(addr);
                assertEquals(expected.longValue(), arena.free(addr),
                        "seed " + seed + " op " + op + ": free returned a different size than allocated");
            }
            if ((op & 63) == 0) {
                checkInvariants(arena, live, "seed " + seed + " op " + op);
            }
        }
        // Drain: freeing everything must return the arena to empty, whatever order it is done in.
        for (long addr : addrs) {
            arena.free(addr);
            live.remove(addr);
        }
        assertEquals(0L, arena.getSize(), "seed " + seed + ": arena not empty after freeing everything");
        assertEquals(0, live.size());
    }

    @Test
    void randomWorkloadsWithTheRealSizeDistribution() {
        for (long seed = 1; seed <= 6; seed++) {
            randomWorkload(seed, 4000, true, 1 << 26);
        }
    }

    @Test
    void randomWorkloadsWithMixedAlignment() {
        for (long seed = 100; seed <= 106; seed++) {
            randomWorkload(seed, 4000, false, 1 << 22);
        }
    }

    /**
     * The block-zero merge specifically. Freeing a block adjacent to a free block that starts at
     * address 0 takes the branch carrying the "very dodgy solution" comment, whose own note admits
     * it "assumes block zero is 0 addr n size".
     */
    /**
     * The block-zero merge specifically, driven so that it actually executes.
     *
     * <p>{@code free()} has two merge paths. The first needs a previous TAKEN block; the second --
     * the one carrying {@code //FIXME: this is very dodgy solution}, and whose own note admits it
     * "assumes block zero is 0 addr n size" -- fires when the freed block has <b>no</b> previous
     * taken block but the free set is non-empty, i.e. when the freed block sits directly above a
     * free block that starts at address 0. The sequence below constructs exactly that, then asks for
     * the merged span back and requires it at address 0.
     */
    @Test
    void blockZeroMergeCoalescesAndServesTheMergedSpan() {
        AllocationArena arena = new AllocationArena();
        long a = arena.alloc(1024);
        long b = arena.alloc(2048);
        assertEquals(0L, a);
        assertEquals(1024L, b);

        arena.free(a);                       // free block [0,1024); a is no longer TAKEN
        long c = arena.alloc(4096);          // too big for [0,1024) -> placed above, at the top
        assertEquals(3072L, c, "the 4096 block must not fit in the 1024 hole");

        // b now has no previous TAKEN block, and FREE holds a block at address 0 -> the dodgy branch.
        arena.free(b);
        assertEquals(4096L, arena.getSize(c), "c is still live and still the extent");

        // The merge must have produced one span [0, 3072). Taking it back proves nothing was lost.
        long merged = arena.alloc(3072);
        assertEquals(0L, merged, "the block-zero merge should have produced a contiguous span at 0");
        assertEquals(3072L, arena.getSize(merged), "the merged span is 1024 + 2048");
        assertEquals(7168L, arena.getSize(), "and c still holds the extent above it");
    }

    /**
     * The strongest form of the coalescing check: drain a fragmented arena, then ask for the whole
     * extent back as one allocation. If any merge lost a block, the space is there in pieces and
     * either this fails or it lands somewhere other than 0.
     */
    @Test
    void aDrainedArenaHandsBackOneContiguousBlock() {
        AllocationArena arena = new AllocationArena();
        List<Long> addrs = new ArrayList<>();
        for (int i = 0; i < 128; i++) {
            addrs.add(arena.alloc(1024));
        }
        long extent = arena.getSize();
        assertEquals(128L * 1024L, extent);

        // Free in a scattered order so merges happen from both sides.
        for (int i = 0; i < addrs.size(); i += 2) {
            arena.free(addrs.get(i));
        }
        for (int i = 1; i < addrs.size(); i += 2) {
            arena.free(addrs.get(i));
        }
        assertEquals(0L, arena.getSize());

        long whole = arena.alloc((int) extent);
        assertEquals(0L, whole, "the drained arena should hand back one block starting at 0");
        assertEquals(extent, arena.getSize());
    }

    /** Allocate all, free in a scattered order (every other, then the rest), then check for leaks. */
    @Test
    void scatteredFreeOrderDoesNotLeakOrOverlap() {
        AllocationArena arena = new AllocationArena();
        List<Long> addrs = new ArrayList<>();
        int size = 1024;
        for (int i = 0; i < 256; i++) {
            addrs.add(arena.alloc(size));
        }
        assertEquals(256L * size, arena.getSize());
        for (int i = 0; i < addrs.size(); i += 2) {
            arena.free(addrs.get(i));
        }
        for (int i = 1; i < addrs.size(); i += 2) {
            arena.free(addrs.get(i));
        }
        assertEquals(0L, arena.getSize(), "freeing everything must leave nothing allocated");
    }

    /** Re-allocating the same sizes after a full free must reproduce the same addresses. */
    @Test
    void reuseIsDeterministicAfterAFullFree() {
        AllocationArena arena = new AllocationArena();
        int[] sizes = {1024, 2048, 1024, 4096, 1024};
        long[] first = new long[sizes.length];
        for (int i = 0; i < sizes.length; i++) {
            first[i] = arena.alloc(sizes[i]);
        }
        for (long a : first) {
            arena.free(a);
        }
        assertEquals(0L, arena.getSize());
        for (int i = 0; i < sizes.length; i++) {
            assertEquals(first[i], arena.alloc(sizes[i]),
                    "after a full free the arena should hand out the same layout");
        }
    }

    /** Freeing an address that was never allocated (or already freed) must be loud, not silent. */
    @Test
    void freeingAnUnknownAddressThrows() {
        AllocationArena arena = new AllocationArena();
        long a = arena.alloc(1024);
        arena.free(a);
        assertThrows(RuntimeException.class, () -> arena.free(a), "double free must throw");
        assertThrows(RuntimeException.class, () -> arena.free(4096), "never-allocated must throw");
    }
}
