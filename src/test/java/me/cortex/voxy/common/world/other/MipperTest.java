package me.cortex.voxy.common.world.other;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Pins the light a mip cell carries, for the one case where the arithmetic is not obvious: a cell
 * whose eight children are all air.
 *
 * <p>This is a regression test for a <b>port</b> defect, not an upstream one. Upstream's
 * {@code Mipper.mip} ends its all-air branch with
 *
 * <pre>blockLight = (blockLight / 8) &amp; 0xF0;  return withLight(I111, blockLight | skyLight);</pre>
 *
 * and the port shipped
 *
 * <pre>blockLight = blockLight / 8;  return withLight(I111, (blockLight &lt;&lt; 4) | skyLight);</pre>
 *
 * <p>The accumulator holds {@code 16 * sum(nibbles)} because every term is masked with {@code 0xF0},
 * so {@code blockLight / 8} is {@code 2 * sum}, not the average. Upstream's {@code & 0xF0} is what
 * turns that back into {@code floor(sum / 8)} placed in the high nibble; the port dropped it and
 * shifted left by four instead, which moves the value up into bits 8+ where {@code withLight}'s
 * {@code & 0xFF} discards it. What survives the mask is {@code (32 * sum) & 0xFF}, i.e.
 * {@code (sum mod 8) << 5} — so the block nibble becomes {@code (sum mod 8) << 1}, and for any
 * {@code sum} that is a multiple of 8 it collapses to zero.
 *
 * <p>Sky is unaffected: it is summed from the {@code 0x0F} halves, divided by 8 and placed in the
 * low nibble, where the corrupted term cannot reach it ({@code 32 * sum} always has a zero low
 * nibble). That is why this bug looks like "torches and lava go dark in the far LOD" and not like
 * the black splotches, and why it was worth separating from them rather than assuming one cause.
 */
class MipperTest {

    /** An air voxel carrying the given light byte; {@code Mipper} special-cases cells by air-ness. */
    private static long air(int light) {
        return Mapper.airWithLight(light);
    }

    private static int skyOf(long id) {
        return Mapper.getLightId(id) & 0x0F;
    }

    private static int blockOf(long id) {
        return (Mapper.getLightId(id) >> 4) & 0x0F;
    }

    private static long mip(long a, long b, long c, long d, long e, long f, long g, long h) {
        // The all-air branch never touches the mapper -- it needs no palette lookup to average
        // light -- so a null mapper is legitimate here and keeps the test independent of Mapper's
        // block registry.
        return Mipper.mip(a, b, c, d, e, f, g, h, null);
    }

    @Test
    void fullyLitAirKeepsBothNibbles() {
        // The case that fails on the port's arithmetic and passes on upstream's: uniform maximum
        // block and sky light, sum = 120 for each, which is a multiple of 8.
        long l = air(0xFF);
        long out = mip(l, l, l, l, l, l, l, l);

        assertEquals(15, blockOf(out), "block light must survive the mip");
        assertEquals(15, skyOf(out), "sky light must survive the mip");
    }

    @Test
    void blockLightIsAveragedIntoTheHighNibbleAtEveryLevel() {
        // Sweep sums that are and are not multiples of 8, since the port's error showed up as a
        // function of (sum mod 8) -- one sample would have missed most of it.
        for (int perCell = 0; perCell <= 15; perCell++) {
            long l = air(perCell << 4);           // block light only, sky 0
            long out = mip(l, l, l, l, l, l, l, l);
            assertEquals(perCell, blockOf(out), "uniform block light " + perCell + " must mip to itself");
        }
    }

    @Test
    void skyLightIsAveragedRoundedUp() {
        // Upstream rounds sky up (Math.ceil) and this pins that, because the same expression is
        // where a "just take the max" instinct would show up and light every cave near the surface.
        long dark = air(0);
        long lit = air(0x0F);
        // one child lit out of eight: ceil(15 / 8) = 2
        assertEquals(2, skyOf(mip(lit, dark, dark, dark, dark, dark, dark, dark)));
        // four of eight: ceil(60 / 8) = 8
        assertEquals(8, skyOf(mip(lit, lit, lit, lit, dark, dark, dark, dark)));
        // all eight: 15
        assertEquals(15, skyOf(mip(lit, lit, lit, lit, lit, lit, lit, lit)));
    }

    @Test
    void theTwoNibblesDoNotLeakIntoEachOther() {
        // The port's bug was a nibble-crossing error, so assert the crossing directly: block light
        // set, sky clear, and the reverse. A leak shows up as one axis moving when the other does.
        long blockOnly = air(0xF0);
        long skyOnly = air(0x0F);

        long allBlock = mip(blockOnly, blockOnly, blockOnly, blockOnly, blockOnly, blockOnly, blockOnly, blockOnly);
        assertEquals(15, blockOf(allBlock));
        assertEquals(0, skyOf(allBlock), "block light must not raise sky light");

        long allSky = mip(skyOnly, skyOnly, skyOnly, skyOnly, skyOnly, skyOnly, skyOnly, skyOnly);
        assertEquals(0, blockOf(allSky), "sky light must not raise block light");
        assertEquals(15, skyOf(allSky));
    }
}
