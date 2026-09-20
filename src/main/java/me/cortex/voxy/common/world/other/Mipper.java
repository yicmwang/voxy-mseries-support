package me.cortex.voxy.common.world.other;

import static me.cortex.voxy.common.world.other.Mapper.withLight;

//Mipper for data
public class Mipper {
    //TODO: compute the opacity of the block then mip w.r.t those blocks
    // as distant horizons done


    //TODO: also pass in the level its mipping from, cause at lower levels you want to preserve block details
    // but at higher details you want more air
    /**
     * The light byte a mip cell should carry: the brightest sky and the brightest block light found
     * anywhere among its eight children, packed as {@code (block << 4) | sky}.
     *
     * <p>Split out from {@link #mip} so the rule is unit-testable without a block registry —
     * {@code mip} needs a {@link Mapper} to pick the representative child, and a {@code Mapper}
     * needs deserialised block states. The rule is the part that was wrong, so the rule is the part
     * worth pinning.
     *
     * <p>Sky and block are maximised independently rather than by comparing packed bytes: a child
     * with sky 15 and a child with block 12 must combine into the brightest cell, not into whichever
     * of the two packed values happens to be numerically larger.
     */
    static int brightestLight(long a, long b, long c, long d,
                              long e, long f, long g, long h) {
        int sky = 0;
        int block = 0;
        for (final long id : new long[] {a, b, c, d, e, f, g, h}) {
            final int light = Mapper.getLightId(id);
            if ((light & 0x0F) > sky) sky = light & 0x0F;
            if (((light >> 4) & 0x0F) > block) block = (light >> 4) & 0x0F;
        }
        return (block << 4) | sky;
    }

    public static long mip(long I000, long I100, long I001, long I101,
                           long I010, long I110, long I011, long I111,
                          Mapper mapper) {
        //TODO: do a stable sort on all the entires, w.r.t the opacity and maybe light as a secondary???
        // then select the highest value
        // UPDATE, dumbass, the highest value _is_ the max/min



        int max = -1;

        //TODO: mip with respect to all the variables, what that means is take whatever has the highest count and return that
        //TODO: also average out the light level and set that as the new light level
        //For now just take the most top corner

        //TODO: i think it needs to compute the _max_ light level, since e.g. if a point is bright irl
        // you can see it from really really damn far away.
        // it could be a heavily weighted average with a huge preference to the top most lighting value
        if (!Mapper.isAir(I111)) {
            max = (mapper.getBlockStateOpacity(I111)<<4)|0b111;
        }
        if (!Mapper.isAir(I110)) {
            max = Math.max((mapper.getBlockStateOpacity(I110)<<4)|0b110, max);
        }
        if (!Mapper.isAir(I011)) {
            max = Math.max((mapper.getBlockStateOpacity(I011)<<4)|0b011, max);
        }
        if (!Mapper.isAir(I010)) {
            max = Math.max((mapper.getBlockStateOpacity(I010)<<4)|0b010, max);
        }
        if (!Mapper.isAir(I101)) {
            max = Math.max((mapper.getBlockStateOpacity(I101)<<4)|0b101, max);
        }
        if (!Mapper.isAir(I100)) {
            max = Math.max((mapper.getBlockStateOpacity(I100)<<4)|0b100, max);
        }
        if (!Mapper.isAir(I001)) {
            max = Math.max((mapper.getBlockStateOpacity(I001)<<4)|0b001, max);
        }
        if (!Mapper.isAir(I000)) {
            max = Math.max((mapper.getBlockStateOpacity(I000)<<4), max);
        }

        if (max != -1) {
            // The chosen child decides the block, the biome and the shape -- but NOT the light.
            //
            // It used to decide the light too, by being returned unchanged, and that is the bug-3
            // mechanism: a solid block's own cell light is 0 in both nibbles (light does not
            // propagate into a solid), so a coarse cell holding solid and air together became a
            // solid cell carrying (0,0). At coarse LOD levels that cell IS the terrain surface, so
            // every face taking its light from it went black. Measured with VOXY_LOD_SHOW_LIGHT:
            // 26.24% of drawn LOD fragments carry a zero light byte, all in the mid-distance band
            // where the LOD is coarse, and 0% near the camera where it is not.
            //
            // Upstream's own TODO prescribes the fix: "i think it needs to compute the _max_ light
            // level, since e.g. if a point is bright irl you can see it from really really damn far
            // away". The brightest child is what a viewer outside the cell would see, and a cell
            // with no lit child -- a sealed one -- still resolves to 0, so caves stay dark.
            final long chosen = switch (max&0b111) {
                case 0 -> I000;
                case 1 -> I001;
                case 2 -> I010;
                case 3 -> I011;
                case 4 -> I100;
                case 5 -> I101;
                case 6 -> I110;
                case 7 -> I111;
                default -> throw new IllegalStateException("Unexpected value: " + (max&0b111));
            };
            return withLight(chosen, brightestLight(I000, I100, I001, I101, I010, I110, I011, I111));
        } else {
            int blockLight = (Mapper.getLightId(I000) & 0xF0) + (Mapper.getLightId(I001) & 0xF0) + (Mapper.getLightId(I010) & 0xF0) + (Mapper.getLightId(I011) & 0xF0) +
                    (Mapper.getLightId(I100) & 0xF0) + (Mapper.getLightId(I101) & 0xF0) + (Mapper.getLightId(I110) & 0xF0) + (Mapper.getLightId(I111) & 0xF0);
            int skyLight = (Mapper.getLightId(I000) & 0x0F) + (Mapper.getLightId(I001) & 0x0F) + (Mapper.getLightId(I010) & 0x0F) + (Mapper.getLightId(I011) & 0x0F) +
                    (Mapper.getLightId(I100) & 0x0F) + (Mapper.getLightId(I101) & 0x0F) + (Mapper.getLightId(I110) & 0x0F) + (Mapper.getLightId(I111) & 0x0F);
            // The accumulator holds 16 * sum(nibbles) because every term above is masked with 0xF0,
            // so `blockLight / 8` is 2 * sum and not the average. `& 0xF0` is what turns it back into
            // floor(sum / 8) already sitting in the high nibble. This is upstream's expression
            // verbatim; the port had `blockLight / 8` then `(blockLight << 4) | skyLight`, which
            // moved the value up into bits 8+ where withLight's `& 0xFF` discards it and left
            // `(32 * sum) & 0xFF` instead -- zero whenever sum is a multiple of 8, i.e. no block
            // light at all in the common uniformly-lit case. Shedding light, not the splotches:
            // sky is summed from different bits and was never touched by the corruption, which is
            // why the symptom was torches and lava going dark in the far LOD rather than terrain
            // going black. Pinned by MipperTest.
            blockLight = (blockLight / 8) & 0xF0;
            skyLight = (int) Math.ceil((double) skyLight / 8);

            return withLight(I111, blockLight | skyLight);
        }
    }
}
