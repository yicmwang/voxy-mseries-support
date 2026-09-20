package me.cortex.voxy.common.world.service;

import me.cortex.voxy.common.voxelization.ILightingSupplier;
import net.minecraft.world.level.chunk.DataLayer;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;

/**
 * Pins the light byte Voxy bakes into a LOD voxel against Minecraft's own notion of what an
 * absent light layer means.
 *
 * <p>This is the regression test for the black LOD splotches. {@code getLightingSupplier} used to
 * return a constant {@code (byte) 0} whenever it could not find both layers, and to force the
 * missing half of a one-layer section to zero. For SKY that is backwards: a section at or above
 * its column's sky-light source stores no {@code DataLayer} <i>because</i> it is uniformly fully
 * lit, and {@code SkyLightSectionStorage.getLightValue} answers 15 for it. The old code answered
 * 0, baking the brightest terrain in the world — open ground with nothing above it to shade it —
 * as pitch black. The block layer really does default to 0
 * ({@code BlockLightSectionStorage.getLightValue}), so only the sky half is inverted.
 *
 * <p>These assertions are claims about Minecraft, not about Voxy's preferences, so they were
 * verified against the shipped bytecode of both storage classes before being written down.
 */
class LightingSupplierTest {

    /** Sky light is the low nibble and block light the high one; see {@code getLighting}. */
    private static int skyOf(byte light) {
        return light & 0x0F;
    }

    private static int blockOf(byte light) {
        return (light >> 4) & 0x0F;
    }

    private static DataLayer filled(int value) {
        DataLayer layer = new DataLayer();
        layer.fill(value);
        return layer;
    }

    @Test
    void absentSkyLayerReadsAsFullyLit() {
        // The regression. Both layers absent is the shape of open, unshaded ground: there is
        // nothing to store, which is exactly why it is fully lit. The old supplier emitted 0x00
        // here, which the fragment shader turns into the darkest lightmap texel -- a black patch.
        ILightingSupplier supplier = VoxelIngestService.lightingSupplier(null, null);

        for (int y = 0; y < 16; y += 5) {
            for (int z = 0; z < 16; z += 5) {
                for (int x = 0; x < 16; x += 5) {
                    byte light = supplier.supply(x, y, z);
                    assertEquals(15, skyOf(light), "absent sky layer must read as fully lit");
                    assertEquals(0, blockOf(light), "absent block layer must read as unlit");
                }
            }
        }
    }

    @Test
    void absentSkyLayerDoesNotOverrideStoredBlockLight() {
        // A torch-lit section with no shading anywhere above it: block layer stored, sky absent.
        // The old code took its "block only" branch and kept the block light but zeroed the sky,
        // which is the red-speckle-inside-black signature in the light-readout frames.
        DataLayer block = new DataLayer();
        block.set(3, 4, 5, 12);
        block.set(0, 0, 0, 7);

        ILightingSupplier supplier = VoxelIngestService.lightingSupplier(block, null);

        assertEquals(12, blockOf(supplier.supply(3, 4, 5)));
        assertEquals(15, skyOf(supplier.supply(3, 4, 5)));
        assertEquals(7, blockOf(supplier.supply(0, 0, 0)));
        assertEquals(15, skyOf(supplier.supply(0, 0, 0)));
    }

    @Test
    void storedAllZeroSkyLayerStaysDark() {
        // The counterpart to the fix, and the reason it keys on null rather than on
        // DataLayer.isEmpty(): a layer that IS stored and is all-zero is genuinely dark -- a
        // sealed cave -- and must stay dark. Naively "fixing" absence by defaulting every
        // missing-or-empty sky layer to 15 would light every cave in the world.
        ILightingSupplier supplier = VoxelIngestService.lightingSupplier(new DataLayer(), new DataLayer());

        assertEquals(0, skyOf(supplier.supply(8, 8, 8)));
        assertEquals(0, blockOf(supplier.supply(8, 8, 8)));
    }

    @Test
    void absentBlockLayerReadsAsUnlitEvenUnderAFullSky() {
        DataLayer sky = new DataLayer();
        sky.set(1, 2, 3, 11);
        sky.set(15, 15, 15, 0);

        ILightingSupplier supplier = VoxelIngestService.lightingSupplier(null, sky);

        assertEquals(0, blockOf(supplier.supply(1, 2, 3)));
        assertEquals(11, skyOf(supplier.supply(1, 2, 3)));
        assertEquals(0, skyOf(supplier.supply(15, 15, 15)));
    }

    @Test
    void bothLayersPresentAreSampledVerbatim() {
        DataLayer block = new DataLayer();
        DataLayer sky = new DataLayer();
        for (int i = 0; i < 16; i++) {
            block.set(i, i, i, i);
            sky.set(i, i, i, 15 - i);
        }

        ILightingSupplier supplier = VoxelIngestService.lightingSupplier(block, sky);

        for (int i = 0; i < 16; i++) {
            byte light = supplier.supply(i, i, i);
            assertEquals(i, blockOf(light), "block nibble at " + i);
            assertEquals(15 - i, skyOf(light), "sky nibble at " + i);
        }
    }

    @Test
    void lightIsPackedSkyLowBlockHigh() {
        // The packing is load-bearing: quads.frag's uint2vec4RGBA(interData.y) and the lightmap
        // axes both depend on it. getLighting in bindings.glsl decodes (i2>>4) as the lightmap U
        // coordinate and (i2&0xF) as V, and MC's lightmap has block light on U and sky on V --
        // so block must be the high nibble. Swapping them would silently shade every face with
        // the wrong axis and look almost plausible.
        assertNotEquals(0, 15 & 0x0F, "sanity: the low nibble is addressable at all");

        DataLayer block = filled(15);
        DataLayer sky = new DataLayer();
        byte light = VoxelIngestService.lightingSupplier(block, sky).supply(0, 0, 0);

        assertEquals(15, blockOf(light), "a full block layer must land in the high nibble");
        assertEquals(0, skyOf(light), "an all-zero sky layer must land in the low nibble");
    }

    @Test
    void unlitChunkReadsAsFullySkyLitRatherThanAsStoredZero() {
        // The other half of the same question. "A stored all-zero layer is genuinely dark" is only
        // true of a chunk Minecraft has finished lighting; a chunk that is not light-correct has had
        // no light computed for it, and its layers read zero because there is nothing there yet.
        // Baking those zeros is permanent -- the LOD has no notion of light arriving later -- and it
        // draws the terrain as a black mass with a correct silhouette.
        DataLayer block = new DataLayer();
        DataLayer sky = new DataLayer();
        ILightingSupplier supplier = VoxelIngestService.lightingSupplier(block, sky, false);

        for (int y = 0; y < 16; y += 5) {
            for (int z = 0; z < 16; z += 5) {
                for (int x = 0; x < 16; x += 5) {
                    byte light = supplier.supply(x, y, z);
                    assertEquals(15, skyOf(light), "an unlit chunk must not bake a stored zero as night");
                    assertEquals(0, blockOf(light), "and must not invent block light either");
                }
            }
        }
    }

    @Test
    void unlitChunkIgnoresEvenAPopulatedLayer() {
        // Not just the all-zero layer: an unlit chunk's layers are not trustworthy at all, so a
        // stray nonzero value in one must not be believed either. Otherwise a half-written layer
        // would bake a half-lit chunk, which is the harder failure to notice.
        DataLayer block = filled(15);
        DataLayer sky = filled(12);
        ILightingSupplier supplier = VoxelIngestService.lightingSupplier(block, sky, false);

        byte light = supplier.supply(8, 8, 8);
        assertEquals(15, skyOf(light));
        assertEquals(0, blockOf(light));
    }

    @Test
    void litChunkStillReadsItsStoredLayersVerbatim() {
        // The gate must key on isLightCorrect() alone and not drift into "always 15": a lit chunk
        // with a genuinely dark stored layer -- a sealed cave -- has to stay dark, which is the
        // assertion storedAllZeroSkyLayerStaysDark makes through the two-argument overload.
        DataLayer block = new DataLayer();
        DataLayer sky = new DataLayer();
        ILightingSupplier lit = VoxelIngestService.lightingSupplier(block, sky, true);
        ILightingSupplier legacy = VoxelIngestService.lightingSupplier(block, sky);

        for (int i = 0; i < 16; i++) {
            assertEquals(legacy.supply(i, i, i), lit.supply(i, i, i),
                    "the third argument must not change what a lit chunk reads");
        }
    }

    @Test
    void sampledValuesAreClampedIntoTheirNibble() {
        // Defensive: getLight() is documented to return 0..15, but a nibble that overflowed its
        // half would corrupt the other channel rather than merely being too bright.
        DataLayer block = filled(15);
        DataLayer sky = filled(15);
        byte light = VoxelIngestService.lightingSupplier(block, sky).supply(7, 7, 7);

        assertEquals(15, blockOf(light));
        assertEquals(15, skyOf(light));
        assertEquals((byte) 0xFF, light, "both channels saturated is the brightest byte");
    }
}
