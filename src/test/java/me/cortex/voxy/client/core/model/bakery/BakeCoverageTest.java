package me.cortex.voxy.client.core.model.bakery;

import org.junit.jupiter.api.Test;
import org.lwjgl.system.MemoryUtil;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Pins the one thing about the bake readback that the LOD correctness rests on: the marker the
 * metadata is derived from describes what the model DREW, not what dilation filled in.
 *
 * <p>This is a regression test for the missing-surface-face class of bug. {@code occludesFace},
 * {@code faceCoversFullBlock} and the packed quad bounds all come from the marker, and the mesher
 * culls a solid block's face wherever its neighbour claims to occlude. Dilation fills every empty
 * texel of any cell that had one opaque pixel, so taking the marker from post-dilation data made
 * every non-cube model claim full coverage — a 4-quad plant then occluded like a stone cube, and the
 * faces it culled left holes in the terrain around it. No GPU is needed to catch that: it is a
 * property of the pack, so it is tested as one.
 */
class BakeCoverageTest {
    private static final int CELL_W = 16;
    private static final int CELL_H = 16;
    private static final int TOTAL_W = CELL_W * 3;
    private static final int CELL_PIXELS = CELL_W * CELL_H;

    private static final int OPAQUE_RED = 0xFFFF0000;
    private static final int CLEAR = 0x00000000;

    /** A cell grid of clear pixels with the given cells painted fully opaque. */
    private static int[] fullCells(int... cells) {
        int[] rgba = new int[TOTAL_W * CELL_H * 2];
        for (int cell : cells) {
            int cx = (cell % 3) * CELL_W;
            int cy = (cell / 3) * CELL_H;
            for (int y = cy; y < cy + CELL_H; y++) {
                for (int x = cx; x < cx + CELL_W; x++) {
                    rgba[y * TOTAL_W + x] = OPAQUE_RED;
                }
            }
        }
        return rgba;
    }

    /** A cell painted with a diagonal line — a stand-in for a plant's edge-on cross planes. */
    private static void paintDiagonal(int[] rgba, int cell) {
        int cx = (cell % 3) * CELL_W;
        int cy = (cell / 3) * CELL_H;
        for (int i = 0; i < CELL_W; i++) {
            rgba[(cy + i) * TOTAL_W + cx + i] = OPAQUE_RED;
        }
    }

    private static int countDrawn(boolean[] drawn, int cell) {
        int cx = (cell % 3) * CELL_W;
        int cy = (cell / 3) * CELL_H;
        int n = 0;
        for (int y = cy; y < cy + CELL_H; y++) {
            for (int x = cx; x < cx + CELL_W; x++) {
                if (drawn[y * TOTAL_W + x]) n++;
            }
        }
        return n;
    }

    @Test
    void dilateLeavesAnEmptyCellEmpty() {
        // An empty face must stay empty: faceExists()==false here is what makes a plant's up/down
        // faces not exist, and needsDoubleSidedQuads true for the model. A dilation that filled empty
        // cells would turn every plant into a full cube.
        int[] rgba = fullCells(0);
        assertEquals(0, BakeCoverage.dilateOpaqueIntoGaps(rgba, TOTAL_W, CELL_W, CELL_H));
        assertEquals(0, countDrawn(BakeCoverage.snapshotDrawn(rgba), 1));
        assertEquals(0, countDrawn(BakeCoverage.snapshotDrawn(rgba), 2));
    }

    @Test
    void dilateFillsEveryTexelOfATouchedCell() {
        int[] rgba = fullCells(0);
        paintDiagonal(rgba, 1);
        assertEquals(CELL_PIXELS - CELL_W, BakeCoverage.dilateOpaqueIntoGaps(rgba, TOTAL_W, CELL_W, CELL_H));
        boolean[] drawn = BakeCoverage.snapshotDrawn(rgba);
        assertEquals(CELL_PIXELS, countDrawn(drawn, 1));
    }

    @Test
    void dilatePreservesTheCellsAverageAlpha() {
        // Water bakes at alpha ~0.7-0.8; the fill must not turn its gaps opaque, or a translucent
        // block's atlas cell stops matching its metadata.
        int[] rgba = new int[TOTAL_W * CELL_H * 2];
        int water = (200 << 24) | 0x0000FF;
        rgba[0] = water;
        BakeCoverage.dilateOpaqueIntoGaps(rgba, TOTAL_W, CELL_W, CELL_H);
        assertEquals(200, rgba[1] >>> 24);
    }

    @Test
    void markerComesFromPreDilationCoverageNotFromTheDilatedColour() {
        // The regression. Cell 1 is a diagonal line: 16 of 256 pixels drawn. Post-dilation every
        // texel of that cell is nonzero, so a marker taken from the colour word would say 256/256 and
        // the model would occlude like a cube.
        int[] rgba = fullCells(0);
        paintDiagonal(rgba, 1);
        boolean[] drawn = BakeCoverage.snapshotDrawn(rgba);
        BakeCoverage.dilateOpaqueIntoGaps(rgba, TOTAL_W, CELL_W, CELL_H);

        long dest = MemoryUtil.nmemAllocChecked(6L * CELL_PIXELS * 8L);
        try {
            BakeCoverage.pack(dest, rgba, drawn, TOTAL_W, CELL_W, CELL_H);

            assertEquals(CELL_PIXELS, markersInFace(dest, 0), "fully drawn face keeps full coverage");
            assertEquals(CELL_W, markersInFace(dest, 1), "diagonal face keeps its true 16 pixels");
            assertEquals(0, markersInFace(dest, 2), "empty face stays unmarked");

            // ...while the colour word still carries the dilated tile the atlas needs.
            assertFalse(hasTransparentColour(dest, 1), "atlas colour for the diagonal face is filled");
        } finally {
            MemoryUtil.nmemFree(dest);
        }
    }

    @Test
    void packIsFaceMajorNotRasterOrder() {
        // Face N must land at byte offset N*cellPixels*8. Walking the raster image straight through
        // would put rows 0..5 of the whole target into "face 0" -- face 0 actually lives at the
        // 16x16 cell at (0,0). Cell 3 is the first cell of the second row, so it is the one that
        // catches a raster-order pack.
        int[] rgba = fullCells(3);
        boolean[] drawn = BakeCoverage.snapshotDrawn(rgba);
        long dest = MemoryUtil.nmemAllocChecked(6L * CELL_PIXELS * 8L);
        try {
            BakeCoverage.pack(dest, rgba, drawn, TOTAL_W, CELL_W, CELL_H);
            assertEquals(0, markersInFace(dest, 0));
            assertEquals(CELL_PIXELS, markersInFace(dest, 3));
            assertEquals(0, markersInFace(dest, 5));
        } finally {
            MemoryUtil.nmemFree(dest);
        }
    }

    @Test
    void packWritesTheMarkerOnlyForDrawnPixels() {
        int[] rgba = fullCells(0, 1, 2, 3, 4, 5);
        boolean[] drawn = BakeCoverage.snapshotDrawn(rgba);
        long dest = MemoryUtil.nmemAllocChecked(6L * CELL_PIXELS * 8L);
        try {
            BakeCoverage.pack(dest, rgba, drawn, TOTAL_W, CELL_W, CELL_H);
            for (int face = 0; face < 6; face++) {
                assertEquals(CELL_PIXELS, markersInFace(dest, face));
            }
            assertTrue(drawn[0]);
        } finally {
            MemoryUtil.nmemFree(dest);
        }
    }

    private static int markersInFace(long dest, int face) {
        long base = dest + (long) face * CELL_PIXELS * 8L;
        int n = 0;
        for (int i = 0; i < CELL_PIXELS; i++) {
            if ((MemoryUtil.memGetInt(base + (long) i * 8L + 4) & 0xFF) != 0) n++;
        }
        return n;
    }

    private static boolean hasTransparentColour(long dest, int face) {
        long base = dest + (long) face * CELL_PIXELS * 8L;
        for (int i = 0; i < CELL_PIXELS; i++) {
            if ((MemoryUtil.memGetInt(base + (long) i * 8L) & 0xFF000000) == 0) return true;
        }
        return false;
    }
}
