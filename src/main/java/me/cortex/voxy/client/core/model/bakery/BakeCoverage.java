package me.cortex.voxy.client.core.model.bakery;

import org.lwjgl.system.MemoryUtil;

/**
 * Pure packing and coverage logic for the Metal bake readback, split out of {@link MetalViewCapture}
 * so it can be unit-tested without a GPU.
 *
 * <p>The bake target is a 3×2 grid of {@code cellW×cellH} face cells. {@code emitToStream} copies it
 * into a face-major destination — face N at byte offset {@code N*cellW*cellH*8} — as a uvec2 per
 * pixel: a colour word, then a metadata word whose low byte is the "the model drew this pixel"
 * marker.
 *
 * <p>The two consumers of that data want opposite things, which is why this exists as its own unit.
 * The <b>atlas</b> is built from the COLOUR word and needs {@link #dilateOpaqueIntoGaps} to have run:
 * a sparse cell mip-averages to alpha 0, the LOD shader's discard then fires, and the horizon tears.
 * The <b>metadata</b> — {@code ModelQueries}/{@code TextureUtils}: {@code occludesFace},
 * {@code faceCoversFullBlock}, the packed quad bounds and depths — is derived from the MARKER, and
 * must not see the dilation. Dilation fills every empty texel of any cell that had one opaque pixel,
 * so a 30%-covered cell would report 100% coverage, its model would claim to occlude faces it never
 * touches, and the mesher would cull real faces off that model's neighbours. That is a hole in the
 * world exactly where a solid block meets a plant, fence, slab or anything else that is not a cube.
 *
 * <p>Keeping the marker on pre-dilation data is therefore not a refinement of the dilation; it is the
 * separation of the two, and {@code BakeCoverageTest} pins it.
 */
public final class BakeCoverage {
    /** Marker bit written into the metadata word's low byte for a pixel the model drew. */
    public static final int MARKER_WRITTEN = 0x80;

    private BakeCoverage() {
    }

    /**
     * Which pixels the model actually drew, from the readback as it stands. Call this BEFORE
     * {@link #dilateOpaqueIntoGaps}: after it, every texel of any touched cell is nonzero and the
     * answer is "all of them" for every face of every model.
     */
    public static boolean[] snapshotDrawn(int[] rgba) {
        boolean[] drawn = new boolean[rgba.length];
        for (int i = 0; i < rgba.length; i++) {
            drawn[i] = (rgba[i] & 0xFF000000) != 0;
        }
        return drawn;
    }

    /**
     * Fill every transparent pixel in each cell with that cell's average written RGBA, so each face
     * cell becomes a uniform-colour tile.
     *
     * <p>The fill alpha is the cell's average written alpha rather than a hard 0xFF: for SOLID/CUTOUT
     * bakes every written pixel is alpha 255 so the fill is bit-identical to before, but TRANSLUCENT
     * bakes (water ≈ 0.7–0.8 alpha) keep their translucency instead of the gaps turning opaque.
     *
     * <p>Cells with no written pixel at all are left untouched — an empty face must stay empty, which
     * is what makes {@code faceExists()} false for a plant's up/down faces and
     * {@code needsDoubleSidedQuads} true for the model.
     *
     * @return total pixels filled, for diagnostics
     */
    public static int dilateOpaqueIntoGaps(int[] rgba, int totalW, int cellW, int cellH) {
        int totalFilled = 0;
        for (int cellRow = 0; cellRow < 2; cellRow++) {
            for (int cellCol = 0; cellCol < 3; cellCol++) {
                final int x0 = cellCol * cellW;
                final int y0 = cellRow * cellH;
                long rSum = 0, gSum = 0, bSum = 0, aSum = 0;
                int opaqueCount = 0;
                for (int y = y0; y < y0 + cellH; y++) {
                    for (int x = x0; x < x0 + cellW; x++) {
                        int p = rgba[y * totalW + x];
                        if ((p & 0xFF000000) == 0) continue;
                        rSum += (p) & 0xFF;
                        gSum += (p >> 8) & 0xFF;
                        bSum += (p >> 16) & 0xFF;
                        aSum += (p >>> 24);
                        opaqueCount++;
                    }
                }
                if (opaqueCount == 0) continue; // entire cell empty — leave it
                int avgR = (int) (rSum / opaqueCount);
                int avgG = (int) (gSum / opaqueCount);
                int avgB = (int) (bSum / opaqueCount);
                // Written pixels have alpha >= 1, so the fill always survives the downstream
                // "was this pixel written" checks on the colour word.
                int avgA = Math.max(1, (int) (aSum / opaqueCount));
                int fill = (avgA << 24) | (avgB << 16) | (avgG << 8) | avgR;
                for (int y = y0; y < y0 + cellH; y++) {
                    for (int x = x0; x < x0 + cellW; x++) {
                        int idx = y * totalW + x;
                        if ((rgba[idx] & 0xFF000000) != 0) continue;
                        rgba[idx] = fill;
                        totalFilled++;
                    }
                }
            }
        }
        return totalFilled;
    }

    /**
     * Copy the 3×2 cell grid into {@code destAddr} face-major: face N occupies
     * {@code [N*cellW*cellH*8, (N+1)*cellW*cellH*8)} with pixels row-major inside the cell.
     *
     * <p>The colour word comes from {@code rgba} (dilated, so the atlas is opaque) and the marker
     * word from {@code drawn} (pre-dilation, so the metadata is honest). Consumers read face N at
     * bytes {@code [2048*N, 2048*(N+1))} — see {@code ModelFactory.processModelResult}.
     *
     * <p>Walking the raster image straight into {@code destAddr} instead would land rows 0..5 of the
     * whole target into "face 0", which is geometrically wrong: face 0 lives at pixels (0..15, 0..15).
     * The legacy GL bakery had a {@code bufferreorder.comp} compute shader for this rearrangement.
     */
    public static void pack(long destAddr, int[] rgba, boolean[] drawn,
                            int totalW, int cellW, int cellH) {
        for (int face = 0; face < 6; face++) {
            int faceX = face % 3;
            int faceY = face / 3;
            long faceDst = destAddr + (long) face * cellW * cellH * 8L;
            for (int ly = 0; ly < cellH; ly++) {
                int srcY = faceY * cellH + ly;
                long dstRow = faceDst + (long) ly * cellW * 8L;
                for (int lx = 0; lx < cellW; lx++) {
                    int srcIdx = srcY * totalW + faceX * cellW + lx;
                    long dstPx = dstRow + lx * 8L;
                    MemoryUtil.memPutInt(dstPx, rgba[srcIdx]);
                    MemoryUtil.memPutInt(dstPx + 4, drawn[srcIdx] ? MARKER_WRITTEN : 0);
                }
            }
        }
    }
}
