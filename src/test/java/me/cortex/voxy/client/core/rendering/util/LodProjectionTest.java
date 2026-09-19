package me.cortex.voxy.client.core.rendering.util;

import org.joml.Matrix4f;
import org.joml.Matrix4fc;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The z row is where this went wrong, so that is what these assert on.
 *
 * <p>A perspective matrix's z row must stay O(1) relative to its w row. When it does not, x/y and w
 * still look right — the geometry is simply projected to a depth outside the clip volume, which on
 * Metal means every fragment is clipped: no pixels, no error, and every draw counter healthy.
 */
class LodProjectionTest {

    private static Matrix4f glPerspective(final float near, final float far) {
        return new Matrix4f().setPerspective(1.2f, 16f / 9f, near, far);
    }

    /** The same projection as Metal's frame uses: z ∈ [0,1], near = 1, far = 0. */
    private static Matrix4f reverseZPerspective(final float near, final float far) {
        return MetalMvpUtil.reverseZRemapOf(glPerspective(near, far));
    }

    /**
     * The construction this replaces: {@code base · P(0.05, rd*16)⁻¹ · P(nearVoxy, 48000)}. Exact
     * when base is the GL-convention matrix it assumes — which is why it survived on the GL
     * backend — and wrong by the ratio of the two near planes when base is reverse-Z.
     */
    private static Matrix4f byCancelling(final Matrix4fc base, final float nearVoxy, final float farVoxy) {
        return new Matrix4f(base)
                .mulLocal(glPerspective(0.05f, 8 * 16f).invert(), new Matrix4f())
                .mulLocal(glPerspective(nearVoxy, farVoxy));
    }

    /** z_row_scale = |m22| / |m23|: ~1 for a real perspective, ~320 for the broken one. */
    private static float zRowScale(final Matrix4f m) {
        return Math.abs(m.m22()) / Math.abs(m.m23());
    }

    @Test
    void buildingFromTheFovIsStableInBothDepthConventions() {
        final Matrix4f fromGl = LodProjection.compute(glPerspective(0.05f, 128f), 16f, 48000f);
        final Matrix4f fromReverseZ = LodProjection.compute(reverseZPerspective(0.05f, 128f), 16f, 48000f);

        assertEquals(1.0f, zRowScale(fromGl), 0.05f, "GL-convention base");
        assertEquals(1.0f, zRowScale(fromReverseZ), 0.05f, "reverse-Z base");
    }

    /**
     * This is the defect, reproduced: cancelling a reverse-Z base leaves the z row ~320x too
     * steep, which is {@code (far0 - near0)/near0} for the 0.05/16 near planes in play. The
     * assertion is deliberately on the magnitude rather than an exact value — what matters is that
     * it is two orders of magnitude away from 1, not the third decimal place.
     */
    @Test
    void cancellingAReverseZBaseBlowsUpTheZRow() {
        final Matrix4f broken = byCancelling(reverseZPerspective(0.05f, 128f), 16f, 48000f);
        assertTrue(zRowScale(broken) > 100f,
                "expected the z row to be ~320x too steep, got " + zRowScale(broken));

        // ...and x/y are untouched, which is exactly why this hides: the frame still looks sane.
        final Matrix4f good = LodProjection.compute(reverseZPerspective(0.05f, 128f), 16f, 48000f);
        assertEquals(good.m00(), broken.m00(), 1e-3f);
        assertEquals(good.m11(), broken.m11(), 1e-3f);
    }

    /** The two near-plane presets, and the Sodium-off override. */
    @Test
    void nearPlaneTracksTheVanillaRenderDistance() {
        assertEquals(16f, LodProjection.nearVoxy(8, false));    // 128 blocks of vanilla terrain
        assertEquals(8f, LodProjection.nearVoxy(2, false));     // 32 blocks -- 16 would clip it
        assertEquals(0.1f, LodProjection.nearVoxy(8, true));    // no vanilla handover at all
    }

    /** A degenerate base must not poison the frame with a NaN aspect. */
    @Test
    void aDegenerateBaseFallsBackToASanePerspective() {
        final Matrix4f out = LodProjection.compute(new Matrix4f(), 16f, 48000f);
        assertTrue(Float.isFinite(out.m00()) && out.m00() != 0f);
        assertEquals(1.0f, zRowScale(out), 0.05f);
    }
}
