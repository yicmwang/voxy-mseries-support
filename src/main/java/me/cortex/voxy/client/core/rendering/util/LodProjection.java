package me.cortex.voxy.client.core.rendering.util;

import org.joml.Matrix4f;
import org.joml.Matrix4fc;

/**
 * Builds the projection Voxy's LOD pass runs on: the camera's own field of view and aspect, but
 * with Voxy's near and far planes.
 *
 * <p>Voxy needs its own near/far because vanilla's near plane is far too close for geometry that
 * starts where the loaded chunks end — at a short render distance the LOD ring begins well beyond
 * 0.05 blocks, and a near plane that close wrecks depth precision across the whole ring.
 *
 * <p><b>Why this is not built by cancelling vanilla's projection.</b> The obvious construction is
 * {@code base · P(near0, far0)⁻¹ · P(nearVoxy, farVoxy)}, which replaces the projection's depth
 * range while leaving its x/y rows exactly as vanilla had them. It is exact — but only when
 * {@code base} really is {@code P(near0, far0)} in the same depth convention, because the
 * cancellation of the z row depends on both matrices agreeing about where z = ±1 sits. Under
 * whole-frame Metal, vanilla's matrix is REVERSE-Z (near = 1, far = 0, greater-equal), so the two
 * z rows do not cancel: the composition leaves the z row scaled by roughly
 * {@code (far0 - near0) / near0} — about 320x at Voxy's 0.05/16 near planes — and every LOD vertex
 * lands far outside Metal's z ∈ [0, w] clip volume. The x/y rows and w survive untouched, so the
 * frame still looks plausible and every draw counter still reads healthy.
 *
 * <p>Reading the field of view and aspect back off {@code base} instead sidesteps the whole
 * question: the x/y rows of a perspective matrix are the same in every depth convention, so
 * recovering fov and aspect from them is convention-independent. The result is a GL-convention
 * matrix (z ∈ [-1, 1]) which the caller then remaps for the target frame — see
 * {@link MetalMvpUtil#applyReverseZRemap}, which maps it to near = 1, far = 0.
 */
public final class LodProjection {

    private LodProjection() {}

    /**
     * @param base     vanilla's projection for this frame, in whatever depth convention the
     *                 backend uses. Only its x/y rows are read.
     * @param nearVoxy Voxy's near plane, in blocks
     * @param farVoxy  Voxy's far plane, in blocks
     */
    public static Matrix4f compute(final Matrix4fc base, final float nearVoxy, final float farVoxy) {
        // A perspective matrix's x/y rows are (f/aspect, f) in every convention, so f (the cotangent
        // of half the vertical fov) and the aspect ratio both fall straight out. Abs because the
        // handedness flip some backends apply to x is not part of the projection's shape.
        final float f = Math.abs(base.m11());
        if (!(f > 1.0e-6f)) {
            // Degenerate (startup/transitional) matrix -- fall back to a sane perspective rather
            // than propagating a NaN aspect through the whole frame.
            return new Matrix4f().setPerspective(1.2f, 16f / 9f, nearVoxy, farVoxy);
        }
        final float aspect = f / Math.abs(base.m00());
        final float fovY = (float) (2.0 * Math.atan(1.0 / f));
        return new Matrix4f().setPerspective(fovY, aspect, nearVoxy, farVoxy);
    }

    /**
     * Voxy's near plane for a given vanilla render distance. Below 32 blocks of vanilla render
     * distance a 16-block near plane would clip geometry the player can see, so it drops to 8.
     *
     * @param disableSodiumChunkRender when Sodium's chunk renderer is off there is no vanilla
     *                                 terrain to hand over from, so the near plane must stay
     *                                 small enough to render close geometry.
     */
    public static float nearVoxy(final int renderDistanceChunks, final boolean disableSodiumChunkRender) {
        final float near = (renderDistanceChunks * 16) <= 32 ? 8f : 16f;
        return disableSodiumChunkRender ? 0.1f : near;
    }
}
