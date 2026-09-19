package me.cortex.voxy.client.core.rendering.util;

import org.joml.Matrix4f;
import org.joml.Vector4f;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Metallum's frame depth buffer is REVERSE-Z: near maps to 1, far to 0, and the live pass compares
 * GreaterEqual (the restored P0 probe reads that straight off the pass:
 * {@code compare=GreaterEqual reverseZ=true}).
 *
 * <p>Voxy's LOD pipeline used {@code DepthState.DEFAULT} = LessEqual with GL-convention depths.
 * Against a reverse-Z buffer that rejects every fragment -- over sky the buffer holds 0 (far) while
 * a Voxy fragment's depth is a large positive number in its own convention, so {@code fragDepth <= 0}
 * is false everywhere. The LOD drew nothing at all, which is why the bug survived every other
 * diagnostic: the draws, the commands, the geometry and the target were all correct, and all of
 * that sits upstream of the depth test.
 *
 * <p>Fixing it needs BOTH halves, and this pins the half that is pure arithmetic: the depth VALUES
 * must be remapped into reverse-Z space. The compare-op half is chosen alongside it in
 * MDICSectionRenderer.
 *
 * <p>Pure joml -- no GPU, no game, no window. Precedent: {@code CullFrustumProjectionTest}.
 */
class ReverseZRemapTest {

    private static final float NEAR = 16f;
    private static final float FAR = 16f * 3000f;
    private static final float FOV_Y = (float) Math.toRadians(70);
    private static final float ASPECT = 16f / 9f;

    /** Verbatim shape of VoxyRenderSystem.makeProjectionMatrix: a GL-convention perspective. */
    private static Matrix4f glProjection() {
        return new Matrix4f().setPerspective(FOV_Y, ASPECT, NEAR, FAR);
    }

    /** NDC z for a point {@code distance} blocks in front of the camera, through {@code proj}. */
    private static float ndcZ(final Matrix4f proj, final float distance) {
        Vector4f clip = proj.transform(new Vector4f(0f, 0f, -distance, 1f));
        return clip.z / clip.w;
    }

    @Test
    void glConventionPutsNearAtMinusOneAndFarAtPlusOne() {
        Matrix4f proj = glProjection();
        assertEquals(-1f, ndcZ(proj, NEAR), 1e-3f, "GL convention: near plane is NDC z = -1");
        assertEquals(1f, ndcZ(proj, FAR), 1e-3f, "GL convention: far plane is NDC z = +1");
    }

    @Test
    void reverseZRemapPutsNearAtOneAndFarAtZero() {
        Matrix4f proj = glProjection();
        MetalMvpUtil.applyReverseZRemap(proj);
        assertEquals(1f, ndcZ(proj, NEAR), 1e-3f,
                "reverse-Z: the near plane must map to depth 1 (what Metallum's buffer means by near)");
        assertEquals(0f, ndcZ(proj, FAR), 1e-3f,
                "reverse-Z: the far plane must map to depth 0 -- this is what makes a LessEqual "
                        + "compare reject every LOD fragment over sky");
    }

    @Test
    void reverseZRemapKeepsDepthOrdering() {
        // Nearer must mean larger depth, which is what a GreaterEqual compare relies on.
        Matrix4f proj = glProjection();
        MetalMvpUtil.applyReverseZRemap(proj);
        float previous = Float.MAX_VALUE;
        for (float d : new float[]{32f, 64f, 128f, 512f, 4096f, FAR}) {
            float z = ndcZ(proj, d);
            assertTrue(z < previous, "depth must decrease with distance (d=" + d + ", z=" + z + ")");
            previous = z;
        }
    }

    @Test
    void everyPointInsideTheFrustumLandsInsideMetalsClipVolume() {
        // Metal's visible clip volume is NDC z in [0, w] -- anything below 0 is clipped by the
        // rasterizer. Only points at or beyond the near plane are in the frustum; anything closer
        // is legitimately clipped and must NOT be asserted on (d=8 with near=16 sits behind the
        // near plane, and an earlier version of this test wrongly demanded it survive).
        Matrix4f proj = glProjection();
        MetalMvpUtil.applyReverseZRemap(proj);
        for (float d : new float[]{NEAR, 24f, 32f, 64f, 128f, 512f, 4096f, FAR}) {
            float z = ndcZ(proj, d);
            assertTrue(z >= -1e-4f && z <= 1f + 1e-4f,
                    "NDC z must be inside Metal's [0,1] for d=" + d + " but was " + z);
        }
    }

    @Test
    void fixesTheNearClipThatTheGlConventionCaused() {
        // The concrete defect this remap also repairs, and the reason "LOD vanishes near the
        // camera" appeared alongside everything else. Under the GL convention the near HALF of the
        // depth range is negative, and Metal clips negative NDC z outright -- so LOD terrain inside
        // the frustum but closer than roughly 2x the near plane was rasterizer-clipped, not
        // depth-rejected. 24 blocks is inside a 16-block near plane, and is the smallest sample
        // that the CullFrustumProjectionTest showed as negative.
        Matrix4f gl = glProjection();
        assertTrue(ndcZ(gl, 24f) < 0f,
                "precondition: the GL convention puts 24 blocks at negative NDC z, so Metal clips it");

        Matrix4f remapped = glProjection();
        MetalMvpUtil.applyReverseZRemap(remapped);
        assertTrue(ndcZ(remapped, 24f) >= 0f,
                "the remap must bring in-frustum geometry back inside Metal's clip volume");
    }

    @Test
    void reverseZRemapLeavesScreenPositionUntouched() {
        // Only the z row may change. If x or y moved, the LOD would be drawn in the wrong place --
        // which would look like a completely different bug.
        Matrix4f plain = glProjection();
        Matrix4f remapped = glProjection();
        MetalMvpUtil.applyReverseZRemap(remapped);
        for (float[] p : new float[][]{{0f, 0f, -100f}, {37f, -12f, -250f}, {-400f, 90f, -800f}}) {
            Vector4f a = plain.transform(new Vector4f(p[0], p[1], p[2], 1f));
            Vector4f b = remapped.transform(new Vector4f(p[0], p[1], p[2], 1f));
            assertEquals(a.x, b.x, 1e-4f, "x must be untouched by the remap");
            assertEquals(a.y, b.y, 1e-4f, "y must be untouched by the remap");
            assertEquals(a.w, b.w, 1e-4f, "w must be untouched by the remap");
        }
    }

    @Test
    void normalZRemapIsTheWrongToolForReverseZ() {
        // Pins WHY the pre-existing remap cannot be reused: it is the normal-Z mapping, the exact
        // opposite of what a reverse-Z buffer needs. Reaching for it here would silently invert
        // every depth comparison.
        Matrix4f proj = glProjection();
        MetalMvpUtil.applyNdcRemap(proj);
        assertEquals(0f, ndcZ(proj, NEAR), 1e-3f, "applyNdcRemap maps near to 0 -- normal Z");
        assertEquals(1f, ndcZ(proj, FAR), 1e-3f, "applyNdcRemap maps far to 1 -- normal Z");
    }

    @Test
    void composingBothRemapsIsNotIdentity() {
        // Not an inverse pair, and worth pinning because it is the kind of thing one assumes.
        // Each maps the same [-1,1] range onto a different unit interval, so composing them gives
        // z' = 0.75 - 0.25*z (near->1, far->0.5), NOT the original. Applying both by accident --
        // e.g. enabling VOXY_LOD_METAL_NDC and VOXY_LOD_REVERSE_Z together -- therefore does not
        // cancel out; the call sites are mutually exclusive for that reason.
        Matrix4f proj = glProjection();
        MetalMvpUtil.applyReverseZRemap(proj);
        MetalMvpUtil.applyNdcRemap(proj);
        assertEquals(1f, ndcZ(proj, NEAR), 1e-3f, "near ends at 1, not back at -1");
        assertEquals(0.5f, ndcZ(proj, FAR), 1e-3f, "far ends at 0.5, not back at +1");
    }
}
