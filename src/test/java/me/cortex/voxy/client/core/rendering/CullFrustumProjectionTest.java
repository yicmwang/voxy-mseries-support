package me.cortex.voxy.client.core.rendering;

import org.joml.FrustumIntersection;
import org.joml.Matrix4f;
import org.joml.Vector4f;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Field;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Is the LOD cull frustum's far plane Voxy's render distance, or the vanilla one?
 *
 * <p>Symptom under test: with a low vanilla render distance, LOD terrain is culled beyond roughly
 * that same distance — a hard edge near the player, and ground that vanishes when you fly higher
 * (it falls outside the plane). Reported on hardware at RD 4, and less pronounced at RD 32.
 *
 * <p>Mechanism: {@code VoxyRenderSystem.computeProjectionMat} does not return a geometric
 * projection. It returns {@code base x inverse(vanillaProj) x voxyProj} — a DEPTH REMAP, so that
 * Voxy's depth values are comparable with vanilla's. That is correct for depth. But
 * {@code Viewport.update()} hands the same matrix to joml's {@link FrustumIntersection}, which
 * derives the near and far planes from the matrix's z and w rows. After a depth remap those rows no
 * longer describe Voxy's view volume, so the extracted far plane can land near the VANILLA render
 * distance instead of Voxy's.
 *
 * <p>This test asserts the far plane is where Voxy's projection puts it. <b>If it fails, the
 * mechanism above is confirmed</b> — the failure message prints both distances. If it passes, the
 * cull is fine and the vanishing geometry is happening somewhere else (the rasterizer clipping
 * Metal's z<0 half, per {@code MetalMvpUtil}, is the other candidate).
 *
 * <p>Pure arithmetic on joml — no game, no GPU, no window.
 */
class CullFrustumProjectionTest {

    private static final float FOV_DEG = 70f;
    private static final float ASPECT = 2940f / 1846f;      // the size the game actually ran at
    private static final float VANILLA_NEAR = 0.05f;
    private static final float VOXY_FAR = 16f * 3000f;      // makeProjectionMatrix(nearVoxy, 16*3000)

    private static Vector4f[] planesOf(Matrix4f m) throws Exception {
        FrustumIntersection fi = new FrustumIntersection();
        fi.set(m, false);
        Field f = FrustumIntersection.class.getDeclaredField("planes");
        f.setAccessible(true);
        return (Vector4f[]) f.get(fi);
    }

    /** Distance along the camera's forward axis (-Z) at which this plane crosses it. */
    private static float axisDistance(Vector4f plane) {
        float len = (float) Math.sqrt(plane.x * plane.x + plane.y * plane.y + plane.z * plane.z);
        if (len == 0f || plane.z == 0f) {
            return Float.NaN;
        }
        return (plane.w / len) / (plane.z / len);
    }

    private static Matrix4f perspective(float near, float far) {
        return new Matrix4f().setPerspective((float) Math.toRadians(FOV_DEG), ASPECT, near, far);
    }

    /** Verbatim transcription of VoxyRenderSystem.computeProjectionMat for the given vanilla RD. */
    private static Matrix4f voxyComposedProjection(int vanillaRenderDistanceChunks) {
        float vanillaFar = vanillaRenderDistanceChunks * 16f;
        float nearVoxy = vanillaFar <= 32.0f ? 8f : 16f;     // VoxyClient.disableSodiumChunkRender() is false
        Matrix4f base = perspective(VANILLA_NEAR, vanillaFar);
        return base.mulLocal(perspective(VANILLA_NEAR, vanillaFar).invert(), new Matrix4f())
                .mulLocal(perspective(nearVoxy, VOXY_FAR));
    }

    private static void report(String label, Matrix4f proj) throws Exception {
        Vector4f[] p = planesOf(new Matrix4f(proj));         // identity model view: camera at origin
        System.out.printf("  %-28s near(plane 4)=%10.1f   far(plane 5)=%12.1f%n",
                label, axisDistance(p[4]), axisDistance(p[5]));
    }

    @Test
    void cullFrustumFarPlaneMatchesVoxysRenderDistance() throws Exception {
        System.out.println("\n=== cull frustum far plane, by vanilla render distance ===");
        System.out.println("  (Voxy's own far plane is " + VOXY_FAR + " blocks)");

        for (int rd : new int[]{4, 8, 32}) {
            System.out.println("-- vanilla render distance " + rd + " chunks ("
                    + (rd * 16) + " blocks)");
            report("geometric (Voxy near/far)", perspective(rd * 16f <= 32 ? 8f : 16f, VOXY_FAR));
            report("composed (what Voxy culls with)", voxyComposedProjection(rd));
        }

        // The cull must see Voxy's render distance, not vanilla's.
        float farAtRd4 = axisDistance(planesOf(voxyComposedProjection(4))[5]);
        System.out.printf("%n  => at RD 4 the cull far plane is %.1f blocks; Voxy's is %.0f%n",
                farAtRd4, VOXY_FAR);

        // Second question, same harness: what NDC z does the composed projection produce?
        // Metal's visible clip volume is z in [0, w] -- anything with NDC z < 0 is CLIPPED by the
        // rasterizer, and MetalMvpUtil's remap (which fixes exactly that) is OFF by default. So the
        // distance at which NDC z crosses zero is a hard visibility edge, independent of the cull.
        System.out.println("\n=== NDC z vs distance through the composed projection ===");
        System.out.println("  (Metal clips NDC z < 0)");
        Matrix4f proj = voxyComposedProjection(4);
        for (float d : new float[]{8f, 16f, 24f, 32f, 48f, 64f, 128f, 512f, 4096f, 48000f}) {
            org.joml.Vector4f clip = proj.transform(new org.joml.Vector4f(0f, 0f, -d, 1f));
            float ndcZ = clip.z / clip.w;
            System.out.printf("  %8.0f blocks -> NDC z = %+9.4f   %s%n",
                    d, ndcZ, ndcZ < 0f ? "<-- CLIPPED BY METAL" : "");
        }

        assertTrue(farAtRd4 > VOXY_FAR * 0.5f,
                "CULL FRUSTUM FAR PLANE IS THE VANILLA RENDER DISTANCE, NOT VOXY'S.\n"
                        + "  At RD 4 (64 blocks) the cull frustum's far plane is " + farAtRd4 + " blocks.\n"
                        + "  Voxy's projection puts it at " + VOXY_FAR + ".\n"
                        + "  The depth remap in computeProjectionMat has replaced the z row, and joml\n"
                        + "  extracts near/far from it -- so the cull discards everything past the\n"
                        + "  vanilla render distance. That is the hard edge and the vanishing ground.");
    }
}
