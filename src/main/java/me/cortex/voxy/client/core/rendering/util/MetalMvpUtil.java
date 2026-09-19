package me.cortex.voxy.client.core.rendering.util;

import me.cortex.voxy.common.Logger;
import org.joml.Matrix4f;
import org.joml.Matrix4fc;

/**
 * Shared GL→Metal clip-space depth remap for raster passes that consume a
 * GL-convention MVP (built from {@code setPerspective} without
 * {@code zZeroToOne} in VoxyRenderSystem.makeProjectionMatrix).
 *
 * Extracted from MDICSectionRenderer's 2026-05-26 experiment
 * ({@code VOXY_LOD_METAL_NDC=1}) so every Metal pass shares ONE depth
 * convention — the LOD terrain pass (quads.frag's {@code gl_FragCoord.z})
 * and the chunk-bound depth mask (ChunkBoundRenderer/outline.vsh) compare
 * depths against each other, so they must remap identically (both on, or
 * both off). On Metal — whose visible clip volume is z ∈ [0, w] — the near
 * half of the GL clip range lands at NDC z &lt; 0 and is clipped by the
 * rasterizer; the remap maps the whole GL range into Metal's [0,1] so
 * nothing in the GL frustum is clipped, and depth ordering is preserved
 * (the remap is monotonic). OFF by default because it shifts every depth
 * value and needs visual confirmation — same class as the bakery m22 fix.
 */
public final class MetalMvpUtil {
    /** Opt-in flag for the remap ({@code VOXY_LOD_METAL_NDC=1}). Callers gate on this AND a non-GL backend. */
    public static final boolean METAL_NDC_REMAP = "1".equals(System.getenv("VOXY_LOD_METAL_NDC"));

    /**
     * Opt-in reverse-Z remap ({@code VOXY_LOD_REVERSE_Z=1}).
     *
     * <p>Metallum's frame depth buffer is REVERSE-Z: near maps to 1, far to 0, and the pass's
     * compare function is GreaterEqual. The P0 probe reads this straight off the live pass and
     * reports {@code compare=GreaterEqual reverseZ=true}.
     *
     * <p>Voxy's LOD pipeline instead used {@link me.cortex.voxy.client.core.gpu.PipelineState.DepthState#DEFAULT}
     * (LessEqual) and a GL-convention projection. Against a reverse-Z buffer that combination
     * rejects every fragment: over sky the buffer holds 0 (far) while a Voxy fragment's depth is a
     * large positive number in its own convention, so {@code fragDepth <= 0} is false everywhere.
     * The draws execute, every fragment is discarded, and the LOD contributes no pixels at all.
     *
     * <p>Fixing it needs both halves: the depth VALUES must be in reverse-Z space (this remap,
     * {@code z' = 0.5 - 0.5*z}, mapping NDC z -1..1 to 1..0) AND the compare must be GreaterEqual.
     * The pre-existing {@link #applyNdcRemap} maps to normal Z (near 0, far 1) and so cannot be
     * used for this.
     */
    /**
     * DEFAULT ON for non-GL backends. Metallum's frame is reverse-Z, so the GL convention is simply
     * wrong here, not merely different -- there is no configuration in which LessEqual against a
     * reverse-Z buffer renders correctly. Callers still gate on a non-GL backend, so GL is
     * unaffected. {@code VOXY_LOD_REVERSE_Z=0} forces it off (useful for a controlled A/B).
     */
    public static final boolean REVERSE_Z_REMAP = !"0".equals(System.getenv("VOXY_LOD_REVERSE_Z"));
    private static boolean reverseZLogged = false;

    /**
     * metalMVP = ZremapRev . mat, mapping NDC z [-1,1] to [1,0] -- near to 1, far to 0, i.e. the
     * reverse-Z convention Metallum's frame buffer already uses. Row form:
     * {@code z' = -0.5*z + 0.5*w}, {@code w' = w}. X/Y untouched. Mutates {@code mat} in place.
     */
    public static void applyReverseZRemap(Matrix4f mat) {
        reverseZRemapOf(mat).get(mat);
        if (!reverseZLogged) {
            reverseZLogged = true;
            Logger.info("[Metal] VOXY_LOD_REVERSE_Z active: render MVP remapped to reverse-Z "
                    + "(near=1, far=0); depth compare must be GreaterEqual");
        }
    }
    /** Non-mutating form of {@link #applyReverseZRemap}. */
    public static Matrix4f reverseZRemapOf(Matrix4fc mat) {
        return new Matrix4f(
                1, 0, 0,     0,
                0, 1, 0,     0,
                0, 0, -0.5f, 0,
                0, 0, 0.5f,  1).mul(mat, new Matrix4f());
    }

    private static boolean logged = false;

    private MetalMvpUtil() {}

    /**
     * metalMVP = Zremap · mat, where Zremap maps NDC z [-1,1] → [0,1].
     * Column-major args (mColRow): m22 = 0.5, m32 = 0.5 give the row
     * form  z' = 0.5·z + 0.5·w,  w' = w. X/Y are untouched so on-screen
     * position is identical; only the clipped/written depth changes.
     * Mutates {@code mat} in place.
     */
    public static void applyNdcRemap(Matrix4f mat) {
        new Matrix4f(
                1, 0, 0,    0,
                0, 1, 0,    0,
                0, 0, 0.5f, 0,
                0, 0, 0.5f, 1).mul(mat, mat);
        if (!logged) {
            logged = true;
            Logger.info("[Metal] VOXY_LOD_METAL_NDC active: render MVP remapped to [0,1] NDC-z");
        }
    }
}
