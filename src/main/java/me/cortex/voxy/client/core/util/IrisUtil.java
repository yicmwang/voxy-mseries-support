package me.cortex.voxy.client.core.util;

import me.cortex.voxy.client.core.VoxyRenderSystem;
import me.cortex.voxy.client.core.rendering.Viewport;
import net.caffeinemc.mods.sodium.client.render.chunk.ChunkRenderMatrices;
import net.caffeinemc.mods.sodium.client.util.FogParameters;

/**
 * Iris support stub.
 *
 * <p>Iris was dropped in P2. It is GL-based by construction — it compiles GLSL into GL programs and
 * renders through the GL pipeline — and whole-frame Metal has no GL context, so there was no path
 * for it to work. Its entire integration surface (the pack contract, the gbuffer injector, the vx
 * material g-buffer planes, the shadow-pass guards, the sampler save/restore) has been removed.
 *
 * <p>The type and method names are kept deliberately: call sites are written as
 * {@code if (IrisUtil.irisShaderPackEnabled()) { ...iris path... } else { ...normal path... }}, so a
 * constant-false stub makes every one of them take its non-Iris branch with no edit, and the
 * remaining guards stay readable rather than being silently deleted.
 *
 * <p>Do not reintroduce an Iris path here without a Metal-native story for shader packs.
 */
public final class IrisUtil {
    /** Retained for the (now removed) Iris viewport capture path. */
    public record CapturedViewportParameters(ChunkRenderMatrices matrices, FogParameters parameters,
                                             double x, double y, double z) {
        public Viewport<?> apply(VoxyRenderSystem vrs) {
            return vrs.setupViewport(this.matrices, this.parameters, this.x, this.y, this.z);
        }
    }

    public static final boolean IRIS_INSTALLED = false;
    public static final boolean SHADER_SUPPORT = false;

    private IrisUtil() {
    }

    public static boolean irisShadowActive() {
        return false;
    }

    /** Was true while Iris rendered its shadow map (which re-enters Sodium's SOLID pass). */
    public static boolean shadowsBeingRendered() {
        return false;
    }

    public static void clearIrisSamplers() {
    }

    public static boolean irisShaderPackEnabled() {
        return false;
    }

    /** Was true when the LOD bridge should be injected into the pack's gbuffer instead of MC's RT. */
    public static boolean irisGbufferInjectMode() {
        return false;
    }

    /** Was true when the active pack shipped a voxy.json contract. */
    public static boolean vxContractActive() {
        return false;
    }

    public static void disableIrisShaders() {
    }
}
