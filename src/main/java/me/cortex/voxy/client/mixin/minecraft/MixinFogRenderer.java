package me.cortex.voxy.client.mixin.minecraft;

import me.cortex.voxy.client.config.VoxyConfig;
import me.cortex.voxy.client.core.IGetVoxyRenderSystem;
import net.minecraft.client.Camera;
import net.minecraft.client.DeltaTracker;
import net.minecraft.client.multiplayer.ClientLevel;
import net.minecraft.client.renderer.fog.FogData;
import net.minecraft.client.renderer.fog.FogRenderer;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

/**
 * Voxy owns the far field's fog, so vanilla's fog range is pushed out of the way.
 *
 * <p>26.2 port: {@code setupFog} now returns {@link FogData} rather than a
 * {@code Vector4f}, and the {@code RenderSystem.getDevice()} call the old injection point
 * anchored on is gone. Injects at RETURN and mutates the returned FogData, matching upstream.
 * Again invisible to javac — mixin resolves injection points at APPLY time.
 */
@Mixin(value = FogRenderer.class, remap = true)
public class MixinFogRenderer {
    @Inject(method = "setupFog", at = @At("RETURN"))
    private void voxy$modifyFog(Camera camera, int renderDistanceInChunks, DeltaTracker deltaTracker,
                                float darkenWorldAmount, ClientLevel level,
                                CallbackInfoReturnable<FogData> cir) {
        if (!(VoxyConfig.CONFIG.enableRendering && VoxyConfig.CONFIG.enabled)) return;

        var vrs = IGetVoxyRenderSystem.getNullable();
        if (vrs == null) return;

        FogData data = cir.getReturnValue();
        if (data == null) return;

        // Diagnostic (VOXY_FOG_TRACE=1): the fog-disable above was observed NOT to take effect --
        // frames captured with Voxy enabled were pixel-identical to frames captured with Voxy
        // disabled (VOXY_FORCE_METAL=0), and with Voxy disabled this mixin returns early, so the
        // fog must have been active in both. Either this injection never runs, or it runs and
        // vanilla ignores the fields it writes. Log both the incoming values and whether we were
        // reached at all, so those two cases are distinguishable.
        final boolean trace = "1".equals(System.getenv("VOXY_FOG_TRACE"));
        if (trace && (FOG_TRACE_COUNT++ % 600) == 0) {
            me.cortex.voxy.common.Logger.info(String.format(java.util.Locale.ROOT,
                    "[FogTrace n=%d] IN  env=[%.1f..%.1f] renderDist=[%.1f..%.1f] skyEnd=%.1f cloudEnd=%.1f",
                    FOG_TRACE_COUNT - 1, data.environmentalStart, data.environmentalEnd,
                    data.renderDistanceStart, data.renderDistanceEnd, data.skyEnd, data.cloudEnd));
        }

        data.renderDistanceStart = 999999999;
        data.renderDistanceEnd = 999999999;
        if (!VoxyConfig.CONFIG.useEnvironmentalFog) {
            data.environmentalStart = 99999999;
            data.environmentalEnd = 99999999;
        }
        // Vanilla still fades sky and clouds on their own ranges; Voxy owns the far field, so
        // push those out too. Without this the horizon keeps a sky-coloured band that reads as
        // the LOD "fading into transparency".
        data.skyEnd = 99999999;
        data.cloudEnd = 99999999;

        if (trace && ((FOG_TRACE_COUNT - 1) % 600) == 0) {
            me.cortex.voxy.common.Logger.info(String.format(java.util.Locale.ROOT,
                    "[FogTrace n=%d] OUT env=[%.1f..%.1f] renderDist=[%.1f..%.1f] skyEnd=%.1f cloudEnd=%.1f",
                    FOG_TRACE_COUNT - 1, data.environmentalStart, data.environmentalEnd,
                    data.renderDistanceStart, data.renderDistanceEnd, data.skyEnd, data.cloudEnd));
        }
    }

    private static long FOG_TRACE_COUNT = 0;
}
