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
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

/**
 * Voxy owns the far field's fog, so vanilla's fog range is pushed out of the way.
 *
 * <p>26.2 port: {@code setupFog} now returns {@link FogData} rather than a
 * {@code Vector4f}, and the {@code RenderSystem.getDevice()} call the old injection point
 * anchored on is gone. Injects at RETURN and mutates the returned FogData, matching upstream.
 * Again invisible to javac — mixin resolves injection points at APPLY time.
 */
@Mixin(value = FogRenderer.class, remap = true, priority = 900)//We must execute before sodium
public class MixinFogRenderer {
    /**
     * The fog the terrain shader actually reads is uploaded to a UBO by
     * {@code FogRenderer.updateBuffer(FogData)}, not read back out of the object {@code setupFog}
     * returns. Mutating only the return value was measured to be ineffective: {@code VOXY_FOG_TRACE}
     * shows this handler running and writing 1e8/1e9 into the returned FogData
     * ({@code OUT env=[1.0E8..1.0E8] renderDist=[1.0E9..1.0E9]}) while vanilla terrain still faded to
     * fog over the last stretch of the render distance. So the upload is intercepted too, at HEAD of
     * whichever overload the frame uses; whatever computes the numbers, the buffer gets Voxy's.
     */
    @Inject(method = "updateBuffer(Lnet/minecraft/client/renderer/fog/FogData;)V", at = @At("HEAD"))
    private void voxy$modifyFogBuffer(FogData data, CallbackInfo ci) {
        if (data == null) return;
        voxy$pushFogOut(data, "updateBuffer");
    }

    @Inject(method = "setupFog", at = @At("RETURN"))
    private void voxy$modifyFog(Camera camera, int renderDistanceInChunks, DeltaTracker deltaTracker,
                                float darkenWorldAmount, ClientLevel level,
                                CallbackInfoReturnable<FogData> cir) {
        if (!(VoxyConfig.CONFIG.enableRendering && VoxyConfig.CONFIG.enabled)) return;

        var vrs = IGetVoxyRenderSystem.getNullable();
        if (vrs == null) return;

        FogData data = cir.getReturnValue();
        if (data == null) return;
        voxy$pushFogOut(data, "setupFog");
    }

    /** Push every fog range Voxy owns out of the way, and report it under {@code VOXY_FOG_TRACE}. */
    private void voxy$pushFogOut(FogData data, String site) {

        // Diagnostic (VOXY_FOG_TRACE=1): the fog-disable was observed NOT to take effect. This logs
        // the incoming and outgoing values from BOTH sites, so "never reached" and "reached but
        // something re-derives the fog afterwards" are distinguishable rather than guessed at.
        final boolean trace = "1".equals(System.getenv("VOXY_FOG_TRACE"));
        if (trace && (FOG_TRACE_COUNT++ % 600) == 0) {
            me.cortex.voxy.common.Logger.info(String.format(java.util.Locale.ROOT,
                    "[FogTrace n=%d @%s] IN  env=[%.1f..%.1f] renderDist=[%.1f..%.1f] skyEnd=%.1f cloudEnd=%.1f",
                    FOG_TRACE_COUNT - 1, site, data.environmentalStart, data.environmentalEnd,
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
                    "[FogTrace n=%d @%s] OUT env=[%.1f..%.1f] renderDist=[%.1f..%.1f] skyEnd=%.1f cloudEnd=%.1f",
                    FOG_TRACE_COUNT - 1, site, data.environmentalStart, data.environmentalEnd,
                    data.renderDistanceStart, data.renderDistanceEnd, data.skyEnd, data.cloudEnd));
        }
    }

    private static long FOG_TRACE_COUNT = 0;
}
