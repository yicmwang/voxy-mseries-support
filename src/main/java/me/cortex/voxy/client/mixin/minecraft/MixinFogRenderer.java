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

        data.renderDistanceStart = 999999999;
        data.renderDistanceEnd = 999999999;
        if (!VoxyConfig.CONFIG.useEnvironmentalFog) {
            data.environmentalStart = 99999999;
            data.environmentalEnd = 99999999;
        }
    }
}
