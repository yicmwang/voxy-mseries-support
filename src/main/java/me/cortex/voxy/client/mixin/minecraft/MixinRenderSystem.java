package me.cortex.voxy.client.mixin.minecraft;


import com.mojang.blaze3d.systems.GpuDevice;
import com.mojang.blaze3d.systems.RenderSystem;
import me.cortex.voxy.client.VoxyClient;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

//Thanks iris for making me need todo this ;-; _irritater_
@Mixin(RenderSystem.class)
public class MixinRenderSystem {
    // We need to inject before iris to initialize our systems.
    // 26.2: initRenderer now takes the created GpuDevice rather than the raw
    // (windowHandle, debugVerbosity, sync, ShaderSource, renderDebugLabels)
    // tuple it took at 1.21.11. Signature drift here is invisible to javac --
    // mixin validates descriptors at APPLY time -- so this only surfaced on a
    // real client launch.
    @Inject(method = "initRenderer", order = 900, remap = false, at = @At("RETURN"))
    private static void voxy$injectInit(GpuDevice device, CallbackInfo ci) {
        VoxyClient.initVoxyClient();
    }
}
