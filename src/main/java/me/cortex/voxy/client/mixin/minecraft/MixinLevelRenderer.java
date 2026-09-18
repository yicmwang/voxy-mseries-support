package me.cortex.voxy.client.mixin.minecraft;

import me.cortex.voxy.client.VoxyClientInstance;
import me.cortex.voxy.client.config.VoxyConfig;
import me.cortex.voxy.client.core.IGetVoxyRenderSystem;
import me.cortex.voxy.client.core.VoxyRenderSystem;
import me.cortex.voxy.client.core.util.IrisUtil;
import me.cortex.voxy.common.Logger;
import me.cortex.voxy.common.world.WorldEngine;
import me.cortex.voxy.commonImpl.VoxyCommon;
import me.cortex.voxy.commonImpl.WorldIdentifier;
import net.minecraft.client.Minecraft;
import net.minecraft.client.multiplayer.ClientLevel;
import net.minecraft.client.renderer.LevelRenderer;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Unique;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/**
 * Owns the Voxy renderer's lifetime, hung off {@link LevelRenderer}.
 *
 * <p>26.2 port. The old version shadowed a {@code level} field and hooked {@code setLevel} and
 * {@code allChanged}; LevelRenderer has none of those any more (the field is gone, and
 * {@code allChanged} was removed outright — see also VoxyCommands). It now reads the level from
 * Minecraft and hooks {@code resetLevelRenderData}, which is the per-level reset and therefore the
 * level-change/reload point this needs.
 */
@Mixin(LevelRenderer.class)
public abstract class MixinLevelRenderer implements IGetVoxyRenderSystem {
    @Unique private VoxyRenderSystem renderer;

    @Override
    public VoxyRenderSystem getVoxyRenderSystem() {
        return this.renderer;
    }

    @Inject(method = "resetLevelRenderData", at = @At("RETURN"), order = 900)//We want to inject before sodium
    private void voxy$reloadRenderer(CallbackInfo ci) {
        this.shutdownRenderer();
        if (Minecraft.getInstance().level != null) {
            this.createRenderer();
        }
    }

    @Inject(method = "close", at = @At("HEAD"))
    private void voxy$injectClose(CallbackInfo ci) {
        this.shutdownRenderer();
    }

    @Override
    public void shutdownRenderer() {
        if (this.renderer != null) {
            this.renderer.shutdown();
            this.renderer = null;
        }
    }

    @Override
    public void createRenderer() {
        if (this.renderer != null) throw new IllegalStateException("Cannot have multiple renderers");
        if (!VoxyConfig.CONFIG.isRenderingEnabled()) {
            Logger.info("Not creating renderer due to disabled");
            return;
        }
        ClientLevel level = Minecraft.getInstance().level;
        if (level == null) {
            Logger.error("Not creating renderer due to null world");
            return;
        }
        var instance = (VoxyClientInstance)VoxyCommon.getInstance();
        if (instance == null) {
            Logger.error("Not creating renderer due to null instance");
            return;
        }
        WorldEngine world = WorldIdentifier.ofEngine(level);
        if (world == null) {
            Logger.error("Null world selected");
            return;
        }
        try {
            this.renderer = new VoxyRenderSystem(world, instance.getServiceManager());
        } catch (RuntimeException e) {
            if (IrisUtil.irisShaderPackEnabled()) {
                IrisUtil.disableIrisShaders();
            } else {
                throw e;
            }
        }
        instance.updateDedicatedThreads();
    }
}
