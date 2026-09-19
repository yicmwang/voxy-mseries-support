package me.cortex.voxy.client.core.rendering.util;

import com.mojang.blaze3d.textures.GpuTextureView;
import me.cortex.voxy.client.core.gpu.IGpuSampler;
import me.cortex.voxy.client.core.gpu.RenderBackendFactory;
import me.cortex.voxy.client.core.gpu.RenderEncoder;
import me.cortex.voxy.client.core.gpu.SamplerDesc;
import me.cortex.voxy.client.core.metal.MetallumAttachmentTexture;
import me.cortex.voxy.client.core.metal.MetallumBridge;
import net.minecraft.client.Minecraft;

/**
 * Routes Minecraft's 16×16 lightmap to Voxy's terrain shaders.
 *
 * <p>On Metal, MC's lightmap is already a {@code MetalGpuTexture} on Metallum's device, so there is
 * nothing to copy across APIs — Voxy binds MC's own texture. This is the same arrangement
 * {@link me.cortex.voxy.client.core.model.bakery.ModelTextureBakery} uses for MC's block atlas:
 * adapt the foreign texture with {@link MetallumAttachmentTexture} and re-{@code refresh()} it each
 * frame, so a resource reload or a texture swap is picked up without re-registering.
 *
 * <p>It used to keep a Shared-storage mirror and copy the lightmap into it with
 * {@code glGetTexImage} once per frame. That is impossible under whole-frame Metal — there is no GL
 * context to read back through — and the copy had been short-circuited to an unconditional
 * {@code return}, leaving the mirror allocated, never written, and bound. Every LOD fragment then
 * sampled lighting of exactly black, which is why the terrain rendered as black blobs with a
 * correct silhouette and a healthy draw count. Sampling MC's texture directly removes the copy, the
 * frame-id gate it needed, and the staging buffer.
 */
public class LightMapHelper {

    /**
     * Adapter over MC's lightmap. Kept as a field rather than built per call because it owns a
     * registration in the backend's handle map — churning that once per terrain pass would leak ids.
     */
    private static MetallumAttachmentTexture lightmap;
    private static IGpuSampler sampler;
    private static boolean warnedUnavailable;

    /**
     * GL path: binds MC's GlTexture directly to the configured texture unit — MC keeps its own
     * lightmap upload current so we just point at it. Unused under whole-frame Metal, which goes
     * through {@link #bindMetal}.
     */
    public static void bind(int lightingIndex) {
        // Nothing to do: the GL path is driven entirely by MC's own texture unit state.
    }

    /**
     * Metal-only path: bind MC's lightmap plus a matching sampler at {@code slot}. Safe to call from
     * any of the three terrain passes in a frame — unlike the copy it replaces there is no per-frame
     * work here, so no frame-id gate is needed.
     */
    public static void bindMetal(RenderEncoder encoder, int slot) {
        if (lightmap == null) {
            lightmap = MetallumAttachmentTexture.ofMetalTexture("mc-lightmap", LightMapHelper::lightmapHandle);
            sampler = RenderBackendFactory.get().createSampler(SamplerDesc.builder()
                    .filter(SamplerDesc.Filter.LINEAR, SamplerDesc.Filter.LINEAR)
                    .mipFilter(SamplerDesc.MipFilter.NOT_MIPMAPPED)
                    .wrap(SamplerDesc.Wrap.CLAMP_TO_EDGE, SamplerDesc.Wrap.CLAMP_TO_EDGE)
                    .label("Voxy.MCLightmapSampler")
                    .build());
        }
        lightmap.refresh();
        if (lightmap.id() == -1) {
            // No handle this frame (closed texture, or a backend where MC's texture is not a
            // MetalGpuTexture). Binding anyway would ask the handle map for id -1 and throw, so
            // leave the slot alone and say so once.
            if (!warnedUnavailable) {
                warnedUnavailable = true;
                me.cortex.voxy.common.Logger.warn("LightMapHelper: MC lightmap has no Metal handle; "
                        + "LOD terrain will render unlit until it does");
            }
            return;
        }
        encoder.setTexture(slot, lightmap);
        encoder.setSampler(slot, sampler);
    }

    /** MC's level lightmap as a Metal handle, or 0 when it is not a Metallum texture. */
    private static long lightmapHandle() {
        GpuTextureView view = Minecraft.getInstance().gameRenderer.levelLightmap();
        // levelLightmap(), not lightmap(): the latter can return the UI variant, which is not the
        // level's lighting.
        return view == null ? 0L : MetallumBridge.textureHandle(view.texture());
    }
}
