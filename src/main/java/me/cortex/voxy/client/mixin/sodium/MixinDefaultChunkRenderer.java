package me.cortex.voxy.client.mixin.sodium;

import com.mojang.blaze3d.buffers.GpuBuffer;
import com.mojang.blaze3d.buffers.GpuBufferSlice;
import com.mojang.blaze3d.textures.GpuSampler;
import me.cortex.voxy.client.core.IGetVoxyRenderSystem;
import me.cortex.voxy.client.core.rendering.Viewport;
import net.caffeinemc.mods.sodium.client.render.chunk.ChunkRenderMatrices;
import net.caffeinemc.mods.sodium.client.render.chunk.DefaultChunkRenderer;
import net.caffeinemc.mods.sodium.client.gpu.device.batch.MultiDrawBatch;
import net.caffeinemc.mods.sodium.client.gpu.device.context.DrawContext;
import net.caffeinemc.mods.sodium.client.render.chunk.lists.ChunkRenderListIterable;
import net.caffeinemc.mods.sodium.client.render.chunk.terrain.DefaultTerrainRenderPasses;
import net.caffeinemc.mods.sodium.client.render.chunk.terrain.TerrainRenderPass;
import net.caffeinemc.mods.sodium.client.render.viewport.CameraTransform;
import net.caffeinemc.mods.sodium.client.util.FogParameters;
import net.minecraft.client.Minecraft;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.Redirect;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/**
 * Voxy's entry point into Sodium's terrain rendering.
 *
 * <p>Rewritten for Sodium 0.9 during the 26.2 port. The previous version extended
 * {@code ShaderChunkRenderer} and pulled in Sodium's GL internals
 * ({@code CommandList}, {@code RenderDevice}, {@code MultiDrawBatch},
 * {@code GlBufferUsage}) which no longer exist. It did two extra jobs that are
 * deliberately dropped here:
 *
 * <ul>
 *   <li><b>Shared-quad-index-buffer preflight</b> — a workaround for Sodium's GL
 *       buffer growing via {@code glMapBufferRange} mid-frame. That is a GL-only
 *       failure mode; under a Metal backend it cannot occur.</li>
 *   <li><b>The compositing bridge</b> ({@code IOSurfaceBridgeCompositor},
 *       {@code VxContractInjector}, {@code MetalVxResolvePass}) — replaced in P2
 *       by drawing directly into Metallum's frame.</li>
 * </ul>
 *
 * <p>What remains is the part that is backend-independent: on Sodium's SOLID pass,
 * set up the viewport and run Voxy's pipeline. The hook fires at SOLID head so the
 * near scene overdraws the LODs through the shared depth buffer.
 */
@Mixin(value = DefaultChunkRenderer.class, remap = false)
public abstract class MixinDefaultChunkRenderer {
    /**
     * VOXY_NO_VANILLA=1 suppresses Sodium's terrain draws entirely, so the LOD can be inspected in
     * isolation -- its exact shape, with no vanilla terrain in the way.
     *
     * <p>Why it has to be done HERE, at the batch draw, rather than by cancelling
     * {@code DefaultChunkRenderer.render}: Voxy's own LOD render is driven from inside that method,
     * by {@link #voxy$injectRender} just before {@code ShaderChunkRenderer.end}. Cancelling the whole
     * method would cancel Voxy with it. Suppressing only the batch draws leaves {@code begin} and
     * {@code end} to bracket the pass as usual, so the injection still fires on a live pass.
     *
     * <p>The cull is deliberately NOT disabled alongside this. Sections where vanilla drew are still
     * culled, which is correct -- the question this view answers is what the LOD does where it is
     * ALLOWED to draw, not what it would draw if unculled.
     *
     * <p>Live risk, from the comment below: Metallum is said to open its render encoder lazily on
     * Sodium's first draw. If that is literally at the draw rather than at {@code begin}/{@code setContext},
     * suppressing draws leaves no live encoder and Voxy has nothing to render into, which reads as a
     * blank screen rather than as a clean view. That is the thing to look for first if this comes up empty.
     */
    private static final boolean NO_VANILLA = "1".equals(System.getenv("VOXY_NO_VANILLA"));

    @Redirect(method = "render", at = @At(value = "INVOKE",
            target = "Lnet/caffeinemc/mods/sodium/client/gpu/device/batch/MultiDrawBatch;draw(Lnet/caffeinemc/mods/sodium/client/gpu/device/context/DrawContext;)V"),
            remap = false)
    private void voxy$suppressVanillaTerrain(MultiDrawBatch batch, DrawContext context) {
        if (NO_VANILLA) {
            return;
        }
        batch.draw(context);
    }

    // Injected at the TAIL of the pass (just before Sodium ends it), not HEAD.
    //
    // Under whole-frame Metal this matters: Metallum opens its render encoder lazily on Sodium's
    // first draw, so at HEAD there is no open encoder and no bound attachments -- Voxy would have
    // nothing to render into. By the tail Sodium has drawn, Metallum's encoder is live, and Voxy
    // can borrow it and share the pass's depth. Depth testing still gives the right result either
    // way: LOD fragments behind the near terrain are rejected, and those showing through sky gaps
    // draw.
    @Inject(method = "render", at = @At(value = "INVOKE",
            target = "Lnet/caffeinemc/mods/sodium/client/render/chunk/ShaderChunkRenderer;end(Lnet/caffeinemc/mods/sodium/client/render/chunk/terrain/TerrainRenderPass;)V",
            shift = At.Shift.BEFORE))
    private void voxy$injectRender(ChunkRenderMatrices matrices,
                                   ChunkRenderListIterable renderLists,
                                   TerrainRenderPass renderPass,
                                   CameraTransform camera,
                                   FogParameters fogParameters,
                                   boolean indexedRenderingEnabled,
                                   GpuSampler terrainSampler,
                                   GpuBufferSlice uniformData,
                                   GpuBuffer sectionTimeInfo,
                                   CallbackInfo ci) {
        if (renderPass != DefaultTerrainRenderPasses.SOLID) {
            return;
        }
        var renderer = IGetVoxyRenderSystem.getNullable();
        if (renderer == null) {
            return;
        }
        Viewport<?> viewport = renderer.setupViewport(matrices, fogParameters, camera.x, camera.y, camera.z);
        renderer.renderOpaque(viewport);
    }
}
