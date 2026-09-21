package me.cortex.voxy.client.mixin.sodium;

import com.mojang.blaze3d.buffers.GpuBuffer;
import com.mojang.blaze3d.buffers.GpuBufferSlice;
import com.mojang.blaze3d.textures.GpuSampler;
import me.cortex.voxy.client.core.IGetVoxyRenderSystem;
import me.cortex.voxy.client.core.rendering.BuiltSectionMask;
import me.cortex.voxy.client.core.rendering.Viewport;
import net.caffeinemc.mods.sodium.client.render.chunk.ChunkRenderMatrices;
import net.caffeinemc.mods.sodium.client.render.chunk.DefaultChunkRenderer;
import net.caffeinemc.mods.sodium.client.render.chunk.LocalSectionIndex;
import net.caffeinemc.mods.sodium.client.gpu.device.batch.MultiDrawBatch;
import net.caffeinemc.mods.sodium.client.gpu.device.context.DrawContext;
import net.caffeinemc.mods.sodium.client.render.chunk.lists.ChunkRenderList;
import net.caffeinemc.mods.sodium.client.render.chunk.lists.ChunkRenderListIterable;
import net.caffeinemc.mods.sodium.client.render.chunk.region.RenderRegion;
import net.caffeinemc.mods.sodium.client.render.chunk.terrain.DefaultTerrainRenderPasses;
import net.caffeinemc.mods.sodium.client.render.chunk.terrain.TerrainRenderPass;
import net.caffeinemc.mods.sodium.client.render.viewport.CameraTransform;
import net.caffeinemc.mods.sodium.client.util.FogParameters;
import net.caffeinemc.mods.sodium.client.util.iterator.ByteIterator;
import net.minecraft.client.Minecraft;
import net.minecraft.core.SectionPos;
import net.minecraft.util.Mth;
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

    /** Whether this frame's feed runs: true only on the frame the camera entered a new section. */
    private static boolean voxy$feedThisFrame;

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
        // The mask is the union over the frame's terrain passes, not just this one. Sodium renders
        // SOLID, CUTOUT then TRANSLUCENT in that order (DefaultTerrainRenderPasses.ALL), so SOLID is
        // the frame boundary: clear there and let every pass contribute. Gating the feed to SOLID
        // alone would leave a section with no SOLID geometry unclaimed, and that is a section whose
        // surface is grass, leaves or a flower -- at RD 2 that is most of the visible surface, so the
        // LOD survived over exactly the cutout terrain.
        //
        // The clear is gated on the CAMERA'S SECTION rather than on the frame, so the culled region's
        // edges step at chunk boundaries instead of sliding along with the player. See
        // BuiltSectionMask.beginFrameIfSectionChanged for the measurement behind that and for the one
        // thing it costs.
        if (renderPass == DefaultTerrainRenderPasses.SOLID) {
            voxy$feedThisFrame = BuiltSectionMask.beginFrameIfSectionChanged(
                    Mth.floor(camera.x) >> 4, Mth.floor(camera.y) >> 4, Mth.floor(camera.z) >> 4);
        }
        if (voxy$feedThisFrame) {
            feedBuiltSectionMask(renderLists);
        }
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

    /**
     * Hand {@link BuiltSectionMask} the sections vanilla is drawing THIS FRAME.
     *
     * <p>This is the fix for the whole cull saga, and it is a deletion rather than a calculation. The
     * mask used to be fed from {@code RenderRegionManager.uploadResults} — what Sodium had MESHED — and
     * a distance rule was then asked to turn that into what Sodium RENDERS. It cannot: a distance is
     * not a frustum. Four versions of that rule were tried and each was wrong somewhere (sphere,
     * cylinder, conjunction, Sodium's own union), because they were all approximating the set that
     * {@code renderLists} already IS. The parameter is right here, it is a {@code ChunkRenderListIterable},
     * and its entries are exactly the sections about to be drawn in this pass.
     *
     * <p>Decoding is upstream's own, not reverse-engineered: a list entry is a
     * {@code RenderSection.getSectionIndex()}, which {@link LocalSectionIndex} packs from the section's
     * position within its region ({@code x & 7, y & 3, z & 7} — the region is 8x4x8 chunks), and the
     * region carries its own origin. So origin + unpack is the section, with no arithmetic of ours to
     * get wrong.
     *
     * <p>The caller clears the mask once per frame, at the first terrain pass, so this accumulates
     * SOLID + CUTOUT + TRANSLUCENT between calls.
     */
    private static void feedBuiltSectionMask(ChunkRenderListIterable renderLists) {
        java.util.Iterator<ChunkRenderList> lists = renderLists.iterator();
        while (lists.hasNext()) {
            ChunkRenderList list = lists.next();
            RenderRegion region = list.getRegion();
            final int originX = region.getChunkX();
            final int originY = region.getChunkY();
            final int originZ = region.getChunkZ();
            ByteIterator it = list.sectionsWithGeometryIterator(false);
            // Sodium returns NULL here, not an empty iterator, for a list with no geometry in this
            // pass -- verified in the shipped bytecode, where `sectionsWithGeometryCount == 0` is an
            // explicit `return null`. The same pattern is on sectionsWithSpritesIterator and the
            // block-entity accessors, so it is worth remembering rather than re-learning as an NPE.
            if (it == null) continue;
            while (it.hasNext()) {
                final int idx = it.nextByteAsInt();
                BuiltSectionMask.addDrawn(SectionPos.asLong(
                        originX + LocalSectionIndex.unpackX(idx),
                        originY + LocalSectionIndex.unpackY(idx),
                        originZ + LocalSectionIndex.unpackZ(idx)));
            }
        }
    }
}
