package me.cortex.voxy.client.mixin.sodium;

import me.cortex.voxy.client.ICheekyClientChunkCache;
import me.cortex.voxy.client.config.VoxyConfig;
import me.cortex.voxy.client.core.IGetVoxyRenderSystem;
import me.cortex.voxy.client.core.VoxyRenderSystem;
import me.cortex.voxy.client.core.util.IrisUtil;
import me.cortex.voxy.common.world.service.VoxelIngestService;
import net.caffeinemc.mods.sodium.client.render.chunk.RenderSection;
import net.caffeinemc.mods.sodium.client.render.chunk.RenderSectionManager;
import net.caffeinemc.mods.sodium.client.render.chunk.data.BuiltSectionInfo;
import net.caffeinemc.mods.sodium.client.render.chunk.map.ChunkTrackerHolder;
import net.caffeinemc.mods.sodium.client.render.chunk.translucent_sorting.SortBehavior;
import net.caffeinemc.mods.sodium.client.render.viewport.Viewport;
import net.caffeinemc.mods.sodium.client.util.FogParameters;
import net.fabricmc.loader.api.FabricLoader;
import net.minecraft.client.multiplayer.ClientLevel;
import net.minecraft.core.SectionPos;
import net.minecraft.world.level.ChunkPos;
import net.minecraft.world.level.LightLayer;
import net.minecraft.world.level.chunk.status.ChunkStatus;
import org.spongepowered.asm.mixin.Final;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.Unique;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.Redirect;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

@Mixin(value = RenderSectionManager.class, remap = false)
public class MixinRenderSectionManager {
    @Unique
    private static final boolean BOBBY_INSTALLED = FabricLoader.getInstance().isModLoaded("bobby");

    @Shadow @Final private ClientLevel level;

    @Inject(method = "<init>", at = @At("TAIL"))
    private void voxy$resetChunkTracker(ClientLevel level, int renderDistance, SortBehavior sortBehavior, CallbackInfo ci) {
        this.bottomSectionY = this.level.getMinY()>>4;
        // Sodium rebuilds its RenderSectionManager on a level change or a render-distance change, and
        // the built-section set starts over with it. Without this the mirror keeps sections from the
        // previous world, and the next ChunkBoundRenderer to seed from it would mask-discard LODs
        // over chunks that no longer exist.
        me.cortex.voxy.client.core.rendering.ChunkBoundRenderer.mirrorReset();
    }

    /**
     * Feed the chunk-bound depth mask with the sections Sodium actually renders.
     *
     * <p>Nothing did this before. {@code ChunkBoundRenderer.addSection} / {@code mirrorAdd} had no
     * caller anywhere, so {@code chunk2idx} stayed empty, {@code renderMetal}'s {@code count > 0}
     * guard never opened, the mask pass issued no draw, and the mask kept the 0 it was cleared to.
     * The test in quads.frag is {@code if (gl_FragCoord.z < voxyBoundDepth) discard;}, and
     * {@code fragDepth < 0} is false for every fragment — so the mask discarded nothing, ever, and
     * the LOD drew over the loaded chunks and wrote depth there, leaving Sodium's terrain to fail
     * its depth test against it. That is "vanilla renders under the LOD".
     *
     * <p>Both the static mirror and the live renderer are fed. The mirror is what a *future*
     * ChunkBoundRenderer seeds from (a renderer reload builds a fresh one while Sodium's sections
     * never re-transition); the live instance needs the incremental events because it only reads the
     * mirror at construction.
     */
    @Inject(method = "onSectionAdded", at = @At("TAIL"))
    private void voxy$boundMaskAdd(int x, int y, int z, CallbackInfo ci) {
        long pos = me.cortex.voxy.client.core.rendering.ChunkBoundRenderer.packSectionPos(x, y, z);
        me.cortex.voxy.client.core.rendering.ChunkBoundRenderer.mirrorAdd(pos);
        VoxyRenderSystem vrs = IGetVoxyRenderSystem.getNullable();
        if (vrs != null) {
            vrs.chunkBoundRenderer.addSection(pos);
        }
    }

    @Inject(method = "onSectionRemoved", at = @At("TAIL"))
    private void voxy$boundMaskRemove(int x, int y, int z, CallbackInfo ci) {
        long pos = me.cortex.voxy.client.core.rendering.ChunkBoundRenderer.packSectionPos(x, y, z);
        me.cortex.voxy.client.core.rendering.ChunkBoundRenderer.mirrorRemove(pos);
        VoxyRenderSystem vrs = IGetVoxyRenderSystem.getNullable();
        if (vrs != null) {
            vrs.chunkBoundRenderer.removeSection(pos);
        }
    }

    @Inject(method = "onChunkRemoved", at = @At("HEAD"))
    private void voxy$injectIngest(int x, int z, CallbackInfo ci) {
        //TODO: Am not quite sure if this is right
        if (VoxyConfig.CONFIG.ingestEnabled && !BOBBY_INSTALLED) {
            var cccm = (ICheekyClientChunkCache)this.level.getChunkSource();
            if (cccm != null) {
                var chunk = cccm.voxy$cheekyGetChunk(x, z);
                if (chunk != null) {
                    VoxelIngestService.tryAutoIngestChunk(chunk);
                }
            }
        }
    }


    @Inject(method = "onChunkAdded", at = @At("HEAD"))
    private void voxy$ingestOnAdd(int x, int z, CallbackInfo ci) {
        if (this.level != null && VoxyConfig.CONFIG.ingestEnabled) {
            var cccm = this.level.getChunkSource();
            if (cccm != null) {
                var chunk = cccm.getChunk(x, z, ChunkStatus.FULL, false);
                if (chunk != null) {
                    VoxelIngestService.tryAutoIngestChunk(chunk);
                }
            }
        }
    }


    @Unique private long cachedChunkPos = -1;
    @Unique private int cachedChunkStatus;
    @Unique private int bottomSectionY;


    @Redirect(method = "updateSectionInfo", at = @At(value = "INVOKE", target = "Lnet/caffeinemc/mods/sodium/client/render/chunk/RenderSection;setInfo(Lnet/caffeinemc/mods/sodium/client/render/chunk/data/BuiltSectionInfo;)I"))
    private int voxy$updateOnUpload(RenderSection instance, BuiltSectionInfo info) {
        boolean isInvisible = instance.isInvisible();
        int changes = instance.setInfo(info);
        VoxyRenderSystem vrs = null;
        if (isInvisible == instance.isInvisible() || changes == 0 || (vrs = IGetVoxyRenderSystem.getNullable()) == null) {
            return changes;
        }
        int x = instance.getChunkX(), y = instance.getChunkY(), z = instance.getChunkZ();

        if (!isInvisible && VoxyConfig.CONFIG.ingestEnabled) {
            var tracker = ((AccessorChunkTracker) ChunkTrackerHolder.get(this.level)).getChunkStatus();
            //in theory the cache value could be wrong but is so soso unlikely and at worst means we either duplicate ingest a chunk
            // which... could be bad ;-; or we dont ingest atall which is ok!
            long key = ChunkPos.pack(x, z);
            if (key != this.cachedChunkPos) {
                this.cachedChunkPos = key;
                this.cachedChunkStatus = tracker.getOrDefault(key, 0);
            }
            if (this.cachedChunkStatus == 3) {//If this chunk still has surrounding chunks
                var cccm = this.level.getChunkSource();
                //var chunk = ((ICheekyClientChunkCache)cccm).voxy$cheekyGetChunk(x, z);
                //Dont thinks need to use cheekyGetChunk here as thats handled by the inject into head of onChunkRemoved
                // but only ingest if the chunkstatus is full and exists
                var chunk = cccm.getChunk(x, z, ChunkStatus.FULL, false);
                if (chunk != null) {
                    var section = chunk.getSection(y - this.bottomSectionY);
                    var lp = this.level.getLightEngine();

                    var csp = SectionPos.of(x, y, z);
                    var blp = lp.getLayerListener(LightLayer.BLOCK).getDataLayerData(csp);
                    var slp = lp.getLayerListener(LightLayer.SKY).getDataLayerData(csp);

                    //Note: we dont do this check and just blindly ingest, it shouldbe ok :tm:
                    //if (blp != null || slp != null)
                    VoxelIngestService.rawIngest(vrs.getEngine(), section, x, y, z, blp == null ? null : blp.copy(), slp == null ? null : slp.copy());
                }
            }
        }

        return changes;
    }
}
