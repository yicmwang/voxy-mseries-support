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
        // A new manager is built on a level change AND on every render-distance change, but only the
        // first means the mask's contents are wrong. Clearing on a distance change is what left the
        // mask empty for the rest of a session (measured: built=0 for 3600 consecutive frames after
        // an RD change), and an empty mask culls nothing. See resetForLevel for why the distance case
        // does not need it.
        me.cortex.voxy.client.core.rendering.BuiltSectionMask.resetForLevel(level);
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
                    // isLightCorrect() is handed over rather than acted on here: this is the site that
                    // fires the moment Sodium meshes a section, which is the moment the standing
                    // "ingested before it was lit" hypothesis predicts the dark ones come from. It
                    // counts them so the hypothesis can be read off instead of argued.
                    VoxelIngestService.rawIngest(vrs.getEngine(), section, x, y, z,
                            blp == null ? null : blp.copy(), slp == null ? null : slp.copy(),
                            chunk.isLightCorrect());
                }
            }
        }

        return changes;
    }
}
