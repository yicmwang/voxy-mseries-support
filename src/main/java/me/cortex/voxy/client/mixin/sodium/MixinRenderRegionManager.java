package me.cortex.voxy.client.mixin.sodium;

import me.cortex.voxy.client.core.IGetVoxyRenderSystem;
import me.cortex.voxy.client.core.rendering.BuiltSectionMask;
import net.caffeinemc.mods.sodium.client.render.chunk.RenderSection;
import net.caffeinemc.mods.sodium.client.render.chunk.UniformBufferManager;
import net.caffeinemc.mods.sodium.client.render.chunk.compile.BuilderTaskOutput;
import net.caffeinemc.mods.sodium.client.render.chunk.region.RenderRegionManager;
import net.minecraft.core.SectionPos;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.ModifyArg;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

import java.util.Collection;

@Mixin(value = RenderRegionManager.class, remap = false)
public class MixinRenderRegionManager {

    /**
     * Tell {@link BuiltSectionMask} which sections Sodium has geometry for, at the moment a mesh
     * upload lands.
     *
     * <p>{@code isBuilt} alone is not the question: it means the build FINISHED, and Sodium marks a
     * section built even when it meshed to nothing, so using it alone would claim vanilla covers
     * empty air. {@code getLastMeshResultSize} is 0 for a section that produced no geometry, which
     * is the thing the cull needs to know.
     *
     * <p>The same collection also carries output for sections whose mesh was just cleared, and
     * those must leave the mask -- hence the else branch rather than only adding.
     */
    @Inject(method = "uploadResults(Ljava/util/Collection;Lnet/caffeinemc/mods/sodium/client/render/chunk/UniformBufferManager;)V",
            at = @At("HEAD"), remap = false)
    private void voxy$feedBuiltSectionMask(Collection<BuilderTaskOutput> outputs, UniformBufferManager ubm,
                                           CallbackInfo ci) {
        for (BuilderTaskOutput output : outputs) {
            RenderSection section = output.section;
            if (section == null) continue;
            SectionPos pos = section.getPosition();
            long key = SectionPos.asLong(pos.x(), pos.y(), pos.z());
            if (section.isBuilt() && section.getLastMeshResultSize() > 0) {
                BuiltSectionMask.add(key);
            } else {
                BuiltSectionMask.remove(key);
            }
        }
    }
    @ModifyArg(method = "uploadResults(Lnet/caffeinemc/mods/sodium/client/render/chunk/region/RenderRegion;Ljava/util/Collection;Lnet/caffeinemc/mods/sodium/client/render/chunk/UniformBufferManager;)V", at = @At(value = "INVOKE", target = "Lnet/caffeinemc/mods/sodium/client/render/chunk/region/RenderRegionManager$PendingSectionMeshUpload;<init>(Lnet/caffeinemc/mods/sodium/client/render/chunk/RenderSection;ILnet/caffeinemc/mods/sodium/client/render/chunk/data/BuiltSectionMeshParts;Lnet/caffeinemc/mods/sodium/client/render/chunk/terrain/TerrainRenderPass;Lnet/caffeinemc/mods/sodium/client/gpu/arena/PendingUpload;)V"), remap = false, index = 1)
    private int voxy$cancelFade(int original) {
        if (original == -1) return original;
        var vrs = IGetVoxyRenderSystem.getNullable();
        if (vrs==null) {
            return original;
        } else {
            return -999999;
        }
    }
}