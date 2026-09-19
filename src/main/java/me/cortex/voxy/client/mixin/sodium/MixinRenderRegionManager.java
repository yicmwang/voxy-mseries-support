package me.cortex.voxy.client.mixin.sodium;

import me.cortex.voxy.client.core.IGetVoxyRenderSystem;
import me.cortex.voxy.client.core.VoxyRenderSystem;
import me.cortex.voxy.client.core.rendering.ChunkBoundRenderer;
import net.caffeinemc.mods.sodium.client.render.chunk.RenderSection;
import net.caffeinemc.mods.sodium.client.render.chunk.compile.BuilderTaskOutput;
import net.caffeinemc.mods.sodium.client.render.chunk.region.RenderRegionManager;
import net.caffeinemc.mods.sodium.client.render.chunk.UniformBufferManager;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.ModifyArg;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

import java.util.Collection;

@Mixin(value = RenderRegionManager.class, remap = false)
public class MixinRenderRegionManager {

    /**
     * Feed the chunk-bound depth mask with the sections Sodium has actually meshed.
     *
     * <p>Nothing fed this mask before: {@code ChunkBoundRenderer.addSection} / {@code mirrorAdd} had
     * no caller anywhere, so {@code chunk2idx} stayed empty, {@code renderMetal}'s {@code count > 0}
     * guard never opened, the mask pass issued no draw, and the mask kept the 0 it was cleared to.
     * quads.frag tests {@code if (gl_FragCoord.z < voxyBoundDepth) discard;}, and
     * {@code fragDepth < 0} is false for every fragment — so the mask discarded nothing, ever, and
     * the LOD drew across the loaded chunks and wrote depth there, leaving Sodium's terrain to fail
     * its depth test against it. That is "vanilla renders under the LOD instead of over it".
     *
     * <p>This is the upload point rather than Sodium's section-loading events because the mask is a
     * volume test, not a surface test: a section with no geometry still occludes everything behind
     * it if it is in the mask, so entries have to mean "there is something here to hide LOD behind".
     * {@code isBuilt} is re-checked because the same collection also carries output for sections
     * whose mesh was just cleared, and those must leave the mask.
     *
     * <p>Both the static mirror and the live renderer are fed. The mirror is what a *future*
     * ChunkBoundRenderer seeds from — a renderer reload builds a fresh instance while Sodium's
     * sections never re-transition — and the live instance needs the incremental events because it
     * only reads the mirror at construction.
     */
    /**
     * Whether this section has anything to hide LOD behind — which is not the same question as
     * {@link RenderSection#isBuilt()}.
     *
     * <p>{@code isBuilt()} means the build finished, and Sodium marks a section built even when it
     * meshed to nothing. Using it as the mask predicate put every air section above the terrain into
     * the mask, so each chunk column masked its full height and the LOD was discarded in the empty
     * air above every chunk — visible as sky-coloured rectangles ringing each vanilla chunk, occluding
     * LOD that had nothing in front of it. {@code getLastMeshResultSize()} is 0 for a section that
     * produced no geometry, which is the question the mask actually needs answered.
     */
    private static boolean hasGeometry(RenderSection section) {
        return section.isBuilt() && section.getLastMeshResultSize() > 0;
    }

    @Inject(method = "uploadResults(Ljava/util/Collection;Lnet/caffeinemc/mods/sodium/client/render/chunk/UniformBufferManager;)V",
            at = @At("HEAD"), remap = false)
    private void voxy$feedBoundMask(Collection<BuilderTaskOutput> outputs, UniformBufferManager ubm, CallbackInfo ci) {
        VoxyRenderSystem vrs = IGetVoxyRenderSystem.getNullable();
        for (BuilderTaskOutput output : outputs) {
            RenderSection section = output.section;
            if (section == null) continue;
            long pos = ChunkBoundRenderer.packSectionPos(
                    section.getPosition().x(), section.getPosition().y(), section.getPosition().z());
            if (hasGeometry(section)) {
                ChunkBoundRenderer.mirrorAdd(pos);
                // Guarded like the removal: a rebuilt section is re-uploaded every time its mesh
                // changes, and an unguarded add would make _addPos log "Chunk already in map" on
                // each rebuild.
                if (vrs != null && !vrs.chunkBoundRenderer.hasSection(pos)) {
                    vrs.chunkBoundRenderer.addSection(pos);
                }
            } else {
                ChunkBoundRenderer.mirrorRemove(pos);
                if (vrs != null && vrs.chunkBoundRenderer.hasSection(pos)) {
                    vrs.chunkBoundRenderer.removeSection(pos);
                }
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