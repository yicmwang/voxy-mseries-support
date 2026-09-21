package me.cortex.voxy.common.world.service;

import me.cortex.voxy.common.Logger;
import me.cortex.voxy.common.thread.Service;
import me.cortex.voxy.common.thread.ServiceManager;
import me.cortex.voxy.common.voxelization.ILightingSupplier;
import me.cortex.voxy.common.voxelization.VoxelizedSection;
import me.cortex.voxy.common.voxelization.WorldConversionFactory;
import me.cortex.voxy.common.world.WorldEngine;
import me.cortex.voxy.common.world.WorldUpdater;
import me.cortex.voxy.common.world.other.Mapper;
import me.cortex.voxy.commonImpl.VoxyCommon;
import me.cortex.voxy.commonImpl.WorldIdentifier;
import net.minecraft.core.SectionPos;
import net.minecraft.world.level.LightLayer;
import net.minecraft.world.level.chunk.DataLayer;
import net.minecraft.world.level.chunk.LevelChunk;
import net.minecraft.world.level.chunk.LevelChunkSection;
import net.minecraft.world.level.lighting.LayerLightSectionStorage;
import org.jetbrains.annotations.NotNull;

import java.util.Arrays;
import java.util.concurrent.ConcurrentLinkedDeque;

public class VoxelIngestService {
    /** M13 diagnostic counters — read by AbstractRenderPipeline's Metal-DIAG dump. */

    /**
     * Fill an all-air, no-light-layer section with air at FULL SKY instead of zero
     * ({@code VOXY_AIR_SECTION_SKY_LIGHT=1}).
     *
     * <p><b>Measured, and it does NOT fix the black.</b> The reasoning was sound: an absent sky layer
     * is, per MC's own {@code SkyLightSectionStorage}, the uniformly-fully-lit case, so storing zero
     * gives every air voxel above the terrain a light of 0, and the surface face of every column takes
     * its light from exactly that voxel. Switching it on leaves the camera-independent counts where
     * they were, though — {@code [Metal-DARKSRC]} dark-from-air 11,358,933 -> 11,522,942 and
     * {@code [Metal-LITAUD] zeroLight} 43.50% -> 42.92% at equal section counts. So the air those dark
     * faces read is NOT in these sections; it is inside partially-solid sections, where
     * {@link #lightingSupplier} already supplies it.
     *
     * <p>Kept as an opt-in rather than deleted because the earlier attempt at it was reverted on a
     * reading (~10-20% dark to 75-83%) that this round showed to be untrustworthy — an unpinned camera
     * and a bake whose metadata was wrong. This is that same change measured properly. It is a real
     * latent bug for fully-lit open-sky sections on its own merits, but it is not the splotches. Off by
     * default.
     */
    private static final boolean AIR_SECTION_SKY_LIGHT =
            "1".equals(System.getenv("VOXY_AIR_SECTION_SKY_LIGHT"));

    private static final ThreadLocal<VoxelizedSection> SECTION_CACHE = ThreadLocal.withInitial(VoxelizedSection::createEmpty);
    private final Service service;
    /**
     * @param raw whether this section came in through {@code rawIngest} (Sodium's section-info redirect
     *            and the border-block-change path) or through {@code enqueueIngest} (the chunk-level
     *            path). The two have different guard strengths and, more importantly, different
     *            timing: the raw path fires the instant Sodium meshes a section, which is not
     *            necessarily the instant its light exists. Every audit below is split by it, because
     *            a fault that only shows up on one path needs a different fix than one on both.
     */
    private record IngestSection(int cx, int cy, int cz, WorldEngine world, LevelChunkSection section, DataLayer blockLight, DataLayer skyLight, boolean raw, boolean lightCorrect){}
    private final ConcurrentLinkedDeque<IngestSection> ingestQueue = new ConcurrentLinkedDeque<>();

    public VoxelIngestService(ServiceManager pool) {
        this.service = pool.createServiceNoCleanup(()->this::processJob, 5000, "Ingest service");
    }

    private void processJob() {
        var task = this.ingestQueue.pop();
        task.world.markActive();

        var section = task.section;
        var vs = SECTION_CACHE.get().setPosition(task.cx, task.cy, task.cz);

        if (section.hasOnlyAir() && task.blockLight==null && task.skyLight==null) {//If the chunk section has lighting data, propagate it
            // An all-air section whose sky layer is ABSENT is, per MC's own SkyLightSectionStorage, a
            // section that is uniformly fully sky-lit -- it stores no DataLayer precisely because every
            // cell is 15, so storing zero here is wrong on its own terms. It is NOT the black splotches
            // though: with VOXY_AIR_SECTION_SKY_LIGHT=1 the camera-independent counts do not move
            // (dark-from-air 11,358,933 -> 11,522,942). See the gate's javadoc.
            boolean skyLitAir = AIR_SECTION_SKY_LIGHT;
            if (skyLitAir) {
                var lit = vs.zero();
                long airAtFullSky = me.cortex.voxy.common.world.other.Mapper.airWithLight(0x0F);
                // The whole array, not just the 16^3 level-0 block: a VoxelizedSection also carries the
                // already-mipped 8^3/4^3/2^3/1 levels, and this branch skips mipSection(), so anything
                // left at zero is what a coarser LOD cell reads. Filling only level 0 would leave the
                // black at distance while fixing it up close.
                java.util.Arrays.fill(lit.section, airAtFullSky);
                WorldUpdater.insertUpdate(task.world, lit);
            } else {
                WorldUpdater.insertUpdate(task.world, vs.zero());
            }
        } else {
            VoxelizedSection csec = WorldConversionFactory.convert(
                    SECTION_CACHE.get(),
                    task.world.getMapper(),
                    section.getStates(),
                    section.getBiomes(),
                    getLightingSupplier(task)
            );
            WorldConversionFactory.mipSection(csec, task.world.getMapper());
            WorldUpdater.insertUpdate(task.world, csec);
        }
    }

    @NotNull
    private static ILightingSupplier getLightingSupplier(IngestSection task) {
        return lightingSupplier(task.blockLight, task.skyLight, task.lightCorrect);
    }

    /**
     * Build the per-voxel light supplier for a section from its two {@link DataLayer}s.
     *
     * <p><b>A null layer is that layer's DEFAULT, not "no light".</b> Minecraft's own light engine
     * is the authority on what the default is, and the two layers disagree:
     * <ul>
     *   <li>{@code BlockLightSectionStorage.getLightValue} returns <b>0</b> when the block layer is
     *       absent — a section with no stored block light is unlit by blocks.</li>
     *   <li>{@code SkyLightSectionStorage.getLightValue} returns <b>15</b> when the sky layer is
     *       absent — a section at or above its column's sky-light source ({@code topSections})
     *       stores no {@code DataLayer} precisely <i>because</i> it is uniformly fully lit, and
     *       below that height the lookup walks upward and still falls back to 15.</li>
     * </ul>
     *
     * <p>This used to answer {@code (byte) 0} whenever it could not find both layers, and to zero
     * whichever half was missing. Sky is the half that matters: forcing it to 0 bakes the
     * brightest terrain in the world — open ground with nothing above it to shade it, which is
     * exactly when the sky layer is absent — as pitch black. On screen that is a black patch with
     * a correct silhouette, a correct draw count and no geometry error to find, sitting next to
     * perfectly lit terrain whose section happens to contain one shadow and therefore does have a
     * stored layer. A section holding a light source but no sky layer lands in the block-only case
     * and keeps its block light with sky zeroed, which is the red-speckle-within-black the
     * light-readout frames show.
     *
     * <p>A layer that <i>is</i> stored and is all-zero is genuinely dark, and still yields 0.
     * That is what {@code DataLayer.isEmpty()} would have conflated with absence.
     */
    static ILightingSupplier lightingSupplier(final DataLayer blockLight, final DataLayer skyLight) {
        return lightingSupplier(blockLight, skyLight, true);
    }

    /**
     * The same, but told whether the section's chunk had been lit at all when it was read.
     *
     * <p>The paragraph above assumes a stored all-zero layer is genuinely dark. That is only true of
     * a chunk Minecraft has finished lighting. A chunk that is <i>not</i> light-correct has had no
     * light computed for it yet, and its layers then read all-zero for the same reason a section
     * above the sky-light source reads all-zero-in-absence: there is nothing there yet. Baking those
     * zeros is permanent — the LOD has no notion of light arriving later — and it draws the terrain
     * as a black mass with a correct silhouette, which is what the black splotches are.
     *
     * <p>So an unlit chunk is answered as fully sky-lit, which is upstream's own rule for an absent
     * sky layer and the same rule this method already applies when the layer is missing. It is the
     * recoverable direction: if the light does arrive, the section is re-ingested and corrected; if it
     * never does, the terrain is over-bright rather than black, and a guess of "open to the sky" is
     * right for the surface of every column that has nothing above it.
     *
     * @param lightCorrect {@code ChunkAccess.isLightCorrect()} at the moment the layers were copied
     */
    /**
     * The same as the two-argument form. {@code lightCorrect} is NOT consulted.
     *
     * <p>It used to be. When it was false this returned a constant sky 15 for the whole section, on the
     * reasoning that a chunk which is not light-correct has had no light computed for it yet, so its
     * layers read zero for the same reason an absent layer does. Two things were wrong with that.
     *
     * <p>First, <b>{@code ChunkAccess.isLightCorrect()} is never set true on a vanilla client.</b> The
     * client's light path is {@code ClientPacketListener.applyLightData -> LevelLightEngine.queueSectionData},
     * which queues light; nothing on it calls {@code setLightCorrect}. So the flag says nothing about
     * whether light has been computed — it is false for every section, always.
     *
     * <p>Second, <b>the correction the old comment promised did not exist.</b> It said "if the light does
     * arrive, the section is re-ingested and corrected"; nothing in the tree re-ingests on light arrival,
     * and no light-update hook exists. So the guess was not recoverable — it was permanent.
     *
     * <p>Together those made sky 15 the DEFAULT outcome of client-side ingest rather than an edge case,
     * and it threw away real light: measured, 54% of ingested sections arrive with both layers null (for
     * which the per-layer rule below already answers sky 15), and the rest carry real data that this
     * branch overwrote.
     *
     * <p>The user chose removal over the correct fix, knowing the trade: present-and-zero layers now bake
     * as darkness again, which is bug 2's original symptom and is sticky in the same way. The correct fix
     * is to gate ingest on the light packet having actually arrived — the readiness signal already exists
     * and is already tested at {@code MixinRenderSectionManager} ({@code cachedChunkStatus == 3}, set from
     * {@code ClientChunkCacheMixin.onChunkLoaded} and {@code ClientPacketListenerMixin.onLightDataReceived})
     * — and then read the data verbatim, which needs no substitution at all.
     */
    static ILightingSupplier lightingSupplier(final DataLayer blockLight, final DataLayer skyLight, final boolean lightCorrect) {
        return (x, y, z) -> {
            final int block = blockLight == null ? 0 : Math.min(15, blockLight.get(x, y, z));
            final int sky = skyLight == null ? 15 : Math.min(15, skyLight.get(x, y, z));
            return (byte) (sky | (block << 4));
        };
    }

    private static boolean shouldIngestSection(LevelChunkSection section, int cx, int cy, int cz) {
        return true;
    }



    public boolean enqueueIngest(WorldEngine engine, LevelChunk chunk) {
        if (!this.service.isLive()) {
            return false;
        }
        if (!engine.isLive()) {
            throw new IllegalStateException("Tried inserting chunk into WorldEngine that was not alive");
        }

        engine.markActive();

        var lightingProvider = chunk.getLevel().getLightEngine();
        boolean gotLighting = false;

        int i = chunk.getMinSectionY() - 1;
        boolean allEmpty = true;
        for (var section : chunk.getSections()) {
            i++;
            if (section == null || !shouldIngestSection(section, chunk.getPos().x(), i, chunk.getPos().z())) continue;
            allEmpty&=section.hasOnlyAir();
            //if (section.isEmpty()) continue;
            var pos = SectionPos.of(chunk.getPos(), i);
            if (lightingProvider.getDebugSectionType(LightLayer.SKY, pos) != LayerLightSectionStorage.SectionType.LIGHT_AND_DATA && lightingProvider.getDebugSectionType(LightLayer.BLOCK, pos) != LayerLightSectionStorage.SectionType.LIGHT_AND_DATA)
                continue;
            gotLighting = true;
        }

        if (allEmpty&&!gotLighting) {
            //Special case all empty chunk columns, we need to clear it out
            i = chunk.getMinSectionY() - 1;
            for (var section : chunk.getSections()) {
                i++;
                if (section == null || !shouldIngestSection(section, chunk.getPos().x(), i, chunk.getPos().z())) continue;
                this.ingestQueue.add(new IngestSection(chunk.getPos().x(), i, chunk.getPos().z(), engine, section, null, null, false, chunk.isLightCorrect()));
                try {
                    this.service.execute();
                } catch (Exception e) {
                    Logger.error("Executing had an error: assume shutting down, aborting",e);
                    break;
                }
            }
        }

        if (!gotLighting) {
            return false;
        }

        var blp = lightingProvider.getLayerListener(LightLayer.BLOCK);
        var slp = lightingProvider.getLayerListener(LightLayer.SKY);


        i = chunk.getMinSectionY() - 1;
        for (var section : chunk.getSections()) {
            i++;
            if (section == null || !shouldIngestSection(section, chunk.getPos().x(), i, chunk.getPos().z())) continue;
            //if (section.isEmpty()) continue;
            var pos = SectionPos.of(chunk.getPos(), i);

            var bl = blp.getDataLayerData(pos);
            if (bl != null) {
                bl = bl.copy();
            }

            var sl = slp.getDataLayerData(pos);
            if (sl != null) {
                sl = sl.copy();
            }

            //If its null for either, assume failure to obtain lighting and ignore section
            //if (blNone && slNone) {
            //    continue;
            //}

            this.ingestQueue.add(new IngestSection(chunk.getPos().x(), i, chunk.getPos().z(), engine, section, bl, sl, false, chunk.isLightCorrect()));//TODO: fixme, this is technically not safe todo on the chunk load ingest, we need to copy the section data so it cant be modified while being read
            try {
                this.service.execute();
            } catch (Exception e) {
                Logger.error("Executing had an error: assume shutting down, aborting",e);
                break;
            }
        }
        return true;
    }

    public int getTaskCount() {
        return this.service.numJobs();
    }

    public void shutdown() {
        this.service.shutdown();
    }

    //Utility method to ingest a chunk into the given WorldIdentifier or world
    public static boolean tryIngestChunk(WorldIdentifier worldId, LevelChunk chunk) {
        if (worldId == null) return false;
        var instance = VoxyCommon.getInstance();
        if (instance == null) return false;
        if (!instance.isIngestEnabled(worldId)) return false;
        var engine = instance.getOrCreate(worldId);
        if (engine == null) return false;
        return instance.getIngestService().enqueueIngest(engine, chunk);
    }

    //Try to automatically ingest the chunk into the correct world
    public static boolean tryAutoIngestChunk(LevelChunk chunk) {
        return tryIngestChunk(WorldIdentifier.of(chunk.getLevel()), chunk);
    }

    private boolean rawIngest0(WorldEngine engine, LevelChunkSection section, int x, int y, int z, DataLayer bl, DataLayer sl, boolean lightCorrect) {
        this.ingestQueue.add(new IngestSection(x, y, z, engine, section, bl, sl, true, lightCorrect));
        try {
            this.service.execute();
            return true;
        } catch (Exception e) {
            Logger.error("Executing had an error: assume shutting down, aborting",e);
            return false;
        }
    }

    public static boolean rawIngest(WorldIdentifier id, LevelChunkSection section, int x, int y, int z, DataLayer bl, DataLayer sl, boolean lightCorrect) {
        if (id == null) return false;
        var engine = id.getOrCreateEngine();
        if (engine == null) return false;
        return rawIngest(engine, section, x, y, z, bl, sl, lightCorrect);
    }

    public static boolean rawIngest(WorldEngine engine, LevelChunkSection section, int x, int y, int z, DataLayer bl, DataLayer sl) {
        return rawIngest(engine, section, x, y, z, bl, sl, true);
    }

    /**
     * @param lightCorrect the caller's {@code ChunkAccess.isLightCorrect()} for the chunk this section
     *                     belongs to. Callers that cannot answer it pass {@code true}, which under-counts
     *                     rather than inventing a number.
     */
    public static boolean rawIngest(WorldEngine engine, LevelChunkSection section, int x, int y, int z,
                                    DataLayer bl, DataLayer sl, boolean lightCorrect) {
        if (!shouldIngestSection(section, x, y, z)) return false;
        if (engine.instanceIn == null) return false;
        if (!engine.instanceIn.isIngestEnabled(null)) return false;//TODO: dont pass in null
        return engine.instanceIn.getIngestService().rawIngest0(engine, section, x, y, z, bl, sl, lightCorrect);
    }
}
