package me.cortex.voxy.common.world.service;

import me.cortex.voxy.common.Logger;
import me.cortex.voxy.common.thread.Service;
import me.cortex.voxy.common.thread.ServiceManager;
import me.cortex.voxy.common.voxelization.ILightingSupplier;
import me.cortex.voxy.common.voxelization.VoxelizedSection;
import me.cortex.voxy.common.voxelization.WorldConversionFactory;
import me.cortex.voxy.common.world.WorldEngine;
import me.cortex.voxy.common.world.WorldUpdater;
import me.cortex.voxy.commonImpl.VoxyCommon;
import me.cortex.voxy.commonImpl.WorldIdentifier;
import net.minecraft.core.SectionPos;
import net.minecraft.world.level.LightLayer;
import net.minecraft.world.level.chunk.DataLayer;
import net.minecraft.world.level.chunk.LevelChunk;
import net.minecraft.world.level.chunk.LevelChunkSection;
import net.minecraft.world.level.lighting.LayerLightSectionStorage;
import org.jetbrains.annotations.NotNull;

import java.util.concurrent.ConcurrentLinkedDeque;

public class VoxelIngestService {
    /** M13 diagnostic counters — read by AbstractRenderPipeline's Metal-DIAG dump. */
    public static final java.util.concurrent.atomic.AtomicLong DIAG_ENQUEUE_COUNT = new java.util.concurrent.atomic.AtomicLong();
    public static final java.util.concurrent.atomic.AtomicLong DIAG_PROCESS_COUNT = new java.util.concurrent.atomic.AtomicLong();
    public static final java.util.concurrent.atomic.AtomicLong DIAG_RAW_INGEST_COUNT = new java.util.concurrent.atomic.AtomicLong();

    private static final ThreadLocal<VoxelizedSection> SECTION_CACHE = ThreadLocal.withInitial(VoxelizedSection::createEmpty);
    private final Service service;
    private record IngestSection(int cx, int cy, int cz, WorldEngine world, LevelChunkSection section, DataLayer blockLight, DataLayer skyLight){}
    private final ConcurrentLinkedDeque<IngestSection> ingestQueue = new ConcurrentLinkedDeque<>();

    public VoxelIngestService(ServiceManager pool) {
        this.service = pool.createServiceNoCleanup(()->this::processJob, 5000, "Ingest service");
    }

    private void processJob() {
        DIAG_PROCESS_COUNT.incrementAndGet();
        var task = this.ingestQueue.pop();
        task.world.markActive();

        var section = task.section;
        var vs = SECTION_CACHE.get().setPosition(task.cx, task.cy, task.cz);

        if (section.hasOnlyAir() && task.blockLight==null && task.skyLight==null) {//If the chunk section has lighting data, propagate it
            WorldUpdater.insertUpdate(task.world, vs.zero());
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
        return lightingSupplier(task.blockLight, task.skyLight);
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
        return (x, y, z) -> {
            final int block = blockLight == null ? 0 : Math.min(15, blockLight.get(x, y, z));
            final int sky = skyLight == null ? 15 : Math.min(15, skyLight.get(x, y, z));
            return (byte) (sky | (block << 4));
        };
    }

    private static boolean shouldIngestSection(LevelChunkSection section, int cx, int cy, int cz) {
        return true;
    }

    /** Total times enqueueIngest was called regardless of outcome. */
    public static final java.util.concurrent.atomic.AtomicLong DIAG_ENQUEUE_CALL_COUNT = new java.util.concurrent.atomic.AtomicLong();
    /** Times enqueueIngest exited early because gotLighting was false. */
    public static final java.util.concurrent.atomic.AtomicLong DIAG_ENQUEUE_NO_LIGHTING_COUNT = new java.util.concurrent.atomic.AtomicLong();

    /**
     * Sections enqueued with BOTH light layers null. {@link #getLightingSupplier} answers these
     * with a constant {@code (byte) 0} — the section is baked as pitch black, with the correct
     * silhouette and a healthy draw count, which is indistinguishable on screen from a lighting
     * bug. Split by ingest path because they have different guard strengths: {@code enqueueIngest}
     * only checks at chunk granularity, {@code rawIngest} does not check at all.
     */
    public static final java.util.concurrent.atomic.AtomicLong DIAG_LIGHT_NONE_CHUNK = new java.util.concurrent.atomic.AtomicLong();
    public static final java.util.concurrent.atomic.AtomicLong DIAG_LIGHT_NONE_RAW = new java.util.concurrent.atomic.AtomicLong();
    /** Sections enqueued with exactly one layer null — half-lit, equally wrong but less visibly. */
    public static final java.util.concurrent.atomic.AtomicLong DIAG_LIGHT_HALF = new java.util.concurrent.atomic.AtomicLong();

    /**
     * Classify a section by what {@link #lightingSupplier} received. A null SKY layer is the
     * interesting case: it is the absent-means-15 default, and it is the section shape the old
     * zero-substituting supplier baked black. These counters measure how much of the world was
     * affected, which is the pass/fail for the fix.
     */
    private static void classifyLighting(DataLayer bl, DataLayer sl, boolean raw) {
        if (sl == null) {
            DIAG_LIGHT_NO_SKY.incrementAndGet();
            if (bl == null) {
                (raw ? DIAG_LIGHT_NONE_RAW : DIAG_LIGHT_NONE_CHUNK).incrementAndGet();
            } else {
                DIAG_LIGHT_HALF.incrementAndGet();
            }
        } else if (bl == null) {
            DIAG_LIGHT_HALF.incrementAndGet();
        }
    }

    /** Sections enqueued with no stored sky layer — the ones the old supplier zeroed. */
    public static final java.util.concurrent.atomic.AtomicLong DIAG_LIGHT_NO_SKY = new java.util.concurrent.atomic.AtomicLong();

    public boolean enqueueIngest(WorldEngine engine, LevelChunk chunk) {
        DIAG_ENQUEUE_CALL_COUNT.incrementAndGet();
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
                this.ingestQueue.add(new IngestSection(chunk.getPos().x(), i, chunk.getPos().z(), engine, section, null, null));
                try {
                    this.service.execute();
                } catch (Exception e) {
                    Logger.error("Executing had an error: assume shutting down, aborting",e);
                    break;
                }
            }
        }

        if (!gotLighting) {
            DIAG_ENQUEUE_NO_LIGHTING_COUNT.incrementAndGet();
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

            DIAG_ENQUEUE_COUNT.incrementAndGet();
            classifyLighting(bl, sl, false);
            this.ingestQueue.add(new IngestSection(chunk.getPos().x(), i, chunk.getPos().z(), engine, section, bl, sl));//TODO: fixme, this is technically not safe todo on the chunk load ingest, we need to copy the section data so it cant be modified while being read
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

    private boolean rawIngest0(WorldEngine engine, LevelChunkSection section, int x, int y, int z, DataLayer bl, DataLayer sl) {
        DIAG_RAW_INGEST_COUNT.incrementAndGet();
        DIAG_ENQUEUE_COUNT.incrementAndGet();
        classifyLighting(bl, sl, true);
        this.ingestQueue.add(new IngestSection(x, y, z, engine, section, bl, sl));
        try {
            this.service.execute();
            return true;
        } catch (Exception e) {
            Logger.error("Executing had an error: assume shutting down, aborting",e);
            return false;
        }
    }

    public static boolean rawIngest(WorldIdentifier id, LevelChunkSection section, int x, int y, int z, DataLayer bl, DataLayer sl) {
        if (id == null) return false;
        var engine = id.getOrCreateEngine();
        if (engine == null) return false;
        return rawIngest(engine, section, x, y, z, bl, sl);
    }

    public static boolean rawIngest(WorldEngine engine, LevelChunkSection section, int x, int y, int z, DataLayer bl, DataLayer sl) {
        if (!shouldIngestSection(section, x, y, z)) return false;
        if (engine.instanceIn == null) return false;
        if (!engine.instanceIn.isIngestEnabled(null)) return false;//TODO: dont pass in null
        return engine.instanceIn.getIngestService().rawIngest0(engine, section, x, y, z, bl, sl);
    }
}
