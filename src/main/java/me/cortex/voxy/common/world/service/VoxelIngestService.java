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
            // TRIED AND REVERTED (2026-09-19): filling this with air at sky 15 instead of plain
            // zero, on the theory that an absent sky layer means the section is open sky. It made
            // the frame dramatically worse -- LOD-band darkness went from ~10-20% to 75-83% and
            // stayed there. Left as zero, which is what upstream does.
            WorldUpdater.insertUpdate(task.world, vs.zero());
        } else {
            VoxelizedSection csec = WorldConversionFactory.convert(
                    SECTION_CACHE.get(),
                    task.world.getMapper(),
                    section.getStates(),
                    section.getBiomes(),
                    getLightingSupplier(task)
            );
            scanIngestOutput(csec);
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

    /**
     * Splits "the LOD is dark" into its two possible halves. Voxels that are solid but carry a
     * light byte of zero, counted at the moment the section leaves the ingest, tell us whether the
     * light engine handed us darkness or whether it was lost afterwards in storage, upload or
     * meshing. Only the first half is an ingest bug; the second is not, and the black-splotch
     * investigation wasted rounds by not separating them.
     *
     * <p>{@code VOXY_INGEST_SCAN=1}, read once — 4096 longs per section is cheap next to the
     * conversion itself, but there is no reason to pay it in a normal run.
     */
    private static final boolean SCAN_INGEST = "1".equals(System.getenv("VOXY_INGEST_SCAN"));
    public static final java.util.concurrent.atomic.AtomicLong DIAG_VOXEL_SOLID = new java.util.concurrent.atomic.AtomicLong();
    public static final java.util.concurrent.atomic.AtomicLong DIAG_VOXEL_SOLID_DARK = new java.util.concurrent.atomic.AtomicLong();

    private static void scanIngestOutput(VoxelizedSection csec) {
        if (!SCAN_INGEST) return;
        long solid = 0;
        long dark = 0;
        long top = 0;
        long topDark = 0;
        // Section index layout is y-major, matching DataLayer and Mipper: (y<<8)|(z<<4)|x.
        for (int z = 0; z < 16; z++) {
            for (int x = 0; x < 16; x++) {
                for (int y = 15; y >= 0; y--) {
                    final int i = (y << 8) | (z << 4) | x;
                    final long v = csec.section[i];
                    if (me.cortex.voxy.common.world.other.Mapper.isAir(v)) continue;
                    solid++;
                    final boolean isDark = (me.cortex.voxy.common.world.other.Mapper.getLightId(v) & 0xFF) == 0;
                    if (isDark) dark++;
                    // The column's topmost solid voxel is what a viewer sees from above, so it is
                    // the one voxel per column that is unambiguously exposed to the sky. Dark
                    // there is a real defect; dark anywhere else is just "underground".
                    if (y == 15) continue;
                    final long above = csec.section[((y + 1) << 8) | (z << 4) | x];
                    if (me.cortex.voxy.common.world.other.Mapper.isAir(above)) {
                        top++;
                        if (isDark) topDark++;
                    }
                    break;
                }
            }
        }
        // The cell that actually lights a surface face. Voxy's mesher takes an opaque block's face
        // light from the ADJACENT cell (RenderDataFactory: `(A&~LM) | (lighter&LM)`), because light
        // does not propagate into solid blocks -- a surface grass block legitimately holds light 0
        // in its own cell. So the meaningful question is whether the AIR resting on the ground
        // carries sky light, and counting dark solid voxels measures the wrong thing entirely.
        long ground = 0;
        long groundDark = 0;
        for (int z = 0; z < 16; z++) {
            for (int x = 0; x < 16; x++) {
                for (int y = 1; y < 16; y++) {
                    final int i = (y << 8) | (z << 4) | x;
                    final long v = csec.section[i];
                    if (!me.cortex.voxy.common.world.other.Mapper.isAir(v)) continue;
                    if (me.cortex.voxy.common.world.other.Mapper.isAir(csec.section[((y - 1) << 8) | (z << 4) | x])) continue;
                    ground++;
                    if ((me.cortex.voxy.common.world.other.Mapper.getLightId(v) & 0xFF) == 0) groundDark++;
                    break;
                }
            }
        }
        DIAG_VOXEL_GROUND.addAndGet(ground);
        DIAG_VOXEL_GROUND_DARK.addAndGet(groundDark);

        DIAG_VOXEL_SOLID.addAndGet(solid);
        DIAG_VOXEL_SOLID_DARK.addAndGet(dark);
        DIAG_VOXEL_TOP.addAndGet(top);
        DIAG_VOXEL_TOP_DARK.addAndGet(topDark);
    }

    /** Air cells resting on solid ground — the cells that light the visible surface faces. */
    public static final java.util.concurrent.atomic.AtomicLong DIAG_VOXEL_GROUND = new java.util.concurrent.atomic.AtomicLong();
    public static final java.util.concurrent.atomic.AtomicLong DIAG_VOXEL_GROUND_DARK = new java.util.concurrent.atomic.AtomicLong();

    /** Solid voxels with air directly above — the section's exposed surface. */
    public static final java.util.concurrent.atomic.AtomicLong DIAG_VOXEL_TOP = new java.util.concurrent.atomic.AtomicLong();
    public static final java.util.concurrent.atomic.AtomicLong DIAG_VOXEL_TOP_DARK = new java.util.concurrent.atomic.AtomicLong();

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

    /**
     * Does Minecraft agree with what we baked?
     *
     * <p>Every layer of this investigation so far has compared Voxy against Voxy. This asks the
     * authority instead: for sampled voxels, it reads the sky light back out of
     * {@code LevelLightEngine} — the same call vanilla's own renderer makes — and compares it with
     * the byte Voxy is about to store. A mismatch means the fault is in how the ingest obtains
     * light, and the direction of the mismatch says which way. Agreement means the LOD is faithfully
     * reproducing a world that really is that dark, and the search belongs somewhere else entirely.
     *
     * <p>{@code VOXY_LIGHT_COMPARE=1}.
     */
    private static final boolean COMPARE_LIGHT = "1".equals(System.getenv("VOXY_LIGHT_COMPARE"));
    public static final java.util.concurrent.atomic.AtomicLong DIAG_CMP_SAMPLES = new java.util.concurrent.atomic.AtomicLong();
    public static final java.util.concurrent.atomic.AtomicLong DIAG_CMP_AGREE = new java.util.concurrent.atomic.AtomicLong();
    public static final java.util.concurrent.atomic.AtomicLong DIAG_CMP_MC_BRIGHTER = new java.util.concurrent.atomic.AtomicLong();
    public static final java.util.concurrent.atomic.AtomicLong DIAG_CMP_VOXY_BRIGHTER = new java.util.concurrent.atomic.AtomicLong();

    /** Sampled sky light at the section's own voxels: MC's answer vs the byte we baked. */
    private static void compareAgainstLightEngine(net.minecraft.world.level.Level level,
                                                  SectionPos pos,
                                                  DataLayer blockLight,
                                                  DataLayer skyLight) {
        if (!COMPARE_LIGHT) return;
        var engine = level.getLightEngine();
        for (int y = 0; y < 16; y += 4) {
            for (int z = 0; z < 16; z += 4) {
                for (int x = 0; x < 16; x += 4) {
                    final int mcSky = engine.getLayerListener(LightLayer.SKY)
                            .getLightValue(pos.origin().offset(x, y, z));
                    final int mcBlock = engine.getLayerListener(LightLayer.BLOCK)
                            .getLightValue(pos.origin().offset(x, y, z));
                    final int voxySky = skyLight == null ? 15 : Math.min(15, skyLight.get(x, y, z));
                    final int voxyBlock = blockLight == null ? 0 : Math.min(15, blockLight.get(x, y, z));
                    DIAG_CMP_SAMPLES.incrementAndGet();
                    if (mcSky == voxySky && mcBlock == voxyBlock) {
                        DIAG_CMP_AGREE.incrementAndGet();
                    } else if (mcSky > voxySky || mcBlock > voxyBlock) {
                        DIAG_CMP_MC_BRIGHTER.incrementAndGet();
                    } else {
                        DIAG_CMP_VOXY_BRIGHTER.incrementAndGet();
                    }
                }
            }
        }
    }

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
            compareAgainstLightEngine(chunk.getLevel(), pos, bl, sl);
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
