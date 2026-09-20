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
        DIAG_PROCESS_COUNT.incrementAndGet();
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
            scanIngestOutput(csec, task.raw);
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
    static ILightingSupplier lightingSupplier(final DataLayer blockLight, final DataLayer skyLight, final boolean lightCorrect) {
        if (!lightCorrect) {
            // Unlit: neither layer is trustworthy, so take both defaults. Sky 15 rather than 0 —
            // see above. Block 0 matches BlockLightSectionStorage's absent-layer answer.
            return (x, y, z) -> (byte) 15;
        }
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
    /**
     * <b>On by default, and that default is the fix to a measurement bug.</b> This was gated behind
     * {@code VOXY_INGEST_SCAN=1}, no run ever set it, so {@link #DIAG_VOXEL_GROUND} stayed at zero and
     * `[Metal-VOXEL2] groundAir=0` was read as "the ingest-side air above ground is never dark" when it
     * actually meant "the audit did not run". It was cited as evidence in that direction. A counter
     * that silently reads zero when it is switched off is worse than no counter: 4096 long reads per
     * section is nothing next to the conversion that follows it, so it runs unless explicitly killed
     * ({@code VOXY_INGEST_SCAN=0}).
     */
    private static final boolean SCAN_INGEST = !"0".equals(System.getenv("VOXY_INGEST_SCAN"));
    public static final java.util.concurrent.atomic.AtomicLong DIAG_VOXEL_SOLID = new java.util.concurrent.atomic.AtomicLong();
    public static final java.util.concurrent.atomic.AtomicLong DIAG_VOXEL_SOLID_DARK = new java.util.concurrent.atomic.AtomicLong();

    private static void scanIngestOutput(VoxelizedSection csec, boolean raw) {
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
        // Split by ingest path. The raw path is the one that fires the moment Sodium meshes a
        // section, so if "ingested before it was lit" is the fault, the dark air is on THIS side and
        // the chunk path is clean -- and a single combined number cannot say that.
        (raw ? DIAG_VOXEL_GROUND_RAW : DIAG_VOXEL_GROUND_CHUNK).addAndGet(ground);
        (raw ? DIAG_VOXEL_GROUND_DARK_RAW : DIAG_VOXEL_GROUND_DARK_CHUNK).addAndGet(groundDark);
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
    /** The same two, split by ingest path: raw = Sodium's section-info redirect, chunk = chunk load. */
    public static final java.util.concurrent.atomic.AtomicLong DIAG_VOXEL_GROUND_RAW = new java.util.concurrent.atomic.AtomicLong();
    public static final java.util.concurrent.atomic.AtomicLong DIAG_VOXEL_GROUND_DARK_RAW = new java.util.concurrent.atomic.AtomicLong();
    public static final java.util.concurrent.atomic.AtomicLong DIAG_VOXEL_GROUND_CHUNK = new java.util.concurrent.atomic.AtomicLong();
    public static final java.util.concurrent.atomic.AtomicLong DIAG_VOXEL_GROUND_DARK_CHUNK = new java.util.concurrent.atomic.AtomicLong();

    /**
     * Sections handed to the ingest while {@code ChunkAccess.isLightCorrect()} was <b>false</b>, split
     * by path. This is the direct measurement of the standing hypothesis — that Voxy bakes a section
     * before Minecraft has lit it and is never told to look again — and it separates it from the
     * alternative, that the light was there and was lost on the way in.
     */
    public static final java.util.concurrent.atomic.AtomicLong DIAG_INGEST_NOT_LIT_RAW = new java.util.concurrent.atomic.AtomicLong();
    public static final java.util.concurrent.atomic.AtomicLong DIAG_INGEST_NOT_LIT_CHUNK = new java.util.concurrent.atomic.AtomicLong();

    /**
     * Of the sections that arrived with a sky layer <i>present</i>, how many had one that is entirely
     * zero ({@code DataLayer.isEmpty()}). A present-and-empty sky layer is the shape of a section
     * whose light data exists as an allocation but has not been computed — as opposed to the absent
     * layer, which {@link #lightingSupplier} already answers with 15. Split by path for the same
     * reason as {@link #DIAG_INGEST_NOT_LIT_RAW}: the raw path is the one that fires early.
     */
    public static final java.util.concurrent.atomic.AtomicLong DIAG_SKY_EMPTY_RAW = new java.util.concurrent.atomic.AtomicLong();
    public static final java.util.concurrent.atomic.AtomicLong DIAG_SKY_EMPTY_CHUNK = new java.util.concurrent.atomic.AtomicLong();

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
        // Present-but-empty is the unlit signature. isEmpty() is the cheap depth value the storage
        // itself tracks, not a 2048-byte scan.
        if (sl != null && sl.isEmpty()) {
            (raw ? DIAG_SKY_EMPTY_RAW : DIAG_SKY_EMPTY_CHUNK).incrementAndGet();
        }
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
            if (!chunk.isLightCorrect()) DIAG_INGEST_NOT_LIT_CHUNK.incrementAndGet();
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
        DIAG_RAW_INGEST_COUNT.incrementAndGet();
        DIAG_ENQUEUE_COUNT.incrementAndGet();
        classifyLighting(bl, sl, true);
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
     *                     rather than inventing a number — see {@link #DIAG_INGEST_NOT_LIT_RAW}.
     */
    public static boolean rawIngest(WorldEngine engine, LevelChunkSection section, int x, int y, int z,
                                    DataLayer bl, DataLayer sl, boolean lightCorrect) {
        if (!shouldIngestSection(section, x, y, z)) return false;
        if (engine.instanceIn == null) return false;
        if (!engine.instanceIn.isIngestEnabled(null)) return false;//TODO: dont pass in null
        if (!lightCorrect) DIAG_INGEST_NOT_LIT_RAW.incrementAndGet();
        return engine.instanceIn.getIngestService().rawIngest0(engine, section, x, y, z, bl, sl, lightCorrect);
    }
}
