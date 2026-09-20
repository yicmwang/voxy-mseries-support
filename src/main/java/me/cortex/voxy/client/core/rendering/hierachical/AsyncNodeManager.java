package me.cortex.voxy.client.core.rendering.hierachical;

import it.unimi.dsi.fastutil.ints.Int2IntOpenHashMap;
import it.unimi.dsi.fastutil.ints.IntConsumer;
import it.unimi.dsi.fastutil.ints.IntOpenHashSet;
import it.unimi.dsi.fastutil.longs.LongOpenHashSet;
import me.cortex.voxy.client.TimingStatistics;
import me.cortex.voxy.client.core.gpu.IGpuBuffer;
import me.cortex.voxy.client.core.gl.shader.Shader;
import me.cortex.voxy.client.core.gl.shader.ShaderType;
import me.cortex.voxy.client.core.rendering.GeometryCache;
import me.cortex.voxy.client.core.rendering.SectionUpdateRouter;
import me.cortex.voxy.client.core.rendering.building.BuiltSection;
import me.cortex.voxy.client.core.rendering.building.RenderGenerationService;
import me.cortex.voxy.client.core.rendering.section.geometry.BasicAsyncGeometryManager;
import me.cortex.voxy.client.core.rendering.section.geometry.BasicSectionGeometryData;
import me.cortex.voxy.client.core.rendering.section.geometry.IGeometryData;
import me.cortex.voxy.client.core.rendering.util.UploadStream;
import me.cortex.voxy.common.Logger;
import me.cortex.voxy.common.util.AllocationArena;
import me.cortex.voxy.common.util.MemoryBuffer;
import me.cortex.voxy.common.util.UnsafeUtil;
import me.cortex.voxy.common.world.WorldEngine;
import me.cortex.voxy.common.world.WorldSection;
import org.lwjgl.system.MemoryUtil;

import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.util.List;
import java.util.concurrent.ConcurrentLinkedDeque;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.LockSupport;
import java.util.concurrent.locks.StampedLock;

import static org.lwjgl.opengl.GL43C.*;

//TODO: create an "async upload stream", that is, the upload stream is a raw mapped buffer pointer that can be written to
// which is then synced to the gpu on "render thread sync",


//An "async host" for a NodeManager, has specific synchonius entry and exit points
// this is done off thread to reduce the amount of work done on the render thread, improving frame stability and reducing runtime overhead
public class AsyncNodeManager {
    private static final VarHandle RESULT_HANDLE;
    private static final VarHandle RESULT_CACHE_1_HANDLE;
    private static final VarHandle RESULT_CACHE_2_HANDLE;
    static {
        try {
            RESULT_HANDLE = MethodHandles.lookup().findVarHandle(AsyncNodeManager.class, "results", SyncResults.class);
            RESULT_CACHE_1_HANDLE = MethodHandles.lookup().findVarHandle(AsyncNodeManager.class, "resultCache1", SyncResults.class);
            RESULT_CACHE_2_HANDLE = MethodHandles.lookup().findVarHandle(AsyncNodeManager.class, "resultCache2", SyncResults.class);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            throw new RuntimeException(e);
        }
    }

    private final Thread thread;
    public final int maxNodeCount;
    private final long geometryCapacity;
    private volatile boolean running = true;

    private final NodeManager manager;
    private final BasicAsyncGeometryManager geometryManager;
    private final IGeometryData geometryData;
    private final SectionUpdateRouter router;

    private final GeometryCache geometryCache = new GeometryCache(1L<<32);

    private final AtomicInteger workCounter = new AtomicInteger();

    @SuppressWarnings("FieldMayBeFinal")
    private volatile SyncResults results = null, resultCache1 = new SyncResults(), resultCache2 = new SyncResults();


    //locals for during iteration
    private final IntOpenHashSet tlnIdChange = new IntOpenHashSet();//"Encoded" add/remove id, first bit indicates if its add or remove, 1 is add
    //Top bit indicates clear or reset
    private final IntOpenHashSet cleanerIdResetClear = new IntOpenHashSet();//Tells the cleaner if it needs to clear the id to 0, or reset the id to the current frame

    private boolean needsWaitForSync = false;

    // ---------------------------------------------------------------------------------------------
    // End-to-end verification of the metadata path (VOXY_METACHK).
    // ---------------------------------------------------------------------------------------------

    /**
     * A section's 32-byte metadata record is what turns a draw command into a quad range: cmdgen
     * reads {@code quadStart} and the eight packed group counts out of it, and the vertex shader
     * reads the position from it. It is produced CPU-side by {@code writeMetadataSplit} into a
     * staging buffer, copied into the upload ring, and finally scattered into the metadata buffer
     * <b>by the GPU</b> -- four steps, three of them asynchronous, and every one of them has to agree
     * before a draw points at the right quads. Any of them failing produces the same visible thing:
     * geometry that is not the section's own, reshuffled as it re-streams, black wherever the donor
     * happens to be interior or unlit.
     *
     * <p>Rather than guess which step, remember the 32 bytes that were <i>intended</i> for each
     * section and read the metadata buffer back once the scatter has certainly run. All of this is on
     * the async thread, which owns the staging buffer, so there is no CPU-side race; the only other
     * writer is the GPU's scatter riding the render thread's command buffer, which the wall-clock lag
     * covers. A mismatch means the metadata the GPU will read is not what the geometry manager wrote
     * -- the one condition that produces the artefact, measured rather than argued.
     */
    private static final boolean METACHK = !"0".equals(System.getenv("VOXY_METACHK"));
    private static final int METACHK_SLOTS = 1 << 14;
    /**
     * How many render ticks must have passed since a record was staged before it is read back. The
     * scatter that carries it runs on a tick AFTER the one that publishes the results, and the
     * command buffer is submitted around there, so one tick is not enough. A wall-clock lag was the
     * first attempt and it produced false positives at startup -- the async thread staged and then
     * verified before the render thread had ticked even once, so the buffer was legitimately still
     * empty -- which is exactly the kind of reading that has misled this investigation before.
     */
    private static final int METACHK_LAG_TICKS = 3;
    private static final long METACHK_LAG_MS = 150;
    private final me.cortex.voxy.common.util.MemoryBuffer metachkExpected =
            new me.cortex.voxy.common.util.MemoryBuffer((long) METACHK_SLOTS * 32L);
    private final int[] metachkId = new int[METACHK_SLOTS];
    private final long[] metachkStamp = new long[METACHK_SLOTS];
    private final long[] metachkStampTick = new long[METACHK_SLOTS];
    private final it.unimi.dsi.fastutil.ints.Int2IntOpenHashMap metachkSlot =
            new it.unimi.dsi.fastutil.ints.Int2IntOpenHashMap(METACHK_SLOTS);
    private int metachkHead = 0;
    /** Advanced by {@link #tick} on the render thread; read by the async thread as a gate. */
    private volatile long renderTick = 0;
    private long metachkChecked = 0, metachkMismatch = 0, metachkEvicted = 0, metachkTooEarly = 0;
    private long metachkLastLog = 0;
    private int metachkExamples = 0;

    {
        java.util.Arrays.fill(this.metachkId, -1);
        this.metachkSlot.defaultReturnValue(-1);
    }

    /** Remember the 32 bytes just staged for {@code id}, to be read back once the scatter has run. */
    private void metachkStage(final int id, final long ptrA, final long ptrB) {
        if (!METACHK) return;
        int slot = this.metachkSlot.get(id);
        if (slot == -1) {
            slot = this.metachkHead;
            this.metachkHead = (this.metachkHead + 1) & (METACHK_SLOTS - 1);
            final int evicted = this.metachkId[slot];
            if (evicted != -1) {
                this.metachkSlot.remove(evicted);
                this.metachkEvicted++;
            }
            this.metachkSlot.put(id, slot);
            this.metachkId[slot] = id;
        }
        final long dst = this.metachkExpected.address + (long) slot * 32L;
        MemoryUtil.memCopy(ptrA, dst, 16L);
        MemoryUtil.memCopy(ptrB, dst + 16L, 16L);
        this.metachkStamp[slot] = System.nanoTime();
        this.metachkStampTick[slot] = this.renderTick;
    }

    /** Read the metadata buffer back for every staged record older than the lag, and compare. */
    private void metachkVerify() {
        if (!METACHK) return;
        if (!(((BasicSectionGeometryData) this.geometryData).getMetadataBuffer()
                instanceof me.cortex.voxy.client.core.metal.MetalBuffer mb)) return;
        final long mp = mb.getContentsPtr();
        if (mp == 0) return;

        final long now = System.nanoTime();
        final long lagNanos = METACHK_LAG_MS * 1_000_000L;
        final long tick = this.renderTick;
        for (int slot = 0; slot < METACHK_SLOTS; slot++) {
            final int id = this.metachkId[slot];
            if (id == -1) continue;
            if (tick - this.metachkStampTick[slot] < METACHK_LAG_TICKS) continue;
            if (now - this.metachkStamp[slot] < lagNanos) continue;
            final long exp = this.metachkExpected.address + (long) slot * 32L;
            final long got = mp + (long) id * 32L;
            int diff = 0;
            for (int k = 0; k < 32; k += 4) {
                diff |= MemoryUtil.memGetInt(exp + k) ^ MemoryUtil.memGetInt(got + k);
            }
            if (diff != 0) {
                this.metachkMismatch++;
                if (this.metachkExamples++ < 8) {
                    Logger.warn("[Metal-METACHK!] section " + id + " metadata mismatch"
                            + " expected=[quadStart=" + Integer.toUnsignedString(MemoryUtil.memGetInt(exp + 12))
                            + " counts=" + Integer.toUnsignedString(MemoryUtil.memGetInt(exp + 16)) + ","
                            + Integer.toUnsignedString(MemoryUtil.memGetInt(exp + 20)) + ","
                            + Integer.toUnsignedString(MemoryUtil.memGetInt(exp + 24)) + ","
                            + Integer.toUnsignedString(MemoryUtil.memGetInt(exp + 28)) + "]"
                            + " buffer=[quadStart=" + Integer.toUnsignedString(MemoryUtil.memGetInt(got + 12))
                            + " counts=" + Integer.toUnsignedString(MemoryUtil.memGetInt(got + 16)) + ","
                            + Integer.toUnsignedString(MemoryUtil.memGetInt(got + 20)) + ","
                            + Integer.toUnsignedString(MemoryUtil.memGetInt(got + 24)) + ","
                            + Integer.toUnsignedString(MemoryUtil.memGetInt(got + 28)) + "]");
                }
            } else {
                this.metachkChecked++;
            }
            this.metachkId[slot] = -1;
            this.metachkSlot.remove(id);
        }

        final long total = this.metachkChecked + this.metachkMismatch;
        // Time-based so the verification RATE is visible. A count-based condition (every 2000) made
        // "verified 1500 records and then nothing was staged again" look identical to "stopped
        // working", which is the same trap as the earlier count-only log.
        if (total > 0 && now - this.metachkLastLog > 5_000_000_000L) {
            this.metachkLastLog = now;
            Logger.info("[Metal-METACHK] checked=" + this.metachkChecked
                    + " mismatch=" + this.metachkMismatch
                    + " evicted=" + this.metachkEvicted
                    + " pending=" + this.metachkSlot.size()
                    + " renderTicks=" + this.renderTick);
        }
    }

    public AsyncNodeManager(int maxNodeCount, IGeometryData geometryData, RenderGenerationService renderService) {
        //Note the current implmentation of ISectionWatcher is threadsafe
        //Note: geometry data is the data store/source, not the management, it is just a raw store of data
        // it MUST ONLY be accessed on the render thread
        // AsyncNodeManager will use an AsyncGeometryManager as the manager for the data store, and sync the results on the render thread
        this.geometryData = geometryData;
        this.geometryCapacity = ((BasicSectionGeometryData)geometryData).getGeometryCapacityBytes();

        this.maxNodeCount = maxNodeCount;

        this.thread = new Thread(()->{
            try {
                while (this.running) {
                    this.run();
                }
            } catch (Exception e) {
                Logger.error("Critical error occurred in async processor, things will be broken", e);
            }
        });
        this.thread.setName("Async Node Manager");
        this.thread.setDaemon(true);// don't block JVM shutdown if this thread is stuck

        this.geometryManager = new BasicAsyncGeometryManager(((BasicSectionGeometryData)geometryData).getMaxSectionCount(), this.geometryCapacity);

        this.router = new SectionUpdateRouter();
        this.router.setCallbacks(pos->{//On initial render gen, try get from geometry cache
            var cachedGeometry = this.geometryCache.remove(pos);
            if (cachedGeometry != null) {//Use the cached geometry
                this.submitGeometryResult(cachedGeometry);
            } else {//Else we need to request it
                renderService.enqueueTask(pos);
            }
        }, renderService::enqueueTask, this::submitChildChange);
        renderService.setResultConsumer(this::submitGeometryResult);

        this.manager = new NodeManager(maxNodeCount, this.geometryManager, this.router);

        //Dont do the move... is just to much effort
        this.manager.setClear(new NodeManager.ICleaner() {
            @Override
            public void alloc(int id) {
                AsyncNodeManager.this.cleanerIdResetClear.remove(id);//Remove clear
                AsyncNodeManager.this.cleanerIdResetClear.add(id|(1<<31));//Add reset
            }

            @Override
            public void move(int from, int to) {
                //noop (sorry :( will cause some perf loss/incorrect cleaning )
            }

            @Override
            public void free(int id) {
                AsyncNodeManager.this.cleanerIdResetClear.remove(id|(1<<31));//Remove reset
                AsyncNodeManager.this.cleanerIdResetClear.add(id);//Add clear
            }
        });
        this.manager.setTLNCallbacks(id->{
            if (!this.tlnIdChange.remove(id)) {
                if (!this.tlnIdChange.add(id|(1<<31))) {
                    throw new IllegalStateException();
                }
            }
        }, id -> {
            if (!this.tlnIdChange.remove(id|(1<<31))) {
                if (!this.tlnIdChange.add(id)) {
                    throw new IllegalStateException();
                }
            }
        });
    }

    private SyncResults getMakeResultObject() {
        SyncResults resultSet = (SyncResults)RESULT_CACHE_1_HANDLE.getAndSet(this, null);
        if (resultSet == null) {//Not in the first object
            resultSet = (SyncResults)RESULT_CACHE_2_HANDLE.getAndSet(this, null);
        }
        if (resultSet == null) {
            throw new IllegalStateException("There should always be an object in the result set cache pair");
        }
        //Reset everything to default
        resultSet.reset();
        return resultSet;
    }

    /** UBO binding for scatter.comp's `Push { uint count; }` block. */
    private static final int SCATTER_PUSH_BINDING = 14;

    private final me.cortex.voxy.client.core.gpu.RenderBackend backend = me.cortex.voxy.client.core.gpu.RenderBackendFactory.get();

    private final me.cortex.voxy.client.core.gpu.IGpuPipeline scatterWrite = this.backend.createComputePipeline(
            new me.cortex.voxy.client.core.gpu.ComputePipelineDesc(
                    me.cortex.voxy.client.core.gl.shader.ShaderLoader.parse("voxy:util/scatter.comp"),
                    java.util.Map.of(
                            "INPUT_BUFFER_BINDING", "0",
                            "OUTPUT_BUFFER1_BINDING", "1",
                            "OUTPUT_BUFFER2_BINDING", "2",
                            "PUSH_BINDING", Integer.toString(SCATTER_PUSH_BINDING)),
                    null, null,
                    128, 1, 1,
                    "AsyncNodeManager.scatterWrite"));

    private final me.cortex.voxy.client.core.gpu.IGpuPipeline multiMemcpy = this.backend.createComputePipeline(
            new me.cortex.voxy.client.core.gpu.ComputePipelineDesc(
                    me.cortex.voxy.client.core.gl.shader.ShaderLoader.parse("voxy:util/memcpy.comp"),
                    java.util.Map.of(
                            "INPUT_HEADER_BUFFER_BINDING", "0",
                            "INPUT_DATA_BUFFER_BINDING", "1",
                            "OUTPUT_BUFFER_BINDING", "2"),
                    null, null,
                    256, 1, 1,
                    "AsyncNodeManager.multiMemcpy"));

    private void run() {
        if (this.workCounter.get() <= 0) {
            //TODO: here, instead of parking, we can do more work on other sub-tasks such as filtering the mesh build queue
            LockSupport.park();
            if (this.workCounter.get() <= 0 || !this.running) {//No work
                return;
            }
            //This is a funny thing, wait a bit, this allows for better batching, but this thread is independent of everything else so waiting a bit should be mostly ok
            try {
                Thread.sleep(10);
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
        }

        if (!this.running) {
            return;
        }

        // Read back any metadata record old enough that its scatter has certainly executed.
        this.metachkVerify();


        int workDone = 0;

        {
            LongOpenHashSet add = null;
            LongOpenHashSet rem = null;
            long stamp = this.tlnLock.writeLock();

            if (!this.tlnAdd.isEmpty()) {
                add = new LongOpenHashSet(this.tlnAdd);
                this.tlnAdd.clear();
            }
            if (!this.tlnRem.isEmpty()) {
                rem = new LongOpenHashSet(this.tlnRem);
                this.tlnRem.clear();
            }

            this.tlnLock.unlockWrite(stamp);
            int work = 0;
            if (rem != null) {
                var iter = rem.longIterator();
                while (iter.hasNext()) {
                    this.manager.removeTopLevelNode(iter.nextLong());
                    work++;
                }
            }

            if (add != null) {
                var iter = add.longIterator();
                while (iter.hasNext()) {
                    this.manager.insertTopLevelNode(iter.nextLong());
                    work++;
                }
            }

            workDone += work;
        }

        do {
            var job = this.childUpdateQueue.poll();
            if (job == null)
                break;
            workDone++;
            this.manager.processChildChange(job.key, job.getNonEmptyChildren());
            job.release();
        } while (true);


        //Limit uploading as well as by geometry capacity being available
        // must have 50 mb of free geometry space to upload
        for (int limit = 0; limit < 300 && ((this.geometryCapacity-this.geometryManager.getGeometryUsedBytes())>50_000_000L); limit++) {
            var job = this.geometryUpdateQueue.poll();
            if (job == null)
                break;
            workDone++;
            this.manager.processGeometryResult(job);
        }

        while (true) {//Process all request batches
            var job = this.requestBatchQueue.poll();
            if (job == null)
                break;
            workDone++;
            long ptr = job.address;
            int count = MemoryUtil.memGetInt(ptr);
            ptr += 8;//Its 8 to keep alignment
            if (job.size < count * 8L + 8) {
                // Counted and skipped, NOT thrown.
                //
                // This used to throw, and the throw is not a log line: the try/catch that catches it
                // is OUTSIDE the `while (this.running)` loop, so the exception exits the loop and the
                // async node manager thread dies permanently. For the rest of the session nothing
                // processes node requests, nothing uploads geometry, nothing propagates child changes
                // -- while the renderer keeps drawing whatever state it was left holding, and other
                // code keeps reusing the memory those stale references point at.
                //
                // The user deferred this as an old nuisance ("that error has existed for a long time,
                // we'll fix it later"). It is being instrumented now because it is a candidate for the
                // flashing splotches: a dead node manager is exactly the kind of thing that produces
                // geometry that no longer matches its section, intermittently, and gets worse the
                // longer a session runs. The count also settles how often the traversal's readback is
                // garbage -- a nonzero count is proof that it happens at all, which INVALID=0 on a
                // clean run cannot tell us.
                DIAG_REQ_BATCH_OVERFLOW.incrementAndGet();
                job.free();
                continue;
            }
            for (int i = 0; i < count; i++) {
                long pos = ((long) MemoryUtil.memGetInt(ptr)) << 32; ptr += 4;
                pos |= Integer.toUnsignedLong(MemoryUtil.memGetInt(ptr)); ptr += 4;
                // Validate what came back from the GPU before acting on it.
                //
                // This is the CPU reading GPU-written traversal output, and it is the one place in
                // the pipeline where a wrong value turns into wrong NODES being loaded and meshed
                // rather than wrong pixels being shaded. The user's splotch report -- shapes that do
                // not resemble their own section, one frame, different every time -- is what that
                // would look like from outside, and the deferred batch-overflow IllegalStateException
                // a few lines above is its sibling: a garbage count there, a garbage-but-in-range
                // position here.
                //
                // The check is a round trip. getWorldSectionId packs level, x, y and z into a long
                // with masked fields, so a value that did not come from that packing cannot survive
                // being unpacked and repacked -- any stray bit lands in a masked-off position and the
                // result differs. That catches corruption without needing to know what it looks like.
                if (!validSectionPos(pos)) {
                    DIAG_REQ_INVALID.incrementAndGet();
                }
                DIAG_REQ_VALIDATED.incrementAndGet();
                this.manager.processRequest(pos);
            }
            job.free();
        }


        do {
            var job = this.removeBatchQueue.poll();
            if (job == null)
                break;
            workDone++;
            long ptr = job.address;
            int zeroCount = 0;
            for (int i = 0; i < NodeCleaner.OUTPUT_COUNT; i++) {
                long pos = ((long) MemoryUtil.memGetInt(ptr)) << 32; ptr += 4;
                pos |= Integer.toUnsignedLong(MemoryUtil.memGetInt(ptr)); ptr += 4;

                if (pos == -1) {
                    //TODO: investigate how or what this happens
                    continue;
                }

                if (pos == 0 && zeroCount++>0) {
                    Logger.error("Remove node pos is 0 " + zeroCount + " times, this is really bad, please report" );
                    continue;
                }

                this.manager.removeNodeGeometry(pos);
            }
            job.free();
        } while (true);

        if (this.workCounter.addAndGet(-workDone) < 0) {
            try {
                Thread.sleep(1000);
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
            //Due to synchronization "issues", wait a millis (give up this time slice)
            if (this.workCounter.get() < 0) {
                Logger.error("Work counter less than zero, hope it fixes itself...");
                //return;
            }
        }

        if (workDone == 0) {//Nothing happened, which is odd, but just return
            //Should probably log that nothing happened, at least once
            return;
        }
        //=====================
        //process output events and atomically sync to results

        //Events into manager
        //manager.insertTopLevelNode();
        //manager.removeTopLevelNode();

        //manager.removeNodeGeometry();

        //manager.processRequest();
        //manager.processChildChange();
        //manager.processGeometryResult();


        //Outputs from manager
        //manager.setClear();
        //manager.setTLNCallbacks();

        //manager.writeChanges()


        //Run in a loop, process all the input events, collect the output events merge with previous and publish
        // note: inner event processing is a loop, is.. should be synced to attomic/volatile variable that is being watched
        // when frametime comes around, want to exit out as quick as possible, or make the event publishing
        // "effectivly immediately", that is, atomicly swap out the render side event updates

        //like
        // var current = <new events>
        // var old = getAndSet(this.events, null);
        // if (old != null) {current = merge(old, current);}
        // getAndSet(this.events, current);
        // if (old == null) {cleanAllEventsUpToThisPoint();}//(i.e. clear any buffers or maps containing data revolving around uncommited render thread data events)

        // this creates a lock free event update loop, allowing the render thread to never stall on waiting

        //TODO: NOTE: THIS MUST BE A SINGLE OBJECT THAT IS EXCHANGED
        // for it to be effectivly synchonized all outgoing events/effects _MUST_ happen at the same time
        // for this to be lock free an entire object containing ALL the events that must be synced must be exchanged


        //TODO: also note! this can be done for the processing of rendered out block models!!
        // (it might be able to also be put in this thread, maybe? but is proabably worth putting in own thread for latency reasons)
        if (this.needsWaitForSync) {
            while (RESULT_HANDLE.get(this) != null && this.running) {
                try {
                    Thread.sleep(10);
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }
            }
        }


        var prev = (SyncResults) RESULT_HANDLE.getAndSet(this, null);
        SyncResults results = null;
        if (prev == null) {
            this.needsWaitForSync = false;
            results = this.getMakeResultObject();
            //Clear old data (if it exists), create a new result set
            results.tlnDelta.addAll(this.tlnIdChange);
            this.tlnIdChange.clear();

            if (!this.geometryManager.getUploads().isEmpty()){//Put in new data into sync set
                var iter = this.geometryManager.getUploads().int2ObjectEntrySet().fastIterator();
                while (iter.hasNext()) {
                    var val = iter.next();
                    results.geometryUpload.upload(val.getIntKey(), val.getValue());
                    val.getValue().free();
                }
                this.geometryManager.getUploads().clear();
            }

            this.geometryManager.getHeapRemovals().clear();//We dont do removals on new data (as there is "none")
            results.cleanerOperations.addAll(this.cleanerIdResetClear); this.cleanerIdResetClear.clear();
        } else {
            results = prev;
            // merge with the previous result set

            if (!this.tlnIdChange.isEmpty()) {//Merge top level node id changes
                var iter = this.tlnIdChange.intIterator();
                while (iter.hasNext()) {
                    int val = iter.nextInt();
                    if (!results.tlnDelta.remove(val ^ (1 << 31))) {//Remove opposite
                        results.tlnDelta.add(val);//Add this if not added
                    }
                }
                this.tlnIdChange.clear();
            }

            if (!this.cleanerIdResetClear.isEmpty()) {//Merge top level node id changes
                var iter = this.cleanerIdResetClear.intIterator();
                while (iter.hasNext()) {
                    int val = iter.nextInt();
                    results.cleanerOperations.remove(val^(1<<31));//Remove opposite
                    results.cleanerOperations.add(val);//Add this
                }
                this.cleanerIdResetClear.clear();
            }

            if (!this.geometryManager.getHeapRemovals().isEmpty()) {//Remove and free all the removed geometry uploads
                var rem = this.geometryManager.getHeapRemovals();
                var iter = rem.intIterator();
                while (iter.hasNext()) {
                    results.geometryUpload.remove(iter.nextInt());
                }
                rem.clear();
            }

            if (!this.geometryManager.getUploads().isEmpty()) {//Add all the new uploads to the result set
                var add = this.geometryManager.getUploads();
                var iter = add.int2ObjectEntrySet().fastIterator();
                while (iter.hasNext()) {
                    var val = iter.next();
                    results.geometryUpload.upload(val.getIntKey(), val.getValue());
                    val.getValue().free();
                }
                add.clear();
            }
        }

        {//This is the same regardless of if is a merge or new result
            //Geometry id metadata updates
            if (!this.geometryManager.getUpdateIds().isEmpty()) {
                var ids = this.geometryManager.getUpdateIds();
                var iter = ids.intIterator();
                while (iter.hasNext()) {
                    int val = iter.nextInt();
                    int scatterAddr = (val<<1)|(1<<31);//Since we write to the second buffer

                    //Geometry buffer is index of 1, so mutate to put it in that location, it is also 32 bytes, so needs to be split into 2 separate scatter writes
                    long ptrA = results.getScatterWritePtr(scatterAddr+0, 1);
                    long ptrB = results.getScatterWritePtr(scatterAddr+1, 0);

                    //Write update data
                    this.geometryManager.writeMetadataSplit(val, ptrA, ptrB);

                    //Remember what was intended for this section so the GPU's scatter can be checked
                    //against it once it has run (VOXY_METACHK).
                    this.metachkStage(val, ptrA, ptrB);
                }
                ids.clear();
            }

            //Node updates
            if (!this.manager.getNodeUpdates().isEmpty()) {
                var ids = this.manager.getNodeUpdates();
                var iter = ids.intIterator();
                while (iter.hasNext()) {
                    int val = iter.nextInt();
                    //Dont need to modify the write location since we write to buffer 0
                    long ptr = results.getScatterWritePtr(val);
                    //Write updated data
                    this.manager.writeNode(val, ptr);
                }
                ids.clear();
            }
        }

        results.geometrySectionCount = this.geometryManager.getSectionCount();
        results.usedGeometry = this.geometryManager.getGeometryUsedBytes();
        results.currentMaxNodeId = this.manager.getCurrentMaxNodeId();

        this.needsWaitForSync |= results.geometryUpload.currentElemCopyAmount*8L > 2L<<20;//2mb limit per frame
        this.needsWaitForSync |= results.cleanerOperations.size() > 1024;
        this.needsWaitForSync |= results.scatterWriteLocationMap.size() > 4096;
        this.needsWaitForSync |= results.tlnDelta.size() > 10;

        if (!RESULT_HANDLE.compareAndSet(this, null, results)) {
            throw new IllegalArgumentException("Should always have null");
        }
    }

    private IntConsumer tlnAddCallback; private IntConsumer tlnRemoveCallback;
    //Render thread synchronization
    public void tick(IGpuBuffer nodeBuffer, NodeCleaner cleaner) {//TODO: dont pass nodeBuffer here??, do something else thats better
        this.renderTick++;
        var results = (SyncResults)RESULT_HANDLE.getAndSet(this, null);//Acquire the results
        if (results == null) {//There are no new results to process, return
            return;
        }
        DIAG_TICK_WITH_RESULTS_COUNT.incrementAndGet();
        DIAG_LAST_TICK_SECTION_COUNT.set(results.geometrySectionCount);
        if (results.geometryUpload != null && !results.geometryUpload.dataUploadPoints.isEmpty()) {
            DIAG_TICK_WITH_UPLOADS_COUNT.incrementAndGet();
        }

        //top level node add/remove
        if (!results.tlnDelta.isEmpty()) {
            var iter = results.tlnDelta.intIterator();
            while (iter.hasNext()) {
                int val = iter.nextInt();
                if ((val&(1<<31))!=0) {//Add node
                    this.tlnAddCallback.accept(val&(-1>>>1));
                } else {
                    this.tlnRemoveCallback.accept(val);
                }
            }
            //Dont need to clear as is not used again
        }

        {//Update basic geometry data
            var store = (BasicSectionGeometryData)this.geometryData;

            store.setSectionCount(results.geometrySectionCount);

            var upload = results.geometryUpload;
            if (!upload.dataUploadPoints.isEmpty()) {
                ((BasicSectionGeometryData)this.geometryData).ensureAccessable(upload.maxElementAccess);
                TimingStatistics.A.start();

                int copies = upload.dataUploadPoints.size();
                int scratchSize = (int) upload.arena.getSize() * 8;
                long ptr = UploadStream.INSTANCE.rawUploadAddress(scratchSize + copies * 16);
                UnsafeUtil.memcpy(upload.scratchHeaderBuffer.address, UploadStream.INSTANCE.getBaseAddress() + ptr, copies * 16L);
                UnsafeUtil.memcpy(upload.scratchDataBuffer.address, UploadStream.INSTANCE.getBaseAddress() + ptr + copies * 16L, scratchSize);
                UploadStream.INSTANCE.commit();//Commit the buffer

                if (copies > 500) {
                    Logger.warn("Large amount of copies, lag will probably happen: " + copies);
                }

                try (var encoder = this.backend.beginComputePass()) {
                    encoder.setPipeline(this.multiMemcpy);
                    // M12: UploadStream's persistent buffer flows through the
                    // encoder's IGpuPersistentBuffer overload now (was a raw
                    // glBindBufferRange against UploadStream.getRawBufferId()
                    // that broke on Metal because the buffer id isn't a GL name).
                    encoder.setBuffer(0, UploadStream.INSTANCE.getUploadBuffer(), ptr, copies * 16L);
                    encoder.setBuffer(1, UploadStream.INSTANCE.getUploadBuffer(), ptr + copies * 16L, scratchSize);
                    encoder.setBuffer(2, ((BasicSectionGeometryData) this.geometryData).getGeometryBuffer(), 0);

                    encoder.barrier(me.cortex.voxy.client.core.gpu.ComputeEncoder.BARRIER_SHADER, me.cortex.voxy.client.core.gpu.ComputeEncoder.BARRIER_SHADER);
                    encoder.dispatch(copies, 1, 1);
                    encoder.barrier(me.cortex.voxy.client.core.gpu.ComputeEncoder.BARRIER_SHADER, me.cortex.voxy.client.core.gpu.ComputeEncoder.BARRIER_SHADER);
                }

                TimingStatistics.A.stop();
            }
        }

        TimingStatistics.B.start();
        if (!results.scatterWriteLocationMap.isEmpty()) {//Scatter write
            int count = results.scatterWriteLocationMap.size();//Number of writes, not chunks or uvec4 count
            int chunks = (count+3)/4;
            int streamSize = chunks*80;//80 bytes per chunk, it is guaranteed the buffer is big enough
            long ptr = UploadStream.INSTANCE.rawUploadAddress(streamSize + 16);//Ensure it is 16 byte aligned
            ptr = (ptr+15L)&~0xFL;//Align up to 16 bytes
            MemoryUtil.memCopy(results.scatterWriteBuffer.address, UploadStream.INSTANCE.getBaseAddress() + ptr, streamSize);
            UploadStream.INSTANCE.commit();//Commit the buffer

            try (var encoder = this.backend.beginComputePass();
                 var stack = org.lwjgl.system.MemoryStack.stackPush()) {
                encoder.setPipeline(this.scatterWrite);
                // M12: cross-backend persistent-buffer bind — see multiMemcpy above.
                encoder.setBuffer(0, UploadStream.INSTANCE.getUploadBuffer(), ptr, streamSize);
                encoder.setBuffer(1, nodeBuffer, 0);
                encoder.setBuffer(2, ((BasicSectionGeometryData) this.geometryData).getMetadataBuffer(), 0);

                long pushAddr = stack.nmalloc(4);
                MemoryUtil.memPutInt(pushAddr, count);
                encoder.setBytes(SCATTER_PUSH_BINDING, pushAddr, 4);

                encoder.barrier(
                        me.cortex.voxy.client.core.gpu.ComputeEncoder.BARRIER_SHADER,
                        me.cortex.voxy.client.core.gpu.ComputeEncoder.BARRIER_SHADER);
                encoder.dispatch((count + 127) / 128, 1, 1);
                encoder.barrier(
                        me.cortex.voxy.client.core.gpu.ComputeEncoder.BARRIER_SHADER,
                        me.cortex.voxy.client.core.gpu.ComputeEncoder.BARRIER_SHADER);
            }
        }
        TimingStatistics.B.stop();

        TimingStatistics.C.start();
        if (!results.cleanerOperations.isEmpty()) {
            cleaner.updateIds(results.cleanerOperations);
        }
        TimingStatistics.C.stop();

        this.currentMaxNodeId = results.currentMaxNodeId;
        this.usedGeometryAmount = results.usedGeometry;

        //Insert the result set into the cache
        if (!RESULT_CACHE_1_HANDLE.compareAndSet(this, null, results)) {
            //Failed to insert into result set 1, insert it into result set 2
            if (!RESULT_CACHE_2_HANDLE.compareAndSet(this, null, results)) {
                throw new IllegalStateException("Could not insert result into cache");
            }
        }
    }


    public void setTLNAddRemoveCallbacks(IntConsumer add, IntConsumer remove) {
        this.tlnAddCallback = add;
        this.tlnRemoveCallback = remove;
    }

    private int currentMaxNodeId = 0;
    public int getCurrentMaxNodeId() {
        return this.currentMaxNodeId;
    }

    private long usedGeometryAmount = 0;
    public long getUsedGeometryCapacity() {
        return this.usedGeometryAmount;
    }

    /**
     * Render-thread, once per frame: hand geometry addresses freed by {@code removeSection} back to the
     * arena, once no in-flight command buffer can still be drawing from them.
     *
     * <p>Delegates to {@link BasicAsyncGeometryManager} rather than being implemented here, because that
     * is the class that owns the allocation arena and therefore the only one that can say when an
     * address is safe to reuse. The render thread is where it has to be driven from, since only it
     * knows the frame counter the delay is measured in.
     */
    public void releaseRetiredFrees(long frameId) {
        this.geometryManager.releaseRetiredFrees(frameId);
    }

    public long getGeometryCapacity() {
        return this.geometryCapacity;
    }


    //==================================================================================================================
    //Incoming events

    //TODO: add atomic counters for each event type probably
    private final ConcurrentLinkedDeque<MemoryBuffer> requestBatchQueue = new ConcurrentLinkedDeque<>();
    private final ConcurrentLinkedDeque<WorldSection> childUpdateQueue = new ConcurrentLinkedDeque<>();
    private final ConcurrentLinkedDeque<BuiltSection> geometryUpdateQueue = new ConcurrentLinkedDeque<>();

    private final ConcurrentLinkedDeque<MemoryBuffer> removeBatchQueue = new ConcurrentLinkedDeque<>();

    private final StampedLock tlnLock = new StampedLock();
    private final LongOpenHashSet tlnAdd = new LongOpenHashSet();
    private final LongOpenHashSet tlnRem = new LongOpenHashSet();

    private void addWork() {
        if (!this.running) throw new IllegalStateException("Not running");
        if (this.workCounter.getAndIncrement() == 0) {
            LockSupport.unpark(this.thread);
        }
    }

    public void submitRequestBatch(MemoryBuffer batch) {//Only called from render thread
        this.requestBatchQueue.add(batch);
        this.addWork();
    }

    private void submitChildChange(WorldSection section) {
        if (!this.running) {
            return;
        }
        section.acquire();//We must acquire the section before putting in the queue
        this.childUpdateQueue.add(section);
        this.addWork();
    }

    /** M13 diagnostic counters — read by AbstractRenderPipeline's Metal-DIAG dump. */
        /**
     * Node requests read back from the GPU and checked for plausibility before being acted on.
     *
     * <p>A request that survives the round trip in {@link #validSectionPos} decoded from the same
     * packing that produced it, so a nonzero {@code DIAG_REQ_INVALID} means the CPU read bytes the
     * traversal did not write -- corruption in flight rather than a wrong value generated. That is
     * the difference between "the node manager asked for garbage" and "the GPU told it garbage",
     * and it is the distinction the splotch investigation needs.
     */
    /**
     * Batches whose GPU-written count did not fit the slot the traversal allocated for it.
     *
     * <p>A count the traversal did not write is the readback returning garbage, which makes this the
     * one direct measurement of that happening. It also used to be fatal to the async node manager
     * thread (the throw escaped the run loop); it is now counted and skipped, so a session survives
     * one and the rate is visible.
     */
    public static final java.util.concurrent.atomic.AtomicLong DIAG_REQ_BATCH_OVERFLOW = new java.util.concurrent.atomic.AtomicLong();

    public static final java.util.concurrent.atomic.AtomicLong DIAG_REQ_INVALID = new java.util.concurrent.atomic.AtomicLong();
    public static final java.util.concurrent.atomic.AtomicLong DIAG_REQ_VALIDATED = new java.util.concurrent.atomic.AtomicLong();

    private static boolean validSectionPos(final long pos) {
        final int lvl = me.cortex.voxy.common.world.WorldEngine.getLevel(pos);
        if (lvl < 0 || lvl > me.cortex.voxy.common.world.WorldEngine.MAX_LOD_LAYER) return false;
        final int x = me.cortex.voxy.common.world.WorldEngine.getX(pos);
        final int y = me.cortex.voxy.common.world.WorldEngine.getY(pos);
        final int z = me.cortex.voxy.common.world.WorldEngine.getZ(pos);
        return me.cortex.voxy.common.world.WorldEngine.getWorldSectionId(lvl, x, y, z) == pos;
    }

public static final java.util.concurrent.atomic.AtomicLong DIAG_WORLD_EVENT_COUNT = new java.util.concurrent.atomic.AtomicLong();
    public static final java.util.concurrent.atomic.AtomicLong DIAG_GEOMETRY_RESULT_COUNT = new java.util.concurrent.atomic.AtomicLong();
    public static final java.util.concurrent.atomic.AtomicLong DIAG_TOP_LEVEL_ADD_COUNT = new java.util.concurrent.atomic.AtomicLong();
    /** Times AsyncNodeManager.tick() ran and consumed a non-null SyncResults. */
    public static final java.util.concurrent.atomic.AtomicLong DIAG_TICK_WITH_RESULTS_COUNT = new java.util.concurrent.atomic.AtomicLong();
    /** Highest geometrySectionCount observed by AsyncNodeManager.tick(). */
    public static final java.util.concurrent.atomic.AtomicLong DIAG_LAST_TICK_SECTION_COUNT = new java.util.concurrent.atomic.AtomicLong();
    /** Times AsyncNodeManager.tick() saw non-empty geometry uploads in the sync results. */
    public static final java.util.concurrent.atomic.AtomicLong DIAG_TICK_WITH_UPLOADS_COUNT = new java.util.concurrent.atomic.AtomicLong();

    private void submitGeometryResult(BuiltSection geometry) {
        DIAG_GEOMETRY_RESULT_COUNT.incrementAndGet();
        if (!this.running) {
            geometry.free();
            return;
        }
        this.geometryUpdateQueue.add(geometry);
        this.addWork();
    }

    public void submitRemoveBatch(MemoryBuffer batch) {//Only called from render thread
        this.removeBatchQueue.add(batch);
        this.addWork();
    }

    public void addTopLevel(long section) {//Only called from render thread
        DIAG_TOP_LEVEL_ADD_COUNT.incrementAndGet();
        if (!this.running) throw new IllegalStateException("Not running");
        long stamp = this.tlnLock.writeLock();
        int state = 0;
        if (!this.tlnRem.remove(section)) {
            state += this.tlnAdd.add(section)?1:0;
        } else {
            state -= 1;
        }
        if (state != 0) {
            if (this.workCounter.getAndAdd(state) == 0) {
                LockSupport.unpark(this.thread);
            }
        }
        this.tlnLock.unlockWrite(stamp);
    }

    public void removeTopLevel(long section) {//Only called from render thread
        if (!this.running) throw new IllegalStateException("Not running");
        long stamp = this.tlnLock.writeLock();
        int state = 0;
        if (!this.tlnAdd.remove(section)) {
            state += this.tlnRem.add(section)?1:0;
        } else {
            state -= 1;
        }
        if (state != 0) {
            if (this.workCounter.getAndAdd(state) == 0) {
                LockSupport.unpark(this.thread);
            }
        }
        this.tlnLock.unlockWrite(stamp);
    }

    //==================================================================================================================

    public void start() {
        this.thread.start();
    }

    public void stop() {
        if (!this.running) {
            throw new IllegalStateException();
        }
        this.running = false;
        LockSupport.unpark(this.thread);
        try {
            while (this.thread.isAlive()) {
                LockSupport.unpark(this.thread);
                this.thread.join(1000);
            }
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }

        while (true) {
            var buffer = this.requestBatchQueue.poll();
            if (buffer == null) break;
            buffer.free();
        }

        while (true) {
            var buffer = this.removeBatchQueue.poll();
            if (buffer == null) break;
            buffer.free();
        }

        while (true) {
            var buffer = this.geometryUpdateQueue.poll();
            if (buffer == null) break;
            buffer.free();
        }

        while (true) {
            var section = this.childUpdateQueue.poll();
            if (section == null) break;
            section.release();
        }

        if (RESULT_HANDLE.get(this) != null) {
            var result = (SyncResults)RESULT_HANDLE.getAndSet(this, null);
            result.geometryUpload.free();
            result.scatterWriteBuffer.free();
        }

        if (RESULT_CACHE_1_HANDLE.get(this) != null) {//Clear cache 1
            var result = (SyncResults)RESULT_CACHE_1_HANDLE.getAndSet(this, null);
            result.geometryUpload.free();
            result.scatterWriteBuffer.free();
        }

        if (RESULT_CACHE_2_HANDLE.get(this) != null) {//Clear cache 2
            var result = (SyncResults)RESULT_CACHE_2_HANDLE.getAndSet(this, null);
            result.geometryUpload.free();
            result.scatterWriteBuffer.free();
        }

        this.scatterWrite.close();
        this.multiMemcpy.close();
        this.geometryCache.free();
    }

    public void addDebug(List<String> debug) {
        debug.add("UC/GC: " + (this.getUsedGeometryCapacity()/(1<<20))+"/"+(this.getGeometryCapacity()/(1<<20)));
        //debug.add("GUQ/NRC: " + this.geometryUpdateQueue.size()+"/"+this.removeBatchQueue.size());
    }

    public boolean hasWork() {
        return this.workCounter.get()!=0 || RESULT_HANDLE.get(this) != null;
    }

    public void worldEvent(WorldSection section, int flags, int neighborMask) {
        DIAG_WORLD_EVENT_COUNT.incrementAndGet();
        //If there is any change, we need to clear the geometry cache before emitting update
        this.geometryCache.clear(section.key);

        this.router.forwardEvent(section, flags);

        if (neighborMask != 0) {//trigger rebuilds for neighbors
            if ((neighborMask&0b000001)!=0) this.router.triggerRemesh(WorldEngine.getWorldSectionId(section.lvl, section.x, section.y-1, section.z));//-y
            if ((neighborMask&0b000010)!=0) this.router.triggerRemesh(WorldEngine.getWorldSectionId(section.lvl, section.x, section.y+1, section.z));//+y
            if ((neighborMask&0b000100)!=0) this.router.triggerRemesh(WorldEngine.getWorldSectionId(section.lvl, section.x-1, section.y, section.z));//-x
            if ((neighborMask&0b001000)!=0) this.router.triggerRemesh(WorldEngine.getWorldSectionId(section.lvl, section.x+1, section.y, section.z));//+x
            if ((neighborMask&0b010000)!=0) this.router.triggerRemesh(WorldEngine.getWorldSectionId(section.lvl, section.x, section.y, section.z-1));//-z
            if ((neighborMask&0b100000)!=0) this.router.triggerRemesh(WorldEngine.getWorldSectionId(section.lvl, section.x, section.y, section.z+1));//+z
        }
    }

    //Results object, which is to be synced between the render thread and worker thread
    private static final class SyncResults {
        //Contains
        // geometry uploads and id invalidations and the data
        // node ids to invalidate/update and its data
        // top level node ids to add/remove
        // cleaner move and set operations

        //Node id updates + size
        private int currentMaxNodeId;// the id of the ending of the node ids

        //TLN add/rem
        private final IntOpenHashSet tlnDelta = new IntOpenHashSet();

        //Deltas for geometry store
        private int geometrySectionCount;
        private long usedGeometry;
        private final ComputeMemoryCopy geometryUpload = new ComputeMemoryCopy();

        //Gpu geometry downloads



        //Scatter writes for both geometry and node metadata
        private MemoryBuffer scatterWriteBuffer = new MemoryBuffer(8192*2);
        private final Int2IntOpenHashMap scatterWriteLocationMap = new Int2IntOpenHashMap(1024);
        {this.scatterWriteLocationMap.defaultReturnValue(-1);}

        //Cleaner operations
        private final IntOpenHashSet cleanerOperations = new IntOpenHashSet();

        public void reset() {
            this.cleanerOperations.clear();
            this.scatterWriteLocationMap.clear();
            this.currentMaxNodeId = 0;
            this.tlnDelta.clear();
            this.geometrySectionCount = 0;
            this.usedGeometry = 0;
            this.geometryUpload.reset();
        }

        //Get or create a scatter write address for the given location
        public long getScatterWritePtr(int location) {
            return this.getScatterWritePtr(location, 0);
        }

        //ensureExtra is used to ensure that allocations are "effectivly" in the same memory block (kinda?)
        public long getScatterWritePtr(int location, int ensureExtra) {
            int loc = this.scatterWriteLocationMap.get(location);
            if (loc == -1) {//Location doesnt exist, create it
                this.ensureScatterBufferCapacity(1+ensureExtra);//Ensure can contain capacity for this + extra
                int baseId = this.scatterWriteLocationMap.size();
                int chunkBase = (baseId/4)*5;//Base uvec4 index
                int innerId   = baseId&3;
                MemoryUtil.memPutInt(this.scatterWriteBuffer.address + (chunkBase*16L) + (innerId*4L), location);//Set the write location
                int writeLocation = (chunkBase+1+innerId);//Write location in uvec4
                this.scatterWriteLocationMap.put(location, writeLocation);
                return this.scatterWriteBuffer.address + (writeLocation*16L);
            } else {
                return this.scatterWriteBuffer.address + (16L*loc);
            }
        }

        private void ensureScatterBufferCapacity(int extra) {
            int requiredChunks = ((this.scatterWriteLocationMap.size()+extra)+3)/4;//4 entries in a chunk
            long requiredSize = requiredChunks*5L*16L;//5 uvec4 per chunk, 16 bytes per uvec4
            if (this.scatterWriteBuffer.size <= requiredSize) {//Needs resize
                long newSize = (long) ((this.scatterWriteBuffer.size*1.5) + extra*80L);
                newSize = ((newSize+79)/80)*80;//Ceil to chunk size

                Logger.info("Expanding scatter update buffer to " + newSize);

                var newBuffer = new MemoryBuffer(newSize);
                this.scatterWriteBuffer.cpyTo(newBuffer.address);
                this.scatterWriteBuffer.free();
                this.scatterWriteBuffer = newBuffer;
            }
        }
    }

    private static class ComputeMemoryCopy {
        public int currentElemCopyAmount;
        public int maxElementAccess;
        private MemoryBuffer scratchHeaderBuffer = new MemoryBuffer(1<<16);
        private MemoryBuffer scratchDataBuffer = new MemoryBuffer(1<<20);

        private final AllocationArena arena = new AllocationArena();
        private final Int2IntOpenHashMap dataUploadPoints = new Int2IntOpenHashMap();//Points to the header index
        {this.dataUploadPoints.defaultReturnValue(-1);}


        public void remove(int point) {
            int header = this.dataUploadPoints.remove(point);
            if (header == -1) {//No upload for point
                return;
            }
            int size = MemoryUtil.memGetInt(this.scratchHeaderBuffer.address + header*16L + 8L);
            this.currentElemCopyAmount -= size;
            //Free the old memory addr from arena
            if (this.arena.free(MemoryUtil.memGetInt(this.scratchHeaderBuffer.address + header*16L)) != size) {
                throw new IllegalStateException("Freed memory not same size as expected");
            }
            if (MemoryUtil.memGetInt(this.scratchHeaderBuffer.address + header*16L + 4L) != point) {
                throw new IllegalStateException("Destination not the same as point");
            }

            //If we were the end upload header, return as we dont need to shuffle
            if (header == this.dataUploadPoints.size()) {
                long A = this.scratchHeaderBuffer.address + header*16L;
                //Zero the memory, for consistancy
                MemoryUtil.memPutLong(A, 0);
                MemoryUtil.memPutLong(A+8, 0);
                return;
            }

            //Else: we need to move the ending upload header from the end to where the freed point was
            int endingPoint = MemoryUtil.memGetInt(this.scratchHeaderBuffer.address + this.dataUploadPoints.size()*16L + 4);
            if (this.dataUploadPoints.get(endingPoint) != this.dataUploadPoints.size()) {
                throw new IllegalStateException("ending header not pointing at end point");
            }

            //Move the end header to the old header location
            long A = this.scratchHeaderBuffer.address + this.dataUploadPoints.size()*16L;
            long B = this.scratchHeaderBuffer.address + header*16L;
            MemoryUtil.memPutLong(B, MemoryUtil.memGetLong(A)); MemoryUtil.memPutLong(A, 0);
            MemoryUtil.memPutLong(B+8, MemoryUtil.memGetLong(A+8)); MemoryUtil.memPutLong(A+8, 0);

            //Update the map
            this.dataUploadPoints.put(endingPoint, header);
        }

        public void upload(int point, MemoryBuffer data) {
            if ((data.size%8)!=0) throw new IllegalStateException("Data must be of size multiple 8");
            int elemSize = (int) (data.size / 8);
            this.maxElementAccess = Math.max(this.maxElementAccess, point + elemSize);
            int header = this.dataUploadPoints.get(point);
            if (header != -1) {
                //If we already have a header location, we just need to reallocate the data
                long headerPtr = this.scratchHeaderBuffer.address + header*16L;
                if (MemoryUtil.memGetInt(headerPtr+4L) != point) {
                    throw new IllegalStateException("Existing destination not the point");
                }
                int pSize = MemoryUtil.memGetInt(headerPtr+8L);//Previous size
                if (pSize == elemSize) {
                    //The data we are replacing is the same size, so just overwrite it, this is the easiest
                    data.cpyTo(this.scratchDataBuffer.address+MemoryUtil.memGetInt(headerPtr)*8L);
                } else {
                    //Dealloc
                    if (this.arena.free(MemoryUtil.memGetInt(headerPtr)) != pSize) {
                        throw new IllegalStateException("Freed allocation not size as expected");
                    }

                    this.currentElemCopyAmount -= pSize;
                    this.currentElemCopyAmount += elemSize;

                    int alloc = this.allocScratchDataPos(elemSize);//New allocation position
                    //Copy data into position
                    data.cpyTo(this.scratchDataBuffer.address+alloc*8L);

                    //Update the header
                    MemoryUtil.memPutInt(headerPtr, alloc);
                    MemoryUtil.memPutInt(headerPtr+8, elemSize);
                }
            } else {
                //We need to create and allocate a new header for the upload
                header = this.dataUploadPoints.size();
                this.dataUploadPoints.put(point, header);

                if (this.scratchHeaderBuffer.size<=header*16L) {
                    //We must resize the header buffer
                    long newSize = Math.max(this.scratchHeaderBuffer.size*2, header*16L);
                    Logger.info("Resizing scratch header buffer to: " + newSize);
                    var newScratch = new MemoryBuffer(newSize);
                    this.scratchHeaderBuffer.cpyTo(newScratch.address);
                    this.scratchHeaderBuffer.free();
                    this.scratchHeaderBuffer = newScratch;
                }

                long headerPtr = this.scratchHeaderBuffer.address + header*16L;//Header resize has happened so this is a stable address

                this.currentElemCopyAmount += elemSize;

                int alloc = this.allocScratchDataPos(elemSize);//New allocation position
                //Copy data into position
                data.cpyTo(this.scratchDataBuffer.address+alloc*8L);

                //Set header data
                MemoryUtil.memPutInt(headerPtr, alloc);
                MemoryUtil.memPutInt(headerPtr+4, point);
                MemoryUtil.memPutInt(headerPtr+8, elemSize);
            }
        }

        //This is done here as it enables easily doing scratch data resizing
        private int allocScratchDataPos(int size) {
            int pos = (int) this.arena.alloc(size);
            if (this.scratchDataBuffer.size <= (pos+size)*8L) {
                //We must resize :cri:
                long newSize = Math.max(this.scratchDataBuffer.size*2, (pos+size)*8L);
                Logger.info("Resizing scratch data buffer to: " + newSize);
                var newScratch = new MemoryBuffer(newSize);
                this.scratchDataBuffer.cpyTo(newScratch.address);
                this.scratchDataBuffer.free();
                this.scratchDataBuffer = newScratch;
            }
            return pos;
        }

        public void reset() {
            this.maxElementAccess = 0;
            this.currentElemCopyAmount = 0;
            this.dataUploadPoints.clear();
            this.arena.reset();
        }

        public void free() {
            this.scratchHeaderBuffer.free(); this.scratchHeaderBuffer = null;
            this.scratchDataBuffer.free(); this.scratchDataBuffer = null;
        }
    }
}
