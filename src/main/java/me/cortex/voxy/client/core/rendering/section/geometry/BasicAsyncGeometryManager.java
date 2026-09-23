package me.cortex.voxy.client.core.rendering.section.geometry;

import it.unimi.dsi.fastutil.ints.Int2ObjectOpenHashMap;
import it.unimi.dsi.fastutil.ints.IntOpenHashSet;
import it.unimi.dsi.fastutil.objects.ObjectArrayList;
import me.cortex.voxy.client.core.rendering.building.BuiltSection;
import me.cortex.voxy.common.util.AllocationArena;
import me.cortex.voxy.common.util.HierarchicalBitSet;
import me.cortex.voxy.common.util.MemoryBuffer;
import org.lwjgl.system.MemoryUtil;

import java.util.function.Consumer;


//Is basicly the manager for an "undefined" data store, the underlying store is irrelevant
// this manager serves as an overlay, that is, it allows an implementation to do "async management" of the data store
public class BasicAsyncGeometryManager implements IGeometryManager {
    public static final int SECTION_METADATA_SIZE = 32;

    /**
     * Bytes per geometry element — <b>the quad record size, and it must track {@code
     * RenderDataFactory.QUAD_BYTES}</b>.
     *
     * <p>It has two jobs and both are the quad record. As the <b>divisor</b> in {@code createMeta}, it
     * turns a section's uploaded byte size into its quad count ({@code itemCount}); as the <b>address
     * unit</b> of {@code allocationHeap}, it scales the arena pointer that the shaders treat as a quad
     * index ({@code cmdgen.comp} writes {@code baseVertex = offset*4}, and the vertex stage reads
     * {@code quadData[baseVertex>>2]} at a 16-byte stride).
     *
     * <p>Leaving it at 8 when the record grew to 16 doubled every section's quad count — {@code
     * itemCount = bytes/8 = 2*quads}, packed into {@code meta.b} and summed by cmdgen as each group's
     * quad count, so {@code cmd.count = quadCount*6} came out 2x — and simultaneously scaled the arena
     * address against a 16-byte stride. Measured as {@code indices} 17360118 against 8962794 at a
     * matched draw list. One constant, both symptoms.
     */
    private static final long GEOMETRY_ELEMENT_SIZE =
            me.cortex.voxy.client.core.rendering.building.RenderDataFactory.QUAD_BYTES;
    private final HierarchicalBitSet allocationSet;
    private final AllocationArena allocationHeap = new AllocationArena();
    private final ObjectArrayList<SectionMeta> sectionMetadata = new ObjectArrayList<>(1<<15);

    //Changes that need to be applied to the underlying data store to match this state
    private final IntOpenHashSet invalidatedIds = new IntOpenHashSet(1024);//Ids that need to be invalidated
    //TODO: maybe change from it pointing to MemoryBuffer, to BuiltSection
    //Note!: the int part is an unsigned int ptr, must be scaled by GEOMETRY_ELEMENT_SIZE
    private final Int2ObjectOpenHashMap<MemoryBuffer> heapUploads = new Int2ObjectOpenHashMap<>(1024);//Uploads into the buffer at the given location
    private final IntOpenHashSet heapRemoveUploads = new IntOpenHashSet(1024);//Any removals are added here, so that it can be properly synced
    private long usedCapacity = 0;

    public BasicAsyncGeometryManager(int maxSectionCount, long geometryCapacity) {
        this.allocationSet = new HierarchicalBitSet(maxSectionCount);
        if (geometryCapacity%GEOMETRY_ELEMENT_SIZE != 0)  throw new IllegalStateException();
        this.allocationHeap.setLimit(geometryCapacity/GEOMETRY_ELEMENT_SIZE);
    }

    @Override
    public int uploadSection(BuiltSection section) {
        return this.uploadReplaceSection(-1, section);
    }

    @Override
    public int uploadReplaceSection(int oldId, BuiltSection section) {
        if (section.isEmpty()) {
            throw new IllegalArgumentException("sectionData is empty, cannot upload nothing");
        }

        //Free the old id and replace it with a new one
        // if oldId is -1, then treat it as not previously existing

        //Free the old data if oldId is supplied
        if (oldId != -1) {
            //Its here just for future optimization potential
            this.removeSection(oldId);
        }

        int newId =  this.allocationSet.allocateNext();
        if (newId == HierarchicalBitSet.SET_FULL) {
            throw new IllegalStateException("Tried adding section when section count is already at capacity");
        }
        if (newId > this.sectionMetadata.size()) {
            throw new IllegalStateException("Size exceeds limits: " + newId + ", " + this.sectionMetadata.size() + ", " + this.allocationSet.getCount());
        }

        if (newId < this.sectionMetadata.size()) {
            if (this.sectionMetadata.get(newId) != null) {
                throw new IllegalStateException();
            }
        }

        var newMeta = this.createMeta(section);

        if (newId == this.sectionMetadata.size()) {
            this.sectionMetadata.add(newMeta);
        } else {
            if (this.sectionMetadata.set(newId, newMeta) != null) {
                throw new IllegalStateException();
            }
        }

        //Invalidate the section id
        this.invalidatedIds.add(newId);

        //HierarchicalOcclusionTraverser.HACKY_SECTION_COUNT = this.allocationSet.getCount();
        return newId;
    }

    @Override
    public void removeSection(int id) {
        if (!this.allocationSet.free(id)) {
            throw new IllegalStateException("Id was not already allocated. id: " + id);
        }
        var oldMetadata = this.sectionMetadata.set(id, null);
        int ptr = oldMetadata.geometryPtr;
        //Free from the heap -- but NOT immediately. See releaseRetiredFrees.
        if (GEOM_FREE_DELAY_FRAMES <= 0) {
            this.usedCapacity -= this.allocationHeap.free(Integer.toUnsignedLong(ptr));
        } else {
            this.pendingFrees.addLast(new long[]{Integer.toUnsignedLong(ptr), this.lastFrame});
        }
        //Free the upload if it was uploading
        var buf = this.heapUploads.remove(ptr);
        if (buf != null) {
            buf.free();
        }
        this.heapRemoveUploads.add(ptr);
        this.invalidatedIds.add(id);
    }

    /**
     * Geometry addresses freed by {@link #removeSection}, held until no in-flight command buffer can
     * still be reading them. Each entry is {@code {address, frameIdAtFreeTime}}.
     *
     * <p>Upstream frees into the arena immediately, and that is safe in GL: a draw, the geometry copy
     * and the metadata write are all commands in ONE ordered stream, so a reuse necessarily appears
     * after the draws that referenced the old occupant. This port runs whole-frame Metal with
     * {@code MAX_SUBMITS_IN_FLIGHT} command buffers in flight, so a buffer that has not executed yet can
     * still be reading an address that has since been handed to a different section -- and the GPU then
     * draws the NEW occupant's quads under the OLD section's command and position: real quads, with
     * their own baked light, at coordinates that do not belong there.
     *
     * <p>That is the shape of bug 3, and it also explains why the artefact's rate tracks the render
     * thread's CPU load rather than anything geometric: the faster the render thread runs, the further
     * ahead it gets of the GPU, and the more often a free-and-reuse lands inside that window.
     *
     * <p>{@code VOXY_GEOM_FREE_DELAY_FRAMES} is the number of frames an address is held before it
     * returns to the arena. 0 restores upstream's immediate free, so an unset run is the control. The
     * value only has to exceed the number of command buffers that can be in flight, plus the CPU's own
     * lead; the renderer's frame counter ages the entries, so this costs a bounded amount of memory
     * rather than leaking. A fence would express the same thing exactly, and is the right end state --
     * this is deliberately the cheap version, to find out whether the mechanism is real before paying
     * for the rigorous one.
     */
    private final java.util.ArrayDeque<long[]> pendingFrees = new java.util.ArrayDeque<>();

    private static final int GEOM_FREE_DELAY_FRAMES = parseGeomFreeDelayFrames();

    private static int parseGeomFreeDelayFrames() {
        String v = System.getenv("VOXY_GEOM_FREE_DELAY_FRAMES");
        if (v == null || v.isBlank()) return 0;
        try {
            return Math.max(0, Integer.parseInt(v.trim()));
        } catch (NumberFormatException e) {
            return 0;
        }
    }

    /** Advanced by the render thread in {@link #releaseRetiredFrees}; read by the async thread to stamp. */
    private volatile long lastFrame = -1;

    @Override
    public void releaseRetiredFrees(long frameId) {
        this.lastFrame = frameId;
        if (GEOM_FREE_DELAY_FRAMES <= 0) return;
        while (!this.pendingFrees.isEmpty()
                && frameId - this.pendingFrees.peekFirst()[1] >= GEOM_FREE_DELAY_FRAMES) {
            this.usedCapacity -= this.allocationHeap.free(this.pendingFrees.pollFirst()[0]);
        }
    }

    private SectionMeta createMeta(BuiltSection section) {
        if ((section.geometryBuffer.size%GEOMETRY_ELEMENT_SIZE)!=0) throw new IllegalStateException();
        int size = (int) (section.geometryBuffer.size/GEOMETRY_ELEMENT_SIZE);
        //clamp size upwards
        int upsized = (size+1023)&~1023;
        //Address
        int addr = (int)this.allocationHeap.alloc(upsized);
        if (addr == -1) {
            throw new IllegalStateException("Geometry OOM. requested allocation size (in elements): " + size + ", Heap size at top remaining: " + (this.allocationHeap.getLimit()-this.allocationHeap.getSize()) + ", used elements: " + this.usedCapacity);
        }
        this.usedCapacity += upsized;
        //Create upload
        if (this.heapUploads.put(addr, section.geometryBuffer) != null) {
            throw new IllegalStateException("Addr: " + addr);
        }
        this.heapRemoveUploads.remove(addr);
        //Create Meta
        return new SectionMeta(section.position, section.aabb, addr, size, section.offsets, section.childExistence);
    }

    @Override
    public void downloadAndRemove(int id, Consumer<BuiltSection> callback) {
        throw new IllegalStateException("Not yet implemented");
    }

    public Int2ObjectOpenHashMap<MemoryBuffer> getUploads() {
        return this.heapUploads;
    }

    public IntOpenHashSet getHeapRemovals() {
        return this.heapRemoveUploads;
    }

    public int getSectionCount() {
        return this.allocationSet.getCount();
    }

    public long getGeometryUsedBytes() {
        return this.usedCapacity * GEOMETRY_ELEMENT_SIZE;
    }

    public IntOpenHashSet getUpdateIds() {
        return this.invalidatedIds;
    }

    public void writeMetadata(int sectionId, long ptr) {
        var sec = this.sectionMetadata.get(sectionId);
        if (sec == null) {
            //Write nothing
            MemoryUtil.memSet(ptr, 0, SECTION_METADATA_SIZE);
        } else {
            sec.writeMetadata(ptr);
        }
    }

    public void writeMetadataSplit(int sectionId, long ptrA, long ptrB) {
        if (SECTION_METADATA_SIZE != 32) {//This system only works with 32 byte metadata
            throw new IllegalStateException();
        }
        var sec = this.sectionMetadata.get(sectionId);
        if (sec == null) {
            //Write nothing
            MemoryUtil.memSet(ptrA, 0, 16);
            MemoryUtil.memSet(ptrB, 0, 16);
        } else {
            sec.writeMetadataSplitParts(ptrA, ptrB);
        }
    }

    private record SectionMeta(long position, int aabb, int geometryPtr, int itemCount, int[] offsets, byte childExistence) {
        public void writeMetadata(long ptr) {
            this.writeMetadataSplitParts(ptr, ptr+16);
        }

        public void writeMetadataSplitParts(long ptrA, long ptrB) {//First 16 bytes are put into ptrA the remaining 16 bytes are put into ptrB
            //Split the long into 2 ints to solve endian issues
            MemoryUtil.memPutInt(ptrA, (int) (this.position>>32)); ptrA += 4;
            MemoryUtil.memPutInt(ptrA, (int) this.position); ptrA += 4;
            MemoryUtil.memPutInt(ptrA, (int) this.aabb); ptrA += 4;
            MemoryUtil.memPutInt(ptrA, this.geometryPtr + this.offsets[0]); ptrA += 4;

            MemoryUtil.memPutInt(ptrB, (this.offsets[1]-this.offsets[0])|((this.offsets[2]-this.offsets[1])<<16)); ptrB += 4;
            MemoryUtil.memPutInt(ptrB, (this.offsets[3]-this.offsets[2])|((this.offsets[4]-this.offsets[3])<<16)); ptrB += 4;
            MemoryUtil.memPutInt(ptrB, (this.offsets[5]-this.offsets[4])|((this.offsets[6]-this.offsets[5])<<16)); ptrB += 4;
            MemoryUtil.memPutInt(ptrB, (this.offsets[7]-this.offsets[6])|((this.itemCount -this.offsets[7])<<16)); ptrB += 4;
        }
    }
}
