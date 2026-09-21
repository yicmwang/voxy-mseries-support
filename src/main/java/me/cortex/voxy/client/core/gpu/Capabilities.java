package me.cortex.voxy.client.core.gpu;

public class Capabilities {

    public static final Capabilities INSTANCE = new Capabilities();

    public final boolean repFragTest;
    public final boolean meshShaders;
    public final boolean INT64_t;
    public final long ssboMaxSize;
    public final boolean isMesa;
    public final boolean canQueryGpuMemory;
    public final long totalDedicatedMemory;//Bytes, dedicated memory
    public final long totalDynamicMemory;//Bytes, total allocation memory - dedicated memory
    public final boolean compute;
    public final boolean indirectParameters;
    public final boolean indirectCount;
    public final boolean isIntel;
    public final boolean subgroup;
    public final boolean sparseBuffer;
    public final boolean isNvidia;
    public final boolean isAmd;
    public final boolean nvBarryCoords;
    public final boolean hasBrokenDepthSampler;

    /**
     * Whole-frame Metal has NO GL context, so there is nothing left to probe
     * here. The Metal backend reports its own capabilities through
     * RenderBackendFactory / RenderBackend; every value below is the neutral
     * set the no-GL-context branch used to produce.
     */
    public Capabilities() {
        this.sparseBuffer = false;
        this.compute = false;
        this.indirectCount = false;
        this.indirectParameters = false;
        this.repFragTest = false;
        this.meshShaders = false;
        this.canQueryGpuMemory = false;
        this.INT64_t = false;
        // Selects the non-subgroup prefix-sum shader. Safe, and the right choice when we
        // cannot prove subgroup support.
        this.subgroup = false;
        // Buffer sizing reads this; Metal's SSBOs are large, so give it a generous bound
        // rather than 0, which would size allocations to nothing.
        this.ssboMaxSize = 1L << 30;
        this.isMesa = false;
        this.isIntel = false;
        this.isNvidia = false;
        this.isAmd = false;
        this.totalDedicatedMemory = -1;
        this.totalDynamicMemory = -1;
        this.nvBarryCoords = false;
        this.hasBrokenDepthSampler = false;
    }

    public static void init() {
    }

    public long getFreeDedicatedGpuMemory() {
        // Always unreachable: canQueryGpuMemory is false, and every caller guards on it.
        throw new IllegalStateException("Cannot query gpu memory, missing extension");
    }

    //TODO: add gpu eviction tracking
}
