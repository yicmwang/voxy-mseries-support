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
        // Selects the prefix-sum shader: `util/prefixsum/inital3.comp` (subgroup) when true,
        // `util/prefixsum/simple.comp` (shared-memory Hillis-Steele) when false.
        //
        // The port set this to false with the note "safe, and the right choice when we cannot prove
        // subgroup support". That reasoning was sound when written, but it is no longer the state of
        // the evidence: `hiz/hiz.comp` already requires GL_KHR_shader_subgroup_arithmetic/basic/clustered
        // and uses subgroupMax, subgroupClusteredMax and subgroupBarrier, and it feeds the Hi-Z pyramid
        // the occlusion cull depends on -- a cull that is verified working. So subgroup ops already
        // compile AND execute correctly through this SPIR-V -> MSL path. Flagging the whole GPU as
        // lacking them, on every launch, is a claim this build contradicts.
        //
        // NOW ON by default, and the verification was not assumed. The prefix sum feeds
        // buildtranslucents.comp, so a wrong result misplaces translucent terrain and the spin-test
        // screenshot diff sees it. Against a same-config same-machine control -- two runs of the
        // fallback, which is the only thing that establishes this configuration's noise floor -- the
        // subgroup arm is IDENTICAL: median mean-abs-diff 2.32 / 6.79% of pixels against the control's
        // 2.35 / 6.74%. (An earlier reading called this "outside tolerance" purely because the tolerance
        // in tools/shots_diff.py is calibrated on the FLAT_FRAG configuration, whose floor is ~0.4;
        // real-colour runs at this sub_division_size sit at ~2.3. The arm was fine and the yardstick was
        // wrong.)
        //
        // VOXY_SUBGROUP=0 restores the fallback.
        this.subgroup = !"0".equals(System.getenv("VOXY_SUBGROUP"));
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
