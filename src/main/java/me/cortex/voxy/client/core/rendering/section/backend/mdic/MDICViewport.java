package me.cortex.voxy.client.core.rendering.section.backend.mdic;

import me.cortex.voxy.client.core.gpu.IGpuBuffer;
import me.cortex.voxy.client.core.gpu.RenderBackendFactory;
import me.cortex.voxy.client.core.rendering.Viewport;
import me.cortex.voxy.client.core.rendering.hierachical.HierarchicalOcclusionTraverser;

public class MDICViewport extends Viewport<MDICViewport> {
    public final IGpuBuffer drawCountCallBuffer = RenderBackendFactory.get().createBuffer(1024).zero();
    public final IGpuBuffer drawCallBuffer = RenderBackendFactory.get().createBuffer(5*4*(400_000+100_000+100_000)).zero();//400k draw calls
    public final IGpuBuffer positionScratchBuffer  = RenderBackendFactory.get().createBuffer(8*400000).zero();//400k positions
    /**
     * Zeroed like its siblings above. It was not, and its FIRST UINT IS NOT DATA -- it is the render
     * list's length, which `prep.comp` turns into a dispatch size and every consumer of the list uses
     * as an array bound. An un-zeroed allocation therefore hands the first frames a garbage bound and
     * the draws walk a garbage range of section ids. The sibling buffers got `.zero()` and this one
     * was missed, which is the kind of asymmetry that reads as intentional when it is not.
     */
    public final IGpuBuffer indirectLookupBuffer = RenderBackendFactory.get().createBuffer(HierarchicalOcclusionTraverser.MAX_QUEUE_SIZE *4+4).zero();//In theory, this could be global/not unique to the viewport
    public final IGpuBuffer visibilityBuffer;

    public MDICViewport(int maxSectionCount) {
        // Zeroed for the same reason as indirectLookupBuffer: `cmdgen` gates every draw on
        // `(visibilityData[sid] & 0x7fffffff) == frameId`, so an un-zeroed entry that happens to
        // equal the current frame id makes a section render on evidence that was never written.
        this.visibilityBuffer = RenderBackendFactory.get().createBuffer(maxSectionCount*4L).zero();
    }

    @Override
    protected void delete0() {
        super.delete0();
        this.visibilityBuffer.free();
        this.indirectLookupBuffer.free();
        this.drawCountCallBuffer.free();
        this.drawCallBuffer.free();
        this.positionScratchBuffer.free();
    }

    @Override
    public IGpuBuffer getRenderList() {
        return this.indirectLookupBuffer;
    }
}
