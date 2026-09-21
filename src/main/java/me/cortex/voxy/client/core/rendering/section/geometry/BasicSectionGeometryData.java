package me.cortex.voxy.client.core.rendering.section.geometry;

import me.cortex.voxy.client.core.gpu.Capabilities;
import me.cortex.voxy.client.core.gpu.IGpuBuffer;
import me.cortex.voxy.client.core.gpu.RenderBackendFactory;
import me.cortex.voxy.common.Logger;

public class BasicSectionGeometryData implements IGeometryData {
    public static final int SECTION_METADATA_SIZE = 32;
    private final IGpuBuffer sectionMetadataBuffer;
    private final IGpuBuffer geometryBuffer;

    private final int maxSectionCount;
    private int currentSectionCount;

    public BasicSectionGeometryData(int maxSectionCount, long geometryCapacity) {
        this.maxSectionCount = maxSectionCount;
        this.sectionMetadataBuffer = RenderBackendFactory.get().createBuffer((long) maxSectionCount * SECTION_METADATA_SIZE);
        //8 Cause a quad is 8 bytes
        if ((geometryCapacity%8)!=0) {
            throw new IllegalStateException();
        }
        long start = System.currentTimeMillis();
        String msg = "Creating and zeroing " + (geometryCapacity/(1024*1024)) + "MB geometry buffer";
        if (Capabilities.INSTANCE.canQueryGpuMemory) {
            msg += " driver states " + (Capabilities.INSTANCE.getFreeDedicatedGpuMemory()/(1024*1024)) + "MB of free memory";
        }
        Logger.info(msg);
        Logger.info("if your game crashes/exits here without any other log message, try manually decreasing the geometry capacity");
        IGpuBuffer buffer = null;
        if (!(Capabilities.INSTANCE.isNvidia)) {// && ThreadUtils.isWindows
            buffer = RenderBackendFactory.get().createBuffer(geometryCapacity, 0, false);//Only do this if we are not on nvidia
            //TODO: FIXME: TEST, see if the issue is that we are trying to zero the entire buffer, try only zeroing increments
            // or dont zero it at all
        } else {
            Logger.info("Running on nvidia, using workaround sparse buffer allocation");
        }
        if (buffer == null) {
            throw new IllegalStateException("Unable to allocate geometry buffer");
        }
        this.geometryBuffer = buffer;
        Logger.info("Successfully allocated the geometry buffer in "
                + (System.currentTimeMillis() - start) + "ms");
    }

    /** No-op on Metal: buffers are always fully committed, there is no sparse range to page in. */
    public void ensureAccessable(int maxElementAccess) {
    }

    public IGpuBuffer getGeometryBuffer() {
        return this.geometryBuffer;
    }

    public IGpuBuffer getMetadataBuffer() {
        return this.sectionMetadataBuffer;
    }

    public int getSectionCount() {
        return this.currentSectionCount;
    }

    public void setSectionCount(int count) {
        this.currentSectionCount = count;
    }

    public int getMaxSectionCount() {
        return this.maxSectionCount;
    }

    public long getGeometryCapacityBytes() {//In bytes
        return this.geometryBuffer.size();
    }

    @Override
    public void free() {
        this.sectionMetadataBuffer.free();
        // The wait-for-release poll that used to sit here (glFinish + getFreeDedicatedGpuMemory, up
        // to 2.5s) is deleted: glFinish does not exist on Metal, and the loop never ran a single
        // iteration anyway -- its condition `elapsed > TIMEOUT` is false on entry because elapsed
        // starts at 0.
        this.geometryBuffer.free();
    }
}
