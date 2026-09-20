package me.cortex.voxy.client.core.rendering.section.geometry;

import me.cortex.voxy.client.core.gpu.GlCompat;

import me.cortex.voxy.client.core.gl.Capabilities;
import me.cortex.voxy.client.core.gpu.IGpuBuffer;
import me.cortex.voxy.client.core.gpu.RenderBackendFactory;
import me.cortex.voxy.common.Logger;

import static org.lwjgl.opengl.ARBSparseBuffer.*;
import static org.lwjgl.opengl.GL11C.*;
import static org.lwjgl.opengl.GL15C.GL_ARRAY_BUFFER;
import static org.lwjgl.opengl.GL15C.glBindBuffer;

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
        // The allocation is backend-agnostic; only the error reporting around it is GL-specific.
        // Calling glGetError with no GL context aborts the JVM, and sparse-buffer commitment is a
        // GL extension Metal has no equivalent for (its buffers are always fully committed).
        boolean glBackend = RenderBackendFactory.get().getType()
                == me.cortex.voxy.client.core.gpu.BackendType.OPENGL;

        if (glBackend) {
            glGetError();//Clear any errors
        }
        IGpuBuffer buffer = null;
        if (!(Capabilities.INSTANCE.isNvidia)) {// && ThreadUtils.isWindows
            // THIS IS THE ONLY BUFFER IN THE RENDERER CREATED UN-ZEROED. Every other allocation goes
            // through RenderBackend.createBuffer, which on Metal hard-codes zero=true (MetalRenderBackend
            // :104-111), so the `.zero()` chains elsewhere in the port are redundant there and this
            // explicit `false` is the one place that actually leaves garbage resident.
            //
            // It matters because this is the buffer holding every LOD quad. A draw reads it only where
            // metadata points, and metadata is only created when a section is meshed -- but nothing
            // forces the geometry copy to land before that metadata becomes visible to `cmdgen`. When
            // the two disagree for a frame, the shader reads never-written memory and decodes it as
            // quads: arbitrary positions, arbitrary model ids, arbitrary light. The user's description
            // of the expected shape of this is exact -- "an unzeroed buffer renders something random
            // for one frame, and then gets written over".
            //
            // The log line above already claims "Creating and zeroing ... geometry buffer", which is
            // false; and the upstream TODO here shows the zeroing was removed deliberately, to avoid
            // paying for it at startup, with the consequence never re-tested.
            //
            // VOXY_GEOMETRY_ZERO=1 restores the zeroing, so that can be tested as one variable on one
            // build. Default off, i.e. today's behaviour, so an unset run is the control.
            final boolean zeroGeometry = "1".equals(System.getenv("VOXY_GEOMETRY_ZERO"));
            buffer = RenderBackendFactory.get().createBuffer(geometryCapacity, 0, zeroGeometry);//Only do this if we are not on nvidia
            if (zeroGeometry) {
                Logger.info("VOXY_GEOMETRY_ZERO=1: zero-filled the " + (geometryCapacity / (1024 * 1024))
                        + "MB geometry buffer (upstream skips this; expect a startup pause)");
            }
        } else {
            Logger.info("Running on nvidia, using workaround sparse buffer allocation");
        }
        if (!glBackend) {
            if (buffer == null) {
                throw new IllegalStateException("Unable to allocate geometry buffer");
            }
            this.geometryBuffer = buffer;
            Logger.info("Successfully allocated the geometry buffer in "
                    + (System.currentTimeMillis() - start) + "ms");
            return;
        }
        int error = glGetError();
        if (error != GL_NO_ERROR || buffer == null) {
            if ((buffer == null || error == GL_OUT_OF_MEMORY) && RenderBackendFactory.get().hasSparseBuffer()) {
                if (buffer != null) {
                    Logger.error("Failed to allocate geometry buffer, attempting workaround with sparse buffers");
                    buffer.free();
                }
                buffer = RenderBackendFactory.get().createBuffer(geometryCapacity, GL_SPARSE_STORAGE_BIT_ARB);
                //buffer.zero();
                error = glGetError();
                if (error != GL_NO_ERROR) {
                    buffer.free();
                    throw new IllegalStateException("Unable to allocate geometry buffer using workaround, got gl error " + error);
                }
            } else {
                throw new IllegalStateException("Unable to allocate geometry buffer, got gl error " + error);
            }
        }
        this.geometryBuffer = buffer;
        long delta = System.currentTimeMillis() - start;
        Logger.info("Successfully allocated the geometry buffer in " + delta + "ms");
    }

    private long sparseCommitment = 0;//Tracks the current range of the allocated sparse buffer
    public void ensureAccessable(int maxElementAccess) {
        long size = (Integer.toUnsignedLong(maxElementAccess)*8L+65535L)&~65535L;
        //If we are a sparse buffer, ensure the memory upto the requested size is allocated
        if (this.geometryBuffer.isSparse()) {
            if (this.sparseCommitment < size) {//if we try to access memory outside the allocation range, allocate it
                glBindBuffer(GL_ARRAY_BUFFER, this.geometryBuffer.id());
                size += 65536L*1024;//increase size by 64mb to prevent driver allocation thrashing
                glBufferPageCommitmentARB(GL_ARRAY_BUFFER, this.sparseCommitment, size-this.sparseCommitment, true);
                glBindBuffer(GL_ARRAY_BUFFER, 0);
                this.sparseCommitment = size;
            }
        }
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

        long gpuMemory = 0;
        if (Capabilities.INSTANCE.canQueryGpuMemory) {
            GlCompat.finish();
            gpuMemory = Capabilities.INSTANCE.getFreeDedicatedGpuMemory();
        }
        if (this.geometryBuffer.isSparse()) {
            glBindBuffer(GL_ARRAY_BUFFER, this.geometryBuffer.id());
            glBufferPageCommitmentARB(GL_ARRAY_BUFFER, 0, this.sparseCommitment, false);
            glBindBuffer(GL_ARRAY_BUFFER, 0);
        }

        GlCompat.finish();
        this.geometryBuffer.free();
        GlCompat.finish();
        if (Capabilities.INSTANCE.canQueryGpuMemory) {
            long releaseSize = (long) (this.geometryBuffer.size()*0.75);//if gpu memory usage drops by 75% of the expected value assume we freed it
            if (this.geometryBuffer.isSparse()) {//If we are using sparse buffers, use the commited size instead
                releaseSize = (long)(this.sparseCommitment*0.75);
            }
            if (Capabilities.INSTANCE.getFreeDedicatedGpuMemory()-gpuMemory<=releaseSize) {
                Logger.info("Attempting to wait for gpu memory to release");
                long start = System.currentTimeMillis();

                long TIMEOUT = 2500;

                while (System.currentTimeMillis() - start > TIMEOUT) {//Wait up to 2.5 seconds for memory to release
                    GlCompat.finish();
                    if (Capabilities.INSTANCE.getFreeDedicatedGpuMemory() - gpuMemory > releaseSize) break;
                }
                if (Capabilities.INSTANCE.getFreeDedicatedGpuMemory() - gpuMemory <= releaseSize) {
                    Logger.warn("Failed to wait for gpu memory to be freed, this could indicate an issue with the driver");
                }
            }
        }
    }
}
