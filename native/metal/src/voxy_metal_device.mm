// voxy_metal_device.mm — MTLDevice, MTLCommandQueue, and basic device info.

#include "voxy_metal.h"

extern "C" JNIEXPORT jlong JNICALL
Java_me_cortex_voxy_client_core_metal_MetalNative_mtlCreateSystemDefaultDevice(
        JNIEnv *, jclass) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) return 0;
        return voxy_handle_from(device);
    }
}

extern "C" JNIEXPORT jlong JNICALL
Java_me_cortex_voxy_client_core_metal_MetalNative_mtlDeviceNewCommandQueue(
        JNIEnv *, jclass, jlong deviceHandle) {
    @autoreleasepool {
        if (deviceHandle == 0) return 0;
        id<MTLDevice> device = voxy_handle_cast<id<MTLDevice>>(deviceHandle);
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (queue == nil) return 0;
        return voxy_handle_from(queue);
    }
}

extern "C" JNIEXPORT jstring JNICALL
Java_me_cortex_voxy_client_core_metal_MetalNative_mtlDeviceGetName(
        JNIEnv *env, jclass, jlong deviceHandle) {
    @autoreleasepool {
        if (deviceHandle == 0) return nullptr;
        id<MTLDevice> device = voxy_handle_cast<id<MTLDevice>>(deviceHandle);
        NSString *name = [device name];
        if (name == nil) return nullptr;
        return env->NewStringUTF([name UTF8String]);
    }
}

extern "C" JNIEXPORT jlong JNICALL
Java_me_cortex_voxy_client_core_metal_MetalNative_mtlDeviceMaxBufferLength(
        JNIEnv *, jclass, jlong deviceHandle) {
    @autoreleasepool {
        if (deviceHandle == 0) return 0;
        id<MTLDevice> device = voxy_handle_cast<id<MTLDevice>>(deviceHandle);
        if (@available(macOS 10.14, *)) {
            return (jlong)[device maxBufferLength];
        }
        return (jlong)(256LL * 1024 * 1024); // Conservative fallback.
    }
}

extern "C" JNIEXPORT jboolean JNICALL
Java_me_cortex_voxy_client_core_metal_MetalNative_mtlDeviceSupportsFamily(
        JNIEnv *, jclass, jlong deviceHandle, jint family) {
    @autoreleasepool {
        if (deviceHandle == 0) return JNI_FALSE;
        id<MTLDevice> device = voxy_handle_cast<id<MTLDevice>>(deviceHandle);
        if (@available(macOS 10.15, *)) {
            return [device supportsFamily:(MTLGPUFamily)family] ? JNI_TRUE : JNI_FALSE;
        }
        return JNI_FALSE;
    }
}

// -------- Command buffer --------

extern "C" JNIEXPORT jlong JNICALL
Java_me_cortex_voxy_client_core_metal_MetalNative_mtlCommandQueueNewCommandBuffer(
        JNIEnv *, jclass, jlong queueHandle) {
    @autoreleasepool {
        if (queueHandle == 0) return 0;
        id<MTLCommandQueue> queue = voxy_handle_cast<id<MTLCommandQueue>>(queueHandle);
        id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
        if (cmdBuf == nil) return 0;
        return voxy_handle_from(cmdBuf);
    }
}

extern "C" JNIEXPORT void JNICALL
Java_me_cortex_voxy_client_core_metal_MetalNative_mtlCommandBufferCommit(
        JNIEnv *, jclass, jlong cmdBufHandle) {
    @autoreleasepool {
        if (cmdBufHandle == 0) return;
        id<MTLCommandBuffer> cmdBuf = voxy_handle_cast<id<MTLCommandBuffer>>(cmdBufHandle);
        [cmdBuf commit];
    }
}

extern "C" JNIEXPORT void JNICALL
Java_me_cortex_voxy_client_core_metal_MetalNative_mtlCommandBufferWaitUntilCompleted(
        JNIEnv *, jclass, jlong cmdBufHandle) {
    @autoreleasepool {
        if (cmdBufHandle == 0) return;
        id<MTLCommandBuffer> cmdBuf = voxy_handle_cast<id<MTLCommandBuffer>>(cmdBufHandle);
        [cmdBuf waitUntilCompleted];
    }
}

extern "C" JNIEXPORT jint JNICALL
Java_me_cortex_voxy_client_core_metal_MetalNative_mtlCommandBufferGetStatus(
        JNIEnv *, jclass, jlong cmdBufHandle) {
    @autoreleasepool {
        if (cmdBufHandle == 0) return 0;
        id<MTLCommandBuffer> cmdBuf = voxy_handle_cast<id<MTLCommandBuffer>>(cmdBufHandle);
        return (jint)[cmdBuf status];
    }
}

// ---------------------------------------------------------------------------
// GPU timing.
//
// `GPUStartTime` / `GPUEndTime` are CFTimeInterval (seconds) reported by Metal itself, and their
// difference is the GPU-side duration of this command buffer's execution. Until now the only
// GPU-cost reading in the tree was [Metal-PERF]'s `submit`, which is commit + waitUntilCompleted --
// a measure of how long the CPU WAITED, not how long the GPU worked. The difference matters: the
// wait also absorbs Metallum's flushFrame and the deferred ordered index-wait that sits in front of
// it (MetalRenderBackend.submit's guest branch), so a fill-vs-draw split read through `submit` is a
// reading of the wrong quantity.
//
// The properties are spelled GPUStartTime / GPUEndTime -- capital GPU. `gpuStartTime` is NOT a
// selector on MTLCommandBuffer, and the compiler says so ("no known instance method for selector"),
// which is how this was found.
//
// BOTH VALUES ARE ONLY MEANINGFUL AFTER THE COMMAND BUFFER COMPLETES. Metal reports 0.0 before that,
// so a caller that reads them without having waited gets 0 ms rather than an error -- check for a
// zero END time and treat it as "not available yet" rather than as "the GPU took no time".
extern "C" JNIEXPORT jdouble JNICALL
Java_me_cortex_voxy_client_core_metal_MetalNative_mtlCommandBufferGetGpuStartTime(
        JNIEnv *, jclass, jlong cmdBufHandle) {
    @autoreleasepool {
        if (cmdBufHandle == 0) return 0.0;
        id<MTLCommandBuffer> cmdBuf = voxy_handle_cast<id<MTLCommandBuffer>>(cmdBufHandle);
        return (jdouble)[cmdBuf GPUStartTime];
    }
}

extern "C" JNIEXPORT jdouble JNICALL
Java_me_cortex_voxy_client_core_metal_MetalNative_mtlCommandBufferGetGpuEndTime(
        JNIEnv *, jclass, jlong cmdBufHandle) {
    @autoreleasepool {
        if (cmdBufHandle == 0) return 0.0;
        id<MTLCommandBuffer> cmdBuf = voxy_handle_cast<id<MTLCommandBuffer>>(cmdBufHandle);
        return (jdouble)[cmdBuf GPUEndTime];
    }
}
