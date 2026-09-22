package com.metallum.render;

import me.cortex.voxy.client.core.metal.MetalNative;

/**
 * Test double for Metallum's real {@code com.metallum.render.MetalInterop}.
 *
 * <p>Deliberately backed by a <b>real</b> {@code MTLDevice} and real {@code MTLTexture} objects
 * rather than fabricated numbers, so the tests exercise the reflective bridge against handles that
 * genuinely behave like Metal objects. The class name and signatures mirror the real facade exactly
 * (see metallum's {@code MetalInterop}); if that API changes shape, these tests stop compiling,
 * which is the point — it is the cross-repo contract.
 */
public final class MetalInterop {

    public static final long DEVICE;
    public static final long COLOR_TEXTURE;
    public static final long DEPTH_TEXTURE;

    static {
        long device = 0L;
        long color = 0L;
        long depth = 0L;
        if (MetalNative.load()) {
            device = MetalNative.mtlCreateSystemDefaultDevice();
            if (device != 0L) {
                // 64x48 RGBA8Unorm colour, 32x24 Depth32Float.
                long colorDesc = MetalNative.mtlNewTextureDescriptor(2, 70, 64, 48, 1, 5, 2);
                if (colorDesc != 0L) {
                    color = MetalNative.mtlDeviceNewTexture(device, colorDesc);
                }
                long depthDesc = MetalNative.mtlNewTextureDescriptor(2, 252, 32, 24, 1, 4, 2);
                if (depthDesc != 0L) {
                    depth = MetalNative.mtlDeviceNewTexture(device, depthDesc);
                }
            }
        }
        DEVICE = device;
        COLOR_TEXTURE = color;
        DEPTH_TEXTURE = depth;
    }

    private MetalInterop() {
    }

    public static boolean isAvailable() {
        return DEVICE != 0L;
    }

    public static long deviceHandle() {
        return DEVICE;
    }

    public static long commandQueueHandle() {
        return 0L;
    }

    public static long currentCommandBufferHandle() {
        return 0L;
    }

    public static long currentRenderEncoderHandle() {
        return 0L;
    }

    public static long currentColorAttachmentHandle() {
        return COLOR_TEXTURE;
    }

    public static long currentDepthAttachmentHandle() {
        return DEPTH_TEXTURE;
    }

    public static int currentViewportWidth() {
        return 64;
    }

    public static int currentViewportHeight() {
        return 48;
    }

    public static void endCurrentEncoder() {
    }

    // ---------------------------------------------------------------------------------------------
    // The five below were MISSING, and their absence silently disabled the whole double.
    //
    // MetallumBridge.resolve() runs its lookups in order and used to latch `resolved` before the try,
    // so the first NoSuchMethodException -- endCurrentEncoderAndReport, the sixth lookup -- left every
    // field after it null for the process lifetime while isAvailable (the first lookup) still made
    // available() report true. Three of MetallumBridgePresentTest's assertions read those later
    // fields and failed with bare zeros, with no stated cause, because SLF4J is NOP in this source
    // set and the error naming the missing method was swallowed.
    //
    // So the drift is one-directional and entirely in this test double: the bridge gained these five
    // lookups on 2026-09-19 (00dccf76, c2185e95) and this file was never updated. The class comment
    // above claims the tests "stop compiling" if the facade changes shape -- they do not, because the
    // bridge reaches the façade reflectively and nothing links these signatures to resolve()'s
    // strings. Kept in step by hand, and MetallumBridgePresentTest is what catches the next drift.
    //
    // Signatures mirror the real facade (metallum MetalInterop:165, :174, :190, :218, :266).
    // ---------------------------------------------------------------------------------------------

    public static long openEncoderHandle() {
        return 0L;
    }

    public static long endCurrentEncoderAndReport() {
        return 0L;
    }

    public static long acquireRenderEncoder(final long colorHandle, final long depthHandle,
                                            final int width, final int height) {
        return 0L;
    }

    public static long textureHandle(final com.mojang.blaze3d.textures.GpuTexture texture) {
        return 0L;
    }

    public static void invalidateRenderPassState() {
    }

    /** Records mid-frame splits so tests can assert Voxy's submit() reached Metallum. */
    public static int flushFrameCalls;

    public static void flushFrame() {
        flushFrameCalls++;
    }
}
