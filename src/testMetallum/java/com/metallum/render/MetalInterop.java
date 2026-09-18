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
}
