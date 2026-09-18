package me.cortex.voxy.client.core.metal;

import me.cortex.voxy.common.Logger;

import java.lang.reflect.Method;

/**
 * Reflective bridge to Metallum's public interop surface
 * ({@code com.metallum.render.MetalInterop}).
 *
 * <p>Under whole-frame Metal, Metallum owns the {@code MTLDevice}, the command queue and the
 * frame's {@code MTLCommandBuffer}; Voxy must encode into that same buffer so its passes are
 * ordered with Sodium's and share the depth attachment. Metallum is a separate mod, so this is
 * resolved reflectively rather than as a compile dependency — {@code MetalInterop} was designed
 * as a thin static facade for exactly that.
 *
 * <p>When Metallum is absent every accessor degrades to 0/{@code false} and the backend falls back
 * to owning its own device and command buffers (the standalone smoke-test path).
 */
public final class MetallumBridge {
    private static final String INTEROP = "com.metallum.render.MetalInterop";

    private static Method mIsAvailable, mDeviceHandle, mCommandQueueHandle, mCommandBufferHandle,
            mEndCurrentEncoder, mRenderEncoderHandle, mColorAttachment, mDepthAttachment,
            mViewportWidth, mViewportHeight;
    private static boolean resolved;

    private MetallumBridge() {
    }

    private static synchronized void resolve() {
        if (resolved) {
            return;
        }
        resolved = true;
        try {
            Class<?> interop = Class.forName(INTEROP);
            mIsAvailable = interop.getMethod("isAvailable");
            mDeviceHandle = interop.getMethod("deviceHandle");
            mCommandQueueHandle = interop.getMethod("commandQueueHandle");
            mCommandBufferHandle = interop.getMethod("currentCommandBufferHandle");
            mEndCurrentEncoder = interop.getMethod("endCurrentEncoder");
            mRenderEncoderHandle = interop.getMethod("currentRenderEncoderHandle");
            mColorAttachment = interop.getMethod("currentColorAttachmentHandle");
            mDepthAttachment = interop.getMethod("currentDepthAttachmentHandle");
            mViewportWidth = interop.getMethod("currentViewportWidth");
            mViewportHeight = interop.getMethod("currentViewportHeight");
            Logger.info("Metallum interop detected; Voxy will encode into Metallum's frame");
        } catch (Throwable t) {
            Logger.info("Metallum interop not present (" + t.getClass().getSimpleName()
                    + "); Voxy will own its own Metal device");
        }
    }

    /** True when Metallum is loaded and has published its device. */
    public static boolean available() {
        resolve();
        if (mIsAvailable == null) {
            return false;
        }
        try {
            return Boolean.TRUE.equals(mIsAvailable.invoke(null));
        } catch (Throwable t) {
            return false;
        }
    }

    /** Metallum's {@code MTLDevice}, or 0. */
    public static long device() {
        return call(mDeviceHandle);
    }

    /** Metallum's {@code MTLCommandQueue}, or 0. */
    public static long commandQueue() {
        return call(mCommandQueueHandle);
    }

    /** The frame's {@code MTLCommandBuffer}, or 0. */
    public static long commandBuffer() {
        return call(mCommandBufferHandle);
    }

    /** The open {@code MTLRenderCommandEncoder}, or 0. */
    public static long renderEncoder() {
        return call(mRenderEncoderHandle);
    }

    public static long colorAttachment() {
        return call(mColorAttachment);
    }

    public static long depthAttachment() {
        return call(mDepthAttachment);
    }

    public static int viewportWidth() {
        return (int) call(mViewportWidth);
    }

    public static int viewportHeight() {
        return (int) call(mViewportHeight);
    }

    /**
     * Closes Metallum's currently open encoder so Voxy can open its own on the same attachments.
     * Metallum reopens lazily on its next draw, preserving what Voxy wrote.
     */
    public static void endCurrentEncoder() {
        resolve();
        if (mEndCurrentEncoder == null) {
            return;
        }
        try {
            mEndCurrentEncoder.invoke(null);
        } catch (Throwable t) {
            Logger.error("MetallumBridge.endCurrentEncoder failed", t);
        }
    }

    private static long call(Method m) {
        resolve();
        if (m == null) {
            return 0L;
        }
        try {
            Object r = m.invoke(null);
            return r instanceof Number n ? n.longValue() : 0L;
        } catch (Throwable t) {
            return 0L;
        }
    }
}
