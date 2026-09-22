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
            mViewportWidth, mViewportHeight, mFlushFrame, mAcquireRenderEncoder, mInvalidateRenderPassState,
            mLogRenderPassCounters, mOpenEncoderHandle, mEndCurrentEncoderAndReport, mTextureHandle,
            mHasPendingColorClear, mHasPendingDepthClear, mPendingColorClearCount,
            mDrawProbeStyleTriangle;
    /** An attempt has been made; do not make another (the lookups are not cheap and cannot start working). */
    private static boolean resolved;
    /**
     * The attempt SUCCEEDED -- every non-optional lookup resolved.
     *
     * <p>Distinct from {@link #resolved}, and the distinction is the whole fix. This method used to set
     * {@code resolved = true} BEFORE its try block, so a lookup that threw partway through left the
     * fields before it set and every field after it null, for the lifetime of the process, while
     * {@code mIsAvailable} -- one of the first to be set -- still made {@link #available()} report
     * true. Voxy would then take the whole "encode into Metallum's frame" path with a null colour
     * attachment, a null render encoder and a 0x0 viewport, and the only evidence was a single log line
     * naming the missing method.
     *
     * <p>That is not hypothetical: it is exactly what the testMetallum double produced, where five
     * non-optional methods were absent and the tests failed with bare zeros from three of them. In
     * production the real facade satisfies all of them, so it never fired -- which is why it survived.
     */
    private static boolean resolveOk;

    private MetallumBridge() {
    }

    /**
     * A method that is nice to have but must not disable the bridge. Diagnostics land here: a
     * mismatch in one of them previously threw out of {@link #resolve()}, which made metallum look
     * absent and sent Voxy down its standalone path.
     */
    private static Method optional(final Class<?> interop, final String name, final Class<?>... params) {
        try {
            return interop.getMethod(name, params);
        } catch (Throwable t) {
            Logger.warn("MetallumBridge: optional interop method '" + name + "' unavailable ("
                    + t.getClass().getSimpleName() + "); that diagnostic is disabled");
            return null;
        }
    }

    private static synchronized void resolve() {
        if (resolved) {
            return;
        }
        resolved = true;
        // `resolveOk` is set only at the END of the try. A throw leaves it false, and because the
        // fields set before the throw are left populated, `available()` must consult this flag rather
        // than `mIsAvailable` -- see the field comment.
        try {
            Class<?> interop = Class.forName(INTEROP);
            mIsAvailable = interop.getMethod("isAvailable");
            mDeviceHandle = interop.getMethod("deviceHandle");
            mCommandQueueHandle = interop.getMethod("commandQueueHandle");
            mCommandBufferHandle = interop.getMethod("currentCommandBufferHandle");
            mEndCurrentEncoder = interop.getMethod("endCurrentEncoder");
            mEndCurrentEncoderAndReport = interop.getMethod("endCurrentEncoderAndReport");
            mOpenEncoderHandle = interop.getMethod("openEncoderHandle");
            mTextureHandle = interop.getMethod("textureHandle", Class.forName("com.mojang.blaze3d.textures.GpuTexture"));
            mRenderEncoderHandle = interop.getMethod("currentRenderEncoderHandle");
            mColorAttachment = interop.getMethod("currentColorAttachmentHandle");
            mDepthAttachment = interop.getMethod("currentDepthAttachmentHandle");
            mViewportWidth = interop.getMethod("currentViewportWidth");
            mViewportHeight = interop.getMethod("currentViewportHeight");
            mFlushFrame = interop.getMethod("flushFrame");
            mAcquireRenderEncoder = interop.getMethod("acquireRenderEncoder", long.class, long.class, int.class, int.class);
            mInvalidateRenderPassState = interop.getMethod("invalidateRenderPassState");
            mLogRenderPassCounters = optional(interop, "logRenderPassCounters");
            mHasPendingColorClear = optional(interop, "hasPendingColorClear", long.class);
            mHasPendingDepthClear = optional(interop, "hasPendingDepthClear", long.class);
            mPendingColorClearCount = optional(interop, "pendingColorClearCount");
            mDrawProbeStyleTriangle = optional(interop, "drawProbeStyleTriangle",
                    long.class, long.class, int.class, int.class, String.class);
            resolveOk = true;
            Logger.info("Metallum interop detected; Voxy will encode into Metallum's frame");
        } catch (Throwable t) {
            // Clear what was set before the throw, so no accessor can hand out a half-resolved bridge.
            // Without this, a caller that reaches an accessor without consulting available() -- and
            // several do, via the direct helpers below -- gets a real device handle and a null
            // everything-else.
            mIsAvailable = mDeviceHandle = mCommandQueueHandle = mCommandBufferHandle = null;
            mEndCurrentEncoder = mRenderEncoderHandle = mColorAttachment = mDepthAttachment = null;
            mViewportWidth = mViewportHeight = mFlushFrame = null;
            mAcquireRenderEncoder = mInvalidateRenderPassState = null;
            mOpenEncoderHandle = mEndCurrentEncoderAndReport = mTextureHandle = null;
            // Name the method. A bare "interop not present" is indistinguishable from metallum
            // genuinely being absent, and it silently changes Voxy's whole rendering path -- it
            // renders standalone instead of into Metallum's frame. That cost three invalid
            // measurement runs before anyone looked at the log line.
            Logger.error("Metallum interop resolution FAILED (" + t.getClass().getSimpleName() + ": "
                    + t.getMessage() + "); Voxy will own its own Metal device and will NOT draw into "
                    + "Metallum's frame. This is a Voxy/metallum version mismatch, not an absent mod.",
                    t);
        }
    }

    /** True when Metallum is loaded and has published its device. */
    public static boolean available() {
        resolve();
        // resolveOk, NOT just mIsAvailable. isAvailable is one of the FIRST lookups resolve() performs,
        // so it is set even when a later one throws and leaves the bridge half-built; gating on it alone
        // is what let a partial resolve masquerade as a working bridge.
        if (!resolveOk || mIsAvailable == null) {
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
        resolve();
        return call(mDeviceHandle);
    }

    /** Metallum's {@code MTLCommandQueue}, or 0. */
    public static long commandQueue() {
        resolve();
        return call(mCommandQueueHandle);
    }

    /** The frame's {@code MTLCommandBuffer}, or 0. */
    public static long commandBuffer() {
        resolve();
        return call(mCommandBufferHandle);
    }

    /** The open {@code MTLRenderCommandEncoder}, or 0. */
    public static long renderEncoder() {
        resolve();
        return call(mRenderEncoderHandle);
    }

    public static long colorAttachment() {
        resolve();
        return call(mColorAttachment);
    }

    public static long depthAttachment() {
        resolve();
        return call(mDepthAttachment);
    }

    public static int viewportWidth() {
        resolve();
        return (int) call(mViewportWidth);
    }

    public static int viewportHeight() {
        resolve();
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

    /**
     * Handle of the encoder Metallum currently has open (render, compute or blit), or {@code 0}.
     * Diagnostics only — call {@link #endCurrentEncoder()} to actually close it.
     */
    public static long openEncoderHandle() {
        resolve();
        if (mOpenEncoderHandle == null) {
            return 0L;
        }
        try {
            Object v = mOpenEncoderHandle.invoke(null);
            return v instanceof Long l ? l : 0L;
        } catch (Throwable t) {
            return 0L;
        }
    }

    /**
     * The raw {@code MTLTexture} handle behind a Blaze3D texture, or {@code 0} when it is not a
     * Metallum texture (or Metallum is absent).
     *
     * <p>This is how Voxy samples Minecraft's own textures under whole-frame Metal — notably the
     * block atlas, which is already on Metallum's device and so needs no readback or mirror.
     */
    public static long textureHandle(Object gpuTexture) {
        resolve();
        if (mTextureHandle == null || gpuTexture == null) {
            return 0L;
        }
        try {
            Object v = mTextureHandle.invoke(null, gpuTexture);
            return v instanceof Long l ? l : 0L;
        } catch (Throwable t) {
            Logger.error("MetallumBridge.textureHandle failed", t);
            return 0L;
        }
    }

    /**
     * {@link #endCurrentEncoder()} reporting the handle it closed, or {@code 0} if none was open.
     * Diagnostics only.
     */
    public static long endCurrentEncoderAndReport() {
        resolve();
        if (mEndCurrentEncoderAndReport == null) {
            return 0L;
        }
        try {
            Object v = mEndCurrentEncoderAndReport.invoke(null);
            return v instanceof Long l ? l : 0L;
        } catch (Throwable t) {
            return 0L;
        }
    }

    /**
     * Submits and waits on Metallum's current frame buffer so GPU-written data becomes CPU-visible,
     * then continues the frame in a fresh buffer.
     *
     * <p>Voxy needs this mid-frame: its draw path reads the GPU-generated draw commands to push
     * {@code baseInstance} (Metal's indirect path does not propagate it), and that read cannot be
     * satisfied from a buffer that has not been committed. Without a mid-frame completion point Voxy
     * cannot encode into Metallum's frame at all.
     *
     * <p>No-op when Metallum is absent or predates the hook.
     */
    public static void flushFrame() {
        resolve();
        if (mFlushFrame == null) {
            return;
        }
        try {
            mFlushFrame.invoke(null);
        } catch (Throwable t) {
            Logger.error("MetallumBridge.flushFrame failed", t);
        }
    }

    /** True when Metallum exposes the mid-frame split (see {@link #flushFrame()}). */
    public static boolean supportsFlushFrame() {
        resolve();
        return mFlushFrame != null;
    }

    /**
     * Borrows Metallum's render encoder for these attachments, reusing Metallum's open encoder when
     * the attachments match. Drawing on Metallum's encoder is the only legal way to share a pass:
     * asking Metal for a second encoder on the same command buffer trips
     * "A command encoder is already encoding to this command buffer".
     *
     * <p>Returns 0 when Metallum is absent; the caller then owns its own encoder.
     */
    public static long acquireRenderEncoder(final long colorHandle, final long depthHandle,
                                            final int width, final int height) {
        resolve();
        if (mAcquireRenderEncoder == null) {
            return 0L;
        }
        try {
            Object r = mAcquireRenderEncoder.invoke(null, colorHandle, depthHandle, width, height);
            return r instanceof Number n ? n.longValue() : 0L;
        } catch (Throwable t) {
            Logger.error("MetallumBridge.acquireRenderEncoder failed", t);
            return 0L;
        }
    }

    /** Marks Metallum's pass state stale after Voxy borrowed and drew on its encoder. */
    /**
     * Whether Metallum still has a clear queued for this texture -- i.e. whether its next render
     * pass on it will clear it. Voxy's borrow path bypasses the bookkeeping that consumes these,
     * so a queued clear means this frame's LOD draws get erased afterwards. Diagnostic only.
     */
    public static boolean hasPendingColorClear(final long textureHandle) {
        resolve();
        if (mHasPendingColorClear == null) return false;
        try {
            return (Boolean) mHasPendingColorClear.invoke(null, textureHandle);
        } catch (Throwable t) {
            return false;
        }
    }

    public static boolean hasPendingDepthClear(final long textureHandle) {
        resolve();
        if (mHasPendingDepthClear == null) return false;
        try {
            return (Boolean) mHasPendingDepthClear.invoke(null, textureHandle);
        } catch (Throwable t) {
            return false;
        }
    }

    public static int pendingColorClearCount() {
        resolve();
        if (mPendingColorClearCount == null) return -1;
        try {
            return (Integer) mPendingColorClearCount.invoke(null);
        } catch (Throwable t) {
            return -1;
        }
    }

    /**
     * Draw a magenta triangle at the current point in the frame, via the P0 probe's own path
     * (a fresh encoder made directly from the command buffer), bypassing
     * {@code renderCommandEncoderForHandles}. Diagnostic only; see the metallum-side javadoc.
     */
    public static boolean drawProbeStyleTriangle(final long colorTexture, final long depthTexture,
                                                 final int width, final int height, final String label) {
        resolve();
        if (mDrawProbeStyleTriangle == null) return false;
        try {
            return (Boolean) mDrawProbeStyleTriangle.invoke(null, colorTexture, depthTexture, width, height, label);
        } catch (Throwable t) {
            Logger.error("MetallumBridge.drawProbeStyleTriangle failed", t);
            return false;
        }
    }

    public static void invalidateRenderPassState() {
        resolve();
        if (mInvalidateRenderPassState == null) {
            return;
        }
        try {
            mInvalidateRenderPassState.invoke(null);
        } catch (Throwable t) {
            Logger.error("MetallumBridge.invalidateRenderPassState failed", t);
        }
    }

    /** Temporary: dump Metallum's render-pass entry counters. */
    public static void logRenderPassCounters() {
        resolve();
        if (mLogRenderPassCounters == null) {
            return;
        }
        try {
            mLogRenderPassCounters.invoke(null);
        } catch (Throwable ignored) {
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
