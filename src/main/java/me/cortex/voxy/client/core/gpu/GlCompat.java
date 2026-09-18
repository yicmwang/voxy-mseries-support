package me.cortex.voxy.client.core.gpu;

import static org.lwjgl.opengl.GL11.glFinish;

/**
 * GL calls Voxy makes outside its backend abstraction, routed so they become no-ops on a non-GL
 * backend.
 *
 * <p>Under whole-frame Metal there is no GL context. LWJGL does not throw in that situation — it
 * aborts the process outright:
 *
 * <pre>FATAL ERROR in native method: No context is current ... The JVM will abort execution.</pre>
 *
 * <p>Being an abort rather than an exception, it cannot be caught, so the call has to be skipped
 * rather than guarded. It also does not care whether it happens mid-frame or during teardown: the
 * first occurrence was a framebuffer query in the model bakery, the second a {@code glFinish()} on
 * world exit, and both killed the JVM the same way.
 */
public final class GlCompat {
    private GlCompat() {
    }

    /** True when the active backend actually has a GL context. */
    public static boolean isGlBackend() {
        return RenderBackendFactory.get().getType() == BackendType.OPENGL;
    }

    /**
     * {@code glFinish()} where it means something, a no-op elsewhere.
     *
     * <p>Every call site pairs this with an explicit backend fence wait, and the fence is what
     * actually provides the synchronisation — {@code glFinish()} is the GL-side belt-and-braces.
     * Skipping it on Metal gives up nothing the fence does not already guarantee.
     */
    public static void finish() {
        if (isGlBackend()) {
            glFinish();
        }
    }
}
