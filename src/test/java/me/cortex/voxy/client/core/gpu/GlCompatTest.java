package me.cortex.voxy.client.core.gpu;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Field;
import java.lang.reflect.Proxy;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Verifies that GL calls Voxy makes outside its backend abstraction are actually skipped on a
 * non-GL backend.
 *
 * <p>These tests are unusual in that they pass by <em>not</em> killing the JVM. Calling
 * {@code glFinish()} with no GL context current does not throw — LWJGL aborts the process:
 *
 * <pre>FATAL ERROR in native method: No context is current ... The JVM will abort execution.</pre>
 *
 * <p>So no assertion can observe the failure; the suite simply dies with SIGABRT if the gate is
 * removed. That is a real regression guard rather than a decorative one — this exact abort is what
 * ended two Metal runs in a crash dump, from two different GL calls (a bakery framebuffer query and
 * a teardown {@code glFinish}).
 */
class GlCompatTest {

    /**
     * Read reflectively rather than via {@code RenderBackendFactory.get()}: calling {@code get()}
     * would construct the real backend, which on an Apple Silicon host means loading the Metal
     * native library or falling back to {@code GlRenderBackend} — the latter aborting the JVM in a
     * test process that has no GL context. That is the very failure this class exists to prevent.
     */
    private static RenderBackend currentInstance() {
        try {
            Field field = RenderBackendFactory.class.getDeclaredField("INSTANCE");
            field.setAccessible(true);
            return (RenderBackend) field.get(null);
        } catch (ReflectiveOperationException e) {
            throw new AssertionError("RenderBackendFactory.INSTANCE is no longer a static field", e);
        }
    }

    private RenderBackend previous;

    @BeforeEach
    void captureBackend() {
        this.previous = currentInstance();
    }

    @AfterEach
    void restoreBackend() {
        RenderBackendFactory.set(this.previous);
    }

    /**
     * A backend that answers {@code getType()} and nothing else. {@link GlCompat} consults only that
     * method, so a proxy is enough — no need to stand up the full interface, and anything else being
     * called shows up as a thrown exception rather than a silent pass.
     */
    private static RenderBackend backendOfType(final BackendType type) {
        return (RenderBackend) Proxy.newProxyInstance(
                RenderBackend.class.getClassLoader(),
                new Class<?>[]{RenderBackend.class},
                (proxy, method, args) -> switch (method.getName()) {
                    case "getType" -> type;
                    case "toString" -> "fake-" + type;
                    case "hashCode" -> System.identityHashCode(proxy);
                    case "equals" -> proxy == args[0];
                    default -> throw new UnsupportedOperationException(
                            "GlCompat must not call " + method.getName() + " on the backend");
                });
    }

    @Test
    void detectsANonGlBackend() {
        RenderBackendFactory.set(backendOfType(BackendType.METAL));
        assertFalse(GlCompat.isGlBackend(), "METAL is not a GL backend");
    }

    @Test
    void detectsTheGlBackend() {
        RenderBackendFactory.set(backendOfType(BackendType.OPENGL));
        assertTrue(GlCompat.isGlBackend(), "OPENGL is a GL backend");
    }

    /** The load-bearing one: without the gate this aborts the JVM instead of failing. */
    @Test
    void finishIsSkippedOnANonGlBackend() {
        RenderBackendFactory.set(backendOfType(BackendType.METAL));
        assertDoesNotThrow(GlCompat::finish);
    }
}
