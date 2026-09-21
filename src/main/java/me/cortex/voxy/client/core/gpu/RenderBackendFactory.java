package me.cortex.voxy.client.core.gpu;

import me.cortex.voxy.common.Logger;

/**
 * Factory for the render backend.
 *
 * <p>There is exactly one: Metal, on Apple Silicon. This project has no OpenGL path and is not going to
 * grow one, so the factory no longer selects between backends — it resolves the only one and fails
 * loudly if it cannot.
 *
 * <p>The fallback to a GL backend that used to live here was worse than useless once GL stopped being a
 * target: it turned a missing native library into a silently different renderer, which is how you spend
 * an afternoon measuring a frame that Metal never drew.
 */
public final class RenderBackendFactory {
    private static RenderBackend INSTANCE;

    private RenderBackendFactory() {}

    /** Gets or creates the singleton RenderBackend instance. */
    public static RenderBackend get() {
        if (INSTANCE == null) {
            INSTANCE = createBackend();
        }
        return INSTANCE;
    }

    /**
     * Force-sets the backend. Kept for tests that need a stand-in; there is no production caller.
     */
    public static void set(RenderBackend backend) {
        INSTANCE = backend;
    }

    private static RenderBackend createBackend() {
        final String os = System.getProperty("os.name", "").toLowerCase();
        final String arch = System.getProperty("os.arch", "").toLowerCase();
        if (!(os.contains("mac") && arch.contains("aarch64"))) {
            throw new IllegalStateException(
                    "Voxy renders through Metal and this machine is " + os + "/" + arch + ". "
                            + "An Apple Silicon Mac (macOS + aarch64) is required; there is no other backend.");
        }
        if (!me.cortex.voxy.client.core.metal.MetalNative.load()) {
            throw new IllegalStateException(
                    "Could not load the Voxy Metal native library (libvoxy_metal.dylib). It ships in "
                            + "src/main/resources/natives/macos-arm64/ and is built by native/metal/CMakeLists.txt; "
                            + "there is no fallback backend.");
        }
        Logger.info("Using Metal render backend (Apple Silicon)");
        // Deliberately NOT wrapped in a catch: a Metal init failure is fatal here, and swallowing it was
        // only ever justified by having somewhere else to go.
        return new me.cortex.voxy.client.core.metal.MetalRenderBackend();
    }
}
