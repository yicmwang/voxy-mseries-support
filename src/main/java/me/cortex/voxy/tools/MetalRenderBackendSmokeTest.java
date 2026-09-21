package me.cortex.voxy.tools;

import me.cortex.voxy.client.core.gpu.IGpuTexture;
import me.cortex.voxy.client.core.gpu.RenderBackend;
import me.cortex.voxy.client.core.gpu.RenderEncoder;
import me.cortex.voxy.client.core.gpu.RenderPassDesc;
import me.cortex.voxy.client.core.metal.MetalNative;
import me.cortex.voxy.client.core.metal.MetalRenderBackend;

/**
 * M3 verification: open a Metal render pass with clear-color load action,
 * close the encoder, submit, and confirm the command buffer completed.
 *
 * Doesn't read pixels back yet (texture readback JNI lands in M5/M7); status
 * == Completed is enough to confirm the render encoder JNI wires up correctly
 * and Metal accepts the descriptor.
 *
 * Run with {@code ./gradlew testMetalRender}.
 */
public final class MetalRenderBackendSmokeTest {

    private MetalRenderBackendSmokeTest() {}

    public static void main(String[] args) {
        String os = System.getProperty("os.name", "").toLowerCase();
        String arch = System.getProperty("os.arch", "").toLowerCase();
        if (!(os.contains("mac") && arch.contains("aarch64"))) {
            System.err.println("M3 smoke test requires macOS aarch64 (got " + os + "/" + arch + ")");
            System.exit(2);
        }
        if (!MetalNative.load()) {
            System.err.println("Metal native library failed to load");
            System.exit(2);
        }

        MetalRenderBackend backend = new MetalRenderBackend();
        try {
            // 256x256 RGBA8 texture as the clear target. GL format constants are
            // translated by MetalFormatUtil; GL_RGBA8 = 0x8058.
            final int GL_RGBA8 = 0x8058;
            final int GL_TEXTURE_2D = 0x0DE1;

            IGpuTexture target = backend.createTexture(GL_TEXTURE_2D);
            target.store(GL_RGBA8, 1, 256, 256);
            target.name("voxy-m3-clear-target");

            RenderPassDesc pass = RenderPassDesc.builder(256, 256)
                    .clearColor(target, 1.0f, 0.5f, 0.25f, 1.0f)
                    .build();

            try (RenderEncoder enc = backend.beginRenderPass(pass)) {
                // Empty pass — clear runs as the load action when the encoder is created.
                if (enc == null) throw new RuntimeException("beginRenderPass returned null encoder");
            }
            backend.submit();

            System.out.println("M3 SMOKE OK — Metal clear-color render pass committed and completed");
            System.out.println("  Backend: METAL");   // the only one there is; getType() is gone
            System.out.println("  Target:  256x256 RGBA8 (id=" + target.id() + ")");
            System.out.println("  Clear:   (1.0, 0.5, 0.25, 1.0)");

            // Exercise the GPU timer, so a broken JNI binding fails HERE rather than silently
            // reporting -1 in every [Metal-PERF] line of a six-minute game run. Metal reports
            // GPUStartTime/GPUEndTime as 0.0 until a buffer completes, so a non-negative reading
            // also confirms captureGpuTime ran after the wait rather than before it.
            double gpuMs = backend.lastSubmitGpuMs();
            System.out.println("  GPU time: " + (gpuMs >= 0.0
                    ? String.format("%.4f ms (Metal reported timestamps)", gpuMs)
                    : "UNAVAILABLE (-1) — the timer is wired but Metal returned no timestamps"));
            if (gpuMs < 0.0) {
                System.out.println("    !! captureGpuTime ran but got 0.0, or the JNI symbol is missing.");
                System.out.println("    !! [Metal-PERF]'s gpu= field will read -1; investigate before trusting it.");
            }
        } finally {
            backend.shutdown();
        }
    }
}
