package me.cortex.voxy.client.core.metal;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * The "Metallum present" half of the bridge contract, run in its own source set so a fake
 * {@code com.metallum.render.MetalInterop} is on the classpath. {@link MetallumBridgeTest} covers the
 * absent half; one JVM cannot see the class both ways.
 *
 * <p>The fake is backed by a real {@code MTLDevice} and real textures, so this verifies the whole
 * chain — reflection, resolution, and that the returned handle is a usable Metal object — rather
 * than just that a number came back.
 */
class MetallumBridgePresentTest {

    @BeforeAll
    static void requireMetal() {
        assumeTrue(MetalNative.load(), "Metal native library unavailable");
        assumeTrue(MetalNative.mtlCreateSystemDefaultDevice() != 0L, "no Metal device");
    }

    @Test
    void detectsMetallum() {
        assertTrue(MetallumBridge.available(), "bridge must find the fake MetalInterop");
    }

    @Test
    void resolvedDeviceHandleIsAUsableMetalDevice() {
        long device = MetallumBridge.device();
        assertNotEquals(0L, device, "device handle must be resolved");
        // The strongest available check: ask Metal itself about the handle we got back.
        String name = MetalNative.mtlDeviceGetName(device);
        assertNotNull(name, "resolved handle must be a real MTLDevice");
        assertFalse(name.isBlank(), "MTLDevice name must not be blank");
    }

    @Test
    void resolvesAttachmentHandles() {
        long color = MetallumBridge.colorAttachment();
        long depth = MetallumBridge.depthAttachment();
        assertNotEquals(0L, color);
        assertNotEquals(0L, depth);
        assertNotEquals(color, depth, "colour and depth must be distinct attachments");
    }

    @Test
    void resolvesViewportFromFake() {
        assertEquals(64, MetallumBridge.viewportWidth());
        assertEquals(48, MetallumBridge.viewportHeight());
    }

    @Test
    void endCurrentEncoderDoesNotThrow() {
        assertDoesNotThrow(MetallumBridge::endCurrentEncoder);
    }
}
