package me.cortex.voxy.client.core.metal;

import me.cortex.voxy.client.core.gpu.IGpuTexture;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Exercises the adapter the P2 bridge rewire depends on, against <b>real</b> Metal textures.
 *
 * <p>The point of {@link MetallumAttachmentTexture} is that the Metal encoder resolves textures
 * through {@link MetalHandleMap}, so this verifies the two things that actually matter: the handle is
 * registered so an encoder lookup succeeds, and the reported dimensions come from the real texture
 * rather than from a guess.
 */
class MetallumAttachmentTextureTest {

    private static final int MTL_TEXTURE_TYPE_2D = 2;
    private static final int RGBA8_UNORM = 70;
    private static final int DEPTH32_FLOAT = 252;
    private static final int USAGE_SHADER_READ_RENDER_TARGET = 5;
    private static final int STORAGE_PRIVATE = 2;

    private static long device;

    @BeforeAll
    static void requireMetal() {
        assumeTrue(MetalNative.load(), "Metal native library unavailable");
        device = MetalNative.mtlCreateSystemDefaultDevice();
        assumeTrue(device != 0L, "no Metal device");
    }

    private static long newTexture(int width, int height, int pixelFormat, int usage) {
        long desc = MetalNative.mtlNewTextureDescriptor(
                MTL_TEXTURE_TYPE_2D, pixelFormat, width, height, 1, usage, STORAGE_PRIVATE);
        assertNotEquals(0L, desc, "texture descriptor");
        long texture = MetalNative.mtlDeviceNewTexture(device, desc);
        assertNotEquals(0L, texture, "texture creation");
        return texture;
    }

    @Test
    void adaptsRealTextureDimensionsAndFormat() {
        long texture = newTexture(64, 48, RGBA8_UNORM, USAGE_SHADER_READ_RENDER_TARGET);
        MetallumAttachmentTexture adapter =
                new MetallumAttachmentTexture("test-color", () -> texture);
        adapter.refresh();

        assertEquals(texture, adapter.metalHandle());
        assertEquals(RGBA8_UNORM, adapter.mtlPixelFormat(),
                "pixel format must come from the real texture");
        assertEquals(MetalNative.mtlTextureGetWidth(texture), adapter.getWidth());
        assertEquals(MetalNative.mtlTextureGetHeight(texture), adapter.getHeight());
        assertEquals(1, adapter.getLevels());
    }

    @Test
    void registersHandleSoEncoderLookupResolves() {
        long texture = newTexture(16, 16, RGBA8_UNORM, USAGE_SHADER_READ_RENDER_TARGET);
        MetallumAttachmentTexture adapter =
                new MetallumAttachmentTexture("test-color", () -> texture);
        adapter.refresh();

        assertNotEquals(-1, adapter.id(), "adapter must expose a handle-map id");
        assertEquals(texture, MetalHandleMap.getHandle(adapter.id()),
                "encoder bufferHandle()/textureHandle() lookup must resolve to the real handle");
        assertEquals(adapter.id(), MetalHandleMap.getId(texture));
    }

    @Test
    void refreshIsIdempotentForAnUnchangedHandle() {
        long texture = newTexture(32, 32, RGBA8_UNORM, USAGE_SHADER_READ_RENDER_TARGET);
        MetallumAttachmentTexture adapter =
                new MetallumAttachmentTexture("test-color", () -> texture);
        adapter.refresh();
        int firstId = adapter.id();

        adapter.refresh();
        adapter.refresh();

        assertEquals(firstId, adapter.id(), "a stable handle must not be re-registered each frame");
    }

    @Test
    void reRegistersWhenTheAttachmentChanges() {
        long first = newTexture(32, 32, RGBA8_UNORM, USAGE_SHADER_READ_RENDER_TARGET);
        long second = newTexture(64, 64, RGBA8_UNORM, USAGE_SHADER_READ_RENDER_TARGET);
        long[] current = {first};
        MetallumAttachmentTexture adapter =
                new MetallumAttachmentTexture("test-color", () -> current[0]);
        adapter.refresh();
        int firstId = adapter.id();

        current[0] = second;
        adapter.refresh();

        assertNotEquals(firstId, adapter.id(), "a changed attachment must get a fresh id");
        assertEquals(second, MetalHandleMap.getHandle(adapter.id()));
        assertEquals(second, adapter.metalHandle());
        assertEquals(MetalNative.mtlTextureGetWidth(second), adapter.getWidth(),
                "dimensions must follow the new attachment");
        assertEquals(-1, MetalHandleMap.getId(first), "the stale handle must be unregistered");
    }

    @Test
    void toleratesAnUnboundAttachment() {
        // A pass with no depth attachment: refresh() must not register 0 or throw.
        long[] current = {0L};
        MetallumAttachmentTexture adapter =
                new MetallumAttachmentTexture("test-depth", () -> current[0]);
        assertDoesNotThrow(adapter::refresh);
        assertEquals(-1, adapter.id());
        assertEquals(0, adapter.getWidth());
        assertThrows(IllegalStateException.class, adapter::assertAllocated);
    }

    @Test
    void depthAttachmentReportsItsOwnFormat() {
        long depth = newTexture(32, 24, DEPTH32_FLOAT, USAGE_SHADER_READ_RENDER_TARGET);
        MetallumAttachmentTexture adapter =
                new MetallumAttachmentTexture("test-depth", () -> depth);
        adapter.refresh();
        assertEquals(DEPTH32_FLOAT, adapter.mtlPixelFormat());
        assertEquals(MetalNative.mtlTextureGetHeight(depth), adapter.getHeight());
    }

    @Test
    void freeDropsTheRegistrationButNotTheTexture() {
        long texture = newTexture(16, 16, RGBA8_UNORM, USAGE_SHADER_READ_RENDER_TARGET);
        MetallumAttachmentTexture adapter =
                new MetallumAttachmentTexture("test-color", () -> texture);
        adapter.refresh();
        int id = adapter.id();

        adapter.free();

        assertTrue(adapter.isFreed());
        assertEquals(-1, MetalHandleMap.getId(texture), "registration must be dropped");
        assertEquals(-1, adapter.id(), "adapter must expose no id once freed");
        assertEquals(0L, adapter.metalHandle());
    }

    @Test
    void isAnIGpuTexture() {
        // Guards the interface contract the render path relies on.
        assertInstanceOf(IGpuTexture.class, MetallumAttachmentTexture.color());
        assertInstanceOf(IGpuTexture.class, MetallumAttachmentTexture.depth());
    }
}
