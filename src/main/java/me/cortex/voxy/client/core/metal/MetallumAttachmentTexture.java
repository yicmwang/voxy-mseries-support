package me.cortex.voxy.client.core.metal;

import me.cortex.voxy.client.core.gpu.IGpuTexture;

import java.util.function.LongSupplier;

/**
 * Adapts a Metallum-owned texture to {@link IGpuTexture} — most often one of the frame's
 * attachments, but any texture already on Metallum's device qualifies (see
 * {@link #ofMetalTexture}).
 *
 * <p>This is the piece that lets the P2 bridge rewire keep the existing render path: instead of
 * rendering into an IOSurface-backed texture and compositing, Voxy renders straight into the frame's
 * colour/depth attachment that Metallum has bound. The Metal encoder resolves textures through
 * {@link MetalHandleMap}, so all that is needed is an {@code IGpuTexture} that registers the raw
 * {@code MTLTexture} handle — the same shape as {@code IOSurfaceBridge.BridgedGpuTexture}.
 *
 * <p>Handles are resolved lazily and re-resolved per frame via {@link #refresh()}, because Metallum
 * may hand out a different attachment between frames (resize, target swap). When the handle is
 * unchanged, {@code refresh()} is a no-op and the registration is preserved.
 */
public final class MetallumAttachmentTexture implements IGpuTexture {

    private final String label;
    private final LongSupplier handleSource;

    private int id = -1;
    private long handle;
    private int width;
    private int height;
    private int pixelFormat;
    private int levels = 1;

    public MetallumAttachmentTexture(String label, LongSupplier handleSource) {
        this.label = label;
        this.handleSource = handleSource;
    }

    /**
     * An {@link IGpuTexture} over an arbitrary Metallum texture — not necessarily a frame attachment.
     *
     * <p>For textures that already live on Metallum's device and therefore need no cross-API mirror,
     * such as Minecraft's block atlas under whole-frame Metal. The supplier is re-read on every
     * {@link #refresh()}, so a resource-pack reload that swaps the texture is picked up.
     */
    public static MetallumAttachmentTexture ofMetalTexture(String label, LongSupplier handleSource) {
        return new MetallumAttachmentTexture(label, handleSource);
    }

    /** Metallum's currently bound colour attachment. */
    public static MetallumAttachmentTexture color() {
        return new MetallumAttachmentTexture("metallum-color", MetallumBridge::colorAttachment);
    }

    /** Metallum's currently bound depth/stencil attachment. */
    public static MetallumAttachmentTexture depth() {
        return new MetallumAttachmentTexture("metallum-depth", MetallumBridge::depthAttachment);
    }

    /**
     * Re-point at the frame's current attachment. Safe to call every frame; re-registers only when
     * the underlying handle actually changed.
     */
    public void refresh() {
        long next = this.handleSource.getAsLong();
        if (next == this.handle) {
            return;
        }
        if (this.id != -1) {
            MetalHandleMap.unregister(this.id);
            this.id = -1;
        }
        this.handle = next;
        if (next == 0L) {
            // Attachment not available this frame (e.g. a pass with no depth buffer).
            this.width = 0;
            this.height = 0;
            this.pixelFormat = 0;
            return;
        }
        this.id = MetalHandleMap.register(next);
        this.width = MetalNative.mtlTextureGetWidth(next);
        this.height = MetalNative.mtlTextureGetHeight(next);
        this.pixelFormat = MetalNative.mtlTextureGetPixelFormat(next);
        this.levels = Math.max(1, MetalNative.mtlTextureGetMipmapLevelCount(next));
    }

    /** The raw {@code MTLTexture} handle currently adapted, or 0. */
    public long metalHandle() {
        return this.handle;
    }

    /** MTL pixel format of the adapted texture (MTLPixelFormat numeric value). */
    public int mtlPixelFormat() {
        return this.pixelFormat;
    }

    public String label() {
        return this.label;
    }

    @Override
    public int id() {
        return this.id;
    }

    @Override
    public int getWidth() {
        return this.width;
    }

    @Override
    public int getHeight() {
        return this.height;
    }

    @Override
    public int getLevels() {
        return this.levels;
    }

    @Override
    public int getFormat() {
        return this.pixelFormat;
    }

    @Override
    public int getType() {
        return 0x0DE1; // GL_TEXTURE_2D — the adapter is conceptually a 2D target.
    }

    @Override
    public IGpuTexture store(int format, int levels, int width, int height) {
        // The storage belongs to Metallum; Voxy does not own or resize it.
        return this;
    }

    @Override
    public IGpuTexture createView() {
        return this;
    }

    @Override
    public IGpuTexture name(String name) {
        return this;
    }

    @Override
    public void assertAllocated() {
        if (this.id == -1) {
            throw new IllegalStateException(
                    "MetallumAttachmentTexture[" + this.label + "] not bound to an attachment");
        }
    }

    @Override
    public void free() {
        // Owned by Metallum — drop our registration only.
        if (this.id != -1) {
            MetalHandleMap.unregister(this.id);
            this.id = -1;
        }
        this.handle = 0L;
    }

    @Override
    public void assertNotFreed() {
    }

    @Override
    public boolean isFreed() {
        return this.id == -1 && this.handle == 0L;
    }
}
