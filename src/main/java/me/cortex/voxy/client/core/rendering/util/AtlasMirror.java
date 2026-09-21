package me.cortex.voxy.client.core.rendering.util;

import me.cortex.voxy.client.core.gpu.IGpuSampler;
import me.cortex.voxy.client.core.gpu.IGpuTexture;
import me.cortex.voxy.client.core.gpu.RenderBackend;
import me.cortex.voxy.client.core.gpu.RenderBackendFactory;
import me.cortex.voxy.client.core.gpu.SamplerDesc;
import me.cortex.voxy.client.core.metal.MetalTexture;
import org.lwjgl.system.MemoryUtil;

import static org.lwjgl.opengl.GL11C.GL_RGBA;
import static org.lwjgl.opengl.GL11C.GL_RGBA8;
import static org.lwjgl.opengl.GL11C.GL_TEXTURE_2D;
import static org.lwjgl.opengl.GL11C.GL_TEXTURE_WIDTH;
import static org.lwjgl.opengl.GL11C.GL_TEXTURE_HEIGHT;
import static org.lwjgl.opengl.GL11C.GL_UNSIGNED_BYTE;
import static org.lwjgl.opengl.GL11C.glBindTexture;
import static org.lwjgl.opengl.GL11C.glGetInteger;
import static org.lwjgl.opengl.GL11C.glGetTexLevelParameteri;
import static org.lwjgl.opengl.GL11C.GL_TEXTURE_BINDING_2D;
import static org.lwjgl.opengl.GL11C.GL_PACK_ALIGNMENT;
import static org.lwjgl.opengl.GL11C.glPixelStorei;
import static org.lwjgl.opengl.GL11.GL_PACK_ROW_LENGTH;
import static org.lwjgl.opengl.GL11.GL_PACK_SKIP_PIXELS;
import static org.lwjgl.opengl.GL11.GL_PACK_SKIP_ROWS;
import static org.lwjgl.opengl.GL12.GL_PACK_IMAGE_HEIGHT;
import static org.lwjgl.opengl.GL12.GL_PACK_SKIP_IMAGES;
import static org.lwjgl.opengl.GL13C.GL_ACTIVE_TEXTURE;
import static org.lwjgl.opengl.GL13C.GL_TEXTURE0;
import static org.lwjgl.opengl.GL13C.glActiveTexture;

/**
 * M13 chunk 1 foundation (2026-05-13): mirrors MC's block atlas GL texture
 * into a Shared-storage Metal texture so the future Metal-native bakery can
 * sample it without crossing the GL ↔ Metal context boundary.
 *
 * <p>The bakery currently auto-gates off on Metal because Apple's GL stack
 * crashes inside the pixel-processor on FBO readback (see
 * {@code project_m12_closed_m13_in_progress} memory). The replacement path
 * keeps the entire bake on the Metal side: the bake target is a Metal
 * texture; this helper provides the source.
 *
 * <p>Lifecycle: lazy on first call to {@link #syncMetal}, single CPU readback
 * of MC's atlas via {@code nglGetTexImage} (Apple GL 4.1 — no DSA available),
 * upload to a Shared Metal texture via {@link IGpuTexture#uploadSubImage2D}.
 * Atlas dimensions are queried at first sync. Re-sync is gated by the source
 * GL texture id — if a resource pack reload swaps the atlas, the mirror gets
 * rebuilt; otherwise it's a free hot path. Mipmap levels are out of scope
 * for the MVP — level 0 only; the bakery's `textureGrad` samples will lose
 * a small amount of distance smoothness, traded for not having to plumb
 * level dimensions through `glGetTexLevelParameteri` per level.
 *
 * <p>Not stenciled for thread-safety — the bakery runs on the render thread
 * inside MC's GL context, same as this helper.
 *
 * <p>Memory: the atlas is typically 1024×1024 or 2048×2048 RGBA8 — 4–16 MB.
 * The staging buffer is a single pinned native allocation; resizes on atlas
 * dimension change.
 */
public final class AtlasMirror {
    private final RenderBackend backend = RenderBackendFactory.get();
    private IGpuTexture mirror;
    private IGpuSampler sampler;
    private long stagingAddr;
    private long stagingSize;
    private int width;
    private int height;
    private int lastSyncedGlId = -1;
    /** Re-sync up to this many times before locking the cache. Lets MC's
     * textures finish loading before the mirror freezes a grey/placeholder.
     * Each sync is a full atlas readback (Apple GL fragile pixel processor),
     * so keep bounded. Most MC worlds finish texture-load within ~10s and the
     * bakery starts after that; this is a safety margin. */
    private static final int WARMUP_MAX_SYNCS = 50;
    private int warmupSyncCount = 0;

    /**
     * Sync the mirror from MC's GL atlas texture {@code mcAtlasGlId}. Returns
     * the Metal-side {@link IGpuTexture} the bakery should bind. Cheap when
     * the source id hasn't changed since the last call (single int compare).
     *
     * <p>Metal-only by construction — there is no GL render path to mirror for.
     */
    public IGpuTexture syncMetal(int mcAtlasGlId) {
        if (mcAtlasGlId == 0) {
            return this.mirror; // possibly null on the very first call before MC is ready
        }
        // M13 chunk 1 polish (2026-05-16): originally we cached on `mcAtlasGlId
        // == lastSyncedGlId`. That caches a stale read from BEFORE MC finishes
        // loading textures — the readback returns a grey/placeholder atlas
        // (the [Metal-REORDER] diagnostic shows bake-target pixels filled
        // with ff515151 = pure-grey 0x51), and the cache hit then prevents any
        // later real-texture sync. The user sees this as LOD chunks rendering
        // grey + transparent. Allow up to N "warm-up" re-syncs so an early
        // bake grabs the atlas as it is at that moment, but later bakes pick
        // up MC's real textures once they're loaded. After N readbacks we
        // stop re-syncing to avoid the per-bake `nglGetTexImage` cost (which
        // hits Apple GL's fragile pixel-processor — keep the rate bounded).
        boolean idMatches = (mcAtlasGlId == this.lastSyncedGlId) && this.mirror != null;
        boolean stillWarming = this.warmupSyncCount < WARMUP_MAX_SYNCS;
        if (idMatches && !stillWarming) {
            return this.mirror;
        }

        // Capture MC's current 2D binding so we can restore it.
        int prevActive = glGetInteger(GL_ACTIVE_TEXTURE);
        glActiveTexture(GL_TEXTURE0);
        int prevBinding = glGetInteger(GL_TEXTURE_BINDING_2D);
        // SIGBUS hardening (2026-05-26): glGetTexImage honours the GL_PACK_*
        // pixel-store state. MC/Sodium — notably a screenshot's glReadPixels —
        // can leave GL_PACK_ROW_LENGTH / skips non-default; the readback would
        // then stride past stagingAddr and _platform_memmove SIGBUSes (same
        // crash class as the LightMapHelper fix). Force pack params to defaults
        // and restore them in finally so MC's state is undisturbed.
        int prevRowLen  = glGetInteger(GL_PACK_ROW_LENGTH);
        int prevSkipPix = glGetInteger(GL_PACK_SKIP_PIXELS);
        int prevSkipRow = glGetInteger(GL_PACK_SKIP_ROWS);
        int prevSkipImg = glGetInteger(GL_PACK_SKIP_IMAGES);
        int prevImgH    = glGetInteger(GL_PACK_IMAGE_HEIGHT);
        int prevAlign   = glGetInteger(GL_PACK_ALIGNMENT);
        try {
            glBindTexture(GL_TEXTURE_2D, mcAtlasGlId);
            int w = glGetTexLevelParameteri(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH);
            int h = glGetTexLevelParameteri(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT);
            if (w <= 0 || h <= 0) {
                return this.mirror; // atlas not yet ready
            }
            ensureResources(w, h);
            glPixelStorei(GL_PACK_ROW_LENGTH, 0);
            glPixelStorei(GL_PACK_SKIP_PIXELS, 0);
            glPixelStorei(GL_PACK_SKIP_ROWS, 0);
            glPixelStorei(GL_PACK_SKIP_IMAGES, 0);
            glPixelStorei(GL_PACK_IMAGE_HEIGHT, 0);
            glPixelStorei(GL_PACK_ALIGNMENT, 4);
            // Read level 0 only. The bakery shaders use textureGrad on the
            // atlas; without mipmaps the LOD selection effectively snaps to
            // the base level. For the MVP this matches what the per-block
            // bake actually needs (each face samples a small UV region at
            // close to 1:1 texel-to-pixel ratio inside the 16×16 bake cell).
            org.lwjgl.opengl.GL11C.nglGetTexImage(
                    GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, this.stagingAddr);
        } finally {
            glPixelStorei(GL_PACK_ROW_LENGTH, prevRowLen);
            glPixelStorei(GL_PACK_SKIP_PIXELS, prevSkipPix);
            glPixelStorei(GL_PACK_SKIP_ROWS, prevSkipRow);
            glPixelStorei(GL_PACK_SKIP_IMAGES, prevSkipImg);
            glPixelStorei(GL_PACK_IMAGE_HEIGHT, prevImgH);
            glPixelStorei(GL_PACK_ALIGNMENT, prevAlign);
            glBindTexture(GL_TEXTURE_2D, prevBinding);
            glActiveTexture(prevActive);
        }

        this.mirror.uploadSubImage2D(0, 0, 0, this.width, this.height,
                GL_RGBA, GL_UNSIGNED_BYTE, this.stagingAddr);
        this.lastSyncedGlId = mcAtlasGlId;
        this.warmupSyncCount++;
        return this.mirror;
    }

    /**
     * Sampler conventions for sampling MC's block atlas in the bakery.
     *
     * <p>Shared by both atlas sources — the mirrored GL texture and the direct Metal one — so a block
     * cannot bake differently depending on which path produced its atlas. NEAREST mip filter (Voxy's
     * terrain shader uses {@code textureGrad} and needs derivatives, but the bakery shaders use
     * {@code textureLod}); MIN/MAG are LINEAR so the bake does not look pixelated when projected
     * through the 6-face cube transforms.
     */
    public static IGpuSampler createAtlasSampler(final RenderBackend backend) {
        return backend.createSampler(SamplerDesc.builder()
                .filter(SamplerDesc.Filter.LINEAR, SamplerDesc.Filter.LINEAR)
                .mipFilter(SamplerDesc.MipFilter.NEAREST)
                .wrap(SamplerDesc.Wrap.CLAMP_TO_EDGE, SamplerDesc.Wrap.CLAMP_TO_EDGE)
                .label("Voxy.MCBlockAtlasSampler")
                .build());
    }

    public IGpuTexture texture() { return this.mirror; }
    public IGpuSampler sampler() { return this.sampler; }
    public int width()  { return this.width; }
    public int height() { return this.height; }

    private void ensureResources(int w, int h) {
        if (this.mirror == null || this.width != w || this.height != h) {
            if (this.mirror != null) this.mirror.free();
            this.width = w;
            this.height = h;
            MetalTexture tex = (MetalTexture) this.backend.createTexture(GL_TEXTURE_2D);
            tex.storeUploadable(GL_RGBA8, 1, w, h);
            tex.name("Voxy.MCBlockAtlasMirror");
            this.mirror = tex;

            long needed = (long) w * h * 4L;
            if (this.stagingAddr != 0L && needed != this.stagingSize) {
                MemoryUtil.nmemFree(this.stagingAddr);
                this.stagingAddr = 0L;
            }
            if (this.stagingAddr == 0L) {
                this.stagingAddr = MemoryUtil.nmemAllocChecked(needed);
                this.stagingSize = needed;
            }
            // Mirror dimensions changed: force a re-upload on next call.
            this.lastSyncedGlId = -1;
        }
        if (this.sampler == null) {
            this.sampler = createAtlasSampler(this.backend);
        }
    }

    public void free() {
        if (this.mirror != null) { this.mirror.free(); this.mirror = null; }
        if (this.sampler != null) { this.sampler.close(); this.sampler = null; }
        if (this.stagingAddr != 0L) {
            MemoryUtil.nmemFree(this.stagingAddr);
            this.stagingAddr = 0L;
            this.stagingSize = 0L;
        }
    }
}
