package me.cortex.voxy.client.core.model;

import me.cortex.voxy.client.core.gpu.IGpuBuffer;
import me.cortex.voxy.client.core.gpu.IGpuTexture;
import me.cortex.voxy.client.core.gpu.RenderBackendFactory;
import net.minecraft.client.Minecraft;
import net.minecraft.client.renderer.texture.TextureAtlas;
import net.minecraft.resources.Identifier;

// GL_RGBA8 is a format TOKEN handed to the backend (MetalFormatUtil.glFormatToMetal), not a GL call.
import static org.lwjgl.opengl.GL11.GL_RGBA8;

public class ModelStore {
    /**
     * Bytes per model entry. 64 by default, which is 36 bytes of data followed by 28 bytes of padding
     * ({@code BlockModel._pad[7]}) — the padding is written by nobody, since {@code ModelFactory} fills
     * offsets 0..35 and stops.
     *
     * <p>{@code VOXY_MODEL_TIGHT=1} drops the padding to 36. It must be set together with the shader
     * define of the same name, because the struct in {@code block_model.glsl} defines the layout the
     * GPU reads; {@code MDICSectionRenderer} injects the define from the same environment variable, so
     * flipping one flag moves both sides.
     *
     * <p>It exists as a bytes-vs-latency discriminator: the load count and every index are identical
     * between the two settings, so a change in {@code gpu} is attributable purely to bytes per element.
     */
    public static final int MODEL_SIZE = "1".equals(System.getenv("VOXY_MODEL_TIGHT")) ? 36 : 64;
    final IGpuBuffer modelBuffer;
    final IGpuBuffer modelColourBuffer;
    final IGpuTexture textures;
    /**
     * Sampler for {@link #textures}, used by the render encoder path
     * (MDIC's renderTerrainMetal).
     */
    public final me.cortex.voxy.client.core.gpu.IGpuSampler atlasSampler;

    public ModelStore() {
        this.modelBuffer = RenderBackendFactory.get().createBuffer(MODEL_SIZE * (1<<16));
        this.modelColourBuffer = RenderBackendFactory.get().createBuffer(4 * (1<<16));
        // Allocate the model atlas as CPU-uploadable: Shared storage so
        // `uploadSubImage2D` can push the bakery results into it.
        this.textures = RenderBackendFactory.get().createTexture()
                .storeUploadable(GL_RGBA8,
                        Integer.numberOfTrailingZeros(ModelFactory.MODEL_TEXTURE_SIZE),
                        ModelFactory.MODEL_TEXTURE_SIZE*3*256,
                        ModelFactory.MODEL_TEXTURE_SIZE*2*256)
                .name("ModelTextures");


        //Limit the mips of the texture to match that of the terrain atlas
        int mipLvl = ((TextureAtlas) Minecraft.getInstance().getTextureManager()
                .getTexture(Identifier.fromNamespaceAndPath("minecraft", "textures/atlas/blocks.png")))
                .maxMipLevel;

        this.atlasSampler = RenderBackendFactory.get().createSampler(
                me.cortex.voxy.client.core.gpu.SamplerDesc.builder()
                        .filter(me.cortex.voxy.client.core.gpu.SamplerDesc.Filter.NEAREST,
                                me.cortex.voxy.client.core.gpu.SamplerDesc.Filter.NEAREST)
                        .mipFilter(me.cortex.voxy.client.core.gpu.SamplerDesc.MipFilter.LINEAR)
                        .wrap(me.cortex.voxy.client.core.gpu.SamplerDesc.Wrap.CLAMP_TO_EDGE,
                                me.cortex.voxy.client.core.gpu.SamplerDesc.Wrap.CLAMP_TO_EDGE)
                        .lod(0, mipLvl)
                        .label("ModelAtlasSampler")
                        .build());
    }


    public void free() {
        this.modelBuffer.free();
        this.modelColourBuffer.free();
        this.textures.free();
        this.atlasSampler.close();
    }


    /**
     * The {@code modelData[]} SSBO the shaders read. Exposed so a diagnostic can decode
     * {@code modelData[stateId(quad)].faceData[face]} on the CPU and check it against what the
     * vertex shader would compute from the same entry.
     */
    public IGpuBuffer getModelBuffer() {
        return this.modelBuffer;
    }

    /**
     * Encoder-aware bind — model + colour SSBOs plus the model atlas
     * texture + sampler. Used by Metal's MDIC render path.
     */
    public void bindBuffers(me.cortex.voxy.client.core.gpu.RenderEncoder encoder,
                            int modelBindingIndex, int colourBindingIndex,
                            int atlasBindingIndex) {
        encoder.setBuffer(modelBindingIndex, this.modelBuffer, 0);
        encoder.setBuffer(colourBindingIndex, this.modelColourBuffer, 0);
        encoder.setTexture(atlasBindingIndex, this.textures);
        encoder.setSampler(atlasBindingIndex, this.atlasSampler);
    }
}
