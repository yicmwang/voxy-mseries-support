package me.cortex.voxy.client.core.rendering.post;

import me.cortex.voxy.client.core.gpu.shader.ShaderLoader;
import me.cortex.voxy.client.core.gpu.GraphicsPipelineDesc;
import me.cortex.voxy.client.core.gpu.IGpuPipeline;
import me.cortex.voxy.client.core.gpu.PipelineState;
import me.cortex.voxy.client.core.gpu.RenderBackendFactory;
import me.cortex.voxy.client.core.gpu.VertexLayout;

import java.util.Map;

import static org.lwjgl.opengl.GL11C.GL_RGBA8;

/**
 * Helper that compiles a vertex+fragment shader pair into a fullscreen-quad
 * pipeline through {@link me.cortex.voxy.client.core.gpu.RenderBackend#createGraphicsPipeline},
 * so the shaders compile cleanly on Metal.
 *
 * <p>The raw-GL surface this used to carry (a cached glProgram, glUseProgram +
 * glDrawArrays(TRIANGLE_STRIP), a lazily-created push UBO, and the empty VAO
 * the attribute-less draw needed) is gone with the GL backend. On Metal the
 * draw goes through a {@code RenderEncoder}, which takes the pipeline from
 * {@link #pipeline()} and the uniform block from {@code RenderEncoder.setBytes}.
 *
 * <p>{@code GL_RGBA8} below is the abstraction's shared format vocabulary, not a
 * GL call: {@code MetalFormatUtil} translates it.
 */
public class FullscreenBlit {

    private final IGpuPipeline pipeline;

    public FullscreenBlit(String fragId) {
        this("voxy:post/fullscreen.vert", fragId, Map.of());
    }

    public FullscreenBlit(String vertId, String fragId) {
        this(vertId, fragId, Map.of());
    }

    /** Backend-agnostic constructor — supply a defines map for shader permutations. */
    public FullscreenBlit(String vertId, String fragId, Map<String, String> defines) {
        // Default pipeline state — callers manage depth/blend/stencil themselves
        // through their own render pass because the FullscreenBlit draws into
        // someone else's framebuffer (e.g. sourceFB from MC's render target).
        GraphicsPipelineDesc desc = new GraphicsPipelineDesc(
                ShaderLoader.parse(vertId),
                ShaderLoader.parse(fragId),
                defines != null ? defines : Map.of(),
                null, null,           // no MSL — runtime compiler produces it on Metal
                null, null,           // no SPIRV — runtime compiler produces it on Vulkan
                GL_RGBA8,             // color format
                VertexLayout.EMPTY,   // gl_VertexID-driven full-screen quad
                PipelineState.DEFAULT,
                fragId);
        this.pipeline = RenderBackendFactory.get().createGraphicsPipeline(desc);
    }

    public void delete() {
        this.pipeline.close();
    }

    /** The compiled pipeline, for callers drawing it through a RenderEncoder. */
    public IGpuPipeline pipeline() {
        return this.pipeline;
    }
}
