package me.cortex.voxy.client.core;

import me.cortex.voxy.client.config.VoxyConfig;
import me.cortex.voxy.client.core.rendering.hierachical.AsyncNodeManager;
import me.cortex.voxy.client.core.rendering.hierachical.HierarchicalOcclusionTraverser;
import me.cortex.voxy.client.core.rendering.hierachical.NodeCleaner;

import java.util.function.BooleanSupplier;

/**
 * The render pipeline. There is one, and it is Metal's.
 *
 * <h2>What used to be here</h2>
 * This class carried a second, complete pipeline for OpenGL: its own colour and SSAO colour textures,
 * a depth-stencil framebuffer and a second framebuffer for the SSAO target, an SSAO compute shader, a
 * final blit with an environmental-fog push block, and the four {@code setup}/{@code finish} hooks that
 * drove them. All of it is deleted, along with the GL backend it existed to drive — this project has no
 * OpenGL path and never will.
 *
 * <p>The Metal path needs none of it. The frame's own attachments are the render targets, the LOD pass
 * is built in {@link AbstractRenderPipeline} against Metallum's colour and depth, and environmental fog
 * is applied per-fragment by {@code quads.frag}'s {@code USE_ENV_FOG} branch using fog parameters packed
 * into the SceneUniform SSBO — so the GL design's separate fog pass, its push block and its final blit
 * have no counterpart here to port.
 *
 * <p>{@link #useEnvFog()} survives because it is a real compile-time decision rather than GL plumbing:
 * it selects the {@code USE_ENV_FOG} define on the terrain pipeline, and {@code MDICSectionRenderer}
 * reads it when building that pipeline's defines.
 */
public class NormalRenderPipeline extends AbstractRenderPipeline {

    private final boolean useEnvFog;

    @Override
    public boolean useEnvFog() {
        return this.useEnvFog;
    }

    protected NormalRenderPipeline(AsyncNodeManager nodeManager, NodeCleaner nodeCleaner,
                                   HierarchicalOcclusionTraverser traversal, BooleanSupplier frexSupplier) {
        super(nodeManager, nodeCleaner, traversal, frexSupplier, false);
        // Metal + Iris-pack gbuffer injection: the pack's own fog/composite shades the injected LOD
        // pixels, so Voxy's per-fragment env fog would double-fog them (and its colour wouldn't match
        // the pack's). Pack state is baked at construction — quads.frag's USE_ENV_FOG is a compile-time
        // define — and VoxyRenderSystem watches for a pack toggle and recreates the renderer (same path
        // as the config toggle).
        //
        // The `metal &&` that used to qualify this was always true: the backend is Metal. Removing it
        // is the whole of the change, not a behaviour change.
        this.useEnvFog = VoxyConfig.CONFIG.useEnvironmentalFog
                && !(me.cortex.voxy.client.core.util.IrisUtil.irisGbufferInjectMode()
                        || me.cortex.voxy.client.core.util.IrisUtil.vxContractActive());
    }

    @Override
    public void free() {
        super.free0();
    }
}
