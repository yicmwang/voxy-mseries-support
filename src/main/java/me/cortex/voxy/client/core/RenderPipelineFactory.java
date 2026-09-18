package me.cortex.voxy.client.core;

import me.cortex.voxy.client.core.rendering.hierachical.AsyncNodeManager;
import me.cortex.voxy.client.core.rendering.hierachical.HierarchicalOcclusionTraverser;
import me.cortex.voxy.client.core.rendering.hierachical.NodeCleaner;

import java.util.function.BooleanSupplier;

/**
 * Chooses the render pipeline.
 *
 * <p>P2 removed the Iris branches. Iris is GL-based by construction — it compiles GLSL into GL
 * programs and renders through the GL pipeline — so under whole-frame Metal there is nothing for it
 * to hook. That also removed the "Metal vx material" pipeline (Phase C), whose only purpose was to
 * produce a material g-buffer for an Iris pack's {@code voxy_opaque}/{@code voxy_translucent} to
 * shade GL-side.
 *
 * <p>What is left is the normal pipeline, which is the whole frame under Metal.
 */
public class RenderPipelineFactory {
    public static AbstractRenderPipeline createPipeline(AsyncNodeManager nodeManager, NodeCleaner nodeCleaner,
                                                        HierarchicalOcclusionTraverser traversal,
                                                        BooleanSupplier frexSupplier) {
        return new NormalRenderPipeline(nodeManager, nodeCleaner, traversal, frexSupplier);
    }
}
