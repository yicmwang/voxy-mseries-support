package me.cortex.voxy.client.core.rendering.section.geometry;

import me.cortex.voxy.client.core.rendering.building.BuiltSection;

import java.util.function.Consumer;

public interface IGeometryManager {
    int uploadSection(BuiltSection section);
    int uploadReplaceSection(int oldId, BuiltSection section);
    void removeSection(int id);

    void downloadAndRemove(int id, Consumer<BuiltSection> callback);

    /**
     * Called once per frame on the RENDER thread, after the frame's draws have been submitted.
     *
     * <p>Its job is to return geometry addresses that were freed by {@link #removeSection} back to the
     * allocation arena, once no in-flight command buffer can still be reading them. See
     * {@code BasicAsyncGeometryManager.releaseRetiredFrees} for why that delay exists at all; the short
     * version is that upstream frees immediately, which is safe in GL (one ordered command stream) and
     * is a race on Metal, where up to {@code MAX_SUBMITS_IN_FLIGHT} command buffers can be drawing from
     * an address that has already been handed to a different section.
     *
     * @param frameId the renderer's frame counter, used only to age the pending frees
     */
    default void releaseRetiredFrees(long frameId) {}
}
