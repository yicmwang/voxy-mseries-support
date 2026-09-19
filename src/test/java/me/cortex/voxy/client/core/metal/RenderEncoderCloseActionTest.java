package me.cortex.voxy.client.core.metal;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;

/**
 * Pins who is responsible for ending a render encoder.
 *
 * <p>Voxy draws into Metallum's frame, so an encoder can come from one of two places, and each has
 * exactly one owner:
 *
 * <ul>
 *   <li><b>Borrowed</b> -- Metallum's encoder, which it keeps drawing on after Voxy is done. Ending
 *       it here would break Metallum's remaining draws, so only its cached state is invalidated.</li>
 *   <li><b>Voxy's own</b> -- created from the pass descriptor. Voxy ends it and releases it.</li>
 * </ul>
 *
 * <p>The rule is extracted because the LOD pass getting it wrong is expensive and silent. A third
 * option was tried and removed: an encoder Metallum could not see at all ("detached"), so that no
 * pass teardown could touch it. That is not a supported arrangement -- Metallum cannot know an
 * encoder is open, so it will happily open a second one on the same command buffer, which asserts
 * with "A command encoder is already encoding to this command buffer". Borrowing badly is worse
 * than not borrowing; the supported answer is to own the encoder and end it promptly, which is what
 * every other Voxy pass already does.
 */
class RenderEncoderCloseActionTest {

    @Test
    void borrowedEncodersAreOnlyInvalidated() {
        // Metallum still owns the pass and keeps drawing on this encoder after we are done.
        assertEquals(MetalRenderEncoder.CloseAction.INVALIDATE_ONLY,
                MetalRenderEncoder.closeActionFor(true));
    }

    @Test
    void ownedEncodersAreEndedAndReleased() {
        // Voxy created it, so Voxy ends it. This is the path every working Voxy pass takes.
        assertEquals(MetalRenderEncoder.CloseAction.END_AND_RELEASE,
                MetalRenderEncoder.closeActionFor(false));
    }

    @Test
    void theTwoSourcesNeverShareAnOwner() {
        // The property that matters: whichever way an encoder was obtained, exactly one party ends
        // it and it is never "nobody".
        assertNotEquals(MetalRenderEncoder.closeActionFor(true), MetalRenderEncoder.closeActionFor(false),
                "a borrowed and an owned encoder must not have the same close behaviour");
    }
}
