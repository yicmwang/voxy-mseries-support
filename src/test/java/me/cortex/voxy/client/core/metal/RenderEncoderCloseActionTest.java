package me.cortex.voxy.client.core.metal;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Pins who is responsible for ending a render encoder. This is the rule whose absence made Voxy's
 * LOD render zero pixels.
 *
 * <p>The bisection that found it: a magenta triangle drawn through Voxy's encoder at the LOD pass
 * produced 0.00% of the frame, while the same triangle drawn through an encoder created directly
 * from the command buffer and ended immediately produced 17.13%. Same point in the frame, same
 * depth state, same pipeline shape -- only the encoder differed. Corroborating it, forcing Voxy to
 * end its borrowed encoder tripped {@code endEncoding has already been called}, i.e. the encoder had
 * died before Voxy drew on it.
 *
 * <p>A GPU-free unit test cannot prove pixels appear, but it can hold still the ownership rule that
 * decides whether an encoder is ever ended by the party that owns it. That rule is one branch, and
 * it is the branch that was wrong.
 */
class RenderEncoderCloseActionTest {

    @Test
    void detachedEncodersAreEndedByUsViaMetallum() {
        // A detached encoder is untracked: no pass teardown will touch it, so if we do not end it,
        // its contents are discarded when the command buffer commits. Ending it through Metallum
        // also drops the wrapper it holds, keeping the ObjC lifetime balanced.
        assertEquals(MetalRenderEncoder.CloseAction.END_VIA_METALLUM,
                MetalRenderEncoder.closeActionFor(true, false));
    }

    @Test
    void borrowedEncodersAreOnlyInvalidated() {
        // Metallum still owns the pass and keeps drawing on this encoder after we are done; ending
        // it here would break Metallum's remaining draws.
        assertEquals(MetalRenderEncoder.CloseAction.INVALIDATE_ONLY,
                MetalRenderEncoder.closeActionFor(false, true));
    }

    @Test
    void ownedEncodersAreEndedAndReleased() {
        // Voxy created it on its own command buffer, so Voxy ends it.
        assertEquals(MetalRenderEncoder.CloseAction.END_AND_RELEASE,
                MetalRenderEncoder.closeActionFor(false, false));
    }

    @Test
    void detachedWinsOverBorrowed() {
        // The two flags should never both be set, but if a future edit sets both the detached
        // reading is the safe one: an encoder Metallum does not know about must not be left open on
        // the assumption that Metallum will close it.
        assertEquals(MetalRenderEncoder.CloseAction.END_VIA_METALLUM,
                MetalRenderEncoder.closeActionFor(true, true));
    }

    @Test
    void everyEncodingSourceHasExactlyOneOwner() {
        // The property that matters: for each way an encoder can be obtained there is exactly one
        // party responsible for ending it, and it is never "nobody" -- which is the state that made
        // the LOD invisible.
        for (boolean detached : new boolean[]{true, false}) {
            for (boolean borrowed : new boolean[]{true, false}) {
                MetalRenderEncoder.CloseAction action = MetalRenderEncoder.closeActionFor(detached, borrowed);
                assertEquals(detached ? MetalRenderEncoder.CloseAction.END_VIA_METALLUM
                                : (borrowed ? MetalRenderEncoder.CloseAction.INVALIDATE_ONLY
                                            : MetalRenderEncoder.CloseAction.END_AND_RELEASE),
                        action,
                        "detached=" + detached + " borrowed=" + borrowed);
            }
        }
    }
}
