package me.cortex.voxy.client.core.rendering.section.backend.mdic;

import me.cortex.voxy.client.core.gpu.PipelineState;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Pins the depth compare the LOD pass uses. This is the decision that made Voxy's LOD render zero
 * pixels, and nothing else could have caught it:
 *
 * <p>Metallum's frame depth buffer is reverse-Z (near=1, far=0, compare GreaterEqual -- the restored
 * P0 probe reads that off the live pass). Voxy compared {@code LessEqual} against it. Over sky the
 * buffer holds 0, while a Voxy fragment's depth is a large positive number in the GL convention, so
 * {@code fragDepth <= 0} is false for EVERY fragment. The draws were issued, the geometry was
 * correct, the commands were valid and the render target was right -- the fragments were simply all
 * rejected. Draw counts, command dumps and the render target identity are all upstream of this, so
 * every one of them measured healthy while the LOD contributed nothing.
 *
 * <p>The rule is small and pure, which is exactly what makes it worth holding still -- the same
 * reasoning as {@code ActiveCommandBufferAdoptionTest}, where the failure mode was also expensive
 * to rediscover and cheap to pin.
 */
class LodDepthStateTest {

    @Test
    void usesGreaterEqualOnAReverseZFrame() {
        PipelineState.DepthState state = MDICSectionRenderer.lodDepthState(true, false, true);
        assertEquals(PipelineState.CompareOp.GREATER_EQUAL, state.compareOp,
                "a reverse-Z buffer needs GreaterEqual; LessEqual rejects every fragment");
        assertTrue(state.testEnabled, "the depth test must still run -- this is a convention fix, not a disable");
        assertTrue(state.writeEnabled, "the LOD must keep writing depth so it self-occludes");
    }

    @Test
    void usesLessEqualOnAConventionalFrame() {
        // The GL path and any non-reverse-Z target keep the previous behaviour, so this fix cannot
        // regress them.
        PipelineState.DepthState state = MDICSectionRenderer.lodDepthState(false, false, true);
        assertEquals(PipelineState.CompareOp.LESS_EQUAL, state.compareOp);
        assertTrue(state.testEnabled);
        assertTrue(state.writeEnabled);
    }

    @Test
    void theTwoConventionsDisagreeOnlyInDirection() {
        // The comparison really is inverted between them -- not merely different constants -- which
        // is what makes getting it wrong silently fatal rather than visibly wrong.
        PipelineState.DepthState reverseZ = MDICSectionRenderer.lodDepthState(true, false, true);
        PipelineState.DepthState normalZ = MDICSectionRenderer.lodDepthState(false, false, true);
        assertEquals(PipelineState.CompareOp.GREATER_EQUAL, reverseZ.compareOp);
        assertEquals(PipelineState.CompareOp.LESS_EQUAL, normalZ.compareOp);
        assertEquals(reverseZ.testEnabled, normalZ.testEnabled);
        assertEquals(reverseZ.writeEnabled, normalZ.writeEnabled);
    }

    @Test
    void theNoDepthSwitchStillWins() {
        // VOXY_LOD_NO_DEPTH=1 is a diagnostic that must keep meaning "no depth test at all",
        // whichever convention the frame uses -- otherwise the switch silently stops working and
        // any bisection built on it reports a false negative.
        for (boolean reverseZ : new boolean[]{true, false}) {
            PipelineState.DepthState state = MDICSectionRenderer.lodDepthState(reverseZ, true, true);
            assertFalse(state.testEnabled, "depth test must be off (reverseZ=" + reverseZ + ")");
            assertFalse(state.writeEnabled, "depth write must be off (reverseZ=" + reverseZ + ")");
            assertEquals(PipelineState.CompareOp.ALWAYS, state.compareOp);
        }
    }

    @Test
    void neverReturnsAStateThatRejectsEverythingOnAReverseZFrame() {
        // The specific regression, stated directly: on a reverse-Z frame the LOD must not end up
        // with the LessEqual default, which is what made it invisible.
        assertFalse(MDICSectionRenderer.lodDepthState(true, false, true).compareOp == PipelineState.CompareOp.LESS_EQUAL,
                "LessEqual against a reverse-Z buffer is the zero-pixel bug");
    }

    /**
     * The translucent pass takes the SAME operator and differs only in the write flag. This is the
     * half that was missed: the opaque pass was fixed for reverse-Z and water was left on the
     * GL-convention presets, which inverted the water test in both directions at once -- water drew
     * in front of terrain it should have been hidden by, and was hidden by terrain it should have
     * drawn in front of.
     */
    @Test
    void translucentSharesTheOperatorAndOnlyDropsTheWrite() {
        PipelineState.DepthState opaque = MDICSectionRenderer.lodDepthState(true, false, true);
        PipelineState.DepthState water = MDICSectionRenderer.lodDepthState(true, false, false);
        assertEquals(opaque.compareOp, water.compareOp,
                "both passes read the same depth buffer, so they must agree on the operator");
        assertTrue(water.testEnabled, "water must still be occluded by opaque terrain");
        assertFalse(water.writeEnabled, "water must not write depth, or it occludes itself");

        PipelineState.DepthState waterGl = MDICSectionRenderer.lodDepthState(false, false, false);
        assertEquals(PipelineState.CompareOp.LESS_EQUAL, waterGl.compareOp);
        assertTrue(waterGl.testEnabled);
        assertFalse(waterGl.writeEnabled);
    }

    @Test
    void waterDebugStillForcesTheTestOffOnEitherConvention() {
        for (boolean reverseZ : new boolean[]{true, false}) {
            PipelineState.DepthState state = MDICSectionRenderer.lodDepthState(reverseZ, true, false);
            assertFalse(state.testEnabled, "VOXY_LOD_WATER_DEBUG must show all water (reverseZ=" + reverseZ + ")");
        }
    }
}
