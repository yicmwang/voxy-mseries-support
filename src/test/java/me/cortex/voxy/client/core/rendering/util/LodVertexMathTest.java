package me.cortex.voxy.client.core.rendering.util;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Pins the CPU replica of the LOD vertex path to values taken off a live run, so a decode that
 * drifts from {@code quad_format.glsl} fails here instead of sending another game run after a
 * phantom. The reference values come from the {@code VOXY_GEOM_TRACE=1} dumps:
 *
 * <pre>
 *   [#0 idx=3096 bi=0 pos=[1342177279,4026531856] baseVtx=34260 quadIdx=8565 quad=[112984065,0]]
 *   [#0 idxCount=66 baseInstance=1 pos=[1073741824,16]]
 *   [#1 idxCount=1806 baseInstance=0 pos=[1074790399,4294967280]]
 * </pre>
 */
class LodVertexMathTest {

    private static final int QUAD_X = 112984065;   // 0x06BC0001
    private static final int QUAD_Y = 0;

    @Test
    void decodesTheDumpedQuadToASaneFaceSizeAndModel() {
        // Eu32v takes AMOUNT first, then shift. Reading it the other way round produces the
        // nonsense modelId 2,030,592 that this test exists to keep from coming back.
        assertEquals(1, LodVertexMath.face(QUAD_X, QUAD_Y));
        assertEquals(1, LodVertexMath.stateId(QUAD_X, QUAD_Y));
        final int[] size = LodVertexMath.size(QUAD_X, QUAD_Y);
        assertEquals(1, size[0]);
        assertEquals(1, size[1]);
    }

    @Test
    void decodesTheDumpedPositionsToSmallRelativeOffsets() {
        // pos=[1073741824,16] -> lod 4, section (1,0,0)
        assertEquals(4, LodVertexMath.lodLevel(1073741824));
        assertArrayEqualsInt(new int[] {1, 0, 0}, LodVertexMath.lodPosition(1073741824, 16));

        // pos=[1074790399,-16] -> lod 4, section (-1,0,-1); the X/Y/Z fields are SIGNED
        assertEquals(4, LodVertexMath.lodLevel(1074790399));
        assertArrayEqualsInt(new int[] {-1, 0, -1}, LodVertexMath.lodPosition(1074790399, -16));
    }

    /**
     * The hypothesis this replaces: that {@code modelData[modelId].faceData[face]} being zero made
     * every quad zero-area -- a valid draw, no fragments, no error. It is false. The
     * {@code +1/16} on the end terms means the smallest possible face extent is 1/16 of a block,
     * so a quad is never degenerate however the model decodes.
     */
    @Test
    void aZeroFaceDataStillYieldsANonDegenerateQuad() {
        final float[] fs = LodVertexMath.faceSize(0);
        // 1/16 from the end term, plus the epsilon the xz term was shrunk by.
        assertEquals(0.06255f, fs[1], 1e-6f);
        assertEquals(0.06255f, fs[3], 1e-6f);
        assertTrue(fs[1] > 0 && fs[3] > 0);
    }

    @Test
    void faceIndentationMatchesTheEncoder() {
        assertEquals(0.0f, LodVertexMath.faceIndentation(0), 1e-6f);
        assertEquals(1.0f, LodVertexMath.faceIndentation(63 << 16), 1e-6f);   // 63 reads as 64
        assertEquals(0.5f, LodVertexMath.faceIndentation(32 << 16), 1e-6f);
    }

    /**
     * The two properties that decide whether a quad can produce fragments at all: its four corners
     * must not collapse onto one another, and they must land inside Metal's clip volume.
     *
     * <p>An identity MVP puts this quad at (336, 464) — far outside [-1,1] — so the matrix here is
     * an identity with the quad's own base point translated to the origin, which is what a real
     * camera-relative MVP does. The quad is built from a zero {@code faceData}, the worst case.
     */
    @Test
    void aNearbyQuadWithNonZeroAreaLandsInsideTheClipVolume() {
        // Scaled down 100x and translated so this quad's base point (335.9992, 464.0, -0.0008)
        // lands at the origin -- i.e. the same job a camera-relative projection does.
        final float[] mvp = {
                0.01f, 0, 0, 0,
                0, 0.01f, 0, 0,
                0, 0, 0.01f, 0,
                -3.359992f, -4.64f, 0.000008f, 1};
        final float[] corners = LodVertexMath.corners(mvp, new int[] {0, 0, 0},
                4 << 28, 0, QUAD_X, QUAD_Y, 0, false);

        float maxSpread = 0;
        for (int i = 0; i < 4; i++) {
            final float x = corners[i * 4], y = corners[i * 4 + 1];
            final float z = corners[i * 4 + 2], w = corners[i * 4 + 3];
            assertEquals(1.0f, w, 1e-6f);
            assertTrue(LodVertexMath.insideClipVolume(x, y, z, w),
                    "corner " + i + " outside the clip volume: " + x + "," + y + "," + z + "," + w);
            maxSpread = Math.max(maxSpread, Math.abs(x - corners[0]) + Math.abs(y - corners[1]));
        }
        assertTrue(maxSpread > 1e-3f, "all four corners coincide -- a zero-area quad draws nothing");
    }

    /** A point behind the eye is outside Metal's clip volume — the w>0 guard is load-bearing. */
    @Test
    void behindTheCameraIsOutsideTheClipVolume() {
        assertFalse(LodVertexMath.insideClipVolume(0, 0, 0.5f, -1f));
        assertFalse(LodVertexMath.insideClipVolume(0, 0, -0.1f, 1f));   // Metal clips z < 0
        assertFalse(LodVertexMath.insideClipVolume(0, 0, 1.1f, 1f));    // ...and z > w
        assertTrue(LodVertexMath.insideClipVolume(0, 0, 0.5f, 1f));
    }

    private static void assertArrayEqualsInt(final int[] expected, final int[] actual) {
        assertEquals(expected.length, actual.length);
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], actual[i], "index " + i);
        }
    }
}
