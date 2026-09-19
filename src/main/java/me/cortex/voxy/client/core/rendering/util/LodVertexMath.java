package me.cortex.voxy.client.core.rendering.util;

/**
 * A CPU replica of the LOD vertex path — {@code setupQuad} + {@code getQuadCornerPos} from
 * {@code assets/voxy/shaders/lod/quad_util.glsl}, together with the {@code quad_format.glsl} /
 * {@code block_model.glsl} decoders they call.
 *
 * <p>Why this exists: everything else in the Metal LOD chain has been cleared one layer at a time
 * (encoder, render pass, attachments, pixel formats, viewport, pipeline state, bindings, indirect
 * commands, depth convention), and the LOD still contributes no pixels while a forced clip-space
 * triangle through the SAME pipeline fills the screen. That leaves the vertex INPUTS — and the
 * only way to tell "the inputs are garbage" from "the inputs are fine and the shader misbehaves"
 * without another GPU experiment is to compute the same transform here and look at the result.
 *
 * <p>It is a deliberate line-by-line transcription, so it must be kept in step with the shaders:
 * the point of the class is to be exactly as wrong as the GPU is, which it cannot be if it drifts.
 * Every non-obvious line below cites the GLSL it mirrors.
 *
 * <p>All positions are {@code float} and all decoding is integer, matching the {@code ivec2} branch
 * of {@code quad_format.glsl} (the one MSL takes, since MSL has no {@code uint64_t} path).
 */
public final class LodVertexMath {

    private LodVertexMath() {}

    /**
     * {@code Eu32v(ivec2 data, int amount, int shift)} — note the argument order: AMOUNT first,
     * then shift. Shifts past bit 31 read {@code data.y}.
     */
    static int eu32v(final int x, final int y, final int amount, final int shift) {
        if (shift > 31) {
            return (y >>> (shift - 32)) & ((1 << amount) - 1);
        }
        return (x >>> shift) & ((1 << amount) - 1);
    }

    /** GLSL {@code bitfieldExtract(int, offset, bits)} — extracts and SIGN-EXTENDS. */
    static int bitfieldExtract(final int value, final int offset, final int bits) {
        final int raw = (value >>> offset) & ((1 << bits) - 1);
        final int sh = 32 - bits;
        return (raw << sh) >> sh;
    }

    /** {@code extractDetail(uvec2)} — {@code encPos.x >> 28}. */
    public static int lodLevel(final int sPosX) {
        return sPosX >>> 28;
    }

    /**
     * {@code extractLoDPosition(uvec2 encPos)} — three GLSL {@code bitfieldExtract} calls. These
     * are signed: the packed coordinates are relative and legitimately negative.
     */
    public static int[] lodPosition(final int sPosX, final int sPosY) {
        final int y = bitfieldExtract(sPosX, 20, 8);
        final int x = bitfieldExtract(sPosY, 4, 24);
        final int z = bitfieldExtract(((sPosX & ((1 << 20) - 1)) << 4) | (sPosY >>> 28), 0, 24);
        return new int[] {x, y, z};
    }

    /** {@code extractFace(quad)} = {@code Eu32v(quad, 3, 0)}. */
    public static int face(final int qx, final int qy) {
        return eu32v(qx, qy, 3, 0);
    }

    /** {@code extractSize(quad)} = {@code ivec2(Eu32v(quad,4,3), Eu32v(quad,4,7)) + 1}. */
    public static int[] size(final int qx, final int qy) {
        return new int[] {eu32v(qx, qy, 4, 3) + 1, eu32v(qx, qy, 4, 7) + 1};
    }

    /** {@code extractStateId(quad)} = {@code Eu32v(quad,6,26) | (Eu32v(quad,14,32) << 6)}. */
    public static int stateId(final int qx, final int qy) {
        return eu32v(qx, qy, 6, 26) | (eu32v(qx, qy, 14, 32) << 6);
    }

    /** {@code extractPos(quad)} = {@code vec3(Eu32v(quad,5,21), Eu32v(quad,5,16), Eu32v(quad,5,11))}. */
    public static float[] pos(final int qx, final int qy) {
        return new float[] {eu32v(qx, qy, 5, 21), eu32v(qx, qy, 5, 16), eu32v(qx, qy, 5, 11)};
    }

    /** {@code extractFaceIndentation(faceData)}. */
    public static float faceIndentation(final int faceData) {
        int enc = (faceData >> 16) & 63;
        if (enc == 63) enc = 64;
        return enc / 64.0f;
    }

    /**
     * {@code getFaceSize(faceData)} — {@code extractFaceSizes} followed by the shrink-by-epsilon /
     * make-end-relative-to-start pair. The epsilon cancels, so {@code yw} end up as the plain
     * end-minus-start extents.
     */
    public static float[] faceSize(final int faceData) {
        final float eps = 0.00005f;
        final float[] v = new float[] {
                (faceData & 0xF) / 16.0f,
                ((faceData >> 4) & 0xF) / 16.0f + 1.0f / 16.0f,
                ((faceData >> 8) & 0xF) / 16.0f,
                ((faceData >> 12) & 0xF) / 16.0f + 1.0f / 16.0f,
        };
        v[0] -= eps;                       // faceOffsetsSizes.xz -= vec2(EPSILON)
        v[2] -= eps;
        v[1] -= v[0];                      // faceOffsetsSizes.yw -= faceOffsetsSizes.xz
        v[3] -= v[2];
        return v;
    }

    /** {@code swizzelDataAxis(axis, data)}. */
    static float[] swizzle(final int axis, final float[] d) {
        if (axis == 1) return new float[] {d[0], d[1], d[2]};
        if (axis == 0) return new float[] {d[0], d[2], d[1]};
        return new float[] {d[2], d[1], d[0]};
    }

    /**
     * The four clip-space corners of one quad, in corner-id order, exactly as
     * {@code setupQuad} + {@code getQuadCornerPos} would produce them on the GPU.
     *
     * @param mvp          the uploaded MVP, JOML/column-major (element {@code [c*4+r]})
     * @param baseSectionPos  the {@code SceneUniform.baseSectionPos} the shader reads
     * @param quad         the {@code quadData[gl_VertexID>>2]} entry, as its two raw ints
     * @param faceData     {@code modelData[stateId(quad)].faceData[face(quad)]}
     * @param absIndent    whether {@code VOXY_LOD_ABS_INDENT} is in effect (Metal injects it)
     * @return 4 corners x (x,y,z,w)
     */
    public static float[] corners(final float[] mvp, final int[] baseSectionPos,
                                  final int sPosX, final int sPosY,
                                  final int quadX, final int quadY,
                                  final int faceData, final boolean absIndent) {
        final int lodLevel = lodLevel(sPosX);
        final float lodScale = (float) (1 << lodLevel);

        final int[] lodPos = lodPosition(sPosX, sPosY);
        final int[] baseSection = new int[] {
                lodPos[0] * (1 << lodLevel) - baseSectionPos[0],
                lodPos[1] * (1 << lodLevel) - baseSectionPos[1],
                lodPos[2] * (1 << lodLevel) - baseSectionPos[2],
        };

        final int face = face(quadX, quadY);
        final int[] quadSize = size(quadX, quadY);
        final float[] fs = faceSize(faceData);

        float depthOffset = faceIndentation(faceData);
        if (absIndent) depthOffset /= lodScale;

        final float[] quadStart = pos(quadX, quadY);
        final float[] faceOffset = swizzle(face >> 1, new float[] {
                fs[0], fs[2], (depthOffset * (1 - (face & 1))) + ((1 - depthOffset) * (face & 1))});
        for (int i = 0; i < 3; i++) quadStart[i] += faceOffset[i];

        final float[] basePoint = new float[] {
                quadStart[0] * lodScale + baseSection[0] * 32.0f,
                quadStart[1] * lodScale + baseSection[1] * 32.0f,
                quadStart[2] * lodScale + baseSection[2] * 32.0f,
        };
        final int axis = face >> 1;
        final float[] quadSizeAddin = new float[] {fs[1] + quadSize[0] - 1, fs[3] + quadSize[1] - 1};

        final float[] out = new float[16];
        for (int cornerId = 0; cornerId < 4; cornerId++) {
            final float[] cornerMask = new float[] {
                    ((cornerId >> 1) & 1) * lodScale, (cornerId & 1) * lodScale};
            final float[] offset = swizzle(axis, new float[] {
                    quadSizeAddin[0] * cornerMask[0], quadSizeAddin[1] * cornerMask[1], 0f});
            final float[] point = new float[] {
                    basePoint[0] + offset[0], basePoint[1] + offset[1], basePoint[2] + offset[2]};
            for (int r = 0; r < 4; r++) {
                out[cornerId * 4 + r] = mvp[0 * 4 + r] * point[0]
                        + mvp[1 * 4 + r] * point[1]
                        + mvp[2 * 4 + r] * point[2]
                        + mvp[3 * 4 + r];
            }
        }
        return out;
    }

    /** {@code -w <= x <= w && -w <= y <= w && 0 <= z <= w} — Metal's visible clip volume. */
    public static boolean insideClipVolume(final float x, final float y, final float z, final float w) {
        return w > 0 && x >= -w && x <= w && y >= -w && y <= w && z >= 0 && z <= w;
    }
}
