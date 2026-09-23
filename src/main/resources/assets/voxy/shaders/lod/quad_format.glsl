#ifdef GL_ARB_gpu_shader_int64
#define Quad uint64_t

#define Eu32(data, amountBits, shift) (uint((data)>>(shift))&((1u<<(amountBits))-1))

vec3 extractPos(uint64_t quad) {
    //TODO: pull out the majic constants into #defines (specifically the shift amount)
    return vec3(Eu32(quad, 5, 21), Eu32(quad, 5, 16), Eu32(quad, 5, 11));
}

ivec2 extractSize(uint64_t quad) {
    return ivec2(Eu32(quad, 4, 3), Eu32(quad, 4, 7)) + ivec2(1);//the + 1 is cause you cant actually have a 0 size quad
}

uint extractFace(uint64_t quad) {
    return Eu32(quad, 3, 0);
}

uint extractStateId(uint64_t quad) {
    return Eu32(quad, 16, 26);
}

uint extractBiomeId(uint64_t quad) {
    return Eu32(quad, 9, 46);
}

uint extractLightId(uint64_t quad) {
    return Eu32(quad, 8, 55);
}

bool isQuadEmpty(uint64_t quad) {
    return quad == uint64_t(0);
}

#else
//TODO: FIXME, ivec2 swaps around the data of the x and y cause its written in little endian

// ivec3, not ivec2: the third int carries the quad's own faceModelData, which the vertex stage needs
// for the face size and the face indentation. Baking it here is what lets the per-vertex path skip the
// `modelData[modelId]` load -- see quad_util.glsl's setupQuad and optimisation.MD §33.
//
// That load measured 14.3 % of the LOD pass's gpu (removing it entirely), and the cost is the LOAD'S
// LATENCY, not the bytes: shrinking the model element 64 -> 36 bytes moved gpu by +0.8 %, i.e. nothing.
// So the fix is to shorten the dependency chain
//   gl_VertexID -> quadData[quadId] -> modelId -> modelData[]
// from two serialised dependent loads per vertex to one, for 3 of the quad's 4 corners.
//
// The element grows 8 -> 16 bytes (std430 gives an ivec3 array a stride of 16): 8 more bytes of
// streaming traffic per quad to delete a 64-byte dependent load, the right trade under a latency cost.
//
// THIS IS THE DEFINE THAT WINS. bindings.glsl guards its own `#define Quad` with `#ifndef Quad`, and
// this file is included first, so a change made only there is dead code -- which is exactly how the
// first version of this change shipped and failed to compile ('z' : vector swizzle selection out of
// range, because the shader still saw an ivec2).
#define Quad ivec3

//#define Eu32(data, amountBits, shift) (uint((data)>>(shift))&((1u<<(amountBits))-1))

// Takes ivec3 now, not ivec2: Quad gained a third component for the baked faceModelData. The body only
// ever reads .x and .y, which a 3-component vector still has, so widening the parameter is the whole
// change -- but it must be widened, because GLSL does not convert between vector sizes at a call.
uint Eu32v(ivec3 data, int amount, int shift) {
    if (shift > 31) {
        shift -= 32;
        return (uint(data.y)>>uint(shift))&((1u<<uint(amount))-1);
    } else {
        return (uint(data.x)>>uint(shift))&((1u<<uint(amount))-1);
    }
}

vec3 extractPos(ivec2 quad) {
    return vec3(Eu32v(quad, 5, 21), Eu32v(quad, 5, 16), Eu32v(quad, 5, 11));
}

ivec2 extractSize(ivec2 quad) {
    return ivec2(Eu32v(quad, 4, 3), Eu32v(quad, 4, 7)) + ivec2(1);//the + 1 is cause you cant actually have a 0 size quad
}

uint extractFace(ivec2 quad) {
    return Eu32v(quad, 3, 0);
}

uint extractStateId(ivec2 quad) {
    //Eu32(quad, 20, 26);
    return Eu32v(quad, 6, 26)|(Eu32v(quad, 14, 32)<<6);
}

uint extractBiomeId(ivec2 quad) {
    return Eu32v(quad, 9, 46);
}

uint extractLightId(ivec2 quad) {
    return Eu32v(quad, 8, 55);
}

bool isQuadEmpty(ivec2 quad) {
    return all(equal(quad, ivec2(0)));
}
#endif