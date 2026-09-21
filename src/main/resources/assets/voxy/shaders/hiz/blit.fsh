#version 430


layout(location = 0) in vec2 uv;
layout(binding = 0) uniform sampler2D depthTex;
#ifdef OUTPUT_COLOUR
layout(location=0) out vec4 colour;
#endif
void main() {
    vec4 depths = textureGather(depthTex, uv, 0); // Get depth values from all surrounding texels.

#ifdef VOXY_HIZ_REVERSE_Z
    // Reverse-Z frame: near is 1, far is 0. The pyramid keeps the tile's FARTHEST occluder, which on
    // this convention is the SMALLEST value, so the reduction is a min rather than a max.
    //
    // The sky patch below is deliberately absent, and that is not an omission: sky is 0.0 here, so a
    // tile containing any sky min-reduces to 0.0 on its own -- exactly the conservative answer the GL
    // patch goes out of its way to produce (it rewrites a sky-containing tile so the max lands on the
    // sky value). The min-reduce gets there for free.
    float res = min(min(depths.x, depths.y), min(depths.z, depths.w));
#else
    bvec4 cv = lessThanEqual(vec4(0.999999999f), depths);
    if (any(cv)) {//Patch holes (its very dodgy but should work :tm:, should clamp it to the first 3 levels)
        depths = mix(vec4(0.0f), depths, cv);
    }
    float res = max(max(depths.x, depths.y), max(depths.z, depths.w));
#endif

    #ifdef OUTPUT_COLOUR
    colour = vec4(res);
    #else
    gl_FragDepth = res; // Write conservative depth.
    #endif
}
