//The hierarchical-Z occlusion test, in ONE place.
//
//Extracted from lod/hierarchical/screenspace.glsl so that the traversal and the
//per-section cull pass (lod/gl46/section_cull.comp) ask the *same* predicate
//about the *same* depth convention. Two copies of this test is exactly how this
//project got a cull whose box frame disagreed with the geometry it was culling
//(cull.MD 8.8), so the definition is shared rather than duplicated.

layout(binding = HIZ_BINDING) uniform sampler2D hizDepthSampler;

// Perspective divide, then the window transform for XY ONLY. The z is returned untouched because it is
// ALREADY depth on this frame: the traversal's VP is vanilla's reverse-Z projection, whose NDC z is in
// [0,1]. Applying the `* 0.5 + 0.5` window transform to z as well -- as the other convention needs --
// squashes box depth into [0.5,1] against a pyramid that holds [0,1], and the test then reads the wrong
// half of the range. One place, so the caller's two corners cannot drift apart.
vec3 toScreenspace(vec4 p) {
    vec3 n = p.xyz / p.w;
    return vec3(n.xy * 0.5f + 0.5f, n.z);
}

//True when the screenspace box [minBB,maxBB] (xy = window coords in [0,1], z = reverse-Z depth) lies
//entirely BEHIND the pyramid's farthest occluder over the box's footprint.
//
//The pyramid's dimensions come from the texture itself rather than from a packed size uniform: the
//caller may not have one (the per-section cull does not), and `textureSize(sampler, ml)` is by
//construction the equivalent of HiZBuffer's mip rule, `max(w>>ml, 1)`, for every level.
bool hizOccluded(vec3 minBB, vec3 maxBB) {
    //Things start breaking down if the area is the entire scree, no idea why, just abort if we hit this case
    if ((maxBB.xy-minBB.xy)==vec2(1.0f)) return false;

    ivec2 ssize = textureSize(hizDepthSampler, 0);
    vec2 size = (maxBB.xy-minBB.xy)*ssize;
    float miplevel = log2(max(max(size.x, size.y),1));

    miplevel = floor(miplevel)-1;
    //miplevel = clamp(miplevel, 0, 6);
    miplevel = clamp(miplevel, 0, textureQueryLevels(hizDepthSampler)-1);

    int ml = int(miplevel);
    ssize = max(ivec2(1), textureSize(hizDepthSampler, ml));
    ivec2 mxbb = min(ivec2(maxBB.xy*ssize),ssize-1);
    ivec2 mnbb = ivec2(minBB.xy*ssize);

    // Reverse-Z frame: near is 1, far is 0, and the pyramid holds the tile's FARTHEST occluder --
    // the smallest value -- so this is a min-reduce. It starts at 1.0 (the near plane) because a min
    // over non-negative depths would otherwise just return the initial value.
    float pointSample = 1.0f;
    bool sampled = false;
    for (int x = mnbb.x; x<=mxbb.x; x++) {
        for (int y = mnbb.y; y<=mxbb.y; y++) {
            float sp = texelFetch(hizDepthSampler, ivec2(x, y), ml).r;
            pointSample = min(sp, pointSample);
            sampled = true;
        }
    }
    // AN EMPTY RANGE MUST NOT READ AS OCCLUDED.
    //
    // 1.0f is the correct IDENTITY for a min-reduce here, but it is also the NEAR PLANE, so if the loop
    // body never runs the seed survives and `1.0 > maxBB.z` is true for every box that is not itself at
    // the near plane. "I found no occluder" would then mean "fully occluded" -- the dangerous direction
    // for a cull, and it silently removes geometry rather than adding it.
    //
    // The range is empty when the clamped box degenerates: mnbb = ivec2(minBB.xy*ssize) but
    // mxbb = min(ivec2(maxBB.xy*ssize), ssize-1), so a box clamped to the far edge (minBB.x == 1.0,
    // i.e. entirely off the right or bottom) gives mnbb == ssize while mxbb == ssize-1. The two clamps
    // disagree, and nothing checked.
    //
    // Returning false is the conservative answer and the one this test's own reasoning implies: an
    // unbuilt pyramid and a sky tile both read as "not occluded" (see the note below), so "no samples"
    // must too, for the same reason -- being wrong in the occluding direction is what loses terrain.
    if (!sampled) {
        return false;
    }
    // A box is occluded when it lies entirely BEHIND the tile's farthest occluder. On reverse-Z
    // "behind" means a SMALLER depth, and the box's nearest point is maxBB.z (its largest corner, since
    // larger is nearer), so the test is `tileFarthest > boxNearest`.
    //
    // No unbuilt-pyramid guard, deliberately. Far is 0.0 on this convention, which makes 0.0 BOTH the
    // unbuilt sentinel AND the commonest legitimate value in the scene -- sky. A guard such as
    // `pointSample <= 0.0 -> not occluded` would therefore reject every box whose footprint touches
    // sky, and in an outdoor scene that is nearly all of them; that guard was in fact present once and
    // is why the cull never fired. The arithmetic already gives the right answer for both cases:
    // `0.0 > maxBB.z` is false for every box, so an unbuilt pyramid and a sky tile both read as
    // "not occluded" without a special case.
    return pointSample > maxBB.z;
}
