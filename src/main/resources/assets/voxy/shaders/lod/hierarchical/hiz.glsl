//The hierarchical-Z occlusion test, in ONE place.
//
//Extracted from lod/hierarchical/screenspace.glsl when a second caller existed: a per-section cull
//pass (lod/gl46/section_cull.comp) that re-asked this same question about every section the traversal
//had already kept. That pass was removed 2026-09-22 — it measured at ~0-6 %, and at exactly 0 for its
//default whole-cell form, which re-asked the traversal's own question about the traversal's own output
//(lod-bugs.MD 2.1). The traversal is the only caller again.
//
//The single definition stays anyway, and the reason is not tidiness. Two copies of this test is how
//this project got a cull whose box frame disagreed with the geometry it was culling (cull.MD 8.8); the
//next caller gets the shared one or the bug comes back.

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
// The pyramid's answer for the last box tested, published for the traversal's diagnostics. See
// traversal_dev.comp's hizSampleHist for what it is for: the cull's decisions have never been observed
// from the inside, and every remaining hypothesis about the over-culling is a claim about this number.
float hizLastSample = 1.0f;

bool hizOccluded(vec3 minBB, vec3 maxBB) {
    // ---------------------------------------------------------------------------------------------
    // UPSTREAM'S GUARD, RESTORED — FOR FIDELITY, NOT BECAUSE IT FIXED ANYTHING.
    //
    // Upstream (`voxy/lod/hierarchical/screenspace.glsl:157`):
    //     if (any(lessThan(abs(_maxBB.xy-_minBB.xy-vec2(1.0f)), vec2(0.000001f)))) return false;
    //
    // The port had `if ((maxBB.xy-minBB.xy)==vec2(1.0f)) return false;`. `==` on a vec2 is an
    // ALL-components comparison, so the port bailed out only when BOTH axes spanned the screen where
    // upstream bails when EITHER does. The port also kept upstream's own warning on the line above --
    // "Things start breaking down if the area is the entire screen, just abort if we hit this case" --
    // so it had weakened the guard its own comment describes.
    //
    // Measured: restoring it changed the artefact's measure by nothing (384 388 sky px against 386 497,
    // where the corrected band is 235 000-253 000). It is kept because it is upstream's form and the
    // port's was demonstrably weaker, NOT because it is a fix, and this note exists so nobody later
    // reads it as one.
    //
    // An earlier version of this comment claimed the guard caught "a distant coarse node at the
    // horizon". That is wrong and worth recording: a distant node projects to a SMALL footprint and
    // cannot span the screen. What this guard catches is a box the camera is inside or straddling,
    // where the divide by a near-zero or negative w flips corners and the clamped box blows up. Those
    // are near nodes, which is not the reported symptom.
    //
    // The clamps below are upstream's too, same class: `ceil` on the far edge and `floor` on the near
    // edge make the sampled range a SUPERSET of the box. The port truncated the far edge with
    // `ivec2(maxBB.xy*ssize)` (`int()` truncates toward zero), which can shrink the range by a texel
    // and bias the min-reduce toward nearer occluders. Also measured neutral, also kept for fidelity.
    // ---------------------------------------------------------------------------------------------
    if (any(lessThan(abs((maxBB.xy - minBB.xy) - vec2(1.0f)), vec2(0.000001f)))) { hizLastSample = 1.0f; return false; }

    ivec2 ssize = textureSize(hizDepthSampler, 0);
    vec2 size = (maxBB.xy-minBB.xy)*ssize;
    float miplevel = log2(max(max(size.x, size.y),1));

    miplevel = floor(miplevel)-1;
    miplevel = clamp(miplevel, 0, textureQueryLevels(hizDepthSampler)-1);

    int ml = int(miplevel);
    ssize = max(ivec2(1), textureSize(hizDepthSampler, ml));
    ivec2 mxbb = min(ivec2(ceil(maxBB.xy*ssize)), ssize-1);
    ivec2 mnbb = ivec2(floor(minBB.xy*ssize));

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
        hizLastSample = 1.0f;
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
    //
    // A CONSERVATIVE MARGIN ON THIS COMPARISON WAS TRIED AND MEASURED NEUTRAL, and is not here.
    // Upstream's rasterized cull brings the box closer to the camera before its depth test
    // (`gl_Position.z += CLOSER_SIGN*0.000001f * gl_Position.w`), and the analytic test inherited no
    // equivalent, which is a real difference worth knowing about. It is not this bug: a relative
    // margin of 1% and of 5% moved the artefact's own measure not at all (386 497 and 393 927 sky
    // pixels against 386 497 for no margin, where the corrected band is 235 000-253 000). Whatever is
    // wrong here is not a marginal comparison. See lod-bugs.MD 11.
    hizLastSample = pointSample;
    return pointSample > maxBB.z;
}
