#version 460 core

// Declares ONLY quads3.vert's location-0 output (`flat uvec4 interData`), and none of location 1.
//
// The partner to tools/uv_only.frag -- see that file for the full reasoning. Between them they split
// the terrain vertex's two-varying interface, which is the last thing standing between the ICB probe
// and an answer.
//
// This file used to argue that "`flat uvec4` is the more suspicious half ... because nothing else in
// the probe carries an integer interpolant". **That was the wrong inference** (optimisation.MD 16).
// NEITHER the integer type NOR `flat` is the trigger: a flat varying with no texture behind it is
// ACCEPTED (case 90), and this case FAILS for a different reason. What it actually isolates is the
// terrain vertex's LIGHTMAP SAMPLE reaching a varying this fragment declares -- `interData.y` carries
// `packVec4(getLighting(...))`, a vertex texture fetch, and that is the rule cases 95-99 measure.
//
// Lives under assets/voxy/shaders/tools/ to mark it as not part of the production rendering pipeline.

layout(location = 0) in flat uvec4 interData;

layout(location = 0) out vec4 outColour0;

void main() {
    // Read interData so the input cannot be optimised out of the fragment.
    outColour0 = vec4(float(interData.x & 0xFFu) / 255.0, 0.5, 0.75, 1.0);
}
