#version 460 core

// Declares ONLY quads3.vert's location-0 output (`flat uvec4 interData`), and none of location 1.
//
// The partner to tools/uv_only.frag -- see that file for the full reasoning. Between them they split
// the terrain vertex's two-varying interface, which is the last thing standing between the ICB probe
// and an answer.
//
// `flat uvec4` is the more suspicious half: it is an UNSIGNED INTEGER varying with flat interpolation,
// and nothing else in the probe carries an integer interpolant. The toy pipeline's `in vec3 vColor` is
// float and is accepted, so if this is the trigger the difference is the integer/flat qualifier rather
// than the existence of an interface.
//
// Lives under assets/voxy/shaders/tools/ to mark it as not part of the production rendering pipeline.

layout(location = 0) in flat uvec4 interData;

layout(location = 0) out vec4 outColour0;

void main() {
    // Read interData so the input cannot be optimised out of the fragment.
    outColour0 = vec4(float(interData.x & 0xFFu) / 255.0, 0.5, 0.75, 1.0);
}
