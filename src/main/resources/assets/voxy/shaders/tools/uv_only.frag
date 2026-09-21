#version 460 core

// Declares ONLY quads3.vert's location-1 output (`vec2 uv`), and none of location 0.
//
// The ICB probe has narrowed to this: the real terrain vertex PASSES with a fragment that declares no
// inputs (case 11) and FAILS with one that declares `flat uvec4 interData` + `vec2 uv` (cases 9, 10).
// So the trigger is somewhere in the vertex->fragment interpolant interface -- but the toy pipeline
// carries `in vec3 vColor` and passes (cases 1, 2, 4, 5, 6), so a plain float interface is fine.
//
// The two candidates are therefore the INTEGER/FLAT varying and the interface's SIZE. This file takes
// the `vec2 uv` half; tools/idata_only.frag takes the `flat uvec4` half.
//
//   case 12 (this file) PASS, case 13 (idata_only) FAIL -> the integer/flat varying is the trigger
//   case 12 FAIL,          case 13 PASS               -> unexpected; re-derive
//   both PASS                                        -> it is the interface as a whole, not one member
//
// Lives under assets/voxy/shaders/tools/ to mark it as not part of the production rendering pipeline.

layout(location = 1) in vec2 uv;

layout(location = 0) out vec4 outColour0;

void main() {
    // Read uv so the input cannot be optimised out of the fragment.
    outColour0 = vec4(0.25, uv.x, 0.75, 1.0);
}
