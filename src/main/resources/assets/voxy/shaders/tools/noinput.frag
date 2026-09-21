#version 460 core

// Toy fragment that declares NO vertex inputs at all -- it just writes a constant.
//
// It exists to separate two things ICB probe case 8 and case 9 disagree about. Both pair the SAME
// real terrain vertex, yet case 8 (with quads.frag under VOXY_NO_ATLAS + VOXY_LOD_FLAT_FRAG) PASSES
// and case 9 (with tools/quads_iface.frag) FAILS on "Vertex shader cannot be used with indirect
// command buffers". The only difference is the fragment.
//
// The leading hypothesis this case tests: quads.frag's FLAT early-out leaves interData/uv unused, so
// SPIRV-Cross drops them from the fragment's MSL inputs entirely -- whereas quads_iface.frag declares
// `flat uvec4 interData` and `vec2 uv`. If a fragment that declares NO inputs passes where one that
// declares matching inputs fails, then it is the fragment's INPUT DECLARATION that trips the
// validator, and Metal's "Vertex shader" attribution is a red herring -- which is exactly what would
// make case 8 and case 9 consistent.
//
//   case 11 PASS -> the trigger is the fragment's input interface, not the vertex. The vertex is fine.
//   case 11 FAIL -> the vertex genuinely is implicated, and case 8's PASS needs another explanation.
//
// Lives under assets/voxy/shaders/tools/ to mark it as not part of the production rendering pipeline.

layout(location = 0) out vec4 outColour0;

void main() {
    outColour0 = vec4(0.25, 0.5, 0.75, 1.0);
}
