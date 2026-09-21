#version 460 core

// Toy fragment that matches quads3.vert's OUTPUT INTERFACE exactly, so a pipeline pairing the real
// terrain vertex with a trivial fragment actually LINKS.
//
// Why this file exists: ICB probe case 9 pairs the real terrain vertex with a toy fragment to isolate
// the vertex from quads.frag. The first attempt used tools/mrt.frag, which declares `vec3 vColor` at
// location 0 while the terrain vertex writes `flat uvec4 interData` there. Metal rejected the pipeline
// with a LINK error -- "Fragment input(s) `user(locn0)` mismatching vertex shader output type(s) or not
// written by vertex shader" -- which says nothing about indirect command buffers. A case that fails to
// link cannot answer an ICB question, so its FAIL would have been read as "the vertex is
// ICB-incompatible" when it proved nothing of the kind.
//
// The interface below is quads3.vert's, for the define set icbCase compiles it with
// (NO_SHADE_FACE_TINT / UP_FACE_TINT / DOWN_FACE_TINT / Z_AXIS_FACE_TINT / X_AXIS_FACE_TINT +
// VOXY_METAL_BI_FIX):
//
//   location 0   flat uvec4 interData   quads3.vert:24  (unconditional)
//   location 1   vec2 uv                quads3.vert:26  (present because USE_NV_BARRY is undefined)
//
// Locations 2, 4, 5 and 7 are CONDITIONAL and none of their defines is set, so the vertex writes
// nothing there: voxyFogDist needs VOXY_NEEDS_FOG_DIST (quads3.vert:66-70), voxyFramePos needs
// VOXY_LOD_CHUNK_CULL, voxyDrawIdOut needs VOXY_LOD_SHOW_DRAWID, quadDebug needs DEBUG_RENDER.
// If icbCase's vertex define map ever gains one of those, this interface must gain the matching input.
//
// Lives under assets/voxy/shaders/tools/ to mark it as not part of the production rendering pipeline.

layout(location = 0) in flat uvec4 interData;
layout(location = 1) in vec2 uv;

layout(location = 0) out vec4 outColour0;

void main() {
    // Read both inputs so the compiler cannot strip either one out of the vertex interface.
    // vec3 + float = 4 components, which is what vec4 takes; appending a third component is 5 and
    // shaderc rejects it ("'constructor' : too many arguments").
    outColour0 = vec4(vec3(float(interData.x & 0xFFu) / 255.0), uv.x);
}
