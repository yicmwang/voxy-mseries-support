#version 460 core

// CORRECTION (2026-09-21, optimisation.MD 16). This file was written to separate the INTEGER TYPE from
// the FLAT QUALIFIER as the ICB blocker. **Both were wrong.** Case 90 (a flat varying with no texture
// behind it) PASSES, so `flat` is not the trigger; and the smooth-carrier replacement fails too, so
// the type is not either. The measured rule is: **a texture sample in the VERTEX whose result reaches
// a varying the FRAGMENT declares** -- here, `interData.y` carries `packVec4(getLighting(...))`, a
// vertex texture fetch. See optimisation.MD 16 for the case table.
//
// This case still FAILS, so it remains a usable negative control -- but it no longer discriminates
// what it was written to discriminate. The reasoning below is kept only as a record of a superseded
// inference; do not act on it.
//
// The FRAGMENT HALF OF A PROPOSED FIX, paired with a case that patches quads3.vert to match.
//
// Case 13 isolated the ICB blocker to `flat uvec4 interData` -- an unsigned-integer, flat-qualified
// varying. That leaves two candidate causes, and they imply very different amounts of work:
//
//   * the INTEGER TYPE  -> the fix is a bitcast: the vertex keeps `flat` but carries the bits in a
//                          vec4 of floats, and this fragment converts back with floatBitsToUint. The
//                          `flat` qualifier is preserved, so the per-face data is still NOT
//                          interpolated -- which matters, because interpolating these bits would
//                          corrupt them. A local change to two shaders, no architecture change.
//   * the FLAT QUALIFIER -> a float varying cannot simply drop `flat` (the bits would be interpolated
//                          and destroyed), so the fix would have to move interData out of the varying
//                          interface entirely -- e.g. a per-vertex SSBO read in the fragment stage.
//                          A much larger change.
//
// The paired case patches quads3.vert with exactly the two edits this file assumes:
//   `out flat uvec4 interData`  ->  `out flat vec4 interData`
//   `interData = quad.attributeData`  ->  `interData = uintBitsToFloat(quad.attributeData)`
// If that case PASSES, the cause is the integer type and the fix is the cheap one.
//
// Lives under assets/voxy/shaders/tools/ to mark it as not part of the production rendering pipeline.

layout(location = 0) in flat vec4 interData;

layout(location = 0) out vec4 outColour0;

void main() {
    // Convert back to uint, so this exercises the real round trip rather than just the interface.
    outColour0 = vec4(float(floatBitsToUint(interData.x) & 0xFFu) / 255.0, 0.5, 0.75, 1.0);
}
