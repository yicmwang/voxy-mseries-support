#version 460 core

// The FRAGMENT HALF OF THE PROPOSED FIX, paired with a case that patches quads3.vert to match.
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
