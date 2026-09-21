#version 460 core

// MRT smoke-test fragment shader: TWO colour outputs carrying known constants, so a pixel readback
// can say which of them the pipeline actually delivered.
//
// The second output is deliberately the same shape as the one that never lands in the game --
// quads.frag's `VOXY_LOD_DEPTH_COLOUR` output, which is the Hi-Z pyramid's source. Here the value is a
// constant rather than gl_FragCoord.z so the readback is unambiguous: a wrong value means the value is
// wrong, a missing value means the attachment never received the write. Those are different faults and
// the twelve measurements in optimisation.MD 9.6 could not separate them.
//
// Lives under assets/voxy/shaders/tools/ to mark it as not part of the production rendering pipeline.

layout(location = 0) in vec3 vColor;

layout(location = 0) out vec4 outColour0;
layout(location = 1) out vec4 outColour1;

void main() {
    outColour0 = vec4(vColor, 1.0);
    // 0.25 / 0.5 / 0.75 / 1.0 quantise to exactly 64 / 128 / 191 / 255 in RGBA8.
    outColour1 = vec4(0.25, 0.5, 0.75, 1.0);
}
