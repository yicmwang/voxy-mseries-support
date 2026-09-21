#version 460 core

// MRT smoke-test fragment shader: TWO colour outputs, so a pixel readback can say which of them the
// pipeline actually delivered.
//
// The second output is the same shape as the one that never lands in the game -- quads.frag's
// `VOXY_LOD_DEPTH_COLOUR` output, which is the Hi-Z pyramid's source. Everything measurable around it
// is now verified correct (pass descriptor: right texture, RGBA8Unorm, Store; pipeline descriptor:
// two RGBA8Unorm attachments, write mask All; MSL: declares and assigns [[color(1)]]; backend: proven
// standalone), and it still reads empty in the game.
//
// So this shader carries the one input never varied: the SHADER'S OWN SHAPE. The defines below are the
// features the game's quads.frag has and this harness did not, and they exist so the difference can be
// bisected in seconds rather than in six-minute runs.
//
// Lives under assets/voxy/shaders/tools/ to mark it as not part of the production rendering pipeline.

layout(location = 0) in vec3 vColor;

layout(location = 0) out vec4 outColour0;
layout(location = 1) out vec4 outColour1;

void main() {
    outColour0 = vec4(vColor, 1.0);

#if defined(MRT_DISCARD)
    // A discard in the shader, mirroring quads.frag's alpha cutout. NEVER TAKEN -- the point is to have
    // the discard path present in the compiled code, not to exercise it -- because if a discard is what
    // kills the second output, its mere presence is enough and a taken branch would confound the test
    // by removing the fragment entirely.
    if (vColor.r < 0.0) {
        discard;
    }
#endif

#ifdef MRT_FRAGCOORD
    // quads.frag writes gl_FragCoord.z here, and declares [[position]] in the process. A constant was
    // tried in the game (occ4) and was also dropped, so the value is not the fault -- but gl_FragCoord
    // brings a [[position]] input into the fragment function, which this harness never had.
    outColour1 = vec4(gl_FragCoord.z, 0.0, 0.0, 1.0);
#else
    // 0.25 / 0.5 / 0.75 / 1.0 quantise to exactly 64 / 128 / 191 / 255 in RGBA8.
    outColour1 = vec4(0.25, 0.5, 0.75, 1.0);
#endif
}
