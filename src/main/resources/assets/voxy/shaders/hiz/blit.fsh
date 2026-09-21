#version 430


layout(location = 0) in vec2 uv;
layout(binding = 0) uniform sampler2D depthTex;
layout(location=0) out vec4 colour;

void main() {
    vec4 depths = textureGather(depthTex, uv, 0); // Get depth values from all surrounding texels.

    // Reverse-Z frame: near is 1, far is 0. The pyramid keeps the tile's FARTHEST occluder, which on
    // this convention is the SMALLEST value, so the reduction is a min rather than a max.
    //
    // There is deliberately no sky-hole patch here. The other convention's version rewrote a
    // sky-containing tile so a max landed on the sky value; on reverse-Z sky is 0.0, so a min-reduce
    // lands on it for free -- exactly the conservative answer that patch produced by hand.
    float res = min(min(depths.x, depths.y), min(depths.z, depths.w));

    // Written as COLOUR, never gl_FragDepth: the pyramid is an R32F colour target (see HiZBuffer's
    // class doc for why a depth-format texture cannot be read back through a `sampler2D` on Metal).
    colour = vec4(res);
}
