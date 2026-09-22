#version 430
#extension GL_ARB_shader_draw_parameters : require

layout(binding = 0, std140) uniform SceneUniform {
    mat4 VP;
    ivec3 camSecPos;
    uint screenW;
    vec3 camSubSecPos;
    uint screenH;
};

#define NODE_DATA_INDEX 1
#import <voxy:lod/hierarchical/node.glsl>

layout(binding = 2, std430) restrict buffer NodeList {
    uint count;
    uint nodeQueue[];
};

layout(location = 1) out flat vec4 colour;

void main() {
    UnpackedNode node;
    unpackNode(node, nodeQueue[gl_InstanceID]);

    // CONVERTED, and this was the last unconverted instance in the corpus. It carried the exact form
    // screenspace.glsl's 2026-05-26 fix removed: `node.pos<<lodLevel` then `<<5`, both left-shifts of a
    // value that is NEGATIVE for nodes left/behind/below the camera section. That is UB, and this
    // backend's SPIRV-Cross/MSL path miscompiles it while GL's drivers happened to do the arithmetic
    // multiply -- so the outline drew in the wrong place for precisely the off-axis nodes it exists to
    // diagnose.
    //
    // It is a debug visualiser and cannot be the cause of any rendering symptom, so it was left while
    // the live cull path was audited first. Fixed anyway, because it is the tool someone reaches for to
    // investigate "nodes missing near the screen edge", and a diagnostic that lies about precisely that
    // case is worse than no diagnostic.
    vec4 base = VP*vec4(vec3(((node.pos*(1<<node.lodLevel))-camSecPos)*32)-camSubSecPos, 1);

    // The `<<(5+node.lodLevel)` is safe as written -- the ivec3 components are gl_VertexID bits, so they
    // are 0 or 1 and never negative -- but it is converted for consistency with the line above, since
    // the distinction is invisible at a glance and that is how this family survives.
    vec4 pos = base + (VP*vec4(ivec3(gl_VertexID&1, (gl_VertexID>>2)&1, (gl_VertexID>>1)&1)*(1<<(5+node.lodLevel)), 1));

    gl_Position = pos;

    //node.nodeId
    uint hash = node.nodeId*1231421+123141;
    hash ^= hash>>16;
    hash = hash*1231421+123141;
    hash ^= hash>>16;
    hash = hash * 1827364925 + 123325621;
    //colour = vec4(vec3(float(hash&15u)/15, float((hash>>4)&15u)/15, float((hash>>8)&15u)/15), 1);
    colour = vec4(vec3(float(hash&31u)/31, float(node.lodLevel)/4, float(node.lodLevel)/4), 1);
}