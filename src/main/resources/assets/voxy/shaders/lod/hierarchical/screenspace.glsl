
//All the screenspace computuation code, hiz culling + size/screenspace AABB size computation
// to determin whether child node should be visited
// it controls the actions of the traversal logic
//NOTEEE!!! SO can do a few things, technically since atm its split not useing persistent threads
// can use mesh shaders to do rasterized occlution directly with a meshdrawindirect, one per layer
//Persistent threads might still be viable/usable since the inital lods supplied to the culler are mixed level
// (basiclly the minimum guarenteed value, like dont supply a top level lod right in front of the camera, since that is guarenteed not to, never be that level)
// do this based on camera distance computation

//changing the base level/root of the graph for some nodes can be really tricky and incorrect so might not be worth it but it should help
// substantually for performance (for both persistent threads and incremental)


layout(binding = HIZ_BINDING) uniform sampler2D hizDepthSampler;

//TODO: maybe do spher bounds aswell? cause they have different accuracies but are both over estimates (liberals (non conservative xD))
// so can do &&

bool within(vec2 a, vec2 b, vec2 c) {
    return all(lessThan(a,b)) && all(lessThan(b, c));
}

bool within(vec3 a, vec3 b, vec3 c) {
    return all(lessThan(a,b)) && all(lessThan(b, c));
}

bool within(float a, float b, float c) {
    return a<b && b<c;
}

float crossMag(vec2 a, vec2 b) {
    return abs(a.x*b.y-b.x*a.y);
}

bool checkPointInView(vec4 point) {
    return within(vec3(-point.w,-point.w,0.0f), point.xyz, vec3(point.w));
}

vec3 minBB = vec3(0.0f);
vec3 maxBB = vec3(0.0f);
bool frustumCulled = false;

float screenSize = 0.0f;

UnpackedNode node22;
//Sets up screenspace with the given node id, returns true on success false on failure/should not continue
//Accesses data that is setup in the main traversal and is just shared to here
void setupScreenspace(in UnpackedNode node) {
    //TODO: Need to do aabb size for the nodes, it must be an overesimate of all the children

    node22 = node;
    /*
    Transform transform = transforms[getTransformIndex(node)];

    vec3 point = VP*(((transform.transform*vec4((node.pos<<node.lodLevel) - transform.originPos.xyz, 1))
                    + (transform.worldPos.xyz-camChunkPos))-camSubChunk);
                    */


    // Metal fix (2026-05-26): the original `((node.pos<<lodLevel)-camSecPos)<<5`
    // left-shifts a SIGNED value that is NEGATIVE for nodes left/behind/below the
    // camera section. Left-shift of a negative int is UNDEFINED — GL drivers
    // treat it as arithmetic (×2^n) and work, but Metal/MSL (fast-math) optimises
    // it differently → garbage basePos for negative-relative nodes → those nodes
    // are frustum-culled wrongly. That is the terrain disappearing at the screen
    // edges (direction-dependent: which nodes are negative depends on facing),
    // while the centre (positive-relative) stays solid. Use multiplication, which
    // is well-defined for negative values: `x<<lod` -> `x*(1<<lod)`, `y<<5` -> `y*32`.
    vec3 basePos = vec3(((node.pos*(1<<node.lodLevel))-camSecPos)*32)-camSubSecPos;

    frustumCulled = outsideFrustum(frustum, basePos, float(32<<node.lodLevel));

    //Fast exit
    if (frustumCulled) {
        return;
    }

    vec4 P000 = VP * vec4(basePos, 1);
    mat3x4 Axis = mat3x4(VP) * float(32<<node.lodLevel);

    vec4 P100 = Axis[0] + P000;
    vec4 P001 = Axis[2] + P000;
    vec4 P101 = Axis[2] + P100;
    vec4 P010 = Axis[1] + P000;
    vec4 P110 = Axis[1] + P100;
    vec4 P011 = Axis[1] + P001;
    vec4 P111 = Axis[1] + P101;


    //Perspective divide + convert to screenspace (i.e. range 0->1 if within viewport)
    vec3 p000 = (P000.xyz/P000.w) * 0.5f + 0.5f;
    vec3 p100 = (P100.xyz/P100.w) * 0.5f + 0.5f;
    vec3 p001 = (P001.xyz/P001.w) * 0.5f + 0.5f;
    vec3 p101 = (P101.xyz/P101.w) * 0.5f + 0.5f;
    vec3 p010 = (P010.xyz/P010.w) * 0.5f + 0.5f;
    vec3 p110 = (P110.xyz/P110.w) * 0.5f + 0.5f;
    vec3 p011 = (P011.xyz/P011.w) * 0.5f + 0.5f;
    vec3 p111 = (P111.xyz/P111.w) * 0.5f + 0.5f;


    {//Compute exact screenspace size
        float ssize = 0;
        {//Faces from 0,0,0

            vec2 A = p100.xy-p000.xy;
            vec2 B = p010.xy-p000.xy;
            vec2 C = p001.xy-p000.xy;
            ssize += crossMag(A,B);
            ssize += crossMag(A,C);
            ssize += crossMag(C,B);
        }
        {//Faces from 1,1,1
            vec2 A = p011.xy-p111.xy;
            vec2 B = p101.xy-p111.xy;
            vec2 C = p110.xy-p111.xy;
            ssize += crossMag(A,B);
            ssize += crossMag(A,C);
            ssize += crossMag(C,B);
        }
        ssize *= 0.5f;//Half the size since we did both back and front area
        screenSize = ssize;
    }

    minBB = min(min(min(p000, p100), min(p001, p101)), min(min(p010, p110), min(p011, p111)));
    maxBB = max(max(max(p000, p100), max(p001, p101)), max(max(p010, p110), max(p011, p111)));

    minBB = clamp(minBB, vec3(0), vec3(1));
    maxBB = clamp(maxBB, vec3(0), vec3(1));
}

//Checks if the node is implicitly culled (outside frustum)
bool outsideFrustum() {
    return frustumCulled;// maxW < 16 is a trick where 16 is the near plane

    //|| any(lessThanEqual(minBB, vec3(0.0f, 0.0f, 0.0f))) || any(lessThanEqual(vec3(1.0f, 1.0f, 1.0f), maxBB));
}

bool isCulledByHiz() {
#ifdef VOXY_NO_HIZ
    // The pyramid is never built on this backend (see HierarchicalOcclusionTraverser.traversalDefines),
    // so sampling it tests noise. The guard below assumes an unbuilt pyramid reads zero, which holds
    // for a zeroed buffer and not for raw allocated texture memory.
    return false;
#endif
    //Things start breaking down if the area is the entire scree, no idea why, just abort if we hit this case
    if ((maxBB.xy-minBB.xy)==vec2(1.0f)) return false;

    ivec2 ssize = ivec2(packedHizSize>>16,packedHizSize&0xFFFF);
    vec2 size = (maxBB.xy-minBB.xy)*ssize;
    float miplevel = log2(max(max(size.x, size.y),1));

    miplevel = floor(miplevel)-1;
    //miplevel = clamp(miplevel, 0, 6);
    miplevel = clamp(miplevel, 0, textureQueryLevels(hizDepthSampler)-1);

    int ml = int(miplevel);
    ssize = max(ivec2(1), ssize>>ml);
    ivec2 mxbb = min(ivec2(maxBB.xy*ssize),ssize-1);
    ivec2 mnbb = ivec2(minBB.xy*ssize);

    float pointSample = -1.0f;
    //float pointSample2 = 0.0f;
    for (int x = mnbb.x; x<=mxbb.x; x++) {
        for (int y = mnbb.y; y<=mxbb.y; y++) {
            float sp = texelFetch(hizDepthSampler, ivec2(x, y), ml).r;
            //pointSample2 = max(sp, pointSample2);
            //sp = mix(sp, pointSample, 0.9999999f<=sp);
            pointSample = max(sp, pointSample);
        }
    }
    //pointSample = mix(pointSample, pointSample2, pointSample<=0.000001f);

    // M13 2026-05-15: on backends that haven't built the HiZ pyramid yet
    // (Metal still uses ensureAllocated only — the zero-init pyramid is
    // a parked stub until cross-context MC-depth import lands), every
    // texelFetch returns 0.0. The original `pointSample <= minBB.z`
    // returns TRUE for any box with minBB.z >= 0 — i.e. every visible
    // box — which culls all top-level LOD nodes and leaves renderList
    // empty. Skip the occlusion test when the HiZ is uninitialised
    // (pointSample == 0) so the stub becomes a true "always pass"
    // instead of an "always reject". Real GL-path HiZ writes 1.0 in
    // sky regions so pointSample is positive everywhere and this guard
    // doesn't change its behaviour.
    if (pointSample <= 0.0) return false;
    return pointSample<=minBB.z;
}



//Returns if we should decend into its children or not
bool shouldDecend() {
    return screenSize > minSSS;
}