
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


//The HiZ sampler, toScreenspace() and the pyramid test itself live in hiz.glsl. They were extracted
//when a per-section cull pass shared this predicate; that pass is gone (2026-09-22, lod-bugs.MD 2.1)
//and the traversal is the only caller again, but the single definition stays — the depth convention is
//what drifts when this test is written down twice.
#import <voxy:lod/hierarchical/hiz.glsl>


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

    // -------------------------------------------------------------------------------------------
    // WHICH FRAME THE BOX IS PROJECTED IN — the whole of Bug A's fix.
    //
    // The Hi-Z pyramid the traversal consults is built from frame N-1's depth, and it CANNOT be made
    // current: the geometry that fills it is drawn by the LOD pass, which runs after this traversal.
    // That is the frame's shape, not a defect. The defect was projecting the box in frame N and testing
    // it against frame N-1's map: with the camera moving, the box's footprint is not where it was when
    // those texels were written, so the test can find the near ground under a distant node and cull it
    // for one frame. Confined to the finest levels, because that is where per-frame screen displacement
    // is largest (lod-bugs.MD 17.7-17.10).
    //
    // So project the box in frame N-1 as well, using prevVP and a base position re-expressed against
    // the previous frame's camera origin (camDelta), and test THAT. Box and map then describe the same
    // instant, and a consistent Hi-Z test cannot contradict itself: the node's own surface is in the
    // map it is tested against, so `pointSample > maxBB.z` is false for anything actually visible.
    //
    // It costs NOTHING in culling -- unlike widening the sample band, measured to buy the blip back
    // with the cull's entire purpose (17.11) -- and needs no MSL pass (the format-view route Metal
    // refused outright, 17.12). The cull simply lags a frame, which is the frame it already lagged.
    //
    // `screenSize` and the box below both come from this matrix, deliberately: the refine decision
    // should describe the same frame as the cull that feeds it, and a node's screen size does not
    // meaningfully change in one frame. The FRUSTUM test above is left on the world-space box, so it is
    // unaffected either way.
    // -------------------------------------------------------------------------------------------
    mat4 boxVP = VP;
    vec3 boxBasePos = basePos;
    if (usePrevView == 1u) {
        boxVP = prevVP;
        boxBasePos = basePos + camDelta;
    }

    vec4 P000 = boxVP * vec4(boxBasePos, 1);
    mat3x4 Axis = mat3x4(boxVP) * float(32<<node.lodLevel);

    vec4 P100 = Axis[0] + P000;
    vec4 P001 = Axis[2] + P000;
    vec4 P101 = Axis[2] + P100;
    vec4 P010 = Axis[1] + P000;
    vec4 P110 = Axis[1] + P100;
    vec4 P011 = Axis[1] + P001;
    vec4 P111 = Axis[1] + P101;


    // Perspective divide + screenspace transform.
    //
    // xy is the same on both backends: NDC xy spans [-1,1], so *0.5+0.5 is the window transform.
    //
    // z is NOT, and getting it wrong is silent. GL's NDC z spans [-1,1], so *0.5+0.5 gives the window
    // depth the pyramid stores. On the whole-frame Metal path the traversal's VP is vanilla's projection
    // unchanged (VoxyRenderSystem.computeProjectionMat returns `base` for non-GL), which is reverse-Z
    // with an infinite far plane -- NDC z is ALREADY the depth, in [0,1] -- and applying the window
    // transform a second time squashes it into [0.5,1] against a pyramid holding [0,1]. A box at the far
    // plane would then be presented to the occlusion test as if it sat at depth ~0.5.
    //
    // The corner CHOICE further down was already correct for reverse-Z (larger z is nearer, so max()
    // picks the box's nearest point). It was the value that was wrong.
    vec3 p000 = toScreenspace(P000);
    vec3 p100 = toScreenspace(P100);
    vec3 p001 = toScreenspace(P001);
    vec3 p101 = toScreenspace(P101);
    vec3 p010 = toScreenspace(P010);
    vec3 p110 = toScreenspace(P110);
    vec3 p011 = toScreenspace(P011);
    vec3 p111 = toScreenspace(P111);


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

//The test itself (with its reasoning about the reverse-Z convention, the min-reduce and the
//deliberate absence of an unbuilt-pyramid guard) moved verbatim to hiz.glsl, where the per-section
//cull pass shares it. This stays a wrapper so the traversal's call site is unchanged.
bool isCulledByHiz() {
    return hizOccluded(minBB, maxBB);
}



//Returns if we should decend into its children or not
bool shouldDecend() {
    return screenSize > minSSS;
}