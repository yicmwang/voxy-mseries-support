#version 460 core
//Use quad shuffling to compute fragment mip
//#extension GL_KHR_shader_subgroup_quad: enable
// M9 Phase 2 patch: bumped to #version 460 so gl_HelperInvocation is a
// core built-in. shaderc/glslang's Vulkan profile doesn't accept
// GL_ARB_shader_helper_invocation as an extension; 4.50+ has it built-in.
#ifdef USE_SINGLE_TRI
#define USE_NV_BARRY
#endif

#ifdef USE_NV_BARRY
#extension GL_NV_fragment_shader_barycentric: require
#endif

layout(binding = 0) uniform sampler2D blockModelAtlas;
layout(binding = 2) uniform sampler2D depthTex;

#ifdef VOXY_METAL_BOUND_SSBO
// Metal-only (round 20): the chunk-bound mask arrives as raw floats in a
// plain buffer, NOT via depthTex — sampling a depth-format texture through
// the texture2d<float> declaration SPIRV-Cross emits for sampler2D silently
// reads ZEROS on Metal (same bug class as the round-18 Iris depth export),
// which left this mask inert since M13 chunk 3. ChunkBoundRenderer blits
// the bound depth into this buffer after the bound pass; width rides in
// the header so no pipeline rebuild is needed on resize.
// Binding 9: 0-5 are the terrain draw's buffer table, 6 is quads3.vert's
// per-draw UBO (Metal setVertexBytes slot — setBuffer would clobber it),
// 7/8 belong to the cmdgen compute defines.
layout(binding = 9, std430) readonly restrict buffer BoundDepthBuffer {
    uint boundWidth;
    uint _boundPad1;
    uint _boundPad2;
    uint _boundPad3;
    float boundDepths[];
};
#endif

#ifdef VOXY_LOD_CHUNK_CULL
// Per-SECTION cull of the LOD against the set of sections vanilla has drawn.
//
// Three dimensions, deliberately. Sodium enforces a vertical render distance as well as a
// horizontal one, so "this chunk column has geometry" does not mean the section above or below it
// does -- and a cull that believes otherwise removes LOD sections vanilla never drew, which is a
// hole in the air. Two earlier versions got this wrong in different ways: one tested a LOD node's
// centre column (a node is 2x2 columns at detail 0, so one bit decided four, leaving a hole ring
// around the rim of vanilla's coverage), and the next tested every column of the node but was
// still two-dimensional. A fragment knows its own position in all three axes, so it can ask about
// its own 16x16x16 section and be exact at every detail level.
//
// Bindings: 0-5 are the terrain draw's buffer table, 6 is quads3.vert's per-draw UBO, 9 is the
// Metal chunk-bound depth buffer, so 10 is free for this.
//
// Header is EIGHT uints so the uvec2 array starts at byte 32 and is 8-byte aligned, which std430
// requires for a vec2 array. Must match BuiltSectionMask.HEADER_UINTS exactly.
layout(binding = VOXY_LOD_CHUNK_CULL_BINDING, std430) readonly restrict buffer BuiltMaskChunkBuffer {
    uint chunkMaskSide;
    // The square's origin is WORLD-anchored, not the camera's column: it only moves when the
    // camera crosses a multiple of the anchor step. Indexing from a fixed world origin is what
    // stops the culled region's edge sliding along with the player.
    int chunkMaskAnchorSecX;
    int chunkMaskCamSecY;
    int chunkMaskAnchorSecZ;
    // The LOD FRAME's origin in world blocks, as FLOATS: viewport.section << 5, i.e. an exact
    // multiple of 32 -- NOT the camera's position, which is what this field used to hold and what
    // the name still half-suggests. A fragment's section is reconstructed as
    // floor(frameOrigin + rel), i.e. ADD BEFORE FLOORING. That order is kept deliberately even
    // though it is now moot -- the origin is integral, so floor(A + r) == A + floor(r) -- because
    // the camera form made it load-bearing:
    //
    //     floor(camX) + floor(fragX - camX)  ==  floor(fragX) - [frac(fragX) < frac(camX)]
    //
    // That stray -1 was a band across the world positioned by frac(camX) -- where the camera sits
    // inside its own block -- which made the culled region's edges crawl with the player at
    // sub-block granularity instead of stepping at chunk boundaries. Removing the camera from this
    // lookup is what fixed it; do not put it back. See cull.MD 8.12.
    float chunkMaskFrameOriginX;
    float chunkMaskFrameOriginY;
    float chunkMaskFrameOriginZ;
    uint _chunkMaskPad;
    // One 64-bit vertical bitmask per column, bit (secY - camSecY) + 32. Two uints rather than a
    // uint64_t because MSL translation of 64-bit GLSL integers is a risk not worth taking for a
    // value that is only ever shifted and tested.
    uvec2 chunkMaskColumnY[];
};
#endif

//#define DEBUG_RENDER

//TODO: need to fix when merged quads have discardAlpha set to false but they span multiple tiles
// however they are not a full block

layout(location = 0) in flat uvec4 interData;
#ifndef USE_NV_BARRY
layout(location = 1) in vec2 uv;
#endif

// M13 chunk 5: per-vertex world distance to camera, interpolated.
// 2026-07-03: decoupled from USE_ENV_FOG — the vx contract forces env fog
// off, which silently compiled out the distance-mip + far-water-alpha
// features in every Iris/BSL session. Must mirror quads3.vert's guard.
#if defined(USE_ENV_FOG) || defined(VOXY_LOD_DIST_MIP) || defined(VOXY_WATER_FAR_ALPHA) || defined(VOXY_TRANS_NEAR_CULL)
#define VOXY_NEEDS_FOG_DIST
#endif
#ifdef VOXY_NEEDS_FOG_DIST
layout(location = 2) in float voxyFogDist;
#endif
// location 3 was the near-cull's horizontal-only offset; see quads3.vert for why it is gone. The
// per-section cull reads location 4 (voxyFramePos) instead -- an earlier version of this comment
// claimed the cull used this varying to find its chunk column, which stopped being true when the cull
// moved from a column to a 16x16x16 section and never was true afterwards.
#ifdef VOXY_LOD_CHUNK_CULL
// The full 3D camera-relative offset, for the per-section cull: it asks about a 16x16x16 section,
// and Sodium's vertical render distance means the horizontal column is not enough to answer it.
// Must mirror quads3.vert's guard and location exactly.
layout(location = 4) in vec3 voxyFramePos;
#endif

#ifdef DEBUG_RENDER
layout(location = 7) in flat uint quadDebug;
#endif


#ifndef PATCHED_SHADER
layout(location = 0) out vec4 outColour;

#ifdef VOXY_LOD_DEPTH_COLOUR
// Second colour attachment: this fragment's depth, as a COLOUR value. This is the Hi-Z pyramid's
// source, and it exists because a depth-format texture sampled through `sampler2D` becomes MSL
// `texture2d<float>`, from which Metal silently reads ZEROS -- it needs `depth2d`. This file already
// records that measurement at the top, from the depth-bound mask's M13 work. Emitting depth as colour
// sidesteps it entirely: the pyramid samples an R32F texture, which is an ordinary texture2d<float>
// read, and no blit, buffer transfer or encoder transition is involved.
layout(location = 1) out float voxyDepthOut;
#endif
#else

//Bind the model buffer and import the model system as we need it
#define MODEL_BUFFER_BINDING 3
#import <voxy:lod/block_model.glsl>

#endif

#import <voxy:lod/gl46/bindings.glsl>

vec4 uint2vec4RGBA(uint colour) {
    return vec4((uvec4(colour)>>uvec4(24,16,8,0))&uvec4(0xFF))/255.0;
}

//bool useMipmaps() {
//    return (interData.x&2u)==0u;
//}

uint tintingState() {
    return (interData.x>>2)&3u;
}

bool useDiscard() {
    return (interData.x&1u)==1u;
}

uint getFace() {
    return (interData.x>>4)&7u;
}

#ifdef PATCHED_SHADER
vec2 getLightmap() {
    return clamp(vec2((interData.y>>4)&0xFu, interData.y&0xFu)/15, vec2(8.0f/256), vec2(248.0f/256));
}
#endif

uint getModelId() {
    return interData.x>>16;
}

vec2 getBaseUV() {
    uint face = getFace();
    uint modelId = interData.x>>16;
    vec2 modelUV = vec2(modelId&0xFFu, (modelId>>8)&0xFFu)*(1.0/(256.0));
    return modelUV + (vec2(face>>1, face&1u) * (1.0/(vec2(3.0, 2.0)*256.0)));
}


#ifdef PATCHED_SHADER
struct VoxyFragmentParameters {
    //TODO: pass in derivative data
    vec4 sampledColour;
    vec2 tile;
    vec2 uv;
    uint face;
    uint modelId;
    vec2 lightMap;
    vec4 tinting;
    uint customId;//Same as iris's modelId
};

void voxy_emitFragment(VoxyFragmentParameters parameters);
#else

vec4 computeColour(vec2 texturePos, vec4 colour) {
    //Conditional tinting, TODO: FIXME: this is better but still not great, try encode data into the top bit of alpha so its per pixel

    uint tintingFunction = tintingState();
    bool doTint = tintingFunction==2;//Always tint if function == 2
    if (tintingFunction == 1) {//partial tint
        vec4 tintTest = textureLod(blockModelAtlas, texturePos, 0);
        if (abs(tintTest.r-tintTest.g) < 0.02f && abs(tintTest.g-tintTest.b) < 0.02f) {
            doTint = true;
        }
    }
    if (doTint) {
        colour *= uint2vec4RGBA(interData.z).yzwx;
    }
    return (colour * uint2vec4RGBA(interData.y)) + vec4(0,0,0,float(interData.w&0xFFu)/255);
}

#endif


#ifdef VOXY_LOD_SHOW_DRAWID
layout(location = 5) in flat uint voxyDrawIdOut;
#endif

void main() {
#ifdef VOXY_LOD_SHOW_DRAWID
    // Paint the section index this vertex stage RECEIVED, 24 bits across RGB. Nothing else runs --
    // no atlas, no light, no discard -- so the image is a direct readout of the per-draw constant.
    // A correctly-delivered index is piecewise-constant over each drawn section and matches the draw
    // order; a coalesced or stuck constant shows the same value over runs of draws, or values that do
    // not change where the geometry does.
    // R is a MARKER, not data: exactly 1.0 means "this pixel was painted by the readout", which is what
    // lets an analysis separate LOD-painted pixels from vanilla terrain and sky without guessing at
    // colours. The index is the low 16 bits in G,B -- enough because a render list is at most
    // MAX_QUEUE_SIZE = 200_000, and the valid bound for any one frame is its listCount (measured at
    // ~5_100 here). So any marked pixel whose decoded index is >= that frame's listCount was never
    // written by cmdgen: the shader received an index that does not exist, which is the coalesced or
    // stale per-draw constant this readout exists to catch.
    outColour = vec4(1.0,
                     float((voxyDrawIdOut >> 8) & 0xFFu) / 255.0,
                     float(voxyDrawIdOut & 0xFFu) / 255.0, 1.0);
    return;
#endif
#ifdef VOXY_LOD_CHUNK_CULL
    // Where the chunk-bound depth mask belongs: before any shading, on every path, so a culled
    // column costs one buffer read and nothing else. Placed above the magenta/debug early-outs so
    // a forced-colour bisection still shows exactly what survives the cull.
    {
        // The fragment's own world section, in all three axes. Add the camera's exact position to
        // the camera-relative offset and floor ONCE, so the arithmetic cancels to floor(fragWorld)
        // exactly; see the header fields for why flooring the offset separately is wrong. Integer
        // floor via an arithmetic shift, not a truncating cast: the world extends either side of the
        // origin and `>>` floors a negative int, which is what BuiltSectionMask's own indexing assumes.
        // Frame origin plus the frame-relative position, floored once. NO CAMERA TERM: the cull asks
        // where the fragment is in the WORLD, and the world position is origin + cornerPoint, both of
        // which are independent of where the player is standing.
        int secX = int(floor(chunkMaskFrameOriginX + voxyFramePos.x)) >> 4;
        int secY = int(floor(chunkMaskFrameOriginY + voxyFramePos.y)) >> 4;
        int secZ = int(floor(chunkMaskFrameOriginZ + voxyFramePos.z)) >> 4;
        int sd = int(chunkMaskSide);
        // 0-based from the world-anchored origin. No centre bias, and nothing here depends on where
        // the camera is inside the square -- which is what stops the boundary being dragged.
        int cx = secX - chunkMaskAnchorSecX;
        int cz = secZ - chunkMaskAnchorSecZ;
#ifdef VOXY_LOD_CULL_DEBUG
        // Paint the SHADER'S OWN ARITHMETIC -- the column index it looked up and the bit it tested --
        // over every LOD fragment, before any discard. R = cx, G = cz, B = bit, low 8 bits each.
        //
        // This exists because the mask's log proves the producer is section-quantised (identical
        // columns and sections across 600 frames inside one section) while the region on screen
        // follows the camera. Both cannot be true unless the LOOKUP is wrong, and nothing in the Java
        // can see the lookup. A world-referenced fragment must keep the same colour as the camera
        // moves; if the colour field is glued to the screen instead, the lookup is camera-referenced
        // and that is the bug, made visible.
        outColour = vec4(float(cx & 0xFF) / 255.0,
                         float(cz & 0xFF) / 255.0,
                         float(((secY - chunkMaskCamSecY) + 32) & 0xFF) / 255.0,
                         1.0);
        return;
#endif
        // Outside the square, or outside the +/-512 blocks the per-column bitmask spans, is NOT
        // covered: keep the LOD. An over-drawn LOD z-fights, an under-drawn one shows the void.
        if (cx >= 0 && cz >= 0 && cx < sd && cz < sd) {
            int bit = (secY - chunkMaskCamSecY) + 32;
            if (bit >= 0 && bit < 64) {
                uvec2 col = chunkMaskColumnY[cz * sd + cx];
                uint word = bit < 32 ? col.x : col.y;
                if ((word & (1u << uint(bit & 31))) != 0u) {
                    // Vanilla draws this exact 16x16x16 section. Removing only this section is what
                    // keeps the section above it -- which vanilla may not draw at all -- intact,
                    // and that vertical case is what the missing LOD chunks turned out to be.
#ifdef VOXY_LOD_CULL_SHOW
                    // CULL_SHOW=1: paint what the cull removes instead of discarding it. Without this
                    // a hole in the LOD has two possible causes that look identical -- the cull
                    // removed it, or there is no LOD geometry there at all -- so a screenshot cannot
                    // tell a cull boundary from the edge of Voxy's own coverage. With it, combined
                    // with VOXY_LOD_FORCE_MAGENTA, the frame separates into three:
                    //   MAGENTA = LOD drawn   RED = culled by the section mask   SKY = no LOD geometry
                    outColour = vec4(1.0, 0.0, 0.0, 1.0);
                    return;
#else
                    discard;
#endif
                }
            }
        }
    }
#endif
#ifdef VOXY_LOD_FLAT_FRAG
    // VOXY_LOD_FLAT_FRAG=1 -- the fill-vs-per-draw probe; see MDICSectionRenderer.buildTerrainDefines.
    // Emits a constant colour and returns, skipping the per-fragment mip computation, all three atlas
    // fetches, the tinting, the alpha cutout and the fog.
    //
    // Deliberately placed AFTER the chunk-cull block rather than at the top of main() like
    // FORCE_MAGENTA. The question is what the SHADING costs, so the culled region must still be culled:
    // a probe that also restored the culled fragments would change what is drawn as well as what is
    // shaded, and would answer neither question.
    outColour = vec4(1.0, 0.0, 1.0, 1.0);
    return;
#endif
#ifdef VOXY_LOD_DEPTH_COLOUR
    // Written here rather than at the top of main(): a culled fragment must not contribute depth to the
    // pyramid, or the cull would occlude terrain behind the region it just culled. Anything that
    // discards LATER (the alpha cutout) still writes, which is a known approximation -- a tile covered
    // entirely by leaves can read as occluded. If that shows up as holes, move this below the cutout
    // test and give each early-out its own write.
    voxyDepthOut = gl_FragCoord.z;
#endif
#ifdef VOXY_LOD_FORCE_MAGENTA
    // VOXY_LOD_FORCE_MAGENTA=1 -- bisection, not a feature. Emits solid magenta as the FIRST
    // statement of main(), before the depth-bound test, the alpha discard, the tile clamp and
    // every early-out, so it answers one question only: does the LOD geometry rasterize at all?
    // Magenta on screen => the vertex stage, clipping, the pipeline state and the render encoder
    // are all fine, and the fault is downstream in shading/discards. No magenta => no fragment
    // of this pipeline ever reaches the framebuffer, whatever the draw counters say.
    outColour = vec4(1.0, 0.0, 1.0, 1.0);
    return;
#endif
#ifdef VOXY_LOD_SHOW_LIGHT
    // VOXY_LOD_SHOW_LIGHT=1 -- reads the LOD's own light data back out, before any shading.
    // Red = sky light, green = block light, each /15 (see makeRemainingAttributes in
    // quad_util.glsl for the packing); blue = the quad's LOD level /7, which is what separates
    // "the far field is dark because it is coarse" from "the far field is dark because those
    // particular voxels lost their light". Pure black in red+green means the voxel was baked
    // with BOTH nibbles zero, and not that geometry is missing: a missing quad draws nothing at
    // all and shows whatever is behind it.
    // Blue is a CONSTANT 0.5 on every fragment this shader emits. That is the whole point: it
    // separates "a fragment was drawn and its light is zero" from "no fragment was drawn here at
    // all and we are looking at whatever is behind". An earlier version of this readout inferred
    // "the black is drawn geometry with light 0" from the fact that non-black pixels varied --
    // which is invalid, because that only proves the readout works where it drew. The user, looking
    // at the real frame, could see through the black into the void, i.e. no fragment at all.
    // With blue pinned, b == 128 is a drawn fragment and anything else is not.
    outColour = vec4(float((interData.w >> 24) & 0xFu) / 15.0,
                     float((interData.w >> 28) & 0xFu) / 15.0,
                     0.5,
                     1.0);
    return;
#endif
#if defined(TRANSLUCENT) && !defined(PATCHED_SHADER)
    #ifdef VOXY_LOD_WATER_DEBUG
    // Diagnostic (VOXY_LOD_WATER_DEBUG=1): render translucent LOD water as
    // unmistakable solid magenta — no fog, no blend — so a screenshot shows
    // EXACTLY where the LOD water geometry rasterizes. Distinguishes "water
    // missing / clipped / depth-rejected" (no magenta where water should be)
    // from "water present but wrong colour/fog" (magenta is there, just the
    // normal path renders it wrong).
    outColour = vec4(1.0, 0.0, 1.0, 1.0);
    return;
    #endif
    #ifdef VOXY_FLAT_WATER
    // Escape hatch (VOXY_LOD_FLAT_WATER=1): the 2026-05-26 interim flat ocean
    // blue, fog-faded like the opaque terrain. The DEFAULT is now the real
    // translucent path below (atlas sample + biome tint + blend) — the old
    // gate was bare TRANSLUCENT, which is injected for EVERY backend in
    // MDICSectionRenderer (~:240), so the flat colour also hijacked plain-GL
    // runs. VOXY_FLAT_WATER is only injected inside the non-GL guard.
    vec3 waterColour = vec3(0.15, 0.42, 0.72);
    #ifdef USE_ENV_FOG
    if (voxyFogColour.a > 0.0) {
        float fogLerp = clamp(fma(voxyFogDist, voxyFogEndParams.x, voxyFogEndParams.y),
                              0.0, voxyFogEndParams.z);
        waterColour = mix(waterColour, voxyFogColour.rgb, fogLerp * voxyFogColour.a);
    }
    #endif
    outColour = vec4(waterColour, 1.0);
    return;
    #endif
#endif

// The translucent near-cull used to live here: a camera-distance test (3D slant, then horizontal
// Chebyshev, then Euclidean) for "is this fragment inside the MC render distance", discarded with a
// tuned margin. Three successive metrics, each fixing the last one's failure -- white veil squares,
// then a no-water ring out to RD*sqrt(2) over the seafloor -- because a distance is a proxy for what
// MC actually draws, and MC draws the sections it has BUILT.
//
// The per-section cull above answers that exactly, and it is injected into the COMMON defines, so the
// translucent pass already has it: LOD water inside a built section is culled by the same membership
// test as the terrain, with no threshold to tune and no metric to get wrong. The near-cull was also
// gated on IrisUtil.vxContractActive(), which this port does not have, so it was doubly inert -- and
// it was a footgun: had the vx contract returned, it would have fought the section cull with a
// distance. Deleted rather than left disabled.

    //vec2 uv = vec2(0);
    //Tile is the tile we are in
    vec2 tile;
    #ifdef USE_NV_BARRY
    #ifdef USE_SINGLE_TRI
    if (gl_BaryCoordNV.x>=0.5||gl_BaryCoordNV.y>=0.5) discard;
    vec2 uv = gl_BaryCoordNV.yx*(vec2((interData.x>>8)&0xFu, (interData.x>>12)&0xFu)+1)*2;
    #else
    vec2 uv = mix(gl_BaryCoordNV.yx, 1-gl_BaryCoordNV.xz, gl_PrimitiveID&1)*(vec2((interData.x>>8)&0xFu, (interData.x>>12)&0xFu)+1);
    #endif
    #endif

    vec2 uv2 = modf(uv, tile)*(1.0/(vec2(3.0,2.0)*256.0));
    vec4 colour;
    vec2 texPos = uv2 + getBaseUV();

#ifdef VOXY_NO_ATLAS
    // M12 Metal path: ModelTextureBakery is still GL-only, so the
    // blockModelAtlas + depthBoundingBuffer textures aren't populated /
    // bound. Skip atlas sampling and emit a deterministic per-quad
    // debug color hashed from `interData.x` (a flat varying carrying the
    // model id + face + flags — varies per quad / section). Then modulate
    // by the *real* MC lightmap colour packed into `interData.y` by the
    // vertex shader (see makeRemainingAttributes in quad_util.glsl —
    // it samples lightSampler and multiplies by computeDirectionalFaceTint,
    // which already encodes UP/DOWN/Z/X face shade from MC's level). M13
    // chunk 2 wired the lightmap sampler on Metal, so this is the real
    // lighting; the synthetic face-Lambertian shade from M12 is gone.
    // Drops the depth-bounding and alpha-discard checks that depend on
    // the unbound atlas textures. `gl_InstanceID` lives only in the
    // vertex stage so we can't use it here; interData.x gives sufficient
    // variation.
    {
        uint hash = interData.x * 2654435761u;
        hash ^= hash >> 13;
        hash *= 1274126177u;
        hash ^= hash >> 16;
        // Map the hash channels into [0.55, 1.0] so every block reads as
        // a saturated bright colour. The plain `(hash & 0xFF) / 255` from
        // earlier let random channels collapse near zero, which combined
        // with lightmap shading would have collapsed too dark for many
        // blocks. Bias the range so the visual contrast is always strong.
        colour = vec4(
            float((hash >>  0) & 0xFFu) / 255.0 * 0.45 + 0.55,
            float((hash >>  8) & 0xFFu) / 255.0 * 0.45 + 0.55,
            float((hash >> 16) & 0xFFu) / 255.0 * 0.45 + 0.55,
            1.0
        );
        // Modulate by the real lightmap + face-shade tinting baked into
        // interData.y by makeRemainingAttributes. Keep alpha at 1.0 —
        // interData.y's alpha channel carries packed face/lod metadata
        // for the non-translucent path (see `addin` in quad_util.glsl)
        // and would zero the fragment.
        colour.rgb *= uint2vec4RGBA(interData.y).rgb;

        // Procedural per-pixel pattern — gives each face a "textured"
        // look instead of a solid colour. Combines a small-grid checker
        // (4x4 cells per quad) with a value-noise speckle so flat block
        // colours read as 3D-textured surfaces. Real model textures from
        // ModelTextureBakery are still M13 chunk 1 — this is the
        // VOXY_NO_ATLAS debug visualization until that lands.
#ifndef USE_NV_BARRY
        {
            uint face = getFace();
            // 4x4 grid checker — gives subtle "tile" structure.
            ivec2 cell = ivec2(floor(uv * 4.0));
            float checker = ((cell.x ^ cell.y) & 1) == 0 ? 1.0 : 0.85;
            // Value-noise speckle — high-frequency variation hiding the
            // flat per-quad fill. Hash from (uv * 16, face) so the
            // pattern is stable per-pixel but uncorrelated across faces.
            vec2 noiseInput = uv * 16.0 + float(face) * 17.0;
            float noise = fract(sin(dot(noiseInput, vec2(12.9898, 78.233))) * 43758.5453);
            float speckle = 0.88 + 0.12 * noise;
            colour.rgb *= checker * speckle;
        }
#endif
    }
#else
//This is deprecated, TODO: remove the non mip code path
    //if (useMipmaps())
    {
#ifdef VOXY_LOD_DIST_MIP
        // Distance-based mip selection (Metal, 2026-07-03). Fixed mip 0 made
        // every distant pixel pick one arbitrary texel of its 16x16 face cell
        // (NEAREST + no minification) — the spyglass moire/shimmer on LOD
        // water and the pixel noise on distant terrain. Screen-space
        // derivatives are NOT trustworthy here (1-2 px quads gave the noisy
        // dFdx that forced fixed-mip in the first place), so compute the mip
        // ANALYTICALLY: one atlas texel covers lodScale/16 world units; one
        // screen pixel covers voxyFogDist * voxyLodParams.x world units
        // (2*tan(fovY/2)/viewportH, per frame — tracks spyglass zoom).
        // Clamp to VOXY_ATLAS_MAX_LOD: bakes upload mips 16/8/4/2 only.
        float voxyAtlasLod = 0.0;
        #ifdef VOXY_NEEDS_FOG_DIST
        if (voxyLodParams.x > 0.0) {
            float texelWorld = float(1u<<((interData.w>>16)&7u)) * (1.0/16.0);
            float pixelWorld = voxyFogDist * voxyLodParams.x;
            voxyAtlasLod = clamp(log2(max(pixelWorld, 1e-6) / texelWorld) + VOXY_LOD_DIST_MIP_BIAS,
                                 0.0, VOXY_ATLAS_MAX_LOD);
        }
        #endif
        colour = textureLod(blockModelAtlas, texPos, voxyAtlasLod);
#elif defined(VOXY_LOD_FIXED_MIP)
        // DIAGNOSTIC (2026-05-25): sample the atlas at a fixed LOD 0 instead of
        // the derivative-based mip. Tests whether the LOD flicker is unstable
        // mip selection on small/distant quads (noisy dFdx/dFdy) — the "small
        // quad is really fking over the mipping level" issue noted below.
        colour = textureLod(blockModelAtlas, texPos, 0.0);
#else
        vec2 uvSmol = uv*(1.0/(vec2(3.0,2.0)*256.0));
        vec2 dx = dFdx(uvSmol);//vec2(lDx, dDx);
        vec2 dy = dFdy(uvSmol);//vec2(lDy, dDy);
        colour = textureGrad(blockModelAtlas, texPos, dx, dy);
#endif
    }// else {
    //    colour = textureLod(blockModelAtlas, texPos, 0);
    //}

    // Metal bakery debug aid. The normal Metal path samples the real model
    // atlas; define VOXY_DEBUG_MAGENTA_MISSING to make any fully empty bake
    // cell visible instead of silently black/discarded.
    #ifdef VOXY_DEBUG_MAGENTA_MISSING
    if (colour.a == 0.0) {
        outColour = vec4(1.0, 0.0, 1.0, 1.0);
        return;
    }
    #endif
#endif

    //If we are in shaders and are a helper invocation, just exit, as it enables extra performance gains for small sized
    // fragments, we do this here after derivative computation
    //Trying it with all shaders
    //#ifdef PATCHED_SHADER
    #ifndef PATCHED_SHADER_ALLOW_DERIVATIVES
    if (gl_HelperInvocation) {
        return;
    }
    #endif
    //#endif

    if (any(notEqual(clamp(tile, vec2(0), vec2((interData.x>>8)&0xFu, (interData.x>>12)&0xFu)), tile))) {
        discard;
        return;
    }

#ifndef VOXY_NO_DEPTH_BOUND
    //Check the minimum bounding texture and ensure we are greater than it.
    // M13 chunk 1 split: this used to live under `#ifndef VOXY_NO_ATLAS` so
    // the atlas-disabled debug path also skipped the depth-bounding check.
    // M13 chunk 3: the chunk-bound depth mask now renders on Metal too
    // (ChunkBoundRenderer.renderMetal → depthBoundingBuffer, bound at
    // texture slot 2), so this check is ON by default on every backend;
    // VOXY_NO_DEPTH_BOUND=1 is the Metal kill switch that removes it.
#ifdef VOXY_METAL_BOUND_SSBO
    float voxyBoundDepth = boundDepths[uint(gl_FragCoord.y) * boundWidth + uint(gl_FragCoord.x)];
#else
    float voxyBoundDepth = texelFetch(depthTex, ivec2(gl_FragCoord.xy), 0).r;
#endif
    if (gl_FragCoord.z < voxyBoundDepth) {
        #ifdef VOXY_BOUND_DEBUG
        // VOXY_BOUND_DEBUG=1 (Metal mask-verification aid): paint the
        // bound-discarded fragments solid red instead of discarding so a
        // screenshot shows exactly where the chunk-bound depth mask bites.
        outColour = vec4(1.0, 0.0, 0.0, 1.0);
        return;
        #else
        discard;
        return;
        #endif
    }
#endif // VOXY_NO_DEPTH_BOUND

#ifndef VOXY_NO_ATLAS
#ifndef VOXY_LOD_NO_DISCARD
    //Also, small quad is really fking over the mipping level somehow
    #ifndef TRANSLUCENT
    if (useDiscard() && (textureLod(blockModelAtlas, texPos, 0).a <= 0.1f)) {
    //if (useDiscard() && (colour.a <= 0.1f)) {
    #else
    if (textureLod(blockModelAtlas, texPos, 0).a == 0.0f) {
    #endif
        //This is stupidly stupidly bad for divergence
        //TODO: FIXME, basicly what this do is sample the exact pixel (no lod) for discarding, this stops mipmapping fucking it over
        #ifndef DEBUG_RENDER
        discard;
        return;
        #endif
    }
#endif // VOXY_LOD_NO_DISCARD
#endif // VOXY_NO_ATLAS — closes the alpha-discard block above

    #ifndef PATCHED_SHADER_ALLOW_DERIVATIVES
    if (gl_HelperInvocation) {
        return;
    }
    #endif

    #ifndef PATCHED_SHADER
#ifdef VOXY_NO_ATLAS
    // Already computed a debug colour up top; skip computeColour (which
    // re-samples blockModelAtlas via textureLod). Emit straight to outColour.
    outColour = colour;
#else
    colour = computeColour(texPos, colour);
    outColour = colour;
    #ifdef VOXY_FORCE_OPAQUE_ALPHA
    // Metal composites the IOSurface back into Minecraft's main render target.
    // The opaque path otherwise writes LOD/face metadata into alpha, which
    // spyglass/post overlays can interpret as real framebuffer transparency.
    outColour.a = 1.0;
    #endif
#endif

    // M13 chunk 5: environmental fog on the Metal terrain path. Mirrors the
    // GL post-pass formula from blit_texture_depth_cutout.frag (lines 71–74)
    // so distant LOD chunks fade into the sky/biome fog colour the same way
    // Sodium's near terrain does. Injected only on the Metal pipeline (see
    // MDICSectionRenderer constructor) — the GL pipeline still applies fog
    // in the depth-cutout post-pass and would double-apply if this branch
    // also ran. fogColour.a == 0 short-circuits so a feature-flagged-off
    // upload (zero alpha) is cheap.
    // Seam-ring brightness parity (Metal): GL runs ssao.comp between opaque
    // and translucent, approximating the vertex AO Sodium bakes into its near
    // terrain; that pass is parked on Metal, leaving LOD ~10% brighter than
    // the AO-darkened Sodium terrain at the render-distance boundary. Applied
    // BEFORE fog so the compensation darkens terrain, not the fog colour.
    // Injected Metal-only with the tunable value (VOXY_LOD_BRIGHTNESS env).
    #ifdef VOXY_LOD_BRIGHTNESS
    outColour.rgb *= VOXY_LOD_BRIGHTNESS;
    #endif

    // Water-ring parity (Metal, translucent program only): LOD water reads
    // pale and shallow next to MC's near water — MC water darkens with depth
    // while LOD water blends a light texture over the bright fog-coloured
    // bridge clear. Darken it and floor its opacity so the boundary reads as
    // continuous deep water. VOXY_WATER_SHADE / VOXY_WATER_MIN_ALPHA envs.
    #ifdef VOXY_WATER_SHADE
    outColour.rgb *= VOXY_WATER_SHADE;
    #endif
    #ifdef VOXY_WATER_MIN_ALPHA
    outColour.a = max(outColour.a, VOXY_WATER_MIN_ALPHA);
    #endif

    // Far-water opacity ramp (Metal, translucent only, 2026-07-03). MC water
    // is alpha 0.706 and that parity MUST hold at the LOD<->MC seam — but a
    // constant 0.706 out to the horizon lets kilometre-deep seafloor/kelp
    // ghost through the surface and lets the fog-coloured bridge clear bleed
    // up through it (the washed-out flat-blue sheet). Physically, the view
    // path through water at those grazing distances is opaque. Smoothstep
    // the alpha from the vanilla texel value at voxyLodParams.y blocks (past
    // the seam, so ring parity is untouched) to voxyLodParams.w at the far
    // end. w == 0 disables (VOXY_WATER_FAR_ALPHA=0 kill switch, params from
    // MDICSectionRenderer.uploadUniform).
    #if defined(TRANSLUCENT) && defined(VOXY_WATER_FAR_ALPHA)
    if (voxyLodParams.w > 0.0) {
        float farLerp = clamp((voxyFogDist - voxyLodParams.y) * voxyLodParams.z, 0.0, 1.0);
        farLerp = farLerp * farLerp * (3.0 - 2.0 * farLerp);
        outColour.a = mix(outColour.a, max(outColour.a, voxyLodParams.w), farLerp);
    }
    #endif

    #ifdef USE_ENV_FOG
    if (voxyFogColour.a > 0.0) {
        float fogLerp = clamp(fma(voxyFogDist, voxyFogEndParams.x, voxyFogEndParams.y),
                              0.0, voxyFogEndParams.z);
        outColour.rgb = mix(outColour.rgb, voxyFogColour.rgb, fogLerp * voxyFogColour.a);
    }
    #endif

    #ifdef DEBUG_RENDER
    uint hash = quadDebug*1231421+123141;
    hash ^= hash>>16;
    hash = hash*1231421+123141;
    hash ^= hash>>16;
    hash = hash * 1827364925 + 123325621;
    outColour = vec4(float(hash&15u)/15, float((hash>>4)&15u)/15, float((hash>>8)&15u)/15, 0);
    #endif

    #else
    uint modelId = getModelId();
    BlockModel model = modelData[modelId];
    uint tintingFunction = tintingState();
    bool doTint = tintingFunction==2;//Always tint if function == 2
    if (tintingFunction==1) {//Partial tint
        vec4 tintTest = texture(blockModelAtlas, texPos, -2);
        if (abs(tintTest.r-tintTest.g) < 0.02f && abs(tintTest.g-tintTest.b) < 0.02f) {
            doTint = true;
        }
    }
    vec4 tint = vec4(1);
    if (doTint) {
        tint = uint2vec4RGBA(interData.z).yzwx;
    }

    uint face = getFace();
    face ^= uint((face&1u)!=uint(gl_FrontFacing!=((face>>1)!=0u)));
    voxy_emitFragment(VoxyFragmentParameters(colour, tile, texPos, face, modelId, getLightmap(), tint, model.customId));

    #endif
}



//#ifdef GL_KHR_shader_subgroup_quad
/*
uint hash = (uint(tile.x)*(1<<16))^uint(tile.y);
uint horiz = subgroupQuadSwapHorizontal(hash);
bool sameTile = horiz==hash;
uint sv = mix(uint(-1), hash, sameTile);
uint vert = subgroupQuadSwapVertical(sv);
sameTile = sameTile&&vert==hash;
mipBias = sameTile?0:-5.0;
*/
/*
vec2 uvSmol = uv*(1.0/(vec2(3.0,2.0)*256.0));
float lDx = subgroupQuadSwapHorizontal(uvSmol.x)-uvSmol.x;
float lDy = subgroupQuadSwapVertical(uvSmol.y)-uvSmol.y;
float dDx = subgroupQuadSwapDiagonal(lDx);
float dDy = subgroupQuadSwapDiagonal(lDy);
vec2 dx = vec2(lDx, dDx);
vec2 dy = vec2(lDy, dDy);
colour = textureGrad(blockModelAtlas, texPos, dx, dy);
*/
//#else
//colour = texture(blockModelAtlas, texPos);
//#endif
