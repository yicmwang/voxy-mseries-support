struct BlockModel {
    uint faceData[6];
    uint flagsA;
    uint colourTint;
    uint customId;
    // VOXY_MODEL_TIGHT removes this padding, matching ModelStore.MODEL_SIZE (36 instead of 64). It is
    // the bytes-vs-latency discriminator for the vertex stage: the load count, the indices and the
    // arithmetic are all identical, so any change in `gpu` is attributable to the bytes per element.
    //
    // Only 36 of the 64 bytes were ever written -- ModelFactory fills faceData (0..23), flagsA (24),
    // colourTint (28) and customId (32) and stops -- so the padding carries no data and removing it
    // needs no change to the baker. 36 is a legal std430 array stride: the struct's alignment is 4,
    // held by its uint members, so the stride is just the size.
    //
    // The measured stakes: replacing this load entirely with a constant (`VOXY_LOD_NO_MODEL_FETCH`)
    // removes 13.0 % of the LOD pass's gpu. If trimming the stride recovers most of that, the cost is
    // bandwidth; if it recovers little, the cost is the dependent load's latency and the fix is to
    // bake the face constants into Quad so the vertex path has one load instead of two.
#ifndef VOXY_MODEL_TIGHT
    uint _pad[7];
#endif
};

//TODO: FIXME: this isnt actually correct cause depending on the face (i think) it could be 1/64 th of a position off
// but im going to assume that since we are dealing with huge render distances, this shouldent matter that much
float extractFaceIndentation(uint faceData) {
    uint enc = (faceData>>16)&63u;
    enc += uint(enc==63u);//convert 63 to 64 cause of pain reasons
    return float(enc)/64.0;
}

vec4 extractFaceSizes(uint faceData) {
    return (vec4(faceData&0xFu, (faceData>>4)&0xFu, (faceData>>8)&0xFu, (faceData>>12)&0xFu)/16.0)+vec4(0.0,1.0/16.0,0.0,1.0/16.0);
}

uint faceHasAlphaCuttout(uint faceData) {
    return (faceData>>22)&1u;
}

//TODO: try and get rid of
uint faceHasAlphaCuttoutOverride(uint faceData) {
    return (faceData>>23)&1u;
}

uint faceTintState(uint faceData) {
    return (faceData>>24)&3u;
}

bool modelHasBiomeLUT(BlockModel model) {
    return ((model.flagsA)&2u) != 0;
}

bool modelIsTranslucent(BlockModel model) {
    return ((model.flagsA)&4u) != 0;
}

bool modelIsShaded(BlockModel model) {
    return ((model.flagsA)&8u) != 0;
}