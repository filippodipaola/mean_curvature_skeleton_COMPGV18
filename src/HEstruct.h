#IFNDEF HESTRUCT_H
#DEFINE HESTRUCT_H

struct HE_edge {
    // the other half edge in the opposite direction
    HE_edge* pairEdge;
    // vertex at the end of this half edge
    HE_vert* endVert;
    // face the half edge borders
    HE_face* adjFace;
    // next half edge
    HE_edge* nextEdge;

};

// vertex struct
struct HE_vert {
    // coordinates
    Vector3d coord;
    // index in V
    int vInd;
    // one half edge coming from this vertex
    HE_edge* edge;
};

// face strcut
struct HE_face {
    // one half edge that borders the face
    HE_edge* edge;
};

#ENDIF