
#include "HEstruct.h"
#include "createHalfEdge.h"




// ========== function to create the half edge data structure from V and F (body)
void createHalfEdge(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, std::vector<HE_vert*> &vertList){

    // a map storing vertex indices pair (key value) and the corresponding half edge (mapped value)
    std::map< std::pair<int, int>, HE_edge* > edgePairsLookup;

    // go through each face
    for (int iF = 0; iF < F.rows(); iF++){

        // make a new face struct
        HE_face* thisFace;
        thisFace = new HE_face;

        // the vertex indices of this face
        int verts[] = {F(iF, 0), F(iF, 1), F(iF, 2), F(iF, 0), F(iF, 1)};

        // print out the 3 vertices of the current face
        // std::cout << "going through face with verts " << verts[0] << ' ' << verts[1] << ' ' << verts[2] << std::endl;

        // go through each edge in this face to link them to thisFace
        for (int iE = 0; iE < 3; iE++){

            // vertex pair corresponding to this half edge
            std::pair<int, int> thisE = std::make_pair(verts[iE], verts[iE+1]);

            // create new edge struct
            edgePairsLookup[thisE] = new HE_edge;
            // link this half edge to thisFace
            edgePairsLookup[thisE]->adjFace = thisFace;

            // if the start vertex of this edge doesn't have a structure
            if (nullptr == vertList[verts[iE]]){
                // create a vert struct
                vertList[verts[iE]] = new HE_vert;

                // save coordinates
                vertList[verts[iE]]->coord << V(verts[iE], 0), V(verts[iE], 1), V(verts[iE], 2);

                // save its index in V
                vertList[verts[iE]]->vInd = verts[iE];

                // link it to the edge originating from this vertex
                vertList[verts[iE]]->edge = edgePairsLookup[thisE];
            }
        }


        // link thisFace to the first edge in the face
        thisFace->edge = edgePairsLookup[std::make_pair(verts[0], verts[1])];


        // go through each edge again to link to each other
        for (int iE = 0; iE < 3; iE++){

            // vertex pairs corresponding to this half edge, next half edge, and the opposite half edge
            std::pair<int, int> thisE = std::make_pair(verts[iE], verts[iE+1]);
            std::pair<int, int> nextE = std::make_pair(verts[iE+1], verts[iE+2]);
            std::pair<int, int> oppositeE = std::make_pair(verts[iE+1], verts[iE]);

            // link this edge to the next edge
            edgePairsLookup[thisE]->nextEdge = edgePairsLookup[nextE];

            // link this edge to its end vertex
            edgePairsLookup[thisE]->endVert = vertList[verts[iE+1]];

            // if the other half edge exists
            if (edgePairsLookup.end() != edgePairsLookup.find(oppositeE)){
                // link the pair of half edges to each other
                edgePairsLookup[thisE]->pairEdge = edgePairsLookup[oppositeE];
                edgePairsLookup[oppositeE]->pairEdge = edgePairsLookup[thisE];
            }
        }
    }
}
