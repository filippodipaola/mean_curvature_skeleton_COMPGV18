#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/StdVector>
#include <MatOp/SparseSymMatProd.h>
#include <MatOp/SparseGenMatProd.h>
#include <MatOp/SparseCholesky.h>
#include <math.h>
#include "buildLapBeltMat.cpp"
#include "HEstruct.h"

using namespace Eigen;

// ========== function to build the two sparse matrices M, C that make up the laplace-beltrami operator
void buildLapBeltMat(const std::vector<HE_vert*> &HE_vertList, SparseMat &Minv_LB, SparseMat &C_LB){

    // two vectors of triplets storing the indices and values of the non-zero values in M and C
    std::vector<Trip> nonZerosList_Minv, nonZerosList_C;

    // go through each vertex
    for (int iV = 0; iV < HE_vertList.size(); iV++) {

        // area sum
        double areaSum = 0;

        // cotan sum
        double cotanSum = 0;

        // coordinate of this vertex
        Vector3d thisVertCoord = HE_vertList[iV]->coord;

        // edge pointer to traverse through all the edges emanating from this vertex
        HE_edge* pEdge = HE_vertList[iV]->edge;

        // iterating over all edges emanating from this vertex
        do{
            // --- first face bordering this edge
            // coordinates of 2 neighbouring vertices
            Vector3d nCoord_1 = pEdge->endVert->coord;
            Vector3d nCoord_2 = pEdge->nextEdge->endVert->coord;

            // two vectors (from neighbour2 to thisVert and neighbour1)
            Vector3d vec1 = thisVertCoord - nCoord_2;
            Vector3d vec2 = nCoord_1 - nCoord_2;

            // cotan beta (cos/sin)
            double cotan_b = vec1.dot(vec2) / (vec1.cross(vec2)).norm();

            // 1/3 of this triangle area
            areaSum += (vec1.cross(vec2)).norm() / 2 / 3;


            // --- the other face bordering this edge
            // coordinates of the two neighbouring vertices
            nCoord_1 = pEdge->pairEdge->nextEdge->endVert->coord;
            nCoord_2 = pEdge->pairEdge->nextEdge->nextEdge->endVert->coord;

            // the two vectors
            vec1 = thisVertCoord - nCoord_1;
            vec2 = nCoord_2 - nCoord_1;

            // cotan alpha (cos/sin)
            double cotan_a = vec1.dot(vec2) / (vec1.cross(vec2)).norm();

            // append the non-zero element (neighbours) to C list
            nonZerosList_C.push_back(Trip(iV, pEdge->endVert->vInd, cotan_a + cotan_b));

            // cotangent sum
            cotanSum -= (cotan_a + cotan_b);

            // move to the next edge emanating from this vertex
            pEdge = pEdge->pairEdge->nextEdge;
        } while (pEdge != HE_vertList[iV]->edge);

        // append the non-zero element (thisVert) to C list
        nonZerosList_C.push_back(Trip(iV, iV, cotanSum));

        // append the non-zero element (thisVert) to Minv list
        nonZerosList_Minv.push_back(Trip(iV, iV, 1/(2 * areaSum)));
    }

    // build sparse matrices from triplets
    Minv_LB.setFromTriplets(nonZerosList_Minv.begin(), nonZerosList_Minv.end());
    C_LB.setFromTriplets(nonZerosList_C.begin(), nonZerosList_C.end());
}