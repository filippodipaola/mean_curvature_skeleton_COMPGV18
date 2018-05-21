#include <iostream>
#include <igl/viewer/Viewer.h>
#include <igl/edges.h>
#include <igl/cotmatrix.h>
#include <nanogui/formhelper.h>
#include <nanogui/screen.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/StdVector>
#include <Eigen/SparseCholesky>
#include <Eigen/OrderingMethods>
#include <numeric>
#include <math.h>
#include <random>
#include <SymEigsSolver.h>
#include <GenEigsSolver.h>
#include <SymGEigsSolver.h>
#include <MatOp/SparseSymMatProd.h>
#include <MatOp/SparseGenMatProd.h>
#include <MatOp/SparseCholesky.h>

#include <igl/massmatrix.h>
#include <igl/invert_diag.h>
#include <igl/per_vertex_normals.h>


// Qhull Library Imports.
#include "libqhullcpp/QhullError.h"
#include "libqhullcpp/QhullQh.h"
#include "libqhullcpp/QhullFacet.h"
#include "libqhullcpp/QhullFacetList.h"
#include "libqhullcpp/QhullFacetSet.h"
#include "libqhullcpp/QhullRidge.h"
#include "libqhullcpp/QhullLinkedList.h"
#include "libqhullcpp/QhullVertex.h"
#include "libqhullcpp/Qhull.h"
#include "libqhullcpp/PointCoordinates.h"
#include "libqhullcpp/RboxPoints.h"
#include "libqhullcpp/QhullPoints.h"

//using namespace Eigen;


// define types for sparse matrix and triplets
typedef Eigen::SparseMatrix<double> SparseMat;
typedef Eigen::Triplet<double> Trip;

// ===============================================================================================
// ==================== half edge data structure

// declare structs
struct HE_edge;
struct HE_vert;
struct HE_face;

// half edge struct
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
// ===============================================================================================


// ========== function to create the half edge data structure from V and F
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



// ========== function to build the laplace-beltrami operator
void buildLaplaceBeltrami(const std::vector<HE_vert*> &HE_vertList, const Eigen::MatrixXd &V, SparseMat &L){

    long vNum = HE_vertList.size();

    // two vectors of triplets storing the indices and values of the non-zero values in M and C
    std::vector<Trip> nonZerosList_Minv, nonZerosList_C;

    // go through each vertex
    for (int iV = 0; iV < vNum; iV++) {

        // area sum
        double areaSum = 0;

        // cotan sum
        double cotanSum = 0;

        // coordinate of this vertex
        Eigen::Vector3d thisVertCoord = V.row(HE_vertList[iV]->vInd);

        // edge pointer to traverse through all the edges emanating from this vertex
        HE_edge* pEdge = HE_vertList[iV]->edge;

        // iterating over all edges emanating from this vertex
        do{
            // --- first face bordering this edge
            // coordinates of 2 neighbouring vertices
            Eigen::Vector3d nCoord_1 = V.row(pEdge->endVert->vInd);
            Eigen::Vector3d nCoord_2 = V.row(pEdge->nextEdge->endVert->vInd);

            // two vectors (from neighbour2 to thisVert and neighbour1)
            Eigen::Vector3d vec1 = thisVertCoord - nCoord_2;
            Eigen::Vector3d vec2 = nCoord_1 - nCoord_2;

            // cotan beta (cos/sin)
            double cotan_b = vec1.dot(vec2) / (vec1.cross(vec2)).norm();

            // 1/3 of this triangle area
            areaSum += (vec1.cross(vec2)).norm() / 2 / 3;


            // --- the other face bordering this edge
            // coordinates of the two neighbouring vertices
            nCoord_1 = V.row(pEdge->pairEdge->nextEdge->endVert->vInd);
            nCoord_2 = V.row(pEdge->pairEdge->nextEdge->nextEdge->endVert->vInd);

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
    SparseMat Minv_LB(vNum, vNum);
    SparseMat C_LB(vNum, vNum);

    Minv_LB.setFromTriplets(nonZerosList_Minv.begin(), nonZerosList_Minv.end());
    C_LB.setFromTriplets(nonZerosList_C.begin(), nonZerosList_C.end());

    L = Minv_LB * C_LB;
}


// ========== function to compute the mean curvature H at each vertex
void getMeanCurv(const Eigen::MatrixXd &V, const SparseMat &L, Eigen::VectorXd &H){

    // matrix to store the results after applying L to the vertices
    Eigen::MatrixXd result(V.rows(), V.cols());

    // apply L to the vertices
    result = L * V;

    // compute H
    H = result.rowwise().norm() / 2;
}






// ===============================================================================================

// ========== visualisation
void displayMesh(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, igl::viewer::Viewer &viewer) {
    // clear viewer
    viewer.data.clear();
    // Plot the mesh
    viewer.data.set_mesh(V, F);
}


// ========== function to get voronoi poles
Eigen::MatrixXd getVoronoiPoles(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F) {
    // FOR VORONOI CALCULATIONS
    std::vector<Eigen::Vector3d> voronoi_vertices;
    std::vector< std::vector<uint> > cells;

    std::vector<int> poleof;				// The index of the voronoi pole the vertex refers to
    std::vector< std::vector<int> > scorr; 	// The index the surface sample to which a pole corresponds to

    std::vector<double> alpha;
    std::vector<double> radii;

    int nvertices, nvornoi;

    // Created the PointCoordinates object in order to use Qhull
    orgQhull::PointCoordinates *points;
    points = new orgQhull::PointCoordinates(V.cols(), " ");

    // This vector is used to convert the matrix V into a 1D list of coordinates.
    std::vector<double> everyPoint;
    // Populating the vector with every point in the Matrix V.
    for (int x = 0; x < V.rows(); x++) {
        for (int y = 0; y<V.cols();y++) {
            everyPoint.push_back(V(x,y));
        }
    }

    // Appending the iD vector of points to the PointCoordinates object for Qhull.
    points->append(everyPoint);
    // Declaring the Qhull object used for the Voronoi Diagram.
    orgQhull::Qhull *qhull;
    // Instantiating AND running the Voronoi algorithm to produce the Voronoi diagram.
    qhull= new orgQhull::Qhull(" ", points->dimension(),points->count(),&*points->coordinates(),"v Qbb");
    //std::cout << "Sucessfully instantiated and ran Qhull" << std::endl;
    //std::cout << "\tNumber of Points: " << points->count() <<std::endl;
    //std::cout << "\tDimensions: " << points->dimension() <<std::endl;

    // Declaring the Voronoi matrix used to hold the Voronoi vertices.
    Eigen::MatrixXd voro_mat(qhull->facetList().size(),V.cols());
    //  Stores the Veronoi Cells used later on in the pole calculations.
    // Resized to be the same size as the number of vertices of the mesh.
    cells.resize( V.rows() );
    int i = 0;

    for (orgQhull::QhullFacet f : qhull->facetList())
    {

        if(f.isUpperDelaunay()) continue;

        orgQhull::QhullPoint qhpnt = f.voronoiVertex();
        Eigen::Vector3d p(qhpnt[0], qhpnt[1], qhpnt[2]);
        // Populate the voronoi matrix with points from the Qhull object.
        voro_mat(i,0) = qhpnt[0];
        voro_mat(i,1) = qhpnt[1];
        voro_mat(i,2) = qhpnt[2];
        //if(!mesh->bbox().contains(p)) continue;
        // Start populating a vertices with voronoi_vertices.
        voronoi_vertices.push_back(p);

        for (orgQhull::QhullVertex v : f.vertices()) {
            cells[v.point().id()].push_back(i);
        }
        i++;
    }

    nvertices = V.rows();
    nvornoi = voronoi_vertices.size();
    //std::cout << "Voronoi Vertices:\n " << voro_mat << std::endl;
    //std::cout << "Voronoi Row Count:\n " << voro_mat.rows() << std::endl;
    Eigen::MatrixXd N_vertices, voronoi_poles(V.rows(), V.cols());
    // Populate N_vertices matrix with the vertex normals from the
    // mesh.
    igl::per_vertex_normals(V,F,N_vertices);

    Eigen::Vector3d surf_vertex;
    Eigen::Vector3d voro_vertex;
    Eigen::Vector3d surf_normal;

    poleof = std::vector<int>(nvertices, 0);
    scorr  = std::vector< std::vector<int> > (nvornoi, std::vector<int>(4, 0));

    std::vector<int> counter(nvornoi, 0);

    for(uint sidx = 0; (int)sidx < nvertices; sidx++) {
        surf_vertex = V.row(sidx);
        surf_normal = N_vertices.row(sidx);

        // Assigns max_neg_t to the largest possible double value
        double max_neg_t = DBL_MAX;
        // Assigns max_neg_i to zero, as max_neg_i must be less than 0.
        double max_neg_i = 0;

        for(int j = 0; j < (int)cells[sidx].size(); j++)
        {
            int vidx = cells[sidx][j];

            voro_vertex = voronoi_vertices[vidx];

            // Mark the fact that (voronoi) vidx corresponds to (surface) sidx
            // (in the next free location and update this location)
            // the freesub-th correspondent of the voronoi vertex is vidx
            int freesub = counter[vidx];
            if( freesub < 4 ){
                counter[vidx]++;
                scorr[vidx][freesub] = sidx;
            }

            // Project the voronoi vertex on the vertex normal & Retain furthest
            double t = (voro_vertex - surf_vertex).dot(surf_normal);
            if(t < 0 && t < max_neg_t){
                max_neg_t = t;
                max_neg_i = vidx;
            }
        }

        // Save pole to which surface corresponds
        // Store index (possibly nan!! into buffer)
        poleof[sidx] = max_neg_i;
        voronoi_poles.row(sidx) = voronoi_vertices[max_neg_i];
    }
    return voronoi_poles;
}


// ========== function for 4.2
void discreteMeanCurvatureFlow(Eigen::MatrixXd &V, const Eigen::MatrixXi &F, const SparseMat &L, const std::vector<double> &W_H) {


    long vNum = V.rows();

//    // identity matrix times w_H
//    SparseMat W_H(vNum, vNum);
//    W_H.setIdentity();
//    W_H = w_H * W_H;

    // triplets for the left hand side sparse matrix A
    std::vector<Trip> A_tripList;

    A_tripList.reserve(L.nonZeros() + vNum);

    // top portion of left hand side (Laplace-beltrami)

    // go through the non-zero elements in L
    for (int k = 0; k < L.outerSize(); ++k) {
        for (SparseMat::InnerIterator it(L, k); it; ++it) {
            A_tripList.push_back(Trip(it.row(), it.col(), it.value()));
        }
    }

    // bottom portion of left hand side (W_H)
    for (int k = 0; k < vNum; k++) {
        A_tripList.push_back(Trip(vNum + k, k, W_H[k]));
    }


    // build the left hand side sparse matrix A from triplets
    SparseMat A(vNum * 2, vNum);
    A.setFromTriplets(A_tripList.begin(), A_tripList.end());

    // make LHS a symmetric matrix A^2 (= A'A) to use cholesky solver
    SparseMat A_trans = A.transpose();
    SparseMat LHS = A_trans * A;


    // create solver that uses cholesky decomposition
    Eigen::SimplicialLLT<SparseMat> cholSolver;


    // left hand side of the symmetric linear system
    cholSolver.compute(LHS);

    if(cholSolver.info()!= Eigen::Success) {
        // decomposition failed
        std::cout << "\n Decomposition failed \n" << std::endl;
        return;
    }

    // right hand side of the equation
    Eigen::MatrixXd b = Eigen::MatrixXd::Zero(vNum * 2, 3);

    // b is [0; W_H * V]
    for (int k = 0; k < vNum; k++) {
        b.row(k + vNum) = W_H[k] * V.row(k);
    }

    // RHS is A' * b
    Eigen::MatrixXd RHS = A_trans * b;


    // solving Ax = b
    V.col(0) = cholSolver.solve(RHS.col(0));
    V.col(1) = cholSolver.solve(RHS.col(1));
    V.col(2) = cholSolver.solve(RHS.col(2));

    if(cholSolver.info()!= Eigen::Success) {
        // solving failed
        std::cout << "\n Solving failed \n" << std::endl;
        return;
    }
}


// ========== function for 4.4 (1) - test degeneracy by checking if local area is of disk topology
void testDegeneracy(const std::vector<HE_vert*> &HE_vertList, const Eigen::MatrixXd &V, double edgeThresh, std::vector<bool> &fixedVerts) {

    long vNum = HE_vertList.size();

    edgeThresh = edgeThresh / 10;

    // for each vertex
    for (int iV = 0; iV < vNum; iV++) {
        // if it's not already fixed
        if (!fixedVerts[iV]) {

            int badEdge = 0;

            // edge pointer to traverse through all the edges emanating from this vertex
            HE_edge* pEdge = HE_vertList[iV]->edge;

            // for each edge emanating from this vertex
            do{
                // two other vertices of this face
                Eigen::Vector3d vert1 = V.row(pEdge->endVert->vInd);
                Eigen::Vector3d vert2 = V.row(pEdge->nextEdge->endVert->vInd);
                // length of the opposite edge
                double oppEdgeLength = (vert1 - vert2).norm();

                // if length is smaller than threshold
                if (oppEdgeLength < edgeThresh) {
                    // increase bad edge counter
                    badEdge++;
                }

                // move to the next edge emanating from this vertex
                pEdge = pEdge->pairEdge->nextEdge;
            } while (pEdge != HE_vertList[iV]->edge);

            // degenerate if more than one bad edge, fix this vertex
            fixedVerts[iV] = (badEdge > 1);
        }
    }


}


// ========== function for 4.4 (2) - update the parameters for fixed vertices
void updateParameters(const std::vector<bool> fixedVerts, double w_L, double w_H, double w_P, std::vector<double> &W_L, std::vector<double> &W_H, std::vector<double> &W_P) {
    for (int iV = 0; iV < fixedVerts.size(); iV++) {
        if (fixedVerts[iV]) {
            // W_L and W_P to zero, W_H to a large number, to fix the points
            W_L[iV] = 0;
            W_H[iV] = 1e10;
            W_P[iV] = 0;

            // print to debug
            std::cout << iV << "th vertex fixed" << std::endl;
        }
        else {
            // ensure the non-fixed vertices have the right parameters
            W_L[iV] = w_L;
            W_H[iV] = w_H;
            W_P[iV] = w_P;
        }
    }
}


// ========== function for 4.5
void medialSkeletonisationFlow (Eigen::MatrixXd &V, const Eigen::MatrixXi &F, const SparseMat &L, const Eigen::MatrixXd &vPoles, const std::vector<double> &W_L, const std::vector<double> &W_H, const std::vector<double> &W_P) {

    long vNum = V.rows();


    // triplets for the left hand side sparse matrix A
    std::vector<Trip> A_tripList;
    A_tripList.reserve(L.nonZeros() + vNum * 2);


    // A is [W_L * L; W_H; W_P]

    // top portion of left hand side (W_L * L)
    // go through the non-zero elements in L
    for (int k = 0; k < L.outerSize(); ++k) {
        for (SparseMat::InnerIterator it(L, k); it; ++it) {
            A_tripList.push_back(Trip(it.row(), it.col(), W_L[it.row()] * it.value()));
        }
    }

    // middle and bottom portion of left hand side (W_H, W_P)
    for (int k = 0; k < vNum; k++) {
        A_tripList.push_back(Trip(vNum + k, k, W_H[k]));
        A_tripList.push_back(Trip(vNum * 2 + k, k, W_P[k]));
    }


    // build the left hand side sparse matrix A from triplets
    SparseMat A(vNum * 3, vNum);
    A.setFromTriplets(A_tripList.begin(), A_tripList.end());

    // make LHS a symmetric matrix A^2 (= A'A) to use cholesky solver
    SparseMat A_trans = A.transpose();
    SparseMat LHS = A_trans * A;


    // create solver that uses cholesky decomposition
    Eigen::SimplicialLLT<SparseMat> cholSolver;


    // left hand side of the symmetric linear system
    cholSolver.compute(LHS);

    if(cholSolver.info()!= Eigen::Success) {
        // decomposition failed
        std::cout << "\n Decomposition failed \n" << std::endl;
        return;
    }

    // right hand side of the equation
    Eigen::MatrixXd b = Eigen::MatrixXd::Zero(vNum * 3, 3);

    // b is [0; W_H * V; W_p * vPoles]
    for (int k = 0; k < vNum; k++) {
        b.row(k + vNum) = W_H[k] * V.row(k);
        b.row(k + vNum * 2) = W_P[k] * vPoles.row(k);
    }

    // RHS is A' * b
    Eigen::MatrixXd RHS = A_trans * b;


    // solving Ax = b
    V.col(0) = cholSolver.solve(RHS.col(0));
    V.col(1) = cholSolver.solve(RHS.col(1));
    V.col(2) = cholSolver.solve(RHS.col(2));

    if(cholSolver.info()!= Eigen::Success) {
        // solving failed
        std::cout << "\n Solving failed \n" << std::endl;
        return;
    }
}




// ===============================================================================================
// ==================== main function
int main(int argc, char *argv[])
{
    Eigen::MatrixXd V, V_bunny, V_bumpy, V_fertility;
    Eigen::MatrixXi F, F_bunny, F_bumpy, F_fertility;

    // Load the mesh
    igl::readOFF("/Users/jimmy/GoogleDrive/MScRobotics/GV18_AcquisitionProcessing3DGeometry/cw3/meshes/sindorelax.off", V_fertility, F_fertility);
    V = V_fertility;
    F = F_fertility;
    // print size
    std::cout << std::endl << "Mesh size: " << "V: " << V.rows() << " x " << V.cols() <<  ", F: " << F.rows() << " x " << F.cols() << std::endl << std::endl;


    // ---------- build the half edge data structure

    long vNum = V.rows();
    // vector of vertex pointers
    std::vector<HE_vert*> HE_vertList(vNum);

    // build half-edge structure from V and F
    createHalfEdge(V, F, HE_vertList);

    std::cout << "Half edge structure created" << std::endl << std::endl;
    // --------------------------------------------------

    // create viewer
    igl::viewer::Viewer viewer;
    // show original mesh
    viewer.data.set_mesh(V, F);


    // ---------- tasks

    // get the voronoi poles
    Eigen::MatrixXd vPoles = getVoronoiPoles(V, F);

    // initialise the laplace-beltrami operator
    SparseMat laplaceBeltrami(vNum, vNum);

    // boolean value for if a vertex position should be fixed
    std::vector<bool> fixedVerts(vNum, false);


    // parameters
    double w_L = 1;
    double w_H = 0.1;
    double w_P = 0.2;

    // initially all the same for every vertex
    std::vector<double> W_L(vNum, w_L);
    std::vector<double> W_H(vNum, w_H);
    std::vector<double> W_P(vNum, w_P);

    // edgeThresh is 0.002 * (bounding box diagonal length)
    Eigen::Vector3d allMax(V.col(0).maxCoeff(), V.col(1).maxCoeff(), V.col(2).maxCoeff());
    Eigen::Vector3d allMin(V.col(0).minCoeff(), V.col(1).minCoeff(), V.col(2).minCoeff());
    double edgeThresh = 0.002 * (allMax - allMin).norm();



    // ----- test viewer menu
    viewer.callback_init = [&](igl::viewer::Viewer &viewer) {
        // make window
        viewer.ngui->addWindow(Eigen::Vector2i(900, 10), "COMPGV18 Acquisition3D, Coursework 3 test");


        // ----- reload button
        viewer.ngui->addButton("Reload", [&]() {
            // reset mesh
            V = V_fertility;
            F = F_fertility;
            // reset parameters to initial condition
            std::fill(fixedVerts.begin(), fixedVerts.end(), false);
            updateParameters(fixedVerts, w_L, w_H, w_P, W_L, W_H, W_P);
            // show mesh
            displayMesh(V, F, viewer);
        });


        // ---------- section 1: test 1
        viewer.ngui->addGroup("Test 1");


        viewer.ngui->addButton("Plot voronoi poles", [&]() {
            // display
            displayMesh(vPoles, F, viewer);
        });

        // ----- add button: test
        viewer.ngui->addButton("Skeletonise", [&]() {
            // update laplaceBeltrami
            buildLaplaceBeltrami(HE_vertList, V, laplaceBeltrami);
            // contraction
            // discreteMeanCurvatureFlow(V, F, laplaceBeltrami, W_H);
            medialSkeletonisationFlow(V, F, laplaceBeltrami, vPoles, W_L, W_H, W_P);
            // test degeneracy
            testDegeneracy(HE_vertList, V, edgeThresh, fixedVerts);
            // update parameters
            updateParameters(fixedVerts, w_L, w_H, w_P, W_L, W_H, W_P);
            // display
            displayMesh(V, F, viewer);
        });

        // ----- text boxes
        viewer.ngui->addVariable("w_L: ", w_L);
        viewer.ngui->addVariable("w_H: ", w_H);
        viewer.ngui->addVariable("w_P: ", w_P);
        viewer.ngui->addVariable("edgeThresh: ", edgeThresh, false);

        // generate layout
        viewer.screen->performLayout();

        return false;
    };


    // launch viewer
    viewer.launch();
}
