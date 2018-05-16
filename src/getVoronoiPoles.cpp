/*
 * @authors Filippo Di Paola, Jimmy Xu
 * 
 * This function takes the two matrix V and F from the loading of a mesh file and calcuates the 
 * Voronoi poles of the mesh using the Qhull library. Code has been used from https://github.com/ataiya/starlab-mcfskel 
 * specifically from the file "QhullVoronoiHelper.h" in the "surfacemesh_filter_voronoi" folder. 
 *
 * @params  V : An Eigen MatrixXd type matrix containing the coordinates of the vertices of our mesh
 *          F : An Eigen MatrixXi type matrix containing the indexs of the vertices of the faces of the mesh
 *
 * @returns : an Eigen MatrixXd type matrix contain the vertices coordinates of the Voronoi poles of the 
 *            mesh inputted into the function. Calculated using the Qhull library.
 */

// Standard Library Imports.
#include <iostream>
#include <float.h>
// LibIGL Library Imports.
#include <igl/readOFF.h>
#include <igl/viewer/Viewer.h>
#include <igl/readPLY.h>
#include <igl/writeOFF.h>
#include <igl/per_vertex_normals.h>
// Eigen Library Imports.
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/Sparse>
#include <Eigen/Core>
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

Eigen::MatrixXd getVoronoiPoles(Eigen::MatrixXd V, Eigen::MatrixXi F)
{
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
