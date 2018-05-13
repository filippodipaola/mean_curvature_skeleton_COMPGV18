#IFNDEF CREATEHALFEDGE_H
#DEFINE CREATEHALFEDGE_H

#include "HEstruct.h"



// ========== function to create the half edge data structure from V and F
void createHalfEdge(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, std::vector<HE_vert*> &vertList);


#ENDIF
