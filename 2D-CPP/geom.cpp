#include <gmsh.h>

#include <iostream>
#include <set>

using namespace std;

vector<double> assembleNodeMap(vector<size_t> nodeTags,
                               vector<double> nodeCoords) {
  // Initialize the size of the vector
  vector<double> nodeMapVec(nodeTags.size() * 4);

  printf("\nNode Tag | [x, y, z]\n");
  for (int i = 0; i < nodeTags.size(); i++) {
    double x_i = nodeCoords[i * 3];
    double y_i = nodeCoords[i * 3 + 1];
    double z_i = nodeCoords[i * 3 + 2];
    printf("Node [%i] : [%f, %f, %f]\n", i, x_i, y_i, z_i);

    // Updated the vector that will be returned
    nodeMapVec[4 * i] = i;        // Node tag
    nodeMapVec[4 * i + 1] = x_i;  // x coord
    nodeMapVec[4 * i + 2] = y_i;  // y coord
    nodeMapVec[4 * i + 3] = z_i;  // z coord
  }
  return nodeMapVec;
}

vector<int> assembleConnectivityVector(vector<size_t> elemTags,
                                       vector<size_t> elemNodeTags) {
  vector<int> connecVec(elemTags.size() * 4);  // init size of connecVec

  printf("\nElement Tag : GMSH Tag | n1 | n2 | n3\n");
  for (int i = 0; i < elemTags.size(); i++) {
    // Loop through each element tag
    int gmsh_tag = elemTags[i];
    int n1 = elemNodeTags[i * 3];      // node 1
    int n2 = elemNodeTags[i * 3 + 1];  // node 2
    int n3 = elemNodeTags[i * 3 + 2];  // node 3

    // Compute area of element. If the area is negative swap 1st and 3rd nodes

    // Updated the connecVec
    connecVec[4 * i] = gmsh_tag;
    connecVec[4 * i + 1] = n1;
    connecVec[4 * i + 2] = n2;
    connecVec[4 * i + 3] = n3;

    printf("Element [%i] : [%i] [%i, %i, %i]\n", i, gmsh_tag, n1, n2, n3);
  }

  return connecVec;
}

int main(int argc, char **argv) {
  bool flag = false;  // Define the flag for visualizing the mesh;

  gmsh::initialize();                              // initialize gmsh
  gmsh::option::setNumber("General.Terminal", 0);  // 0 = No print, 1 = Print
  gmsh::model::add("model");                       // Specify the model

  // Draw the points that define the boundary
  double lc = 9e-1;
  gmsh::model::geo::addPoint(0.0, 0.0, 0.0, lc, 1);
  gmsh::model::geo::addPoint(0.8, 0.0, 0.0, lc, 2);
  gmsh::model::geo::addPoint(0.8, 0.8, 0.0, lc, 3);
  gmsh::model::geo::addPoint(0.0, 0.8, 0, lc, 4);

  // Draw the lines
  gmsh::model::geo::addLine(1, 2, 1);
  gmsh::model::geo::addLine(2, 3, 2);
  gmsh::model::geo::addLine(3, 4, 3);
  gmsh::model::geo::addLine(4, 1, 4);

  // Define the curve loop for the model boundary
  gmsh::model::geo::addCurveLoop({1, 2, 3, 4}, 1, true);

  // Define the planar surface
  gmsh::model::geo::addPlaneSurface({1}, 1);

  // Call sychronize prior to meshing
  gmsh::model::geo::synchronize();

  // Generate 2D mesh
  gmsh::model::mesh::generate(2);
  if (flag == true) {
    set<string> args(argv, argv + argc);
    if (!args.count("-nopopup")) gmsh::fltk::run();
  }

  // Save the mesh
  gmsh::write("model.msh");

  // Extract the nodes for the mesh
  vector<size_t> nodeTags;
  vector<double> nodeCoords, nodeParams;
  gmsh::model::mesh::getNodes(nodeTags, nodeCoords, nodeParams, -1, -1);
  printf("Total Number of Nodes = %lu\n", nodeTags.size());

  // Get the types of elements in the mesh
  vector<int> elementTypes;
  gmsh::model::mesh::getElementTypes(elementTypes);

  // Interested in extracting the triangle element type
  int elementType = elementTypes[1];

  // Extract the elemTags and elemNodesTags for linear triangular element
  vector<size_t> elemTags, elemNodeTags;
  gmsh::model::mesh::getElementsByType(elementType, elemTags, elemNodeTags);
  printf("Total Number of Elements = %lu\n", elemTags.size());

  // Vector that maps node to xyz coordinates
  vector<double> nodeMapVec;
  nodeMapVec = assembleNodeMap(nodeTags, nodeCoords);
  // for (auto it : nodeMapVec) cout << it << " ";

  // Create the connectivity vector
  vector<int> connecVec;
  connecVec = assembleConnectivityVector(elemTags, elemNodeTags);
  // for (auto it : connecVec) cout << it << " ";

  // Must finalize the model at the end of the call
  gmsh::finalize();
  return 0;
}
