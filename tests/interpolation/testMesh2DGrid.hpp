/**
 * @file testMesh2DGrid.hpp
 * @author Mark Krumholz
 * @brief Unit tests for the Mesh2DGrid class
 * @details
 * This file contains unit tests for the Mesh2DGrid
 * class.
 * @date 2024-06-19
 */

#ifndef TESTMESH2DGRID_HPP
#define TESTMESH2DGRID_HPP

/**
 * @brief Unit test for the Mesh2DGrid class.
 * @return 0 if the test passes, 1 if it fails.
 * @details
 * This function tests the Mesh2DGrid class. The specific
 * tests carried out are: (1) correct construction of a
 * mesh, including correct calculation of its slopes, spine
 * lengths, and convexity; (2) correct calculation of whether
 * a point is within the mesh, and identification of its
 * host cell if it is.
 */
auto testMesh2DGrid() -> int;

#endif // TESTMESH2DGRID_HPP