/**
 * @file testInterpolationAll.cpp
 * @author Mark Krumholz
 * @brief Unit tests for the interpolation classes.
 * @details
 * This file runs unit tests for all the classes in interpolation.
 * @date 2024-06-19
 */

#include "testMesh2DGrid.hpp"

auto main() -> int {
    int result = 0;
    result += testMesh2DGrid();
    return result;
}