/**
 * @file testInterpolationAll.cpp
 * @author Mark Krumholz
 * @brief Unit tests for the interpolation classes.
 * @details
 * This file runs unit tests for all the classes in interpolation.
 * @date 2024-06-19
 */

#include "testInterpolator1D.hpp"
#include "testMesh2DGrid.hpp"
#include "testMesh2DInterpolator.hpp"
#include "testMesh3DInterpolator.hpp"
#include <exception>
#include <iostream>

auto main() -> int {
    try
    {
        int result = 0;
        result += testInterpolator1D();
        result += testMesh2DGrid();
        result += testMesh2DInterpolator();
        result += testMesh3DInterpolator();
        return result;
    }
    catch (const std::exception& error)
    {
        std::cerr << "testInterpolationAll: uncaught exception: "
            << error.what() << "\n";
        return 1;
    }
}
