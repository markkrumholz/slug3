/**
 * @file testMesh2DGrid.cpp
 * @author Mark Krumholz
 * @brief Implementation of the testMesh2DGrid function
 * @date 2024-06-19
 */

#include "../src/interpolation/Mesh2DGrid.hpp"
#include "../src/utils/MiscUtils.hpp"
#include "testMesh2DGrid.hpp"
#include <array>
#include <cstdlib>
#include <iostream>
#include <mdspan>

// NOLINTBEGIN(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)

auto testMesh2DGrid() -> int
{
    // Construct a simple convex mesh
    constexpr size_t nx = 4;
    constexpr size_t ny = 3;
    std::array<double,nx*ny> xData = { 0 };
    std::array<double,ny> yData = { 0 };
    const double fac = 0.1;
    const std::mdspan<double, std::extents<size_t, nx, ny>> 
        x(xData.data());
    const std::mdspan<double, std::extents<size_t, ny>> 
        y(yData.data());
    for (size_t j = 0; j < ny; ++j) {
        y[j] = static_cast<double>(j);
        for (size_t i = 0; i < nx; ++i) {
            x[i,j] = static_cast<double>(i) +
                (fac * static_cast<double>(j));
        }
    }
    const interp::Mesh2DGrid m2d(x, y);

    // Check basics: dimensions, convexity, limits
    if (m2d.nx() != nx) {
        std::cerr << "testMesh2DGrid: expected nx = " << nx
            << ", got nx = " << m2d.nx() << "\n";
        return 1;
    }
    if (m2d.ny() != ny) {
        std::cerr << "testMesh2DGrid: expected ny = " << ny
            << ", got ny = " << m2d.ny() << "\n";
        return 1;
    }
    if (!m2d.convex()) {
        std::cerr << "testMesh2DGrid: convex grid reported as non-convex\n";
        return 1;
    }
    if (!utils::approxEqual(m2d.xMin(), 0.0)) {
        std::cerr << "testMesh2DGrid: expected xMin = " << 0.0
            << ", found xMin = " << m2d.xMin() << "\n";
        return 1;
    }
    if (!utils::approxEqual(m2d.xMax(), nx - 1 + (fac * (ny-1)))) {
        std::cerr << "testMesh2DGrid: expected xMax = " << nx - 1 + (fac * (ny-1))
            << ", found xMax = " << m2d.xMax() << "\n";
        return 1;        
    }
    if (!utils::approxEqual(m2d.yMin(), 0)) {
        std::cerr << "testMesh2DGrid: expected yMin = " << 0.0
            << ", found yMin = " << m2d.yMin() << "\n";
        return 1;        
    }
    if (!utils::approxEqual(m2d.yMax(), ny-1))
    {
        std::cerr << "testMesh2DGrid: expected yMax = " << ny-1
            << ", found yMax = " << m2d.yMax() << "\n";
        return 1;
    }
    const double yTest = 1.5;
    if (!utils::approxEqual(m2d.xMin(yTest), fac*yTest)) {
        std::cerr << "testMesh2DGrid: expected xMin = " << fac*yTest
            << " at y = " << yTest << " for test mesh; found "
            << m2d.xMin(yTest) << "\n";
        return 1;
    }
    if (!utils::approxEqual(m2d.xMax(yTest), nx-1 + (fac*yTest))) {
        std::cerr << "testMesh2DGrid: expected xMax = " << nx-1 + (fac*yTest)
            << " at y = " << yTest << " for test mesh; found "
            << m2d.xMax(yTest) << "\n";
        return 1;
    }
    const double xOut = 0.0;
    const double xIn = 2.5;
    if (m2d.contains(xOut, yTest)) {
        std::cerr << "testMesh2DGrid: point (" << xOut 
            << ", " << yTest << ") incorrectly reported"
            " as inside the mesh\n";
        return 1;
    }
    if (!m2d.contains(xIn, yTest)) {
        std::cerr << "testMesh2DGrid: point (" << xIn 
            << ", " << yTest << ") incorrectly reported"
            " as outside the mesh\n";
        return 1;
    }
    return 0; // Passed
}

// NOLINTEND(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
