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
#include <ranges>

// NOLINTBEGIN(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access, misc-use-anonymous-namespace)

// Test basic mesh functions
static auto
testMeshBasics(const interp::Mesh2DGrid& m2d,
    const size_t nx, const size_t ny, const double fac)
{
    // Check dimensions, convexity, and limits
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

    // Test containment and indexing
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
    const auto [iIdx,jIdx] = m2d.xyIdx(xIn, yTest);
    if (iIdx != 2 || jIdx != 1) {
        std::cerr << "testMesh2DGrid: for point (" << xIn 
            << ", " << yTest << "), expected (i,j) = (2,1), "
            "instead found (" << iIdx << ", " << jIdx << ")\n";
        return 1;
    }

    return 0; // Success
}


// Test ability to find intersections with mesh at fixed x
static auto
testXIntersect(const interp::Mesh2DGrid& m2d)
{
    // Shorten name
    using xInt = interp::Mesh2DGrid::xIntersectionDescriptor;
    using xIntType = interp::Mesh2DGrid::IntersectionType;

    // Test points and expected results
    std::vector<double> xTest = { -0.5, 0.05, 0.5, 3.05, 3.1 };
    std::vector<std::vector<xInt>> xIntersect(xTest.size());
    std::vector<std::vector<std::pair<double, double>>> yLim(xTest.size());
    xIntersect[0] = std::vector<xInt>();  // No intersections
    yLim[0] = std::vector<std::pair<double, double>>();
    xIntersect[1] = {
        { 0, 0.05, xIntType::rib, 0, false },
        { 0.5, std::sqrt(0.5*0.5 + 0.05*0.05), xIntType::spine, 0, true }
    };
    yLim[1] = { { 0, 0.5 } };
    xIntersect[2] = {
        { 0, 0.5, xIntType::rib, 0, false },
        { 1, 0.5, xIntType::rib, 1, false },
        { 2, 0.5, xIntType::rib, 2, true }
    };
    yLim[2] = { { 0, 2 }};
    xIntersect[3] = {
        { 0.5, std::sqrt(0.05*0.05 + 0.5*0.5), xIntType::spine, 3, false },
        { 1, 3.05, xIntType::rib, 1, false },
        { 2, 3.05, xIntType::rib, 2, true }
    };
    yLim[3] = { { 0.5, 2 }};
    xIntersect[4] = {
        { 1, std::sqrt(0.1*0.1 + 1*1), xIntType::spine, 3, false },
        { 2, 3.1, xIntType::rib, 2, true }
    };
    yLim[4] = { { 1, 2 } };

    // Check against expected results
    for (const auto& [x, xi, yl] : std::views::zip(xTest, xIntersect, yLim))
    {
        const auto yLimTest = m2d.yLim(x);
        const auto xIntersectTest = m2d.xIntersect(x);
        if (yl.size() != yLimTest.size())
        {
            std::cerr << "testMesh2DGrid: checked for y limits at x = "
                << x << "; expected to find " << yl.size()
                << ", instead got " << yLimTest.size() << "\n";
            return 1;
        }
        if (xi.size() != xIntersectTest.size())
        {
            std::cerr << "testMesh2DGrid: searched for intersections at "
                "x = " << x << ": expected to find " << xi.size()
                << " intersection points, got " << xIntersectTest.size() << "\n";
            return 1;
        }
        for (const auto& [r, lim] : std::views::zip(yLimTest, yl))
        {
            if (!utils::approxEqual(r.first, lim.first) || 
                !utils::approxEqual(r.second, lim.second))
            {
                std::cerr << "testMesh2DGrid: y limits not as expected at "
                    "x = " << x << ": expected " << lim.first << " - " << lim.second
                    << ", instead found " << r.first << " - " << r.second
                    << "\n";
                return 1;
            }
        }
        for (const auto& [r, intersect] : std::views::zip(xIntersectTest, xi))
        {
            if (!utils::approxEqual(r.y, intersect.y) ||
                !utils::approxEqual(r.xs, intersect.xs) ||
                r.t != intersect.t ||
                r.idx != intersect.idx ||
                r.meshExit != intersect.meshExit)
            {
                std::cerr << "testMesh2DGrid: intersection points not as "
                    "expected at x = " << x << ": expected (y, xs, t, idx, exit) = "
                    << intersect.y << ", " << intersect.xs << ", " << static_cast<int>(intersect.t) 
                    << ", " << intersect.idx << ", " << intersect.meshExit
                    << ", instead found "
                    << r.y << ", " << r.xs << ", " << static_cast<int>(r.t) 
                    << ", " << r.idx << ", " << r.meshExit
                    << "\n";
                return 1;
            }
        }
    }

    return 0;  // Success
}


auto testMesh2DGrid() -> int
{
    // Construct a simple convex mesh
    constexpr size_t nx = 4;
    constexpr size_t ny = 3;
    const double fac = 0.1;
    std::array<double,nx*ny> xData = { 0 };
    std::array<double,ny> yData = { 0 };
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

    // Do basic tests
    if (testMeshBasics(m2d, nx, ny, fac)) { return 1; };
    
    // Do intersection tests
    if (testXIntersect(m2d)) { return 1; };

    return 0; // Return success
}

// NOLINTEND(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access, misc-use-anonymous-namespace)
