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
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <mdspan>
#include <ranges>
#include <utility>
#include <vector>

// NOLINTBEGIN(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access, misc-use-anonymous-namespace)

// Test basic mesh functions
static auto
testMeshBasics(const interp::Mesh2DGrid& m2d,
    const interp::Mesh2DGrid& m2dNC,
    const size_t nx, const size_t ny, const double fac)
{
    // Check dimensions, convexity, and limits
    const auto nxm1 = static_cast<double>(nx - 1);
    const auto nym1 = static_cast<double>(ny - 1);
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
    if (!utils::approxEqual(m2d.xMax(), nxm1 + (fac * nym1))) {
        std::cerr << "testMesh2DGrid: expected xMax = " << nxm1 + (fac * nym1)
            << ", found xMax = " << m2d.xMax() << "\n";
        return 1;        
    }
    if (!utils::approxEqual(m2d.yMin(), 0)) {
        std::cerr << "testMesh2DGrid: expected yMin = " << 0.0
            << ", found yMin = " << m2d.yMin() << "\n";
        return 1;        
    }
    if (!utils::approxEqual(m2d.yMax(), nym1))
    {
        std::cerr << "testMesh2DGrid: expected yMax = " << ny-1
            << ", found yMax = " << m2d.yMax() << "\n";
        return 1;
    }
    const double yTest = 1.5;
    if (!utils::approxEqual(m2d.xMin(yTest), fac*yTest))
    {
        std::cerr << "testMesh2DGrid: expected xMin = " << fac*yTest
            << " at y = " << yTest << " for test mesh; found "
            << m2d.xMin(yTest) << "\n";
        return 1;
    }
    if (!utils::approxEqual(m2d.xMax(yTest), 
        static_cast<double>(nx)-1 + (fac*yTest)))
    {
        std::cerr << "testMesh2DGrid: expected xMax = " 
            << static_cast<double>(nx)-1 + (fac*yTest)
            << " at y = " << yTest << " for test mesh; found "
            << m2d.xMax(yTest) << "\n";
        return 1;
    }

    // Test containment and indexing
    const double xOut = 0.0;
    const double xIn = 2.5;
    if (m2d.contains(xOut, yTest))
    {
        std::cerr << "testMesh2DGrid: point (" << xOut 
            << ", " << yTest << ") incorrectly reported"
            " as inside the mesh\n";
        return 1;
    }
    if (!m2d.contains(xIn, yTest))
    {
        std::cerr << "testMesh2DGrid: point (" << xIn 
            << ", " << yTest << ") incorrectly reported"
            " as outside the mesh\n";
        return 1;
    }
    const auto [iIdx,jIdx] = m2d.xyIdx(xIn, yTest);
    if (iIdx != 2 || jIdx != 1)
    {
        std::cerr << "testMesh2DGrid: for point (" << xIn 
            << ", " << yTest << "), expected (i,j) = (2,1), "
            "instead found (" << iIdx << ", " << jIdx << ")\n";
        return 1;
    }

    // Test that non-convex mesh is recorded correctly as non-convex,
    // and that it gets the correct xMin and xMax assigned
    if (m2dNC.convex())
    {
        std::cerr << "testMesh2DGrid: non-convex grid reported as convex\n";
        return 1;
    }
    if (!utils::approxEqual(m2dNC.xMin(), 0.0))
    {
        std::cerr << "testMesh2DGrid: expected xMin = " << 0.0
            << ", found xMin = " << m2dNC.xMin() << "\n";
        return 1;
    }
    if (!utils::approxEqual(m2dNC.xMax(), 
        static_cast<double>(nx) - 1 + fac))
    {
        std::cerr << "testMesh2DGrid: expected xMax = " 
            << static_cast<double>(nx) - 1 + fac
            << ", found xMax = " << m2dNC.xMax() << "\n";
        return 1;        
    }


    return 0; // Success
}


// Test ability to find intersections with convex mesh at fixed x
static auto
testXIntersectConvex(const interp::Mesh2DGrid& m2d,
    const size_t nx, const double fac)
{
    // Shorten names
    using XInt = interp::Mesh2DGrid::xIntersectionDescriptor;
    using XIntType = interp::Mesh2DGrid::IntersectionType;

    // Test points and expected results
    const std::vector<double> xTest = { -0.5, 0.05, 0.5, 3.05, 3.1 };
    std::vector<std::vector<XInt>> xIntersect(xTest.size());
    std::vector<std::vector<std::pair<double, double>>> yLim(xTest.size());
    xIntersect[0] = std::vector<XInt>();  // No intersections
    yLim[0] = std::vector<std::pair<double, double>>();
    xIntersect[1] = {
        { .y = 0, .xs = xTest[1], .t = XIntType::rib, 
            .idx = 0, .meshExit = false },
        { .y = xTest[1]/fac, 
            .xs = std::sqrt(std::pow(xTest[1]/fac,2) + 
            std::pow(xTest[1],2)), 
            .t = XIntType::spine, .idx = 0, .meshExit = true }
    };
    yLim[1] = { { 0, 0.5 } };
    xIntersect[2] = {
        { .y = 0, .xs = xTest[2], .t = XIntType::rib, 
            .idx = 0, .meshExit = false },
        { .y = 1, .xs = xTest[2], .t = XIntType::rib,
            .idx = 1, .meshExit = false },
        { .y = 2, .xs = xTest[2], .t= XIntType::rib, 
            .idx = 2, .meshExit = true }
    };
    yLim[2] = { { 0, 2 }};
    const double nxm1 = static_cast<double>(nx) - 1;
    xIntersect[3] = {
        { .y = (xTest[3]-nxm1)/fac, 
            .xs = 
            std::sqrt(std::pow(xTest[3]-nxm1,2) + 
            std::pow((xTest[3]-nxm1)/fac,2)), 
            .t = XIntType::spine, .idx = 3, .meshExit = false },
        { .y = 1, .xs = xTest[3], .t = XIntType::rib, 
            .idx = 1, .meshExit = false },
        { .y = 2, .xs = xTest[3], .t = XIntType::rib, 
            .idx = 2, .meshExit = true }
    };
    yLim[3] = { { 0.5, 2 }};
    xIntersect[4] = {
        { .y = 1, .xs = std::sqrt(1 + std::pow(0.1,2)), 
            .t = XIntType::spine, .idx = 3, 
            .meshExit = false },
        { .y = 2, .xs = 3.1, .t = XIntType::rib, 
            .idx = 2, .meshExit = true }
    };
    yLim[4] = { { 1, 2 } };

    // Check against expected results
    for (const auto& [x, xi, yl] : std::views::zip(xTest, xIntersect, yLim))
    {
        const auto yLimTest = m2d.yLim(x);
        const auto xIntersectTest = m2d.xIntersect(x);
        if (yl.size() != yLimTest.size())
        {
            std::cerr << "testMesh2DGrid: checked convex mesh for y limits at x = "
                << x << "; expected to find " << yl.size()
                << ", instead got " << yLimTest.size() << "\n";
            return 1;
        }
        if (xi.size() != xIntersectTest.size())
        {
            std::cerr << "testMesh2DGrid: searched convex mesh for intersections at "
                "x = " << x << ": expected to find " << xi.size()
                << " intersection points, got " << xIntersectTest.size() << "\n";
            return 1;
        }
        for (const auto& [r, lim] : std::views::zip(yLimTest, yl))
        {
            if (!utils::approxEqual(r.first, lim.first) || 
                !utils::approxEqual(r.second, lim.second))
            {
                std::cerr << "testMesh2DGrid: convex mesh y limits not as expected at "
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
                std::cerr << "testMesh2DGrid: convex mesh intersection points not as "
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

// Test ability to find intersections with convex mesh at fixed x
static auto
testXIntersectNonConvex(const interp::Mesh2DGrid& m2dNC)
{
    // Shorten names
    using XInt = interp::Mesh2DGrid::xIntersectionDescriptor;
    using XIntType = interp::Mesh2DGrid::IntersectionType;

    // Do test for non-convex mesh, in locations where non-convexity matters
    std::vector<double> xTestNC = { 0.05, 3.05 };
    std::vector<std::vector<XInt>> xIntersectNC(xTestNC.size());
    std::vector<std::vector<std::pair<double, double>>> yLimNC(xTestNC.size());
    const double num = std::sqrt(std::pow(0.5,2) + std::pow(0.05,2)); // Convenience
    xIntersectNC[0] = {
        { .y = 0, .xs = 0.05, .t = XIntType::rib, 
            .idx = 0, .meshExit = false },
        { .y = 0.5, .xs = num, .t = XIntType::spine, 
            .idx = 0, .meshExit = true },
        { .y = 1.5,
            .xs = std::sqrt(1 + std::pow(0.1,2)) + num,
            .t = XIntType::spine, .idx = 0, .meshExit = false },
        { .y = 2, .xs = 0.05, .t = XIntType::rib, 
            .idx = 2, .meshExit = true }
    };
    yLimNC[0] = { { 0, 0.5 }, { 1.5, 2 } };
    xIntersectNC[1] = {
        { .y = 0.5, .xs = num, .t = XIntType::spine, 
            .idx = 3, .meshExit = false },
        { .y = 1, .xs = 3.05, .t = XIntType::rib, 
            .idx = 1, .meshExit = false },
        { .y = 1.5, .xs = std::sqrt(1 + std::pow(0.1,2)) + num,
            .t = XIntType::spine, .idx = 3, .meshExit = true }
    };
    yLimNC[1] = { { 0.5, 1.5 } };

    // Check against expected results
    for (const auto& [x, xi, yl] : std::views::zip(xTestNC, xIntersectNC, yLimNC))
    {
        const auto yLimTest = m2dNC.yLim(x);
        const auto xIntersectTest = m2dNC.xIntersect(x);
        if (yl.size() != yLimTest.size())
        {
            std::cerr << "testMesh2DGrid: checked non-convex mesh for y limits at x = "
                << x << "; expected to find " << yl.size()
                << ", instead got " << yLimTest.size() << "\n";
            return 1;
        }
        if (xi.size() != xIntersectTest.size())
        {
            std::cerr << "testMesh2DGrid: searched non-convex mesh for intersections at "
                "x = " << x << ": expected to find " << xi.size()
                << " intersection points, got " << xIntersectTest.size() << "\n";
            return 1;
        }
        for (const auto& [r, lim] : std::views::zip(yLimTest, yl))
        {
            if (!utils::approxEqual(r.first, lim.first) || 
                !utils::approxEqual(r.second, lim.second))
            {
                std::cerr << "testMesh2DGrid: non-convex mesh y limits not as expected at "
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
                std::cerr << "testMesh2DGrid: non-convex mesh intersection points not as "
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

// Test ability to find intersections with mesh at fixed y
static auto
testYIntersect(const interp::Mesh2DGrid& m2d,
    const size_t nx, const double fac)
{
    // Shorten name
    using YInt = interp::Mesh2DGrid::yIntersectionDescriptor;

    const std::vector<double> yTest = { -0.5, 0.5, 2 };
    std::vector<std::vector<YInt>> expected(yTest.size());
    for (size_t i = 1; i < yTest.size(); i++)
    {
        expected[i].resize(nx);
        for (size_t j = 0; j < nx; j++)
        {
            expected[i][j] = {
                .x = static_cast<double>(j) + (yTest[i] * fac),
                .s = yTest[i] * std::sqrt(1 + std::pow(fac,2)),
                .idx = j
            };
        }
    }

    // Check against expected results
    for (const auto& [y, ex] : std::views::zip(yTest, expected))
    {
        const auto yIntersectTest = m2d.yIntersect(y);
        if (ex.size() != yIntersectTest.size())
        {
            std::cerr << "testMesh2DGrid: searched convex mesh for intersections at "
                "y = " << y << ": expected to find " << ex.size()
                << " intersection points, got " << yIntersectTest.size() << "\n";
            return 1;
        }
        for (const auto& [r, e] : std::views::zip(yIntersectTest, ex))
        {
            if (!utils::approxEqual(r.x, e.x) ||
                !utils::approxEqual(r.s, e.s) ||
                r.idx != e.idx)
            {
                std::cerr << "testMesh2DGrid: convex mesh intersection points not as "
                    "expected at y = " << y 
                    << ": expected (x, s, idx) = "
                    << e.x << ", " << e.s << ", " 
                    << e.idx
                    << ", instead found "
                    << r.x << ", " << r.s << ", "
                    << r.idx << "\n";
                return 1;
            }
        }
    }

    return 0; // Success
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

    // Construct a non-convex mesh
    std::array<double,nx*ny> xDataNC = { 0 };
    std::array<double,ny> yDataNC = { 0 };
    const std::mdspan<double, std::extents<size_t, nx, ny>> 
        xNC(xDataNC.data());
    const std::mdspan<double, std::extents<size_t, ny>> 
        yNC(yDataNC.data());
    for (size_t j = 0; j < ny; ++j) {
        yNC[j] = static_cast<double>(j);
        for (size_t i = 0; i < nx; ++i) {
            if (j % 2 == 1)
            {
                xNC[i,j] = static_cast<double>(i) +
                    (fac * static_cast<double>(j));
            }
            else
            {
                xNC[i,j] = static_cast<double>(i);
            }
        }
    }
    const interp::Mesh2DGrid m2dNC(xNC, yNC);

    // Result accumulator
    int test = 0;

    // Do basic tests
    test += testMeshBasics(m2d, m2dNC, nx, ny, fac);

    // Do intersection tests
    test += testXIntersectConvex(m2d, nx, fac);
    test += testXIntersectNonConvex(m2dNC);
    test += testYIntersect(m2d, nx, fac);

    return test; // Return success
}

// NOLINTEND(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access, misc-use-anonymous-namespace)
