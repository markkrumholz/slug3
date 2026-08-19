/**
 * @file testMesh2DGrid.cpp
 * @author Mark Krumholz
 * @brief Implementation of the testMesh2DGrid function
 * @date 2024-06-19
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
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
#include <string>
#include <utility>
#include <vector>

// NOLINTBEGIN(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)

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

// Helper for testXIntersectN: check xIntersectN(x, y, n) against the
// corresponding window of the already-tested xIntersect(x) list, for
// every valid n and for both starting types of y -- exactly on an
// intersection point (rib or spine), and strictly interior to the mesh
// between two consecutive intersection points. The expected window is
// computed by simulating the same down-first-alternating search the
// real algorithm uses: each round, take one point below if available,
// then check completion, then take one point above if available.
//
// This deliberately does not compare meshExit: that flag is direction-
// dependent (it means "the search cannot continue past this point in
// the direction it was searching"), and xIntersect always searches
// upward from the bottom, so its meshExit values only reflect that one
// direction. xIntersectN can reach the very same point via a downward
// sub-search instead, where meshExit legitimately flips (e.g. the
// mesh's lower-coverage boundary for this x is not an exit when
// xIntersect starts there, but is a genuine exit when xIntersectN's
// downward search terminates there) without either being wrong.
// meshExit correctness is instead covered by the hand-verified
// testXIntersectNDownwardSpine regression test below.
static auto
checkXIntersectNAgainstFull(
    const interp::Mesh2DGrid& m,
    const double x,
    const std::string& label) -> int
{
    const auto full = m.xIntersect(x);
    if (full.size() < 2)
    {
        std::cerr << "checkXIntersectNAgainstFull: " << label
            << ": xIntersect(x = " << x << ") returned fewer than 2 "
            "points, which is too few to exercise xIntersectN\n";
        return 1;
    }

    const auto checkWindow = [&](
        const double y,
        const size_t nBelowAvail,
        const size_t nAboveAvail,
        const size_t startTotal,
        const size_t startIdx0) -> int
    {
        for (size_t n = 1; n <= full.size() + 2; ++n)
        {
            const auto got = m.xIntersectN(x, y, n);

            size_t nBelow = 0;
            size_t nAbove = 0;
            size_t total = startTotal;
            bool contDown = true;
            bool contUp = true;
            while (contDown || contUp)
            {
                if (total == n) { break; }
                if (contDown)
                {
                    if (nBelow < nBelowAvail) { nBelow++; total++; }
                    else { contDown = false; }
                }
                if (total == n) { break; }
                if (contUp)
                {
                    if (nAbove < nAboveAvail) { nAbove++; total++; }
                    else { contUp = false; }
                }
            }

            if (got.size() != total)
            {
                std::cerr << "checkXIntersectNAgainstFull: " << label
                    << ": x = " << x << ", y = " << y << ", n = " << n
                    << ": expected " << total << " points, got "
                    << got.size() << "\n";
                return 1;
            }
            const size_t windowStart = startIdx0 - nBelow;
            for (size_t idx = 0; idx < got.size(); ++idx)
            {
                const auto& e = full[windowStart + idx];
                const auto& r = got[idx];
                if (!utils::approxEqual(r.y, e.y) ||
                    !utils::approxEqual(r.xs, e.xs) ||
                    r.t != e.t || r.idx != e.idx)
                {
                    std::cerr << "checkXIntersectNAgainstFull: " << label
                        << ": x = " << x << ", y = " << y << ", n = " << n
                        << ": point " << idx << " expected (y, xs, t, idx) = "
                        << e.y << ", " << e.xs << ", "
                        << static_cast<int>(e.t) << ", " << e.idx
                        << ", instead found " << r.y << ", "
                        << r.xs << ", " << static_cast<int>(r.t) << ", "
                        << r.idx << "\n";
                    return 1;
                }
            }
        }
        return 0;
    };

    // On a non-convex mesh, xIntersect's full list can span multiple
    // disconnected segments (separated by a meshExit point where the
    // line leaves and later re-enters the mesh). xIntersectN performs
    // a single continuous search from its starting point and correctly
    // stops at such a boundary rather than jumping the gap to a
    // different segment. meshExit only ever marks the top of a segment
    // (an entry point is always meshExit = false), so the two
    // directions are not symmetric: walking up from startIdx, a
    // meshExit point is still the reachable top of the current
    // segment, so it counts before stopping; walking down, encountering
    // any meshExit point at a lower index means that point belongs to
    // an entirely different, disconnected segment, so it must be
    // excluded rather than counted. And if startIdx itself is already a
    // segment's top (meshExit = true), nothing above it is reachable at
    // all, regardless of what a later, disconnected segment contains.
    const auto countAvailBelow = [&](const size_t startIdx) -> size_t
    {
        size_t count = 0;
        for (size_t idx = startIdx; idx-- > 0; )
        {
            if (full[idx].meshExit) { break; }
            count++;
        }
        return count;
    };
    const auto countAvailAbove = [&](const size_t startIdx) -> size_t
    {
        if (full[startIdx].meshExit) { return 0; }
        size_t count = 0;
        for (size_t idx = startIdx + 1; idx < full.size(); ++idx)
        {
            count++;
            if (full[idx].meshExit) { break; }
        }
        return count;
    };

    int result = 0;

    // Starting exactly on each intersection point (rib and spine alike);
    // the starting point itself always counts, so startTotal = 1
    for (size_t k = 0; k < full.size(); ++k)
    {
        result += checkWindow(full[k].y, countAvailBelow(k), countAvailAbove(k), 1, k);
        if (result != 0) { return result; }
    }

    // Starting strictly interior to the mesh, at the midpoint between
    // each pair of consecutive intersection points; nothing is found at
    // the start, so startTotal = 0. Skip pairs straddling a segment
    // gap (full[k] itself a meshExit point): the midpoint there falls
    // outside the mesh entirely, which xIntersectN does not accept.
    for (size_t k = 0; k + 1 < full.size(); ++k)
    {
        if (full[k].meshExit) { continue; }
        const double y = 0.5 * (full[k].y + full[k+1].y);
        if (utils::approxEqual(y, full[k].y) ||
            utils::approxEqual(y, full[k+1].y)) { continue; }
        result += checkWindow(y, countAvailBelow(k + 1), countAvailAbove(k), 0, k + 1);
        if (result != 0) { return result; }
    }

    return result;
}

// Test xIntersectN's ability to find up to n intersection points of
// the mesh with a line of constant x, centered on a given y, by
// differential testing against the already-tested xIntersect
static auto
testXIntersectN(
    const interp::Mesh2DGrid& m2d,
    const interp::Mesh2DGrid& m2dNC) -> int
{
    int result = 0;
    result += checkXIntersectNAgainstFull(m2d, 0.05, "convex");
    result += checkXIntersectNAgainstFull(m2d, 3.05, "convex");
    result += checkXIntersectNAgainstFull(m2dNC, 0.05, "non-convex");
    result += checkXIntersectNAgainstFull(m2dNC, 3.05, "non-convex");
    return result;
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


// Test that querying yIntersect at a sequence of consecutive, exact
// grid y values reproduces each column's raw x data exactly. This
// exercises Mesh2DGrid's cached row-index search (yIdx): previously,
// when a query landed exactly on the upper edge of the currently
// cached row interval, yIdx failed to advance its cached index,
// leaving the row-offset dy computed as the full interval width
// instead of zero. Rather than reading the queried column directly,
// yIntersect would then trace a spurious path interpolated between
// the wrong pair of columns via floating-point division, introducing
// rounding error that (among other things) could violate the strict
// ordering required downstream by Interpolator1D. Querying the grid
// rows in increasing order, one row at a time, reproduces exactly the
// query pattern that surfaced this bug.
static auto
testYIntersectGridBoundary()
{
    constexpr size_t nx = 4;
    constexpr size_t ny = 4;
    std::array<double, nx*ny> xData = { 0 };
    std::array<double, ny> yData = { 0 };
    const std::mdspan<double, std::extents<size_t, nx, ny>> x(xData.data());
    const std::mdspan<double, std::extents<size_t, ny>> y(yData.data());
    for (size_t j = 0; j < ny; ++j) { y[j] = static_cast<double>(j); }

    // Give each column a distinct set of x values, so that resolving
    // a query against the wrong column produces detectably different
    // results from resolving it against the right one
    constexpr std::array<std::array<double, nx>, ny> xCol = {{
        { 0.0, 1.0, 2.0, 3.0 },
        { 0.0, 1.5, 2.5, 4.0 },
        { 0.0, 2.0, 3.0, 5.0 },
        { 0.0, 2.5, 4.0, 7.0 }
    }};
    for (size_t j = 0; j < ny; ++j) {
        for (size_t i = 0; i < nx; ++i) { x[i,j] = xCol[j][i]; } // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- i, j are loop indices bounded by constexpr nx/ny
    }
    const interp::Mesh2DGrid m2d(x, y);

    // Query each grid row in increasing order, one step at a time; a
    // query landing exactly on row j should reproduce that column's
    // raw x values exactly (no interpolation should occur)
    for (size_t j = 0; j < ny; ++j)
    {
        const auto result = m2d.yIntersect(static_cast<double>(j));
        if (result.size() != nx)
        {
            std::cerr << "testMesh2DGrid: querying yIntersect at grid "
                "row j = " << j << " returned " << result.size()
                << " points, expected " << nx << "\n";
            return 1;
        }
        for (size_t i = 0; i < nx; ++i)
        {
            if (result[i].x != xCol[j][i] || result[i].idx != i) // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- i, j are loop indices bounded by constexpr nx/ny
            {
                std::cerr << "testMesh2DGrid: querying yIntersect at "
                    "grid row j = " << j << ", point i = " << i
                    << ": expected x = " << xCol[j][i] << ", idx = " << i // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- i, j are loop indices bounded by constexpr nx/ny
                    << ", instead found x = " << result[i].x
                    << ", idx = " << result[i].idx << "\n";
                return 1;
            }
        }
    }

    return 0; // Success
}

// Test the x-direction analog of testYIntersectGridBoundary: querying
// xIdx at a sequence of consecutive, exact grid x values along a
// single row should return each point's own index, rather than
// leaving the cached index stuck one step behind (the same class of
// bug as in yIdx, fixed in the same way)
static auto
testXIdxGridBoundary()
{
    constexpr size_t nx = 5;
    constexpr std::array<double, nx> xRow = { 0.0, 1.0, 2.5, 4.0, 8.0 };
    std::array<double, nx*2> xData = { 0 };
    std::array<double, 2> yData = { 0.0, 1.0 };
    const std::mdspan<double, std::extents<size_t, nx, 2>> x(xData.data());
    const std::mdspan<double, std::extents<size_t, 2>> y(yData.data());
    for (size_t i = 0; i < nx; ++i) {
        x[i,0] = xRow[i]; // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- i is a loop index bounded by constexpr nx
        x[i,1] = xRow[i] + 0.5; // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- Second row just needs to be valid; i is a loop index bounded by constexpr nx
    }
    const interp::Mesh2DGrid m2d(x, y);

    // Query each grid point along row 0 in increasing order; each
    // query should resolve to its own index, except the last point,
    // which resolves to nx - 2 (the start of the final cell), matching
    // the same top-of-range convention used by yIdx
    for (size_t k = 0; k < nx; ++k)
    {
        const auto i = m2d.xIdx(xRow[k], 0); // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- k is a loop index bounded by constexpr nx
        const auto expected = (k == nx - 1) ? nx - 2 : k;
        if (i != expected)
        {
            std::cerr << "testMesh2DGrid: querying xIdx at grid point "
                "k = " << k << " (x = " << xRow[k] << ") returned i = " // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- k is a loop index bounded by constexpr nx
                << i << ", expected " << expected << "\n";
            return 1;
        }
    }

    return 0; // Success
}

// Regression test for a bug in findIntersectDistance, found while
// implementing xIntersectN: the distance-to-rib calculation always
// measured against the cell's top edge (y_[jSaveLoc+1]), which is
// correct for upward searches but wrong for downward ones, where it
// should measure against the cell's bottom edge (y_[jSaveLoc])
// instead. This never surfaced before xIntersectN existed, because
// every other caller of this traversal machinery (xIntersect) only
// ever searches upward, from yMin_ to yMax_. This mesh has a single
// cell whose right spine runs from (1, 0) to (3, 2) while its left
// edge stays pinned at x = 0; querying x = 1.5 places the start point
// outside the cell at the bottom (y = 0, where the cell only spans
// x in [0, 1]) but inside it higher up (the right edge reaches x =
// 1.5 at y = 0.5). A downward search from an interior y in (0.5, 2)
// must therefore hit the right spine at y = 0.5 before it would ever
// reach the bottom rib at y = 0; the buggy version instead jumped
// straight past the spine to the bottom rib.
static auto
testXIntersectNDownwardSpine()
{
    using XInt = interp::Mesh2DGrid::xIntersectionDescriptor;
    using XIntType = interp::Mesh2DGrid::IntersectionType;

    constexpr size_t nx = 2;
    constexpr size_t ny = 2;
    std::array<double, nx*ny> xData = { 0 };
    std::array<double, ny> yData = { 0.0, 2.0 };
    const std::mdspan<double, std::extents<size_t, nx, ny>> x(xData.data());
    const std::mdspan<double, std::extents<size_t, ny>> y(yData.data());
    x[0,0] = 0.0; x[0,1] = 0.0; // Left edge: vertical at x = 0
    x[1,0] = 1.0; x[1,1] = 3.0; // Right edge: from (1, 0) to (3, 2)
    const interp::Mesh2DGrid m2d(x, y);

    constexpr double xQuery = 1.5;
    constexpr double ySpine = 0.5; // Where the right spine crosses xQuery
    const double xsSpine = std::sqrt(std::pow(xQuery - 1.0, 2) +
        std::pow(ySpine, 2)); // s along the right spine at (xQuery, ySpine)

    const std::vector<XInt> expected = {
        { .y = ySpine, .xs = xsSpine, .t = XIntType::spine,
            .idx = 1, .meshExit = true },
        { .y = 2.0, .xs = xQuery, .t = XIntType::rib,
            .idx = 1, .meshExit = true }
    };

    // A downward-only search (n = 1) from an interior y above the
    // spine crossing must find the spine, not the bottom rib
    for (const double yQuery : { 0.6, 1.2, 1.9 })
    {
        const auto got1 = m2d.xIntersectN(xQuery, yQuery, 1);
        if (got1.size() != 1 ||
            !utils::approxEqual(got1[0].y, expected[0].y) ||
            !utils::approxEqual(got1[0].xs, expected[0].xs) ||
            got1[0].t != expected[0].t || got1[0].idx != expected[0].idx ||
            got1[0].meshExit != expected[0].meshExit)
        {
            std::cerr << "testMesh2DGrid: xIntersectN(x = " << xQuery
                << ", y = " << yQuery << ", n = 1) expected to find the "
                "right spine crossing at y = " << expected[0].y
                << ", instead got " << got1.size() << " points"
                << (got1.empty() ? "" :
                    (std::string(", first at y = ") +
                    std::to_string(got1[0].y))) << "\n";
            return 1;
        }

        // With n = 2, the search should also find the top rib above
        const auto got2 = m2d.xIntersectN(xQuery, yQuery, 2);
        if (got2.size() != 2)
        {
            std::cerr << "testMesh2DGrid: xIntersectN(x = " << xQuery
                << ", y = " << yQuery << ", n = 2) expected 2 points, "
                "got " << got2.size() << "\n";
            return 1;
        }
        for (const auto& [r, e] : std::views::zip(got2, expected))
        {
            if (!utils::approxEqual(r.y, e.y) || !utils::approxEqual(r.xs, e.xs) ||
                r.t != e.t || r.idx != e.idx || r.meshExit != e.meshExit)
            {
                std::cerr << "testMesh2DGrid: xIntersectN(x = " << xQuery
                    << ", y = " << yQuery << ", n = 2): expected (y, xs, t, "
                    "idx, exit) = " << e.y << ", " << e.xs << ", "
                    << static_cast<int>(e.t) << ", " << e.idx << ", "
                    << e.meshExit << ", instead found " << r.y << ", "
                    << r.xs << ", " << static_cast<int>(r.t) << ", "
                    << r.idx << ", " << r.meshExit << "\n";
                return 1;
            }
        }

        // n = 3 asks for more points than exist along this line; only
        // the 2 that actually exist should come back
        const auto got3 = m2d.xIntersectN(xQuery, yQuery, 3);
        if (got3.size() != 2)
        {
            std::cerr << "testMesh2DGrid: xIntersectN(x = " << xQuery
                << ", y = " << yQuery << ", n = 3) expected only the 2 "
                "available points, got " << got3.size() << "\n";
            return 1;
        }
    }

    return 0; // Success
}

// Regression test for a bug in xIntersectN's "on rib" corner-case
// handling, found while hooking xIntersectN up to real stellar track
// data. When a search starts exactly on a rib, at a column i whose
// index exactly matches the query x (i.e. x == x_[i,jSaveLoc]), the
// code needs to suppress that column's own spine recheck in whichever
// direction it's searching, since that spine trivially starts at the
// exact point being searched from and would otherwise be
// re-"discovered" at zero distance. The existing code only did this
// for a positive-sloped spine (decrementing into the next column and
// setting the corresponding flag); for a spine sloping <= 0 at that
// same column, neither flag was set, so the zero-distance spine won
// the nearest-edge comparison and was reported as if the search had
// jumped straight to the mass at its far end -- in the real-data case
// that exposed this, a search from a live, present-day mass up to the
// very next (much higher, much shorter-lived) mass in the grid
// incorrectly reported that higher mass as still alive at a time long
// after it had actually died, instead of the true intersection with
// the mesh's edge somewhere in between. This mesh reproduces the same
// shape as that real case in miniature: two masses (5 and 20), where
// mass 5's "time" column keeps increasing (0, 2, 4, 6) while mass 20's
// stops early and pads by repeating its final value (0, 1, 1, 1),
// giving the column-2 spine connecting them a negative slope
// (15 / (1 - 4) = -5) at the exact query point (x = 4, y = 5).
static auto
testXIntersectNRibCornerNegativeSlope()
{
    using XInt = interp::Mesh2DGrid::xIntersectionDescriptor;
    using XIntType = interp::Mesh2DGrid::IntersectionType;

    constexpr size_t nx = 4;
    constexpr size_t ny = 2;
    std::array<double, nx*ny> xData = { 0 };
    std::array<double, ny> yData = { 5.0, 20.0 };
    const std::mdspan<double, std::extents<size_t, nx, ny>> x(xData.data());
    const std::mdspan<double, std::extents<size_t, ny>> y(yData.data());
    for (size_t i = 0; i < nx; ++i) { x[i,0] = static_cast<double>(i) * 2.0; }
    for (size_t i = 0; i < nx; ++i)
    {
        x[i,1] = static_cast<double>(std::min(i, size_t{1}));
    }
    const interp::Mesh2DGrid m2d(x, y);

    const auto got = m2d.xIntersectN(4.0, 5.0, 2);

    const std::vector<XInt> expected = {
        { .y = 5.0, .xs = 4.0, .t = XIntType::rib,
            .idx = 0, .meshExit = false },
        { .y = 11.0, .xs = std::sqrt(40.0), .t = XIntType::spine,
            .idx = 3, .meshExit = true }
    };

    if (got.size() != expected.size())
    {
        std::cerr << "testMesh2DGrid: xIntersectN(4.0, 5.0, 2) expected "
            << expected.size() << " points, got " << got.size() << "\n";
        return 1;
    }
    for (const auto& [r, e] : std::views::zip(got, expected))
    {
        if (!utils::approxEqual(r.y, e.y) || !utils::approxEqual(r.xs, e.xs) ||
            r.t != e.t || r.idx != e.idx || r.meshExit != e.meshExit)
        {
            std::cerr << "testMesh2DGrid: xIntersectN(4.0, 5.0, 2): "
                "expected (y, xs, t, idx, exit) = " << e.y << ", " << e.xs
                << ", " << static_cast<int>(e.t) << ", " << e.idx << ", "
                << e.meshExit << ", instead found " << r.y << ", " << r.xs
                << ", " << static_cast<int>(r.t) << ", " << r.idx << ", "
                << r.meshExit << "\n";
            return 1;
        }
    }

    return 0; // Success
}

// Regression test for yIntersect at the exact yMin_/yMax_ boundary.
// yIdx's cell-index clamp (needed so it always returns a valid
// cell-start index) forces a nonzero mass offset from the boundary
// column, unlike every interior exact grid mass (where the offset is
// naturally zero); combined with a degenerate (collapsed) boundary
// cell -- reproduced here by giving the last two columns the same
// value at the last row -- this used to corrupt the traversal into
// producing non-monotonic results (see getTrack(mMax()) in
// testTracksAll.cpp for the practical regression this fixes). This
// checks that yIntersect(yMin_) and yIntersect(yMax_) instead
// reproduce the boundary column's own x data exactly.
static auto
testYIntersectBoundary()
{
    constexpr size_t nx = 3;
    constexpr size_t ny = 4;
    std::array<double, nx*ny> xData = { 0 };
    std::array<double, ny> yData = { 0 };
    const std::mdspan<double, std::extents<size_t, nx, ny>> x(xData.data());
    const std::mdspan<double, std::extents<size_t, ny>> y(yData.data());
    for (size_t j = 0; j < ny; ++j) { y[j] = static_cast<double>(j); }

    // Column ny-2 and column ny-1 share the same value at the last
    // row, making that boundary cell degenerate
    constexpr std::array<std::array<double, nx>, ny> xCol = {{
        { -10.0, -5.0, 1.0 },
        { -9.0,  -4.0, 2.0 },
        { -8.0,  -3.0, 3.0 },
        { -7.0,  -2.0, 3.0 }
    }};
    for (size_t j = 0; j < ny; ++j) {
        for (size_t i = 0; i < nx; ++i) { x[i,j] = xCol[j][i]; } // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- i, j are loop indices bounded by constexpr nx/ny
    }
    const interp::Mesh2DGrid m2d(x, y);

    for (const size_t jBoundary : { size_t{0}, ny - 1 })
    {
        const auto result = m2d.yIntersect(static_cast<double>(jBoundary));
        if (result.size() != nx)
        {
            std::cerr << "testMesh2DGrid: yIntersect at boundary row j = "
                << jBoundary << " returned " << result.size()
                << " points, expected " << nx << "\n";
            return 1;
        }
        for (size_t i = 0; i < nx; ++i)
        {
            if (result[i].x != xCol[jBoundary][i] || result[i].idx != i) // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- jBoundary, i are bounded by constexpr ny/nx
            {
                std::cerr << "testMesh2DGrid: yIntersect at boundary row "
                    "j = " << jBoundary << ", point i = " << i
                    << ": expected x = " << xCol[jBoundary][i] << ", idx = " << i // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- jBoundary, i are bounded by constexpr ny/nx
                    << ", instead found x = " << result[i].x
                    << ", idx = " << result[i].idx << "\n";
                return 1;
            }
        }
    }

    return 0; // Success
}

// Regression test for xIntersect at the exact xMin_/xMax_ boundary,
// including the meshExit rule for columns whose boundary-row value
// coincides with xMin_/xMax_: a column's point is marked meshExit =
// false (does not end a segment) only if the next column with a
// distinct y value (skipping any that tie its own y value, as can
// happen with concatenated, overlapping mass ranges) also lies at
// xMin_/xMax_, and meshExit = true otherwise. This mesh exercises all
// three shapes at once: an isolated tangent point (column 0, alone at
// xMax_), a contiguous multi-column run ending at the true edge
// (columns 0-1, both at xMin_), and a mass tie that must be skipped
// when checking for continuation (columns 2-3 share a y value, but
// only column 2 is at xMax_; checking for continuation past column 2
// must look past column 3 to column 4, not stop at column 3).
static auto
testXIntersectBoundary()
{
    constexpr size_t nx = 3;
    constexpr size_t ny = 6;
    std::array<double, nx*ny> xData = { 0 };
    std::array<double, ny> yData = { 0.0, 1.0, 2.0, 2.0, 3.0, 4.0 };
    const std::mdspan<double, std::extents<size_t, nx, ny>> x(xData.data());
    const std::mdspan<double, std::extents<size_t, ny>> y(yData.data());

    constexpr std::array<std::array<double, nx>, ny> xCol = {{
        { -10.0, -5.0, 5.0 }, // touches xMin (-10, tied w/ col 1) and xMax (5)
        { -10.0, -4.0, 3.0 }, // touches xMin (-10, tied w/ col 0); not xMax
        { -8.0,  -3.0, 5.0 }, // touches xMax (5); mass tied with col 3
        { -7.0,  -2.0, 3.0 }, // mass tied with col 2; touches neither bound
        { -6.0,  -1.0, 5.0 }, // touches xMax (5)
        { -5.0,   0.0, 2.0 }  // touches neither bound
    }};
    for (size_t j = 0; j < ny; ++j) {
        for (size_t i = 0; i < nx; ++i) { x[i,j] = xCol[j][i]; } // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- i, j are loop indices bounded by constexpr nx/ny
    }
    const interp::Mesh2DGrid m2d(x, y);

    using XInt = interp::Mesh2DGrid::xIntersectionDescriptor;
    using XIntType = interp::Mesh2DGrid::IntersectionType;

    const auto checkBoundary = [&](
        const double xBoundary,
        const std::vector<XInt>& expected) -> int
    {
        const auto result = m2d.xIntersect(xBoundary);
        if (result.size() != expected.size())
        {
            std::cerr << "testMesh2DGrid: xIntersect at x = " << xBoundary
                << " returned " << result.size() << " points, expected "
                << expected.size() << "\n";
            return 1;
        }
        for (const auto& [r, e] : std::views::zip(result, expected))
        {
            if (!utils::approxEqual(r.y, e.y) || !utils::approxEqual(r.xs, e.xs) ||
                r.t != e.t || r.idx != e.idx || r.meshExit != e.meshExit)
            {
                std::cerr << "testMesh2DGrid: xIntersect at x = " << xBoundary
                    << ": expected (y, xs, t, idx, exit) = " << e.y << ", "
                    << e.xs << ", " << static_cast<int>(e.t) << ", " << e.idx
                    << ", " << e.meshExit << ", instead found " << r.y << ", "
                    << r.xs << ", " << static_cast<int>(r.t) << ", " << r.idx
                    << ", " << r.meshExit << "\n";
                return 1;
            }
        }
        return 0;
    };

    int result = 0;

    // xMax_ = 5: an isolated tangent point (column 0), then a
    // contiguous pair (columns 2, 4) whose continuation check must
    // skip past the mass tie at column 3
    result += checkBoundary(5.0, {
        { .y = 0.0, .xs = 5.0, .t = XIntType::rib, .idx = 0, .meshExit = true },
        { .y = 2.0, .xs = 5.0, .t = XIntType::rib, .idx = 2, .meshExit = false },
        { .y = 3.0, .xs = 5.0, .t = XIntType::rib, .idx = 4, .meshExit = true }
    });

    // xMin_ = -10: a contiguous run spanning columns 0-1
    result += checkBoundary(-10.0, {
        { .y = 0.0, .xs = -10.0, .t = XIntType::rib, .idx = 0, .meshExit = false },
        { .y = 1.0, .xs = -10.0, .t = XIntType::rib, .idx = 1, .meshExit = true }
    });

    return result;
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
    test += testXIntersectN(m2d, m2dNC);
    test += testYIntersect(m2d, nx, fac);

    // Regression tests for the cached-index boundary bug in yIdx/xIdx
    test += testYIntersectGridBoundary();
    test += testXIdxGridBoundary();

    // Regression test for the downward-search distance bug in
    // findIntersectDistance, found while implementing xIntersectN
    test += testXIntersectNDownwardSpine();
    test += testXIntersectNRibCornerNegativeSlope();

    // Regression tests for yIntersect/xIntersect at the exact
    // yMin_/yMax_ and xMin_/xMax_ boundaries
    test += testYIntersectBoundary();
    test += testXIntersectBoundary();

    return test; // Return success
}

// NOLINTEND(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
