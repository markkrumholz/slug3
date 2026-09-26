/**
 * @file testMesh3DInterpolator.cpp
 * @author Mark Krumholz
 * @brief Implementation of the testMesh3DInterpolator function
 * @date 2026-07-03
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#include "../src/interpolation/Interpolator1D.hpp"
#include "../src/interpolation/Mesh2DInterpolator.hpp"
#include "../src/interpolation/Mesh3DInterpolator.hpp"
#include "../src/utils/MiscUtils.hpp"
#include "testMesh3DInterpolator.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <gsl/gsl_interp.h>
#include <iostream>
#include <mdspan>
#include <memory>
#include <ranges>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// NOLINTBEGIN(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)

// Number of fields in test
constexpr size_t nF = 2;

// Standard test mesh dimensions shared by every test below
constexpr size_t nx = 3;
constexpr size_t ny = 4;
constexpr size_t nz = 5;

// xTest = 0.777 lies strictly between the i=0 and i=1 mesh columns for
// every slice tested below (all grid x values are multiples of 0.01,
// so this choice can never land exactly on a mesh point), so each
// slice tested produces exactly one segment spanning the full range of
// the slice's other coordinate
constexpr double xTest = 0.777;
static const std::vector<double> zQuery = { 0.0, 1.3, 2.0, 4.0 }; // NOLINT(bugprone-throwing-static-initialization,cert-err58-cpp) -- built from a fixed literal list, so allocation failure aside (which would abort regardless), this can't actually throw
static const std::vector<double> yQuery = { 0.0, 0.7, 2.0, 3.0 }; // NOLINT(bugprone-throwing-static-initialization,cert-err58-cpp) -- built from a fixed literal list, so allocation failure aside (which would abort regardless), this can't actually throw

// The test mesh is filled with an affine function of the physical
// (x, y, z) coordinates, so linear interpolation (gsl_interp_linear)
// reproduces it exactly no matter where in the mesh it is evaluated
static auto
fExpected(double x, double y, double z) -> std::array<double, nF>
{
    return { x + y + z, (2.0 * x) - y + (0.5 * z) };
}

// Check that interpConstX on a 2D slice produces a single segment
// spanning [loExpected, hiExpected], and that evaluating it at a set of
// query points along that segment reproduces fExpected. fixedIsY
// selects whether fixedCoord plays the role of y (slice at constant y,
// so the segment interpolates over z) or z (slice at constant z, so the
// segment interpolates over y).
static auto
testSliceXInterp(const interp::Mesh2DInterpolator<nF>& slice,
    double xTest,
    double fixedCoord,
    bool fixedIsY,
    const std::vector<double>& queryPts,
    double loExpected,
    double hiExpected) -> int
{
    auto xInterp = slice.interpConstX(xTest);

    if (xInterp.size() != 1)
    {
        std::cerr << "testMesh3DInterpolator: at x = " << xTest
            << " expected 1 segment, found " << xInterp.size() << "\n";
        return 1;
    }

    const auto [lo, hi] = xInterp[0]->xRange();
    if (!utils::approxEqual(lo, loExpected) || !utils::approxEqual(hi, hiExpected))
    {
        std::cerr << "testMesh3DInterpolator: at x = " << xTest
            << " expected segment limits " << loExpected << "-" << hiExpected
            << ", instead found " << lo << "-" << hi << "\n";
        return 1;
    }

    for (const auto& q : queryPts)
    {
        const auto fval = (*xInterp[0])(q);
        const auto expected = fixedIsY ?
            fExpected(xTest, fixedCoord, q) :
            fExpected(xTest, q, fixedCoord);
        for (auto [fi, fe] : std::views::zip(fval, expected))
        {
            if (!utils::approxEqual(fi, fe))
            {
                std::cerr << "testMesh3DInterpolator: at x = " << xTest
                    << ", query = " << q << ", expected f = " << fe
                    << ", found " << fi << "\n";
                return 1;
            }
        }
    }
    return 0; // Success
}

// Test a y-degenerate mesh (ny = 1): sliceConstY should work
// trivially (no interpolation needed, since the mesh already lies
// entirely at the single y value), while sliceConstZ and
// sliceConstZCopy should throw, since there is no way to build a
// valid (x, y) slice when the y axis has only one point
static auto
testYDegenerateMesh() -> int
{
    constexpr size_t ny1 = 1;
    std::array<double, nx*ny1*nz> xDataYDeg = { 0 };
    std::array<double, ny1> yDataYDeg = { 0 };
    std::array<double, nz> zDataYDeg = { 0 };
    std::array<double, nx*ny1*nz*nF> fDataYDeg = { 0 };
    const std::mdspan<double, std::extents<size_t, nx, ny1, nz>>
        xYDeg(xDataYDeg.data());
    const std::mdspan<double, std::extents<size_t, ny1>>
        yYDeg(yDataYDeg.data());
    const std::mdspan<double, std::extents<size_t, nz>>
        zYDeg(zDataYDeg.data());
    const std::mdspan<double, std::extents<size_t, nx, ny1, nz, nF>>
        fYDeg(fDataYDeg.data());
    yYDeg[0] = 7.0;
    for (size_t k = 0; k < nz; ++k) { zYDeg[k] = static_cast<double>(k); }
    for (size_t i = 0; i < nx; ++i) {
        for (size_t k = 0; k < nz; ++k) {
            xYDeg[i,0,k] = static_cast<double>(i) +
                (0.01 * static_cast<double>(k));
            const auto fv = fExpected(xYDeg[i,0,k], yYDeg[0], zYDeg[k]);
            for (size_t n = 0; n < nF; ++n) { fYDeg[i,0,k,n] = fv[n]; } //NOLINT(cppcoreguidelines-pro-bounds-constant-array-index)
        }
    }
    const interp::Mesh3DInterpolator<nF> m3dYDeg(
        xYDeg, yYDeg, zYDeg, fYDeg, gsl_interp_linear);

    const auto& sliceYDeg = m3dYDeg.sliceConstY(7.0);
    if (testSliceXInterp(sliceYDeg, xTest, 7.0, true, zQuery, 0.0, 4.0) == 1)
    {
        return 1;
    }
    const auto sliceYDegCopy = m3dYDeg.sliceConstYCopy(7.0);
    if (testSliceXInterp(sliceYDegCopy, xTest, 7.0, true, zQuery, 0.0, 4.0) == 1)
    {
        return 1;
    }

    try
    {
        const auto& badSlice = m3dYDeg.sliceConstZ(2.0);
        (void)badSlice;
        std::cerr << "testMesh3DInterpolator: sliceConstZ on a "
            "y-degenerate mesh should have thrown, but did not\n";
        return 1;
    }
    catch (const std::runtime_error&) { /* this is the expected outcome */ } // NOLINT(bugprone-empty-catch) -- verifying that this call throws is the entire point of this test
    try
    {
        auto badSlice = m3dYDeg.sliceConstZCopy(2.0);
        (void)badSlice;
        std::cerr << "testMesh3DInterpolator: sliceConstZCopy on a "
            "y-degenerate mesh should have thrown, but did not\n";
        return 1;
    }
    catch (const std::runtime_error&) { /* this is the expected outcome */ } // NOLINT(bugprone-empty-catch) -- verifying that this call throws is the entire point of this test

    return 0; // Success
}

// Test a z-degenerate mesh (nz = 1): the mirror image of the
// y-degenerate case above
static auto
testZDegenerateMesh() -> int
{
    constexpr size_t nz1 = 1;
    std::array<double, nx*ny*nz1> xDataZDeg = { 0 };
    std::array<double, ny> yDataZDeg = { 0 };
    std::array<double, nz1> zDataZDeg = { 0 };
    std::array<double, nx*ny*nz1*nF> fDataZDeg = { 0 };
    const std::mdspan<double, std::extents<size_t, nx, ny, nz1>>
        xZDeg(xDataZDeg.data());
    const std::mdspan<double, std::extents<size_t, ny>>
        yZDeg(yDataZDeg.data());
    const std::mdspan<double, std::extents<size_t, nz1>>
        zZDeg(zDataZDeg.data());
    const std::mdspan<double, std::extents<size_t, nx, ny, nz1, nF>>
        fZDeg(fDataZDeg.data());
    zZDeg[0] = 9.0;
    for (size_t j = 0; j < ny; ++j) { yZDeg[j] = static_cast<double>(j); }
    for (size_t i = 0; i < nx; ++i) {
        for (size_t j = 0; j < ny; ++j) {
            xZDeg[i,j,0] = static_cast<double>(i) +
                (0.1 * static_cast<double>(j));
            const auto fv = fExpected(xZDeg[i,j,0], yZDeg[j], zZDeg[0]);
            for (size_t n = 0; n < nF; ++n) { fZDeg[i,j,0,n] = fv[n]; } //NOLINT(cppcoreguidelines-pro-bounds-constant-array-index)
        }
    }
    const interp::Mesh3DInterpolator<nF> m3dZDeg(
        xZDeg, yZDeg, zZDeg, fZDeg, gsl_interp_linear);

    const auto& sliceZDeg = m3dZDeg.sliceConstZ(9.0);
    if (testSliceXInterp(sliceZDeg, xTest, 9.0, false, yQuery, 0.0, 3.0) == 1)
    {
        return 1;
    }
    const auto sliceZDegCopy = m3dZDeg.sliceConstZCopy(9.0);
    if (testSliceXInterp(sliceZDegCopy, xTest, 9.0, false, yQuery, 0.0, 3.0) == 1)
    {
        return 1;
    }

    try
    {
        const auto& badSlice = m3dZDeg.sliceConstY(1.0);
        (void)badSlice;
        std::cerr << "testMesh3DInterpolator: sliceConstY on a "
            "z-degenerate mesh should have thrown, but did not\n";
        return 1;
    }
    catch (const std::runtime_error&) { /* this is the expected outcome */ } // NOLINT(bugprone-empty-catch) -- verifying that this call throws is the entire point of this test
    try
    {
        auto badSlice = m3dZDeg.sliceConstYCopy(1.0);
        (void)badSlice;
        std::cerr << "testMesh3DInterpolator: sliceConstYCopy on a "
            "z-degenerate mesh should have thrown, but did not\n";
        return 1;
    }
    catch (const std::runtime_error&) { /* this is the expected outcome */ } // NOLINT(bugprone-empty-catch) -- verifying that this call throws is the entire point of this test

    return 0; // Success
}

// Test that a mesh with both ny = 1 and nz = 1 is rejected outright
static auto
testFullyDegenerateMeshRejected() -> int
{
    constexpr size_t ny1 = 1;
    constexpr size_t nz1 = 1;
    std::array<double, nx*ny1*nz1> xDataBoth = { 0 };
    std::array<double, ny1> yDataBoth = { 0 };
    std::array<double, nz1> zDataBoth = { 0 };
    std::array<double, nx*ny1*nz1*nF> fDataBoth = { 0 };
    const std::mdspan<double, std::extents<size_t, nx, ny1, nz1>>
        xBoth(xDataBoth.data());
    const std::mdspan<double, std::extents<size_t, ny1>>
        yBoth(yDataBoth.data());
    const std::mdspan<double, std::extents<size_t, nz1>>
        zBoth(zDataBoth.data());
    const std::mdspan<double, std::extents<size_t, nx, ny1, nz1, nF>>
        fBoth(fDataBoth.data());

    try
    {
        const interp::Mesh3DInterpolator<nF> m3dBoth(
            xBoth, yBoth, zBoth, fBoth, gsl_interp_linear);
        (void)m3dBoth;
        std::cerr << "testMesh3DInterpolator: constructing a mesh "
            "with ny = 1 and nz = 1 should have thrown, but did not\n";
        return 1;
    }
    catch (const std::runtime_error&) { /* this is the expected outcome */ } // NOLINT(bugprone-empty-catch) -- verifying that this call throws is the entire point of this test

    return 0; // Success
}

// Relative agreement required between lazily- and eagerly-evaluated
// slices: the two apply the same arithmetic, differing only in that the
// lazy slice's local rib/spine interpolators may round differently
static auto closeEnough(const double a, const double b) -> bool
{
    constexpr double tol = 1e-12;
    return std::abs(a - b) <= tol * std::max({ 1.0, std::abs(a), std::abs(b) });
}

// Compare two Interpolator1D objects (or null pointers) over their
// common range at a set of evenly spaced points; returns the number of
// mismatches found, printing each with label
template <size_t NF>
static auto compareInterp1D(const interp::Interpolator1D<NF>* lazy,
    const interp::Interpolator1D<NF>* eager, const std::string& label) -> int
{
    if ((lazy == nullptr) != (eager == nullptr))
    {
        std::cerr << "testMesh3DInterpolator: lazySlice: " << label
            << ": one slice returned null and the other did not\n";
        return 1;
    }
    if (lazy == nullptr) { return 0; }
    if (!closeEnough(lazy->xMin(), eager->xMin()) || !closeEnough(lazy->xMax(), eager->xMax()))
    {
        std::cerr << "testMesh3DInterpolator: lazySlice: " << label
            << ": ranges differ: [" << lazy->xMin() << ", " << lazy->xMax() << "] vs ["
            << eager->xMin() << ", " << eager->xMax() << "]\n";
        return 1;
    }
    constexpr int nPt = 17;
    int nBad = 0;
    for (int p = 0; p <= nPt; ++p)
    {
        // Clamped to each interpolator's own range: q itself can land
        // just outside even eager's range by rounding, and the two
        // ranges (having been checked to agree above) may still differ
        // by rounding
        const double q = eager->xMin() + ((eager->xMax() - eager->xMin()) * p / nPt);
        const auto a = (*lazy)(std::clamp(q, lazy->xMin(), lazy->xMax()));
        const auto b = (*eager)(std::clamp(q, eager->xMin(), eager->xMax()));
        for (size_t n = 0; n < NF; ++n)
        {
            if (!closeEnough(a[n], b[n])) // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index)
            {
                std::cerr << "testMesh3DInterpolator: lazySlice: " << label << " at " << q
                    << ", field " << n << ": lazy " << a[n] << " vs eager " << b[n] << "\n"; // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index)
                ++nBad;
            }
        }
    }
    return nBad;
}

// Build the mesh used by the lazy-slice tests: large enough for
// steffen interpolation's local stencils to matter (15 x 8 x 4), with
// curved x coordinates and non-linear function values, so that
// agreement between lazy and eager slices is a real test of the lazy
// slice's own local rib/spine interpolators rather than of
// interpolating an affine function exactly. If dupRun is true, the x
// coordinates at i = 5 through 8 are all set equal to that at i = 5 (a
// run of duplicates along every rib, which Interpolator1D collapses),
// with f left distinct.
static auto makeLazyTestMesh(const bool dupRun,
    const gsl_interp_type* interpType = gsl_interp_steffen) -> interp::Mesh3DInterpolator<nF>
{
    constexpr size_t nxL = 15;
    constexpr size_t nyL = 8;
    constexpr size_t nzL = 4;
    std::vector<double> xData(nxL * nyL * nzL);
    std::vector<double> yData(nyL);
    std::vector<double> zData(nzL);
    std::vector<double> fData(nxL * nyL * nzL * nF);
    const std::mdspan<double, std::dextents<size_t, 3>> x(xData.data(), nxL, nyL, nzL);
    const std::mdspan<double, std::dextents<size_t, 1>> y(yData.data(), nyL);
    const std::mdspan<double, std::dextents<size_t, 1>> z(zData.data(), nzL);
    const std::mdspan<double, std::dextents<size_t, 4>> f(fData.data(), nxL, nyL, nzL, nF);
    for (size_t j = 0; j < nyL; ++j) { y[j] = 1.0 + (0.5 * static_cast<double>(j * j)); }
    for (size_t k = 0; k < nzL; ++k) { z[k] = -1.0 + (0.7 * static_cast<double>(k)); }
    for (size_t i = 0; i < nxL; ++i) {
        for (size_t j = 0; j < nyL; ++j) {
            for (size_t k = 0; k < nzL; ++k) {
                const auto di = static_cast<double>(i);
                // Each row non-decreasing in i, with a start and end
                // that vary with both j and k, so the mesh edges are
                // curved
                x[i,j,k] = (0.4 * di) + (0.03 * di * di / (1.0 + y[j])) +
                    (0.1 * z[k] * std::sqrt(di)) - (0.05 * y[j]);
                f[i,j,k,0] = std::sin(x[i,j,k]) + (0.3 * y[j] * y[j]) + std::cos(z[k]);
                f[i,j,k,1] = (std::exp(-0.2 * x[i,j,k]) * y[j]) + (z[k] * z[k]);
            }
        }
    }
    if (dupRun)
    {
        for (size_t i = 6; i <= 8; ++i) {
            for (size_t j = 0; j < nyL; ++j) {
                for (size_t k = 0; k < nzL; ++k) { x[i,j,k] = x[5,j,k]; }
            }
        }
    }
    return { x, y, z, f, interpType };
}

// Compare lazySliceConstZ() against sliceConstZCopy() of the same mesh
// for every kind of query a Tracks3D makes of a slice (see
// testLazySliceMatchesEager()'s own comment), at z values between grid
// planes and exactly on them; lazyMesh and eagerMesh must hold the same
// data (they differ only in testLazySliceSurvivesMove()). Returns the
// number of mismatches.
static auto compareLazyEager(const interp::Mesh3DInterpolator<nF>& lazyMesh,
    const interp::Mesh3DInterpolator<nF>& eagerMesh) -> int
{
    const auto z = eagerMesh.z();
    const size_t nzL = z.extent(0);
    int nBad = 0;
    for (const double z0 : { -0.83, -0.3, 0.4, 0.95, 1.05, z[0], z[2], z[nzL-1] })
    {
        const auto eager = eagerMesh.sliceConstZCopy(z0);
        const auto lazy = lazyMesh.lazySliceConstZ(z0);
        const std::string zLabel = "z = " + std::to_string(z0);

        if (!closeEnough(lazy.xMin(), eager.xMin()) || !closeEnough(lazy.xMax(), eager.xMax()) ||
            lazy.convex() != eager.convex())
        {
            std::cerr << "testMesh3DInterpolator: lazySlice: " << zLabel
                << ": mesh extent or convexity differs\n";
            ++nBad;
        }

        // Tracks (constant y), and edge positions at that y
        for (const double y0 : { 1.0, 1.3, 2.9, 7.7, 13.5, 25.0 })
        {
            const auto lt = lazy.interpConstY(y0);
            const auto et = eager.interpConstY(y0);
            nBad += compareInterp1D<nF>(lt.get(), et.get(),
                zLabel + ", interpConstY(" + std::to_string(y0) + ")");
            if (!closeEnough(lazy.xMax(y0), eager.xMax(y0)) ||
                !closeEnough(lazy.xMin(y0), eager.xMin(y0)))
            {
                std::cerr << "testMesh3DInterpolator: lazySlice: " << zLabel
                    << ": xMin/xMax(" << y0 << ") differs\n";
                ++nBad;
            }
        }

        // Isochrones (constant x), and the y range and edge slopes there
        const double xLo = eager.xMin();
        const double xHi = eager.xMax();
        for (int p = 1; p < 12; ++p)
        {
            const double x0 = xLo + ((xHi - xLo) * p / 12.0);
            const auto ls = lazy.interpConstX(x0);
            const auto es = eager.interpConstX(x0);
            if (ls.size() != es.size())
            {
                std::cerr << "testMesh3DInterpolator: lazySlice: " << zLabel
                    << ": interpConstX(" << x0 << ") gave " << ls.size() << " segments vs "
                    << es.size() << "\n";
                ++nBad;
                continue;
            }
            for (size_t sgm = 0; sgm < ls.size(); ++sgm)
            {
                nBad += compareInterp1D<nF>(ls[sgm].get(), es[sgm].get(),
                    zLabel + ", interpConstX(" + std::to_string(x0) + ")");
            }
            const auto lLim = lazy.yLim(x0);
            const auto eLim = eager.yLim(x0);
            const auto lEdge = lazy.yEdgeSlope(x0, false);
            const auto eEdge = eager.yEdgeSlope(x0, false);
            bool limOk = lLim.size() == eLim.size() && lEdge.size() == eEdge.size();
            for (size_t r = 0; limOk && r < lLim.size(); ++r)
            {
                limOk = closeEnough(lLim[r].first, eLim[r].first) &&
                    closeEnough(lLim[r].second, eLim[r].second);
            }
            for (size_t r = 0; limOk && r < lEdge.size(); ++r)
            {
                limOk = closeEnough(lEdge[r].first, eEdge[r].first) &&
                    closeEnough(lEdge[r].second, eEdge[r].second);
            }
            if (!limOk)
            {
                std::cerr << "testMesh3DInterpolator: lazySlice: " << zLabel
                    << ": yLim/yEdgeSlope(" << x0 << ") differ\n";
                ++nBad;
            }

            // Single-point evaluation (and containment) at a few y
            // values along this isochrone
            for (const double y0 : { 1.2, 3.3, 9.0, 20.0 })
            {
                if (lazy.contains(x0, y0) != eager.contains(x0, y0))
                {
                    std::cerr << "testMesh3DInterpolator: lazySlice: " << zLabel
                        << ": contains(" << x0 << ", " << y0 << ") differs\n";
                    ++nBad;
                    continue;
                }
                if (!eager.contains(x0, y0)) { continue; }
                const auto lv = lazy(x0, y0);
                const auto ev = eager(x0, y0);
                for (size_t n = 0; n < nF; ++n)
                {
                    if (!closeEnough(lv[n], ev[n])) // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index)
                    {
                        std::cerr << "testMesh3DInterpolator: lazySlice: " << zLabel
                            << ": (" << x0 << ", " << y0 << ") field " << n << ": lazy "
                            << lv[n] << " vs eager " << ev[n] << "\n"; // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index)
                        ++nBad;
                    }
                }
            }
        }
    }
    return nBad;
}

// Verify that Mesh3DInterpolator::lazySliceConstZ() agrees with
// sliceConstZCopy() for every kind of query a Tracks3D makes of a
// slice -- interpConstY (a track), interpConstX (an isochrone),
// operator() (a single star), xMax(y), yLim(x), yEdgeSlope(x) and
// contains(x, y) -- both on a mesh with distinct coordinates along
// every rib and on one with a run of duplicate coordinates, which the
// lazy slice's local stencils must widen past to see the same
// deduplicated points as the eager slice's full rib interpolators.
static auto testLazySliceMatchesEager() -> int
{
    int result = 0;
    for (const bool dupRun : { false, true })
    {
        const auto m3d = makeLazyTestMesh(dupRun);
        const int nBad = compareLazyEager(m3d, m3d);
        if (nBad > 0)
        {
            std::cerr << "testMesh3DInterpolator: lazySlice: " << nBad << " mismatches"
                << (dupRun ? " with duplicate rib coordinates\n" : "\n");
            result = 1;
        }
    }
    return result;
}

// Verify that a lazy slice stays valid after the Mesh3DInterpolator it
// was taken from is moved: its queries must still agree with eager
// slices of the moved-to object.
static auto testLazySliceSurvivesMove() -> int
{
    auto m3d = makeLazyTestMesh(false);
    const auto lazy = m3d.lazySliceConstZ(0.4);
    const interp::Mesh3DInterpolator<nF> moved(std::move(m3d));
    const auto eager = moved.sliceConstZCopy(0.4);
    int nBad = 0;
    for (const double y0 : { 1.3, 7.7, 25.0 })
    {
        nBad += compareInterp1D<nF>(lazy.interpConstY(y0).get(), eager.interpConstY(y0).get(),
            "after move, interpConstY(" + std::to_string(y0) + ")");
    }
    const double x0 = 0.5 * (eager.xMin() + eager.xMax());
    const auto ls = lazy.interpConstX(x0);
    const auto es = eager.interpConstX(x0);
    if (ls.size() != es.size()) { ++nBad; }
    for (size_t sgm = 0; sgm < std::min(ls.size(), es.size()); ++sgm)
    {
        nBad += compareInterp1D<nF>(ls[sgm].get(), es[sgm].get(), "after move, interpConstX");
    }
    if (nBad > 0)
    {
        std::cerr << "testMesh3DInterpolator: lazySliceSurvivesMove: " << nBad << " mismatches\n";
        return 1;
    }
    return 0;
}

// Verify that lazySliceConstZ() rejects a mesh built with a global
// interpolation type (cspline), for which a local stencil cannot
// reproduce the full rib/spine interpolant.
static auto testLazySliceRejectsGlobalInterp() -> int
{
    const auto m3d = makeLazyTestMesh(false, gsl_interp_cspline);
    try
    {
        [[maybe_unused]] const auto lazy = m3d.lazySliceConstZ(0.4);
    }
    catch (const std::runtime_error&)
    {
        return 0;
    }
    std::cerr << "testMesh3DInterpolator: lazySliceRejectsGlobalInterp: expected a "
        "cspline mesh's lazy slice to throw\n";
    return 1;
}

auto testMesh3DInterpolator() -> int
{
    // Construct a test non-square 3D mesh of size (3, 4, 5) where the
    // data points are
    // y_j = j
    // z_k = k
    // x_ijk = i + (0.1 * j) + (0.01 * k)
    // f_ijk = { x_ijk + y_j + z_k, (2 * x_ijk) - y_j + (0.5 * z_k) }
    std::array<double, nx*ny*nz> xData = { 0 };
    std::array<double, ny> yData = { 0 };
    std::array<double, nz> zData = { 0 };
    std::array<double, nx*ny*nz*nF> fData = { 0 };
    const std::mdspan<double, std::extents<size_t, nx, ny, nz>>
        x(xData.data());
    const std::mdspan<double, std::extents<size_t, ny>>
        y(yData.data());
    const std::mdspan<double, std::extents<size_t, nz>>
        z(zData.data());
    const std::mdspan<double, std::extents<size_t, nx, ny, nz, nF>>
        f(fData.data());
    for (size_t j = 0; j < ny; ++j) { y[j] = static_cast<double>(j); }
    for (size_t k = 0; k < nz; ++k) { z[k] = static_cast<double>(k); }
    for (size_t i = 0; i < nx; ++i) {
        for (size_t j = 0; j < ny; ++j) {
            for (size_t k = 0; k < nz; ++k) {
                x[i,j,k] = static_cast<double>(i) +
                    (0.1 * static_cast<double>(j)) +
                    (0.01 * static_cast<double>(k));
                const auto fv = fExpected(x[i,j,k], y[j], z[k]);
                for (size_t n = 0; n < nF; ++n) { f[i,j,k,n] = fv[n]; } //NOLINT(cppcoreguidelines-pro-bounds-constant-array-index)
            }
        }
    }

    // Construct interpolator
    const interp::Mesh3DInterpolator<nF> m3d(x, y, z, f, gsl_interp_linear);

    // Test slices at constant y: an exact grid match at each end of the
    // mesh, and a value that falls strictly between two grid points
    for (const auto& y0 : { 0.0, 1.7, 3.0 })
    {
        const auto& slice = m3d.sliceConstY(y0);
        if (testSliceXInterp(slice, xTest, y0, true, zQuery, 0.0, 4.0) == 1)
        {
            return 1;
        }
    }

    // Test slices at constant z: an exact grid match at each end of the
    // mesh, and a value that falls strictly between two grid points
    for (const auto& z0 : { 0.0, 2.3, 4.0 })
    {
        const auto& slice = m3d.sliceConstZ(z0);
        if (testSliceXInterp(slice, xTest, z0, false, yQuery, 0.0, 3.0) == 1)
        {
            return 1;
        }
    }

    // Test that values of x outside the mesh range produce no segments
    const auto& sliceOut = m3d.sliceConstY(1.0);
    if (!sliceOut.interpConstX(-1.0).empty() || !sliceOut.interpConstX(3.0).empty())
    {
        std::cerr << "testMesh3DInterpolator: expected no segments for "
            "x outside mesh range\n";
        return 1;
    }

    // Test a y-degenerate mesh (ny = 1), a z-degenerate mesh (nz = 1),
    // and that a mesh with both ny = 1 and nz = 1 is rejected outright
    if (testYDegenerateMesh() == 1) { return 1; }
    if (testZDegenerateMesh() == 1) { return 1; }
    if (testFullyDegenerateMeshRejected() == 1) { return 1; }
    if (testLazySliceMatchesEager() == 1) { return 1; }
    if (testLazySliceSurvivesMove() == 1) { return 1; }
    if (testLazySliceRejectsGlobalInterp() == 1) { return 1; }

    return 0; // Success
}

// NOLINTEND(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
