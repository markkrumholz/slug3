/**
 * @file testMesh3DInterpolator.cpp
 * @author Mark Krumholz
 * @brief Implementation of the testMesh3DInterpolator function
 * @date 2026-07-03
 */

#include "../src/interpolation/Interpolator1D.hpp"
#include "../src/interpolation/Mesh2DInterpolator.hpp"
#include "../src/interpolation/Mesh3DInterpolator.hpp"
#include "../src/utils/MiscUtils.hpp"
#include "testMesh3DInterpolator.hpp"
#include <array>
#include <cstddef>
#include <gsl/gsl_interp.h>
#include <iostream>
#include <mdspan>
#include <memory>
#include <ranges>
#include <stdexcept>
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

    return 0; // Success
}

// NOLINTEND(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
