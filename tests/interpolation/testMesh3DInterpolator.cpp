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
#include <utility>
#include <vector>

// NOLINTBEGIN(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access, misc-use-anonymous-namespace)

// Number of fields in test
constexpr size_t nF = 2;

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

    for (double q : queryPts)
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

auto testMesh3DInterpolator() -> int
{
    // Construct a test non-square 3D mesh of size (3, 4, 5) where the
    // data points are
    // y_j = j
    // z_k = k
    // x_ijk = i + (0.1 * j) + (0.01 * k)
    // f_ijk = { x_ijk + y_j + z_k, (2 * x_ijk) - y_j + (0.5 * z_k) }
    constexpr size_t nx = 3;
    constexpr size_t ny = 4;
    constexpr size_t nz = 5;
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
                for (size_t n = 0; n < nF; ++n) { f[i,j,k,n] = fv[n]; }
            }
        }
    }

    // Construct interpolator
    const interp::Mesh3DInterpolator<nF> m3d(x, y, z, f, gsl_interp_linear);

    // xTest = 0.777 lies strictly between the i=0 and i=1 mesh columns
    // for every slice tested below (all grid x values are multiples of
    // 0.01, so this choice can never land exactly on a mesh point), so
    // each slice below produces exactly one segment spanning the full
    // range of the slice's other coordinate
    const double xTest = 0.777;
    const std::vector<double> zQuery = { 0.0, 1.3, 2.0, 4.0 };
    const std::vector<double> yQuery = { 0.0, 0.7, 2.0, 3.0 };

    // Test slices at constant y: an exact grid match at each end of the
    // mesh, and a value that falls strictly between two grid points
    for (double y0 : { 0.0, 1.7, 3.0 })
    {
        auto slice = m3d.sliceConstY(y0);
        if (testSliceXInterp(slice, xTest, y0, true, zQuery, 0.0, 4.0) == 1)
        {
            return 1;
        }
    }

    // Test slices at constant z: an exact grid match at each end of the
    // mesh, and a value that falls strictly between two grid points
    for (double z0 : { 0.0, 2.3, 4.0 })
    {
        auto slice = m3d.sliceConstZ(z0);
        if (testSliceXInterp(slice, xTest, z0, false, yQuery, 0.0, 3.0) == 1)
        {
            return 1;
        }
    }

    // Test that values of x outside the mesh range produce no segments
    auto sliceOut = m3d.sliceConstY(1.0);
    if (!sliceOut.interpConstX(-1.0).empty() || !sliceOut.interpConstX(3.0).empty())
    {
        std::cerr << "testMesh3DInterpolator: expected no segments for "
            "x outside mesh range\n";
        return 1;
    }

    return 0; // Success
}

// NOLINTEND(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access, misc-use-anonymous-namespace)
