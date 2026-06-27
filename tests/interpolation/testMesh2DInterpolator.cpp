/**
 * @file testMesh2DInterpolator.cpp
 * @author Mark Krumholz
 * @brief Implementation of the testMesh2DInterpolator function
 * @date 2024-06-19
 */

#include "../src/interpolation/Mesh2DInterpolator.hpp"
#include "../src/utils/MiscUtils.hpp"
#include "testMesh2DInterpolator.hpp"
#include <array>
#include <cmath>
#include <gsl/gsl_interp.h>
#include <mdspan>

// NOLINTBEGIN(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access, misc-use-anonymous-namespace)

auto testMesh2DInterpolator() -> int
{
    // Construct a test data set of size (4, 3) where the data points are 
    // y_j = j
    // x_ij = i + 0.1 * j
    // f_ij = { x_ij + y_j, sin(x_{ij} + y_j), cos(x_{ij} + y_j) }
    constexpr size_t nx = 4;
    constexpr size_t ny = 3;
    constexpr size_t nF = 3;
    const double fac = 0.1;
    std::array<double,nx*ny> xData = { 0 };
    std::array<double,ny> yData = { 0 };
    std::array<double,nx*ny*nF> fData = { 0 };
    const std::mdspan<double, std::extents<size_t, nx, ny>> 
        x(xData.data());
    const std::mdspan<double, std::extents<size_t, ny>> 
        y(yData.data());
    const std::mdspan<double, std::extents<size_t, nx, ny, nF>>
        f(fData.data());
    for (size_t j = 0; j < ny; ++j) {
        y[j] = static_cast<double>(j);
        for (size_t i = 0; i < nx; ++i) {
            x[i,j] = static_cast<double>(i) +
                (fac * static_cast<double>(j));
            f[i,j,0] = x[i,j] + y[j];
            f[i,j,1] = std::sin(f[i,j,0]);
            f[i,j,2] = std::cos(f[i,j,0]);
        }
    }

    // Construct interpolator
    interp::Mesh2DInterpolator<3> m2d(x, y, f, gsl_interp_linear);

    // Construct a data set with duplication of x, to verify that
    // functions properly; this data set is identical to the standard test,
    // except that the first row is changed to { 0, 2, 2, 3} instead of
    // {0, 1, 2, 3}
    auto xDataDup = xData;
    const std::mdspan<double, std::extents<size_t, nx, ny>> 
        xDup(xDataDup.data());
    xDup[2,0] = 2.0;

    // Construct interpolator with duplication
    interp::Mesh2DInterpolator<3> m2dDup(xDup, y, f, gsl_interp_linear);

    // Test making interpolators from the mesh
    auto yInterp = m2d.interpConstY(0.5);

    return 0; // Success
}

// NOLINTEND(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access, misc-use-anonymous-namespace)
