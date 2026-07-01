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
#include <iostream>
#include <gsl/gsl_interp.h>
#include <mdspan>

// NOLINTBEGIN(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access, misc-use-anonymous-namespace)

auto testMesh2DInterpolator() -> int
{
    // Construct a test data set of size (4, 3) where the data points are 
    // y_j = j
    // x_ij = i + (j % 2) (0.1 * j)
    // f_ij = { x_ij + y_j, sin(x_{ij} + y_j) }
    constexpr size_t nx = 4;
    constexpr size_t ny = 3;
    constexpr size_t nF = 2;
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
                ((j % 2) * fac * static_cast<double>(j));
            f[i,j,0] = x[i,j] + y[j];
            f[i,j,1] = std::sin(f[i,j,0]);
        }
    }

    // Construct interpolator
    const interp::Mesh2DInterpolator<nF> m2d(x, y, f, gsl_interp_linear);

    // Construct a data set with duplication of x, to verify that
    // functions properly; this data set is identical to the standard test,
    // except that the first row is changed to { 0, 2, 2, 3} instead of
    // {0, 1, 2, 3}
    auto xDataDup = xData;
    const std::mdspan<double, std::extents<size_t, nx, ny>> 
        xDup(xDataDup.data());
    xDup[2,0] = 2.0;

    // Construct interpolator with duplication
    const interp::Mesh2DInterpolator<nF> m2dDup(xDup, y, f, gsl_interp_linear);

    // Test that x interpolators produce the correct number of segments
    // and limits on them, and that interpolation produces correct values
    const std::vector<double> xTest = { -0.5, 0.05, 2, 3.05 };
    const std::vector<std::vector<double>> yPtXTest = {  
        { }, // xTest = -0.5
        { 0, 0.5, 1.5, 2.0 }, // xTest = 0.05
        { 0, 1, 2}, // xTest = 2
        { 0.5, 1, 1.5 } // xTest = 3.05
    };
    const std::vector<std::vector<std::array<double,2>>> xLim = {
        { }, // xTest = -0.5
        { { 0,0.5 }, { 1.5,2.0 } }, // xTest = 0.05
        { { 0, 2} }, // xTest = 2
        { { 0.5, 1.5} } // xTest = 3.05
    };
    const std::vector<std::vector<std::array<double,nF>>> fXTest = {
        { }, // xTest = -0.5
        { { 0.95 * f[0,0,0] + 0.05 * f[1,0,0],
            0.95 * f[0,0,1] + 0.05 * f[1,0,1] },
            { 0.5 * f[0,0,0] + 0.5 * f[0,1,0],
            0.5 * f[0,0,1] + 0.5 * f[0,1,1] },
            { 0.5 * f[0,1,0] + 0.5 * f[0,2,0],
            0.5 * f[0,1,1] + 0.5 * f[0,2,1] }, 
            { 0.95 * f[0,2,0] + 0.05 * f[1,2,0],
            0.95 * f[0,2,1] + 0.05 * f[1,2,1] } }, // xTest = 0.05
        { { f[2,0,0], f[2,0,1] }, 
            { 0.1 * f[1,1,0] + 0.9 * f[2,1,0],
            0.1 * f[1,1,1] + 0.9 * f[2,1,1] },
            { f[2,2,0], f[2,2,1] } }, // xTest = 2
        { { 0.5 * f[3,0,0] + 0.5 * f[3,1,0],
            0.5 * f[3,0,1] + 0.5 * f[3,1,1] },
            { 0.05 * f[2,1,0] + 0.95 * f[3,1,0],
            0.05 * f[2,1,1] + 0.95 * f[3,1,1] },
            { 0.5 * f[3,1,0] + 0.5 * f[3,2,0],
            0.5 * f[3,1,1] + 0.5 * f[3,2,1] } } // xTest = 3.05
    };
    for (auto [xt, xl, ypt, fpt] : 
        std::views::zip(xTest, xLim, yPtXTest, fXTest))
    {
        // Construct interpolator
        auto xInterp = m2d.interpConstX(xt);

        // Verify that it has the correct size and limits
        if (xInterp.size() != xl.size())
        {
            std::cerr << "testMesh2DInterpolator: at x = " << xt
                << " expected to find " << xl.size() 
                << " segements, instead found " << xInterp.size() << "\n";
            return 1;
        }
        for (auto [seg, xlseg] : 
            std::views::zip(xInterp, xl))
        {
            auto [seglo, seghi] = seg->xRange();
            if (!utils::approxEqual(seglo, xlseg[0]) || 
                !utils::approxEqual(seghi, xlseg[1]))
            {
                std::cerr << "testMesh2DInterpolator: at x = " << xt
                    << " expected to find segment with limits "
                    << xlseg[0] << "-" << xlseg[1] << ", instead found "
                    << seglo << "-" << seghi << "\n";
                return 1;
            }
        }

        // Check that interpolation produces correct values at the
        // test points
        for (auto [yval, fval] : std::views::zip(ypt, fpt))
        {
            for (const auto& xi : xInterp)
            {
                if (yval >= xi->xMin() && yval <= xi->xMax())
                {
                    auto finterp = (*xi)(yval);
                    for (auto [fi, fv] : std::views::zip(finterp, fval))
                    {
                        if (!utils::approxEqual(fi,fv))
                        {
                            std::cerr << "testMesh2DInterpolator: at x = " << xt
                                << ", y = " << yval << ", expected f = "
                                << fv << ", found " << fi << "\n";
                            return 1;
                        }
                    }
                }
            }
        }
    }

    // Test that y interpolators produce the correct number of segments
    // and limits on them, and that interpolation produces correct values
    const std::vector<double> yTest = { -0.5, 0.5 };
    const std::vector<std::vector<double>> xPtYTest = {  
        { }, // yTest = -0.5
        { 0.05, 0.55, 2.05 } // yTest = 0.5
    };
    const std::vector<std::vector<double>> yLim = {
        { }, // yTest = -0.5
        { { 0.05, 3.05 } } // yTest = 0.5
    };
    const std::vector<std::vector<std::array<double,nF>>> fYTest = {
        { }, // yTest = -0.5
        { { 0.5 * f[0,0,0] + 0.5 * f[0,1,0],
            0.5 * f[0,0,1] + 0.5 * f[0,1,1] },
            { 0.25 * f[0,0,0] + 0.25 * f[0,1,0] +
            0.25 * f[1,0,0] + 0.25 * f[1,1,0],
            0.25 * f[0,0,1] + 0.25 * f[0,1,1] +
            0.25 * f[1,0,1] + 0.25 * f[1,1,1]},
            { 0.5 * f[2,0,0] + 0.5 * f[2,1,0],
            0.5 * f[2,0,1] + 0.5 * f[2,1,1]}
        } // yTest = 0.5
    };
    
    for (auto [yt, yl, xpt, fpt] : 
        std::views::zip(yTest, yLim, xPtYTest, fYTest))
    {
        // Construct interpolator
        auto yInterp = m2d.interpConstY(yt);

        // Verify that it has the correct size and limits
        if (yInterp == nullptr && yl.empty()) { continue; }
        if (yInterp == nullptr && !yl.empty())  
        {
            std::cerr << "testMesh2DInterpolator: at y = " << yt
                << " expected to find " << yl.size() 
                << " segements, instead found 0\n";
            return 1;
        }
        if (yInterp != nullptr && yInterp->xRange() != std::make_pair(yl[0], yl[1]))
        {
            std::cerr << "testMesh2DInterpolator: at y = " << yt
                << " expected segment limits " << yl[0] << "-" << yl[1]
                << ", instead found " << yInterp->xRange().first << "-" << yInterp->xRange().second << "\n";
            return 1;
        }

        // Check that interpolation produces correct values at the
        // test points
        for (auto [xval, fval] : std::views::zip(xpt, fpt))
        {
            auto finterp = (*yInterp)(xval);
            for (auto [fi, fv] : std::views::zip(finterp, fval))
            {
                if (!utils::approxEqual(fi,fv))
                {
                    std::cerr << "testMesh2DInterpolator: at y = " << yt
                        << ", x = " << xval << ", expected f = "
                        << fv << ", found " << fi << "\n";
                    return 1;
                }
            }
        }
    }

    return 0; // Success
}

// NOLINTEND(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access, misc-use-anonymous-namespace)
