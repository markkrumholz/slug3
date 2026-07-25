/**
 * @file testInterpolation1D.hpp
 * @author Mark Krumholz
 * @brief Unit tests for the Interpolation1D class
 * @details
 * This file contains unit tests for the Interpolation1D class.
 * @date 2024-06-27
 */

#ifndef TESTINTERPOLATION1D_HPP
#define TESTINTERPOLATION1D_HPP

#include "../src/interpolation/Interpolator1D.hpp"
#include "../src/utils/MiscUtils.hpp"
#include <array>
#include <cmath>
#include <gsl/gsl_interp.h>
#include <iostream>
#include <ranges>
#include <vector>

/**
 * @brief Unit test for the Interpolation1D class
 * @returns 0 if the test passes, 1 if it fails
 */
auto testInterpolator1D() -> int
{
    // Make some test data
    constexpr size_t nF = 3;
    std::vector<double> x = { 1, 2, 3, 4 };
    std::array<std::vector<double>, nF> f;
    for (size_t i = 0; i < nF; i++)
    {
        f[i].resize(x.size());
        for (auto && [xi, fi] : std::views::zip(x, f[i]))
        {
            fi = std::pow(xi, i);
        }
    }

    // Construct interpolators for both a single data slice
    // and all data quantities
    interp::Interpolator1D<> interp1(x, f[1], gsl_interp_linear);
    interp::Interpolator1D<nF> interpN(x, f, gsl_interp_linear);

    // Check that interpolators give correct values
    std::vector<double> xTest = { 1.5, 2, 3.7 };
    std::vector<std::array<double, nF>> expected(xTest.size());
    expected[0] = { 1, 1.5, 
        0.5 * (std::pow(1.0,2) + std::pow(2.0,2)) };
    expected[1] = { 1, 2, std::pow(2.0,2) };
    expected[2] = { 1, 3.7, 
                0.3 * std::pow(3.0,2) + 0.7 * std::pow(4.0,2) };
    for (const auto& [xT, ex] : std::views::zip(xTest, expected))
    {
        auto fx1 = interp1(xT); // Single quantity
        auto fxN = interpN(xT); // Vector of quantities
        auto fxN1 = interpN(xT, 1); // Single quantity from vector
        if (!utils::approxEqual(fx1, ex[1])) {
            std::cerr << "testInterpolator1D: at x = " << xT
                << " expected single-quantity interpolator to return "
                << ex[1] << ", instead got " << fx1 << "\n";
            return 1;
        }
        if (!utils::approxEqual(fxN1, ex[1])) {
            std::cerr << "testInterpolator1D: at x = " << xT
                << " expected single-quantity from vector interpolator to return "
                << ex[1] << ", instead got " << fxN1 << "\n";
            return 1;
        }
        if (fxN.size() != nF)
        {
            std::cerr << "testInterpolator1D: at x = " << xT
                << " expected vector interpolator to return " << nF
                << " quantities, instead got " << fxN.size() << "\n";
            return 1;
        }
        for (const auto& [fxNElem, exElem] : std::views::zip(fxN, ex))
        {
            if (!utils::approxEqual(fxNElem, exElem))
            {
                std::cerr << "testInterpolator1D: at x = " << xT
                    << " expected vector interpolator to return "
                    << ex[0] << " " << ex[1] << " " << ex[2]
                    << ", instead got " 
                    << fxN[0] << " " << fxN[1] << " " << fxN[2] << "\n";
                return 1;
            }            
        }

    }

    return 0; // Success
}

/**
 * @brief Unit test for Interpolator1D::integ
 * @returns 0 if the test passes, 1 if it fails
 * @details
 * Reuses testInterpolator1D's own (x, f) setup: x = {1, 2, 3, 4} and,
 * for quantity i, f = x^i. For i = 0 (constant) and i = 1 (linear),
 * linear interpolation reproduces f exactly, so the integral of the
 * interpolant over any [x0, x1] equals the exact analytic integral of
 * f itself. For i = 2 (quadratic), linear interpolation between grid
 * points does not reproduce f exactly, so the expected values instead
 * come from the trapezoid-rule area under the piecewise-linear
 * interpolant directly -- computed by hand from the same (x, f) grid
 * points integ() itself interpolates between, rather than from x^2.
 */
auto testInterpolator1DInteg() -> int
{
    // Make the same test data as testInterpolator1D
    constexpr size_t nF = 3;
    std::vector<double> x = { 1, 2, 3, 4 };
    std::array<std::vector<double>, nF> f;
    for (size_t i = 0; i < nF; i++)
    {
        f[i].resize(x.size());
        for (auto && [xi, fi] : std::views::zip(x, f[i]))
        {
            fi = std::pow(xi, i);
        }
    }

    interp::Interpolator1D<> interp1(x, f[1], gsl_interp_linear);
    interp::Interpolator1D<nF> interpN(x, f, gsl_interp_linear);

    // Three test ranges: a single grid interval ([1, 2]), two full
    // grid intervals ([2, 4]), and a sub-interval of a single grid
    // interval that doesn't land on grid points at either end
    // ([1.2, 1.8], entirely within [1, 2]) -- exercising both the
    // full-segment and partial-segment cases of the underlying
    // piecewise-linear integration.
    struct Range { double x0; double x1; std::array<double, nF> expected; };
    const std::vector<Range> ranges = {
        // [1, 2]: i = 0 -> 1 * (2 - 1); i = 1 -> 0.5 * (2^2 - 1^2);
        // i = 2 -> trapezoid on (1, f=1)-(2, f=4)
        { 1.0, 2.0, { 1.0 * (2.0 - 1.0), 0.5 * (4.0 - 1.0), 0.5 * (1.0 + 4.0) * (2.0 - 1.0) } },
        // [2, 4]: i = 0 -> 1 * (4 - 2); i = 1 -> 0.5 * (4^2 - 2^2);
        // i = 2 -> trapezoid on (2, f=4)-(3, f=9) plus (3, f=9)-(4, f=16)
        { 2.0, 4.0, { 1.0 * (4.0 - 2.0), 0.5 * (16.0 - 4.0),
            (0.5 * (4.0 + 9.0) * (3.0 - 2.0)) + (0.5 * (9.0 + 16.0) * (4.0 - 3.0)) } },
        // [1.2, 1.8], within [1, 2]: the interpolant there is
        // 1 + 3 * (x - 1) (slope from f(1) = 1 to f(2) = 4), so
        // i = 2's expected value is the trapezoid of that line's own
        // endpoint values at 1.2 and 1.8
        { 1.2, 1.8, { 1.0 * (1.8 - 1.2), 0.5 * ((1.8 * 1.8) - (1.2 * 1.2)),
            0.5 * ((1.0 + (3.0 * 0.2)) + (1.0 + (3.0 * 0.8))) * (1.8 - 1.2) } },
    };

    for (const auto& r : ranges)
    {
        const auto integ1 = interp1.integ(r.x0, r.x1); // Single quantity
        const auto integN = interpN.integ(r.x0, r.x1); // Vector of quantities
        const auto integN1 = interpN.integ(r.x0, r.x1, 1); // Single quantity from vector

        if (!utils::approxEqual(integ1, r.expected[1])) {
            std::cerr << "testInterpolator1DInteg: integrating over ["
                << r.x0 << ", " << r.x1 << "] expected single-quantity "
                "interpolator to return " << r.expected[1]
                << ", instead got " << integ1 << "\n";
            return 1;
        }
        if (!utils::approxEqual(integN1, r.expected[1])) {
            std::cerr << "testInterpolator1DInteg: integrating over ["
                << r.x0 << ", " << r.x1 << "] expected single-quantity "
                "from vector interpolator to return " << r.expected[1]
                << ", instead got " << integN1 << "\n";
            return 1;
        }
        if (integN.size() != nF)
        {
            std::cerr << "testInterpolator1DInteg: integrating over ["
                << r.x0 << ", " << r.x1 << "] expected vector interpolator "
                "to return " << nF << " quantities, instead got "
                << integN.size() << "\n";
            return 1;
        }
        for (const auto& [integNElem, exElem] : std::views::zip(integN, r.expected))
        {
            if (!utils::approxEqual(integNElem, exElem))
            {
                std::cerr << "testInterpolator1DInteg: integrating over ["
                    << r.x0 << ", " << r.x1 << "] expected vector "
                    "interpolator to return " << r.expected[0] << " "
                    << r.expected[1] << " " << r.expected[2]
                    << ", instead got " << integN[0] << " " << integN[1]
                    << " " << integN[2] << "\n";
                return 1;
            }
        }
    }

    return 0; // Success
}

#endif // TESTINTERPOLATION1D_HPP