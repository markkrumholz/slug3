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

/**
 * @brief Unit test for the Interpolator1D::integ(g, x0, x1) and
 *        Interpolator1D::integ(g, x0, x1, idx) overloads
 * @returns 0 if the test passes, 1 if it fails
 * @details
 * All cases use gsl_interp_linear so expected values can be computed exactly
 * from the trapezoid rule.
 *
 * Case 1 – same grid, two fields, via the all-fields overload and the
 * idx overload.  f[0] = x and g[0] = 1, f[1] = 1 and g[1] = x, both on
 * x = {1, 2, 3, 4}. For both fields the product is x, so the exact integral
 * over [1, 4] is (4^2 - 1^2) / 2 = 7.5.
 *
 * Case 2 – different grids, domain clipping by g.  f is on {0, 2, 4} with
 * f = x; g is on {1, 3, 5} with g = 1 (constant).  With x0 = 0, x1 = 5
 * the effective range is [max(0,1,0), min(4,5,5)] = [1, 4].  The merged
 * grid becomes {1, 2, 3, 4}, and f * g = x there, so the expected integral
 * is again 7.5.
 *
 * Case 3 – domain clipping by g from above.  f is on {1, 2, 3, 4} with
 * f = x; g is on {2, 3} with g = 1.  With x0 = 1, x1 = 4 the effective
 * range is [2, 3].  The only merged-grid points are xLo = 2 and xHi = 3,
 * so the product interpolant is the straight line through (2, 2) and (3, 3),
 * giving integral 0.5 * (2 + 3) * 1 = 2.5.
 *
 * Case 4 – non-overlapping domains → zero.  f is on {1, 2} and g is on
 * {3, 4}; any integration limits give xLo >= xHi, so both overloads
 * must return 0.
 */
auto testInterpolator1DProductInteg() -> int
{
    // Case 1: same grid, NF = 2
    {
        constexpr size_t nF = 2;
        const std::vector<double> x = { 1.0, 2.0, 3.0, 4.0 };
        // f[0] = x, f[1] = 1
        const std::array<std::vector<double>, nF> fData = {
            std::vector<double>{ 1.0, 2.0, 3.0, 4.0 },
            std::vector<double>{ 1.0, 1.0, 1.0, 1.0 }
        };
        // g[0] = 1, g[1] = x
        const std::array<std::vector<double>, nF> gData = {
            std::vector<double>{ 1.0, 1.0, 1.0, 1.0 },
            std::vector<double>{ 1.0, 2.0, 3.0, 4.0 }
        };
        interp::Interpolator1D<nF> fInterp(x, fData, gsl_interp_linear);
        interp::Interpolator1D<nF> gInterp(x, gData, gsl_interp_linear);

        constexpr double expected = 7.5; // integral of x from 1 to 4
        constexpr double x0 = 1.0;
        constexpr double x1 = 4.0;

        // All-fields overload
        const auto resultAll = fInterp.integ(gInterp, x0, x1);
        for (size_t i = 0; i < nF; ++i)
        {
            if (!utils::approxEqual(resultAll[i], expected)) // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index)
            {
                std::cerr << "testInterpolator1DProductInteg case 1 (all-fields):"
                    " field " << i << " expected " << expected
                    << ", got " << resultAll[i] << "\n"; // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index)
                return 1;
            }
        }

        // idx overload
        for (size_t i = 0; i < nF; ++i)
        {
            const auto resultIdx = fInterp.integ(gInterp, x0, x1, i);
            if (!utils::approxEqual(resultIdx, expected))
            {
                std::cerr << "testInterpolator1DProductInteg case 1 (idx=" << i
                    << "): expected " << expected << ", got " << resultIdx << "\n";
                return 1;
            }
        }
    }

    // Case 2: different grids, effective domain [1, 4], NF = 1
    {
        // f = x on {0, 2, 4}; g = 1 on {1, 3, 5}
        interp::Interpolator1D<> fInterp(
            { 0.0, 2.0, 4.0 }, { 0.0, 2.0, 4.0 }, gsl_interp_linear);
        interp::Interpolator1D<> gInterp(
            { 1.0, 3.0, 5.0 }, { 1.0, 1.0, 1.0 }, gsl_interp_linear);

        constexpr double expected = 7.5; // integral of x from 1 to 4

        const auto resultAll = fInterp.integ(gInterp, 0.0, 5.0);
        if (!utils::approxEqual(resultAll, expected))
        {
            std::cerr << "testInterpolator1DProductInteg case 2 (all-fields):"
                " expected " << expected << ", got " << resultAll << "\n";
            return 1;
        }
        const auto resultIdx = fInterp.integ(gInterp, 0.0, 5.0, 0);
        if (!utils::approxEqual(resultIdx, expected))
        {
            std::cerr << "testInterpolator1DProductInteg case 2 (idx=0):"
                " expected " << expected << ", got " << resultIdx << "\n";
            return 1;
        }
    }

    // Case 3: g's domain clips the range from above, NF = 1
    {
        // f = x on {1, 2, 3, 4}; g = 1 on {2, 3}
        interp::Interpolator1D<> fInterp(
            { 1.0, 2.0, 3.0, 4.0 }, { 1.0, 2.0, 3.0, 4.0 }, gsl_interp_linear);
        interp::Interpolator1D<> gInterp(
            { 2.0, 3.0 }, { 1.0, 1.0 }, gsl_interp_linear);

        constexpr double expected = 2.5; // trapezoid: 0.5*(2+3)*1

        const auto resultAll = fInterp.integ(gInterp, 1.0, 4.0);
        if (!utils::approxEqual(resultAll, expected))
        {
            std::cerr << "testInterpolator1DProductInteg case 3 (all-fields):"
                " expected " << expected << ", got " << resultAll << "\n";
            return 1;
        }
        const auto resultIdx = fInterp.integ(gInterp, 1.0, 4.0, 0);
        if (!utils::approxEqual(resultIdx, expected))
        {
            std::cerr << "testInterpolator1DProductInteg case 3 (idx=0):"
                " expected " << expected << ", got " << resultIdx << "\n";
            return 1;
        }
    }

    // Case 4: non-overlapping domains → both overloads must return 0, NF = 1
    {
        interp::Interpolator1D<> fInterp(
            { 1.0, 2.0 }, { 1.0, 2.0 }, gsl_interp_linear);
        interp::Interpolator1D<> gInterp(
            { 3.0, 4.0 }, { 1.0, 1.0 }, gsl_interp_linear);

        const auto resultAll = fInterp.integ(gInterp, 1.0, 4.0);
        if (!utils::approxEqual(resultAll, 0.0))
        {
            std::cerr << "testInterpolator1DProductInteg case 4 (all-fields):"
                " expected 0, got " << resultAll << "\n";
            return 1;
        }
        const auto resultIdx = fInterp.integ(gInterp, 1.0, 4.0, 0);
        if (!utils::approxEqual(resultIdx, 0.0))
        {
            std::cerr << "testInterpolator1DProductInteg case 4 (idx=0):"
                " expected 0, got " << resultIdx << "\n";
            return 1;
        }
    }

    return 0; // Success
}

#endif // TESTINTERPOLATION1D_HPP