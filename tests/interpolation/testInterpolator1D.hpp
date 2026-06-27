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

#endif // TESTINTERPOLATION1D_HPP