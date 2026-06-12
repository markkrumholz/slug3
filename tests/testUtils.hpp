/**
 * @file testUtils.hpp
 * @author Mark Krumholz
 * @brief Utility functions for unit tests.
 * @details
 * This file contains utility functions that are used across multiple unit tests in
 * the test suite. These functions include helpers for comparing
 * floating-point numbers and other common tasks that arise in unit testing.
 * @date 2024-06-12
 */

#ifndef TESTUTILS_HPP
#define TESTUTILS_HPP

#include <cmath>

namespace testUtils {

    /**
     * @brief Compare two floating-point numbers for approximate equality.
     * @param a The first number to compare.
     * @param b The second number to compare.
     * @param tol The tolerance for comparison (default is 1e-6).
     * @return True if the numbers are approximately equal within the specified tolerance, false otherwise.
     */
    inline bool approxEqual(double a, double b, double tol = 1e-6) {
        return std::fabs(a - b) < tol;
    }

}

#endif // TESTUTILS_HPP