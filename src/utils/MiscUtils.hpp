/**
 * @file MiscUtils.hpp
 * @author Mark Krumholz
 * @brief Miscellaneous utility functions
 * @date 2024-06-19
 */

#ifndef MISCUTILS_HPP
#define MISCUTILS_HPP

#include <cmath>

namespace utils
{
    /**
     * @brief Compare two floating-point numbers for approximate equality.
     * @param a The first number to compare.
     * @param b The second number to compare.
     * @param tol The tolerance for comparison (default is 1e-6).
     * @return True if the numbers are approximately equal within the specified tolerance, false otherwise.
     */
    inline auto approxEqual(double a, double b, double tol = 1e-6) -> bool {
        return std::fabs(a - b) < tol;
    }

} // namespace utils

#endif // MISCUTILS_HPP