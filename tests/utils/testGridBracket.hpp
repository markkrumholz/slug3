/**
 * @file testGridBracket.hpp
 * @author Mark Krumholz
 * @brief Unit tests for utils::findBracket.
 * @date 2026-09-12
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#ifndef TESTGRIDBRACKET_HPP
#define TESTGRIDBRACKET_HPP

#include "../../src/utils/GridBracket.hpp"
#include <cmath>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

auto checkGridBracket(const std::string& label, const utils::Bracket& b,
    const std::size_t expectedLo, const std::size_t expectedHi, const double expectedT) -> int
{
    constexpr double tol = 1e-12;
    if (b.lo_ != expectedLo || b.hi_ != expectedHi || std::abs(b.t_ - expectedT) > tol)
    {
        std::cerr << label << ": expected (lo=" << expectedLo << ", hi=" << expectedHi <<
            ", t=" << expectedT << "), got (lo=" << b.lo_ << ", hi=" << b.hi_ <<
            ", t=" << b.t_ << ")\n";
        return 1;
    }
    return 0;
}

/**
 * @brief A size-1 grid always brackets to (0, 0, 0.0), regardless of query value
 * @return 0 if the test passes, 1 if it fails
 */
auto testGridBracketSingularGrid() -> int
{
    const std::vector<double> grid{5.0};
    std::size_t cache = 0;
    int result = 0;
    result += checkGridBracket("testGridBracketSingularGrid (at the one grid value)",
        utils::findBracket(grid, 5.0, cache), 0, 0, 0.0);
    result += checkGridBracket("testGridBracketSingularGrid (a different value)",
        utils::findBracket(grid, 999.0, cache), 0, 0, 0.0);
    return result;
}

/**
 * @brief An exact hit on a grid point at the front, an interior point, and grid.back()
 * @return 0 if the test passes, 1 if it fails
 */
auto testGridBracketExactHits() -> int
{
    const std::vector<double> grid{0.0, 1.0, 2.0, 3.0, 4.0};
    std::size_t cache = 0;
    int result = 0;
    result += checkGridBracket("testGridBracketExactHits (front)",
        utils::findBracket(grid, 0.0, cache), 0, 1, 0.0);
    result += checkGridBracket("testGridBracketExactHits (interior)",
        utils::findBracket(grid, 2.0, cache), 2, 3, 0.0);
    // grid.back(): the last interval, not a degenerate one past it -- see
    // findBracket()'s own comment on this edge case
    result += checkGridBracket("testGridBracketExactHits (back)",
        utils::findBracket(grid, 4.0, cache), 3, 4, 1.0);
    return result;
}

/**
 * @brief Ordinary interior interpolation weights, on a coarse, unevenly-indexed grid
 * @return 0 if the test passes, 1 if it fails
 */
auto testGridBracketInterpolation() -> int
{
    const std::vector<double> grid{0.0, 10.0, 20.0};
    std::size_t cache = 0;
    int result = 0;
    result += checkGridBracket("testGridBracketInterpolation (first cell, midpoint)",
        utils::findBracket(grid, 5.0, cache), 0, 1, 0.5);
    result += checkGridBracket("testGridBracketInterpolation (second cell, quarter)",
        utils::findBracket(grid, 12.5, cache), 1, 2, 0.25);
    return result;
}

/**
 * @brief cacheIdx is reused for a nearby query and updated for a distant one
 * @return 0 if the test passes, 1 if it fails
 * @details
 * findBracket()'s own cacheIdx parameter exists purely to accelerate
 * repeated nearby queries (see its own comment); this checks both
 * halves of that contract directly, not just that the returned
 * Bracket is correct regardless (already covered by the other tests
 * here): that cacheIdx stays put across a query still inside the same
 * cell, and does advance for a query that has moved to a different one.
 */
auto testGridBracketCache() -> int
{
    const std::vector<double> grid{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::size_t cache = 0;
    int result = 0;

    const auto b1 = utils::findBracket(grid, 4.5, cache);
    result += checkGridBracket("testGridBracketCache (initial)", b1, 4, 5, 0.5);
    if (cache != 4)
    {
        std::cerr << "testGridBracketCache: expected cacheIdx == 4 after "
            "the initial query, got " << cache << "\n";
        result = 1;
    }

    // Still inside the same [4, 5) cell: cacheIdx must not move
    const auto b2 = utils::findBracket(grid, 4.9, cache);
    result += checkGridBracket("testGridBracketCache (nearby, same cell)", b2, 4, 5, 0.9);
    if (cache != 4)
    {
        std::cerr << "testGridBracketCache: cacheIdx should remain 4 for a "
            "query still inside the same cell, got " << cache << "\n";
        result = 1;
    }

    // Far from the cached cell: cacheIdx must update to the new bracket
    const auto b3 = utils::findBracket(grid, 0.5, cache);
    result += checkGridBracket("testGridBracketCache (distant)", b3, 0, 1, 0.5);
    if (cache != 0)
    {
        std::cerr << "testGridBracketCache: expected cacheIdx == 0 after "
            "the distant query, got " << cache << "\n";
        result = 1;
    }

    return result;
}

#endif // TESTGRIDBRACKET_HPP
