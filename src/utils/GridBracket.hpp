/**
 * @file GridBracket.hpp
 * @author Mark Krumholz
 * @brief Cache-accelerated bracket search on a sorted 1D grid
 * @date 2026-09-12
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#ifndef GRIDBRACKET_HPP
#define GRIDBRACKET_HPP

#include <cstddef>
#include <vector>

namespace utils
{
    /**
     * @brief A bracketing pair of grid indices, plus an interpolation weight
     * @details
     * lo_ and hi_ are the indices of the grid points immediately below
     * and above (or equal to) a query value, and t_ is the fractional
     * distance of the query value between them, so that
     * (1 - t_) * grid[lo_] + t_ * grid[hi_] recovers the query value.
     * For a grid of size 1 (a degenerate/singular axis with no actual
     * extent -- e.g. a yield model tabulated at only one [Fe/H]), lo_
     * == hi_ == 0 and t_ == 0, so a caller that always applies this
     * same (1 - t_) * grid[lo_] + t_ * grid[hi_] weighting handles that
     * case correctly with no special-casing of its own.
     */
    struct Bracket
    {
        std::size_t lo_; /**< Index of the grid point at or below the query value */
        std::size_t hi_; /**< Index of the grid point at or above the query value */
        double t_;       /**< Fractional distance of the query value between grid[lo_] and grid[hi_] */
    };

    /**
     * @brief Find the bracketing grid points of a sorted grid
     * @param grid A sorted (ascending), non-empty grid of values
     * @param value The query value; assumed to already lie within
     *   [grid.front(), grid.back()]
     * @param cacheIdx The calling thread's cached bracket index for
     *   this axis (e.g. one element of a ThreadVec<size_t>, already
     *   resolved to the element private to this thread); updated in
     *   place to the bracket this call finds, so the next call -- if
     *   its own query value is still within, or close to, this same
     *   cell -- can reuse it
     * @returns The bracketing Bracket for value
     * @details
     * Locates the bracket via a binary search accelerated by cacheIdx
     * -- see GridBracket.cpp's own definition for the full rationale.
     * Shared by every tensor-grid interpolator in this codebase (the
     * spectral synthesis libraries in src/specsyn, and
     * yields::YieldChannel) rather than each reimplementing its own
     * copy of the same bracket-search logic.
     */
    auto findBracket(const std::vector<double>& grid, double value,
        std::size_t& cacheIdx) -> Bracket;

} // namespace utils

#endif // GRIDBRACKET_HPP
