/**
 * @file GridBracket.cpp
 * @author Mark Krumholz
 * @brief Implementation of GridBracket.hpp
 * @date 2026-09-12
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#include "GridBracket.hpp"
#include <algorithm>
#include <cstddef>
#include <vector>

namespace utils
{
    namespace
    {
        /**
         * @brief Binary search for the bracketing index of a sorted grid, within given bounds
         * @param grid A sorted (ascending), non-empty grid of values
         * @param value The query value
         * @param idxLo Lower index for the search
         * @param idxHi Upper index for the search
         * @returns The index lo such that [grid[lo], grid[lo + 1]]
         *   brackets value, or idxHi itself if value == grid[idxHi]
         * @details
         * A helper for findBracket, carrying out a binary search
         * between idxLo and idxHi -- deliberately not the grid's full
         * extent, so that findBracket can narrow the search using its
         * cached index. No checking is done here to ensure value
         * actually lies between grid[idxLo] and grid[idxHi]; the
         * caller is responsible for that. Mirrors
         * interp::Mesh2DGrid::ySearch exactly, including its
         * value == grid[idxHi] edge case.
         */
        auto bracketSearch(const std::vector<double>& grid, const double value, //NOLINT(llvm-prefer-static-over-anonymous-namespace)
            const std::size_t idxLo, const std::size_t idxHi) -> std::size_t
        {
            std::size_t lo = idxLo;
            std::size_t hi = idxHi;
            while (hi > lo + 1)
            {
                const std::size_t mid = (lo + hi) / 2;
                if (value > grid[mid]) { lo = mid; } // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- lo, hi, mid are all < grid.size() by construction
                else { hi = mid; }
            }
            if (value == grid[hi]) { return hi; } // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- see above
            return lo;
        }
    } // namespace

    auto findBracket(const std::vector<double>& grid, const double value,
        std::size_t& cacheIdx) -> Bracket
    {
        const std::size_t n = grid.size();
        if (n == 1) { return { 0, 0, 0.0 }; }

        // NOLINTBEGIN(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- cacheIdx is always < n - 1 by construction, both on entry (see below) and after every branch here
        if (value < grid[cacheIdx])
        {
            cacheIdx = bracketSearch(grid, value, 0, cacheIdx); // below cached position: search left
        }
        else if (value >= grid[cacheIdx + 1])
        {
            // above cached position, or exactly on its upper edge; in
            // the latter case cacheIdx must still advance, since
            // leaving it unchanged would compute t as 1 instead of 0
            // for the next cell up
            cacheIdx = bracketSearch(grid, value, cacheIdx, n - 1); // search right
        }
        if (cacheIdx == n - 1) { --cacheIdx; } // value == grid.back(): use the last interval, not a degenerate one past it
        // NOLINTEND(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)

        const std::size_t lo = cacheIdx;
        const std::size_t hi = cacheIdx + 1;
        const double t = std::clamp(
            (value - grid[lo]) / (grid[hi] - grid[lo]), 0.0, 1.0); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- lo, hi < n by construction
        return { lo, hi, t };
    }

} // namespace utils
