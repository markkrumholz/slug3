/**
 * @file SpecsynLib.cpp
 * @author Mark Krumholz
 * @brief Implementation of SpecsynLib.hpp
 * @date 2026-07-20
 */

#include "SpecsynLib.hpp"
#include "../interpolation/Interpolator1D.hpp"
#include "SpecsynCommons.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace specsyn
{
    namespace
    {
        /**
         * @brief A bracketing pair of grid indices, plus an interpolation weight
         * @details
         * lo_ and hi_ are the indices of the grid points immediately
         * below and above (or equal to) a query value, and t_ is the
         * fractional distance of the query value between them, so
         * that (1 - t_) * grid[lo_] + t_ * grid[hi_] recovers the
         * query value. For a grid of size 1 (a degenerate axis with
         * no actual extent), lo_ == hi_ == 0 and t_ == 0.
         */
        struct Bracket
        {
            size_t lo_;
            size_t hi_;
            double t_;
        };

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
            const size_t idxLo, const size_t idxHi) -> size_t
        {
            size_t lo = idxLo;
            size_t hi = idxHi;
            while (hi > lo + 1)
            {
                const size_t mid = (lo + hi) / 2;
                if (value > grid[mid]) { lo = mid; } // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- lo, hi, mid are all < grid.size() by construction
                else { hi = mid; }
            }
            if (value == grid[hi]) { return hi; } // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- see above
            return lo;
        }

        /**
         * @brief Find the bracketing grid points of a sorted grid
         * @param grid A sorted (ascending), non-empty grid of values
         * @param value The query value; assumed to already lie within
         *   [grid.front(), grid.back()]
         * @param cacheIdx The calling thread's cached bracket index
         *   for this axis (one of SpecsynLib::dim1Cache_/dim2Cache_/
         *   dim3Cache_, already resolved to the element private to
         *   this thread); updated in place to the bracket this call
         *   finds, so the next call -- if its own query value is
         *   still within, or close to, this same cell -- can reuse it
         * @returns The bracketing Bracket for value
         * @details
         * Locates the bracket via a binary search accelerated by
         * cacheIdx, exactly as interp::Mesh2DGrid::yIdx accelerates
         * its own search with jSave_: if value already falls within
         * [grid[cacheIdx], grid[cacheIdx + 1]), the cached bracket is
         * reused with no search at all; otherwise a binary search
         * narrows from the cached index towards whichever side value
         * fell outside of, rather than searching the full axis from
         * scratch. This is an O(log n) search in the worst case
         * (same as a plain binary search from scratch), but O(1) on a
         * cache hit and faster on average whenever successive calls'
         * query values are sorted, or nearly so, along this axis --
         * the common case, since spec() is typically called for a
         * long series of stars in Teff/logg order. None of the three
         * tensor-grid axes can be assumed evenly spaced -- e.g. BOSZ's
         * [Fe/H] grid is uniform, but TLUSTY's (log10 of a fixed set
         * of archival Z values) is not -- so a search, cached or not,
         * is needed regardless.
         */
        auto findBracket(const std::vector<double>& grid, const double value, //NOLINT(llvm-prefer-static-over-anonymous-namespace)
            size_t& cacheIdx) -> Bracket
        {
            const size_t n = grid.size();
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

            const size_t lo = cacheIdx;
            const size_t hi = cacheIdx + 1;
            const double t = std::clamp(
                (value - grid[lo]) / (grid[hi] - grid[lo]), 0.0, 1.0); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- lo, hi < n by construction
            return { lo, hi, t };
        }
    } // namespace

    template <OOBPolicy Policy>
    auto SpecsynLib<Policy>::outOfBoundsResult(const std::string& message) -> std::vector<double>
    {
        // if constexpr, rather than a plain if, so that whichever
        // branch does NOT apply to this Policy is discarded rather
        // than compiled into spec()'s hot path
        if constexpr (Policy == OOBPolicy::raise)
        {
            throw std::runtime_error(message);
        }
        else
        {
            return {};
        }
    }

    template <OOBPolicy Policy>
    auto SpecsynLib<Policy>::spec(const double d1, const double d2, const double d3) const -> std::vector<double> // NOLINT(readability-function-cognitive-complexity) -- bracket-finding, gap-checking, and nested trilinear interpolation are each simple on their own; splitting them into separate functions would only add indirection, not clarity
    {
        // Locate the tensor-grid cell containing (d1, d2, d3), and the
        // trilinear interpolation weights within it, via a
        // cache-accelerated binary search on each axis -- none of
        // dim1_, dim2_, or dim3_ can be assumed evenly spaced (see
        // findBracket). Each dimNCache_() call below binds the
        // element of that axis's ThreadVec private to this thread,
        // which findBracket then reads and updates in place.
        const auto b1 = findBracket(dim1_, d1, dim1Cache_());
        const auto b2 = findBracket(dim2_, d2, dim2Cache_());
        const auto b3 = findBracket(dim3_, d3, dim3Cache_());

        // NOLINTBEGIN(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- b1/b2/b3 indices are all < the corresponding grid's size by construction, and the interpolation loop below is a hot path where the cost of bounds checking matters
        // Every one of the 8 neighboring grid points must actually
        // have a spectrum -- interpolating across an unpopulated
        // point would be meaningless -- or this query point counts
        // as out of bounds. Under OOBPolicy::coerce, a query point is
        // only out of bounds if none of its 8 neighbors has a
        // spectrum; if at least one does, spec() instead interpolates
        // using only the valid neighbors (see below), rather than
        // requiring every corner to be populated.
        bool hasValidNeighbor = false; // NOLINT(misc-const-correctness) -- only ever reassigned inside the OOBPolicy::coerce branch of the if constexpr below; clang-tidy's const-correctness check doesn't see that write when checking a non-coerce instantiation, and wrongly concludes the variable could be const for every Policy
        for (const size_t i1 : { b1.lo_, b1.hi_ })
        {
            for (const size_t i2 : { b2.lo_, b2.hi_ })
            {
                for (const size_t i3 : { b3.lo_, b3.hi_ })
                {
                    if constexpr (Policy == OOBPolicy::coerce)
                    {
                        if (!grid_[i1, i2, i3].empty())
                        {
                            hasValidNeighbor = true;
                            break;
                        }
                    }
                    else
                    {
                        if (grid_[i1, i2, i3].empty())
                        {
                            return outOfBoundsResult(
                                "SpecsynLib: point (" + std::to_string(d1) + ", " +
                                std::to_string(d2) + ", " + std::to_string(d3) +
                                ") falls in a gap in this library's grid");
                        }
                    }
                }
            }
        }
        if constexpr (Policy == OOBPolicy::coerce)
        {
            if (!hasValidNeighbor)
            {
                return outOfBoundsResult(
                    "SpecsynLib: point (" + std::to_string(d1) + ", " +
                    std::to_string(d2) + ", " + std::to_string(d3) +
                    ") has no valid neighboring grid points to coerce to");
            }
        }

        // Trilinear interpolation of the stored quantity over the 8
        // neighboring grid points; no scaling (e.g. by surface area)
        // is applied here -- that is left entirely to the caller.
        // Under OOBPolicy::coerce, an unpopulated corner simply
        // contributes nothing, and wSum (the total weight of the
        // valid corners actually used, guaranteed to be 1 if every
        // corner is populated) renormalizes the result at the end so
        // it still represents a properly weighted average rather than
        // an artificially dimmed spectrum.
        std::vector<double> result(wl_.size(), 0.0);
        double wSum = 0.0; // NOLINT(misc-const-correctness) -- see the identical NOLINT on hasValidNeighbor above; only ever incremented inside the OOBPolicy::coerce branch of the if constexpr below
        for (int b1i = 0; b1i < 2; ++b1i)
        {
            const size_t i1 = (b1i == 0) ? b1.lo_ : b1.hi_;
            const double wgt1 = (b1i == 0) ? (1.0 - b1.t_) : b1.t_;
            for (int b2i = 0; b2i < 2; ++b2i)
            {
                const size_t i2 = (b2i == 0) ? b2.lo_ : b2.hi_;
                const double wgt2 = (b2i == 0) ? (1.0 - b2.t_) : b2.t_;
                for (int b3i = 0; b3i < 2; ++b3i)
                {
                    const size_t i3 = (b3i == 0) ? b3.lo_ : b3.hi_;
                    const double wgt3 = (b3i == 0) ? (1.0 - b3.t_) : b3.t_;

                    const double weight = wgt1 * wgt2 * wgt3;
                    if (weight == 0.0) { continue; } // degenerate axis or exact grid hit: skip a zero-weight corner

                    const auto& corner = grid_[i1, i2, i3];
                    if constexpr (Policy == OOBPolicy::coerce)
                    {
                        if (corner.empty()) { continue; }
                        wSum += weight;
                    }
                    for (size_t w = 0; w < result.size(); ++w)
                    {
                        result[w] += weight * corner[w];
                    }
                }
            }
        }
        if constexpr (Policy == OOBPolicy::coerce)
        {
            // wSum can still be exactly 0 here despite hasValidNeighbor
            // being true above: hasValidNeighbor only checks whether
            // any of the 8 corners is populated, regardless of weight,
            // while the loop just above skips zero-weight corners (an
            // exact grid hit on some axis) before ever checking
            // whether they're populated. So a populated corner that
            // happens to carry zero weight contributes nothing to
            // wSum, and if every nonzero-weight corner is unpopulated,
            // wSum ends up 0 even though hasValidNeighbor was true.
            // Dividing by 0 there would silently produce a NaN/Inf
            // "result" instead of a clean out-of-bounds one, so treat
            // it the same as having no valid neighbor at all.
            if (wSum == 0.0)
            {
                return outOfBoundsResult(
                    "SpecsynLib: point (" + std::to_string(d1) + ", " +
                    std::to_string(d2) + ", " + std::to_string(d3) +
                    ") has no valid neighboring grid points with nonzero "
                    "weight to coerce to");
            }
            for (auto& v : result) { v /= wSum; }
        }
        return result;
        // NOLINTEND(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
    }

    template <OOBPolicy Policy>
    void SpecsynLib<Policy>::resample(const std::vector<double>& wlNew)
    {
        for (size_t i1 = 0; i1 < dim1_.size(); ++i1) // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- i1, i2, i3 are all < the corresponding grid's size by construction
        {
            for (size_t i2 = 0; i2 < dim2_.size(); ++i2)
            {
                for (size_t i3 = 0; i3 < dim3_.size(); ++i3)
                {
                    auto& spectrum = grid_[i1, i2, i3];
                    if (spectrum.empty()) { continue; } // unpopulated grid point: leave empty

                    spectrum = resample(wl_, wlNew, spectrum);
                }
            }
        }

        wl_ = wlNew;
    }

    template <OOBPolicy Policy>
    auto SpecsynLib<Policy>::resample(const std::vector<double>& wl,
        const std::vector<double>& wlNew, const std::vector<double>& spectrum) -> std::vector<double>
    {
        const interp::Interpolator1D<1> interpolator(wl, spectrum);
        std::vector<double> resampled(wlNew.size(), 0.0);
        for (size_t w = 0; w < wlNew.size(); ++w)
        {
            // Bin edges around wlNew[w], geometric (i.e. log-space)
            // midpoints of wlNew[w] and its neighbors, so that
            // resampled[w] represents the flux averaged over the bin
            // actually surrounding wlNew[w] rather than just its
            // value there -- point-sampling like that (the previous
            // approach) is fine when upsampling, but can badly
            // misrepresent narrow spectral features when downsampling
            // onto a coarser grid, since the new grid can simply miss
            // them. At the two ends of wlNew, where there is no
            // neighbor on one side, the missing edge is instead placed
            // symmetrically in log space around wlNew[w] itself, using
            // the spacing to the one neighbor that does exist.
            const double wl0 = (w == 0) ?
                std::sqrt(wlNew[w] * wlNew[w] * wlNew[w] / wlNew[w + 1]) :
                std::sqrt(wlNew[w - 1] * wlNew[w]);
            const double wl1 = (w == wlNew.size() - 1) ?
                std::sqrt(wlNew[w] * wlNew[w] * wlNew[w] / wlNew[w - 1]) :
                std::sqrt(wlNew[w] * wlNew[w + 1]);

            // Truncate to wl's own range, since the interpolant is
            // only valid there, then integrate over whatever of the
            // bin survives truncation, dividing by its width to
            // recover a flux density rather than a bin-integrated
            // flux; a bin that truncates to nothing (entirely outside
            // wl's range) leaves resampled[w] at the zero it was
            // initialized with. This is not strictly flux-conserving,
            // since the interpolant is itself built by treating wl/
            // spectrum as sample points rather than bin averages
            // (nothing conservative like PPM), but it is much closer
            // than point-sampling, and the library data being
            // resampled here are themselves sample points rather than
            // bin averages, so a conservative interpolation scheme
            // would not actually gain anything.
            const double wl0Trunc = std::max(wl0, wl.front());
            const double wl1Trunc = std::min(wl1, wl.back());
            if (wl1Trunc > wl0Trunc)
            {
                resampled[w] = interpolator.integ(wl0Trunc, wl1Trunc) / (wl1Trunc - wl0Trunc);
            }
            // else leave as the zero flux resampled was initialized with
        }
        return resampled;
    }

    // Explicit instantiation for every OOBPolicy value actually used;
    // this keeps the class's implementation in this .cpp file, as with
    // every other class in src/specsyn, rather than forcing it into
    // the header just because it is a template.
    template class SpecsynLib<OOBPolicy::raise>;
    template class SpecsynLib<OOBPolicy::silent>;
    template class SpecsynLib<OOBPolicy::coerce>;

} // namespace specsyn
