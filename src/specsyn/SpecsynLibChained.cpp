/**
 * @file SpecsynLibChained.cpp
 * @author Mark Krumholz
 * @brief Implementation of SpecsynLibChained.hpp
 * @date 2026-07-21
 */

#include "SpecsynLibChained.hpp"
#include "Specsyn.hpp"
#include "SpecsynCommons.hpp"
#include "SpecsynLib.hpp"
#include "SpecsynLibNoWind.hpp"
// misc-include-cleaner can't attribute std::ranges::lower_bound/upper_bound
// (used below) to this header on some libc++ versions -- see the identical
// NOLINT on SpecsynLib.cpp's own findBracket -- so both the include itself
// and each call site need a NOLINT.
#include <algorithm> // NOLINT(misc-include-cleaner)
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace specsyn
{
    namespace
    {
        /**
         * @brief Count how many points of grid fall within a merge window
         * @param grid A sorted (ascending) wavelength grid
         * @param lo Window's lower bound
         * @param hi Window's upper bound
         * @param lastWindow Whether this is the very last window overall
         * @returns The number of points in [lo, hi) -- or [lo, hi] if
         *   lastWindow, so the very last window includes the overall
         *   maximum wavelength rather than dropping it -- using
         *   half-open windows everywhere else so a wavelength shared
         *   by two adjacent windows (one of the edges themselves) is
         *   only ever counted, and later emitted, once
         */
        auto countPointsInWindow( //NOLINT(llvm-prefer-static-over-anonymous-namespace)
            const std::vector<double>& grid, const double lo, const double hi, const bool lastWindow) -> size_t
        {
            const auto itLo = std::ranges::lower_bound(grid, lo); // NOLINT(misc-include-cleaner)
            const auto itHi = lastWindow ?
                std::ranges::upper_bound(grid, hi) : // NOLINT(misc-include-cleaner)
                std::ranges::lower_bound(grid, hi); // NOLINT(misc-include-cleaner)
            return static_cast<size_t>(itHi - itLo);
        }

        /**
         * @brief Find whichever grid has the most points in a merge window
         * @param wlGrids Every library's own wavelength grid
         * @param lo Window's lower bound
         * @param hi Window's upper bound
         * @param lastWindow Whether this is the very last window overall
         * @param requireFullContainment If true, only consider grids
         *   whose own range fully contains [lo, hi] (front <= lo and
         *   back >= hi); if false, consider every grid
         * @returns The index into wlGrids of whichever considered grid
         *   has the most points in the window, or an empty optional if
         *   none has any
         * @details
         * A grid that merely touches lo or hi -- e.g. one whose own
         * range ends exactly at this window's lo -- has nothing beyond
         * that point, so requiring full containment first keeps such
         * an edge point from spuriously tying against (or even
         * beating) another grid that actually spans the window, which
         * would otherwise clip the merged grid short of that grid's
         * true reach. See makeCommonWlGrid for how the two passes
         * (with and without that requirement) are combined.
         */
        auto findBestGrid( //NOLINT(llvm-prefer-static-over-anonymous-namespace)
            const std::vector<std::vector<double>>& wlGrids,
            const double lo, const double hi, const bool lastWindow,
            const bool requireFullContainment) -> std::optional<size_t>
        {
            bool found = false;
            size_t bestCount = 0;
            size_t bestLib = 0;
            for (size_t i = 0; i < wlGrids.size(); ++i)
            {
                const auto& grid = wlGrids[i];
                if (requireFullContainment && (grid.front() > lo || grid.back() < hi))
                {
                    continue; // doesn't fully cover this window
                }

                const auto count = countPointsInWindow(grid, lo, hi, lastWindow);
                if (count == 0) { continue; } // no actual points in the window
                if (!found || count > bestCount)
                {
                    bestCount = count;
                    bestLib = i;
                    found = true;
                }
            }
            return found ? std::optional<size_t>(bestLib) : std::nullopt;
        }
    } // namespace

    SpecsynLibChained::SpecsynLibChained(
        const std::vector<std::string>& spectraName,
        const double fehMin,
        const double fehMax,
        const double afe,
        const double cfe,
        const std::vector<double>& microTurb,
        const double r,
        const std::string& registryName,
        const double z) :
        Specsyn(z)
    {
        if (spectraName.empty())
        {
            throw std::runtime_error(
                "SpecsynLibChained: spectraName must contain at least one library name");
        }
        if (!microTurb.empty() && microTurb.size() != spectraName.size())
        {
            throw std::runtime_error(
                "SpecsynLibChained: microTurb must be empty or have the same "
                "number of entries as spectraName");
        }

        // Load every library on its own native wavelength grid first.
        // All but the last use OOBPolicy::silent (so a star outside
        // their grid simply falls through to the next library); the
        // last uses OOBPolicy::Throw. Since OOBPolicy is a compile-time
        // template parameter, the silent libraries and the throw
        // library are genuinely different types -- there are only ever
        // two such types in play here, so they are kept in two
        // separate, concretely-typed containers rather than the
        // type-erased libs_ vector, so that resample() (a SpecsynLib
        // method, not part of the polymorphic Specsyn interface) can
        // still be called on each of them below. Each is constructed
        // as a SpecsynLibNoWind -- the only SpecsynLib specialization
        // that exists so far -- and immediately upcast to the
        // SpecsynLib<Policy> it's stored as, since every function this
        // class actually calls on them (resample(), wl(), spec()) lives
        // on that parent. TODO: once SpecsynLibWR exists, let callers
        // pick which specialization each chained library actually is,
        // rather than hardcoding SpecsynLibNoWind for all of them.
        // An empty microTurb means "use each library's own default":
        // pass NaN through to SpecsynLibNoWind for that entry, which
        // resolves it from the library's own micro_default in the
        // registry (see SpecsynLibNoWind's constructor), rather than
        // forcing every library in the chain to share one hardcoded
        // value
        constexpr double useLibraryDefault = std::numeric_limits<double>::quiet_NaN();

        const size_t n = spectraName.size();
        std::vector<std::unique_ptr<SpecsynLib<OOBPolicy::silent>>> silentLibs;
        silentLibs.reserve(n - 1);
        for (size_t i = 0; i + 1 < n; ++i)
        {
            const double mt = microTurb.empty() ? useLibraryDefault : microTurb[i];
            silentLibs.push_back(std::make_unique<SpecsynLibNoWind<OOBPolicy::silent>>(
                spectraName[i], fehMin, fehMax, afe, cfe, mt, r, registryName, z));
        }
        const double lastMt = microTurb.empty() ? useLibraryDefault : microTurb[n - 1];
        std::unique_ptr<SpecsynLib<OOBPolicy::Throw>> throwLib =
            std::make_unique<SpecsynLibNoWind<OOBPolicy::Throw>>(
                spectraName[n - 1], fehMin, fehMax, afe, cfe, lastMt, r, registryName, z);

        // Build a common wavelength grid spanning every library's own
        // native grid, and resample every library onto it, so that
        // every chained library shares the same wl()
        std::vector<std::vector<double>> wlGrids;
        wlGrids.reserve(n);
        for (const auto& lib : silentLibs) { wlGrids.push_back(lib->wl()); }
        wlGrids.push_back(throwLib->wl());

        const auto commonWl = makeCommonWlGrid(wlGrids);
        for (auto& lib : silentLibs) { lib->resample(commonWl); }
        throwLib->resample(commonWl);

        // Move every library, still in priority order, into libs_
        libs_.reserve(n);
        for (auto& lib : silentLibs) { libs_.push_back(std::move(lib)); }
        libs_.push_back(std::move(throwLib));

        wl_ = commonWl;
    }

    auto SpecsynLibChained::spec(const StarData& props, const double feh) const -> std::vector<double>
    {
        for (size_t i = 0; i + 1 < libs_.size(); ++i)
        {
            auto result = libs_[i]->spec(props, feh);
            if (!result.empty()) { return result; }
        }
        return libs_.back()->spec(props, feh);
    }

    auto SpecsynLibChained::makeCommonWlGrid(
        const std::vector<std::vector<double>>& wlGrids) -> std::vector<double>
    {
        if (wlGrids.empty())
        {
            throw std::runtime_error(
                "SpecsynLibChained::makeCommonWlGrid: wlGrids must not be empty");
        }

        // Every grid's minimum and maximum wavelength delineates a
        // window boundary; sorting and deduplicating them gives the
        // full set of windows to fill in
        std::set<double> edgeSet;
        for (const auto& grid : wlGrids)
        {
            if (grid.empty())
            {
                throw std::runtime_error(
                    "SpecsynLibChained::makeCommonWlGrid: every grid in wlGrids "
                    "must be non-empty");
            }
            edgeSet.insert(grid.front());
            edgeSet.insert(grid.back());
        }
        const std::vector<double> edges(edgeSet.begin(), edgeSet.end());

        std::vector<double> result;
        for (size_t w = 0; w + 1 < edges.size(); ++w)
        {
            const double lo = edges[w]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- w, w + 1 < edges.size() by the loop bound
            const double hi = edges[w + 1]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- see above
            const bool lastWindow = (w + 2 == edges.size());

            // Try the fully-containing grids first (see findBestGrid);
            // if none has any actual points here (most often because
            // this window sits between two grids that don't overlap
            // at all, but possibly just because the grid(s) that do
            // span it happen to have no sample here), fall back to
            // whichever grid simply has the most points physically
            // present in the window. This fallback is what lets a
            // grid's own trailing edge point still appear in the
            // result when nothing else reaches anywhere near it (a
            // genuine coverage gap on the other side), while still
            // losing fairly to a grid that actually spans the window
            // when one exists.
            auto best = findBestGrid(wlGrids, lo, hi, lastWindow, true);
            if (!best) { best = findBestGrid(wlGrids, lo, hi, lastWindow, false); }

            // Truly no grid has any point in this window at all -- a
            // genuine coverage gap -- so contribute nothing here
            // rather than inventing samples
            if (!best) { continue; }

            const auto& grid = wlGrids[*best]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- *best < wlGrids.size() by construction
            const auto itLo = std::ranges::lower_bound(grid, lo); // NOLINT(misc-include-cleaner)
            const auto itHi = lastWindow ?
                std::ranges::upper_bound(grid, hi) : // NOLINT(misc-include-cleaner)
                std::ranges::lower_bound(grid, hi); // NOLINT(misc-include-cleaner)
            result.insert(result.end(), itLo, itHi);
        }

        return result;
    }

} // namespace specsyn
