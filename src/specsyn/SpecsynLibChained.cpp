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
#include <algorithm>
#include <cstddef>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace specsyn
{
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
        // still be called on each of them below.
        const size_t n = spectraName.size();
        std::vector<std::unique_ptr<SpecsynLib<OOBPolicy::silent>>> silentLibs;
        silentLibs.reserve(n - 1);
        for (size_t i = 0; i + 1 < n; ++i)
        {
            const double mt = microTurb.empty() ? defaultMicroTurb : microTurb[i];
            silentLibs.push_back(std::make_unique<SpecsynLib<OOBPolicy::silent>>(
                spectraName[i], fehMin, fehMax, afe, cfe, mt, r, registryName, z));
        }
        const double lastMt = microTurb.empty() ? defaultMicroTurb : microTurb[n - 1];
        auto throwLib = std::make_unique<SpecsynLib<OOBPolicy::Throw>>(
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

        // Count how many points grid i has within [lo, hi) -- or
        // [lo, hi] if lastWindow, so the very last window includes
        // the overall maximum wavelength rather than dropping it --
        // using half-open windows everywhere else so a wavelength
        // shared by two adjacent windows (one of the edges
        // themselves) is only ever counted, and later emitted, once.
        const auto countIn = [](const std::vector<double>& grid,
            const double lo, const double hi, const bool lastWindow) -> size_t
        {
            const auto itLo = std::ranges::lower_bound(grid, lo);
            const auto itHi = lastWindow ?
                std::ranges::upper_bound(grid, hi) : std::ranges::lower_bound(grid, hi);
            return static_cast<size_t>(itHi - itLo);
        };

        std::vector<double> result;
        for (size_t w = 0; w + 1 < edges.size(); ++w)
        {
            const double lo = edges[w]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- w, w + 1 < edges.size() by the loop bound
            const double hi = edges[w + 1]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- see above
            const bool lastWindow = (w + 2 == edges.size());

            // First, restrict to grids whose own range fully contains
            // this window (front <= lo and back >= hi), and pick
            // whichever of those has the most points here. A grid
            // that merely touches lo or hi -- e.g. one whose own
            // range ends exactly at this window's lo -- has nothing
            // beyond that point, so it isn't a genuine owner of the
            // window even though its own edge point technically falls
            // inside it; without this restriction such an edge point
            // can spuriously tie against (or even beat) another grid
            // that actually spans the window, clipping the merged
            // grid short of that grid's true reach.
            bool found = false;
            size_t bestCount = 0;
            size_t bestLib = 0;
            for (size_t i = 0; i < wlGrids.size(); ++i)
            {
                const auto& grid = wlGrids[i];
                if (grid.front() > lo || grid.back() < hi) { continue; } // doesn't fully cover this window

                const auto count = countIn(grid, lo, hi, lastWindow);
                if (count == 0) { continue; } // covers the window but has no actual points in it
                if (!found || count > bestCount)
                {
                    bestCount = count;
                    bestLib = i;
                    found = true;
                }
            }

            // No grid fully spans this window with actual points in
            // it -- most often because it sits between two grids that
            // don't overlap at all, but possibly just because the
            // grid(s) that do span it happen to have no sample here --
            // so fall back to whichever grid simply has the most
            // points physically present in the window, regardless of
            // whether its own range fully contains it. This is what
            // lets a grid's own trailing edge point still appear in
            // the result when nothing else reaches anywhere near it
            // (a genuine coverage gap on the other side), while still
            // losing fairly to a grid that actually spans the window
            // when one exists.
            if (!found)
            {
                for (size_t i = 0; i < wlGrids.size(); ++i)
                {
                    const auto count = countIn(wlGrids[i], lo, hi, lastWindow);
                    if (count == 0) { continue; }
                    if (!found || count > bestCount)
                    {
                        bestCount = count;
                        bestLib = i;
                        found = true;
                    }
                }
            }

            // Truly no grid has any point in this window at all -- a
            // genuine coverage gap -- so contribute nothing here
            // rather than inventing samples
            if (!found) { continue; }

            const auto& grid = wlGrids[bestLib]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- bestLib < wlGrids.size() by construction
            const auto itLo = std::ranges::lower_bound(grid, lo);
            const auto itHi = lastWindow ?
                std::ranges::upper_bound(grid, hi) : std::ranges::lower_bound(grid, hi);
            result.insert(result.end(), itLo, itHi);
        }

        return result;
    }

} // namespace specsyn
