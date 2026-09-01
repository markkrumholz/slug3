/**
 * @file SpecsynLibChained.cpp
 * @author Mark Krumholz
 * @brief Implementation of SpecsynLibChained.hpp
 * @date 2026-07-21
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#include "SpecsynLibChained.hpp"
#include "../io/SimControls.hpp"
#include "../tracks/TrackCommons.hpp"
#include "../utils/MiscUtils.hpp"
#include "Specsyn.hpp"
#include "SpecsynCommons.hpp"
#include "SpecsynLib.hpp"
#include "SpecsynLibNoWind.hpp"
#include "SpecsynLibWD.hpp"
#include "SpecsynLibWR.hpp"
#include "SpecsynUtils.hpp"
// misc-include-cleaner can't attribute std::ranges::lower_bound/upper_bound
// (used below) to this header on some libc++ versions -- see the identical
// NOLINT on SpecsynLib.cpp's own findBracket -- so both the include itself
// and each call site need a NOLINT.
#include <algorithm> // NOLINT(misc-include-cleaner)
#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <toml.hpp>
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

        /**
         * @brief Construct one chained library, dispatching on WR_grid/WD_grid
         * @tparam Policy OOBPolicy for the constructed library
         * @param name Spectral model name
         * @param isWR Whether this model's registry entry has
         *   WR_grid = true
         * @param isWD Whether this model's registry entry has
         *   WD_grid = true; mutually exclusive with isWR
         * @param fehMin Minimum [Fe/H] value; ignored for a WD library,
         *   which has no [Fe/H] axis at all (see SpecsynLibWD)
         * @param fehMax Maximum [Fe/H] value; ignored for a WD library
         * @param afe Value of [alpha/Fe]; ignored for a WR or WD library, which
         *   has no afe axis (see SpecsynLibWR/SpecsynLibWD)
         * @param cfe Value of [C/Fe]; ignored for a WR or WD library
         * @param microTurb Microturbulent velocity, in km/s; ignored
         *   for a WR or WD library
         * @param r Spectral resolution; ignored for a WR or WD library
         * @param registryName Name of the spectral library registry file
         * @param wlMin Minimum wavelength of the output grid, in
         *   Angstrom; see SpecsynLibChained's own constructor
         * @param wlMax Maximum wavelength of the output grid, in
         *   Angstrom; see wlMin
         * @param nWl Number of points in the output grid; see wlMin
         * @param controls Simulation controls, forwarded unchanged to
         *   the constructed library's own constructor
         * @returns The constructed library, upcast to SpecsynLib<Policy>
         * @throws std::runtime_error if isWR and isWD are both true --
         *   a malformed registry entry, since a spectral library can't
         *   be both a Wolf-Rayet grid and a white dwarf grid at once
         * @details
         * Wolf-Rayet libraries -- parameterized by transformed radius
         * and stellar temperature rather than logg and Teff -- need
         * SpecsynLibWR; white dwarf atmosphere grids -- parameterized
         * by logg and Teff alone, with no [Fe/H]/[alpha/Fe]/[C/Fe]/
         * microturbulence/resolution axis at all -- need SpecsynLibWD;
         * every other library needs SpecsynLibNoWind.
         */
        template <OOBPolicy Policy>
        auto makeChainedLib( //NOLINT(llvm-prefer-static-over-anonymous-namespace)
            const std::string& name, const bool isWR, const bool isWD,
            const double fehMin, const double fehMax,
            const double afe, const double cfe, const double microTurb,
            const double r, const std::string& registryName,
            const double wlMin, const double wlMax, const std::size_t nWl,
            const io::SimControls& controls)
        -> std::unique_ptr<SpecsynLib<Policy>>
        {
            if (isWR && isWD)
            {
                throw std::runtime_error(
                    "SpecsynLibChained: spectral model '" + name +
                    "' has both WR_grid and WD_grid set in its registry entry");
            }
            if (isWR)
            {
                return std::make_unique<SpecsynLibWR<Policy>>(
                    name, fehMin, fehMax, registryName, wlMin, wlMax, nWl, controls);
            }
            if (isWD)
            {
                return std::make_unique<SpecsynLibWD<Policy>>(
                    name, registryName, wlMin, wlMax, nWl, controls);
            }
            return std::make_unique<SpecsynLibNoWind<Policy>>(
                name, fehMin, fehMax, afe, cfe, microTurb, r, registryName,
                wlMin, wlMax, nWl, controls);
        }

        // Number of real GridType enumerators -- see SpecsynCommons.hpp
        constexpr auto gridTypeCount = static_cast<std::size_t>(GridType::nGridType);

        /**
         * @brief Widen lo[t]/hi[t] to also cover one chained library's own logTeff() range
         * @tparam Policy OOBPolicy of lib
         * @param lib A single chained library, as constructed by makeChainedLib
         * @param lo Running per-GridType minimum, widened in place at
         *   whichever index t corresponds to lib's own GridType
         * @param hi Running per-GridType maximum, widened in place; see lo
         * @details
         * lib is only ever actually a SpecsynLibNoWind<Policy>, a
         * SpecsynLibWR<Policy>, or a SpecsynLibWD<Policy> (see
         * makeChainedLib), upcast to the common SpecsynLib<Policy> it's
         * stored as -- none of which exposes a logTeff() of its own at
         * that base-class level, so this dynamic_casts back down to
         * whichever concrete type lib actually is to reach it, and
         * uses that to pick which GridType index to widen. lo[t]/hi[t]
         * start at quiet_NaN() (see the constructor) rather than
         * +/-infinity, so the first library of a given GridType simply
         * adopts its own range outright instead of comparing against a
         * NaN that would otherwise never be beaten.
         */
        template <OOBPolicy Policy>
        void updateLogTeffRange( //NOLINT(llvm-prefer-static-over-anonymous-namespace)
            const SpecsynLib<Policy>& lib,
            std::array<double, gridTypeCount>& lo, std::array<double, gridTypeCount>& hi)
        {
            auto widen = [](double& loRef, double& hiRef, const double front, const double back)
            {
                loRef = std::isnan(loRef) ? front : std::min(loRef, front);
                hiRef = std::isnan(hiRef) ? back : std::max(hiRef, back);
            };

            if (const auto* noWind = dynamic_cast<const SpecsynLibNoWind<Policy>*>(&lib))
            {
                const auto t = static_cast<std::size_t>(GridType::normalGrid);
                widen(lo[t], hi[t], noWind->logTeff().front(), noWind->logTeff().back()); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- t < gridTypeCount by construction
            }
            else if (const auto* wr = dynamic_cast<const SpecsynLibWR<Policy>*>(&lib))
            {
                const auto t = static_cast<std::size_t>(GridType::wrGrid);
                widen(lo[t], hi[t], wr->logTeff().front(), wr->logTeff().back()); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- see above
            }
            else if (const auto* wd = dynamic_cast<const SpecsynLibWD<Policy>*>(&lib))
            {
                const auto t = static_cast<std::size_t>(GridType::wdGrid);
                widen(lo[t], hi[t], wd->logTeff().front(), wd->logTeff().back()); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- see above
            }
        }

        /**
         * @brief Widen lo[t]/hi[t] to also cover one chained library's own logg() range, if it has one
         * @tparam Policy OOBPolicy of lib
         * @param lib A single chained library, as constructed by makeChainedLib
         * @param lo Running per-GridType minimum, widened in place at
         *   whichever index t corresponds to lib's own GridType
         * @param hi Running per-GridType maximum, widened in place; see lo
         * @details
         * SpecsynLibNoWind and SpecsynLibWD both expose a logg() --
         * SpecsynLibWR has no logg axis at all, since Wolf-Rayet
         * atmospheres are parameterized by transformed radius instead
         * -- so a lib that dynamic_casts to SpecsynLibWR<Policy> simply
         * leaves lo/hi untouched, and GridType::wrGrid's entry in each
         * stays at quiet_NaN().
         */
        template <OOBPolicy Policy>
        void updateLoggRange( //NOLINT(llvm-prefer-static-over-anonymous-namespace)
            const SpecsynLib<Policy>& lib,
            std::array<double, gridTypeCount>& lo, std::array<double, gridTypeCount>& hi)
        {
            auto widen = [](double& loRef, double& hiRef, const double front, const double back)
            {
                loRef = std::isnan(loRef) ? front : std::min(loRef, front);
                hiRef = std::isnan(hiRef) ? back : std::max(hiRef, back);
            };

            if (const auto* noWind = dynamic_cast<const SpecsynLibNoWind<Policy>*>(&lib))
            {
                const auto t = static_cast<std::size_t>(GridType::normalGrid);
                widen(lo[t], hi[t], noWind->logg().front(), noWind->logg().back()); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- t < gridTypeCount by construction
            }
            else if (const auto* wd = dynamic_cast<const SpecsynLibWD<Policy>*>(&lib))
            {
                const auto t = static_cast<std::size_t>(GridType::wdGrid);
                widen(lo[t], hi[t], wd->logg().front(), wd->logg().back()); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- see above
            }
        }

        /**
         * @brief Widen ranges[i]'s own [min, max] to also cover one chained library's own logTeff() range, if it's a WNL bucket
         * @tparam Policy OOBPolicy of lib
         * @param lib A single chained library, as constructed by makeChainedLib
         * @param ranges Running per-WNL-bucket [min, max], widened in
         *   place at whichever index corresponds to lib's own type(),
         *   if lib is a SpecsynLibWR whose type() is WNLH20/WNLH40/
         *   WNLH60; left untouched for every other kind of library
         *   (including a non-WNL SpecsynLibWR, e.g. WNE or WC, which
         *   getWRType never needs a range for -- see its own comment)
         * @details
         * Mirrors updateLogTeffRange/updateLoggRange's own dynamic_cast
         * pattern, but keyed by SpecsynLibWR::type() rather than
         * GridType, since all three WNL buckets share GridType::wrGrid
         * and so need a finer-grained index than that alone provides.
         */
        template <OOBPolicy Policy>
        void updateWNLTeffRanges( //NOLINT(llvm-prefer-static-over-anonymous-namespace)
            const SpecsynLib<Policy>& lib,
            std::array<std::pair<double, double>, 3>& ranges)
        {
            const auto* wr = dynamic_cast<const SpecsynLibWR<Policy>*>(&lib);
            if (wr == nullptr) { return; }

            const auto type = wr->type();
            if (type != SpecsynLibWR<Policy>::WRType::WNLH20 &&
                type != SpecsynLibWR<Policy>::WRType::WNLH40 &&
                type != SpecsynLibWR<Policy>::WRType::WNLH60)
            {
                return;
            }
            const auto idx = static_cast<std::size_t>(type) -
                static_cast<std::size_t>(SpecsynLibWR<Policy>::WRType::WNLH20);

            auto& [rangeMin, rangeMax] = ranges.at(idx);
            rangeMin = std::isnan(rangeMin) ? wr->logTeff().front() : std::min(rangeMin, wr->logTeff().front());
            rangeMax = std::isnan(rangeMax) ? wr->logTeff().back() : std::max(rangeMax, wr->logTeff().back());
        }

        /**
         * @brief Determine the common wavelength grid the constructor's own chained libraries all get resampled onto
         * @param wlMin See the constructor's own wlMin parameter
         * @param wlMax See the constructor's own wlMax parameter
         * @param nWl See the constructor's own nWl parameter
         * @param allLibs Every chained library, each still on its own
         *   native wavelength grid at this point
         * @return wlMin/wlMax/nWl directly (via utils::logspace), if
         *   wlMin != 0.0; a logspace grid of nWl points spanning
         *   allLibs' own combined native range, if only nWl was given;
         *   or SpecsynLibChained::makeCommonWlGrid() of every library's
         *   own native grid, if neither was given
         * @details
         * Factored out of the constructor purely to keep its own
         * cognitive complexity down -- see
         * SpecsynLibChained::propagateWNLTeffRanges()'s own identical
         * rationale. A free function, rather than a private member
         * like propagateWNLTeffRanges()/updateFeHRanges(), since it
         * needs no access to any SpecsynLibChained member -- only
         * allLibs, passed in directly -- and SpecsynLib<Policy> itself
         * (unlike Specsyn, wrLibs_'s/wdLibs_'s/normalLibs_'s own
         * element type) is only ever visible in this .cpp, not
         * SpecsynLibChained.hpp.
         */
        template <OOBPolicy Policy>
        auto buildChainedWlGrid( //NOLINT(llvm-prefer-static-over-anonymous-namespace)
            const double wlMin, const double wlMax, const std::size_t nWl,
            const std::vector<std::unique_ptr<SpecsynLib<Policy>>>& allLibs) -> std::vector<double>
        {
            if (wlMin != 0.0)
            {
                // The caller fully specified the output grid -- use it
                // directly rather than deriving one from the
                // individual libraries' own native grids at all
                return utils::logspace(wlMin, wlMax, nWl);
            }
            if (nWl != 0)
            {
                // The caller requested a point count but not a range
                // -- span the combined native range of every library
                // in the chain at that many points
                double globalWlMin = std::numeric_limits<double>::infinity();
                double globalWlMax = -std::numeric_limits<double>::infinity();
                for (const auto& lib : allLibs)
                {
                    globalWlMin = std::min(globalWlMin, lib->wl().front());
                    globalWlMax = std::max(globalWlMax, lib->wl().back());
                }
                return utils::logspace(globalWlMin, globalWlMax, nWl);
            }
            // No output grid requested at all -- combine every
            // library's own native grid into one that spans them all
            std::vector<std::vector<double>> wlGrids;
            wlGrids.reserve(allLibs.size());
            for (const auto& lib : allLibs) { wlGrids.push_back(lib->wl()); }
            return SpecsynLibChained::makeCommonWlGrid(wlGrids);
        }

        /**
         * @brief Classify a star into the GridType whose clamp should apply to it
         * @param props Stellar properties to classify
         * @param logg props' own log(g), from Specsyn::getSAandLogg;
         *   passed in rather than computed here since getSAandLogg is
         *   protected on Specsyn, reachable only from a Specsyn (or
         *   derived) member function, not this free function
         * @param logTeffMax Per-GridType log(Teff) maximums (some entries may be quiet_NaN())
         * @param loggMax Per-GridType log(g) maximums; see logTeffMax
         * @param logTeffMin Per-GridType log(Teff) minimums; see logTeffMax
         * @param loggMin Per-GridType log(g) minimums; see logTeffMax
         * @param wnlTeffRanges See SpecsynLibWR::getWRType's own
         *   wnlTeffRanges parameter, which this is simply forwarded to
         * @returns The GridType whose clamp spec() should apply to props
         * @details
         * A Wolf-Rayet star (per SpecsynLibWR::getWRType) is always
         * GridType::wrGrid, checked first and on props' own raw,
         * unclamped values -- WR-ness is a property of the star's own
         * surface composition (and, for a candidate WNL subtype, its
         * log(Teff) actually falling within that subtype's real grid
         * coverage -- see getWRType's own comment), not something that
         * could be affected by any clamp decided afterward.
         *
         * Otherwise, this star is GridType::wdGrid if -- and only if
         * -- both of the following hold on its raw, unclamped log(Teff)
         * (from props directly) and log(g) (the logg parameter):
         *   - it lies above the normal grids' own coverage on at least
         *     one axis (log(Teff) > logTeffMax[normalGrid], or
         *     log(g) > loggMax[normalGrid]) -- i.e. it is a real gap in
         *     normal-star coverage, not merely a star the normal grids
         *     already handle;
         *   - it also lies above the WD grids' own floor on *both*
         *     axes (log(Teff) > logTeffMin[wdGrid] and
         *     log(g) > loggMin[wdGrid]) -- i.e. the WD grids can
         *     actually plausibly cover it, rather than just being the
         *     nearest thing to clamp to.
         * Any GridType whose relevant bound is quiet_NaN() (no chained
         * library of that kind) can never satisfy either condition,
         * so a star is never classified into a GridType with no actual
         * grid backing it.
         *
         * Every other star -- including one with no clamp data at all,
         * i.e. tClamp was false -- is GridType::normalGrid.
         */
        auto classifyGridType( //NOLINT(llvm-prefer-static-over-anonymous-namespace)
            const Specsyn::StarData& props, const double logg,
            const std::array<double, gridTypeCount>& logTeffMin,
            const std::array<double, gridTypeCount>& logTeffMax,
            const std::array<double, gridTypeCount>& loggMin,
            const std::array<double, gridTypeCount>& loggMax,
            const std::array<std::pair<double, double>, 3>& wnlTeffRanges) -> GridType
        {
            if (SpecsynLibWR<OOBPolicy::raise>::getWRType(props, wnlTeffRanges) !=
                SpecsynLibWR<OOBPolicy::raise>::WRType::None)
            {
                return GridType::wrGrid;
            }

            const auto normal = static_cast<std::size_t>(GridType::normalGrid);
            const auto wd = static_cast<std::size_t>(GridType::wdGrid);
            // NOLINTBEGIN(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- normal, wd are both < gridTypeCount by construction
            const double logTeff = props[static_cast<std::size_t>(tracks::FieldIdx::logTe)];

            const bool aboveNormal =
                (!std::isnan(logTeffMax[normal]) && logTeff > logTeffMax[normal]) ||
                (!std::isnan(loggMax[normal]) && logg > loggMax[normal]);
            const bool withinWDFloor =
                !std::isnan(logTeffMin[wd]) && logTeff > logTeffMin[wd] &&
                !std::isnan(loggMin[wd]) && logg > loggMin[wd];
            // NOLINTEND(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)

            return (aboveNormal && withinWDFloor) ? GridType::wdGrid : GridType::normalGrid;
        }

        // std::array's own default member-initialization would leave its
        // doubles indeterminate rather than NaN, so every GridType-indexed
        // range member is instead constructed with this filled-array
        // helper in the constructor's own initializer list below --
        // satisfies cppcoreguidelines-pro-type-member-init while keeping
        // every entry at quiet_NaN() until updateLogTeffRange/
        // updateLoggRange (or nothing, if tClamp is false) overwrite it.
        template <std::size_t N>
        constexpr auto filledArray(double value) -> std::array<double, N>
        {
            std::array<double, N> arr{};
            for (double& x : arr)
            {
                x = value;
            }
            return arr;
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
        const double wlMin,
        const double wlMax,
        const std::size_t nWl,
        const bool tClamp,
        const io::SimControls& controls) :
        Specsyn(controls),
        // Unconditional, regardless of tClamp: every GridType entry
        // starts (and, for tClamp == false, stays) at quiet_NaN() --
        // see filledArray's own comment for why this can't just be a
        // default member initializer or a fill() call in the body.
        logTeffMin_(filledArray<nGridType>(std::numeric_limits<double>::quiet_NaN())),
        logTeffMax_(filledArray<nGridType>(std::numeric_limits<double>::quiet_NaN())),
        loggMin_(filledArray<nGridType>(std::numeric_limits<double>::quiet_NaN())),
        loggMax_(filledArray<nGridType>(std::numeric_limits<double>::quiet_NaN())),
        wnlTeffRanges_({{
            {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()},
            {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()},
            {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()},
        }})
        // fehMin_/fehMax_/loggLibMin_/loggLibMax_/libNames_ each
        // default-construct to nGridType empty vectors, which is
        // exactly what they need to start as -- updateFeHRanges()/
        // updateLoggRanges() (via the sort loop just before them)
        // resize and fill each GridType's own vectors together, since
        // their eventual length isn't known until then. See fehMin_'s
        // own comment.
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

        // Load every library on its own native wavelength grid first,
        // all with OOBPolicy::coerce -- see the constructor's own
        // comment for why there is no longer a distinct raise-policy
        // library at the chain's end. Each is constructed by
        // makeChainedLib, which picks SpecsynLibWR, SpecsynLibWD, or
        // SpecsynLibNoWind per entry of spectraName (see its own
        // comment), and immediately upcast to the
        // SpecsynLib<OOBPolicy::coerce> it's stored as, since every
        // function needed on it below -- resample(), wl() -- lives on
        // that parent; allTypes tracks each entry's own GridType
        // alongside it, so the final sort into wrLibs_/wdLibs_/
        // normalLibs_ doesn't need to re-derive it via dynamic_cast.
        // An empty microTurb means "use each library's own default":
        // pass NaN through to SpecsynLibNoWind for that entry (ignored
        // for a WR or WD entry, neither of which has a microTurb axis
        // at all), which resolves it from the library's own
        // micro_default in the registry (see SpecsynLibNoWind's
        // constructor), rather than forcing every library in the
        // chain to share one hardcoded value
        constexpr double useLibraryDefault = std::numeric_limits<double>::quiet_NaN();

        // Mirrors SimControls::readSpectra's own WR_grid check: a
        // single parse of the registry tells us, for each entry of
        // spectraName, whether it needs SpecsynLibWR (WR_grid = true),
        // SpecsynLibWD (WD_grid = true), or SpecsynLibNoWind (neither
        // flag set).
        const auto registry = parseRegistry(registryName).first;
        auto isWRGrid = [&registry](const std::string& name) -> bool
        {
            return registry.at_path(name).at_path("WR_grid").value<bool>().value_or(false);
        };
        auto isWDGrid = [&registry](const std::string& name) -> bool
        {
            return registry.at_path(name).at_path("WD_grid").value<bool>().value_or(false);
        };

        // Each individual library is constructed on its own native
        // wavelength grid (0/0/0 for wlMin/wlMax/nWl), regardless of
        // what the caller passed to this constructor -- the requested
        // output grid, if any, is instead built and applied once below,
        // after every library exists, to resample each of them exactly
        // once. Passing the caller's wlMin/wlMax/nWl through here too
        // would resample every library twice over: once to the
        // caller's grid (or, if only nWl was given, to that many points
        // over the library's own native range) here, and then again to
        // wl_ below.
        const size_t n = spectraName.size();
        std::vector<std::unique_ptr<SpecsynLib<OOBPolicy::coerce>>> allLibs;
        std::vector<GridType> allTypes;
        allLibs.reserve(n);
        allTypes.reserve(n);
        for (size_t i = 0; i < n; ++i)
        {
            const double mt = microTurb.empty() ? useLibraryDefault : microTurb[i];
            const bool isWR = isWRGrid(spectraName[i]);
            const bool isWD = isWDGrid(spectraName[i]);
            allLibs.push_back(makeChainedLib<OOBPolicy::coerce>(
                spectraName[i], isWR, isWD,
                fehMin, fehMax, afe, cfe, mt, r, registryName,
                0.0, 0.0, 0, controls));
            GridType type = GridType::normalGrid;
            if (isWR) { type = GridType::wrGrid; }
            else if (isWD) { type = GridType::wdGrid; }
            allTypes.push_back(type);
        }

        // Determine the common wavelength grid every chained library
        // will share, then resample every library onto it exactly
        // once -- see buildChainedWlGrid's own comment for the three
        // ways this grid can be determined
        wl_ = buildChainedWlGrid<OOBPolicy::coerce>(wlMin, wlMax, nWl, allLibs);
        for (auto& lib : allLibs) { lib->resample(wl_); }

        // Widen logTeffMin_/logTeffMax_ and loggMin_/loggMax_, per
        // GridType, across every chained library's own logTeff()/
        // logg() range, so classifyGridType (called from spec()) can
        // decide which chain a star belongs to -- see this
        // constructor's own comment for why. Left at quiet_NaN() (set
        // just above) for every GridType if tClamp is false.
        if (tClamp)
        {
            for (const auto& lib : allLibs)
            {
                updateLogTeffRange(*lib, logTeffMin_, logTeffMax_);
                updateLoggRange(*lib, loggMin_, loggMax_);
            }
        }

        // Widen wnlTeffRanges_ across every chained SpecsynLibWR
        // library's own type()/logTeff() range, unconditionally
        // (regardless of tClamp) -- see wnlTeffRanges_'s own comment
        // for why this isn't an optional clamp the way the arrays
        // above are.
        for (const auto& lib : allLibs) { updateWNLTeffRanges(*lib, wnlTeffRanges_); }

        // Sort every library, still in its own original relative
        // order within spectraName, into whichever of wrLibs_/wdLibs_/
        // normalLibs_ matches its own already-known GridType, carrying
        // its own spectraName entry along into the matching libNames_
        // bucket in the same pass -- see libNames_'s own comment for
        // why.
        wrLibs_.reserve(n);
        wdLibs_.reserve(n);
        normalLibs_.reserve(n);
        for (size_t i = 0; i < n; ++i)
        {
            const auto ti = static_cast<std::size_t>(allTypes[i]); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- i < n == allTypes.size() by construction
            libNames_[ti].push_back(spectraName[i]); // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- ti < nGridType by construction
            switch (allTypes[i]) // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- see above
            {
                case GridType::wrGrid: wrLibs_.push_back(std::move(allLibs[i])); break; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- see above
                case GridType::wdGrid: wdLibs_.push_back(std::move(allLibs[i])); break; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- see above
                default: normalLibs_.push_back(std::move(allLibs[i])); break; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- see above
            }
        }

        // Hand the combined wnlTeffRanges_ down to every chained
        // SpecsynLibWR library, so its own spec() classifies stars via
        // getWRType exactly as classifyGridType does -- see
        // wnlTeffRanges_'s own comment for why the two must agree.
        propagateWNLTeffRanges();

        // fehMin_/fehMax_/libNames_, per individual chained library --
        // unconditionally (regardless of tClamp, like wnlTeffRanges_
        // above) -- see updateFeHRanges()'s own comment for exactly
        // what this computes and why.
        updateFeHRanges();

        // loggLibMin_/loggLibMax_, per individual chained library --
        // see updateLoggRanges()'s own comment for exactly what this
        // computes and why.
        updateLoggRanges();

        // Warn (once, up front) for every chained library whose own
        // [Fe/H] coverage the requested range exceeds -- see
        // warnIfFeHClamped()'s own comment.
        warnIfFeHClamped(fehMin, fehMax);
    }

    void SpecsynLibChained::propagateWNLTeffRanges()
    {
        for (const auto& lib : wrLibs_)
        {
            if (auto* wr = dynamic_cast<SpecsynLibWR<OOBPolicy::coerce>*>(lib.get()))
            {
                wr->setWNLTeffRanges(wnlTeffRanges_);
            }
        }
    }

    void SpecsynLibChained::updateFeHRanges()
    {
        // Each GridType's own three vectors (fehMin_[t]/fehMax_[t] --
        // libNames_[t] is already filled, by the sort loop just above
        // this call in the constructor) are sized to chainFor(t)'s own
        // length and filled one entry per chained library, in the same
        // order. An entry is left at quiet_NaN() if that specific
        // library has no [Fe/H] axis at all (SpecsynLibWD; its
        // base-class fehMin()/fehMax() are the unrestricted
        // -infinity/+infinity, not copied through, so downstream
        // isnan() checks treat it the same as "no restriction").
        for (const auto t : { GridType::wrGrid, GridType::wdGrid, GridType::normalGrid })
        {
            const auto& chain = chainFor(t);
            const auto ti = static_cast<std::size_t>(t);
            fehMin_[ti].assign(chain.size(), std::numeric_limits<double>::quiet_NaN()); // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- ti < nGridType by construction
            fehMax_[ti].assign(chain.size(), std::numeric_limits<double>::quiet_NaN()); // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- see above
            for (std::size_t i = 0; i < chain.size(); ++i)
            {
                const double lo = chain[i]->fehMin();
                const double hi = chain[i]->fehMax();
                if (std::isfinite(lo)) { fehMin_[ti][i] = lo; } // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- see above
                if (std::isfinite(hi)) { fehMax_[ti][i] = hi; } // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- see above
            }
        }
    }

    void SpecsynLibChained::updateLoggRanges()
    {
        // Mirrors updateFeHRanges() exactly -- see its own comment --
        // but reads loggMin()/loggMax() instead of fehMin()/fehMax():
        // an entry is left at quiet_NaN() if that specific library has
        // no log(g) axis at all (any SpecsynLibWR).
        for (const auto t : { GridType::wrGrid, GridType::wdGrid, GridType::normalGrid })
        {
            const auto& chain = chainFor(t);
            const auto ti = static_cast<std::size_t>(t);
            loggLibMin_[ti].assign(chain.size(), std::numeric_limits<double>::quiet_NaN()); // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- ti < nGridType by construction
            loggLibMax_[ti].assign(chain.size(), std::numeric_limits<double>::quiet_NaN()); // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- see above
            for (std::size_t i = 0; i < chain.size(); ++i)
            {
                const double lo = chain[i]->loggMin();
                const double hi = chain[i]->loggMax();
                if (std::isfinite(lo)) { loggLibMin_[ti][i] = lo; } // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- see above
                if (std::isfinite(hi)) { loggLibMax_[ti][i] = hi; } // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- see above
            }
        }
    }

    auto SpecsynLibChained::clampFehForLibrary(
        const double feh, const double logg, const GridType type, const std::size_t libIdx) const
        -> std::pair<double, double>
    {
        const auto ti = static_cast<std::size_t>(type);

        double clampedFeh = feh;
        const double fehLo = fehMin_[ti][libIdx]; // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- ti < nGridType, libIdx < chainFor(type).size() by construction
        const double fehHi = fehMax_[ti][libIdx]; // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- see above
        if (!std::isnan(fehLo)) { clampedFeh = std::max(clampedFeh, fehLo); }
        if (!std::isnan(fehHi)) { clampedFeh = std::min(clampedFeh, fehHi); }

        double clampedLogg = logg;
        const double loggLo = loggLibMin_[ti][libIdx]; // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- see above
        const double loggHi = loggLibMax_[ti][libIdx]; // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- see above
        if (!std::isnan(loggLo)) { clampedLogg = std::max(clampedLogg, loggLo); }
        if (!std::isnan(loggHi)) { clampedLogg = std::min(clampedLogg, loggHi); }

        return { clampedFeh, clampedLogg };
    }

    auto SpecsynLibChained::propsWithClampedLogg(
        StarData props, const double currentLogg, const double clampedLogg) -> StarData
    {
        if (clampedLogg != currentLogg)
        {
            // See this function's own header comment for why this nudge
            // is needed at all: +eps when clamping up to a floor (so the
            // round-tripped log(g) lands safely at/above it), -eps when
            // clamping down to a ceiling (so it lands safely at/below it).
            constexpr double loggClampEps = 1e-6;
            const double sign = (clampedLogg > currentLogg) ? 1.0 : -1.0;
            const auto massIdx = static_cast<std::size_t>(tracks::FieldIdx::mass);
            props[massIdx] *= std::pow(10.0, clampedLogg - currentLogg) *
                (1.0 + (sign * loggClampEps)); // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- massIdx < StarData::size() by construction
        }
        return props;
    }

    void SpecsynLibChained::warnIfFeHClamped(const double fehMin, const double fehMax) const
    {
        for (const auto t : { GridType::wrGrid, GridType::wdGrid, GridType::normalGrid })
        {
            const auto ti = static_cast<std::size_t>(t);
            const auto& mins = fehMin_[ti]; // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- ti < nGridType by construction
            const auto& maxs = fehMax_[ti]; // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- see above
            const auto& names = libNames_[ti]; // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- see above
            for (std::size_t i = 0; i < mins.size(); ++i)
            {
                if (std::isnan(mins[i])) { continue; } // no [Fe/H] axis on this library
                if (fehMin < mins[i] || fehMax > maxs[i])
                {
                    std::cout << "slug: warning: chained spectral library " << names[i] <<
                        " covers only [Fe/H] = [" << mins[i] << ", " << maxs[i] <<
                        "], narrower than the requested [Fe/H] = [" << fehMin << ", " << fehMax <<
                        "]; stars outside that range will be clamped to " << names[i] <<
                        "'s nearest available [Fe/H] when it is tried\n";
                }
            }
        }
    }

    auto SpecsynLibChained::chainFor(const GridType type) const -> const std::vector<std::unique_ptr<Specsyn>>&
    {
        switch (type)
        {
            case GridType::wrGrid: return wrLibs_;
            case GridType::wdGrid: return wdLibs_;
            default: return normalLibs_;
        }
    }

    auto SpecsynLibChained::spec(const StarData& props, const double feh) const -> std::vector<double>
    {
        const double rawLogg = getSAandLogg(props).second;
        const auto type = classifyGridType(props, rawLogg, logTeffMin_, logTeffMax_, loggMin_, loggMax_, wnlTeffRanges_);
        const auto& chain = chainFor(type);

        if (chain.empty())
        {
            throw std::runtime_error(
                "SpecsynLibChained: no chained library of the required type "
                "is available for this star");
        }

        // Normal-grid stars specifically are also clamped up to the
        // chain's own combined lower log(Teff) floor, if they fall
        // below it -- see clampNormalLogTeffFloor()'s own comment.
        const StarData queryProps = clampNormalLogTeffFloor(props, type);
        const double queryLogg = getSAandLogg(queryProps).second;

        // First pass: try every library in the chain, in priority
        // order, at the star's true (unclamped) feh and log(g). Chain
        // order encodes physical reliability, not just [Fe/H]/log(g)
        // coverage (e.g. TLUSTY's NLTE hot-star models are preferred
        // over CK04's older, coarser physics even where both cover the
        // same star) -- so a library later in the chain that happens
        // to cover the true feh/log(g) natively must not preempt an
        // earlier, more reliable library that also covers it.
        for (const auto& lib : chain)
        {
            auto result = lib->spec(queryProps, feh);
            if (!result.empty()) { return result; }
        }

        // Second pass, only reached if no library in the chain covers
        // the true feh/log(g) at all: retry each library with feh and
        // log(g) clamped to *that library's own* real coverage -- see
        // this function's own header comment for why per-library, not
        // chain-wide. No atmosphere library actually has uniform
        // coverage across its own kind of chain in practice (a hot O/B
        // star metal-poor enough to fall outside TLUSTY_O/TLUSTY_B's
        // own narrower [Fe/H] range needs the same kind of per-library
        // rescue a WR star metal-poor enough to fall outside the
        // chained WR library's own range already did; a very
        // metal-poor massive MIST star compact enough to exceed every
        // hot-atmosphere library's own log(g) ceiling needs the same
        // kind of rescue on that axis instead). A clamped-but-
        // otherwise-preferred library is still tried before an
        // unclamped-but-less-reliable one, since a modest clamp from a
        // physically better library can outperform an exact match from
        // a worse one -- whether that tradeoff is actually favorable
        // depends on how far the clamp reaches, which this two-pass
        // approach does not attempt to weigh. log(g) is clamped by
        // rescaling queryProps' own mass -- see propsWithClampedLogg()'s
        // own comment for why that, uniquely among props' fields, moves
        // log(g) alone, leaving the star's real luminosity and
        // temperature (and so its spectrum's own physical shape and
        // amplitude) untouched.
        for (std::size_t i = 0; i < chain.size(); ++i)
        {
            const auto [clampedFeh, clampedLogg] = clampFehForLibrary(feh, queryLogg, type, i);
            const StarData libProps = propsWithClampedLogg(queryProps, queryLogg, clampedLogg);
            auto result = chain[i]->spec(libProps, clampedFeh);
            if (!result.empty()) { return result; }
        }
        const auto [finalFeh, finalLogg] = clampFehForLibrary(feh, queryLogg, type, chain.size() - 1);
        const StarData finalProps = propsWithClampedLogg(queryProps, queryLogg, finalLogg);
        return chain.back()->specForce(finalProps, finalFeh);
    }

    auto SpecsynLibChained::specForIntegration(const StarData& props, const double feh) const -> std::vector<double>
    {
        return spec(props, feh);
    }

    auto SpecsynLibChained::clampNormalLogTeffFloor(StarData props, const GridType type) const -> StarData
    {
        if (type == GridType::normalGrid)
        {
            const auto t = static_cast<std::size_t>(type);
            const auto logTeIdx = static_cast<std::size_t>(tracks::FieldIdx::logTe);
            // NOLINTBEGIN(cppcoreguidelines-pro-bounds-constant-array-index) -- t < gridTypeCount, logTeIdx < StarData::size() by construction
            if (!std::isnan(logTeffMin_[t]) && props[logTeIdx] < logTeffMin_[t])
            {
                props[logTeIdx] = logTeffMin_[t];
            }
            // NOLINTEND(cppcoreguidelines-pro-bounds-constant-array-index)
        }
        return props;
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
