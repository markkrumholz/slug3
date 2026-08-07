/**
 * @file SpecsynLibChained.cpp
 * @author Mark Krumholz
 * @brief Implementation of SpecsynLibChained.hpp
 * @date 2026-07-21
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
                const auto t = static_cast<std::size_t>(GridType::NormalGrid);
                widen(lo[t], hi[t], noWind->logTeff().front(), noWind->logTeff().back()); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- t < gridTypeCount by construction
            }
            else if (const auto* wr = dynamic_cast<const SpecsynLibWR<Policy>*>(&lib))
            {
                const auto t = static_cast<std::size_t>(GridType::WRGrid);
                widen(lo[t], hi[t], wr->logTeff().front(), wr->logTeff().back()); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- see above
            }
            else if (const auto* wd = dynamic_cast<const SpecsynLibWD<Policy>*>(&lib))
            {
                const auto t = static_cast<std::size_t>(GridType::WDGrid);
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
         * leaves lo/hi untouched, and GridType::WRGrid's entry in each
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
                const auto t = static_cast<std::size_t>(GridType::NormalGrid);
                widen(lo[t], hi[t], noWind->logg().front(), noWind->logg().back()); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- t < gridTypeCount by construction
            }
            else if (const auto* wd = dynamic_cast<const SpecsynLibWD<Policy>*>(&lib))
            {
                const auto t = static_cast<std::size_t>(GridType::WDGrid);
                widen(lo[t], hi[t], wd->logg().front(), wd->logg().back()); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- see above
            }
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
         * @returns The GridType whose clamp spec() should apply to props
         * @details
         * A Wolf-Rayet star (per SpecsynLibWR::getWRType) is always
         * GridType::WRGrid, checked first and on props' own raw,
         * unclamped values -- WR-ness is a property of the star's own
         * surface composition (see getWRType), not something that
         * could be affected by any clamp decided afterward.
         *
         * Otherwise, this star is GridType::WDGrid if -- and only if
         * -- both of the following hold on its raw, unclamped log(Teff)
         * (from props directly) and log(g) (the logg parameter):
         *   - it lies above the normal grids' own coverage on at least
         *     one axis (log(Teff) > logTeffMax[NormalGrid], or
         *     log(g) > loggMax[NormalGrid]) -- i.e. it is a real gap in
         *     normal-star coverage, not merely a star the normal grids
         *     already handle;
         *   - it also lies above the WD grids' own floor on *both*
         *     axes (log(Teff) > logTeffMin[WDGrid] and
         *     log(g) > loggMin[WDGrid]) -- i.e. the WD grids can
         *     actually plausibly cover it, rather than just being the
         *     nearest thing to clamp to.
         * Any GridType whose relevant bound is quiet_NaN() (no chained
         * library of that kind) can never satisfy either condition,
         * so a star is never classified into a GridType with no actual
         * grid backing it.
         *
         * Every other star -- including one with no clamp data at all,
         * i.e. tClamp was false -- is GridType::NormalGrid.
         */
        auto classifyGridType( //NOLINT(llvm-prefer-static-over-anonymous-namespace)
            const Specsyn::StarData& props, const double logg,
            const std::array<double, gridTypeCount>& logTeffMin,
            const std::array<double, gridTypeCount>& logTeffMax,
            const std::array<double, gridTypeCount>& loggMin,
            const std::array<double, gridTypeCount>& loggMax) -> GridType
        {
            if (SpecsynLibWR<OOBPolicy::raise>::getWRType(props) !=
                SpecsynLibWR<OOBPolicy::raise>::WRType::None)
            {
                return GridType::WRGrid;
            }

            const auto normal = static_cast<std::size_t>(GridType::NormalGrid);
            const auto wd = static_cast<std::size_t>(GridType::WDGrid);
            // NOLINTBEGIN(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- normal, wd are both < gridTypeCount by construction
            const double logTeff = props[static_cast<std::size_t>(tracks::FieldIdx::logTe)];

            const bool aboveNormal =
                (!std::isnan(logTeffMax[normal]) && logTeff > logTeffMax[normal]) ||
                (!std::isnan(loggMax[normal]) && logg > loggMax[normal]);
            const bool withinWDFloor =
                !std::isnan(logTeffMin[wd]) && logTeff > logTeffMin[wd] &&
                !std::isnan(loggMin[wd]) && logg > loggMin[wd];
            // NOLINTEND(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)

            return (aboveNormal && withinWDFloor) ? GridType::WDGrid : GridType::NormalGrid;
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
        Specsyn(controls)
    {
        // Unconditional, regardless of tClamp: every GridType entry
        // starts (and, for tClamp == false, stays) at quiet_NaN(),
        // rather than relying on a default member initializer, since
        // std::array's own default member-initialization would leave
        // its doubles indeterminate rather than NaN.
        constexpr double unset = std::numeric_limits<double>::quiet_NaN();
        logTeffMin_.fill(unset);
        logTeffMax_.fill(unset);
        loggMin_.fill(unset);
        loggMax_.fill(unset);

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
        // All but the last use OOBPolicy::coerce, so a star outside a
        // library's grid (or in one of its gaps) is still handled by
        // that same library if it has at least one valid neighboring
        // grid point, and falls through to the next library in the
        // chain only if it truly has none; the last uses
        // OOBPolicy::raise, so a star nothing in the chain can cover at
        // all -- not even by coercion -- still produces an error
        // rather than silently vanishing. Since OOBPolicy is a
        // compile-time template parameter, the coerce libraries and the
        // raise library are genuinely different types -- there are only
        // ever two such types in play here, so they are kept in two
        // separate, concretely-typed containers rather than the
        // type-erased libs_ vector, so that resample() (a SpecsynLib
        // method, not part of the polymorphic Specsyn interface) can
        // still be called on each of them below. Each is constructed
        // by makeChainedLib, which picks SpecsynLibWR, SpecsynLibWD, or
        // SpecsynLibNoWind per entry of spectraName (see its own
        // comment), and immediately upcast to the SpecsynLib<Policy>
        // it's stored as, since every function this class actually
        // calls on them (resample(), wl(), spec()) lives on that
        // parent. An empty microTurb means "use each library's own
        // default": pass NaN through to SpecsynLibNoWind for that
        // entry (ignored for a WR entry, which has no microTurb axis
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
        std::vector<std::unique_ptr<SpecsynLib<OOBPolicy::coerce>>> coerceLibs;
        coerceLibs.reserve(n - 1);
        for (size_t i = 0; i + 1 < n; ++i)
        {
            const double mt = microTurb.empty() ? useLibraryDefault : microTurb[i];
            coerceLibs.push_back(makeChainedLib<OOBPolicy::coerce>(
                spectraName[i], isWRGrid(spectraName[i]), isWDGrid(spectraName[i]),
                fehMin, fehMax, afe, cfe, mt, r, registryName,
                0.0, 0.0, 0, controls));
        }
        const double lastMt = microTurb.empty() ? useLibraryDefault : microTurb[n - 1];
        std::unique_ptr<SpecsynLib<OOBPolicy::raise>> raiseLib = makeChainedLib<OOBPolicy::raise>(
            spectraName[n - 1], isWRGrid(spectraName[n - 1]), isWDGrid(spectraName[n - 1]),
            fehMin, fehMax, afe, cfe, lastMt, r, registryName,
            0.0, 0.0, 0, controls);

        // Determine the common wavelength grid every chained library
        // will share, in one of three ways, then resample every
        // library onto it exactly once
        if (wlMin != 0.0)
        {
            // The caller fully specified the output grid -- use it
            // directly rather than deriving one from the individual
            // libraries' own native grids at all
            wl_ = utils::logspace(wlMin, wlMax, nWl);
        }
        else if (nWl != 0)
        {
            // The caller requested a point count but not a range --
            // span the combined native range of every library in the
            // chain at that many points
            double globalWlMin = std::numeric_limits<double>::infinity();
            double globalWlMax = -std::numeric_limits<double>::infinity();
            for (const auto& lib : coerceLibs)
            {
                globalWlMin = std::min(globalWlMin, lib->wl().front());
                globalWlMax = std::max(globalWlMax, lib->wl().back());
            }
            globalWlMin = std::min(globalWlMin, raiseLib->wl().front());
            globalWlMax = std::max(globalWlMax, raiseLib->wl().back());
            wl_ = utils::logspace(globalWlMin, globalWlMax, nWl);
        }
        else
        {
            // No output grid requested at all -- combine every
            // library's own native grid into one that spans them all
            std::vector<std::vector<double>> wlGrids;
            wlGrids.reserve(n);
            for (const auto& lib : coerceLibs) { wlGrids.push_back(lib->wl()); }
            wlGrids.push_back(raiseLib->wl());
            wl_ = makeCommonWlGrid(wlGrids);
        }

        for (auto& lib : coerceLibs) { lib->resample(wl_); }
        raiseLib->resample(wl_);

        // Widen logTeffMin_/logTeffMax_ and loggMin_/loggMax_, per
        // GridType, across every chained library's own logTeff()/
        // logg() range, so spec() can clamp a star's log(Teff) (any
        // library) and log(g) (SpecsynLibNoWind/SpecsynLibWD libraries
        // only) into whatever this chain actually covers -- see this
        // constructor's own comment for why. Left at quiet_NaN() (set
        // just above) for every GridType if tClamp is false, so
        // spec() simply skips both clamps.
        if (tClamp)
        {
            for (const auto& lib : coerceLibs)
            {
                updateLogTeffRange(*lib, logTeffMin_, logTeffMax_);
                updateLoggRange(*lib, loggMin_, loggMax_);
            }
            updateLogTeffRange(*raiseLib, logTeffMin_, logTeffMax_);
            updateLoggRange(*raiseLib, loggMin_, loggMax_);
        }

        // Move every library, still in priority order, into libs_
        libs_.reserve(n);
        for (auto& lib : coerceLibs) { libs_.push_back(std::move(lib)); }
        libs_.push_back(std::move(raiseLib));
    }

    auto SpecsynLibChained::spec(const StarData& props, const double feh) const -> std::vector<double>
    {
        // Classified on props' own raw, unclamped values -- see
        // classifyGridType's own comment -- before any clamping below
        // can change them.
        const double rawLogg = getSAandLogg(props).second;
        const auto type = classifyGridType(props, rawLogg, logTeffMin_, logTeffMax_, loggMin_, loggMax_);
        const auto idx = static_cast<std::size_t>(type);

        StarData clampedProps = props;
        if (!std::isnan(logTeffMin_[idx])) // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- idx < gridTypeCount by construction
        {
            double& logTeff = clampedProps[static_cast<size_t>(tracks::FieldIdx::logTe)]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- StarData is a fixed-size std::array, and logTe is one of its compile-time-known indices
            logTeff = std::clamp(logTeff, logTeffMin_[idx], logTeffMax_[idx]); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- see above
        }

        // A logg clamp is only meaningful for a non-WR star -- a
        // Wolf-Rayet star isn't placed on a (feh, logg, logTeff) grid
        // at all (see SpecsynLibNoWind), so it has no logg to clamp;
        // SpecsynLibWR::spec() already clamps its own analogous
        // transformed-radius coordinate internally. This falls out
        // automatically here, without an explicit WR check, since
        // loggMin_[WRGrid]/loggMax_[WRGrid] are never populated (see
        // updateLoggRange) and so are always quiet_NaN() whenever type
        // is GridType::WRGrid. Unlike the logTeff clamp above, logg
        // isn't a native StarData field -- it's derived from mass,
        // log(L), and log(Teff) via Specsyn::getSAandLogg -- so
        // clamping it means adjusting one of those three instead:
        // mass, specifically, rather than log(L) or log(Teff), since
        // the latter two are exactly what getSAandLogg derives this
        // star's surface area from, and perturbing the surface area
        // would distort the emergent spectrum's overall scale for no
        // physical reason. Since log(g) = log10(G * mass / R^2) and R
        // depends only on log(L) and log(Teff) (the latter already
        // clamped, if at all, just above), log(g) is exactly linear in
        // log10(mass) with unit slope -- so scaling mass by
        // 10^(loggTarget - logg) lands log(g) at exactly loggTarget,
        // regardless of logg's own magnitude (unlike scaling by the
        // ratio logg / loggTarget directly, which under- or
        // over-corrects whenever |logg| is far from 1). The extra
        // +/- 1e-10 in the exponent pushes the rescaled mass just past
        // the relevant bound, so the log(g) recomputed from it
        // afterwards (inside getSAandLogg, when spec() is called
        // below) doesn't land just outside that bound again due to
        // floating-point roundoff.
        if (!std::isnan(loggMin_[idx])) // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- idx < gridTypeCount by construction
        {
            const double logg = getSAandLogg(clampedProps).second;
            const double loggMin = loggMin_[idx]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- see above
            const double loggMax = loggMax_[idx]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- see above
            if (logg < loggMin || logg > loggMax)
            {
                double& mass = clampedProps[static_cast<size_t>(tracks::FieldIdx::mass)]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- StarData is a fixed-size std::array, and mass is one of its compile-time-known indices
                mass *= (logg < loggMin) ?
                    std::pow(10.0, (loggMin - logg) + 1e-10) :
                    std::pow(10.0, (loggMax - logg) - 1e-10);
            }
        }

        for (size_t i = 0; i + 1 < libs_.size(); ++i)
        {
            auto result = libs_[i]->spec(clampedProps, feh);
            if (!result.empty()) { return result; }
        }
        return libs_.back()->spec(clampedProps, feh);
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
