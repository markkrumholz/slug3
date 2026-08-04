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
#include "SpecsynLibWR.hpp"
#include "SpecsynUtils.hpp"
// misc-include-cleaner can't attribute std::ranges::lower_bound/upper_bound
// (used below) to this header on some libc++ versions -- see the identical
// NOLINT on SpecsynLib.cpp's own findBracket -- so both the include itself
// and each call site need a NOLINT.
#include <algorithm> // NOLINT(misc-include-cleaner)
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
         * @brief Construct one chained library, dispatching on WR_grid
         * @tparam Policy OOBPolicy for the constructed library
         * @param name Spectral model name
         * @param isWR Whether this model's registry entry has
         *   WR_grid = true
         * @param fehMin Minimum [Fe/H] value
         * @param fehMax Maximum [Fe/H] value
         * @param afe Value of [alpha/Fe]; ignored for a WR library, which
         *   has no afe axis (see SpecsynLibWR)
         * @param cfe Value of [C/Fe]; ignored for a WR library
         * @param microTurb Microturbulent velocity, in km/s; ignored
         *   for a WR library
         * @param r Spectral resolution; ignored for a WR library
         * @param registryName Name of the spectral library registry file
         * @param wlMin Minimum wavelength of the output grid, in
         *   Angstrom; see SpecsynLibChained's own constructor
         * @param wlMax Maximum wavelength of the output grid, in
         *   Angstrom; see wlMin
         * @param nWl Number of points in the output grid; see wlMin
         * @param controls Simulation controls, forwarded unchanged to
         *   the constructed library's own constructor
         * @returns The constructed library, upcast to SpecsynLib<Policy>
         * @details
         * Wolf-Rayet libraries -- parameterized by transformed radius
         * and stellar temperature rather than logg and Teff -- need
         * SpecsynLibWR; every other library needs SpecsynLibNoWind.
         */
        template <OOBPolicy Policy>
        auto makeChainedLib( //NOLINT(llvm-prefer-static-over-anonymous-namespace)
            const std::string& name, const bool isWR,
            const double fehMin, const double fehMax,
            const double afe, const double cfe, const double microTurb,
            const double r, const std::string& registryName,
            const double wlMin, const double wlMax, const std::size_t nWl,
            const io::SimControls& controls)
        -> std::unique_ptr<SpecsynLib<Policy>>
        {
            if (isWR)
            {
                return std::make_unique<SpecsynLibWR<Policy>>(
                    name, fehMin, fehMax, registryName, wlMin, wlMax, nWl, controls);
            }
            return std::make_unique<SpecsynLibNoWind<Policy>>(
                name, fehMin, fehMax, afe, cfe, microTurb, r, registryName,
                wlMin, wlMax, nWl, controls);
        }

        /**
         * @brief Widen [lo, hi] to also cover one chained library's own logTeff() range
         * @tparam Policy OOBPolicy of lib
         * @param lib A single chained library, as constructed by makeChainedLib
         * @param lo Running global minimum, widened in place
         * @param hi Running global maximum, widened in place
         * @details
         * lib is only ever actually a SpecsynLibNoWind<Policy> or a
         * SpecsynLibWR<Policy> (see makeChainedLib), upcast to the
         * common SpecsynLib<Policy> it's stored as -- neither of which
         * exposes a logTeff() of its own, so this dynamic_casts back
         * down to whichever concrete type lib actually is to reach it.
         */
        template <OOBPolicy Policy>
        void updateLogTeffRange( //NOLINT(llvm-prefer-static-over-anonymous-namespace)
            const SpecsynLib<Policy>& lib, double& lo, double& hi)
        {
            if (const auto* noWind = dynamic_cast<const SpecsynLibNoWind<Policy>*>(&lib))
            {
                lo = std::min(lo, noWind->logTeff().front());
                hi = std::max(hi, noWind->logTeff().back());
            }
            else if (const auto* wr = dynamic_cast<const SpecsynLibWR<Policy>*>(&lib))
            {
                lo = std::min(lo, wr->logTeff().front());
                hi = std::max(hi, wr->logTeff().back());
            }
        }

        /**
         * @brief Widen [lo, hi] to also cover one chained library's own logg() range, if it has one
         * @tparam Policy OOBPolicy of lib
         * @param lib A single chained library, as constructed by makeChainedLib
         * @param lo Running global minimum, widened in place
         * @param hi Running global maximum, widened in place
         * @details
         * Only SpecsynLibNoWind exposes a logg() -- SpecsynLibWR has no
         * logg axis at all, since Wolf-Rayet atmospheres are
         * parameterized by transformed radius instead -- so a lib that
         * dynamic_casts to SpecsynLibWR<Policy> simply leaves [lo, hi]
         * untouched.
         */
        template <OOBPolicy Policy>
        void updateLoggRange( //NOLINT(llvm-prefer-static-over-anonymous-namespace)
            const SpecsynLib<Policy>& lib, double& lo, double& hi)
        {
            if (const auto* noWind = dynamic_cast<const SpecsynLibNoWind<Policy>*>(&lib))
            {
                lo = std::min(lo, noWind->logg().front());
                hi = std::max(hi, noWind->logg().back());
            }
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
        // by makeChainedLib, which picks SpecsynLibWR or
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
        // spectraName, whether it needs SpecsynLibWR (WR_grid = true)
        // or SpecsynLibNoWind (WR_grid absent or false).
        const auto registry = parseRegistry(registryName).first;
        auto isWRGrid = [&registry](const std::string& name) -> bool
        {
            return registry.at_path(name).at_path("WR_grid").value<bool>().value_or(false);
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
                spectraName[i], isWRGrid(spectraName[i]),
                fehMin, fehMax, afe, cfe, mt, r, registryName,
                0.0, 0.0, 0, controls));
        }
        const double lastMt = microTurb.empty() ? useLibraryDefault : microTurb[n - 1];
        std::unique_ptr<SpecsynLib<OOBPolicy::raise>> raiseLib = makeChainedLib<OOBPolicy::raise>(
            spectraName[n - 1], isWRGrid(spectraName[n - 1]),
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

        // Widen [logTeffMin_, logTeffMax_] and [loggMin_, loggMax_]
        // across every chained library's own logTeff()/logg() range,
        // so spec() can clamp a star's log(Teff) (any library) and
        // log(g) (SpecsynLibNoWind libraries only) into whatever this
        // chain actually covers -- see this constructor's own comment
        // for why. Left at their default quiet_NaN() (see the header)
        // if tClamp is false, so spec() simply skips both clamps.
        if (tClamp)
        {
            double loTeff = std::numeric_limits<double>::infinity();
            double hiTeff = -std::numeric_limits<double>::infinity();
            double loLogg = std::numeric_limits<double>::infinity();
            double hiLogg = -std::numeric_limits<double>::infinity();
            for (const auto& lib : coerceLibs)
            {
                updateLogTeffRange(*lib, loTeff, hiTeff);
                updateLoggRange(*lib, loLogg, hiLogg);
            }
            updateLogTeffRange(*raiseLib, loTeff, hiTeff);
            updateLoggRange(*raiseLib, loLogg, hiLogg);
            logTeffMin_ = loTeff;
            logTeffMax_ = hiTeff;
            loggMin_ = loLogg;
            loggMax_ = hiLogg;
        }

        // Move every library, still in priority order, into libs_
        libs_.reserve(n);
        for (auto& lib : coerceLibs) { libs_.push_back(std::move(lib)); }
        libs_.push_back(std::move(raiseLib));
    }

    auto SpecsynLibChained::spec(const StarData& props, const double feh) const -> std::vector<double>
    {
        StarData clampedProps = props;
        if (!std::isnan(logTeffMin_))
        {
            double& logTeff = clampedProps[static_cast<size_t>(tracks::FieldIdx::logTe)]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- StarData is a fixed-size std::array, and logTe is one of its compile-time-known indices
            logTeff = std::clamp(logTeff, logTeffMin_, logTeffMax_);
        }

        // A logg clamp is only meaningful for a non-WR star -- a
        // Wolf-Rayet star isn't placed on a (feh, logg, logTeff) grid
        // at all (see SpecsynLibNoWind), so it has no logg to clamp;
        // SpecsynLibWR::spec() already clamps its own analogous
        // transformed-radius coordinate internally. Unlike the logTeff
        // clamp above, logg isn't a native StarData field -- it's
        // derived from mass, log(L), and log(Teff) via
        // Specsyn::getSAandLogg -- so clamping it means adjusting one
        // of those three instead: mass, specifically, rather than
        // log(L) or log(Teff), since the latter two are exactly what
        // getSAandLogg derives this star's surface area from, and
        // perturbing the surface area would distort the emergent
        // spectrum's overall scale for no physical reason. Since
        // log(g) = log10(G * mass / R^2) and R depends only on log(L)
        // and log(Teff) (both left untouched here), log(g) is exactly
        // linear in log10(mass) with unit slope -- so scaling mass by
        // 10^(loggTarget - logg) lands log(g) at exactly loggTarget,
        // regardless of logg's own magnitude (unlike scaling by the
        // ratio logg / loggTarget directly, which under- or
        // over-corrects whenever |logg| is far from 1). The extra
        // +/- 1e-10 in the exponent pushes the rescaled mass just past
        // the relevant bound, so the log(g) recomputed from it
        // afterwards (inside getSAandLogg, when spec() is called
        // below) doesn't land just outside that bound again due to
        // floating-point roundoff.
        if (!std::isnan(loggMin_) &&
            SpecsynLibWR<OOBPolicy::raise>::getWRType(clampedProps) == SpecsynLibWR<OOBPolicy::raise>::WRType::None)
        {
            const double logg = getSAandLogg(clampedProps).second;
            if (logg < loggMin_ || logg > loggMax_)
            {
                double& mass = clampedProps[static_cast<size_t>(tracks::FieldIdx::mass)]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- StarData is a fixed-size std::array, and mass is one of its compile-time-known indices
                mass *= (logg < loggMin_) ?
                    std::pow(10.0, (loggMin_ - logg) + 1e-10) :
                    std::pow(10.0, (loggMax_ - logg) - 1e-10);
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
