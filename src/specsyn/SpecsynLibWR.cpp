/**
 * @file SpecsynLibWR.cpp
 * @author Mark Krumholz
 * @brief Implementation of SpecsynLibWR.hpp
 * @date 2026-07-22
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#include "SpecsynLibWR.hpp"
#include "../io/SimControls.hpp"
#include "../tracks/TrackCommons.hpp"
#include "../utils/Constants.hpp"
#include "../utils/HDF5Utils.hpp"
#include "../utils/MiscUtils.hpp"
#include "Specsyn.hpp"
#include "SpecsynCommons.hpp"
#include "SpecsynLib.hpp"
#include "SpecsynLibChained.hpp"
#include "SpecsynUtils.hpp"
#include "hdf5.h" // NOLINT(misc-include-cleaner)
#include <algorithm> // NOLINT(misc-include-cleaner) -- see the identical NOLINT on SpecsynLib.cpp's findBracket for why std::ranges::lower_bound needs this
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <limits>
#include <mdspan> // NOLINT(misc-include-cleaner)
#include <numbers>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace specsyn
{
    namespace
    {
        /**
         * @brief A bracketing pair of grid indices, plus an interpolation weight
         * @details
         * Identical in purpose to SpecsynLib.cpp's own (private, so not
         * reusable from here) Bracket/findBracket -- lo_ and hi_ are the
         * indices of the grid points immediately below and above (or
         * equal to) a query value, and t_ is the fractional distance of
         * the query value between them, so that
         * (1 - t_) * grid[lo_] + t_ * grid[hi_] recovers the query
         * value. For a grid of size 1, lo_ == hi_ == 0 and t_ == 0.
         */
        struct Bracket
        {
            size_t lo_;
            size_t hi_;
            double t_;
        };

        /**
         * @brief Find the bracketing grid points of a sorted grid
         * @param grid A sorted (ascending), non-empty grid of values
         * @param value The query value; clamped to [grid.front(),
         *   grid.back()] if it falls outside that range, rather than
         *   extrapolated
         * @returns The bracketing Bracket for value
         */
        auto findBracket(const std::vector<double>& grid, const double value) //NOLINT(llvm-prefer-static-over-anonymous-namespace)
            -> Bracket
        {
            const size_t n = grid.size();
            if (n == 1) { return { .lo_ = 0, .hi_ = 0, .t_ = 0.0 }; }

            const auto it = std::ranges::upper_bound(grid, value); //NOLINT(misc-include-cleaner) -- some libc++ versions can't find a header to attribute std::ranges::upper_bound to, even with <algorithm> already included
            size_t hi = (it == grid.end()) ?
                (n - 1) : static_cast<size_t>(it - grid.begin());
            if (hi == 0) { hi = 1; } // value == grid.front(): use the first interval
            const size_t lo = hi - 1;
            const double t = std::clamp(
                (value - grid[lo]) / (grid[hi] - grid[lo]), 0.0, 1.0); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- lo, hi < n by construction
            return { .lo_ = lo, .hi_ = hi, .t_ = t };
        }

        /**
         * @brief Trilinearly interpolate a scalar tensor grid at a bracketed point
         * @param grid The scalar grid to interpolate on (e.g. logLGrid_)
         * @param b1 Bracket along the grid's first axis
         * @param b2 Bracket along the grid's second axis
         * @param b3 Bracket along the grid's third axis
         * @returns The interpolated scalar value
         * @details
         * logLGrid_ is only ever populated at exactly the same points
         * as SpecsynLib::spectra_ (see SpecsynLibWR's constructor), and
         * unpopulated points hold quiet_NaN() (see logL_'s own
         * initialization), so an unpopulated corner is detected here
         * directly via std::isnan rather than needing its own
         * populated/unpopulated tracking. Under OOBPolicy::raise or
         * ::silent this never actually matters -- a caller only
         * reaches this function after confirming the spectrum
         * interpolation at this same point already succeeded, which
         * for those policies means every corner is populated -- but
         * under ::coerce, spec() can succeed using only a subset of
         * the 8 corners (see SpecsynLib::spec()'s own coerce handling),
         * so this mirrors that same skip-and-renormalize logic: an
         * earlier version of this function assumed every corner used
         * here was always populated, which was true before ::coerce
         * could actually succeed on a genuinely incomplete cell, but
         * silently blended in an unpopulated corner's NaN (via a
         * nonzero weight) into the result once it could.
         */
        auto trilinearScalar( //NOLINT(llvm-prefer-static-over-anonymous-namespace, readability-function-cognitive-complexity) -- see the identical NOLINT on SpecsynLib.cpp's own spec(double, double, double), whose nested trilinear-interpolation loop this mirrors
            const std::mdspan<double, std::dextents<std::size_t, 3>>& grid, // NOLINT(misc-include-cleaner) -- see the identical NOLINT on SpecsynLib.hpp's SpectraGrid alias
            const Bracket& b1, const Bracket& b2, const Bracket& b3) -> double
        {
            double result = 0.0;
            double wSum = 0.0;
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

                        const double value = grid[i1, i2, i3]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- i1, i2, i3 are all < the corresponding grid's size by construction
                        if (std::isnan(value)) { continue; } // unpopulated corner

                        wSum += weight;
                        result += weight * value;
                    }
                }
            }
            return result / wSum;
        }

        /**
         * @brief Find the (log(R_t), log(Teff)) grid indices, at a given feh index, with a populated spectrum closest to a target point
         * @param grid The spectrum tensor grid (SpecsynLibWR::grid_)
         * @param logRt The log(R_t) values spanned by grid's own second axis
         * @param logTeff The log(Teff) values spanned by grid's own third axis
         * @param f Index into grid's own first (feh) axis to search at
         * @param targetLogRt The log(R_t) value to search for the nearest match to
         * @param targetLogTeff The log(Teff) value to search for the nearest match to
         * @return The (r, t) indices whose grid[f, r, t] is populated
         *   and closest to (targetLogRt, targetLogTeff) -- by ordinary
         *   Euclidean distance in (log(R_t), log(Teff)) space -- or
         *   std::nullopt if grid[f, ., .] is unpopulated for every
         *   (r, t)
         * @details
         * Factored out of SpecsynLibWR::specForce() purely to keep its
         * own cognitive complexity down -- see its own comment for how
         * this is used.
         */
        auto nearestPopulatedRtTeff( //NOLINT(llvm-prefer-static-over-anonymous-namespace)
            const std::mdspan<std::vector<double>, std::dextents<std::size_t, 3>>& grid, // NOLINT(misc-include-cleaner) -- see the identical NOLINT on SpecsynLib.hpp's SpectraGrid alias
            const std::vector<double>& logRt, const std::vector<double>& logTeff,
            const size_t f, const double targetLogRt, const double targetLogTeff)
            -> std::optional<std::pair<size_t, size_t>>
        {
            bool found = false;
            double bestDist = std::numeric_limits<double>::infinity();
            size_t bestR = 0;
            size_t bestT = 0;
            for (size_t r = 0; r < logRt.size(); ++r)
            {
                for (size_t t = 0; t < logTeff.size(); ++t)
                {
                    if (grid[f, r, t].empty()) { continue; } // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- f, r, t all < the corresponding axis's size by construction
                    const double dr = logRt[r] - targetLogRt; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- r < logRt.size() by the loop bound
                    const double dt = logTeff[t] - targetLogTeff; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- t < logTeff.size() by the loop bound
                    const double dist = (dr * dr) + (dt * dt);
                    if (dist < bestDist)
                    {
                        bestDist = dist;
                        bestR = r;
                        bestT = t;
                        found = true;
                    }
                }
            }
            return found ? std::optional<std::pair<size_t, size_t>>({bestR, bestT}) : std::nullopt;
        }
    } // namespace

    template <OOBPolicy Policy>
    SpecsynLibWR<Policy>::SpecsynLibWR( // NOLINT(readability-function-cognitive-complexity) -- seven sequential steps (find matching spectra, scan attributes, allocate the tensor grid, read flux/wavelength data, merge wavelength grids, regrid, store), each simple on its own; splitting them into separate functions would only add indirection, not clarity
        const std::string& spectraName,
        const double fehMin,
        const double fehMax,
        const std::string& registryName,
        double wlMin,
        double wlMax,
        const std::size_t nWl,
        const io::SimControls& controls) :
        SpecsynLib<Policy>(controls),
        FeH_(this->dim1_),
        logRt_(this->dim2_),
        logTeff_(this->dim3_)
    {
        // Determine which WR subtype this library covers from
        // spectraName (e.g. "POWR_WNE" -> WRType::WNE, "POWR_WNL_H40"
        // -> WRType::WNLH40), checked case-insensitively since nothing
        // guarantees a caller passes exactly the upper-case naming
        // fetch_powr.py's registry entries use (POWR_WNE/
        // POWR_WNL_H20/POWR_WNL_H40/POWR_WNL_H60/POWR_WC). A WNL
        // library must further specify which of PoWR's three
        // surface-hydrogen grids (H20/H40/H60) it is -- see WRType's
        // own comment for why there is no longer a single bare WNL.
        std::string nameLower = spectraName;
        std::ranges::transform(nameLower, nameLower.begin(), // NOLINT(misc-include-cleaner) -- see the identical NOLINT on SpecsynLib.cpp's findBracket for why std::ranges functions need this despite <algorithm> already being included
            [](const unsigned char c) -> char { return static_cast<char>(std::tolower(c)); });
        if (nameLower.contains("wnl"))
        {
            if (nameLower.contains("h20")) { type_ = WRType::WNLH20; }
            else if (nameLower.contains("h40")) { type_ = WRType::WNLH40; }
            else if (nameLower.contains("h60")) { type_ = WRType::WNLH60; }
            else
            {
                throw std::runtime_error(
                    "SpecsynLibWR: could not determine WR subtype from spectraName "
                    + spectraName + " (a WNL library's spectraName must also "
                    "contain h20, h40, or h60)");
            }
        }
        else if (nameLower.contains("wne")) { type_ = WRType::WNE; }
        else if (nameLower.contains("wc")) { type_ = WRType::WC; }
        else
        {
            throw std::runtime_error(
                "SpecsynLibWR: could not determine WR subtype from spectraName "
                + spectraName + " (expected it to contain wne, wc, or wnl "
                "together with h20, h40, or h60)");
        }

        // Step 1: find the set of spectra matching the input criteria.
        // afe/cfe/microTurb/r don't apply to PoWR's WR registry entries
        // (see fetch_powr.py) -- pass their library-wide defaults
        // through anyway, since findMatchingSpectra treats a group
        // missing one of those attributes as matching regardless of
        // the corresponding input value, so their actual values here
        // never matter.
        auto [fehVals, groupNames] = findMatchingSpectra(
            spectraName, fehMin, fehMax, tracks::defaultAFe, defaultCFe,
            defaultMicroTurb, defaultR, registryName);
        FeH_ = std::move(fehVals); //NOLINT(cppcoreguidelines-prefer-member-initializer)
        if (FeH_.empty())
        {
            throw std::runtime_error(
                "SpecsynLibWR: no spectra found matching the input criteria");
        }

        // Re-derive the path to the HDF5 file holding these spectra, the
        // same way findMatchingSpectra does internally
        auto [registry, registryPath] = parseRegistry(registryName);
        const auto h5name =
            registry.at_path(spectraName).at_path("file").value_or(std::string{});
        const auto h5path = registryPath.parent_path() / h5name;

        // NOLINTBEGIN(misc-include-cleaner)
        const hid_t file = H5Fopen(h5path.string().c_str(),
            H5F_ACC_RDONLY, H5P_DEFAULT);
        if (file < 0)
        {
            throw std::runtime_error(
                "SpecsynLibWR: unable to open HDF5 file " + h5path.string());
        }

        const size_t nfeh = FeH_.size();

        // Step 2: scan every matching group's datasets (without reading
        // their flux or wavelength data yet) to read the log_teff,
        // log_rt, and logl attributes fetch_powr.py stores on each
        // flux dataset, and thereby the unique sets of log(R_t) and
        // log(Teff) values that, together with FeH_, generate the
        // tensor grid on which this library's spectra sit. Each flux
        // dataset has a "_wave" companion dataset holding that
        // spectrum's own wavelength grid (see fetch_powr.py) rather
        // than a shared library-wide grid, so those are skipped here
        // entirely -- they carry no log_teff/log_rt/logl attributes of
        // their own anyway. Also reads each group's own dinf attribute
        // (the wind clumping density contrast, constant across every
        // model in the group, so stored once per [Fe/H] rather than
        // per model -- see fetch_powr.py).
        std::vector<std::vector<std::pair<std::string, std::tuple<double, double, double>>>>
            groupEntries(nfeh);
        std::set<double> logTeffSet;
        std::set<double> logRtSet;
        dInf_.resize(nfeh);
        for (size_t f = 0; f < nfeh; ++f)
        {
            const hid_t grp = H5Gopen2(file, groupNames[f].c_str(), H5P_DEFAULT);
            if (grp < 0)
            {
                H5Fclose(file);
                throw std::runtime_error(
                    "SpecsynLibWR: unable to open group " + groupNames[f]);
            }
            dInf_[f] = utils::readRequiredScalarAttr(grp, "dinf", "SpecsynLibWR");
            for (const auto& name : utils::listGroupDatasetNames(grp))
            {
                if (name.ends_with("_wave")) { continue; } // wavelength companion, not a flux dataset

                const hid_t dset = H5Dopen2(grp, name.c_str(), H5P_DEFAULT);
                if (dset < 0)
                {
                    H5Gclose(grp);
                    H5Fclose(file);
                    throw std::runtime_error(
                        "SpecsynLibWR: unable to open dataset " + name);
                }
                const double logTeff = utils::readRequiredScalarAttr(dset, "log_teff", "SpecsynLibWR");
                const double logRt = utils::readRequiredScalarAttr(dset, "log_rt", "SpecsynLibWR");
                const double logL = utils::readRequiredScalarAttr(dset, "logl", "SpecsynLibWR");
                H5Dclose(dset);

                groupEntries[f].emplace_back(name, std::make_tuple(logTeff, logRt, logL));
                logTeffSet.insert(logTeff);
                logRtSet.insert(logRt);
            }
            H5Gclose(grp);
        }
        logTeff_.assign(logTeffSet.begin(), logTeffSet.end());
        logRt_.assign(logRtSet.begin(), logRtSet.end());
        const size_t nrt = logRt_.size();
        const size_t nteff = logTeff_.size();

        // Step 3: allocate storage for the (FeH, logRt, logTeff)
        // tensor grid of spectra, and point grid_ at it; also allocate
        // a same-shaped temporary to hold each populated point's own
        // wavelength grid -- unlike SpecsynLibNoWind's single shared
        // grid, every PoWR model has its own distinct wavelength
        // sampling (see fetch_powr.py), so there is no single grid to
        // read once here. Read unconditionally regardless of nWl:
        // even when the caller has requested an explicit output grid
        // (nWl != 0), step 6 below still needs each populated point's
        // own native wavelength grid to interpolate its flux from,
        // just onto the caller's commonWl instead of a native-derived
        // one -- only step 5's own derivation of commonWl actually
        // differs on nWl.
        using SpectraGrid = typename SpecsynLib<Policy>::SpectraGrid;
        this->spectra_.assign(nfeh * nrt * nteff, std::vector<double>{});
        this->grid_ = SpectraGrid(this->spectra_.data(), nfeh, nrt, nteff);
        std::vector<std::vector<double>> waveTemp(nfeh * nrt * nteff);
        const SpectraGrid waveGrid(waveTemp.data(), nfeh, nrt, nteff);
        logL_.assign(nfeh * nrt * nteff, std::numeric_limits<double>::quiet_NaN());
        logLGrid_ = ScalarGrid(logL_.data(), nfeh, nrt, nteff);

        // Step 4: read each populated point's flux and its own
        // wavelength grid, placing each at its point in the tensor grid
        for (size_t f = 0; f < nfeh; ++f)
        {
            const hid_t grp = H5Gopen2(file, groupNames[f].c_str(), H5P_DEFAULT);
            if (grp < 0)
            {
                H5Fclose(file);
                throw std::runtime_error(
                    "SpecsynLibWR: unable to open group " + groupNames[f]);
            }
            for (const auto& [name, attrs] : groupEntries[f])
            {
                const auto [logTeffVal, logRtVal, logLVal] = attrs;
                const auto iTeff = static_cast<size_t>(
                    std::ranges::lower_bound(logTeff_, logTeffVal) - logTeff_.begin());
                const auto iRt = static_cast<size_t>(
                    std::ranges::lower_bound(logRt_, logRtVal) - logRt_.begin());
                this->grid_[f, iRt, iTeff] = utils::readDataset1D(grp, name, "SpecsynLibWR");
                waveGrid[f, iRt, iTeff] = utils::readDataset1D(grp, name + "_wave", "SpecsynLibWR");
                logLGrid_[f, iRt, iTeff] = logLVal;
            }
            H5Gclose(grp);
        }

        H5Fclose(file);
        // NOLINTEND(misc-include-cleaner)

        // Step 5: build a single common wavelength grid spanning every
        // populated point's own native grid -- unless the caller
        // requested an explicit output grid (nWl != 0), in which case
        // commonWl is simply nWl points log-spaced from wlMin to
        // wlMax instead, and step 6 below resamples every populated
        // point onto it exactly as it would onto a native-derived grid
        std::vector<double> commonWl;
        if (nWl == 0)
        {
            std::vector<std::vector<double>> wlGrids;
            wlGrids.reserve(nfeh * nrt * nteff);
            for (const auto& wave : waveTemp)
            {
                if (!wave.empty()) { wlGrids.push_back(wave); }
            }
            if (wlGrids.empty())
            {
                throw std::runtime_error(
                    "SpecsynLibWR: no populated grid points found for " + spectraName);
            }
            commonWl = SpecsynLibChained::makeCommonWlGrid(wlGrids);
        }
        else
        {
            // wlMin == 0 here means only nWl was actually requested
            // (wlMin/wlMax are still at SimControls::readSpectra's
            // "not supplied" sentinel), so fall back to the global
            // range spanned by every populated point's own native
            // wavelength grid -- the same grids the nWl == 0 branch
            // above would otherwise have merged via makeCommonWlGrid
            // -- keeping the caller's requested point count.
            if (wlMin == 0.0)
            {
                double globalMin = std::numeric_limits<double>::infinity();
                double globalMax = -std::numeric_limits<double>::infinity();
                for (const auto& wave : waveTemp)
                {
                    if (wave.empty()) { continue; }
                    globalMin = std::min(globalMin, wave.front());
                    globalMax = std::max(globalMax, wave.back());
                }
                if (!std::isfinite(globalMin))
                {
                    throw std::runtime_error(
                        "SpecsynLibWR: no populated grid points found for " + spectraName);
                }
                wlMin = globalMin;
                wlMax = globalMax;
            }
            commonWl = utils::logspace(wlMin, wlMax, nWl);
        }

        // Step 6: regrid every populated spectrum from its own native
        // wavelength grid onto the common one, via the same
        // single-spectrum resampling SpecsynLib::resample uses
        for (size_t f = 0; f < nfeh; ++f) // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- f, rt, t are all < the corresponding grid's size by construction
        {
            for (size_t rt = 0; rt < nrt; ++rt)
            {
                for (size_t t = 0; t < nteff; ++t)
                {
                    auto& spectrum = this->grid_[f, rt, t];
                    if (spectrum.empty()) { continue; } // unpopulated grid point: leave empty
                    const auto& wave = waveGrid[f, rt, t];

                    spectrum = SpecsynLib<Policy>::resample(wave, commonWl, spectrum);
                }
            }
        }

        // Step 7: store the common grid as this library's wavelength grid
        this->wl_ = commonWl;

        // Step 8: if this library is one of the three WNL buckets, seed
        // wnlTeffRanges_'s own entry with this library's own logTeff_
        // range -- the only range this instance can know about without
        // help from a sibling library -- so getWRType behaves sensibly
        // even when this instance is never told about its siblings via
        // setWNLTeffRanges() (e.g. used standalone). See wnlTeffRanges_'s
        // own comment.
        if (type_ == WRType::WNLH20 || type_ == WRType::WNLH40 || type_ == WRType::WNLH60)
        {
            const auto idx = static_cast<std::size_t>(type_) - static_cast<std::size_t>(WRType::WNLH20);
            wnlTeffRanges_.at(idx) = {logTeff_.front(), logTeff_.back()};
        }
    }

    template <OOBPolicy Policy>
    auto SpecsynLibWR<Policy>::getWRType(
        const Specsyn::StarData& props,
        const std::array<std::pair<double, double>, 3>& wnlTeffRanges) -> WRType
    {
        // NOLINTBEGIN(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- StarData is a fixed-size std::array, and every index used here is compile-time-known
        const double heSurf = props[static_cast<size_t>(tracks::FieldIdx::heSurf)];
        const double logTeff = props[static_cast<size_t>(tracks::FieldIdx::logTe)];
        const double mass = props[static_cast<size_t>(tracks::FieldIdx::mass)];
        constexpr double massMin = 5.0; // Msun -- see this function's own comment
        if (mass < massMin) { return WRType::None; }

        if (heSurf >= 0.4 && heSurf <= 0.9)
        {
            const double hSurf = props[static_cast<size_t>(tracks::FieldIdx::hSurf)];
            std::size_t idx = 0;
            WRType candidate = WRType::WNLH20;
            if (hSurf < 0.3) { idx = 0; candidate = WRType::WNLH20; }
            else if (hSurf <= 0.5) { idx = 1; candidate = WRType::WNLH40; }
            else { idx = 2; candidate = WRType::WNLH60; }

            const auto [teffMin, teffMax] = wnlTeffRanges.at(idx);
            if (!std::isnan(teffMin) && logTeff >= teffMin && logTeff <= teffMax)
            {
                return candidate;
            }
            // Composition matches a WNL bucket, but this star's own
            // log(Teff) doesn't fall within that bucket's real grid
            // coverage (or that bucket's range is unknown entirely) --
            // fall through to the WNE/WC check below rather than
            // returning WRType::None here, since a hot enough star in
            // this He window could still genuinely be a WC/WO star.
        }

        constexpr double logTeffHotMin = 4.6989700043360187; // log10(50000)
        if (logTeff > logTeffHotMin)
        {
            const double cSurf = props[static_cast<size_t>(tracks::FieldIdx::cSurf)];
            const double nSurf = props[static_cast<size_t>(tracks::FieldIdx::nSurf)];
            return (cSurf < nSurf) ? WRType::WNE : WRType::WC;
        }

        return WRType::None;
        // NOLINTEND(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
    }

    template <OOBPolicy Policy>
    auto SpecsynLibWR<Policy>::computeRawLogRt(const Specsyn::StarData& props, const double feh) const -> double
    {
        // NOLINTBEGIN(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- StarData is a fixed-size std::array, and every index used here is compile-time-known
        const double logL = props[static_cast<size_t>(tracks::FieldIdx::logL)];
        const double mdot = props[static_cast<size_t>(tracks::FieldIdx::mdot)]; // Msun/yr
        // NOLINTEND(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)

        // D_infinity by linear interpolation in [Fe/H] on dInf_
        const auto bFeh = findBracket(FeH_, feh);
        const double dInf = ((1.0 - bFeh.t_) * dInf_[bFeh.lo_]) + (bFeh.t_ * dInf_[bFeh.hi_]); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- bFeh.lo_/hi_ < dInf_.size() by construction (both sized nfeh)

        // Wind velocity vWind = L / (mdot c), in cgs
        const double lumCgs = std::pow(10.0, logL) * utils::Lsun;      // erg/s
        const double mdotCgs = mdot * utils::Msun / utils::yr;         // g/s
        const double vWind = lumCgs / (mdotCgs * utils::c);            // cm/s

        // Transformed radius Rt (Todt et al. 2015, eq. 2), via the
        // star's own radius -- derived from its surface area, itself
        // derived (by getSAandLogg) from L and Teff -- expressed in
        // Rsun to match the grid's own log_rt units (see
        // fetch_powr.py's R_TRANS [Rsun] -> log10(R_t) conversion)
        constexpr double pi = std::numbers::pi_v<double>;
        const double area = Specsyn::getSAandLogg(props).first; // cm^2
        const double rStarRsun = std::sqrt(area / (4.0 * pi)) / utils::Rsun;

        constexpr double vWindNorm = 2500.0e5; // 2500 km/s, in cm/s
        constexpr double mdotNorm = 1.0e-4;    // Msun/yr
        const double ratio = (vWind / vWindNorm) / (mdot * std::sqrt(dInf) / mdotNorm);
        return std::log10(rStarRsun * std::pow(ratio, 2.0 / 3.0));
    }

    template <OOBPolicy Policy>
    auto SpecsynLibWR<Policy>::spec(const Specsyn::StarData& props, const double feh) const -> std::vector<double> // NOLINT(readability-function-cognitive-complexity) -- WRType check, bounds check, logRt search, and the final logL rescaling are each simple on their own; splitting them into separate functions would only add indirection, not clarity
    {
        // Step 1: a WRType mismatch means this library's spectra don't
        // apply to this star at all
        if (getWRType(props, wnlTeffRanges_) != type_)
        {
            return SpecsynLib<Policy>::outOfBoundsResult(
                "SpecsynLibWR: star's WRType does not match this library's type");
        }

        // NOLINTBEGIN(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- StarData is a fixed-size std::array, and every index used here is compile-time-known
        const double logL = props[static_cast<size_t>(tracks::FieldIdx::logL)];
        const double logTeff = props[static_cast<size_t>(tracks::FieldIdx::logTe)];
        // NOLINTEND(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)

        // Step 2: derive the transformed radius this star maps to
        const double rawLogRt = computeRawLogRt(props, feh);

        // Bounds check: (feh, logTeff) must fall within this library's
        // grid before delegating to the parent class's own spec(),
        // which assumes its caller has already done so (it only checks
        // that the 8 bracketing corners are populated, not that the
        // query point itself is in range). logRt is exempt from this
        // check -- see the comment on its own clamping below.
        if (feh < FeH_.front() || feh > FeH_.back() ||
            logTeff < logTeff_.front() || logTeff > logTeff_.back())
        {
            return SpecsynLib<Policy>::outOfBoundsResult(
                "SpecsynLibWR: star with feh = " + std::to_string(feh) +
                ", logRt = " + std::to_string(rawLogRt) +
                ", logTeff = " + std::to_string(logTeff) +
                " is outside this library's grid");
        }
        const auto bFeh = findBracket(FeH_, feh);
        const auto bTeff = findBracket(logTeff_, logTeff);

        // vWind above is a single-scattering estimate (wind momentum
        // mdot * vWind = L / c) rather than a true wind speed: fits
        // like Nugis & Lamers (2000) -- the actual source of MIST's WR
        // mass loss rates -- give a more direct estimate, but
        // extrapolating them outside the limited [Fe/H] range they
        // were fit over produces nonsense (wind speeds approaching c).
        // Single scattering is a cruder but non-crazy stand-in that
        // holds up over most of parameter space; the tradeoff is that
        // a handful of high-mdot/L points (like this one) push logRt
        // below any real WR grid's coverage, since real WR winds are
        // multiply scattered and so faster (and hence a larger
        // transformed radius) than this estimate implies. There is no
        // well-established correction for this, so rather than reject
        // these stars outright, move logRt to real data instead.
        //
        // PoWR's own (feh, logRt, logTeff) grids are ragged, not
        // rectangular, and get sparser still at the hottest edge --
        // e.g. WC's own grid has only 3 populated log_rt values at its
        // hottest log_teff point, versus ~20 one grid step down --
        // exactly the kind of gap a star whose logTeff has been
        // clamped onto that hot edge in SpecsynLibChained::spec() (so
        // bTeff.t_ is exactly 0 or 1, collapsing all interpolation
        // weight onto a single logTeff column) can land in. So first
        // look only at the (up to 4) feh/logTeff corner combinations
        // that actually carry nonzero interpolation weight -- an exact
        // hit on either axis makes the corresponding "other side"
        // irrelevant regardless of what it has populated -- and move
        // logRt (feh and logTeff both left fixed) to whichever
        // populated grid value is closest to rawLogRt and has data at
        // every one of those weight-carrying corners simultaneously,
        // mirroring the mass rescaling that clamps log(g) in
        // SpecsynLibChained::spec(): both replace an out-of-range
        // continuous estimate with the nearest point guaranteed to
        // actually have data, rather than interpolating across a gap.
        const std::array<std::pair<size_t, size_t>, 4> fehTeffCorners = {{
            { bFeh.lo_, bTeff.lo_ }, { bFeh.lo_, bTeff.hi_ },
            { bFeh.hi_, bTeff.lo_ }, { bFeh.hi_, bTeff.hi_ },
        }};
        const std::array<double, 4> fehTeffWeights = {{
            (1.0 - bFeh.t_) * (1.0 - bTeff.t_), (1.0 - bFeh.t_) * bTeff.t_,
            bFeh.t_ * (1.0 - bTeff.t_), bFeh.t_ * bTeff.t_,
        }};
        double logRt = std::numeric_limits<double>::quiet_NaN();
        double closestValidDist = std::numeric_limits<double>::infinity();
        for (size_t r = 0; r < logRt_.size(); ++r)
        {
            bool populatedAtEveryWeightedCorner = true;
            for (size_t c = 0; c < fehTeffCorners.size(); ++c)
            {
                if (fehTeffWeights[c] == 0.0) { continue; } // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index,cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- c < fehTeffWeights.size() by construction; this corner carries no interpolation weight, so it doesn't matter whether it's populated
                const auto [f, t] = fehTeffCorners[c]; // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index,cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- c < fehTeffCorners.size() by construction
                if (this->grid_[f, r, t].empty())
                {
                    populatedAtEveryWeightedCorner = false;
                    break;
                }
            }
            if (!populatedAtEveryWeightedCorner) { continue; }
            const double dist = std::abs(logRt_[r] - rawLogRt); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- r < logRt_.size() by construction
            if (dist < closestValidDist)
            {
                closestValidDist = dist;
                logRt = logRt_[r]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- see above
            }
        }

        if (std::isnan(logRt))
        {
            // No single logRt value has data at every weight-carrying
            // corner -- fall back to clamping into the union of
            // whatever each of the (up to 4) bracketing columns has
            // populated at all, regardless of weight, so the parent
            // class's own OOBPolicy::coerce handling still gets a
            // chance to salvage a partial-weight interpolation.
            // SpecsynLib::spec() reports this as out of bounds exactly
            // as it would have before this fallback existed if even
            // that comes up with nothing populated at all.
            double populatedRtMin = std::numeric_limits<double>::infinity();
            double populatedRtMax = -std::numeric_limits<double>::infinity();
            for (const size_t f : { bFeh.lo_, bFeh.hi_ })
            {
                for (const size_t t : { bTeff.lo_, bTeff.hi_ })
                {
                    for (size_t r = 0; r < logRt_.size(); ++r)
                    {
                        if (!this->grid_[f, r, t].empty())
                        {
                            populatedRtMin = std::min(populatedRtMin, logRt_[r]); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- r < logRt_.size() by construction
                            populatedRtMax = std::max(populatedRtMax, logRt_[r]); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- see above
                        }
                    }
                }
            }
            logRt = std::isfinite(populatedRtMin) ?
                std::clamp(rawLogRt, populatedRtMin, populatedRtMax) :
                std::clamp(rawLogRt, logRt_.front(), logRt_.back());
        }

        // Step 5: the actual trilinear interpolation, handled entirely
        // by the parent class -- also returns an OOB result if any of
        // the 8 neighboring grid points is unpopulated
        auto result = this->SpecsynLib<Policy>::spec(feh, logRt, logTeff);
        if (result.empty()) { return result; }

        // Step 6: rescale from the interpolated grid point's own
        // luminosity to this star's actual luminosity -- the model
        // spectra are each normalized to their own model's L, not
        // necessarily this particular star's
        const auto bRt = findBracket(logRt_, logRt);
        const double logLGrid = trilinearScalar(logLGrid_, bFeh, bRt, bTeff);
        const double scale = std::pow(10.0, logL - logLGrid);
        for (auto& v : result) { v *= scale; }

        // Step 7
        return result;
    }

    template <OOBPolicy Policy>
    auto SpecsynLibWR<Policy>::specForce(const Specsyn::StarData& props, const double feh) const -> std::vector<double>
    {
        // Reject only if feh itself can't even be bracketed -- neither
        // logRt nor logTeff is checked against its own range here,
        // since the whole point of this function is to search for a
        // populated (log(R_t), log(Teff)) point regardless of how far
        // out of range this star's own values start (see this
        // function's own header comment for why both, not just
        // log(R_t), are searched).
        if (feh < FeH_.front() || feh > FeH_.back())
        {
            throw std::runtime_error(
                "SpecsynLibWR::specForce: star with feh = " + std::to_string(feh) +
                " is entirely outside this library's feh range");
        }

        // NOLINTBEGIN(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- StarData is a fixed-size std::array, and every index used here is compile-time-known
        const double logL = props[static_cast<size_t>(tracks::FieldIdx::logL)];
        const double logTeff = props[static_cast<size_t>(tracks::FieldIdx::logTe)];
        // NOLINTEND(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
        const double rawLogRt = computeRawLogRt(props, feh);

        const auto bFeh = findBracket(FeH_, feh);
        const std::array<std::pair<size_t, double>, 2> fehSides = {{
            {bFeh.lo_, 1.0 - bFeh.t_}, {bFeh.hi_, bFeh.t_}}};

        // At each of the (up to two) bracketing feh corners, find
        // whichever populated (log(R_t), log(Teff)) grid point is
        // closest to this star's own raw values, rescale that point's
        // own spectrum to this star's luminosity, and blend across feh
        // corners by the ordinary feh bilinear weight, renormalized by
        // the sum of weights actually used -- see this function's own
        // header comment for the full rationale.
        std::vector<double> result(this->wl_.size(), 0.0);
        double wSum = 0.0;
        for (const auto& [f, wgtF] : fehSides)
        {
            if (wgtF == 0.0) { continue; } // degenerate axis or exact grid hit: skip a zero-weight corner

            const auto nearest = nearestPopulatedRtTeff(this->grid_, logRt_, logTeff_, f, rawLogRt, logTeff);
            if (!nearest) { continue; } // this feh corner has no populated grid point at all
            const auto [r, t] = *nearest;

            wSum += wgtF;
            const double logLGrid = logLGrid_[f, r, t]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- f, r, t all < the corresponding axis's size by construction
            const double scale = std::pow(10.0, logL - logLGrid);
            const auto& spectrum = this->grid_[f, r, t]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- see above
            for (size_t w = 0; w < result.size(); ++w) { result[w] += wgtF * scale * spectrum[w]; }
        }

        if (wSum == 0.0)
        {
            throw std::runtime_error(
                "SpecsynLibWR::specForce: no populated (log(R_t), log(Teff)) grid "
                "point found for feh = " + std::to_string(feh) + ", logTeff = " +
                std::to_string(logTeff));
        }
        for (auto& v : result) { v /= wSum; }
        return result;
    }

    // Explicit instantiation for every OOBPolicy value actually used;
    // this keeps the constructor's implementation in this .cpp file,
    // as with every other class in src/specsyn.
    template class SpecsynLibWR<OOBPolicy::raise>;
    template class SpecsynLibWR<OOBPolicy::silent>;
    template class SpecsynLibWR<OOBPolicy::coerce>;

} // namespace specsyn
