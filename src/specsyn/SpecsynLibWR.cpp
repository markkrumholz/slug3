/**
 * @file SpecsynLibWR.cpp
 * @author Mark Krumholz
 * @brief Implementation of SpecsynLibWR.hpp
 * @date 2026-07-22
 */

#include "SpecsynLibWR.hpp"
#include "../io/SimControls.hpp"
#include "../tracks/TrackCommons.hpp"
#include "../utils/Constants.hpp"
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
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace specsyn
{
    // Suppress clang-tidy warnings iun this namespace caused by just including
    // hdf5.h, instead of the individual HDF5 headers, since this is the paradigm
    // that HDF5 wants
    // NOLINTBEGIN(misc-include-cleaner)
    namespace
    {
        /**
         * @brief Read a 1D double dataset from an HDF5 group
         * @param grp Handle to the group containing the dataset
         * @param name Name of the dataset
         * @returns The dataset contents
         */
        auto readDataset1D(const hid_t grp, const std::string& name) //NOLINT(llvm-prefer-static-over-anonymous-namespace)
            -> std::vector<double>
        {
            const hid_t dset = H5Dopen2(grp, name.c_str(), H5P_DEFAULT);
            if (dset < 0)
            {
                throw std::runtime_error(
                    "SpecsynLibWR: unable to open dataset " + name);
            }
            const hid_t space = H5Dget_space(dset);
            hsize_t dims = 0;
            H5Sget_simple_extent_dims(space, &dims, nullptr);
            std::vector<double> data(dims);
            H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                H5P_DEFAULT, data.data());
            H5Sclose(space);
            H5Dclose(dset);
            return data;
        }

        /**
         * @brief List the names of every dataset directly inside an HDF5 group
         * @param grp Handle to the group
         * @returns The names of the group's child datasets
         */
        auto listGroupDatasetNames(const hid_t grp) //NOLINT(llvm-prefer-static-over-anonymous-namespace)
            -> std::vector<std::string>
        {
            H5G_info_t ginfo{};
            H5Gget_info(grp, &ginfo);

            std::vector<std::string> names;
            names.reserve(ginfo.nlinks);
            for (hsize_t i = 0; i < ginfo.nlinks; ++i)
            {
                const auto nameLen = H5Lget_name_by_idx(grp, ".",
                    H5_INDEX_NAME, H5_ITER_INC, i, nullptr, 0, H5P_DEFAULT);
                if (nameLen < 0) { continue; }
                std::vector<char> nameBuf(static_cast<size_t>(nameLen) + 1);
                H5Lget_name_by_idx(grp, ".", H5_INDEX_NAME, H5_ITER_INC, i,
                    nameBuf.data(), nameBuf.size(), H5P_DEFAULT);
                names.emplace_back(nameBuf.data());
            }
            return names;
        }

        /**
         * @brief Read a required scalar double attribute from an HDF5 object
         * @param obj Handle to the object (a group or a dataset)
         * @param name Name of the attribute
         * @returns The attribute's value
         * @throws std::runtime_error if obj has no attribute of that name
         */
        auto readRequiredScalarAttr(const hid_t obj, const std::string& name) //NOLINT(llvm-prefer-static-over-anonymous-namespace)
            -> double
        {
            if (H5Aexists(obj, name.c_str()) <= 0)
            {
                throw std::runtime_error(
                    "SpecsynLibWR: missing required attribute " + name);
            }
            const hid_t attr = H5Aopen(obj, name.c_str(), H5P_DEFAULT);
            if (attr < 0)
            {
                throw std::runtime_error(
                    "SpecsynLibWR: unable to open attribute " + name);
            }
            double value = 0.0;
            H5Aread(attr, H5T_NATIVE_DOUBLE, &value);
            H5Aclose(attr);
            return value;
        }
    } // namespace
    // NOLINTEND(misc-include-cleaner)

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
        const double z,
        const io::SimControls& controls) :
        SpecsynLib<Policy>(),
        FeH_(this->dim1_),
        logRt_(this->dim2_),
        logTeff_(this->dim3_)
    {
        this->z_ = z;
        this->intRelTol_ = controls.intRelTol();
        this->intAbsTol_ = controls.intAbsTol();
        this->intMaxIter_ = controls.intMaxIter();

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
            dInf_[f] = readRequiredScalarAttr(grp, "dinf");
            for (const auto& name : listGroupDatasetNames(grp))
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
                const double logTeff = readRequiredScalarAttr(dset, "log_teff");
                const double logRt = readRequiredScalarAttr(dset, "log_rt");
                const double logL = readRequiredScalarAttr(dset, "logl");
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
                this->grid_[f, iRt, iTeff] = readDataset1D(grp, name);
                waveGrid[f, iRt, iTeff] = readDataset1D(grp, name + "_wave");
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
            // (wlMin/wlMax are still at SimPhysics::readSpectra's
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
    }

    template <OOBPolicy Policy>
    auto SpecsynLibWR<Policy>::getWRType(const Specsyn::StarData& props) -> WRType
    {
        // NOLINTBEGIN(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- StarData is a fixed-size std::array, and every index used here is compile-time-known
        const double heSurf = props[static_cast<size_t>(tracks::FieldIdx::heSurf)];
        const double logTeff = props[static_cast<size_t>(tracks::FieldIdx::logTe)];
        constexpr double logTeffNonWRMax = 4.6989700043360187; // log10(50000)
        constexpr double logTeffAbsoluteMin = 4.0; // log10(10000)
        if ((heSurf < 0.4 && logTeff < logTeffNonWRMax) || logTeff < logTeffAbsoluteMin)
        {
            return WRType::None;
        }
        constexpr double logTeffWNLMax = 5.0; // log10(1e5 K)
        if (heSurf <= 0.9 && logTeff < logTeffWNLMax)
        {
            const double hSurf = props[static_cast<size_t>(tracks::FieldIdx::hSurf)];
            if (hSurf < 0.3) { return WRType::WNLH20; }
            if (hSurf <= 0.5) { return WRType::WNLH40; }
            return WRType::WNLH60;
        }
        const double cSurf = props[static_cast<size_t>(tracks::FieldIdx::cSurf)];
        const double nSurf = props[static_cast<size_t>(tracks::FieldIdx::nSurf)];
        if (cSurf < nSurf)
        {
            return WRType::WNE;
        }
        return WRType::WC;
        // NOLINTEND(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
    }

    template <OOBPolicy Policy>
    auto SpecsynLibWR<Policy>::spec(const Specsyn::StarData& props, const double feh) const -> std::vector<double> // NOLINT(readability-function-cognitive-complexity) -- WRType check, dInf/vWind/Rt derivation, bounds check, and the final logL rescaling are each simple on their own; splitting them into separate functions would only add indirection, not clarity
    {
        // Step 1: a WRType mismatch means this library's spectra don't
        // apply to this star at all
        if (getWRType(props) != type_)
        {
            return SpecsynLib<Policy>::outOfBoundsResult(
                "SpecsynLibWR: star's WRType does not match this library's type");
        }

        // NOLINTBEGIN(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- StarData is a fixed-size std::array, and every index used here is compile-time-known
        const double logL = props[static_cast<size_t>(tracks::FieldIdx::logL)];
        const double logTeff = props[static_cast<size_t>(tracks::FieldIdx::logTe)];
        const double mdot = props[static_cast<size_t>(tracks::FieldIdx::mdot)]; // Msun/yr
        // NOLINTEND(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)

        // Step 2: D_infinity by linear interpolation in [Fe/H] on dInf_
        const auto bFeh = findBracket(FeH_, feh);
        const double dInf = ((1.0 - bFeh.t_) * dInf_[bFeh.lo_]) + (bFeh.t_ * dInf_[bFeh.hi_]); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- bFeh.lo_/hi_ < dInf_.size() by construction (both sized nfeh)

        // Step 3: wind velocity vWind = L / (mdot c), in cgs
        const double lumCgs = std::pow(10.0, logL) * utils::Lsun;      // erg/s
        const double mdotCgs = mdot * utils::Msun / utils::yr;         // g/s
        const double vWind = lumCgs / (mdotCgs * utils::c);            // cm/s

        // Step 4: transformed radius Rt (Todt et al. 2015, eq. 2), via
        // the star's own radius -- derived from its surface area,
        // itself derived (by getSAandLogg) from L and Teff -- expressed
        // in Rsun to match the grid's own log_rt units (see
        // fetch_powr.py's R_TRANS [Rsun] -> log10(R_t) conversion)
        constexpr double pi = std::numbers::pi_v<double>;
        const double area = Specsyn::getSAandLogg(props).first; // cm^2
        const double rStarRsun = std::sqrt(area / (4.0 * pi)) / utils::Rsun;

        constexpr double vWindNorm = 2500.0e5; // 2500 km/s, in cm/s
        constexpr double mdotNorm = 1.0e-4;    // Msun/yr
        const double ratio = (vWind / vWindNorm) / (mdot * std::sqrt(dInf) / mdotNorm);
        const double rawLogRt = std::log10(rStarRsun * std::pow(ratio, 2.0 / 3.0));

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

    // Explicit instantiation for every OOBPolicy value actually used;
    // this keeps the constructor's implementation in this .cpp file,
    // as with every other class in src/specsyn.
    template class SpecsynLibWR<OOBPolicy::raise>;
    template class SpecsynLibWR<OOBPolicy::silent>;
    template class SpecsynLibWR<OOBPolicy::coerce>;

} // namespace specsyn
