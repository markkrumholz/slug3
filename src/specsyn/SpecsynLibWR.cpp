/**
 * @file SpecsynLibWR.cpp
 * @author Mark Krumholz
 * @brief Implementation of SpecsynLibWR.hpp
 * @date 2026-07-22
 */

#include "SpecsynLibWR.hpp"
#include "../interpolation/Interpolator1D.hpp"
#include "../tracks/TrackCommons.hpp"
#include "Specsyn.hpp"
#include "SpecsynCommons.hpp"
#include "SpecsynLib.hpp"
#include "SpecsynLibChained.hpp"
#include "SpecsynUtils.hpp"
#include "hdf5.h" // NOLINT(misc-include-cleaner)
#include <algorithm> // NOLINT(misc-include-cleaner) -- see the identical NOLINT on SpecsynLib.cpp's findBracket for why std::ranges::lower_bound needs this
#include <cctype>
#include <cmath>
#include <cstddef>
#include <gsl/gsl_const_cgsm.h> // NOLINT(misc-include-cleaner)
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

            const auto it = std::ranges::upper_bound(grid, value); //NOLINT(misc-include-cleaner) -- see the identical NOLINT on SpecsynLib.cpp's findBracket
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
         * Unlike SpecsynLib::spec(double, double, double), this has no
         * populated/unpopulated notion of its own to check -- logLGrid_
         * is only ever populated at exactly the same points as
         * SpecsynLib::spectra_ (see SpecsynLibWR's constructor), so a
         * caller that has already confirmed the spectrum interpolation
         * succeeded at this same point knows every corner used here is
         * meaningful too.
         */
        auto trilinearScalar( //NOLINT(llvm-prefer-static-over-anonymous-namespace, readability-function-cognitive-complexity) -- see the identical NOLINT on SpecsynLib.cpp's own spec(double, double, double), whose nested trilinear-interpolation loop this mirrors
            const std::mdspan<double, std::dextents<std::size_t, 3>>& grid, // NOLINT(misc-include-cleaner) -- see the identical NOLINT on SpecsynLib.hpp's SpectraGrid alias
            const Bracket& b1, const Bracket& b2, const Bracket& b3) -> double
        {
            double result = 0.0;
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

                        result += weight * grid[i1, i2, i3]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- i1, i2, i3 are all < the corresponding grid's size by construction
                    }
                }
            }
            return result;
        }
    } // namespace

    template <OOBPolicy Policy>
    SpecsynLibWR<Policy>::SpecsynLibWR( // NOLINT(readability-function-cognitive-complexity) -- seven sequential steps (find matching spectra, scan attributes, allocate the tensor grid, read flux/wavelength data, merge wavelength grids, regrid, store), each simple on its own; splitting them into separate functions would only add indirection, not clarity
        const std::string& spectraName,
        const double fehMin,
        const double fehMax,
        const std::string& registryName,
        const double z) :
        SpecsynLib<Policy>(),
        FeH_(this->dim1_),
        logRt_(this->dim2_),
        logTeff_(this->dim3_)
    {
        this->z_ = z;

        // Determine which WR subtype this library covers from
        // spectraName (e.g. "POWR_WNE" -> WRType::WNE), checked
        // case-insensitively since nothing guarantees a caller passes
        // exactly the upper-case naming fetch_powr.py's registry
        // entries use (POWR_WNE/POWR_WNL/POWR_WC).
        std::string nameLower = spectraName;
        std::ranges::transform(nameLower, nameLower.begin(), // NOLINT(misc-include-cleaner) -- see the identical NOLINT on SpecsynLib.cpp's findBracket for why std::ranges functions need this despite <algorithm> already being included
            [](const unsigned char c) -> char { return static_cast<char>(std::tolower(c)); });
        if (nameLower.contains("wnl")) { type_ = WRType::WNL; }
        else if (nameLower.contains("wne")) { type_ = WRType::WNE; }
        else if (nameLower.contains("wc")) { type_ = WRType::WC; }
        else
        {
            throw std::runtime_error(
                "SpecsynLibWR: could not determine WR subtype from spectraName "
                + spectraName + " (expected it to contain wne, wnl, or wc)");
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
        const size_t nfeh = FeH_.size();
        if (nfeh == 0)
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
        // read once here.
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
        // populated point's own native grid
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
        const auto commonWl = SpecsynLibChained::makeCommonWlGrid(wlGrids);

        // Step 6: regrid every populated spectrum onto the common
        // wavelength grid, exactly as SpecsynLib::resample does --
        // an Interpolator1D of the point's own flux versus its own
        // wavelength grid, evaluated at every wavelength in commonWl,
        // with wavelengths outside the point's native range assigned
        // zero flux rather than extrapolated
        for (size_t f = 0; f < nfeh; ++f) // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- f, rt, t are all < the corresponding grid's size by construction
        {
            for (size_t rt = 0; rt < nrt; ++rt)
            {
                for (size_t t = 0; t < nteff; ++t)
                {
                    auto& spectrum = this->grid_[f, rt, t];
                    if (spectrum.empty()) { continue; } // unpopulated grid point: leave empty
                    const auto& wave = waveGrid[f, rt, t];

                    const interp::Interpolator1D<1> interpolator(wave, spectrum);
                    std::vector<double> resampled(commonWl.size(), 0.0);
                    for (size_t w = 0; w < commonWl.size(); ++w)
                    {
                        if (commonWl[w] >= wave.front() && commonWl[w] <= wave.back())
                        {
                            resampled[w] = interpolator(commonWl[w]);
                        }
                        // else leave as the zero flux resampled was initialized with
                    }
                    spectrum = std::move(resampled);
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
        const double logTeff = props[static_cast<size_t>(tracks::FieldIdx::logTe)];
        const double hSurf = props[static_cast<size_t>(tracks::FieldIdx::hSurf)];
        if (logTeff < 4.0 || hSurf > 0.3)
        {
            return WRType::None;
        }
        if (hSurf > 1e-5)
        {
            return WRType::WNL;
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
        constexpr double solarLuminosity = 3.828e33; // erg/s, IAU 2015 nominal value (matches Specsyn::getSAandLogg)
        constexpr double solarMass = GSL_CONST_CGSM_SOLAR_MASS;
        constexpr double speedOfLight = GSL_CONST_CGSM_SPEED_OF_LIGHT;
        constexpr double secPerYear = 3.15576e7; // Julian year (365.25 d), s -- GSL's cgsm constants have no year unit of their own

        const double lumCgs = std::pow(10.0, logL) * solarLuminosity; // erg/s
        const double mdotCgs = mdot * solarMass / secPerYear;         // g/s
        const double vWind = lumCgs / (mdotCgs * speedOfLight);       // cm/s

        // Step 4: transformed radius Rt (Todt et al. 2015, eq. 2), via
        // the star's own radius -- derived from its surface area,
        // itself derived (by getSAandLogg) from L and Teff -- expressed
        // in Rsun to match the grid's own log_rt units (see
        // fetch_powr.py's R_TRANS [Rsun] -> log10(R_t) conversion)
        constexpr double solarRadius = 6.957e10; // cm, IAU 2015 nominal value
        constexpr double pi = std::numbers::pi_v<double>;
        const double area = Specsyn::getSAandLogg(props).first; // cm^2
        const double rStarRsun = std::sqrt(area / (4.0 * pi)) / solarRadius;

        constexpr double vWindNorm = 2500.0e5; // 2500 km/s, in cm/s
        constexpr double mdotNorm = 1.0e-4;    // Msun/yr
        const double ratio = (vWind / vWindNorm) / (mdot * std::sqrt(dInf) / mdotNorm);
        const double logRt = std::log10(rStarRsun * std::pow(ratio, 2.0 / 3.0));

        // Bounds check: (feh, logRt, logTeff) must fall within this
        // library's grid before delegating to the parent class's own
        // spec(), which assumes its caller has already done so (it
        // only checks that the 8 bracketing corners are populated, not
        // that the query point itself is in range)
        if (feh < FeH_.front() || feh > FeH_.back() ||
            logRt < logRt_.front() || logRt > logRt_.back() ||
            logTeff < logTeff_.front() || logTeff > logTeff_.back())
        {
            return SpecsynLib<Policy>::outOfBoundsResult(
                "SpecsynLibWR: star with feh = " + std::to_string(feh) +
                ", logRt = " + std::to_string(logRt) +
                ", logTeff = " + std::to_string(logTeff) +
                " is outside this library's grid");
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
        const auto bTeff = findBracket(logTeff_, logTeff);
        const double logLGrid = trilinearScalar(logLGrid_, bFeh, bRt, bTeff);
        const double scale = std::pow(10.0, logL - logLGrid);
        for (auto& v : result) { v *= scale; }

        // Step 7
        return result;
    }

    // Explicit instantiation for every OOBPolicy value actually used;
    // this keeps the constructor's implementation in this .cpp file,
    // as with every other class in src/specsyn.
    template class SpecsynLibWR<OOBPolicy::Throw>;
    template class SpecsynLibWR<OOBPolicy::silent>;

} // namespace specsyn
