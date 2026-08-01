/**
 * @file SpecsynLibNoWind.cpp
 * @author Mark Krumholz
 * @brief Implementation of SpecsynLibNoWind.hpp
 * @date 2026-07-22
 */

#include "SpecsynLibNoWind.hpp"
#include "../io/SimControls.hpp"
#include "../tracks/TrackCommons.hpp"
#include "../utils/HDF5Utils.hpp"
#include "../utils/MiscUtils.hpp"
#include "Specsyn.hpp"
#include "SpecsynCommons.hpp"
#include "SpecsynLib.hpp"
#include "SpecsynUtils.hpp"
#include "hdf5.h" // NOLINT(misc-include-cleaner)
#include <algorithm> // NOLINT(misc-include-cleaner) -- see the identical NOLINT on SpecsynLib.cpp's findBracket for why std::ranges::lower_bound needs this
#include <cmath>
#include <cstddef>
#include <iterator>
#include <map>
#include <mdspan> // NOLINT(misc-include-cleaner)
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace specsyn
{
    // Maps a (logTeff, logg) grid point to the name of the dataset
    // holding its spectrum, for one HDF5 group (i.e. one feh/afe
    // combination).
    using DatasetMap = std::map<std::pair<double, double>, std::string>;

    // Holds everything the constructor needs to finish building a
    // SpecsynLibNoWind once the (FeH, logg, logTeff) grid and its
    // spectra have been assembled -- shared by both the exact-AFe-match
    // and interpolated-AFe construction paths below, factored out of
    // the constructor itself to keep its cognitive complexity down.
    struct GridBuildResult
    {
        std::vector<double> feh_;
        std::vector<double> logg_;
        std::vector<double> logTeff_;
        std::vector<std::vector<double>> spectra_;
    };

    // Helper used by the interpolating AFe path: scan an HDF5 group's
    // datasets and return a map from (logTeff, logg) to dataset name,
    // updating logTeffSet and loggSet with the values found.
    // NOLINTBEGIN(misc-include-cleaner)
    static auto scanGroupForTeffLogg(
        const hid_t file,
        const std::string& groupName,
        std::set<double>& logTeffSet,
        std::set<double>& loggSet)
        -> DatasetMap
    {
        const hid_t grp = H5Gopen2(file, groupName.c_str(), H5P_DEFAULT);
        if (grp < 0)
        {
            throw std::runtime_error(
                "SpecsynLibNoWind: unable to open group " + groupName);
        }
        DatasetMap entries;
        for (const auto& name : utils::listGroupDatasetNames(grp))
        {
            const hid_t dset = H5Dopen2(grp, name.c_str(), H5P_DEFAULT);
            if (dset < 0)
            {
                H5Gclose(grp);
                throw std::runtime_error(
                    "SpecsynLibNoWind: unable to open dataset " + name);
            }
            const double logTeff = std::log10(utils::readRequiredScalarAttr(dset, "teff", "SpecsynLibNoWind"));
            const double logg = utils::readRequiredScalarAttr(dset, "logg", "SpecsynLibNoWind");
            H5Dclose(dset);
            entries[{logTeff, logg}] = name;
            logTeffSet.insert(logTeff);
            loggSet.insert(logg);
        }
        H5Gclose(grp);
        return entries;
    }
    // NOLINTEND(misc-include-cleaner)

    // Builds the (FeH, logg, logTeff) grid and its spectra for the
    // exact-AFe-match construction path: fehVals/groupNames are the
    // groups already known (via findMatchingSpectra) to match afe
    // exactly, so this just needs to scan each one's (logTeff, logg)
    // coverage and read every spectrum found into its point in the
    // tensor grid.
    // NOLINTBEGIN(misc-include-cleaner)
    static auto buildExactMatchGrid(
        const hid_t file,
        std::vector<double> fehVals,
        const std::vector<std::string>& groupNames) -> GridBuildResult
    {
        const size_t nfeh = fehVals.size();
        std::vector<DatasetMap> groupEntries(nfeh);
        std::set<double> logTeffSet;
        std::set<double> loggSet;
        for (size_t f = 0; f < nfeh; ++f)
        {
            groupEntries[f] =
                scanGroupForTeffLogg(file, groupNames[f], logTeffSet, loggSet);
        }

        GridBuildResult result;
        result.feh_ = std::move(fehVals);
        result.logTeff_.assign(logTeffSet.begin(), logTeffSet.end());
        result.logg_.assign(loggSet.begin(), loggSet.end());
        const size_t nteff = result.logTeff_.size();
        const size_t nlogg = result.logg_.size();
        result.spectra_.assign(nfeh * nlogg * nteff, std::vector<double>{});

        for (size_t f = 0; f < nfeh; ++f)
        {
            const hid_t grp = H5Gopen2(file, groupNames[f].c_str(), H5P_DEFAULT);
            if (grp < 0)
            {
                throw std::runtime_error(
                    "SpecsynLibNoWind: unable to open group " + groupNames[f]);
            }
            for (const auto& [teffLogg, name] : groupEntries[f])
            {
                const auto [logTeff, logg] = teffLogg;
                const auto iTeff = static_cast<size_t>(
                    std::ranges::lower_bound(result.logTeff_, logTeff) -
                    result.logTeff_.begin());
                const auto iLogg = static_cast<size_t>(
                    std::ranges::lower_bound(result.logg_, logg) -
                    result.logg_.begin());
                result.spectra_[(f * nlogg + iLogg) * nteff + iTeff] =
                    utils::readDataset1D(grp, name, "SpecsynLibNoWind");
            }
            H5Gclose(grp);
        }
        return result;
    }
    // NOLINTEND(misc-include-cleaner)

    // Finds the two values in the sorted vector afeVals that bracket
    // afe; throws if afe lies outside the range afeVals covers.
    static auto findAfeBracket(const std::vector<double>& afeVals, const double afe)
        -> std::pair<double, double>
    {
        const auto hiIt = std::ranges::upper_bound(afeVals, afe); //NOLINT(misc-include-cleaner)
        if (hiIt == afeVals.end() || hiIt == afeVals.begin())
        {
            throw std::runtime_error(
                "SpecsynLibNoWind: no spectra found matching the input criteria "
                "(requested afe=" + std::to_string(afe) +
                " is outside the library's available range)");
        }
        return { *std::prev(hiIt), *hiIt };
    }

    // Builds the sorted union of two FeH value sets, used as the FeH_
    // axis of the interpolated-AFe grid.
    static auto buildUnionFeh(
        const std::vector<double>& loFehVals, const std::vector<double>& hiFehVals)
        -> std::vector<double>
    {
        std::vector<double> unionFeh = loFehVals;
        for (const double v : hiFehVals)
        {
            if (std::ranges::find_if(unionFeh, [v](const double u) {
                    return utils::approxEqual(u, v);
                }) == unionFeh.end())
            {
                unionFeh.push_back(v);
            }
        }
        std::ranges::sort(unionFeh);
        return unionFeh;
    }

    // Maps a feh value to its index in vals, or nullopt if it is not
    // present there.
    static auto findFehIndex(const std::vector<double>& vals, const double feh)
        -> std::optional<size_t>
    {
        for (size_t i = 0; i < vals.size(); ++i)
        {
            if (utils::approxEqual(vals[i], feh))
            {
                return i;
            }
        }
        return std::nullopt;
    }

    // For grid row f, reads and linearly interpolates every (logTeff,
    // logg) point present in both bracketing groups' dataset maps,
    // writing the result into spectra at that point's flat index.
    // Points missing from either bracket are left as-is (empty).
    // NOLINTBEGIN(misc-include-cleaner)
    static void interpolateFehRow(
        const hid_t loGrp, const hid_t hiGrp,
        const DatasetMap& loDsets, const DatasetMap& hiDsets,
        const std::vector<double>& logTeff, const std::vector<double>& logg,
        const double alpha, const size_t f,
        std::vector<std::vector<double>>& spectra)
    {
        const size_t nteff = logTeff.size();
        const size_t nlogg = logg.size();
        for (size_t t = 0; t < nteff; ++t)
        {
            for (size_t g = 0; g < nlogg; ++g)
            {
                const auto key = std::make_pair(logTeff[t], logg[g]);
                const auto loIt = loDsets.find(key);
                const auto hiIt = hiDsets.find(key);
                if (loIt == loDsets.end() || hiIt == hiDsets.end())
                {
                    continue; // missing from one bracket → leave empty
                }

                auto loFlux = utils::readDataset1D(loGrp, loIt->second, "SpecsynLibNoWind");
                auto hiFlux = utils::readDataset1D(hiGrp, hiIt->second, "SpecsynLibNoWind");
                const size_t nw = loFlux.size();
                std::vector<double> interp(nw);
                for (size_t w = 0; w < nw; ++w)
                {
                    interp[w] = (1.0 - alpha) * loFlux[w] + alpha * hiFlux[w];
                }
                spectra[(f * nlogg + g) * nteff + t] = std::move(interp);
            }
        }
    }
    // NOLINTEND(misc-include-cleaner)

    // Builds the (FeH, logg, logTeff) grid and its spectra for the
    // interpolated-AFe construction path: no group matches afe
    // exactly, so this brackets it between the two nearest available
    // afe values and linearly interpolates every grid point present in
    // both.
    // NOLINTBEGIN(misc-include-cleaner)
    static auto buildInterpolatedGrid(
        const hid_t file,
        const std::string& spectraName,
        const double fehMin, const double fehMax,
        const double afe, const double cfe, const double microTurb,
        const double r, const std::string& registryName) -> GridBuildResult
    {
        const auto afeVals = findAfeValues(spectraName, cfe, microTurb, r, registryName);
        if (afeVals.empty())
        {
            throw std::runtime_error(
                "SpecsynLibNoWind: no spectra found matching the input criteria");
        }
        const auto [loAfe, hiAfe] = findAfeBracket(afeVals, afe);
        const double alpha = (afe - loAfe) / (hiAfe - loAfe);

        auto [loFehVals, loGroupNames] = findMatchingSpectra(
            spectraName, fehMin, fehMax, loAfe, cfe, microTurb, r, registryName);
        auto [hiFehVals, hiGroupNames] = findMatchingSpectra(
            spectraName, fehMin, fehMax, hiAfe, cfe, microTurb, r, registryName);
        if (loFehVals.empty() || hiFehVals.empty())
        {
            throw std::runtime_error(
                "SpecsynLibNoWind: no spectra found matching the input criteria");
        }

        GridBuildResult result;
        result.feh_ = buildUnionFeh(loFehVals, hiFehVals);
        const size_t nfeh = result.feh_.size();

        std::set<double> logTeffSet;
        std::set<double> loggSet;
        std::vector<DatasetMap> loEntries(loFehVals.size());
        for (size_t i = 0; i < loFehVals.size(); ++i)
        {
            loEntries[i] =
                scanGroupForTeffLogg(file, loGroupNames[i], logTeffSet, loggSet);
        }
        std::vector<DatasetMap> hiEntries(hiFehVals.size());
        for (size_t i = 0; i < hiFehVals.size(); ++i)
        {
            hiEntries[i] =
                scanGroupForTeffLogg(file, hiGroupNames[i], logTeffSet, loggSet);
        }

        result.logTeff_.assign(logTeffSet.begin(), logTeffSet.end());
        result.logg_.assign(loggSet.begin(), loggSet.end());
        const size_t nteff = result.logTeff_.size();
        const size_t nlogg = result.logg_.size();
        result.spectra_.assign(nfeh * nlogg * nteff, std::vector<double>{});

        // For each feh grid point present in both bracketing sets, read
        // and interpolate every (logTeff, logg) point both contain;
        // points missing from either bracketing feh value (not just
        // either bracketing spectrum) are left empty entirely.
        for (size_t f = 0; f < nfeh; ++f)
        {
            const auto loFIdx = findFehIndex(loFehVals, result.feh_[f]);
            const auto hiFIdx = findFehIndex(hiFehVals, result.feh_[f]);
            if (!loFIdx || !hiFIdx)
            {
                continue;
            }

            const hid_t loGrp = H5Gopen2(file, loGroupNames[*loFIdx].c_str(), H5P_DEFAULT);
            const hid_t hiGrp = H5Gopen2(file, hiGroupNames[*hiFIdx].c_str(), H5P_DEFAULT);
            if (loGrp < 0 || hiGrp < 0)
            {
                if (loGrp >= 0)
                {
                    H5Gclose(loGrp);
                }
                if (hiGrp >= 0)
                {
                    H5Gclose(hiGrp);
                }
                throw std::runtime_error(
                    "SpecsynLibNoWind: unable to open bracketing afe group");
            }

            interpolateFehRow(loGrp, hiGrp, loEntries[*loFIdx], hiEntries[*hiFIdx],
                result.logTeff_, result.logg_, alpha, f, result.spectra_);

            H5Gclose(loGrp);
            H5Gclose(hiGrp);
        }

        return result;
    }
    // NOLINTEND(misc-include-cleaner)

    template <OOBPolicy Policy>
    SpecsynLibNoWind<Policy>::SpecsynLibNoWind(
        const std::string& spectraName,
        const double fehMin,
        const double fehMax,
        const double afe,
        const double cfe,
        const double microTurb,
        const double r,
        const std::string& registryName,
        double wlMin,
        double wlMax,
        const std::size_t nWl,
        const double z,
        const io::SimControls& controls) :
        SpecsynLib<Policy>(),
        FeH_(this->dim1_),
        logg_(this->dim2_),
        logTeff_(this->dim3_),
        AFe_(afe),
        CFe_(cfe),
        // A NaN microTurb means "use this library's own default":
        // resolved here, in the member initializer list, rather than
        // in the constructor body, so that findMatchingSpectra below
        // (which uses microTurb_ as a filter) sees the resolved value
        // too, not the NaN sentinel
        microTurb_(std::isnan(microTurb) ?
            getMicroDefault(spectraName, registryName) : microTurb),
        r_(r)
    {
        this->z_ = z;
        this->intRelTol_ = controls.intRelTol();
        this->intAbsTol_ = controls.intAbsTol();
        this->intMaxIter_ = controls.intMaxIter();

        // Step 1: find the set of spectra matching the input criteria
        auto [fehVals, groupNames] = findMatchingSpectra(
            spectraName, fehMin, fehMax, afe, cfe, microTurb_, r, registryName);

        // Re-derive the path to the HDF5 file holding these spectra, the
        // same way findMatchingSpectra does internally
        auto [registry, registryPath] = parseRegistry(registryName);
        const auto h5name =
            registry.at_path(spectraName).at_path("file").value_or(std::string{});
        const auto h5path = registryPath.parent_path() / h5name;

        // Suppress clang-tidy warnings iun this namespace caused by just
        // including hdf5.h, instead of the individual HDF5 headers,
        // since this is the paradigm that HDF5 wants
        // NOLINTBEGIN(misc-include-cleaner)

        const hid_t file = H5Fopen(h5path.string().c_str(),
            H5F_ACC_RDONLY, H5P_DEFAULT);
        if (file < 0)
        {
            throw std::runtime_error(
                "SpecsynLibNoWind: unable to open HDF5 file " + h5path.string());
        }

        // Everything from here through the matching H5Fclose below is
        // wrapped in a try/catch so the file handle always gets closed,
        // regardless of which of the many things that can go wrong
        // while reading the wavelength grid or building the (FeH,
        // logg, logTeff) spectra grid actually does.
        try
        {
            // Step 2: read the wavelength grid shared by every matching
            // spectrum, normally stored under wavelengths/r<r> (all matches
            // share the same r, since it is one of the matching criteria).
            // If no such dataset exists but the library only has a single
            // wavelength grid at all, fall back to that sole entry instead
            // -- e.g. TLUSTY, whose downsampling means no "r" value is
            // truly meaningful (see fetch_tlusty.py), stores its one grid
            // under a non-numeric name instead. This mirrors
            // findMatchingSpectra's treatment of a library with no r
            // attribute on its spectra groups: absent a way to tell r
            // values apart, anything the caller asks for matches.
            const hid_t waveGrp = H5Gopen2(file, "wavelengths", H5P_DEFAULT);
            if (waveGrp < 0)
            {
                throw std::runtime_error(
                    "SpecsynLibNoWind: unable to open group wavelengths in " +
                    h5path.string());
            }
            auto wlName = "r" + std::to_string(std::llround(r));
            if (H5Lexists(waveGrp, wlName.c_str(), H5P_DEFAULT) <= 0)
            {
                const auto waveNames = utils::listGroupDatasetNames(waveGrp);
                if (waveNames.size() == 1)
                {
                    wlName = waveNames.front();
                }
            }
            this->wl_ = utils::readDataset1D(waveGrp, wlName, "SpecsynLibNoWind");
            H5Gclose(waveGrp);
            if (this->wl_.empty())
            {
                throw std::runtime_error(
                    "SpecsynLibNoWind: wavelength grid " + wlName + " in " +
                    h5path.string() + " is empty");
            }

            // Steps 3-5: build the (FeH, logg, logTeff) tensor grid and
            // its spectra, via the exact-AFe-match path if Step 1 found
            // any groups matching afe exactly, or else the
            // interpolated-AFe path, which brackets afe between the two
            // nearest available values and linearly interpolates. See
            // buildExactMatchGrid and buildInterpolatedGrid above for
            // the details of each.
            auto result = !fehVals.empty() ?
                buildExactMatchGrid(file, std::move(fehVals), groupNames) :
                buildInterpolatedGrid(file, spectraName, fehMin, fehMax, afe,
                    cfe, microTurb_, r, registryName);

            FeH_ = std::move(result.feh_);         //NOLINT(cppcoreguidelines-prefer-member-initializer)
            logg_ = std::move(result.logg_);       //NOLINT(cppcoreguidelines-prefer-member-initializer)
            logTeff_ = std::move(result.logTeff_); //NOLINT(cppcoreguidelines-prefer-member-initializer)
            this->spectra_ = std::move(result.spectra_);
            using SpectraGrid = typename SpecsynLib<Policy>::SpectraGrid;
            this->grid_ = SpectraGrid(
                this->spectra_.data(), FeH_.size(), logg_.size(), logTeff_.size());
        }
        catch (...)
        {
            H5Fclose(file);
            throw;
        }

        H5Fclose(file);

        // NOLINTEND(misc-include-cleaner)

        // If the caller requested an explicit output grid rather than
        // this library's own native one, resample onto it now. wlMin
        // == 0 here means only nWl was actually requested (wlMin/
        // wlMax are still at SimPhysics::readSpectra's "not supplied"
        // sentinel), so fall back to this library's own just-read
        // native range in that case, keeping the caller's requested
        // point count.
        if (nWl != 0)
        {
            if (wlMin == 0.0)
            {
                wlMin = this->wl_.front();
                wlMax = this->wl_.back();
            }
            this->resample(utils::logspace(wlMin, wlMax, nWl));
        }
    }

    template <OOBPolicy Policy>
    auto SpecsynLibNoWind<Policy>::spec(const Specsyn::StarData& props, const double feh) const -> std::vector<double>
    {
        // Step 1: check feh and log(Teff) against the grid's bounds
        // before paying for the surface-area/log(g) calculation below
        const double logTeff = props[static_cast<size_t>(tracks::FieldIdx::logTe)]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- StarData is fixed-size, index is compile-time-known
        if (feh < FeH_.front() || feh > FeH_.back() ||
            logTeff < logTeff_.front() || logTeff > logTeff_.back())
        {
            return SpecsynLib<Policy>::outOfBoundsResult(
                "SpecsynLibNoWind: star with feh = " + std::to_string(feh) +
                ", log(Teff) = " + std::to_string(logTeff) +
                " is outside this library's grid");
        }

        // Step 2: surface area and log(g), then bounds-check log(g)
        const auto [area, logg] = this->getSAandLogg(props);
        if (logg < logg_.front() || logg > logg_.back())
        {
            return SpecsynLib<Policy>::outOfBoundsResult(
                "SpecsynLibNoWind: star with log(g) = " + std::to_string(logg) +
                " is outside this library's grid");
        }

        // Steps 3-5 (bracket-finding, gap-checking, and trilinear
        // interpolation) are handled entirely by the parent class,
        // which knows nothing about surface area -- scale its result
        // (step 6) to convert specific flux at the surface into
        // specific luminosity
        auto result = this->SpecsynLib<Policy>::spec(feh, logg, logTeff);
        for (auto& v : result) { v *= area; }
        return result;
    }

    // Explicit instantiation for every OOBPolicy value actually used;
    // this keeps the constructor's implementation in this .cpp file,
    // as with every other class in src/specsyn, rather than forcing it
    // into the header just because it is now a template.
    template class SpecsynLibNoWind<OOBPolicy::raise>;
    template class SpecsynLibNoWind<OOBPolicy::silent>;
    template class SpecsynLibNoWind<OOBPolicy::coerce>;

} // namespace specsyn
