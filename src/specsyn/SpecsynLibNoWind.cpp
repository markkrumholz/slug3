/**
 * @file SpecsynLibNoWind.cpp
 * @author Mark Krumholz
 * @brief Implementation of SpecsynLibNoWind.hpp
 * @date 2026-07-22
 */

#include "SpecsynLibNoWind.hpp"
#include "../tracks/TrackCommons.hpp"
#include "Specsyn.hpp"
#include "SpecsynCommons.hpp"
#include "SpecsynLib.hpp"
#include "SpecsynUtils.hpp"
#include "hdf5.h" // NOLINT(misc-include-cleaner)
#include <algorithm> // NOLINT(misc-include-cleaner) -- see the identical NOLINT on SpecsynLib.cpp's findBracket for why std::ranges::lower_bound needs this
#include <cmath>
#include <cstddef>
#include <mdspan> // NOLINT(misc-include-cleaner)
#include <set>
#include <stdexcept>
#include <string>
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
                    "SpecsynLibNoWind: unable to open dataset " + name);
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
                    "SpecsynLibNoWind: missing required attribute " + name);
            }
            const hid_t attr = H5Aopen(obj, name.c_str(), H5P_DEFAULT);
            if (attr < 0)
            {
                throw std::runtime_error(
                    "SpecsynLibNoWind: unable to open attribute " + name);
            }
            double value = 0.0;
            H5Aread(attr, H5T_NATIVE_DOUBLE, &value);
            H5Aclose(attr);
            return value;
        }
    } // namespace
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
        const double z) :
        SpecsynLib<Policy>(),
        FeH_(this->dim1_),
        logg_(this->dim2_),
        Teff_(this->dim3_),
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

        // Step 1: find the set of spectra matching the input criteria
        auto [fehVals, groupNames] = findMatchingSpectra(
            spectraName, fehMin, fehMax, afe, cfe, microTurb_, r, registryName);
        FeH_ = std::move(fehVals); //NOLINT(cppcoreguidelines-prefer-member-initializer)
        const size_t nfeh = FeH_.size();
        if (nfeh == 0)
        {
            throw std::runtime_error(
                "SpecsynLibNoWind: no spectra found matching the input criteria");
        }

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
            H5Fclose(file);
            throw std::runtime_error(
                "SpecsynLibNoWind: unable to open group wavelengths in " +
                h5path.string());
        }
        auto wlName = "r" + std::to_string(std::llround(r));
        if (H5Lexists(waveGrp, wlName.c_str(), H5P_DEFAULT) <= 0)
        {
            const auto waveNames = listGroupDatasetNames(waveGrp);
            if (waveNames.size() == 1)
            {
                wlName = waveNames.front();
            }
        }
        this->wl_ = readDataset1D(waveGrp, wlName);
        H5Gclose(waveGrp);
        if (this->wl_.empty())
        {
            H5Fclose(file);
            throw std::runtime_error(
                "SpecsynLibNoWind: wavelength grid " + wlName + " in " +
                h5path.string() + " is empty");
        }

        // Step 3: scan every matching group's datasets (without reading
        // their flux data yet) to read the Teff and logg attributes
        // fetch_bosz.py (or any other spectral-library fetch script
        // following the same convention) stores on each one, and
        // thereby the unique sets of logg and Teff values that,
        // together with FeH_, generate the tensor grid on which the
        // library's spectra sit. Not every (FeH, logg, Teff) point in
        // that grid need have a spectrum, so groupEntries also records
        // the (name, Teff, logg) triples for each group, to avoid
        // re-reading these attributes when actually reading the
        // spectra in step 5. Deliberately reads Teff/logg from each
        // dataset's own attributes rather than parsing them out of its
        // name, since not every spectral library on disk can be
        // assumed to encode them in the name at all, let alone in the
        // same way.
        std::vector<std::vector<std::pair<std::string, std::pair<double, double>>>>
            groupEntries(nfeh);
        std::set<double> teffSet;
        std::set<double> loggSet;
        for (size_t f = 0; f < nfeh; ++f)
        {
            const hid_t grp = H5Gopen2(file, groupNames[f].c_str(), H5P_DEFAULT);
            if (grp < 0)
            {
                H5Fclose(file);
                throw std::runtime_error(
                    "SpecsynLibNoWind: unable to open group " + groupNames[f]);
            }
            for (const auto& name : listGroupDatasetNames(grp))
            {
                const hid_t dset = H5Dopen2(grp, name.c_str(), H5P_DEFAULT);
                if (dset < 0)
                {
                    H5Gclose(grp);
                    H5Fclose(file);
                    throw std::runtime_error(
                        "SpecsynLibNoWind: unable to open dataset " + name);
                }
                const double teff = readRequiredScalarAttr(dset, "teff");
                const double logg = readRequiredScalarAttr(dset, "logg");
                H5Dclose(dset);

                groupEntries[f].emplace_back(name, std::make_pair(teff, logg));
                teffSet.insert(teff);
                loggSet.insert(logg);
            }
            H5Gclose(grp);
        }
        Teff_.assign(teffSet.begin(), teffSet.end());
        logg_.assign(loggSet.begin(), loggSet.end());
        const size_t nteff = Teff_.size();
        const size_t nlogg = logg_.size();

        // Step 4: allocate storage for the (FeH, logg, Teff) tensor
        // grid of spectra, and point grid_ at it for convenient
        // indexing. Every entry starts out as an empty vector, which
        // is how unpopulated grid points are represented once step 5
        // has filled in the populated ones.
        using SpectraGrid = typename SpecsynLib<Policy>::SpectraGrid;
        this->spectra_.assign(nfeh * nlogg * nteff, std::vector<double>{});
        this->grid_ = SpectraGrid(this->spectra_.data(), nfeh, nlogg, nteff);

        // Step 5: read the actual spectra, placing each one at its
        // point in the tensor grid
        for (size_t f = 0; f < nfeh; ++f)
        {
            const hid_t grp = H5Gopen2(file, groupNames[f].c_str(), H5P_DEFAULT);
            if (grp < 0)
            {
                H5Fclose(file);
                throw std::runtime_error(
                    "SpecsynLibNoWind: unable to open group " + groupNames[f]);
            }
            for (const auto& [name, teffLogg] : groupEntries[f])
            {
                const auto [teff, logg] = teffLogg;
                const auto iTeff = static_cast<size_t>(
                    std::ranges::lower_bound(Teff_, teff) - Teff_.begin());
                const auto iLogg = static_cast<size_t>(
                    std::ranges::lower_bound(logg_, logg) - logg_.begin());
                this->grid_[f, iLogg, iTeff] = readDataset1D(grp, name);
            }
            H5Gclose(grp);
        }

        H5Fclose(file);

        // NOLINTEND(misc-include-cleaner)
    }

    template <OOBPolicy Policy>
    auto SpecsynLibNoWind<Policy>::spec(const Specsyn::StarData& props, const double feh) const -> std::vector<double>
    {
        // Step 1: check feh and Teff against the grid's bounds before
        // paying for the surface-area/log(g) calculation below
        const double logTeff = props[static_cast<size_t>(tracks::FieldIdx::logTe)]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- StarData is fixed-size, index is compile-time-known
        const double teff = std::pow(10.0, logTeff);
        if (feh < FeH_.front() || feh > FeH_.back() ||
            teff < Teff_.front() || teff > Teff_.back())
        {
            return SpecsynLib<Policy>::outOfBoundsResult(
                "SpecsynLibNoWind: star with feh = " + std::to_string(feh) +
                ", Teff = " + std::to_string(teff) +
                " K is outside this library's grid");
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
        auto result = this->SpecsynLib<Policy>::spec(feh, logg, teff);
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
