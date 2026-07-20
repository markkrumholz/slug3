/**
 * @file SpecsynLib.cpp
 * @author Mark Krumholz
 * @brief Implementation of SpecsynLib.hpp
 * @date 2026-07-20
 */

#include "SpecsynLib.hpp"
#include "SpecsynUtils.hpp"
#include "hdf5.h" // NOLINT(misc-include-cleaner)
#include <algorithm>
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
                    "SpecsynLib: unable to open dataset " + name);
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
         * @brief Parse the Teff and logg values encoded in a spectrum dataset's name
         * @param name Dataset name, of the form "t<Teff>_g<logg>"
         *   (e.g. "t10000_g+2.0"), the naming convention used throughout
         *   the BOSZ library fetched by data/tools/fetch_bosz.py
         * @returns The (Teff, logg) pair encoded in name
         */
        auto parseSpectrumName(const std::string& name) //NOLINT(llvm-prefer-static-over-anonymous-namespace)
            -> std::pair<double, double>
        {
            const auto gPos = name.find("_g");
            if (name.empty() || name.front() != 't' || gPos == std::string::npos)
            {
                throw std::runtime_error(
                    "SpecsynLib: dataset name " + name +
                    " does not match the expected t<Teff>_g<logg> format");
            }
            const double teff = std::stod(name.substr(1, gPos - 1));
            const double logg = std::stod(name.substr(gPos + 2));
            return { teff, logg };
        }
    } // namespace
    // NOLINTEND(misc-include-cleaner)

    template <OOBPolicy Policy>
    SpecsynLib<Policy>::SpecsynLib(
        const std::string& spectraName,
        const double fehMin,
        const double fehMax,
        const double afe,
        const double cfe,
        const double microTurb,
        const double r,
        const std::string& registryName,
        const double z) :
        Specsyn(z),
        AFe_(afe),
        CFe_(cfe),
        microTurb_(microTurb),
        r_(r)
    {
        // Step 1: find the set of spectra matching the input criteria
        auto [fehVals, groupNames] = findMatchingSpectra(
            spectraName, fehMin, fehMax, afe, cfe, microTurb, r, registryName);
        FeH_ = std::move(fehVals); //NOLINT(cppcoreguidelines-prefer-member-initializer)
        const size_t nfeh = FeH_.size();
        if (nfeh == 0)
        {
            throw std::runtime_error(
                "SpecsynLib: no spectra found matching the input criteria");
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
                "SpecsynLib: unable to open HDF5 file " + h5path.string());
        }

        // Step 2: read the wavelength grid shared by every matching
        // spectrum, stored under wavelengths/r<r> (all matches share
        // the same r, since it is one of the matching criteria)
        const hid_t waveGrp = H5Gopen2(file, "wavelengths", H5P_DEFAULT);
        if (waveGrp < 0)
        {
            H5Fclose(file);
            throw std::runtime_error(
                "SpecsynLib: unable to open group wavelengths in " +
                h5path.string());
        }
        const auto wlName = "r" + std::to_string(static_cast<long long>(std::llround(r)));
        wl_ = readDataset1D(waveGrp, wlName);
        H5Gclose(waveGrp);
        if (wl_.empty())
        {
            H5Fclose(file);
            throw std::runtime_error(
                "SpecsynLib: wavelength grid " + wlName + " in " +
                h5path.string() + " is empty");
        }

        // Step 3: scan every matching group's datasets (without reading
        // their data yet) to find the Teff and logg values encoded in
        // their names, and thereby the unique sets of logg and Teff
        // values that, together with FeH_, generate the tensor grid on
        // which the library's spectra sit. Not every (FeH, logg, Teff)
        // point in that grid need have a spectrum, so groupEntries also
        // records the (name, Teff, logg) triples for each group, to
        // avoid re-parsing dataset names when actually reading the
        // spectra in step 5.
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
                    "SpecsynLib: unable to open group " + groupNames[f]);
            }
            for (const auto& name : listGroupDatasetNames(grp))
            {
                const auto teffLogg = parseSpectrumName(name);
                groupEntries[f].emplace_back(name, teffLogg);
                teffSet.insert(teffLogg.first);
                loggSet.insert(teffLogg.second);
            }
            H5Gclose(grp);
        }
        Teff_.assign(teffSet.begin(), teffSet.end());
        logg_.assign(loggSet.begin(), loggSet.end());
        const size_t nteff = Teff_.size();
        const size_t nlogg = logg_.size();

        // Step 4: allocate storage for the (FeH, logg, Teff) tensor
        // grid of spectra, and wrap it in an mdspan for convenient
        // indexing. Every entry starts out as an empty vector, which
        // is how unpopulated grid points are represented once step 5
        // has filled in the populated ones.
        using SpectraGrid = std::mdspan<std::vector<double>, std::dextents<size_t, 3>>;
        spectra_.assign(nfeh * nlogg * nteff, std::vector<double>{});
        const SpectraGrid grid(spectra_.data(), nfeh, nlogg, nteff);

        // Step 5: read the actual spectra, placing each one at its
        // point in the tensor grid
        for (size_t f = 0; f < nfeh; ++f)
        {
            const hid_t grp = H5Gopen2(file, groupNames[f].c_str(), H5P_DEFAULT);
            if (grp < 0)
            {
                H5Fclose(file);
                throw std::runtime_error(
                    "SpecsynLib: unable to open group " + groupNames[f]);
            }
            for (const auto& [name, teffLogg] : groupEntries[f])
            {
                const auto [teff, logg] = teffLogg;
                const auto iTeff = static_cast<size_t>(
                    std::ranges::lower_bound(Teff_, teff) - Teff_.begin());
                const auto iLogg = static_cast<size_t>(
                    std::ranges::lower_bound(logg_, logg) - logg_.begin());
                grid[f, iLogg, iTeff] = readDataset1D(grp, name);
            }
            H5Gclose(grp);
        }

        H5Fclose(file);

        // NOLINTEND(misc-include-cleaner)
    }

    // Explicit instantiation for every OOBPolicy value actually used;
    // this keeps the constructor's implementation in this .cpp file,
    // as with every other class in src/specsyn, rather than forcing it
    // into the header just because it is now a template.
    template class SpecsynLib<OOBPolicy::Throw>;
    template class SpecsynLib<OOBPolicy::Silent>;

} // namespace specsyn
