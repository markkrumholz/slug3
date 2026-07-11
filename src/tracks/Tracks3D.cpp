/**
 * @file Tracks3D.cpp
 * @author Mark Krumholz
 * @brief Implementation of Tracks3D.hpp
 * @date 2024-07-10
 */

#include "Tracks3D.hpp"
#include "../interpolation/Mesh3DInterpolator.hpp"
#include "TrackCommons.hpp"
#include "TrackUtils.hpp"
#include "hdf5.h"  // NOLINT(misc-include-cleaner)
#include <algorithm>
#include <array>
#include <cstddef>
#include <format>
#include <gsl/gsl_interp.h>
#include <iterator>
#include <mdspan>  // NOLINT(misc-include-cleaner)
#include <memory>
#include <ranges>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// NOLINTBEGIN(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)

namespace tracks
{

    // FieldIdx::nTrackQty as a size_t
    constexpr size_t nQty = static_cast<size_t>(FieldIdx::nTrackQty);

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
        auto readDataset1D(const hid_t grp, const std::string& name)  //NOLINT(llvm-prefer-static-over-anonymous-namespace)
            -> std::vector<double>
        {
            const hid_t dset = H5Dopen2(grp, name.c_str(), H5P_DEFAULT);
            if (dset < 0)
            {
                throw std::runtime_error(
                    "Tracks3D: unable to open dataset " + name);
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
         * @brief Read a full 2D double dataset from an HDF5 group
         * @param grp Handle to the group containing the dataset
         * @param name Name of the dataset
         * @returns The dataset contents, and its (nrow, ncol) shape
         */
        auto readDataset2D(const hid_t grp, const std::string& name)  //NOLINT(llvm-prefer-static-over-anonymous-namespace)
            -> std::pair<std::vector<double>, std::pair<size_t, size_t>>
        {
            const hid_t dset = H5Dopen2(grp, name.c_str(), H5P_DEFAULT);
            if (dset < 0)
            {
                throw std::runtime_error(
                    "Tracks3D: unable to open dataset " + name);
            }
            const hid_t space = H5Dget_space(dset);
            std::array<hsize_t,2> dims = {0, 0};
            H5Sget_simple_extent_dims(space, static_cast<hsize_t *>(dims.data()), nullptr);
            std::vector<double> data(dims[0] * dims[1]);
            H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                H5P_DEFAULT, data.data());
            H5Sclose(space);
            H5Dclose(dset);
            return { std::move(data), { dims[0], dims[1] } };
        }

        /**
         * @brief Read the field_names attribute of an HDF5 group
         * @param grp Handle to the group
         * @returns The names of the fields stored in each track dataset,
         *   in the order in which they appear
         */
        auto readFieldNames(const hid_t grp) -> std::vector<std::string> //NOLINT(llvm-prefer-static-over-anonymous-namespace)
        {
            const hid_t attr = H5Aopen(grp, "field_names", H5P_DEFAULT);
            if (attr < 0)
            {
                throw std::runtime_error(
                    "Tracks3D: unable to open field_names attribute");
            }
            const hid_t aspace = H5Aget_space(attr);
            const auto npoints =
                static_cast<size_t>(H5Sget_simple_extent_npoints(aspace));

            // Use the attribute's own (variable-length, UTF-8) type as
            // the memory type; building a fresh H5T_C_S1-based type
            // instead fails to convert because the on-disk character
            // set is UTF-8, not ASCII
            const hid_t memtype = H5Aget_type(attr);

            std::vector<char*> buf(npoints);
            H5Aread(attr, memtype, static_cast<void *>(buf.data())); // NOLINT(bugprone-multi-level-implicit-pointer-conversion)
            std::vector<std::string> names;
            names.reserve(npoints);
            for (const auto* s : buf) { names.emplace_back(s); }

            // Use H5Dvlen_reclaim rather than its replacement,
            // H5Treclaim, since the latter was only added in HDF5
            // 1.14 and isn't available in the older HDF5 that Ubuntu's
            // libhdf5-dev package ships
            H5Dvlen_reclaim(memtype, aspace, H5P_DEFAULT,
                static_cast<void *>(buf.data())); // NOLINT(bugprone-multi-level-implicit-pointer-conversion)
            H5Tclose(memtype);
            H5Sclose(aspace);
            H5Aclose(attr);

            return names;
        }

        /**
         * @brief Read and pad the track data for every matching feh group
         * @param file Handle to the open HDF5 file containing the tracks
         * @param groupNames Names of the matching groups, in ascending feh order
         * @param massData Sorted masses shared by every group
         * @param fieldNames Names of the fields in each track dataset
         * @param ageIdx Index of the 'age' field within fieldNames
         * @param qtyIdx Indices within fieldNames of the nQty quantities to read
         * @param nt Number of time points to pad every track to
         * @returns The times and field data, both laid out with shape
         *   (nmass, nt, nfeh[, nQty]) -- i.e. mass varying slowest, time
         *   next, then feh, and (for field data) quantity fastest
         * @details
         * Loops over every matching feh group in turn, and within each,
         * over every mass from lowest to highest, padding the end of
         * any track with fewer than nt time points by repeating its
         * final value. On error, closes file (and, if already open,
         * the current group) before throwing, since the caller opened
         * file and is not expecting to have to close it itself if this
         * function throws.
         */
        auto readAllTrackData(  //NOLINT(llvm-prefer-static-over-anonymous-namespace)
            const hid_t file,
            const std::vector<std::string>& groupNames,
            const std::vector<double>& massData,
            const std::vector<std::string>& fieldNames,
            const std::ptrdiff_t ageIdx,
            const std::vector<size_t>& qtyIdx,
            const size_t nt)
            -> std::pair<std::vector<double>, std::vector<double>>
        {
            using Array2D = std::mdspan<double, std::dextents<size_t, 2>>;
            using Array3D = interp::Mesh3DInterpolator<nQty>::Array3D;
            using Array4D = interp::Mesh3DInterpolator<nQty>::Array4D;

            const size_t nmass = massData.size();
            const size_t nfeh = groupNames.size();

            std::vector<double> timesData(nmass * nt * nfeh);
            const Array3D times(timesData.data(), nmass, nt, nfeh);
            std::vector<double> fieldDataVec(nmass * nt * nfeh * nQty);
            const Array4D fieldData(fieldDataVec.data(), nmass, nt, nfeh, nQty);

            for (size_t f = 0; f < nfeh; ++f)
            {
                const hid_t grp = H5Gopen2(file, groupNames[f].c_str(), H5P_DEFAULT);
                if (grp < 0)
                {
                    H5Fclose(file);
                    throw std::runtime_error(
                        "Tracks3D: unable to open group " + groupNames[f]);
                }

                for (size_t i = 0; i < nmass; ++i)
                {
                    const auto name = std::format("track_m{:.3f}", massData[i]);
                    auto [data, shape] = readDataset2D(grp, name);
                    const auto [nrow, ncol] = shape;
                    if (ncol != fieldNames.size())
                    {
                        H5Gclose(grp);
                        H5Fclose(file);
                        throw std::runtime_error(
                            "Tracks3D: dataset " + name +
                            " has an unexpected number of fields");
                    }
                    const Array2D track(data.data(), nrow, ncol);

                    for (size_t j = 0; j < nt; ++j)
                    {
                        const size_t src = std::min(j, nrow - 1);
                        times[i, j, f] = track[src, ageIdx];
                        for (size_t k = 0; k < nQty; ++k)
                        {
                            fieldData[i, j, f, k] = track[src, qtyIdx[k]];
                        }
                    }
                }

                H5Gclose(grp);
            }

            return { std::move(timesData), std::move(fieldDataVec) };
        }
    } // namespace
    // NOLINTEND(misc-include-cleaner)

    namespace
    {
        /**
         * @brief Transpose track data into the layout Mesh3DInterpolator requires
         * @param times Times, with shape (nmass, nt, nfeh)
         * @param fieldData Field data, with shape (nmass, nt, nfeh, nQty)
         * @param nmass Number of masses
         * @param nt Number of time points
         * @param nfeh Number of feh values
         * @returns The transposed x and f arrays, with shape
         *   (nt, nmass, nfeh) and (nt, nmass, nfeh, nQty) respectively
         * @details
         * Mesh3DInterpolator requires its x and f arrays to have shape
         * (nt, nmass, nfeh) and (nt, nmass, nfeh, nQty) respectively --
         * the transpose of times and fieldData in their first two
         * dimensions -- because masses and feh, not times, form the
         * tensor (shared) directions of the mesh.
         */
        auto transposeTrackData(  //NOLINT(llvm-prefer-static-over-anonymous-namespace)
            const interp::Mesh3DInterpolator<nQty>::Array3D& times,
            const interp::Mesh3DInterpolator<nQty>::Array4D& fieldData,
            const size_t nmass,
            const size_t nt,
            const size_t nfeh)
            -> std::pair<std::vector<double>, std::vector<double>>
        {
            using Array3D = interp::Mesh3DInterpolator<nQty>::Array3D;
            using Array4D = interp::Mesh3DInterpolator<nQty>::Array4D;

            std::vector<double> xData(nt * nmass * nfeh);
            const Array3D x(xData.data(), nt, nmass, nfeh);
            std::vector<double> fData(nt * nmass * nfeh * nQty);
            const Array4D f(fData.data(), nt, nmass, nfeh, nQty);
            for (size_t i = 0; i < nmass; ++i)
            {
                for (size_t j = 0; j < nt; ++j)
                {
                    for (size_t ff = 0; ff < nfeh; ++ff)
                    {
                        x[j, i, ff] = times[i, j, ff];
                        for (size_t k = 0; k < nQty; ++k)
                        {
                            f[j, i, ff, k] = fieldData[i, j, ff, k];
                        }
                    }
                }
            }

            return { std::move(xData), std::move(fData) };
        }
    } // namespace

    Tracks3D::Tracks3D(
        const std::string& registryName,
        const std::string& trackName,
        const double fehMin,
        const double fehMax,
        const double vvcrit,
        const double afe) :
        AFe_(afe),
        vVcrit_(vvcrit)
    {
        using Array1D = interp::Mesh3DInterpolator<nQty>::Array1D;
        using Array3D = interp::Mesh3DInterpolator<nQty>::Array3D;
        using Array4D = interp::Mesh3DInterpolator<nQty>::Array4D;

        // Step 1: find the set of tracks matching the input criteria.
        // Building a valid 3D mesh in the feh direction requires enough
        // points to support GSL's default interpolation type (steffen),
        // so the bracketing feh range found by findMatchingTracks is
        // expanded by half that type's minimum point count on each
        // side. gsl_interp_type_min_size is a plain C function from
        // GSL, so this can't be made constexpr.
        const auto nExpand = static_cast<unsigned int>(
            gsl_interp_type_min_size(gsl_interp_steffen) / 2);
        auto [fehVals, groupNames] = findMatchingTracks(
            registryName, trackName, fehMin, fehMax, vvcrit, afe, nExpand);
        // fehVals can't be moved into the member initializer list for
        // FeH_ because it's produced by the same findMatchingTracks call
        // that also yields groupNames, which is needed throughout the
        // rest of this constructor
        FeH_ = std::move(fehVals); //NOLINT(cppcoreguidelines-prefer-member-initializer)
        const size_t nfeh = FeH_.size();
        if (nfeh == 0)
        {
            throw std::runtime_error(
                "Tracks3D: no tracks found matching the input criteria");
        }

        // Re-derive the path to the HDF5 file holding these tracks, the
        // same way findMatchingTracks does internally
        auto [registry, registryPath] = parseRegistry(registryName);
        const auto h5name =
            registry.at_path(trackName).at_path("file").value_or(std::string{});
        const auto h5path = registryPath.parent_path() / h5name;

        // Step 2: scan every matching group to find the number of
        // masses (which must be the same in every group, since
        // Mesh3DInterpolator requires a single, shared mass grid) and
        // the maximum number of time points across all of them
        size_t nmass = 0;
        size_t nt = 0;
        for (size_t f = 0; f < nfeh; ++f)
        {
            const auto [nm, ntime] = getTrackSize(h5path, groupNames[f]);
            if (f == 0) { nmass = nm; }
            else if (nm != nmass)
            {
                throw std::runtime_error(
                    "Tracks3D: group " + groupNames[f] + " has " +
                    std::to_string(nm) + " masses, but group " +
                    groupNames[0] + " has " + std::to_string(nmass) +
                    "; all matching groups must share the same mass grid");
            }
            nt = std::max(nt, ntime);
        }

        // Suppress clang-tidy warnings iun this namespace caused by just
        // including hdf5.h, instead of the individual HDF5 headers,
        // since this is the paradigm that HDF5 wants
        // NOLINTBEGIN(misc-include-cleaner)

        // Open the HDF5 file to read the actual track data
        const hid_t file = H5Fopen(h5path.string().c_str(),
            H5F_ACC_RDONLY, H5P_DEFAULT);
        if (file < 0)
        {
            throw std::runtime_error(
                "Tracks3D: unable to open HDF5 file " + h5path.string());
        }

        // Read the masses and field_names from the first (lowest-feh)
        // matching group; these are assumed to be the same in every
        // group, since Mesh3DInterpolator requires a single, shared
        // mass grid and field layout
        const hid_t refGrp = H5Gopen2(file, groupNames[0].c_str(), H5P_DEFAULT);
        if (refGrp < 0)
        {
            H5Fclose(file);
            throw std::runtime_error(
                "Tracks3D: unable to open group " + groupNames[0]);
        }
        std::vector<double> massData = readDataset1D(refGrp, "masses");
        std::ranges::sort(massData.begin(), massData.end());
        const Array1D masses(massData.data(), nmass);

        const auto fieldNames = readFieldNames(refGrp);
        const auto ageIt = std::ranges::find(fieldNames.begin(), fieldNames.end(), "age");
        if (ageIt == fieldNames.end())
        {
            H5Gclose(refGrp);
            H5Fclose(file);
            throw std::runtime_error(
                "Tracks3D: group " + groupNames[0] + " has no 'age' field");
        }
        const auto ageIdx = std::distance(fieldNames.begin(), ageIt);
        std::vector<size_t> qtyIdx;
        for (const auto& fName : fieldNames)
        {
            const auto* itf = std::ranges::find(fieldStr.begin(), fieldStr.end(), fName);
            if (itf == fieldStr.end()) { continue; }
            qtyIdx.push_back(std::distance(fieldStr.begin(), itf));
        }
        if (qtyIdx.size() != fieldStr.size())
        {
            H5Gclose(refGrp);
            H5Fclose(file);
            throw std::runtime_error(
                "Tracks3D: group " + groupNames[0] +
                " does not have enough fields to fill track data");
        }
        H5Gclose(refGrp);

        // Step 3: read and pad the times and track data of each
        // (mass, time, feh) triple
        auto [timesData, fieldDataVec] = readAllTrackData(
            file, groupNames, massData, fieldNames, ageIdx, qtyIdx, nt);
        const Array3D times(timesData.data(), nmass, nt, nfeh);
        const Array4D fieldData(fieldDataVec.data(), nmass, nt, nfeh, nQty);

        H5Fclose(file);

        // NOLINTEND(misc-include-cleaner)

        // Step 4: transpose times and fieldData into the layout
        // Mesh3DInterpolator requires
        auto [xData, fData] = transposeTrackData(times, fieldData, nmass, nt, nfeh);
        const Array3D x(xData.data(), nt, nmass, nfeh);
        const Array4D f(fData.data(), nt, nmass, nfeh, nQty);

        // The feh values, sorted ascending by findMatchingTracks, form
        // the z coordinate of the mesh
        const Array1D fehCoord(FeH_.data(), nfeh);

        // Build the interpolator
        interp_ = std::make_unique<interp::Mesh3DInterpolator<nQty>>(
            x, masses, fehCoord, f);
    }

} // namespace tracks

// NOLINTEND(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
