/**
 * @file Tracks2D.cpp
 * @author Mark Krumholz
 * @brief Implementation of Tracks2D.hpp
 * @date 2024-07-09
 */

#include "Tracks2D.hpp"
#include "../interpolation/Mesh2DInterpolator.hpp"
#include "hdf5.h"  // NOLINT(misc-include-cleaner)
#include <algorithm>
#include <array>
#include <cstddef>
#include <format>
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
                    "Tracks2D: unable to open dataset " + name);
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
         * @brief Get the shape of a 2D dataset without reading its data
         * @param grp Handle to the group containing the dataset
         * @param name Name of the dataset
         * @returns The (nrow, ncol) shape of the dataset
         */
        auto dataset2DShape(const hid_t grp, const std::string& name) //NOLINT(llvm-prefer-static-over-anonymous-namespace)
            -> std::pair<size_t, size_t>
        {
            const hid_t dset = H5Dopen2(grp, name.c_str(), H5P_DEFAULT);
            if (dset < 0)
            {
                throw std::runtime_error(
                    "Tracks2D: unable to open dataset " + name);
            }
            const hid_t space = H5Dget_space(dset);
            std::array<hsize_t,2> dims = {0, 0};
            H5Sget_simple_extent_dims(space, static_cast<hsize_t *>(dims.data()), nullptr);
            H5Sclose(space);
            H5Dclose(dset);
            return { dims[0], dims[1] }; 
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
                    "Tracks2D: unable to open dataset " + name);
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
                    "Tracks2D: unable to open field_names attribute");
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
    } // namespace
    // NOLINTEND(misc-include-cleaner)

    Tracks2D::Tracks2D(const hid_t grp,
        const size_t ntMin)
    {
        using Array1D = interp::Mesh2DInterpolator<nQty>::Array1D;
        using Array2D = interp::Mesh2DInterpolator<nQty>::Array2D;
        using Array3D = interp::Mesh2DInterpolator<nQty>::Array3D;

        // Read the masses of the tracks in this group; the masses
        // dataset is not guaranteed to be stored in ascending order
        // (e.g. some PARSEC files concatenate several mass ranges that
        // were fetched separately), but Mesh2DInterpolator requires
        // its y coordinate -- the masses, here -- to be non-decreasing,
        // and we in any case want to process tracks from lowest mass
        // to highest, so sort them here
        std::vector<double> massData = readDataset1D(grp, "masses");
        std::ranges::sort(massData.begin(), massData.end());
        const size_t nmass = massData.size();
        const Array1D masses(massData.data(), nmass);

        // Read the field_names attribute, and use it to identify which
        // column of each track dataset holds the age, and which columns
        // hold the nQty quantities to be stored as track data; any
        // fields beyond the first nQty non-age fields are ignored
        const auto fieldNames = readFieldNames(grp);
        size_t ageIdx = fieldNames.size();
        for (size_t i = 0; i < fieldNames.size(); ++i)
        {
            if (fieldNames[i] == "age") { ageIdx = i; break; }
        }
        if (ageIdx == fieldNames.size())
        {
            throw std::runtime_error(
                "Tracks2D: group has no 'age' field");
        }
        std::vector<size_t> qtyIdx;
        for (size_t i = 0; i < fieldNames.size(); ++i)
        {
            for (size_t j = 0; j < nQty; ++j)
            {
                if (fieldNames[i] == FieldStr[j])
                {
                    qtyIdx.push_back(j);
                    break;
                }
            }
            if (qtyIdx.size() == nQty) { break; }
        }
        if (qtyIdx.size() != nQty)
        {
            throw std::runtime_error(
                "Tracks2D: did not find all the expected fields in track file");
        }

        // First pass: scan every track dataset to find the number of
        // time points it holds, so we can determine nt, the number of
        // time points in the final, padded set of tracks
        std::vector<size_t> ntime(nmass);
        for (size_t i = 0; i < nmass; ++i)
        {
            const auto name = std::format("track_m{:.3f}", massData[i]);
            const auto [nrow, ncol] = dataset2DShape(grp, name);
            if (ncol != fieldNames.size())
            {
                throw std::runtime_error(
                    "Tracks2D: dataset " + name +
                    " has an unexpected number of fields");
            }
            ntime[i] = nrow;
        }
        const size_t nt = std::max(ntMin,
            *std::ranges::max_element(ntime.begin(), ntime.end()));

        // Allocate storage for the times and track data of each mass,
        // padded to nt time points each
        std::vector<double> timesData(nmass * nt);
        const Array2D times(timesData.data(), nmass, nt);
        std::vector<double> fieldDataVec(nmass * nt * nQty);
        const Array3D fieldData(fieldDataVec.data(), nmass, nt, nQty);

        // Second pass: read each track in turn, from lowest mass to
        // highest, padding the end of any track with fewer than nt
        // time points by repeating its final value
        for (size_t i = 0; i < nmass; ++i)
        {
            const auto name = std::format("track_m{:.3f}", massData[i]);
            auto [data, shape] = readDataset2D(grp, name);
            const auto [nrow, ncol] = shape;
            const Array2D track(data.data(), nrow, ncol);

            for (size_t j = 0; j < nt; ++j)
            {
                const size_t src = std::min(j, nrow - 1);
                times[i, j] = track[src, ageIdx];
                for (size_t k = 0; k < nQty; ++k)
                {
                    fieldData[i, j, k] = track[src, qtyIdx[k]];
                }
            }
        }

        // Mesh2DInterpolator requires its x and f arrays to have shape
        // (nt, nmass) and (nt, nmass, nQty) respectively -- the
        // transpose of times and fieldData -- because masses, not
        // times, form the tensor (shared) direction of the mesh; build
        // those transposed views here before constructing interp_
        std::vector<double> xData(nt * nmass);
        const Array2D x(xData.data(), nt, nmass);
        std::vector<double> fData(nt * nmass * nQty);
        const Array3D f(fData.data(), nt, nmass, nQty);
        for (size_t i = 0; i < nmass; ++i)
        {
            for (size_t j = 0; j < nt; ++j)
            {
                x[j, i] = times[i, j];
                for (size_t k = 0; k < nQty; ++k)
                {
                    f[j, i, k] = fieldData[i, j, k];
                }
            }
        }

        // Build the interpolator
        interp_ = std::make_unique<interp::Mesh2DInterpolator<nQty>>(
            x, masses, f);
    }

} // namespace tracks

// NOLINTEND(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
