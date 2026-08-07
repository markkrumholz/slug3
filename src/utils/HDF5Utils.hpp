/**
 * @file HDF5Utils.hpp
 * @author Mark Krumholz
 * @brief Shared utility functions for reading data out of HDF5 files
 * @date 2026-08-01
 */

#ifndef HDF5UTILS_HPP
#define HDF5UTILS_HPP

// Suppress clang-tidy warnings caused by just including hdf5.h, instead of
// the individual HDF5 headers, since this is the paradigm that HDF5 wants
// NOLINTBEGIN(misc-include-cleaner)
#include "hdf5.h"
#include <array>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace utils
{
    /**
     * @brief Read a scalar double attribute from an HDF5 object, if present
     * @param obj Handle to the object (a group or a dataset)
     * @param name Name of the attribute
     * @returns The attribute's value, or an empty optional if the object
     *   has no attribute of that name
     */
    inline auto readScalarAttrIfPresent(const hid_t obj,
        const std::string& name) -> std::optional<double>
    {
        if (H5Aexists(obj, name.c_str()) <= 0) { return std::nullopt; }
        const hid_t attr = H5Aopen(obj, name.c_str(), H5P_DEFAULT);
        if (attr < 0) { return std::nullopt; }
        double value = 0.0;
        H5Aread(attr, H5T_NATIVE_DOUBLE, &value);
        H5Aclose(attr);
        return value;
    }

    /**
     * @brief Read a required scalar double attribute from an HDF5 object
     * @param obj Handle to the object (a group or a dataset)
     * @param name Name of the attribute
     * @param context Prefix used in thrown error messages (typically the
     *   name of the calling class or function)
     * @returns The attribute's value
     * @throws std::runtime_error if obj has no attribute of that name
     */
    inline auto readRequiredScalarAttr(const hid_t obj, const std::string& name,
        const std::string& context) -> double
    {
        if (H5Aexists(obj, name.c_str()) <= 0)
        {
            throw std::runtime_error(
                context + ": missing required attribute " + name);
        }
        const hid_t attr = H5Aopen(obj, name.c_str(), H5P_DEFAULT);
        if (attr < 0)
        {
            throw std::runtime_error(
                context + ": unable to open attribute " + name);
        }
        double value = 0.0;
        H5Aread(attr, H5T_NATIVE_DOUBLE, &value);
        H5Aclose(attr);
        return value;
    }

    /**
     * @brief Read a 1D double dataset from an HDF5 group (or file)
     * @param grp Handle to the group (or already-open file) containing the dataset
     * @param name Name of the dataset
     * @param context Prefix used in thrown error messages (typically the
     *   name of the calling class or function)
     * @returns The dataset contents
     */
    inline auto readDataset1D(const hid_t grp, const std::string& name,
        const std::string& context) -> std::vector<double>
    {
        const hid_t dset = H5Dopen2(grp, name.c_str(), H5P_DEFAULT);
        if (dset < 0)
        {
            throw std::runtime_error(context + ": unable to open dataset " + name);
        }
        const hid_t space = H5Dget_space(dset);
        hsize_t dims = 0;
        H5Sget_simple_extent_dims(space, &dims, nullptr);
        std::vector<double> data(dims);
        H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
        H5Sclose(space);
        H5Dclose(dset);
        return data;
    }

    /**
     * @brief Get the shape of a 2D dataset without reading its data
     * @param grp Handle to the group containing the dataset
     * @param name Name of the dataset
     * @param context Prefix used in thrown error messages (typically the
     *   name of the calling class or function)
     * @returns The (nrow, ncol) shape of the dataset
     */
    inline auto dataset2DShape(const hid_t grp, const std::string& name,
        const std::string& context) -> std::pair<size_t, size_t>
    {
        const hid_t dset = H5Dopen2(grp, name.c_str(), H5P_DEFAULT);
        if (dset < 0)
        {
            throw std::runtime_error(context + ": unable to open dataset " + name);
        }
        const hid_t space = H5Dget_space(dset);
        std::array<hsize_t, 2> dims = {0, 0};
        H5Sget_simple_extent_dims(space, dims.data(), nullptr);
        H5Sclose(space);
        H5Dclose(dset);
        return { dims[0], dims[1] }; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
    }

    /**
     * @brief Read a full 2D double dataset from an HDF5 group
     * @param grp Handle to the group containing the dataset
     * @param name Name of the dataset
     * @param context Prefix used in thrown error messages (typically the
     *   name of the calling class or function)
     * @returns The dataset contents, and its (nrow, ncol) shape
     */
    inline auto readDataset2D(const hid_t grp, const std::string& name,
        const std::string& context)
        -> std::pair<std::vector<double>, std::pair<size_t, size_t>>
    {
        const hid_t dset = H5Dopen2(grp, name.c_str(), H5P_DEFAULT);
        if (dset < 0)
        {
            throw std::runtime_error(context + ": unable to open dataset " + name);
        }
        const hid_t space = H5Dget_space(dset);
        std::array<hsize_t, 2> dims = {0, 0};
        H5Sget_simple_extent_dims(space, dims.data(), nullptr);
        std::vector<double> data(dims[0] * dims[1]); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
        H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
        H5Sclose(space);
        H5Dclose(dset);
        return { std::move(data), { dims[0], dims[1] } }; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
    }

    /**
     * @brief Read a full 3D double dataset from an HDF5 group
     * @param grp Handle to the group (or already-open file) containing the dataset
     * @param name Name of the dataset
     * @param context Prefix used in thrown error messages (typically the
     *   name of the calling class or function)
     * @returns The dataset contents, flattened in the same row-major
     *   (C) order HDF5/h5py itself stores a multidimensional dataset
     *   in -- i.e. element (i, j, k) of a dataset shaped (n0, n1, n2)
     *   is at flat index (i * n1 + j) * n2 + k -- and its (n0, n1, n2) shape
     */
    inline auto readDataset3D(const hid_t grp, const std::string& name,
        const std::string& context)
        -> std::pair<std::vector<double>, std::array<hsize_t, 3>>
    {
        const hid_t dset = H5Dopen2(grp, name.c_str(), H5P_DEFAULT);
        if (dset < 0)
        {
            throw std::runtime_error(context + ": unable to open dataset " + name);
        }
        const hid_t space = H5Dget_space(dset);
        std::array<hsize_t, 3> dims = {0, 0, 0};
        H5Sget_simple_extent_dims(space, dims.data(), nullptr);
        std::vector<double> data(dims[0] * dims[1] * dims[2]); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
        H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
        H5Sclose(space);
        H5Dclose(dset);
        return { std::move(data), dims };
    }

    /**
     * @brief Read the field_names attribute of an HDF5 group
     * @param grp Handle to the group
     * @param context Prefix used in thrown error messages (typically the
     *   name of the calling class or function)
     * @returns The names of the fields stored in each track dataset,
     *   in the order in which they appear
     */
    inline auto readFieldNames(const hid_t grp, const std::string& context)
        -> std::vector<std::string>
    {
        const hid_t attr = H5Aopen(grp, "field_names", H5P_DEFAULT);
        if (attr < 0)
        {
            throw std::runtime_error(
                context + ": unable to open field_names attribute");
        }
        const hid_t aspace = H5Aget_space(attr);
        const auto npoints =
            static_cast<size_t>(H5Sget_simple_extent_npoints(aspace));

        // Use the attribute's own (variable-length, UTF-8) type as the
        // memory type; building a fresh H5T_C_S1-based type instead fails
        // to convert because the on-disk character set is UTF-8, not ASCII
        const hid_t memtype = H5Aget_type(attr);

        std::vector<char*> buf(npoints);
        H5Aread(attr, memtype, static_cast<void*>(buf.data())); // NOLINT(bugprone-multi-level-implicit-pointer-conversion)
        std::vector<std::string> names;
        names.reserve(npoints);
        for (const auto* s : buf) { names.emplace_back(s); }

        // Use H5Dvlen_reclaim rather than its replacement, H5Treclaim,
        // since the latter was only added in HDF5 1.14 and isn't
        // available in the older HDF5 that Ubuntu's libhdf5-dev package
        // ships
        H5Dvlen_reclaim(memtype, aspace, H5P_DEFAULT,
            static_cast<void*>(buf.data())); // NOLINT(bugprone-multi-level-implicit-pointer-conversion)
        H5Tclose(memtype);
        H5Sclose(aspace);
        H5Aclose(attr);

        return names;
    }

    /**
     * @brief List the names of every dataset directly inside an HDF5 group
     * @param grp Handle to the group
     * @returns The names of the group's child datasets
     */
    inline auto listGroupDatasetNames(const hid_t grp) -> std::vector<std::string>
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

} // namespace utils
// NOLINTEND(misc-include-cleaner)

#endif // HDF5UTILS_HPP
