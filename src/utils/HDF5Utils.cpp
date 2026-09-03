/**
 * @file HDF5Utils.cpp
 * @author Mark Krumholz
 * @brief Implementation of shared utility functions for reading and writing HDF5 files
 * @date 2026-09-03
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#include "HDF5Utils.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// Suppress clang-tidy warnings caused by just including hdf5.h, instead of
// the individual HDF5 headers, since this is the paradigm that HDF5 wants
// NOLINTBEGIN(misc-include-cleaner)

auto utils::readScalarAttrIfPresent(const hid_t obj,
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

auto utils::readRequiredScalarAttr(const hid_t obj, const std::string& name,
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

auto utils::readDataset1D(const hid_t grp, const std::string& name,
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

auto utils::datasetRank(const hid_t grp, const std::string& name,
    const std::string& context) -> int
{
    const hid_t dset = H5Dopen2(grp, name.c_str(), H5P_DEFAULT);
    if (dset < 0)
    {
        throw std::runtime_error(context + ": unable to open dataset " + name);
    }
    const hid_t space = H5Dget_space(dset);
    const int rank = H5Sget_simple_extent_ndims(space);
    H5Sclose(space);
    H5Dclose(dset);
    return rank;
}

auto utils::dataset2DShape(const hid_t grp, const std::string& name,
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

auto utils::readDataset2D(const hid_t grp, const std::string& name,
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

auto utils::readStringDataset1D(const hid_t grp, const std::string& name,
    const std::string& context) -> std::vector<std::string>
{
    const hid_t dset = H5Dopen2(grp, name.c_str(), H5P_DEFAULT);
    if (dset < 0)
    {
        throw std::runtime_error(context + ": unable to open dataset " + name);
    }
    const hid_t space = H5Dget_space(dset);
    hsize_t dims = 0;
    H5Sget_simple_extent_dims(space, &dims, nullptr);
    const hid_t dtype = H5Dget_type(dset);
    const size_t strSize = H5Tget_size(dtype);
    std::vector<char> buf(dims * strSize);
    H5Dread(dset, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data());
    H5Tclose(dtype);
    H5Sclose(space);
    H5Dclose(dset);

    std::vector<std::string> result;
    result.reserve(dims);
    for (hsize_t i = 0; i < dims; ++i)
    {
        const char* start = buf.data() + (i * strSize); // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        const std::string_view bounded(start, strSize);
        result.emplace_back(bounded.substr(0, bounded.find('\0')));
    }
    return result;
}

auto utils::readDataset3D(const hid_t grp, const std::string& name,
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

auto utils::readFieldNames(const hid_t grp, const std::string& context)
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

auto utils::listGroupDatasetNames(const hid_t grp) -> std::vector<std::string>
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

auto utils::vlenStrType() -> hid_t
{
    const hid_t strType = H5Tcopy(H5T_C_S1);
    H5Tset_size(strType, H5T_VARIABLE);
    H5Tset_cset(strType, H5T_CSET_UTF8);
    return strType;
}

auto utils::fixedStrType(const size_t size) -> hid_t
{
    const hid_t strType = H5Tcopy(H5T_C_S1);
    H5Tset_size(strType, size);
    return strType;
}

void utils::writeStringAttr(const hid_t loc, const std::string& name,
    const std::string& value)
{
    const hid_t strType = utils::vlenStrType();
    const hid_t space = H5Screate(H5S_SCALAR);
    const hid_t attr = H5Acreate2(loc, name.c_str(), strType, space,
        H5P_DEFAULT, H5P_DEFAULT);
    if (attr < 0)
    {
        H5Sclose(space);
        H5Tclose(strType);
        throw std::runtime_error(
            "utils::writeStringAttr: unable to create attribute " + name);
    }
    const char* cstr = value.c_str();
    H5Awrite(attr, strType, static_cast<const void*>(&cstr)); // NOLINT(bugprone-multi-level-implicit-pointer-conversion)
    H5Aclose(attr);
    H5Sclose(space);
    H5Tclose(strType);
}

void utils::writeULongAttr(const hid_t loc, const std::string& name,
    const unsigned long value)
{
    const hid_t space = H5Screate(H5S_SCALAR);
    const hid_t attr = H5Acreate2(loc, name.c_str(), H5T_NATIVE_ULONG, space,
        H5P_DEFAULT, H5P_DEFAULT);
    if (attr < 0)
    {
        H5Sclose(space);
        throw std::runtime_error(
            "utils::writeULongAttr: unable to create attribute " + name);
    }
    H5Awrite(attr, H5T_NATIVE_ULONG, static_cast<const void*>(&value));
    H5Aclose(attr);
    H5Sclose(space);
}

auto utils::readULongAttr(const hid_t loc, const std::string& name) -> unsigned long
{
    const hid_t attr = H5Aopen(loc, name.c_str(), H5P_DEFAULT);
    if (attr < 0)
    {
        throw std::runtime_error(
            "utils::readULongAttr: unable to open attribute " + name);
    }
    unsigned long value = 0;
    H5Aread(attr, H5T_NATIVE_ULONG, static_cast<void*>(&value));
    H5Aclose(attr);
    return value;
}

void utils::writeStringDataset(const hid_t loc, const std::string& name,
    const std::string& value)
{
    const hid_t strType = utils::vlenStrType();
    const hid_t space = H5Screate(H5S_SCALAR);
    const hid_t dset = H5Dcreate2(loc, name.c_str(), strType, space,
        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dset < 0)
    {
        H5Sclose(space);
        H5Tclose(strType);
        throw std::runtime_error(
            "utils::writeStringDataset: unable to create dataset " + name);
    }
    const char* cstr = value.c_str();
    H5Dwrite(dset, strType, H5S_ALL, H5S_ALL, H5P_DEFAULT, static_cast<const void*>(&cstr)); // NOLINT(bugprone-multi-level-implicit-pointer-conversion)
    H5Dclose(dset);
    H5Sclose(space);
    H5Tclose(strType);
}

void utils::writeStringArrayAttr(const hid_t loc, const std::string& name,
    const std::vector<std::string>& values)
{
    const hid_t strType = utils::vlenStrType();
    const auto n = static_cast<hsize_t>(values.size());
    const hid_t space = H5Screate_simple(1, &n, nullptr);
    const hid_t attr = H5Acreate2(loc, name.c_str(), strType, space,
        H5P_DEFAULT, H5P_DEFAULT);
    if (attr < 0)
    {
        H5Sclose(space);
        H5Tclose(strType);
        throw std::runtime_error(
            "utils::writeStringArrayAttr: unable to create attribute " + name);
    }
    std::vector<const char*> cstrs;
    cstrs.reserve(values.size());
    for (const auto& value : values) { cstrs.push_back(value.c_str()); }
    H5Awrite(attr, strType, static_cast<const void*>(cstrs.data())); // NOLINT(bugprone-multi-level-implicit-pointer-conversion)
    H5Aclose(attr);
    H5Sclose(space);
    H5Tclose(strType);
}

namespace
{
    // Chunk size (in elements) used for every extensible dataset
    // created below -- an implementation detail private to this file,
    // not exposed via HDF5Utils.hpp, since no caller needs to know or
    // vary it
    constexpr hsize_t extensibleChunkSize = 256;
} // namespace

auto utils::createExtensible1dDataset(const hid_t loc, const std::string& name,
    const hid_t type) -> hid_t
{
    constexpr hsize_t initDims = 0;
    constexpr hsize_t maxDims = H5S_UNLIMITED;
    const hid_t space = H5Screate_simple(1, &initDims, &maxDims);

    const hid_t propList = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(propList, 1, &extensibleChunkSize);

    const hid_t dset = H5Dcreate2(loc, name.c_str(), type, space,
        H5P_DEFAULT, propList, H5P_DEFAULT);
    H5Pclose(propList);
    H5Sclose(space);
    if (dset < 0)
    {
        throw std::runtime_error(
            "utils::createExtensible1dDataset: unable to create dataset " + name);
    }
    return dset;
}

auto utils::createExtensible2dDataset(const hid_t loc, const std::string& name,
    const hid_t type, const hsize_t nCols) -> hid_t
{
    const std::array<hsize_t, 2> initDims = { 0, nCols };
    const std::array<hsize_t, 2> maxDims = { H5S_UNLIMITED, nCols };
    const hid_t space = H5Screate_simple(2, initDims.data(), maxDims.data());

    const hid_t propList = H5Pcreate(H5P_DATASET_CREATE);
    const std::array<hsize_t, 2> chunkDims = { extensibleChunkSize, nCols };
    H5Pset_chunk(propList, 2, chunkDims.data());

    const hid_t dset = H5Dcreate2(loc, name.c_str(), type, space,
        H5P_DEFAULT, propList, H5P_DEFAULT);
    H5Pclose(propList);
    H5Sclose(space);
    if (dset < 0)
    {
        throw std::runtime_error(
            "utils::createExtensible2dDataset: unable to create dataset " + name);
    }
    return dset;
}

void utils::writeFixed1dDataset(const hid_t loc, const std::string& name,
    const hid_t type, const void* data, const hsize_t len, const std::string& units)
{
    const hid_t space = H5Screate_simple(1, &len, nullptr);
    const hid_t dset = H5Dcreate2(loc, name.c_str(), type, space,
        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dset < 0)
    {
        H5Sclose(space);
        throw std::runtime_error(
            "utils::writeFixed1dDataset: unable to create dataset " + name);
    }
    H5Dwrite(dset, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
    utils::writeStringAttr(dset, "units", units);
    H5Dclose(dset);
    H5Sclose(space);
}

void utils::appendToDataset(const hid_t loc, const std::string& name,
    const hid_t memType, const void* value)
{
    const hid_t dset = H5Dopen2(loc, name.c_str(), H5P_DEFAULT);
    if (dset < 0)
    {
        throw std::runtime_error(
            "utils::appendToDataset: unable to open dataset " + name);
    }

    hid_t fileSpace = H5Dget_space(dset);
    hsize_t curLen = 0;
    hsize_t maxLen = 0;
    H5Sget_simple_extent_dims(fileSpace, &curLen, &maxLen);
    H5Sclose(fileSpace);

    const hsize_t newLen = curLen + 1;
    H5Dset_extent(dset, &newLen);

    fileSpace = H5Dget_space(dset);
    constexpr hsize_t count = 1;
    H5Sselect_hyperslab(fileSpace, H5S_SELECT_SET, &curLen, nullptr, &count, nullptr);

    const hid_t memSpace = H5Screate_simple(1, &count, nullptr);
    H5Dwrite(dset, memType, memSpace, fileSpace, H5P_DEFAULT, value);

    H5Sclose(memSpace);
    H5Sclose(fileSpace);
    H5Dclose(dset);
}

void utils::appendRowToDataset2d(const hid_t loc, const std::string& name,
    const hid_t memType, const void* rowData)
{
    const hid_t dset = H5Dopen2(loc, name.c_str(), H5P_DEFAULT);
    if (dset < 0)
    {
        throw std::runtime_error(
            "utils::appendRowToDataset2d: unable to open dataset " + name);
    }

    hid_t fileSpace = H5Dget_space(dset);
    std::array<hsize_t, 2> curDims{};
    std::array<hsize_t, 2> maxDims{};
    H5Sget_simple_extent_dims(fileSpace, curDims.data(), maxDims.data());
    H5Sclose(fileSpace);

    const std::array<hsize_t, 2> newDims = { curDims.at(0) + 1, curDims.at(1) };
    H5Dset_extent(dset, newDims.data());

    fileSpace = H5Dget_space(dset);
    const std::array<hsize_t, 2> start = { curDims.at(0), 0 };
    const std::array<hsize_t, 2> count = { 1, curDims.at(1) };
    H5Sselect_hyperslab(fileSpace, H5S_SELECT_SET, start.data(), nullptr, count.data(), nullptr);

    const hid_t memSpace = H5Screate_simple(2, count.data(), nullptr);
    H5Dwrite(dset, memType, memSpace, fileSpace, H5P_DEFAULT, rowData);

    H5Sclose(memSpace);
    H5Sclose(fileSpace);
    H5Dclose(dset);
}

auto utils::childNameByIdx(const hid_t loc, const hsize_t idx) -> std::string
{
    const auto len = H5Lget_name_by_idx(loc, ".", H5_INDEX_NAME, H5_ITER_NATIVE,
        idx, nullptr, 0, H5P_DEFAULT);
    std::string name(static_cast<std::size_t>(len), '\0');
    H5Lget_name_by_idx(loc, ".", H5_INDEX_NAME, H5_ITER_NATIVE,
        idx, name.data(), static_cast<std::size_t>(len) + 1, H5P_DEFAULT);
    return name;
}

auto utils::isExtensibleDataset(const hid_t dset) -> bool
{
    const hid_t space = H5Dget_space(dset);
    const int rank = H5Sget_simple_extent_ndims(space);
    std::vector<hsize_t> maxDims(static_cast<std::size_t>(rank));
    H5Sget_simple_extent_dims(space, nullptr, maxDims.data());
    H5Sclose(space);
    return std::ranges::any_of(maxDims,
        [](const hsize_t d) { return d == H5S_UNLIMITED; });
}

void utils::appendDatasetContents(const hid_t srcDset, const hid_t dstDset)
{
    const hid_t srcSpace = H5Dget_space(srcDset);
    const int rank = H5Sget_simple_extent_ndims(srcSpace);
    std::vector<hsize_t> srcDims(static_cast<std::size_t>(rank));
    H5Sget_simple_extent_dims(srcSpace, srcDims.data(), nullptr);
    H5Sclose(srcSpace);
    const hsize_t nRows = srcDims.front();
    if (nRows == 0) { return; }

    // Every extensible dataset this is ever called on uses a fixed-size
    // element type (H5T_NATIVE_ULONG, H5T_NATIVE_DOUBLE, or a fixed-
    // width string type) as its on-disk type directly, so reading/
    // writing through that same type as both the file and memory type
    // (rather than converting to/from some other native type)
    // round-trips the raw bytes exactly, with no variable-length data
    // to separately manage
    const hid_t type = H5Dget_type(srcDset);
    const std::size_t elemSize = H5Tget_size(type);
    auto nElem = static_cast<std::size_t>(nRows);
    for (int i = 1; i < rank; ++i) { nElem *= static_cast<std::size_t>(srcDims.at(static_cast<std::size_t>(i))); }
    std::vector<std::byte> buf(nElem * elemSize);
    H5Dread(srcDset, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data());

    const hid_t dstSpaceBefore = H5Dget_space(dstDset);
    std::vector<hsize_t> dstDims(static_cast<std::size_t>(rank));
    H5Sget_simple_extent_dims(dstSpaceBefore, dstDims.data(), nullptr);
    H5Sclose(dstSpaceBefore);
    const hsize_t oldRows = dstDims.front();

    std::vector<hsize_t> newDims = dstDims;
    newDims.front() = oldRows + nRows;
    H5Dset_extent(dstDset, newDims.data());

    const hid_t dstSpace = H5Dget_space(dstDset);
    std::vector<hsize_t> start(static_cast<std::size_t>(rank), 0);
    start.front() = oldRows;
    std::vector<hsize_t> count = srcDims;
    H5Sselect_hyperslab(dstSpace, H5S_SELECT_SET, start.data(), nullptr, count.data(), nullptr);

    const hid_t memSpace = H5Screate_simple(rank, count.data(), nullptr);
    H5Dwrite(dstDset, type, memSpace, dstSpace, H5P_DEFAULT, buf.data());

    H5Sclose(memSpace);
    H5Sclose(dstSpace);
    H5Tclose(type);
}

void utils::appendFileContents(const hid_t srcFile, const hid_t dstFile)
{
    H5G_info_t rootInfo{};
    H5Gget_info(srcFile, &rootInfo);
    for (hsize_t i = 0; i < rootInfo.nlinks; ++i)
    {
        const std::string groupName = utils::childNameByIdx(srcFile, i);
        const hid_t srcChild = H5Oopen(srcFile, groupName.c_str(), H5P_DEFAULT);
        if (H5Iget_type(srcChild) != H5I_GROUP) { H5Oclose(srcChild); continue; }

        const hid_t dstGroup = H5Gopen2(dstFile, groupName.c_str(), H5P_DEFAULT);

        H5G_info_t groupInfo{};
        H5Gget_info(srcChild, &groupInfo);
        for (hsize_t j = 0; j < groupInfo.nlinks; ++j)
        {
            const std::string dsetName = utils::childNameByIdx(srcChild, j);
            const hid_t srcObj = H5Oopen(srcChild, dsetName.c_str(), H5P_DEFAULT);
            if (H5Iget_type(srcObj) == H5I_DATASET && utils::isExtensibleDataset(srcObj))
            {
                const hid_t dstDset = H5Dopen2(dstGroup, dsetName.c_str(), H5P_DEFAULT);
                utils::appendDatasetContents(srcObj, dstDset);
                H5Dclose(dstDset);
            }
            H5Oclose(srcObj);
        }

        H5Gclose(dstGroup);
        H5Oclose(srcChild);
    }
}

// NOLINTEND(misc-include-cleaner)
