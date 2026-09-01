/**
 * @file OutputManagerH5.cpp
 * @author Mark Krumholz
 * @brief Implementation of OutputManagerH5
 * @date 2026-07-17
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#include "OutputManagerH5.hpp"
#include "../core/Cluster.hpp"
#include "../core/Galaxy.hpp"
#include "../phot/FilterCollection.hpp"
#include "../specsyn/Specsyn.hpp"
#include "../utils/RngThread.hpp"
#include "../utils/ThreadVec.hpp"
#include "OutputManager.hpp"
#include "SimControls.hpp"
#include "hdf5.h" // NOLINT(misc-include-cleaner)
#include "io/SlugVersion.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <filesystem>
#include <iomanip> // NOLINT(misc-include-cleaner) -- used for std::setw/std::setfill in the constructor's own thread_NNNN.h5 filename construction
#include <sstream>
#include <stdexcept>
#include <string>
#include <toml.hpp>
#include <vector>
#ifdef _OPENMP
#   include <omp.h>
#endif

// Suppress clang-tidy warnings in this namespace caused by just
// including hdf5.h, instead of the individual HDF5 headers, since
// this is the paradigm that HDF5 wants
// NOLINTBEGIN(misc-include-cleaner)

// Create (and return) a variable-length, UTF-8 HDF5 string datatype
static auto vlenStrType() -> hid_t
{
    const hid_t strType = H5Tcopy(H5T_C_S1);
    H5Tset_size(strType, H5T_VARIABLE);
    H5Tset_cset(strType, H5T_CSET_UTF8);
    return strType;
}

// Create (and return) a fixed-length HDF5 string datatype of the
// given size, in bytes -- used for the "rng" dataset, whose elements
// are utils::RngState's own fixed-width, null-padded buffers rather
// than variable-length text
static auto fixedStrType(const size_t size) -> hid_t
{
    const hid_t strType = H5Tcopy(H5T_C_S1);
    H5Tset_size(strType, size);
    return strType;
}

// Write a scalar string attribute called name, with the given
// value, on the HDF5 object loc
static void writeStringAttr(const hid_t loc, const std::string& name,
    const std::string& value)
{
    const hid_t strType = vlenStrType();
    const hid_t space = H5Screate(H5S_SCALAR);
    const hid_t attr = H5Acreate2(loc, name.c_str(), strType, space,
        H5P_DEFAULT, H5P_DEFAULT);
    if (attr < 0)
    {
        H5Sclose(space);
        H5Tclose(strType);
        throw std::runtime_error(
            "OutputManagerH5: unable to create attribute " + name);
    }
    const char* cstr = value.c_str();
    H5Awrite(attr, strType, static_cast<const void*>(&cstr)); // NOLINT(bugprone-multi-level-implicit-pointer-conversion)
    H5Aclose(attr);
    H5Sclose(space);
    H5Tclose(strType);
}

// Write a scalar unsigned long attribute called name, with the given
// value, on the HDF5 object loc
static void writeULongAttr(const hid_t loc, const std::string& name,
    const unsigned long value)
{
    const hid_t space = H5Screate(H5S_SCALAR);
    const hid_t attr = H5Acreate2(loc, name.c_str(), H5T_NATIVE_ULONG, space,
        H5P_DEFAULT, H5P_DEFAULT);
    if (attr < 0)
    {
        H5Sclose(space);
        throw std::runtime_error(
            "OutputManagerH5: unable to create attribute " + name);
    }
    H5Awrite(attr, H5T_NATIVE_ULONG, static_cast<const void*>(&value));
    H5Aclose(attr);
    H5Sclose(space);
}

// Write a scalar string dataset called name, with the given value,
// into the HDF5 group loc
static void writeStringDataset(const hid_t loc, const std::string& name,
    const std::string& value)
{
    const hid_t strType = vlenStrType();
    const hid_t space = H5Screate(H5S_SCALAR);
    const hid_t dset = H5Dcreate2(loc, name.c_str(), strType, space,
        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dset < 0)
    {
        H5Sclose(space);
        H5Tclose(strType);
        throw std::runtime_error(
            "OutputManagerH5: unable to create dataset " + name);
    }
    const char* cstr = value.c_str();
    H5Dwrite(dset, strType, H5S_ALL, H5S_ALL, H5P_DEFAULT, static_cast<const void*>(&cstr)); // NOLINT(bugprone-multi-level-implicit-pointer-conversion)
    H5Dclose(dset);
    H5Sclose(space);
    H5Tclose(strType);
}

// Write a 1d array-of-strings attribute called name, with the given
// values, on the HDF5 object loc
static void writeStringArrayAttr(const hid_t loc, const std::string& name,
    const std::vector<std::string>& values)
{
    const hid_t strType = vlenStrType();
    const auto n = static_cast<hsize_t>(values.size());
    const hid_t space = H5Screate_simple(1, &n, nullptr);
    const hid_t attr = H5Acreate2(loc, name.c_str(), strType, space,
        H5P_DEFAULT, H5P_DEFAULT);
    if (attr < 0)
    {
        H5Sclose(space);
        H5Tclose(strType);
        throw std::runtime_error(
            "OutputManagerH5: unable to create attribute " + name);
    }
    std::vector<const char*> cstrs;
    cstrs.reserve(values.size());
    for (const auto& value : values) { cstrs.push_back(value.c_str()); }
    H5Awrite(attr, strType, static_cast<const void*>(cstrs.data())); // NOLINT(bugprone-multi-level-implicit-pointer-conversion)
    H5Aclose(attr);
    H5Sclose(space);
    H5Tclose(strType);
}

// Chunk size (in elements) used for the extensible cluster datasets
static constexpr hsize_t clustersChunkSize = 256;

// Create a 1d, extensible dataset called name, of the given HDF5
// datatype, in the HDF5 group loc, chunked with clustersChunkSize
// elements per chunk
static auto createExtensible1dDataset(const hid_t loc, const std::string& name,
    const hid_t type) -> hid_t
{
    constexpr hsize_t initDims = 0;
    constexpr hsize_t maxDims = H5S_UNLIMITED;
    const hid_t space = H5Screate_simple(1, &initDims, &maxDims);

    const hid_t propList = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(propList, 1, &clustersChunkSize);

    const hid_t dset = H5Dcreate2(loc, name.c_str(), type, space,
        H5P_DEFAULT, propList, H5P_DEFAULT);
    H5Pclose(propList);
    H5Sclose(space);
    if (dset < 0)
    {
        throw std::runtime_error(
            "OutputManagerH5: unable to create dataset " + name);
    }
    return dset;
}

// Create a 2d dataset called name, of the given HDF5 datatype, in
// the HDF5 group loc, with nCols columns (fixed) and extensible in
// the number of rows, chunked with clustersChunkSize rows per chunk
static auto createExtensible2dDataset(const hid_t loc, const std::string& name,
    const hid_t type, const hsize_t nCols) -> hid_t
{
    const std::array<hsize_t, 2> initDims = { 0, nCols };
    const std::array<hsize_t, 2> maxDims = { H5S_UNLIMITED, nCols };
    const hid_t space = H5Screate_simple(2, initDims.data(), maxDims.data());

    const hid_t propList = H5Pcreate(H5P_DATASET_CREATE);
    const std::array<hsize_t, 2> chunkDims = { clustersChunkSize, nCols };
    H5Pset_chunk(propList, 2, chunkDims.data());

    const hid_t dset = H5Dcreate2(loc, name.c_str(), type, space,
        H5P_DEFAULT, propList, H5P_DEFAULT);
    H5Pclose(propList);
    H5Sclose(space);
    if (dset < 0)
    {
        throw std::runtime_error(
            "OutputManagerH5: unable to create dataset " + name);
    }
    return dset;
}

// Create a 1d, non-extensible dataset called name, of the given
// HDF5 datatype, in the HDF5 group loc, immediately write data (an
// array of len elements) into it, and tag it with a scalar "units"
// string attribute
static void writeFixed1dDataset(const hid_t loc, const std::string& name,
    const hid_t type, const void* data, const hsize_t len, const std::string& units)
{
    const hid_t space = H5Screate_simple(1, &len, nullptr);
    const hid_t dset = H5Dcreate2(loc, name.c_str(), type, space,
        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dset < 0)
    {
        H5Sclose(space);
        throw std::runtime_error(
            "OutputManagerH5: unable to create dataset " + name);
    }
    H5Dwrite(dset, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
    writeStringAttr(dset, "units", units);
    H5Dclose(dset);
    H5Sclose(space);
}

// Append a single element, of the given HDF5 memory datatype, to
// the end of the extensible 1d dataset called name in the HDF5
// group loc
static void appendToDataset(const hid_t loc, const std::string& name,
    const hid_t memType, const void* value)
{
    const hid_t dset = H5Dopen2(loc, name.c_str(), H5P_DEFAULT);
    if (dset < 0)
    {
        throw std::runtime_error(
            "OutputManagerH5: unable to open dataset " + name);
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

// Append a single row (the dataset's fixed number of columns, taken
// from its own extent) to the end of the extensible 2d dataset
// called name in the HDF5 group loc
static void appendRowToDataset2d(const hid_t loc, const std::string& name,
    const hid_t memType, const void* rowData)
{
    const hid_t dset = H5Dopen2(loc, name.c_str(), H5P_DEFAULT);
    if (dset < 0)
    {
        throw std::runtime_error(
            "OutputManagerH5: unable to open dataset " + name);
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

// Create the "spec_neb" 2D dataset in the given spectra group (sized
// nWl, the same wavelength grid as the group's own "spec" dataset), if
// a nebular emission grid was requested -- a no-op otherwise. Factored
// out of openClusterSpectraGroup()/openGalaxySpectraGroup() to keep
// each within its cognitive-complexity budget.
static void createSpecNebDataset(const io::SimControls& simControls, const hid_t group, const hsize_t nWl)
{
    if (simControls.nebular() == nullptr) { return; }
    const hid_t dset = createExtensible2dDataset(group, "spec_neb", H5T_NATIVE_DOUBLE, nWl);
    writeStringAttr(dset, "units", "erg/(s Angstrom)");
    H5Dclose(dset);
}

// Create the "spec_neb_extinct" 2D dataset in the given spectra group
// (sized nWlExtinct, the same restricted grid as the group's own
// "spec_extinct" dataset), if a nebular emission grid was requested --
// a no-op otherwise. Must only be called when SimControls::extinct()
// is already known to be non-null (specNebExtinct() has no meaning
// without an extinction curve). Factored out of
// openClusterSpectraGroup()/openGalaxySpectraGroup() to keep each
// within its cognitive-complexity budget.
static void createSpecNebExtinctDataset(const io::SimControls& simControls, const hid_t group,
    const hsize_t nWlExtinct)
{
    if (simControls.nebular() == nullptr) { return; }
    const hid_t dset = createExtensible2dDataset(group, "spec_neb_extinct", H5T_NATIVE_DOUBLE, nWlExtinct);
    writeStringAttr(dset, "units", "erg/(s Angstrom)");
    H5Dclose(dset);
}

// Create the "phot_neb" and (if extinction was also requested)
// "phot_neb_extinct" 2D datasets (each sized nRealFilters, excluding
// any appended "Lbol" entry -- see this function's callers' own
// comments) in the given photometry group, if a nebular emission grid
// was requested -- a no-op otherwise. Must only be called when
// SimControls::filters() is already known to be non-null. Factored out
// of openClusterPhotGroup()/openGalaxyPhotGroup() to keep each within
// its cognitive-complexity budget.
static void createPhotNebDatasets(const io::SimControls& simControls, const hid_t group,
    const hsize_t nRealFilters, const std::vector<std::string>& realFilterUnits)
{
    if (simControls.nebular() == nullptr) { return; }
    const hid_t photNebDset = createExtensible2dDataset(group, "phot_neb", H5T_NATIVE_DOUBLE, nRealFilters);
    writeStringArrayAttr(photNebDset, "units", realFilterUnits);
    H5Dclose(photNebDset);

    if (simControls.extinct() == nullptr) { return; }
    const hid_t photNebExtinctDset = createExtensible2dDataset(
        group, "phot_neb_extinct", H5T_NATIVE_DOUBLE, nRealFilters);
    writeStringArrayAttr(photNebExtinctDset, "units", realFilterUnits);
    H5Dclose(photNebExtinctDset);
}

// Append one row to the "phot_neb" and (if extinction was also
// requested) "phot_neb_extinct" datasets in the given photometry
// group, if a nebular emission grid was requested -- a no-op
// otherwise. Must only be called when SimControls::filters() is
// already known to be non-null, and from inside the same critical
// section as the caller's other appendRowToDataset2d() calls.
// Factored out of writeClusterPhot()/writeGalaxyPhot() to keep each
// within its cognitive-complexity budget.
static void appendPhotNebRows(const io::SimControls& simControls, const hid_t group,
    const std::vector<double>& photNeb, const std::vector<double>& photNebExtinct)
{
    if (simControls.nebular() == nullptr || simControls.filters() == nullptr) { return; }
    appendRowToDataset2d(group, "phot_neb", H5T_NATIVE_DOUBLE, photNeb.data());

    if (simControls.extinct() == nullptr) { return; }
    appendRowToDataset2d(group, "phot_neb_extinct", H5T_NATIVE_DOUBLE, photNebExtinct.data());
}

// Create the "line_wl"/"line_label" fixed datasets and the
// "neb_lines" (and, if extinction was also requested,
// "neb_lines_extinct") extensible 2D datasets in the given spectra
// group, if a nebular emission grid was requested -- a no-op
// otherwise. line_label is written as a fixed-length string dataset
// (see utils::readStringDataset1D's own comment), sized to the
// longest of SimControls::nebular()'s own lineLabel() strings plus a
// null terminator -- computed fresh here each time, since Nebular
// itself does not retain whatever fixed length its own source table
// used. Factored out of openClusterSpectraGroup()/
// openGalaxySpectraGroup() to keep each within its cognitive-
// complexity budget.
static void createNebLinesDatasets(const io::SimControls& simControls, const hid_t group)
{
    const auto* neb = simControls.nebular();
    if (neb == nullptr) { return; }

    const auto& lineWl = neb->lineWl();
    const auto& lineLabel = neb->lineLabel();
    const auto nLines = static_cast<hsize_t>(lineWl.size());

    writeFixed1dDataset(group, "line_wl", H5T_NATIVE_DOUBLE, lineWl.data(), nLines, "Angstrom");

    std::size_t maxLabelLen = 0;
    for (const auto& label : lineLabel) { maxLabelLen = std::max(maxLabelLen, label.size()); }
    const std::size_t labelStrSize = maxLabelLen + 1;
    std::vector<char> labelBuf(lineLabel.size() * labelStrSize, '\0');
    for (std::size_t ell = 0; ell < lineLabel.size(); ++ell)
    {
        std::ranges::copy(lineLabel.at(ell), labelBuf.begin() + static_cast<std::ptrdiff_t>(ell * labelStrSize));
    }
    const hid_t labelType = fixedStrType(labelStrSize);
    writeFixed1dDataset(group, "line_label", labelType, labelBuf.data(), nLines, "");
    H5Tclose(labelType);

    const hid_t linesDset = createExtensible2dDataset(group, "neb_lines", H5T_NATIVE_DOUBLE, nLines);
    writeStringAttr(linesDset, "units", "erg/s");
    H5Dclose(linesDset);

    if (simControls.extinct() == nullptr) { return; }
    const hid_t linesExtinctDset = createExtensible2dDataset(group, "neb_lines_extinct", H5T_NATIVE_DOUBLE, nLines);
    writeStringAttr(linesExtinctDset, "units", "erg/s");
    H5Dclose(linesExtinctDset);
}

// Append one row to the "neb_lines" and (if extinction was also
// requested) "neb_lines_extinct" datasets in the given spectra group,
// if a nebular emission grid was requested -- a no-op otherwise. Must
// be called from inside the same critical section as the caller's
// other appendRowToDataset2d() calls. Factored out of
// writeClusterSpec()/writeGalaxySpec() to keep each within its
// cognitive-complexity budget.
static void appendNebLinesRow(const io::SimControls& simControls, const hid_t group,
    const std::vector<double>& lineLum, const std::vector<double>& lineLumExtinct)
{
    if (simControls.nebular() == nullptr) { return; }
    appendRowToDataset2d(group, "neb_lines", H5T_NATIVE_DOUBLE, lineLum.data());

    if (simControls.extinct() == nullptr) { return; }
    appendRowToDataset2d(group, "neb_lines_extinct", H5T_NATIVE_DOUBLE, lineLumExtinct.data());
}

// Return the name of the idx-th direct child link of the HDF5 group
// (or file, which is itself a group) loc
static auto childNameByIdx(const hid_t loc, const hsize_t idx) -> std::string
{
    const auto len = H5Lget_name_by_idx(loc, ".", H5_INDEX_NAME, H5_ITER_NATIVE,
        idx, nullptr, 0, H5P_DEFAULT);
    std::string name(static_cast<std::size_t>(len), '\0');
    H5Lget_name_by_idx(loc, ".", H5_INDEX_NAME, H5_ITER_NATIVE,
        idx, name.data(), static_cast<std::size_t>(len) + 1, H5P_DEFAULT);
    return name;
}

// A dataset is "extensible" if any one of its dimensions has an
// unlimited maximum extent -- true of every dataset this class itself
// creates via createExtensible1dDataset()/createExtensible2dDataset(),
// false of every one it creates via writeFixed1dDataset() (e.g. "wl")
// or writeStringDataset() (e.g. input_deck's own "toml"), so this is
// exactly the distinction consolidateFiles() needs to know which
// datasets hold one row per write (and so need merging across thread
// files) versus fixed, shared metadata (already correctly present,
// identically, in every thread file, so nothing to do)
static auto isExtensibleDataset(const hid_t dset) -> bool
{
    const hid_t space = H5Dget_space(dset);
    const int rank = H5Sget_simple_extent_ndims(space);
    std::vector<hsize_t> maxDims(static_cast<std::size_t>(rank));
    H5Sget_simple_extent_dims(space, nullptr, maxDims.data());
    H5Sclose(space);
    return std::ranges::any_of(maxDims,
        [](const hsize_t d) { return d == H5S_UNLIMITED; });
}

// Read every row of the extensible dataset srcDset in full, and
// append them onto the end of dstDset -- srcDset and dstDset are
// assumed to share the same rank, element type, and (for a 2d
// dataset) number of columns, which every thread_NNNN.h5 file
// guarantees, since each was built by an identical openOutputFile()
// call. A no-op if srcDset is currently empty (no rows to append).
static void appendDatasetContents(const hid_t srcDset, const hid_t dstDset)
{
    const hid_t srcSpace = H5Dget_space(srcDset);
    const int rank = H5Sget_simple_extent_ndims(srcSpace);
    std::vector<hsize_t> srcDims(static_cast<std::size_t>(rank));
    H5Sget_simple_extent_dims(srcSpace, srcDims.data(), nullptr);
    H5Sclose(srcSpace);
    const hsize_t nRows = srcDims.front();
    if (nRows == 0) { return; }

    // Every extensible dataset this class creates uses a fixed-size
    // element type (H5T_NATIVE_ULONG, H5T_NATIVE_DOUBLE, or the
    // fixed-width "rng" string type) as its on-disk type directly, so
    // reading/writing through that same type as both the file and
    // memory type (rather than converting to/from some other native
    // type) round-trips the raw bytes exactly, with no variable-length
    // data to separately manage
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

// Append every extensible dataset in every group of srcFile onto the
// correspondingly-named dataset of the same group in dstFile -- see
// consolidateFiles()'s own header comment
static void appendFileContents(const hid_t srcFile, const hid_t dstFile)
{
    H5G_info_t rootInfo{};
    H5Gget_info(srcFile, &rootInfo);
    for (hsize_t i = 0; i < rootInfo.nlinks; ++i)
    {
        const std::string groupName = childNameByIdx(srcFile, i);
        const hid_t srcChild = H5Oopen(srcFile, groupName.c_str(), H5P_DEFAULT);
        if (H5Iget_type(srcChild) != H5I_GROUP) { H5Oclose(srcChild); continue; }

        const hid_t dstGroup = H5Gopen2(dstFile, groupName.c_str(), H5P_DEFAULT);

        H5G_info_t groupInfo{};
        H5Gget_info(srcChild, &groupInfo);
        for (hsize_t j = 0; j < groupInfo.nlinks; ++j)
        {
            const std::string dsetName = childNameByIdx(srcChild, j);
            const hid_t srcObj = H5Oopen(srcChild, dsetName.c_str(), H5P_DEFAULT);
            if (H5Iget_type(srcObj) == H5I_DATASET && isExtensibleDataset(srcObj))
            {
                const hid_t dstDset = H5Dopen2(dstGroup, dsetName.c_str(), H5P_DEFAULT);
                appendDatasetContents(srcObj, dstDset);
                H5Dclose(dstDset);
            }
            H5Oclose(srcObj);
        }

        H5Gclose(dstGroup);
        H5Oclose(srcChild);
    }
}

// NOLINTEND(misc-include-cleaner)

// H5 constructor: see this class's own header comment for what "the
// output file(s)" means here. Without OpenMP, this opens a single
// outDir/modelName.h5, exactly as before. With OpenMP, each thread
// instead gets its own private outDir/modelName/thread_NNNN.h5 --
// eliminating any concurrent access to a single HDF5 file, which HDF5
// does not safely support unless built with its own (opt-in)
// thread-safety support (see this class's own header comment) -- and
// the destructor consolidates them back into a single
// outDir/modelName.h5 once the run completes, unless
// SimControls::outputMode() is OutputMode::h5divided. See
// openNewOutputFiles()'s own comment for the full detail of what
// happens below.
io::OutputManagerH5::OutputManagerH5(
    const SimControls& simControls,
    const toml::table& inputDeck) :
    OutputManager(simControls, inputDeck)
{
    openNewOutputFiles();
}

// See this method's own header comment for the full design, in
// particular the guarantees it relies on its caller to have already
// established before calling this
void io::OutputManagerH5::checkpoint(const unsigned long trialsCompleted)
{
#ifdef _OPENMP
#pragma omp parallel
    {
        closeOutputFile(trialsCompleted);
    }
#else
    closeOutputFile(trialsCompleted);
#endif
    ++checkpointNumber_;
    openNewOutputFiles();
}

// See this method's own header comment
auto io::OutputManagerH5::checkpointModelName(const unsigned long checkpointNum) const -> std::string
{
    std::ostringstream name;
    name << simControls_.modelName() << "_chk" <<
        std::setfill('0') << std::setw(5) << checkpointNum;
    return name.str();
}

// See this method's own header comment for the full design
void io::OutputManagerH5::openNewOutputFiles()
{
    const std::string modelName = (simControls_.checkpointInterval() != 0) ?
        checkpointModelName(checkpointNumber_) : simControls_.modelName();

#ifdef _OPENMP
    const auto threadDir = std::filesystem::path(simControls_.outDir()) / modelName;
    const auto finalPath = std::filesystem::path(simControls_.outDir()) / (modelName + ".h5");
    if (std::filesystem::exists(threadDir) || std::filesystem::exists(finalPath))
    {
        throw std::runtime_error(
            "OutputManagerH5: output " + threadDir.string() + " or " +
            finalPath.string() + " already exists");
    }
    std::filesystem::create_directory(threadDir);

#pragma omp parallel
    {
        std::ostringstream threadFile;
        threadFile << "thread_" << std::setfill('0') << std::setw(4) <<
            omp_get_thread_num() << ".h5";
        openOutputFile(threadDir / threadFile.str());
    }
#else
    const auto finalPath = std::filesystem::path(simControls_.outDir()) / (modelName + ".h5");
    if (std::filesystem::exists(finalPath))
    {
        throw std::runtime_error(
            "OutputManagerH5: output file " + finalPath.string() + " already exists");
    }
    openOutputFile(finalPath);
#endif
}

// Create one HDF5 file at path, write its header (slug-hash, date,
// time, rng_state) as top-level attributes, then dump the toml input
// deck into an input_deck group, and finally create whichever of the
// clusters/cluster_spectra/cluster_phot/galaxy/galaxy_spectra/
// galaxy_phot groups are enabled -- see this method's own header
// comment
void io::OutputManagerH5::openOutputFile(const std::filesystem::path& path)
{
    // Reset this thread's own handles to -1 (the "not open" sentinel
    // every hid_t handle in this class starts from) before opening
    // its file: openClustersGroup()/etc. below each leave their own
    // handle untouched, rather than explicitly setting it to -1, when
    // the corresponding output.write_* flag is false, relying on it
    // already being -1 here -- see openNewOutputFiles()'s own comment
    // for why this happens per-thread, right here, rather than as a
    // separate, whole-ThreadVec reset up front the way earlier,
    // pre-checkpointing code did.
    file_() = -1;
    clustersGroup_() = -1;
    clusterSpectraGroup_() = -1;
    clusterPhotGroup_() = -1;
    galaxyGroup_() = -1;
    galaxySpectraGroup_() = -1;
    galaxyPhotGroup_() = -1;

    // The HDF5 build linked here is not configured with its own
    // (opt-in) thread-safety support (see this class's own header
    // comment), which turns out to mean it is not safe to call from
    // two threads concurrently AT ALL -- even when each thread only
    // ever touches its own, entirely separate file. A per-file
    // critical section (as every write* method below once used) does
    // not protect against this: two threads each creating their own
    // file can still race on the library's own internal global state
    // and crash. So every call into HDF5 anywhere in this class,
    // across every thread, shares this single, global critical
    // section, regardless of which file/object it targets.
#ifdef _OPENMP
#pragma omp critical(h5ThreadSafety)
#endif
    {
        // NOLINTBEGIN(misc-include-cleaner)
        file_() = H5Fcreate(path.string().c_str(), H5F_ACC_EXCL, H5P_DEFAULT, H5P_DEFAULT);
        if (file_() < 0)
        {
            throw std::runtime_error(
                "OutputManagerH5: unable to create output file " + path.string());
        }

        const auto [date, time] = currentDateAndTime();
        writeStringAttr(file_(), "slug-hash", slugGitHash);
        writeStringAttr(file_(), "date", date);
        writeStringAttr(file_(), "time", time);
        writeStringAttr(file_(), "rng_state", currentRngStateString());

        const hid_t inputDeckGrp = H5Gcreate2(file_(), "input_deck",
            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (inputDeckGrp < 0)
        {
            H5Fclose(file_());
            throw std::runtime_error(
                "OutputManagerH5: unable to create input_deck group");
        }
        std::ostringstream tomlStream;
        tomlStream << inputDeck_;
        writeStringDataset(inputDeckGrp, "toml", tomlStream.str());
        H5Gclose(inputDeckGrp);
        // NOLINTEND(misc-include-cleaner)

        openClustersGroup();
        openClusterSpectraGroup();
        openClusterPhotGroup();
        openGalaxyGroup();
        openGalaxySpectraGroup();
        openGalaxyPhotGroup();
    }
}

// Create the clusters group and its datasets, if output.write_cluster
// (optional, defaults to true) was not set to false
void io::OutputManagerH5::openClustersGroup()
{
    if (!writeCluster_) { return; }

    // NOLINTBEGIN(misc-include-cleaner)
    clustersGroup_() = H5Gcreate2(file_(), "clusters",
        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (clustersGroup_() < 0)
    {
        H5Fclose(file_());
        throw std::runtime_error(
            "OutputManagerH5: unable to create clusters group");
    }

    const hid_t trialDset = createExtensible1dDataset(
        clustersGroup_(), "trial", H5T_NATIVE_ULONG);
    writeStringAttr(trialDset, "units", "");
    H5Dclose(trialDset);
    const hid_t uidDset = createExtensible1dDataset(
        clustersGroup_(), "uid", H5T_NATIVE_ULONG);
    writeStringAttr(uidDset, "units", "");
    H5Dclose(uidDset);
    const hid_t targetMassDset = createExtensible1dDataset(
        clustersGroup_(), "target_mass", H5T_NATIVE_DOUBLE);
    writeStringAttr(targetMassDset, "units", "Msun");
    H5Dclose(targetMassDset);
    const hid_t birthMassDset = createExtensible1dDataset(
        clustersGroup_(), "birth_mass", H5T_NATIVE_DOUBLE);
    writeStringAttr(birthMassDset, "units", "Msun");
    H5Dclose(birthMassDset);
    const hid_t formTimeDset = createExtensible1dDataset(
        clustersGroup_(), "form_time", H5T_NATIVE_DOUBLE);
    writeStringAttr(formTimeDset, "units", "yr");
    H5Dclose(formTimeDset);
    const hid_t fehDset = createExtensible1dDataset(
        clustersGroup_(), "feh", H5T_NATIVE_DOUBLE);
    writeStringAttr(fehDset, "units", "");
    H5Dclose(fehDset);
    const hid_t rngType = fixedStrType(utils::rngStateWidth);
    const hid_t rngDset = createExtensible1dDataset(
        clustersGroup_(), "rng", rngType);
    writeStringAttr(rngDset, "units", "");
    H5Dclose(rngDset);
    H5Tclose(rngType);

    if (simControls_.extinct() != nullptr)
    {
        const hid_t aVDset = createExtensible1dDataset(
            clustersGroup_(), "A_V", H5T_NATIVE_DOUBLE);
        writeStringAttr(aVDset, "units", "magnitudes");
        H5Dclose(aVDset);
    }
    // NOLINTEND(misc-include-cleaner)
}

// Create the cluster_spectra group and its datasets, if a spectral
// synthesizer was requested for this simulation and output.write_cluster_spec
// (optional, defaults to true) was not set to false -- spectra can be
// wanted only as an intermediate for computing photometry, in which
// case writing them out as well just wastes disk space
void io::OutputManagerH5::openClusterSpectraGroup()
{
    if (simControls_.specsyn() == nullptr) { return; }
    if (!writeClusterSpec_) { return; }

    const auto& synth = *simControls_.specsyn();
    const std::vector<double> wlObs = synth.wlObs();
    const auto nWl = static_cast<hsize_t>(wlObs.size());

    // NOLINTBEGIN(misc-include-cleaner)
    clusterSpectraGroup_() = H5Gcreate2(file_(), "cluster_spectra",
        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (clusterSpectraGroup_() < 0)
    {
        H5Fclose(file_());
        throw std::runtime_error(
            "OutputManagerH5: unable to create cluster_spectra group");
    }

    writeFixed1dDataset(clusterSpectraGroup_(), "wl", H5T_NATIVE_DOUBLE,
        wlObs.data(), nWl, "Angstrom");

    const hid_t trialSpecDset = createExtensible1dDataset(
        clusterSpectraGroup_(), "trial", H5T_NATIVE_ULONG);
    writeStringAttr(trialSpecDset, "units", "");
    H5Dclose(trialSpecDset);
    const hid_t timeDset = createExtensible1dDataset(
        clusterSpectraGroup_(), "time", H5T_NATIVE_DOUBLE);
    writeStringAttr(timeDset, "units", "yr");
    H5Dclose(timeDset);
    const hid_t uidSpecDset = createExtensible1dDataset(
        clusterSpectraGroup_(), "uid", H5T_NATIVE_ULONG);
    writeStringAttr(uidSpecDset, "units", "");
    H5Dclose(uidSpecDset);
    const hid_t specDset = createExtensible2dDataset(
        clusterSpectraGroup_(), "spec", H5T_NATIVE_DOUBLE, nWl);
    writeStringAttr(specDset, "units", "erg/(s Angstrom)");
    H5Dclose(specDset);

    // specNeb (if requested) lives on the same wl grid as spec -- no
    // separate wavelength dataset is needed
    createSpecNebDataset(simControls_, clusterSpectraGroup_(), nWl);
    createNebLinesDatasets(simControls_, clusterSpectraGroup_());

    if (simControls_.extinct() != nullptr)
    {
        const std::vector<double> wlExtinctObs = simControls_.extinct()->wlObs();
        const auto nWlExtinct = static_cast<hsize_t>(wlExtinctObs.size());
        writeFixed1dDataset(clusterSpectraGroup_(), "wl_extinct", H5T_NATIVE_DOUBLE,
            wlExtinctObs.data(), nWlExtinct, "Angstrom");
        const hid_t specExtinctDset = createExtensible2dDataset(
            clusterSpectraGroup_(), "spec_extinct", H5T_NATIVE_DOUBLE, nWlExtinct);
        writeStringAttr(specExtinctDset, "units", "erg/(s Angstrom)");
        H5Dclose(specExtinctDset);

        // specNebExtinct (if requested) lives on the same
        // (extinction-curve-restricted) grid as specExtinct -- reuses
        // wl_extinct
        createSpecNebExtinctDataset(simControls_, clusterSpectraGroup_(), nWlExtinct);
    }
    // NOLINTEND(misc-include-cleaner)
}

// Create the cluster_phot group and its datasets, if a filter
// collection or the bolometric luminosity was requested for this
// simulation
void io::OutputManagerH5::openClusterPhotGroup()
{
    if (simControls_.filters() == nullptr && !simControls_.computeLbol()) { return; }
    if (!writeClusterPhot_) { return; }

    std::vector<std::string> filterNames;
    std::vector<std::string> filterUnits;
    if (simControls_.filters() != nullptr)
    {
        filterNames = simControls_.filters()->filterNames();
        filterUnits = simControls_.filters()->filterUnits();
    }
    if (simControls_.computeLbol())
    {
        filterNames.emplace_back("Lbol");
        filterUnits.emplace_back("Lsun");
    }
    const auto nFilters = static_cast<hsize_t>(filterNames.size());

    // NOLINTBEGIN(misc-include-cleaner)
    clusterPhotGroup_() = H5Gcreate2(file_(), "cluster_phot",
        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (clusterPhotGroup_() < 0)
    {
        H5Fclose(file_());
        throw std::runtime_error(
            "OutputManagerH5: unable to create cluster_phot group");
    }

    writeStringArrayAttr(clusterPhotGroup_(), "filters", filterNames);

    const hid_t trialPhotDset = createExtensible1dDataset(
        clusterPhotGroup_(), "trial", H5T_NATIVE_ULONG);
    writeStringAttr(trialPhotDset, "units", "");
    H5Dclose(trialPhotDset);
    const hid_t timePhotDset = createExtensible1dDataset(
        clusterPhotGroup_(), "time", H5T_NATIVE_DOUBLE);
    writeStringAttr(timePhotDset, "units", "yr");
    H5Dclose(timePhotDset);
    const hid_t uidPhotDset = createExtensible1dDataset(
        clusterPhotGroup_(), "uid", H5T_NATIVE_ULONG);
    writeStringAttr(uidPhotDset, "units", "");
    H5Dclose(uidPhotDset);
    const hid_t photDset = createExtensible2dDataset(
        clusterPhotGroup_(), "phot", H5T_NATIVE_DOUBLE, nFilters);
    // Each filter can have its own unit (e.g. a photon-count filter's
    // "photon/s" alongside another filter's magnitude system), so
    // this is a per-column string array -- unlike every other dataset
    // here, whose units are uniform across the whole dataset -- in
    // the same order as the "filters" attribute above
    writeStringArrayAttr(photDset, "units", filterUnits);
    H5Dclose(photDset);

    // phot_extinct/phot_neb/phot_neb_extinct only ever cover real
    // filters, never Lbol -- Lbol is a separate bolometric quantity
    // computed directly from the stellar tracks (see
    // Cluster::computeLbol()), not from the (extincted/nebular-
    // inclusive or otherwise) spectrum, so it has no extincted or
    // nebular-inclusive counterpart; sized independently of
    // nFilters/phot above rather than reusing them, since those may
    // include the "Lbol" entry appended earlier in this function
    if (simControls_.filters() != nullptr)
    {
        const auto& realFilterNames = simControls_.filters()->filterNames();
        const auto& realFilterUnits = simControls_.filters()->filterUnits();
        const auto nRealFilters = static_cast<hsize_t>(realFilterNames.size());

        if (simControls_.extinct() != nullptr)
        {
            const hid_t photExtinctDset = createExtensible2dDataset(
                clusterPhotGroup_(), "phot_extinct", H5T_NATIVE_DOUBLE, nRealFilters);
            writeStringArrayAttr(photExtinctDset, "units", realFilterUnits);
            H5Dclose(photExtinctDset);
        }

        createPhotNebDatasets(simControls_, clusterPhotGroup_(), nRealFilters, realFilterUnits);
    }
    // NOLINTEND(misc-include-cleaner)
}

// Create the galaxy group and its datasets, for a galaxy-type
// simulation. A no-op for a cluster-type simulation, which has no
// Galaxy object at all.
void io::OutputManagerH5::openGalaxyGroup()
{
    if (simControls_.simType() != SimControls::SimType::galaxy) { return; }
    if (!writeGalaxy_) { return; }

    // NOLINTBEGIN(misc-include-cleaner)
    galaxyGroup_() = H5Gcreate2(file_(), "galaxy",
        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (galaxyGroup_() < 0)
    {
        H5Fclose(file_());
        throw std::runtime_error(
            "OutputManagerH5: unable to create galaxy group");
    }

    const hid_t trialDset = createExtensible1dDataset(
        galaxyGroup_(), "trial", H5T_NATIVE_ULONG);
    writeStringAttr(trialDset, "units", "");
    H5Dclose(trialDset);
    const hid_t timeDset = createExtensible1dDataset(
        galaxyGroup_(), "time", H5T_NATIVE_DOUBLE);
    writeStringAttr(timeDset, "units", "yr");
    H5Dclose(timeDset);
    const hid_t targetMassDset = createExtensible1dDataset(
        galaxyGroup_(), "target_mass", H5T_NATIVE_DOUBLE);
    writeStringAttr(targetMassDset, "units", "Msun");
    H5Dclose(targetMassDset);
    const hid_t actualMassDset = createExtensible1dDataset(
        galaxyGroup_(), "actual_mass", H5T_NATIVE_DOUBLE);
    writeStringAttr(actualMassDset, "units", "Msun");
    H5Dclose(actualMassDset);
    // NOLINTEND(misc-include-cleaner)
}

// Create the galaxy_spectra group and its datasets, for a galaxy-type
// simulation with a spectral synthesizer requested. A no-op for a
// cluster-type simulation, or if no spectral synthesizer was
// requested.
void io::OutputManagerH5::openGalaxySpectraGroup()
{
    if (simControls_.simType() != SimControls::SimType::galaxy) { return; }
    if (simControls_.specsyn() == nullptr) { return; }
    if (!writeGalaxySpec_) { return; }

    const auto& synth = *simControls_.specsyn();
    const std::vector<double> wlObs = synth.wlObs();
    const auto nWl = static_cast<hsize_t>(wlObs.size());

    // NOLINTBEGIN(misc-include-cleaner)
    galaxySpectraGroup_() = H5Gcreate2(file_(), "galaxy_spectra",
        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (galaxySpectraGroup_() < 0)
    {
        H5Fclose(file_());
        throw std::runtime_error(
            "OutputManagerH5: unable to create galaxy_spectra group");
    }

    writeFixed1dDataset(galaxySpectraGroup_(), "wl", H5T_NATIVE_DOUBLE,
        wlObs.data(), nWl, "Angstrom");

    const hid_t trialSpecDset = createExtensible1dDataset(
        galaxySpectraGroup_(), "trial", H5T_NATIVE_ULONG);
    writeStringAttr(trialSpecDset, "units", "");
    H5Dclose(trialSpecDset);
    const hid_t timeDset = createExtensible1dDataset(
        galaxySpectraGroup_(), "time", H5T_NATIVE_DOUBLE);
    writeStringAttr(timeDset, "units", "yr");
    H5Dclose(timeDset);
    const hid_t specDset = createExtensible2dDataset(
        galaxySpectraGroup_(), "spec", H5T_NATIVE_DOUBLE, nWl);
    writeStringAttr(specDset, "units", "erg/(s Angstrom)");
    H5Dclose(specDset);

    // specNeb (if requested) lives on the same wl grid as spec -- see
    // openClusterSpectraGroup()'s own identical comment
    createSpecNebDataset(simControls_, galaxySpectraGroup_(), nWl);
    createNebLinesDatasets(simControls_, galaxySpectraGroup_());

    if (simControls_.extinct() != nullptr)
    {
        const std::vector<double> wlExtinctObs = simControls_.extinct()->wlObs();
        const auto nWlExtinct = static_cast<hsize_t>(wlExtinctObs.size());
        writeFixed1dDataset(galaxySpectraGroup_(), "wl_extinct", H5T_NATIVE_DOUBLE,
            wlExtinctObs.data(), nWlExtinct, "Angstrom");
        const hid_t specExtinctDset = createExtensible2dDataset(
            galaxySpectraGroup_(), "spec_extinct", H5T_NATIVE_DOUBLE, nWlExtinct);
        writeStringAttr(specExtinctDset, "units", "erg/(s Angstrom)");
        H5Dclose(specExtinctDset);

        createSpecNebExtinctDataset(simControls_, galaxySpectraGroup_(), nWlExtinct);
    }
    // NOLINTEND(misc-include-cleaner)
}

// Create the galaxy_phot group and its datasets, for a galaxy-type
// simulation with a filter collection or the bolometric luminosity
// requested -- mirrors openClusterPhotGroup()'s own filter-list
// construction. A no-op for a cluster-type simulation, or if neither
// was requested.
void io::OutputManagerH5::openGalaxyPhotGroup()
{
    if (simControls_.simType() != SimControls::SimType::galaxy) { return; }
    if (simControls_.filters() == nullptr && !simControls_.computeLbol()) { return; }
    if (!writeGalaxyPhot_) { return; }

    std::vector<std::string> filterNames;
    std::vector<std::string> filterUnits;
    if (simControls_.filters() != nullptr)
    {
        filterNames = simControls_.filters()->filterNames();
        filterUnits = simControls_.filters()->filterUnits();
    }
    if (simControls_.computeLbol())
    {
        filterNames.emplace_back("Lbol");
        filterUnits.emplace_back("Lsun");
    }
    const auto nFilters = static_cast<hsize_t>(filterNames.size());

    // NOLINTBEGIN(misc-include-cleaner)
    galaxyPhotGroup_() = H5Gcreate2(file_(), "galaxy_phot",
        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (galaxyPhotGroup_() < 0)
    {
        H5Fclose(file_());
        throw std::runtime_error(
            "OutputManagerH5: unable to create galaxy_phot group");
    }

    writeStringArrayAttr(galaxyPhotGroup_(), "filters", filterNames);

    const hid_t trialPhotDset = createExtensible1dDataset(
        galaxyPhotGroup_(), "trial", H5T_NATIVE_ULONG);
    writeStringAttr(trialPhotDset, "units", "");
    H5Dclose(trialPhotDset);
    const hid_t timePhotDset = createExtensible1dDataset(
        galaxyPhotGroup_(), "time", H5T_NATIVE_DOUBLE);
    writeStringAttr(timePhotDset, "units", "yr");
    H5Dclose(timePhotDset);
    const hid_t photDset = createExtensible2dDataset(
        galaxyPhotGroup_(), "phot", H5T_NATIVE_DOUBLE, nFilters);
    // Each filter can have its own unit, as in openClusterPhotGroup()
    writeStringArrayAttr(photDset, "units", filterUnits);
    H5Dclose(photDset);

    // phot_extinct/phot_neb/phot_neb_extinct only ever cover real
    // filters, never Lbol -- see openClusterPhotGroup()'s own
    // identical comment
    if (simControls_.filters() != nullptr)
    {
        const auto& realFilterNames = simControls_.filters()->filterNames();
        const auto& realFilterUnits = simControls_.filters()->filterUnits();
        const auto nRealFilters = static_cast<hsize_t>(realFilterNames.size());

        if (simControls_.extinct() != nullptr)
        {
            const hid_t photExtinctDset = createExtensible2dDataset(
                galaxyPhotGroup_(), "phot_extinct", H5T_NATIVE_DOUBLE, nRealFilters);
            writeStringArrayAttr(photExtinctDset, "units", realFilterUnits);
            H5Dclose(photExtinctDset);
        }

        createPhotNebDatasets(simControls_, galaxyPhotGroup_(), nRealFilters, realFilterUnits);
    }
    // NOLINTEND(misc-include-cleaner)
}

// Close every thread's own file(s) -- see this class's own header
// comment -- then, with OpenMP, consolidate them back into a single
// outDir/modelName.h5 (or, per checkpoint, outDir/modelName_chkNNNNN.h5
// -- see checkpointModelName()) unless SimControls::outputMode() is
// OutputMode::h5divided, in which case they are left as-is.
// Consolidation happens once per checkpoint (or once, unchecked, if
// checkpointing is disabled), single-threaded, after every thread has
// closed its own file, since it needs every thread_NNNN.h5 file
// closed (and so fully flushed to disk) before reading them back in.
// Every checkpoint from 0 to checkpointNumber_ is consolidated here,
// not just the last: checkpoint() itself deliberately never
// consolidates (see its own comment), so every earlier checkpoint's
// own thread_NNNN.h5 files are still sitting there, unmerged, until
// this runs.
io::OutputManagerH5::~OutputManagerH5()
{
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        // If we're being destroyed at all, every trial in the run
        // must already be complete -- there is no partial-run
        // trialsCompleted to pass on here the way checkpoint() has,
        // just simControls_.nTrial() itself
        closeOutputFile(simControls_.nTrial());
    }

#ifdef _OPENMP
    if (simControls_.outputMode() == SimControls::OutputMode::h5)
    {
        if (simControls_.checkpointInterval() != 0)
        {
            for (unsigned long chk = 0; chk <= checkpointNumber_; ++chk)
            {
                consolidateFiles(std::filesystem::path(simControls_.outDir()) /
                    checkpointModelName(chk));
            }
        }
        else
        {
            consolidateFiles(std::filesystem::path(simControls_.outDir()) /
                simControls_.modelName());
        }
    }
#endif
}

// Close whichever of this thread's own groups are open, then its
// file -- see openOutputFile()'s own comment for why this shares its
// critical section
void io::OutputManagerH5::closeOutputFile(const unsigned long trialsCompleted)
{
#ifdef _OPENMP
#pragma omp critical(h5ThreadSafety)
#endif
    {
        // NOLINTBEGIN(misc-include-cleaner)
        if (clustersGroup_() >= 0) { H5Gclose(clustersGroup_()); }
        if (clusterSpectraGroup_() >= 0) { H5Gclose(clusterSpectraGroup_()); }
        if (clusterPhotGroup_() >= 0) { H5Gclose(clusterPhotGroup_()); }
        if (galaxyGroup_() >= 0) { H5Gclose(galaxyGroup_()); }
        if (galaxySpectraGroup_() >= 0) { H5Gclose(galaxySpectraGroup_()); }
        if (galaxyPhotGroup_() >= 0) { H5Gclose(galaxyPhotGroup_()); }
        writeULongAttr(file_(), "trials_completed", trialsCompleted);
        H5Fclose(file_());
        // NOLINTEND(misc-include-cleaner)
    }
}

// See this class's own header comment for the full merge algorithm
void io::OutputManagerH5::consolidateFiles(const std::filesystem::path& path)
{
    if (!std::filesystem::is_directory(path))
    {
        throw std::runtime_error(
            "OutputManagerH5::consolidateFiles: " + path.string() + " is not a directory");
    }

    std::vector<std::filesystem::path> threadFiles;
    for (const auto& entry : std::filesystem::directory_iterator(path))
    {
        if (!entry.is_regular_file()) { continue; }
        const auto& name = entry.path().filename().string();
        if (name.starts_with("thread_") && entry.path().extension() == ".h5")
        { threadFiles.push_back(entry.path()); }
    }
    if (threadFiles.empty())
    {
        throw std::runtime_error(
            "OutputManagerH5::consolidateFiles: no thread_*.h5 files found in " + path.string());
    }
    // Zero-padded thread numbers (see the constructor's own
    // thread_NNNN.h5 naming) sort lexicographically in the same order
    // as numerically, so a plain path sort puts thread_0000.h5 first
    std::ranges::sort(threadFiles);

    const auto destPath = path.string() + ".h5";
    std::filesystem::copy_file(threadFiles.front(), destPath);

    // NOLINTBEGIN(misc-include-cleaner)
    const hid_t destFile = H5Fopen(destPath.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
    if (destFile < 0)
    {
        throw std::runtime_error(
            "OutputManagerH5::consolidateFiles: unable to open " + destPath + " for updating");
    }

    for (std::size_t i = 1; i < threadFiles.size(); ++i)
    {
        const hid_t srcFile = H5Fopen(threadFiles.at(i).string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        if (srcFile < 0)
        {
            H5Fclose(destFile);
            throw std::runtime_error(
                "OutputManagerH5::consolidateFiles: unable to open " + threadFiles.at(i).string());
        }
        appendFileContents(srcFile, destFile);
        H5Fclose(srcFile);
    }
    H5Fclose(destFile);
    // NOLINTEND(misc-include-cleaner)

    for (const auto& threadFile : threadFiles) { std::filesystem::remove(threadFile); }
    std::filesystem::remove(path);
}

// Append one element to each of the clusters datasets. A no-op if
// cluster output was not enabled for this simulation. Guards the
// actual writes with the same critical section openOutputFile() uses
// -- see its own comment for why every call into HDF5 needs this,
// not just concurrent calls that happen to target the same file.
void io::OutputManagerH5::writeCluster(
    const unsigned long trial, const core::Cluster& cluster)
{
    if (clustersGroup_() < 0) { return; }

    const unsigned long uid = cluster.uid();
    const double targetMass = cluster.targetMass();
    const double birthMass = cluster.birthMass();
    const double formTime = cluster.formTime();
    const double feH = cluster.feH();
    const auto& rngState = cluster.rngState();

#ifdef _OPENMP
#pragma omp critical(h5ThreadSafety)
#endif
    {
        // NOLINTBEGIN(misc-include-cleaner)
        appendToDataset(clustersGroup_(), "trial", H5T_NATIVE_ULONG, &trial);
        appendToDataset(clustersGroup_(), "uid", H5T_NATIVE_ULONG, &uid);
        appendToDataset(clustersGroup_(), "target_mass", H5T_NATIVE_DOUBLE, &targetMass);
        appendToDataset(clustersGroup_(), "birth_mass", H5T_NATIVE_DOUBLE, &birthMass);
        appendToDataset(clustersGroup_(), "form_time", H5T_NATIVE_DOUBLE, &formTime);
        appendToDataset(clustersGroup_(), "feh", H5T_NATIVE_DOUBLE, &feH);
        const hid_t rngType = fixedStrType(utils::rngStateWidth);
        appendToDataset(clustersGroup_(), "rng", rngType, rngState.data());
        H5Tclose(rngType);
        if (simControls_.extinct() != nullptr)
        {
            const double aV = cluster.aV();
            appendToDataset(clustersGroup_(), "A_V", H5T_NATIVE_DOUBLE, &aV);
        }

        // Flush this thread's own file now, rather than waiting for
        // its eventual H5Fclose() (in this class's own destructor):
        // the cluster's own rng state, just written above, is what a
        // later crash (e.g. an uncaught exception from spectral
        // synthesis below, which aborts the whole process before any
        // destructor can run) needs to be recoverable from disk, to
        // deterministically reproduce that exact cluster afterward
        // (see Cluster's own second constructor overload, which
        // replays a cluster's stochastic draws from a saved rng
        // state). Costs extra I/O on every cluster write, deemed
        // worth it for making every future crash like this one
        // investigable from the output file alone.
        H5Fflush(file_(), H5F_SCOPE_LOCAL);
        // NOLINTEND(misc-include-cleaner)
    }
}

// Append one element to each of the trial/time/uid/spec
// cluster-spectra datasets. A no-op if spectral synthesis was not
// enabled for this simulation (the cluster_spectra group does not
// exist), or if the cluster has disrupted -- a disrupted cluster is
// no longer an observable object, though its light still belongs in
// the total galaxy spectrum, which is handled elsewhere.
void io::OutputManagerH5::writeClusterSpec(
    const unsigned long trial, const double time, core::Cluster& cluster)
{
    if (clusterSpectraGroup_() < 0) { return; }
    if (cluster.isDisrupted()) { return; }

    // spec()/specExtinct() are computed here, outside the critical
    // section below, since they may need to lazily (re)compute the
    // cluster's spectrum -- potentially expensive work that must not
    // run while holding the critical section, which would needlessly
    // serialize it across threads (see this class's own header
    // comment on write* taking non-const Cluster&/Galaxy&)
    const unsigned long uid = cluster.uid();
    const auto& spec = cluster.spec();
    const auto& specExtinct = cluster.specExtinct();
    const auto& specNeb = cluster.specNeb();
    const auto& specNebExtinct = cluster.specNebExtinct();
    const auto& lineLum = cluster.lineLum();
    const auto& lineLumExtinct = cluster.lineLumExtinct();

#ifdef _OPENMP
#pragma omp critical(h5ThreadSafety)
#endif
    {
        // NOLINTBEGIN(misc-include-cleaner)
        appendToDataset(clusterSpectraGroup_(), "trial", H5T_NATIVE_ULONG, &trial);
        appendToDataset(clusterSpectraGroup_(), "time", H5T_NATIVE_DOUBLE, &time);
        appendToDataset(clusterSpectraGroup_(), "uid", H5T_NATIVE_ULONG, &uid);
        appendRowToDataset2d(clusterSpectraGroup_(), "spec", H5T_NATIVE_DOUBLE, spec.data());
        if (simControls_.nebular() != nullptr)
        {
            appendRowToDataset2d(clusterSpectraGroup_(), "spec_neb",
                H5T_NATIVE_DOUBLE, specNeb.data());
        }
        if (simControls_.extinct() != nullptr)
        {
            appendRowToDataset2d(clusterSpectraGroup_(), "spec_extinct",
                H5T_NATIVE_DOUBLE, specExtinct.data());
            if (simControls_.nebular() != nullptr)
            {
                appendRowToDataset2d(clusterSpectraGroup_(), "spec_neb_extinct",
                    H5T_NATIVE_DOUBLE, specNebExtinct.data());
            }
        }
        appendNebLinesRow(simControls_, clusterSpectraGroup_(), lineLum, lineLumExtinct);
        // NOLINTEND(misc-include-cleaner)
    }
}

// Append one element to each of the trial/time/uid/phot cluster_phot
// datasets. A no-op if no filter collection or bolometric luminosity
// was requested for this simulation (the cluster_phot group does not
// exist), or if the cluster has disrupted -- a disrupted cluster is
// no longer an observable object.
void io::OutputManagerH5::writeClusterPhot(
    const unsigned long trial, const double time, core::Cluster& cluster)
{
    if (clusterPhotGroup_() < 0) { return; }
    if (cluster.isDisrupted()) { return; }

    // phot()/photExtinct()/lbol() are computed here, outside the
    // critical section below -- see writeClusterSpec's own comment
    const unsigned long uid = cluster.uid();
    auto phot = cluster.phot();
    if (simControls_.computeLbol()) { phot.push_back(cluster.lbol()); }
    const auto& photExtinct = cluster.photExtinct();
    const auto& photNeb = cluster.photNeb();
    const auto& photNebExtinct = cluster.photNebExtinct();

#ifdef _OPENMP
#pragma omp critical(h5ThreadSafety)
#endif
    {
        // NOLINTBEGIN(misc-include-cleaner)
        appendToDataset(clusterPhotGroup_(), "trial", H5T_NATIVE_ULONG, &trial);
        appendToDataset(clusterPhotGroup_(), "time", H5T_NATIVE_DOUBLE, &time);
        appendToDataset(clusterPhotGroup_(), "uid", H5T_NATIVE_ULONG, &uid);
        appendRowToDataset2d(clusterPhotGroup_(), "phot", H5T_NATIVE_DOUBLE, phot.data());
        if (simControls_.extinct() != nullptr && simControls_.filters() != nullptr)
        {
            appendRowToDataset2d(clusterPhotGroup_(), "phot_extinct",
                H5T_NATIVE_DOUBLE, photExtinct.data());
        }
        appendPhotNebRows(simControls_, clusterPhotGroup_(), photNeb, photNebExtinct);
        // NOLINTEND(misc-include-cleaner)
    }
}

// Append one element to each of the trial/time/target_mass/actual_mass
// galaxy datasets, then call writeCluster() on every currently-alive
// (non-disrupted) cluster in galaxy, so each is also recorded in the
// clusters datasets. A no-op if galaxy output was not enabled for this
// simulation (the galaxy group does not exist).
void io::OutputManagerH5::writeGalaxy(
    const unsigned long trial, const double time, core::Galaxy& galaxy)
{
    if (galaxyGroup_() < 0) { return; }

    const double targetMass = galaxy.targetMass();
    const double actualMass = galaxy.actualMass();

    // Released before the writeCluster() calls below, each of which
    // takes this same critical section itself; OpenMP critical
    // regions are not reentrant, so those calls must happen outside
    // this block.
#ifdef _OPENMP
#pragma omp critical(h5ThreadSafety)
#endif
    {
        // NOLINTBEGIN(misc-include-cleaner)
        appendToDataset(galaxyGroup_(), "trial", H5T_NATIVE_ULONG, &trial);
        appendToDataset(galaxyGroup_(), "time", H5T_NATIVE_DOUBLE, &time);
        appendToDataset(galaxyGroup_(), "target_mass", H5T_NATIVE_DOUBLE, &targetMass);
        appendToDataset(galaxyGroup_(), "actual_mass", H5T_NATIVE_DOUBLE, &actualMass);
        // NOLINTEND(misc-include-cleaner)
    }

    for (const auto& cluster : galaxy.clusters()) { writeCluster(trial, cluster); }
}

// Append one element to each of the trial/time/spec galaxy_spectra
// datasets, then call writeClusterSpec() on every currently-alive
// (non-disrupted) cluster in galaxy. A no-op if spectral synthesis
// was not enabled for this simulation (the galaxy_spectra group does
// not exist).
void io::OutputManagerH5::writeGalaxySpec(
    const unsigned long trial, const double time, core::Galaxy& galaxy)
{
    if (galaxySpectraGroup_() < 0) { return; }

    // spec()/specExtinct() are computed here, outside the critical
    // section below -- see writeClusterSpec's own comment
    const auto& spec = galaxy.spec();
    const auto& specExtinct = galaxy.specExtinct();
    const auto& specNeb = galaxy.specNeb();
    const auto& specNebExtinct = galaxy.specNebExtinct();
    const auto& lineLum = galaxy.lineLum();
    const auto& lineLumExtinct = galaxy.lineLumExtinct();

    // See writeGalaxy's own comment on this critical section
#ifdef _OPENMP
#pragma omp critical(h5ThreadSafety)
#endif
    {
        // NOLINTBEGIN(misc-include-cleaner)
        appendToDataset(galaxySpectraGroup_(), "trial", H5T_NATIVE_ULONG, &trial);
        appendToDataset(galaxySpectraGroup_(), "time", H5T_NATIVE_DOUBLE, &time);
        appendRowToDataset2d(galaxySpectraGroup_(), "spec", H5T_NATIVE_DOUBLE, spec.data());
        if (simControls_.nebular() != nullptr)
        {
            appendRowToDataset2d(galaxySpectraGroup_(), "spec_neb",
                H5T_NATIVE_DOUBLE, specNeb.data());
        }
        if (simControls_.extinct() != nullptr)
        {
            appendRowToDataset2d(galaxySpectraGroup_(), "spec_extinct",
                H5T_NATIVE_DOUBLE, specExtinct.data());
            if (simControls_.nebular() != nullptr)
            {
                appendRowToDataset2d(galaxySpectraGroup_(), "spec_neb_extinct",
                    H5T_NATIVE_DOUBLE, specNebExtinct.data());
            }
        }
        appendNebLinesRow(simControls_, galaxySpectraGroup_(), lineLum, lineLumExtinct);
        // NOLINTEND(misc-include-cleaner)
    }

    for (auto& cluster : galaxy.clusters()) { writeClusterSpec(trial, time, cluster); }
}

// Append one element to each of the trial/time/phot galaxy_phot
// datasets, then call writeClusterPhot() on every currently-alive
// (non-disrupted) cluster in galaxy. A no-op if no filter collection
// or bolometric luminosity was requested for this simulation (the
// galaxy_phot group does not exist).
void io::OutputManagerH5::writeGalaxyPhot(
    const unsigned long trial, const double time, core::Galaxy& galaxy)
{
    if (galaxyPhotGroup_() < 0) { return; }

    // phot()/photExtinct()/lbol() are computed here, outside the
    // critical section below -- see writeClusterSpec's own comment
    auto phot = galaxy.phot();
    if (simControls_.computeLbol()) { phot.push_back(galaxy.lbol()); }
    const auto& photExtinct = galaxy.photExtinct();
    const auto& photNeb = galaxy.photNeb();
    const auto& photNebExtinct = galaxy.photNebExtinct();

    // See writeGalaxy's own comment on this critical section
#ifdef _OPENMP
#pragma omp critical(h5ThreadSafety)
#endif
    {
        // NOLINTBEGIN(misc-include-cleaner)
        appendToDataset(galaxyPhotGroup_(), "trial", H5T_NATIVE_ULONG, &trial);
        appendToDataset(galaxyPhotGroup_(), "time", H5T_NATIVE_DOUBLE, &time);
        appendRowToDataset2d(galaxyPhotGroup_(), "phot", H5T_NATIVE_DOUBLE, phot.data());
        if (simControls_.extinct() != nullptr && simControls_.filters() != nullptr)
        {
            appendRowToDataset2d(galaxyPhotGroup_(), "phot_extinct",
                H5T_NATIVE_DOUBLE, photExtinct.data());
        }
        appendPhotNebRows(simControls_, galaxyPhotGroup_(), photNeb, photNebExtinct);
        // NOLINTEND(misc-include-cleaner)
    }

    for (auto& cluster : galaxy.clusters()) { writeClusterPhot(trial, time, cluster); }
}
