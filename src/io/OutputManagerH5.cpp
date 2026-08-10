/**
 * @file OutputManagerH5.cpp
 * @author Mark Krumholz
 * @brief Implementation of OutputManagerH5
 * @date 2026-07-17
 */

#include "OutputManagerH5.hpp"
#include "../core/Cluster.hpp"
#include "../phot/FilterCollection.hpp"
#include "../specsyn/Specsyn.hpp"
#include "../utils/ParseUtils.hpp"
#include "../utils/RngThread.hpp"
#include "OutputManager.hpp"
#include "SimControls.hpp"
#include "hdf5.h" // NOLINT(misc-include-cleaner)
#include "io/SlugVersion.hpp"
#include <array>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <string>
#include <toml.hpp>
#include <vector>

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

// NOLINTEND(misc-include-cleaner)

// H5 constructor: create the output file, write the header
// (slug-hash, date, time) as top-level attributes, then dump the toml
// input deck into an input_deck group, and leave the file open
io::OutputManagerH5::OutputManagerH5(
    const SimControls& simControls,
    const toml::table& inputDeck) :
    OutputManager(simControls, inputDeck)
{
    const auto path = std::filesystem::path(simControls_.outDir()) /
        (simControls_.modelName() + ".h5");
    if (std::filesystem::exists(path))
    {
        throw std::runtime_error(
            "OutputManagerH5: output file " + path.string() + " already exists");
    }

    // NOLINTBEGIN(misc-include-cleaner)
    file_ = H5Fcreate(path.string().c_str(), H5F_ACC_EXCL, H5P_DEFAULT, H5P_DEFAULT);
    if (file_ < 0)
    {
        throw std::runtime_error(
            "OutputManagerH5: unable to create output file " + path.string());
    }

    const auto [date, time] = currentDateAndTime();
    writeStringAttr(file_, "slug-hash", slugGitHash);
    writeStringAttr(file_, "date", date);
    writeStringAttr(file_, "time", time);
    writeStringAttr(file_, "rng_state", currentRngStateString());

    const hid_t inputDeckGrp = H5Gcreate2(file_, "input_deck",
        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (inputDeckGrp < 0)
    {
        H5Fclose(file_);
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

// Create the clusters group and its datasets, if cluster output is
// enabled for this simulation
void io::OutputManagerH5::openClustersGroup()
{
    if (simControls_.simType() != SimControls::SimType::cluster &&
        !simControls_.outputClusters())
    {
        return;
    }

    // NOLINTBEGIN(misc-include-cleaner)
    clustersGroup_ = H5Gcreate2(file_, "clusters",
        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (clustersGroup_ < 0)
    {
        H5Fclose(file_);
        throw std::runtime_error(
            "OutputManagerH5: unable to create clusters group");
    }

    const hid_t trialDset = createExtensible1dDataset(
        clustersGroup_, "trial", H5T_NATIVE_ULONG);
    writeStringAttr(trialDset, "units", "");
    H5Dclose(trialDset);
    const hid_t uidDset = createExtensible1dDataset(
        clustersGroup_, "uid", H5T_NATIVE_ULONG);
    writeStringAttr(uidDset, "units", "");
    H5Dclose(uidDset);
    const hid_t targetMassDset = createExtensible1dDataset(
        clustersGroup_, "target_mass", H5T_NATIVE_DOUBLE);
    writeStringAttr(targetMassDset, "units", "Msun");
    H5Dclose(targetMassDset);
    const hid_t birthMassDset = createExtensible1dDataset(
        clustersGroup_, "birth_mass", H5T_NATIVE_DOUBLE);
    writeStringAttr(birthMassDset, "units", "Msun");
    H5Dclose(birthMassDset);
    const hid_t formTimeDset = createExtensible1dDataset(
        clustersGroup_, "form_time", H5T_NATIVE_DOUBLE);
    writeStringAttr(formTimeDset, "units", "yr");
    H5Dclose(formTimeDset);
    const hid_t fehDset = createExtensible1dDataset(
        clustersGroup_, "feh", H5T_NATIVE_DOUBLE);
    writeStringAttr(fehDset, "units", "");
    H5Dclose(fehDset);
    const hid_t rngType = fixedStrType(utils::rngStateWidth);
    const hid_t rngDset = createExtensible1dDataset(
        clustersGroup_, "rng", rngType);
    writeStringAttr(rngDset, "units", "");
    H5Dclose(rngDset);
    H5Tclose(rngType);

    if (simControls_.extinct() != nullptr)
    {
        const hid_t aVDset = createExtensible1dDataset(
            clustersGroup_, "A_V", H5T_NATIVE_DOUBLE);
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

    const auto writeClusterSpecInput = utils::getTOMLKeyWithError<bool>(
        inputDeck_, "output.write_cluster_spec");
    if (writeClusterSpecInput.has_value() && !writeClusterSpecInput.value()) { return; }

    const auto& synth = *simControls_.specsyn();
    const std::vector<double> wlObs = synth.wlObs();
    const auto nWl = static_cast<hsize_t>(wlObs.size());

    // NOLINTBEGIN(misc-include-cleaner)
    clusterSpectraGroup_ = H5Gcreate2(file_, "cluster_spectra",
        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (clusterSpectraGroup_ < 0)
    {
        H5Fclose(file_);
        throw std::runtime_error(
            "OutputManagerH5: unable to create cluster_spectra group");
    }

    writeFixed1dDataset(clusterSpectraGroup_, "wl", H5T_NATIVE_DOUBLE,
        wlObs.data(), nWl, "Angstrom");

    const hid_t trialSpecDset = createExtensible1dDataset(
        clusterSpectraGroup_, "trial", H5T_NATIVE_ULONG);
    writeStringAttr(trialSpecDset, "units", "");
    H5Dclose(trialSpecDset);
    const hid_t timeDset = createExtensible1dDataset(
        clusterSpectraGroup_, "time", H5T_NATIVE_DOUBLE);
    writeStringAttr(timeDset, "units", "yr");
    H5Dclose(timeDset);
    const hid_t uidSpecDset = createExtensible1dDataset(
        clusterSpectraGroup_, "uid", H5T_NATIVE_ULONG);
    writeStringAttr(uidSpecDset, "units", "");
    H5Dclose(uidSpecDset);
    const hid_t specDset = createExtensible2dDataset(
        clusterSpectraGroup_, "spec", H5T_NATIVE_DOUBLE, nWl);
    writeStringAttr(specDset, "units", "erg/(s Angstrom)");
    H5Dclose(specDset);

    if (simControls_.extinct() != nullptr)
    {
        const std::vector<double> wlExtinctObs = simControls_.extinct()->wlObs();
        const auto nWlExtinct = static_cast<hsize_t>(wlExtinctObs.size());
        writeFixed1dDataset(clusterSpectraGroup_, "wl_extinct", H5T_NATIVE_DOUBLE,
            wlExtinctObs.data(), nWlExtinct, "Angstrom");
        const hid_t specExtinctDset = createExtensible2dDataset(
            clusterSpectraGroup_, "spec_extinct", H5T_NATIVE_DOUBLE, nWlExtinct);
        writeStringAttr(specExtinctDset, "units", "erg/(s Angstrom)");
        H5Dclose(specExtinctDset);
    }
    // NOLINTEND(misc-include-cleaner)
}

// Create the cluster_phot group and its datasets, if a filter
// collection or the bolometric luminosity was requested for this
// simulation
void io::OutputManagerH5::openClusterPhotGroup()
{
    if (simControls_.filters() == nullptr && !simControls_.computeLbol()) { return; }

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
    clusterPhotGroup_ = H5Gcreate2(file_, "cluster_phot",
        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (clusterPhotGroup_ < 0)
    {
        H5Fclose(file_);
        throw std::runtime_error(
            "OutputManagerH5: unable to create cluster_phot group");
    }

    writeStringArrayAttr(clusterPhotGroup_, "filters", filterNames);

    const hid_t trialPhotDset = createExtensible1dDataset(
        clusterPhotGroup_, "trial", H5T_NATIVE_ULONG);
    writeStringAttr(trialPhotDset, "units", "");
    H5Dclose(trialPhotDset);
    const hid_t timePhotDset = createExtensible1dDataset(
        clusterPhotGroup_, "time", H5T_NATIVE_DOUBLE);
    writeStringAttr(timePhotDset, "units", "yr");
    H5Dclose(timePhotDset);
    const hid_t uidPhotDset = createExtensible1dDataset(
        clusterPhotGroup_, "uid", H5T_NATIVE_ULONG);
    writeStringAttr(uidPhotDset, "units", "");
    H5Dclose(uidPhotDset);
    const hid_t photDset = createExtensible2dDataset(
        clusterPhotGroup_, "phot", H5T_NATIVE_DOUBLE, nFilters);
    // Each filter can have its own unit (e.g. a photon-count filter's
    // "photon/s" alongside another filter's magnitude system), so
    // this is a per-column string array -- unlike every other dataset
    // here, whose units are uniform across the whole dataset -- in
    // the same order as the "filters" attribute above
    writeStringArrayAttr(photDset, "units", filterUnits);
    H5Dclose(photDset);

    // phot_extinct only ever covers real filters, never Lbol -- Lbol is
    // a separate bolometric quantity computed directly from the
    // stellar tracks (see Cluster::computeLbol()), not from the
    // (extincted or otherwise) spectrum, so it has no extincted
    // counterpart; sized independently of nFilters/phot above rather
    // than reusing them, since those may include the "Lbol" entry
    // appended earlier in this function
    if (simControls_.extinct() != nullptr && simControls_.filters() != nullptr)
    {
        const auto& realFilterNames = simControls_.filters()->filterNames();
        const auto& realFilterUnits = simControls_.filters()->filterUnits();
        const auto nRealFilters = static_cast<hsize_t>(realFilterNames.size());
        const hid_t photExtinctDset = createExtensible2dDataset(
            clusterPhotGroup_, "phot_extinct", H5T_NATIVE_DOUBLE, nRealFilters);
        writeStringArrayAttr(photExtinctDset, "units", realFilterUnits);
        H5Dclose(photExtinctDset);
    }
    // NOLINTEND(misc-include-cleaner)
}

// Create the galaxy group and its datasets, for a galaxy-type
// simulation. A no-op for a cluster-type simulation, which has no
// Galaxy object at all.
void io::OutputManagerH5::openGalaxyGroup()
{
    if (simControls_.simType() != SimControls::SimType::galaxy) { return; }

    // NOLINTBEGIN(misc-include-cleaner)
    galaxyGroup_ = H5Gcreate2(file_, "galaxy",
        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (galaxyGroup_ < 0)
    {
        H5Fclose(file_);
        throw std::runtime_error(
            "OutputManagerH5: unable to create galaxy group");
    }

    const hid_t trialDset = createExtensible1dDataset(
        galaxyGroup_, "trial", H5T_NATIVE_ULONG);
    writeStringAttr(trialDset, "units", "");
    H5Dclose(trialDset);
    const hid_t timeDset = createExtensible1dDataset(
        galaxyGroup_, "time", H5T_NATIVE_DOUBLE);
    writeStringAttr(timeDset, "units", "yr");
    H5Dclose(timeDset);
    const hid_t targetMassDset = createExtensible1dDataset(
        galaxyGroup_, "target_mass", H5T_NATIVE_DOUBLE);
    writeStringAttr(targetMassDset, "units", "Msun");
    H5Dclose(targetMassDset);
    const hid_t actualMassDset = createExtensible1dDataset(
        galaxyGroup_, "actual_mass", H5T_NATIVE_DOUBLE);
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

    const auto& synth = *simControls_.specsyn();
    const std::vector<double> wlObs = synth.wlObs();
    const auto nWl = static_cast<hsize_t>(wlObs.size());

    // NOLINTBEGIN(misc-include-cleaner)
    galaxySpectraGroup_ = H5Gcreate2(file_, "galaxy_spectra",
        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (galaxySpectraGroup_ < 0)
    {
        H5Fclose(file_);
        throw std::runtime_error(
            "OutputManagerH5: unable to create galaxy_spectra group");
    }

    writeFixed1dDataset(galaxySpectraGroup_, "wl", H5T_NATIVE_DOUBLE,
        wlObs.data(), nWl, "Angstrom");

    const hid_t trialSpecDset = createExtensible1dDataset(
        galaxySpectraGroup_, "trial", H5T_NATIVE_ULONG);
    writeStringAttr(trialSpecDset, "units", "");
    H5Dclose(trialSpecDset);
    const hid_t timeDset = createExtensible1dDataset(
        galaxySpectraGroup_, "time", H5T_NATIVE_DOUBLE);
    writeStringAttr(timeDset, "units", "yr");
    H5Dclose(timeDset);
    const hid_t specDset = createExtensible2dDataset(
        galaxySpectraGroup_, "spec", H5T_NATIVE_DOUBLE, nWl);
    writeStringAttr(specDset, "units", "erg/(s Angstrom)");
    H5Dclose(specDset);

    if (simControls_.extinct() != nullptr)
    {
        const std::vector<double> wlExtinctObs = simControls_.extinct()->wlObs();
        const auto nWlExtinct = static_cast<hsize_t>(wlExtinctObs.size());
        writeFixed1dDataset(galaxySpectraGroup_, "wl_extinct", H5T_NATIVE_DOUBLE,
            wlExtinctObs.data(), nWlExtinct, "Angstrom");
        const hid_t specExtinctDset = createExtensible2dDataset(
            galaxySpectraGroup_, "spec_extinct", H5T_NATIVE_DOUBLE, nWlExtinct);
        writeStringAttr(specExtinctDset, "units", "erg/(s Angstrom)");
        H5Dclose(specExtinctDset);
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
    galaxyPhotGroup_ = H5Gcreate2(file_, "galaxy_phot",
        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (galaxyPhotGroup_ < 0)
    {
        H5Fclose(file_);
        throw std::runtime_error(
            "OutputManagerH5: unable to create galaxy_phot group");
    }

    writeStringArrayAttr(galaxyPhotGroup_, "filters", filterNames);

    const hid_t trialPhotDset = createExtensible1dDataset(
        galaxyPhotGroup_, "trial", H5T_NATIVE_ULONG);
    writeStringAttr(trialPhotDset, "units", "");
    H5Dclose(trialPhotDset);
    const hid_t timePhotDset = createExtensible1dDataset(
        galaxyPhotGroup_, "time", H5T_NATIVE_DOUBLE);
    writeStringAttr(timePhotDset, "units", "yr");
    H5Dclose(timePhotDset);
    const hid_t photDset = createExtensible2dDataset(
        galaxyPhotGroup_, "phot", H5T_NATIVE_DOUBLE, nFilters);
    // Each filter can have its own unit, as in openClusterPhotGroup()
    writeStringArrayAttr(photDset, "units", filterUnits);
    H5Dclose(photDset);

    // phot_extinct only ever covers real filters, never Lbol -- see
    // openClusterPhotGroup()'s own identical comment
    if (simControls_.extinct() != nullptr && simControls_.filters() != nullptr)
    {
        const auto& realFilterNames = simControls_.filters()->filterNames();
        const auto& realFilterUnits = simControls_.filters()->filterUnits();
        const auto nRealFilters = static_cast<hsize_t>(realFilterNames.size());
        const hid_t photExtinctDset = createExtensible2dDataset(
            galaxyPhotGroup_, "phot_extinct", H5T_NATIVE_DOUBLE, nRealFilters);
        writeStringArrayAttr(photExtinctDset, "units", realFilterUnits);
        H5Dclose(photExtinctDset);
    }
    // NOLINTEND(misc-include-cleaner)
}

io::OutputManagerH5::~OutputManagerH5()
{
    // NOLINTBEGIN(misc-include-cleaner)
    if (clustersGroup_ >= 0) { H5Gclose(clustersGroup_); }
    if (clusterSpectraGroup_ >= 0) { H5Gclose(clusterSpectraGroup_); }
    if (clusterPhotGroup_ >= 0) { H5Gclose(clusterPhotGroup_); }
    if (galaxyGroup_ >= 0) { H5Gclose(galaxyGroup_); }
    if (galaxySpectraGroup_ >= 0) { H5Gclose(galaxySpectraGroup_); }
    if (galaxyPhotGroup_ >= 0) { H5Gclose(galaxyPhotGroup_); }
    H5Fclose(file_);
    // NOLINTEND(misc-include-cleaner)
}

// Append one element to each of the clusters datasets. A no-op if
// cluster output was not enabled for this simulation.
void io::OutputManagerH5::writeCluster(
    const unsigned long trial, const core::Cluster& cluster)
{
    if (clustersGroup_ < 0) { return; }

    const unsigned long uid = cluster.uid();
    const double targetMass = cluster.targetMass();
    const double birthMass = cluster.birthMass();
    const double formTime = cluster.formTime();
    const double feH = cluster.feH();
    const auto& rngState = cluster.rngState();

    // Guard the actual writes against concurrent callers from other
    // threads; unlike the constructor, this method is expected to be
    // called from inside an openMP parallel region
#ifdef _OPENMP
#pragma omp critical(clusterOutputWrite)
#endif
    {
        // NOLINTBEGIN(misc-include-cleaner)
        appendToDataset(clustersGroup_, "trial", H5T_NATIVE_ULONG, &trial);
        appendToDataset(clustersGroup_, "uid", H5T_NATIVE_ULONG, &uid);
        appendToDataset(clustersGroup_, "target_mass", H5T_NATIVE_DOUBLE, &targetMass);
        appendToDataset(clustersGroup_, "birth_mass", H5T_NATIVE_DOUBLE, &birthMass);
        appendToDataset(clustersGroup_, "form_time", H5T_NATIVE_DOUBLE, &formTime);
        appendToDataset(clustersGroup_, "feh", H5T_NATIVE_DOUBLE, &feH);
        const hid_t rngType = fixedStrType(utils::rngStateWidth);
        appendToDataset(clustersGroup_, "rng", rngType, rngState.data());
        H5Tclose(rngType);
        if (simControls_.extinct() != nullptr)
        {
            const double aV = cluster.aV();
            appendToDataset(clustersGroup_, "A_V", H5T_NATIVE_DOUBLE, &aV);
        }
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
    const unsigned long trial, const double time, const core::Cluster& cluster)
{
    if (clusterSpectraGroup_ < 0) { return; }
    if (cluster.isDisrupted()) { return; }

    const unsigned long uid = cluster.uid();
    const auto& spec = cluster.spec();

    // Guard the actual writes against concurrent callers from other
    // threads, and against writeCluster's own writes to the same
    // file handle -- shares writeCluster's critical section rather
    // than using a separate one, since HDF5 is not thread-safe
    // across concurrent calls into the same file even when they
    // target different groups/datasets, unless built with its
    // (opt-in) thread-safety support
#ifdef _OPENMP
#pragma omp critical(clusterOutputWrite)
#endif
    {
        // NOLINTBEGIN(misc-include-cleaner)
        appendToDataset(clusterSpectraGroup_, "trial", H5T_NATIVE_ULONG, &trial);
        appendToDataset(clusterSpectraGroup_, "time", H5T_NATIVE_DOUBLE, &time);
        appendToDataset(clusterSpectraGroup_, "uid", H5T_NATIVE_ULONG, &uid);
        appendRowToDataset2d(clusterSpectraGroup_, "spec", H5T_NATIVE_DOUBLE, spec.data());
        if (simControls_.extinct() != nullptr)
        {
            appendRowToDataset2d(clusterSpectraGroup_, "spec_extinct",
                H5T_NATIVE_DOUBLE, cluster.specExtinct().data());
        }
        // NOLINTEND(misc-include-cleaner)
    }
}

// Append one element to each of the trial/time/uid/phot cluster_phot
// datasets. A no-op if no filter collection or bolometric luminosity
// was requested for this simulation (the cluster_phot group does not
// exist), or if the cluster has disrupted -- a disrupted cluster is
// no longer an observable object.
void io::OutputManagerH5::writeClusterPhot(
    const unsigned long trial, const double time, const core::Cluster& cluster)
{
    if (clusterPhotGroup_ < 0) { return; }
    if (cluster.isDisrupted()) { return; }

    const unsigned long uid = cluster.uid();
    auto phot = cluster.phot();
    if (simControls_.computeLbol()) { phot.push_back(cluster.lbol()); }

    // Guard the actual writes against concurrent callers from other
    // threads, and against writeCluster's/writeClusterSpec's own
    // writes to the same file handle -- shares their critical
    // section rather than using a separate one, since HDF5 is not
    // thread-safe across concurrent calls into the same file even
    // when they target different groups/datasets, unless built with
    // its (opt-in) thread-safety support
#ifdef _OPENMP
#pragma omp critical(clusterOutputWrite)
#endif
    {
        // NOLINTBEGIN(misc-include-cleaner)
        appendToDataset(clusterPhotGroup_, "trial", H5T_NATIVE_ULONG, &trial);
        appendToDataset(clusterPhotGroup_, "time", H5T_NATIVE_DOUBLE, &time);
        appendToDataset(clusterPhotGroup_, "uid", H5T_NATIVE_ULONG, &uid);
        appendRowToDataset2d(clusterPhotGroup_, "phot", H5T_NATIVE_DOUBLE, phot.data());
        if (simControls_.extinct() != nullptr && simControls_.filters() != nullptr)
        {
            appendRowToDataset2d(clusterPhotGroup_, "phot_extinct",
                H5T_NATIVE_DOUBLE, cluster.photExtinct().data());
        }
        // NOLINTEND(misc-include-cleaner)
    }
}
