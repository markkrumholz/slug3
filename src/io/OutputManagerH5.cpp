/**
 * @file OutputManagerH5.cpp
 * @author Mark Krumholz
 * @brief Implementation of OutputManagerH5
 * @date 2026-07-17
 */

#include "OutputManagerH5.hpp"
#include "../core/Cluster.hpp"
#include "../specsyn/Specsyn.hpp"
#include "OutputManager.hpp"
#include "SimControls.hpp"
#include "SimPhysics.hpp"
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
// HDF5 datatype, in the HDF5 group loc, and immediately write data
// (an array of len elements) into it
static void writeFixed1dDataset(const hid_t loc, const std::string& name,
    const hid_t type, const void* data, const hsize_t len)
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
    const SimControls& simControls, const SimPhysics& simPhysics,
    const toml::table& inputDeck) :
    OutputManager(simControls, simPhysics, inputDeck)
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
    H5Dclose(trialDset);
    const hid_t uidDset = createExtensible1dDataset(
        clustersGroup_, "uid", H5T_NATIVE_ULONG);
    H5Dclose(uidDset);
    const hid_t targetMassDset = createExtensible1dDataset(
        clustersGroup_, "target_mass", H5T_NATIVE_DOUBLE);
    H5Dclose(targetMassDset);
    const hid_t birthMassDset = createExtensible1dDataset(
        clustersGroup_, "birth_mass", H5T_NATIVE_DOUBLE);
    H5Dclose(birthMassDset);
    const hid_t formTimeDset = createExtensible1dDataset(
        clustersGroup_, "form_time", H5T_NATIVE_DOUBLE);
    H5Dclose(formTimeDset);
    const hid_t fehDset = createExtensible1dDataset(
        clustersGroup_, "feh", H5T_NATIVE_DOUBLE);
    H5Dclose(fehDset);
    // NOLINTEND(misc-include-cleaner)
}

// Create the cluster_spectra group and its datasets, if a spectral
// synthesizer was requested for this simulation
void io::OutputManagerH5::openClusterSpectraGroup()
{
    if (simPhysics_.specsyn() == nullptr) { return; }

    const auto& synth = *simPhysics_.specsyn();
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
        wlObs.data(), nWl);

    const hid_t trialSpecDset = createExtensible1dDataset(
        clusterSpectraGroup_, "trial", H5T_NATIVE_ULONG);
    H5Dclose(trialSpecDset);
    const hid_t timeDset = createExtensible1dDataset(
        clusterSpectraGroup_, "time", H5T_NATIVE_DOUBLE);
    H5Dclose(timeDset);
    const hid_t uidSpecDset = createExtensible1dDataset(
        clusterSpectraGroup_, "uid", H5T_NATIVE_ULONG);
    H5Dclose(uidSpecDset);
    const hid_t specDset = createExtensible2dDataset(
        clusterSpectraGroup_, "spec", H5T_NATIVE_DOUBLE, nWl);
    H5Dclose(specDset);
    // NOLINTEND(misc-include-cleaner)
}

io::OutputManagerH5::~OutputManagerH5()
{
    // NOLINTBEGIN(misc-include-cleaner)
    if (clustersGroup_ >= 0) { H5Gclose(clustersGroup_); }
    if (clusterSpectraGroup_ >= 0) { H5Gclose(clusterSpectraGroup_); }
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
        // NOLINTEND(misc-include-cleaner)
    }
}
