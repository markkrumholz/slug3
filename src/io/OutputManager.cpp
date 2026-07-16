/**
 * @file OutputManager.cpp
 * @author Mark Krumholz
 * @brief Implementation of OutputManager
 * @date 2026-07-16
 */

#include "OutputManager.hpp"
#include "SimControls.hpp"
#include "hdf5.h" // NOLINT(misc-include-cleaner)
#include "io/SlugVersion.hpp"
#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <ios>
#include <sstream>
#include <stdexcept>
#include <string>
#include <toml.hpp>
#include <utility>

// Return the current local date (YYYY-MM-DD) and time (HH:MM:SS) as
// a pair of strings
static auto currentDateAndTime() -> std::pair<std::string, std::string>
{
    const auto now = std::chrono::system_clock::now();
    const std::time_t nowC = std::chrono::system_clock::to_time_t(now);
    std::tm tmBuf{};
    localtime_r(&nowC, &tmBuf);

    std::ostringstream dateStream;
    dateStream << std::put_time(&tmBuf, "%Y-%m-%d");
    std::ostringstream timeStream;
    timeStream << std::put_time(&tmBuf, "%H:%M:%S");

    return { dateStream.str(), timeStream.str() };
}

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
            "OutputManager: unable to create attribute " + name);
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
            "OutputManager: unable to create dataset " + name);
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
            "OutputManager: unable to create dataset " + name);
    }
    return dset;
}

// NOLINTEND(misc-include-cleaner)

// Column headings, and the field widths used to lay them out, for
// the ascii cluster output file. uidWidth accommodates a 9-digit
// integer; numWidth accommodates a number in exponential notation
// with six decimal places (e.g. "-1.234567e+01"). Both include a
// couple of extra characters of padding so that columns are visibly
// separated.
static constexpr int uidWidth = 12;
static constexpr int numWidth = 16;

// Write the cluster-output ascii header (column names followed by a
// dashed rule) to file
static void writeClustersHeader(std::ofstream& file)
{
    file << std::right << std::setw(uidWidth) << "trial"
         << std::setw(uidWidth) << "uid"
         << std::setw(numWidth) << "target_mass"
         << std::setw(numWidth) << "birth_mass"
         << std::setw(numWidth) << "form_time"
         << std::setw(numWidth) << "feh" << "\n";
    constexpr auto numColumns = 4;
    file << std::string(static_cast<std::string::size_type>(2) * uidWidth, '-')
         << std::string(static_cast<std::string::size_type>(numColumns) * numWidth, '-') << "\n";
}

// Ascii constructor: open the summary file, write the header
// (slug-hash, date, time), then dump the toml input deck, and close
// the file. If the simulation outputs individual clusters, also
// open the cluster output file, write its column-header rows, and
// leave it open for later writing.
io::OutputManager<io::SimControls::OutputMode::ascii>::OutputManager(
    const SimControls& simControls, const toml::table& inputDeck) :
    simControls_(simControls),
    inputDeck_(inputDeck)
{
    const auto path = std::filesystem::path(simControls_.outDir()) /
        (simControls_.modelName() + "_summary.txt");
    if (std::filesystem::exists(path))
    {
        throw std::runtime_error(
            "OutputManager: output file " + path.string() + " already exists");
    }

    std::ofstream file(path);
    if (!file)
    {
        throw std::runtime_error(
            "OutputManager: unable to open output file " + path.string());
    }

    const auto [date, time] = currentDateAndTime();
    file << "slug-hash  " << slugGitHash << "\n"
         << "date  " << date << "\n"
         << "time  " << time << "\n";

    file << "input_deck\n" << inputDeck_ << "\n";

    file.close();

    if (simControls_.simType() == SimControls::SimType::cluster ||
        simControls_.outputClusters())
    {
        const auto clustersPath = std::filesystem::path(simControls_.outDir()) /
            (simControls_.modelName() + "_clusters.txt");
        if (std::filesystem::exists(clustersPath))
        {
            throw std::runtime_error(
                "OutputManager: output file " + clustersPath.string() + " already exists");
        }

        clustersFile_.open(clustersPath);
        if (!clustersFile_)
        {
            throw std::runtime_error(
                "OutputManager: unable to open output file " + clustersPath.string());
        }
        writeClustersHeader(clustersFile_);
    }
}

io::OutputManager<io::SimControls::OutputMode::ascii>::~OutputManager()
{
    if (clustersFile_.is_open()) { clustersFile_.close(); }
}

// H5 constructor: create the output file, write the header
// (slug-hash, date, time) as top-level attributes, then dump the toml
// input deck into an input_deck group, and leave the file open
io::OutputManager<io::SimControls::OutputMode::h5>::OutputManager(
    const SimControls& simControls, const toml::table& inputDeck) :
    simControls_(simControls),
    inputDeck_(inputDeck)
{
    const auto path = std::filesystem::path(simControls_.outDir()) /
        (simControls_.modelName() + ".h5");
    if (std::filesystem::exists(path))
    {
        throw std::runtime_error(
            "OutputManager: output file " + path.string() + " already exists");
    }

    // NOLINTBEGIN(misc-include-cleaner)
    file_ = H5Fcreate(path.string().c_str(), H5F_ACC_EXCL, H5P_DEFAULT, H5P_DEFAULT);
    if (file_ < 0)
    {
        throw std::runtime_error(
            "OutputManager: unable to create output file " + path.string());
    }

    const auto [date, time] = currentDateAndTime();
    writeStringAttr(file_, "slug-hash", slugGitHash);
    writeStringAttr(file_, "date", date);
    writeStringAttr(file_, "time", time);

    const hid_t inputDeckGrp = H5Gcreate2(file_, "input_deck",
        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (inputDeckGrp < 0)
    {
        H5Fclose(file_);
        throw std::runtime_error(
            "OutputManager: unable to create input_deck group");
    }
    std::ostringstream tomlStream;
    tomlStream << inputDeck_;
    writeStringDataset(inputDeckGrp, "toml", tomlStream.str());
    H5Gclose(inputDeckGrp);
    // NOLINTEND(misc-include-cleaner)

    if (simControls_.simType() == SimControls::SimType::cluster ||
        simControls_.outputClusters())
    {
        // NOLINTBEGIN(misc-include-cleaner)
        clustersGroup_ = H5Gcreate2(file_, "clusters",
            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (clustersGroup_ < 0)
        {
            H5Fclose(file_);
            throw std::runtime_error(
                "OutputManager: unable to create clusters group");
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
}

io::OutputManager<io::SimControls::OutputMode::h5>::~OutputManager()
{
    // NOLINTBEGIN(misc-include-cleaner)
    if (clustersGroup_ >= 0) { H5Gclose(clustersGroup_); }
    H5Fclose(file_);
    // NOLINTEND(misc-include-cleaner)
}
