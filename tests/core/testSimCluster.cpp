/**
 * @file testSimCluster.cpp
 * @author Mark Krumholz
 * @brief Unit tests for the SimCluster class.
 * @date 2026-07-17
 */

#include "../src/core/SimCluster.hpp"
#include "../src/io/OutputManager.hpp"
#include "../src/io/OutputManagerH5.hpp"
#include "../src/io/SimControls.hpp"
#include "../src/io/SimPhysics.hpp"
#include "hdf5.h" // NOLINT(misc-include-cleaner)
#include "testSimCluster.hpp"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <toml.hpp>
#include <utility>
#include <vector>
#ifdef _OPENMP
#   include <omp.h>
#endif // _OPENMP

// Number of trials to run and number of threads to run them across;
// nTrial is deliberately not a multiple of nThreads, so that dynamic
// scheduling has to hand a leftover, uneven remainder of trials to
// some thread
static constexpr unsigned long nTrial = 23;
static constexpr int nThreads = 4;

// clusters.CMF in testCluster.in is a fixed value, so every cluster's
// target mass should come out identical regardless of [Fe/H]
static constexpr double expectedTargetMass = 1e3;

using TrialMap = std::map<unsigned long, std::vector<unsigned long>>;

// Build a cluster-simulation input deck with usable stellar physics
// (IMF, tracks, CMF, [Fe/H]), reusing the deck already exercised by
// testCluster/testSimPhysics, with model_name, out_dir, and n_trial
// injected so it can drive a real end-to-end run. If fehDistPath is
// non-empty, stars.FeH is overridden to point at it, replacing the
// deck's default fixed [Fe/H] value with a distribution, so
// SimPhysics::constFeH() comes out false.
static auto makeInputDeck(const std::string& modelName,
    const std::filesystem::path& outDir,
    const std::string& fehDistPath = "") -> toml::table
{
    toml::table inputDeck = toml::parse_file("tests/core/assets/testCluster.in");
    inputDeck.insert("output", toml::table{ { "model_name", modelName } });
    inputDeck.at_path("outputs").as_table()->insert("out_dir", outDir.string());
    inputDeck.insert("n_trial", static_cast<int64_t>(nTrial));
    if (!fehDistPath.empty())
    {
        inputDeck.at_path("stars").as_table()->insert_or_assign("FeH", fehDistPath);
    }
    return inputDeck;
}

// Read an entire 1d extensible dataset of the given HDF5 native type
// into a vector of the corresponding C++ type
template <typename T>
static auto readDataset(const hid_t group, const char* name, const hid_t memType) // NOLINT(misc-include-cleaner)
    -> std::vector<T>
{
    // NOLINTBEGIN(misc-include-cleaner)
    const hid_t dset = H5Dopen2(group, name, H5P_DEFAULT);
    const hid_t space = H5Dget_space(dset);
    hsize_t len = 0;
    H5Sget_simple_extent_dims(space, &len, nullptr);
    std::vector<T> result(len);
    H5Dread(dset, memType, H5S_ALL, H5S_ALL, H5P_DEFAULT, result.data());
    H5Sclose(space);
    H5Dclose(dset);
    // NOLINTEND(misc-include-cleaner)
    return result;
}

// Parse the deck, build SimControls/SimPhysics/OutputManager/
// SimCluster (mirroring main.cpp's end-to-end setup), and run,
// forcing real multi-threaded execution so SimCluster::run's parallel
// for loop actually spans multiple threads. Returns the resulting
// SimPhysics::constFeH(), so callers can confirm the deck actually
// exercised the code path they intended.
static auto runEndToEnd(const toml::table& inputDeck) -> bool
{
    const io::SimControls simControls(inputDeck);
    const io::SimPhysics simPhysics(inputDeck, simControls.simType());

    if (simControls.outputMode() != io::SimControls::OutputMode::h5)
    {
        throw std::runtime_error(
            "testSimCluster: expected default output mode to be h5");
    }
    std::unique_ptr<io::OutputManager> outputManager =
        std::make_unique<io::OutputManagerH5>(simControls, simPhysics, inputDeck);

#ifdef _OPENMP
    omp_set_num_threads(nThreads);
#endif // _OPENMP

    const bool constFeH = simPhysics.constFeH();
    core::SimCluster simCluster(simControls, simPhysics, std::move(outputManager));
    simCluster.run();
    return constFeH;
}

// Read back the trial/uid/target_mass/form_time columns written by
// runEndToEnd
namespace
{
    struct OutputColumns
    {
        std::vector<unsigned long> trial_;
        std::vector<unsigned long> uid_;
        std::vector<double> targetMass_;
        std::vector<double> formTime_;
    };
} // namespace

static auto readOutputColumns(const std::filesystem::path& h5Path) -> OutputColumns
{
    // NOLINTBEGIN(misc-include-cleaner)
    const hid_t file = H5Fopen(h5Path.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0)
    {
        throw std::runtime_error(
            "testSimCluster: unable to reopen " + h5Path.string());
    }
    const hid_t grp = H5Gopen2(file, "clusters", H5P_DEFAULT);

    OutputColumns cols{
        .trial_ = readDataset<unsigned long>(grp, "trial", H5T_NATIVE_ULONG),
        .uid_ = readDataset<unsigned long>(grp, "uid", H5T_NATIVE_ULONG),
        .targetMass_ = readDataset<double>(grp, "target_mass", H5T_NATIVE_DOUBLE),
        .formTime_ = readDataset<double>(grp, "form_time", H5T_NATIVE_DOUBLE)
    };

    H5Gclose(grp);
    H5Fclose(file);
    // NOLINTEND(misc-include-cleaner)
    return cols;
}

// Verify that every row has the expected constant target_mass and
// form_time, and group each row's uid by its trial number
static auto checkRowsAndGroupByTrial(const OutputColumns& cols, TrialMap& rowsByTrial) -> int
{
    constexpr double tol = 1e-6;
    for (size_t i = 0; i < cols.trial_.size(); ++i)
    {
        rowsByTrial[cols.trial_.at(i)].push_back(cols.uid_.at(i));
        if (std::abs(cols.targetMass_.at(i) - expectedTargetMass) >
            tol * expectedTargetMass)
        {
            std::cerr << "testSimCluster: row " << i << " has target_mass "
                << cols.targetMass_.at(i) << ", expected " << expectedTargetMass << "\n";
            return 1;
        }
        if (cols.formTime_.at(i) != 0.0)
        {
            std::cerr << "testSimCluster: row " << i << " has form_time "
                << cols.formTime_.at(i) << ", expected 0\n";
            return 1;
        }
    }
    return 0;
}

// Verify that every trial from 0 to nTrial - 1 appears exactly once
// (writeCluster is called a single time per trial, regardless of the
// number of output times, since it only writes properties fixed at
// cluster formation), and that no uid is reused across trials -- this
// would fail if the dynamic schedule dropped or duplicated a trial,
// if the omp critical guard around the output writes let two
// threads' rows corrupt each other, or if utils::getID() handed out
// a duplicate ID under concurrent use
static auto checkTrialsAndUids(const TrialMap& rowsByTrial) -> int
{
    std::set<unsigned long> seenUids;
    for (unsigned long trial = 0; trial < nTrial; ++trial)
    {
        const auto it = rowsByTrial.find(trial);
        if (it == rowsByTrial.end())
        {
            std::cerr << "testSimCluster: trial " << trial
                << " is missing from the output\n";
            return 1;
        }
        const auto& uids = it->second;
        if (uids.size() != 1)
        {
            std::cerr << "testSimCluster: trial " << trial << " has "
                << uids.size() << " rows, expected 1\n";
            return 1;
        }
        if (!seenUids.insert(uids.front()).second)
        {
            std::cerr << "testSimCluster: uid " << uids.front()
                << " (trial " << trial << ") was already used by another "
                "trial -- utils::getID() handed out a duplicate\n";
            return 1;
        }
    }
    return 0;
}

// Run one end-to-end scenario (either the deck's default fixed [Fe/H],
// or a variable-[Fe/H] distribution) and verify its output. scenario
// names the case for error messages; expectConstFeH is what
// SimPhysics::constFeH() should come out to for the given deck, which
// this also verifies, so a mistake in deck construction can't
// silently turn a scenario into a no-op duplicate of the other one.
static auto runScenario(const std::string& scenario,
    const toml::table& inputDeck,
    const std::filesystem::path& h5Path,
    const bool expectConstFeH) -> int
{
    OutputColumns cols;
    try
    {
        const bool constFeH = runEndToEnd(inputDeck);
        if (constFeH != expectConstFeH)
        {
            std::cerr << "testSimCluster: " << scenario << ": expected "
                "SimPhysics::constFeH() to be " << expectConstFeH
                << ", got " << constFeH << "\n";
            return 1;
        }
        cols = readOutputColumns(h5Path);
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSimCluster: " << scenario
            << ": end-to-end run failed: " << error.what() << "\n";
        return 1;
    }

    const size_t expectedRows = nTrial;
    if (cols.trial_.size() != expectedRows || cols.uid_.size() != expectedRows)
    {
        std::cerr << "testSimCluster: " << scenario << ": expected "
            << expectedRows << " output rows, got " << cols.trial_.size()
            << " trial entries and " << cols.uid_.size() << " uid entries\n";
        return 1;
    }

    TrialMap rowsByTrial;
    if (checkRowsAndGroupByTrial(cols, rowsByTrial) != 0) { return 1; }

    if (rowsByTrial.size() != nTrial)
    {
        std::cerr << "testSimCluster: " << scenario << ": expected "
            << nTrial << " distinct trial numbers, got "
            << rowsByTrial.size() << "\n";
        return 1;
    }

    return checkTrialsAndUids(rowsByTrial);
}

auto testSimCluster() -> int
{
    const auto outDir = std::filesystem::temp_directory_path() / "slugTestSimCluster";
    std::filesystem::remove_all(outDir);
    std::filesystem::create_directories(outDir);

    // Fixed [Fe/H] (constFeH() == true): every cluster references the
    // single Tracks2D/Mesh2DGrid/Interpolator1D set SimPhysics builds
    // once, single-threaded, at construction time
    const std::string constModelName = "test_sim_cluster_const_feh";
    int result = runScenario("const [Fe/H]",
        makeInputDeck(constModelName, outDir),
        outDir / (constModelName + ".h5"),
        true);

    // Variable [Fe/H] (constFeH() == false): every cluster constructor
    // calls Tracks3D::sliceConstFeH from inside SimCluster::run's
    // parallel loop, building its own fresh Mesh2DGrid/Interpolator1D
    // (and thus fresh ThreadVec's) from inside an active parallel
    // region -- this is exactly the case that used to crash before
    // ThreadVec was sized via omp_get_max_threads() instead of a
    // nested parallel region
    const std::string variableModelName = "test_sim_cluster_variable_feh";
    result += runScenario("variable [Fe/H]",
        makeInputDeck(variableModelName, outDir,
            "tests/core/assets/testClusterFeHDist.toml"),
        outDir / (variableModelName + ".h5"),
        false);

    return result;
}
