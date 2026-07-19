/**
 * @file testSimCluster.cpp
 * @author Mark Krumholz
 * @brief Unit tests for the SimCluster class.
 * @date 2026-07-17
 */

#include "../src/core/SimCluster.hpp"
#include "../src/io/OutputManager.hpp"
#include "../src/io/OutputManagerAscii.hpp"
#include "../src/io/OutputManagerH5.hpp"
#include "../src/io/SimControls.hpp"
#include "../src/io/SimPhysics.hpp"
#include "../src/specsyn/SpecsynBlackbody.hpp"
#include "hdf5.h" // NOLINT(misc-include-cleaner)
#include "testSimCluster.hpp"
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
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

// outTimes for tests/core/assets/testCluster.in is a 3-point grid
// (start_time = 0, end_time = 10, ntime = 3), so each trial writes
// exactly nTime spectra
static constexpr std::size_t nTime = 3;

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

// Read the (rows, cols) extent of a 2d dataset
static auto readDataset2dShape(const hid_t group, const char* name) // NOLINT(misc-include-cleaner)
    -> std::pair<hsize_t, hsize_t>
{
    // NOLINTBEGIN(misc-include-cleaner)
    const hid_t dset = H5Dopen2(group, name, H5P_DEFAULT);
    const hid_t space = H5Dget_space(dset);
    std::array<hsize_t, 2> dims{};
    H5Sget_simple_extent_dims(space, dims.data(), nullptr);
    H5Sclose(space);
    H5Dclose(dset);
    // NOLINTEND(misc-include-cleaner)
    return { dims.at(0), dims.at(1) };
}

// Parse the deck, build SimControls/SimPhysics/OutputManager/
// SimCluster (mirroring main.cpp's end-to-end setup), and run,
// forcing real multi-threaded execution so SimCluster::run's parallel
// for loop actually spans multiple threads. The output manager type
// (h5 or ascii) is chosen from the deck's own outputs.output_mode.
// Returns the resulting SimPhysics::constFeH(), so callers can
// confirm the deck actually exercised the code path they intended.
static auto runEndToEnd(const toml::table& inputDeck) -> bool
{
    const io::SimControls simControls(inputDeck);
    const io::SimPhysics simPhysics(inputDeck, simControls.simType());

    std::unique_ptr<io::OutputManager> outputManager;
    if (simControls.outputMode() == io::SimControls::OutputMode::h5)
    {
        outputManager = std::make_unique<io::OutputManagerH5>(simControls, simPhysics, inputDeck);
    }
    else
    {
        outputManager = std::make_unique<io::OutputManagerAscii>(simControls, simPhysics, inputDeck);
    }

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

// The wavelength grid a default-constructed (z = 0) SpecsynBlackbody
// produces, matching the one SimPhysics builds internally for
// spectra.model = "blackbody", so tests can check the output's
// wavelength grid without reaching into SimPhysics
static auto referenceWlObs() -> std::vector<double>
{
    return specsyn::SpecsynBlackbody().wlObs();
}

// End-to-end check of HDF5 cluster-spectrum output: run with
// spectra.model = "blackbody" (already set in testCluster.in) and
// verify the cluster_spectra group's datasets have the expected
// shapes, and that its wl dataset matches the wavelength grid a
// SpecsynBlackbody produces. We have no independent way to know what
// the spectra themselves should look like, so this checks the form
// of the output rather than its numerical content.
static auto testSimClusterSpectraH5() -> int
{
    const auto outDir = std::filesystem::temp_directory_path() / "slugTestSimClusterSpectraH5";
    std::filesystem::remove_all(outDir);
    std::filesystem::create_directories(outDir);
    const std::string modelName = "test_sim_cluster_spectra_h5";
    const auto h5Path = outDir / (modelName + ".h5");

    try
    {
        runEndToEnd(makeInputDeck(modelName, outDir));

        const auto expectedWl = referenceWlObs();
        const auto nWl = expectedWl.size();
        const auto expectedRows = static_cast<hsize_t>(nTrial) * static_cast<hsize_t>(nTime);

        // NOLINTBEGIN(misc-include-cleaner)
        const hid_t file = H5Fopen(h5Path.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        if (file < 0)
        {
            std::cerr << "testSimCluster: spectraH5: unable to reopen "
                << h5Path.string() << "\n";
            return 1;
        }
        if (H5Lexists(file, "cluster_spectra", H5P_DEFAULT) <= 0)
        {
            std::cerr << "testSimCluster: spectraH5: missing cluster_spectra group\n";
            H5Fclose(file);
            return 1;
        }
        const hid_t grp = H5Gopen2(file, "cluster_spectra", H5P_DEFAULT);

        const auto wl = readDataset<double>(grp, "wl", H5T_NATIVE_DOUBLE);
        const auto trial = readDataset<unsigned long>(grp, "trial", H5T_NATIVE_ULONG);
        const auto time = readDataset<double>(grp, "time", H5T_NATIVE_DOUBLE);
        const auto uid = readDataset<unsigned long>(grp, "uid", H5T_NATIVE_ULONG);
        const auto [specRows, specCols] = readDataset2dShape(grp, "spec");

        H5Gclose(grp);
        H5Fclose(file);
        // NOLINTEND(misc-include-cleaner)

        if (wl.size() != nWl)
        {
            std::cerr << "testSimCluster: spectraH5: wl dataset has size "
                << wl.size() << ", expected " << nWl << "\n";
            return 1;
        }
        for (std::size_t i = 0; i < nWl; ++i)
        {
            if (wl.at(i) != expectedWl.at(i))
            {
                std::cerr << "testSimCluster: spectraH5: wl[" << i << "] = "
                    << wl.at(i) << ", expected " << expectedWl.at(i) << "\n";
                return 1;
            }
        }

        if (trial.size() != expectedRows || time.size() != expectedRows ||
            uid.size() != expectedRows)
        {
            std::cerr << "testSimCluster: spectraH5: expected " << expectedRows
                << " rows in trial/time/uid, got " << trial.size() << "/"
                << time.size() << "/" << uid.size() << "\n";
            return 1;
        }

        if (specRows != expectedRows || specCols != nWl)
        {
            std::cerr << "testSimCluster: spectraH5: spec dataset has shape ("
                << specRows << ", " << specCols << "), expected ("
                << expectedRows << ", " << nWl << ")\n";
            return 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSimCluster: spectraH5 test failed: "
            << error.what() << "\n";
        return 1;
    }
    return 0;
}

// End-to-end check of ascii cluster-spectrum output: run with
// spectra.model = "blackbody" and outputs.output_mode = "ascii", and
// verify the cluster_spectra.txt file has the expected number of
// data lines (nTrial * nTime * nWl -- one line per wavelength, per
// output time, per trial) and that every block of nWl consecutive
// lines carries the expected wavelength grid, in order.
static auto testSimClusterSpectraAscii() -> int
{
    const auto outDir = std::filesystem::temp_directory_path() / "slugTestSimClusterSpectraAscii";
    std::filesystem::remove_all(outDir);
    std::filesystem::create_directories(outDir);
    const std::string modelName = "test_sim_cluster_spectra_ascii";
    const auto specPath = outDir / (modelName + "_cluster_spectra.txt");

    try
    {
        toml::table inputDeck = makeInputDeck(modelName, outDir);
        inputDeck.at_path("outputs").as_table()->insert_or_assign(
            "output_mode", std::string("ascii"));

        runEndToEnd(inputDeck);

        const auto expectedWl = referenceWlObs();
        const auto nWl = expectedWl.size();

        std::ifstream file(specPath);
        if (!file)
        {
            std::cerr << "testSimCluster: spectraAscii: unable to open "
                << specPath.string() << "\n";
            return 1;
        }

        std::string headerLine;
        std::string ruleLine;
        std::getline(file, headerLine);
        std::getline(file, ruleLine);

        std::vector<std::string> dataLines;
        std::string line;
        while (std::getline(file, line))
        {
            if (!line.empty()) { dataLines.push_back(line); }
        }

        const std::size_t expectedLines = nTrial * nTime * nWl;
        if (dataLines.size() != expectedLines)
        {
            std::cerr << "testSimCluster: spectraAscii: expected " << expectedLines
                << " data lines, got " << dataLines.size() << "\n";
            return 1;
        }

        constexpr double wlTol = 1e-5;
        for (std::size_t block = 0; block < dataLines.size() / nWl; ++block)
        {
            for (std::size_t i = 0; i < nWl; ++i)
            {
                std::istringstream lineStream(dataLines.at((block * nWl) + i));
                unsigned long readTrial = 0;
                double readTime = 0.0;
                unsigned long readUid = 0;
                double readWl = 0.0;
                double readSpec = 0.0;
                lineStream >> readTrial >> readTime >> readUid >> readWl >> readSpec;

                if (readTrial >= nTrial)
                {
                    std::cerr << "testSimCluster: spectraAscii: block " << block
                        << " line " << i << " has out-of-range trial "
                        << readTrial << "\n";
                    return 1;
                }
                if (std::abs(readWl - expectedWl.at(i)) > wlTol * expectedWl.at(i))
                {
                    std::cerr << "testSimCluster: spectraAscii: block " << block
                        << " line " << i << " has wl " << readWl
                        << ", expected " << expectedWl.at(i) << "\n";
                    return 1;
                }
            }
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSimCluster: spectraAscii test failed: "
            << error.what() << "\n";
        return 1;
    }
    return 0;
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

    result += testSimClusterSpectraH5();
    result += testSimClusterSpectraAscii();

    return result;
}
