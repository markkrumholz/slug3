/**
 * @file testSimCluster.cpp
 * @author Mark Krumholz
 * @brief Unit tests for the SimCluster class.
 * @date 2026-07-17
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#include "../src/core/SimCluster.hpp"
#include "../src/io/OutputManager.hpp"
#include "../src/io/OutputManagerAscii.hpp"
#include "../src/io/OutputManagerH5.hpp"
#include "../src/io/SimControls.hpp"
#include "../src/specsyn/SpecsynBlackbody.hpp"
#include "../src/specsyn/SpecsynCommons.hpp"
#include "../src/utils/UniqueIDManager.hpp"
#include "hdf5.h" // NOLINT(misc-include-cleaner)
#include "testSimCluster.hpp"
#include <array>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
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
// testCluster/testSimControls, with model_name, out_dir, and n_trial
// injected so it can drive a real end-to-end run. If fehDistPath is
// non-empty, stars.FeH is overridden to point at it, replacing the
// deck's default fixed [Fe/H] value with a distribution, so
// SimControls::constFeH() comes out false. deckPath selects which base
// deck to start from -- defaulting to testCluster.in, but overridable
// (e.g. to testClusterPhot.in, for the photometry-enabled scenarios)
// since not every deck needs to carry every optional section.
static auto makeInputDeck(const std::string& modelName,
    const std::filesystem::path& outDir,
    const std::string& fehDistPath = "",
    const std::string& deckPath = "tests/core/assets/testCluster.in") -> toml::table
{
    toml::table inputDeck = toml::parse_file(deckPath);
    if (toml::table* outputTbl = inputDeck["output"].as_table())
    { outputTbl->insert("model_name", modelName); }
    else { inputDeck.insert("output", toml::table{ { "model_name", modelName } }); }
    if (toml::table* outputsTbl = inputDeck["outputs"].as_table())
    { outputsTbl->insert("out_dir", outDir.string()); }
    else { inputDeck.insert("outputs", toml::table{ { "out_dir", outDir.string() } }); }
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

// Read a 1d array-of-strings attribute called name on the HDF5 object
// loc -- used to check the "filters" attribute openClusterPhotGroup
// writes on the cluster_phot group, independently of the writing code
// itself
static auto readStringArrayAttr(const hid_t loc, const char* name) // NOLINT(misc-include-cleaner)
    -> std::vector<std::string>
{
    // NOLINTBEGIN(misc-include-cleaner)
    const hid_t attr = H5Aopen(loc, name, H5P_DEFAULT);
    if (attr < 0)
    {
        throw std::runtime_error(
            "testSimCluster: unable to open attribute " + std::string(name));
    }
    const hid_t aspace = H5Aget_space(attr);
    const auto npoints =
        static_cast<size_t>(H5Sget_simple_extent_npoints(aspace));
    const hid_t memtype = H5Aget_type(attr);

    std::vector<char*> buf(npoints);
    H5Aread(attr, memtype, static_cast<void*>(buf.data())); // NOLINT(bugprone-multi-level-implicit-pointer-conversion)
    std::vector<std::string> values;
    values.reserve(npoints);
    for (const auto* s : buf) { values.emplace_back(s); }

    H5Dvlen_reclaim(memtype, aspace, H5P_DEFAULT,
        static_cast<void*>(buf.data())); // NOLINT(bugprone-multi-level-implicit-pointer-conversion)
    H5Tclose(memtype);
    H5Sclose(aspace);
    H5Aclose(attr);
    // NOLINTEND(misc-include-cleaner)

    return values;
}

// Read a scalar unsigned long attribute called name on the HDF5
// object loc -- used to check the "trials_completed" attribute
// closeOutputFile() writes on each checkpoint's own top-level file,
// independently of the writing code itself
static auto readULongAttr(const hid_t loc, const char* name) -> unsigned long // NOLINT(misc-include-cleaner)
{
    // NOLINTBEGIN(misc-include-cleaner)
    const hid_t attr = H5Aopen(loc, name, H5P_DEFAULT);
    if (attr < 0)
    {
        throw std::runtime_error(
            "testSimCluster: unable to open attribute " + std::string(name));
    }
    unsigned long value = 0;
    H5Aread(attr, H5T_NATIVE_ULONG, static_cast<void*>(&value));
    H5Aclose(attr);
    // NOLINTEND(misc-include-cleaner)
    return value;
}

// Parse the deck, build SimControls/OutputManager/SimCluster
// (mirroring main.cpp's end-to-end setup), and run, forcing real
// multi-threaded execution so SimCluster::run's parallel for loop
// actually spans multiple threads. The output manager type (h5 or
// ascii) is chosen from the deck's own outputs.output_mode (h5 and
// h5divided both route to OutputManagerH5, exactly as main.cpp itself
// does). restart is threaded through to both OutputManagerH5's and
// SimCluster's own constructors -- see testSimClusterRestartFromCheckpoint()
// for the only caller that passes true. Returns the resulting
// SimControls::constFeH(), so callers can confirm the deck actually
// exercised the code path they intended.
static auto runEndToEnd(const toml::table& inputDeck, const bool restart = false) -> bool
{
    const io::SimControls simControls(inputDeck);

    // Must happen before OutputManagerH5 is constructed below: its
    // constructor opens one HDF5 file per OpenMP thread by running its
    // own "#pragma omp parallel" region, so the team size it sees has
    // to already be nThreads, matching the team size SimCluster::run()
    // will later use -- otherwise threads that only show up in run()'s
    // parallel for never get a file/group of their own, and silently
    // drop every trial they process.
#ifdef _OPENMP
    omp_set_num_threads(nThreads);
#endif // _OPENMP

    std::unique_ptr<io::OutputManager> outputManager;
    if (simControls.outputMode() == io::SimControls::OutputMode::h5 ||
        simControls.outputMode() == io::SimControls::OutputMode::h5divided)
    {
        outputManager = std::make_unique<io::OutputManagerH5>(simControls, inputDeck, restart);
    }
    else
    {
        outputManager = std::make_unique<io::OutputManagerAscii>(simControls, inputDeck);
    }

    // Captured before outputManager is moved into simCluster below,
    // since restartTrialsDone() is only meaningful when restart is
    // true (see OutputManager::restartTrialsDone()'s own comment) --
    // used only to compute the expected post-run() trialsCompleted()
    // below, mirroring exactly what SimCluster::run() itself does
    // with it internally.
    const unsigned long startTrial = restart ? outputManager->restartTrialsDone() : 0;

    const bool constFeH = simControls.constFeH();
    core::SimCluster simCluster(simControls, std::move(outputManager), restart);
    if (simCluster.trialsCompleted() != 0)
    {
        throw std::runtime_error("testSimCluster: trialsCompleted() should start at 0");
    }
    simCluster.run();
    const unsigned long expectedTrialsCompleted = simControls.nTrial() - startTrial;
    if (simCluster.trialsCompleted() != expectedTrialsCompleted)
    {
        throw std::runtime_error(
            "testSimCluster: trialsCompleted() should equal nTrial() - startTrial "
            "after run() completes (got " + std::to_string(simCluster.trialsCompleted()) +
            ", expected " + std::to_string(expectedTrialsCompleted) + "; nTrial=" +
            std::to_string(simControls.nTrial()) + ", startTrial=" + std::to_string(startTrial) + ")");
    }
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

// Verify that every trial from 0 to totalTrials - 1 appears exactly
// once (writeCluster is called a single time per trial, regardless of
// the number of output times, since it only writes properties fixed
// at cluster formation), and that no uid is reused across trials --
// this would fail if the dynamic schedule dropped or duplicated a
// trial, if the omp critical guard around the output writes let two
// threads' rows corrupt each other, or if utils::uniqueID() handed
// out a duplicate ID under concurrent use. totalTrials defaults to
// this file's own module-level nTrial constant, which every caller
// except testSimClusterSigtermGracefulExit() (a deliberately smaller,
// separately-sized run) actually uses.
static auto checkTrialsAndUids(const TrialMap& rowsByTrial,
    const unsigned long totalTrials = nTrial) -> int
{
    std::set<unsigned long> seenUids;
    for (unsigned long trial = 0; trial < totalTrials; ++trial)
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
                "trial -- utils::uniqueID() handed out a duplicate\n";
            return 1;
        }
    }
    return 0;
}

// Run one end-to-end scenario (either the deck's default fixed [Fe/H],
// or a variable-[Fe/H] distribution) and verify its output. scenario
// names the case for error messages; expectConstFeH is what
// SimControls::constFeH() should come out to for the given deck, which
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
                "SimControls::constFeH() to be " << expectConstFeH
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

// The wavelength grid a SpecsynBlackbody built at SimControls's own
// default wavelength range/resolution (z = 0) produces, matching the
// one SimControls builds internally for spectra.model = "blackbody"
// when the deck doesn't override spectra.wl_min/wl_max/nwl, so tests
// can check the output's wavelength grid without reaching into
// SimControls. The SimControls this SpecsynBlackbody references is a
// throwaway local: nothing here reads its integrator tolerances, only
// wlObs(), so it doesn't need to outlive this function's own return.
static auto referenceWlObs() -> std::vector<double>
{
    const io::SimControls controls;
    return specsyn::SpecsynBlackbody(
        specsyn::defaultWlMin, specsyn::defaultWlMax, specsyn::defaultNWl,
        controls).wlObs();
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
        std::string unitsLine;
        std::string ruleLine;
        std::getline(file, headerLine);
        std::getline(file, unitsLine);
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

// The columns tests/core/assets/testClusterPhot.in's phot.filters
// resolves to in cluster_phot output: "Lbol" is popped out of
// FilterCollection's own filter list (see testCluster.cpp's own
// testClusterPhot() for where that expectation is derived from), but
// OutputManagerH5/Ascii both append it back as a final column when
// SimControls::computeLbol() is true
static auto expectedPhotFilters() -> std::vector<std::string>
{
    return { "SLUGTEST.CAM1.G500", "ideal_phot_700_1500", "Lbol" };
}

// End-to-end check of HDF5 cluster-photometry output: run with
// tests/core/assets/testClusterPhot.in (phot.filters set, "Lbol"
// included) and verify the cluster_phot group's "filters" attribute
// and dataset shapes come out as expected. As with
// testSimClusterSpectraH5, there is no independent way to know what
// the photometric values themselves should be from here, so this
// checks the form of the output rather than its numerical content --
// testCluster.cpp's own testClusterPhot() already cross-checks the
// numerical content of Cluster::phot() against FilterCollection::phot().
static auto testSimClusterPhotH5() -> int
{
    const auto outDir = std::filesystem::temp_directory_path() / "slugTestSimClusterPhotH5";
    std::filesystem::remove_all(outDir);
    std::filesystem::create_directories(outDir);
    const std::string modelName = "test_sim_cluster_phot_h5";
    const auto h5Path = outDir / (modelName + ".h5");

    try
    {
        runEndToEnd(makeInputDeck(modelName, outDir, "",
            "tests/core/assets/testClusterPhot.in"));

        const auto expectedFilters = expectedPhotFilters();
        const auto nFilt = expectedFilters.size();
        const auto expectedRows = static_cast<hsize_t>(nTrial) * static_cast<hsize_t>(nTime);

        // NOLINTBEGIN(misc-include-cleaner)
        const hid_t file = H5Fopen(h5Path.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        if (file < 0)
        {
            std::cerr << "testSimCluster: photH5: unable to reopen "
                << h5Path.string() << "\n";
            return 1;
        }
        if (H5Lexists(file, "cluster_phot", H5P_DEFAULT) <= 0)
        {
            std::cerr << "testSimCluster: photH5: missing cluster_phot group\n";
            H5Fclose(file);
            return 1;
        }
        const hid_t grp = H5Gopen2(file, "cluster_phot", H5P_DEFAULT);

        const auto filterNames = readStringArrayAttr(grp, "filters");
        const auto trial = readDataset<unsigned long>(grp, "trial", H5T_NATIVE_ULONG);
        const auto time = readDataset<double>(grp, "time", H5T_NATIVE_DOUBLE);
        const auto uid = readDataset<unsigned long>(grp, "uid", H5T_NATIVE_ULONG);
        const auto [photRows, photCols] = readDataset2dShape(grp, "phot");

        H5Gclose(grp);
        H5Fclose(file);
        // NOLINTEND(misc-include-cleaner)

        if (filterNames != expectedFilters)
        {
            std::cerr << "testSimCluster: photH5: \"filters\" attribute did not "
                "match the expected filter names (with \"Lbol\" popped out)\n";
            return 1;
        }

        if (trial.size() != expectedRows || time.size() != expectedRows ||
            uid.size() != expectedRows)
        {
            std::cerr << "testSimCluster: photH5: expected " << expectedRows
                << " rows in trial/time/uid, got " << trial.size() << "/"
                << time.size() << "/" << uid.size() << "\n";
            return 1;
        }

        if (photRows != expectedRows || photCols != nFilt)
        {
            std::cerr << "testSimCluster: photH5: phot dataset has shape ("
                << photRows << ", " << photCols << "), expected ("
                << expectedRows << ", " << nFilt << ")\n";
            return 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSimCluster: photH5 test failed: "
            << error.what() << "\n";
        return 1;
    }
    return 0;
}

// End-to-end check of ascii cluster-photometry output: run with
// tests/core/assets/testClusterPhot.in and outputs.output_mode =
// "ascii", and verify the cluster_phot.txt file has the expected
// number of data lines (nTrial * nTime -- one line per cluster, per
// output time, unlike the per-wavelength cluster-spectra file) and
// that every line has the expected number of whitespace-separated
// photometry fields (trial, time, uid, then one per filter).
static auto testSimClusterPhotAscii() -> int
{
    const auto outDir = std::filesystem::temp_directory_path() / "slugTestSimClusterPhotAscii";
    std::filesystem::remove_all(outDir);
    std::filesystem::create_directories(outDir);
    const std::string modelName = "test_sim_cluster_phot_ascii";
    const auto photPath = outDir / (modelName + "_cluster_phot.txt");

    const auto nFilt = expectedPhotFilters().size();

    try
    {
        toml::table inputDeck = makeInputDeck(modelName, outDir, "",
            "tests/core/assets/testClusterPhot.in");
        inputDeck.at_path("outputs").as_table()->insert_or_assign(
            "output_mode", std::string("ascii"));

        runEndToEnd(inputDeck);

        std::ifstream file(photPath);
        if (!file)
        {
            std::cerr << "testSimCluster: photAscii: unable to open "
                << photPath.string() << "\n";
            return 1;
        }

        std::string headerLine;
        std::string unitsLine;
        std::string ruleLine;
        std::getline(file, headerLine);
        std::getline(file, unitsLine);
        std::getline(file, ruleLine);

        std::vector<std::string> dataLines;
        std::string line;
        while (std::getline(file, line))
        {
            if (!line.empty()) { dataLines.push_back(line); }
        }

        const std::size_t expectedLines = nTrial * nTime;
        if (dataLines.size() != expectedLines)
        {
            std::cerr << "testSimCluster: photAscii: expected " << expectedLines
                << " data lines, got " << dataLines.size() << "\n";
            return 1;
        }

        for (std::size_t i = 0; i < dataLines.size(); ++i)
        {
            std::istringstream lineStream(dataLines.at(i));
            unsigned long readTrial = 0;
            double readTime = 0.0;
            unsigned long readUid = 0;
            lineStream >> readTrial >> readTime >> readUid;

            if (readTrial >= nTrial)
            {
                std::cerr << "testSimCluster: photAscii: line " << i
                    << " has out-of-range trial " << readTrial << "\n";
                return 1;
            }

            std::size_t nPhotValues = 0;
            double photValue = 0.0;
            while (lineStream >> photValue) { ++nPhotValues; }
            if (nPhotValues != nFilt)
            {
                std::cerr << "testSimCluster: photAscii: line " << i
                    << " has " << nPhotValues << " photometry columns, "
                    "expected " << nFilt << "\n";
                return 1;
            }
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSimCluster: photAscii test failed: "
            << error.what() << "\n";
        return 1;
    }
    return 0;
}

// Format outDir/modelName_chkNNNNN.h5, NNNNN being checkpointNum
// zero-padded to 5 digits -- mirrors
// OutputManagerH5::checkpointModelName()'s own format exactly, so
// tests can predict the path a given checkpoint's consolidated output
// ends up at.
static auto checkpointH5Path(const std::filesystem::path& outDir,
    const std::string& modelName, const unsigned long checkpointNum) -> std::filesystem::path
{
    std::ostringstream name;
    name << modelName << "_chk" << std::setfill('0') << std::setw(5) << checkpointNum;
    return outDir / (name.str() + ".h5");
}

// End-to-end check of checkpointed HDF5 output: with
// outputs.checkpoint_interval set, SimCluster::run() should roll over
// to a new modelName_chkNNNNN.h5 file every checkpointInterval trials
// (see SimControls::checkpointInterval()/OutputManagerH5::checkpoint()),
// and the destructor's consolidateFiles() should merge every one of
// them (not just the last -- see its own comment) once the run
// completes. nTrial (23) is deliberately not an exact multiple of
// checkpointInterval (5), so the final checkpoint is deliberately
// partial (3 trials, not 5) -- exercises that case too, not just
// evenly-divided ones. Checks that exactly the expected set of
// modelName_chkNNNNN.h5 files exists (no more, no fewer), that each
// one's own trial numbers fall within the range that checkpoint was
// supposed to cover, that each one's own top-level "trials_completed"
// attribute (see OutputManagerH5::closeOutputFile()) matches the
// cumulative trial count that checkpoint was closed at, and --
// pooling every checkpoint file's own rows together -- that every
// trial from 0 to nTrial - 1 appears exactly once with a unique uid,
// mirroring runScenario()'s own completeness/uniqueness check for the
// unchunked case.
static auto testSimClusterCheckpointedH5() -> int
{
    const auto outDir = std::filesystem::temp_directory_path() / "slugTestSimClusterCheckpointedH5";
    std::filesystem::remove_all(outDir);
    std::filesystem::create_directories(outDir);
    const std::string modelName = "test_sim_cluster_checkpointed_h5";

    constexpr unsigned long checkpointInterval = 5;
    const unsigned long nCheckpoints =
        (nTrial + checkpointInterval - 1) / checkpointInterval; // ceiling division

    try
    {
        toml::table inputDeck = makeInputDeck(modelName, outDir);
        inputDeck.at_path("outputs").as_table()->insert(
            "checkpoint_interval", static_cast<int64_t>(checkpointInterval));

        runEndToEnd(inputDeck);

        TrialMap rowsByTrial;
        for (unsigned long chk = 0; chk < nCheckpoints; ++chk)
        {
            const auto h5Path = checkpointH5Path(outDir, modelName, chk);
            if (!std::filesystem::exists(h5Path))
            {
                std::cerr << "testSimCluster: checkpointedH5: missing "
                    << h5Path.string() << "\n";
                return 1;
            }
            const auto cols = readOutputColumns(h5Path);

            const unsigned long expectedLo = chk * checkpointInterval;
            const unsigned long expectedHi = // exclusive
                std::min((chk + 1) * checkpointInterval, nTrial);

            // NOLINTBEGIN(misc-include-cleaner)
            const hid_t file = H5Fopen(h5Path.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
            const auto trialsCompleted = readULongAttr(file, "trials_completed");
            H5Fclose(file);
            // NOLINTEND(misc-include-cleaner)
            if (trialsCompleted != expectedHi)
            {
                std::cerr << "testSimCluster: checkpointedH5: "
                    << h5Path.string() << " has trials_completed="
                    << trialsCompleted << ", expected " << expectedHi << "\n";
                return 1;
            }

            for (std::size_t i = 0; i < cols.trial_.size(); ++i)
            {
                const auto trial = cols.trial_.at(i);
                if (trial < expectedLo || trial >= expectedHi)
                {
                    std::cerr << "testSimCluster: checkpointedH5: "
                        << h5Path.string() << " has trial " << trial
                        << ", expected in [" << expectedLo << ", "
                        << expectedHi << ")\n";
                    return 1;
                }
                rowsByTrial[trial].push_back(cols.uid_.at(i));
            }
        }

        // No checkpoint beyond nCheckpoints should have been created
        const auto extraPath = checkpointH5Path(outDir, modelName, nCheckpoints);
        if (std::filesystem::exists(extraPath))
        {
            std::cerr << "testSimCluster: checkpointedH5: unexpected extra "
                "checkpoint file " << extraPath.string() << "\n";
            return 1;
        }

        // The unchunked modelName.h5 should never have been created --
        // checkpointing replaces it entirely with modelName_chkNNNNN.h5
        if (std::filesystem::exists(outDir / (modelName + ".h5")))
        {
            std::cerr << "testSimCluster: checkpointedH5: unexpected "
                "unchunked output file " << (outDir / (modelName + ".h5")).string()
                << "\n";
            return 1;
        }

        if (rowsByTrial.size() != nTrial)
        {
            std::cerr << "testSimCluster: checkpointedH5: expected " << nTrial
                << " distinct trial numbers across every checkpoint, got "
                << rowsByTrial.size() << "\n";
            return 1;
        }
        return checkTrialsAndUids(rowsByTrial);
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSimCluster: checkpointedH5 test failed: "
            << error.what() << "\n";
        return 1;
    }
}

// End-to-end check of restarting from a checkpoint. Phase 1 is an
// ordinary (non-restart) run that only covers the first
// firstPhaseTrials trials (a whole number of checkpointInterval-sized
// batches), leaving behind checkpointH5Path()'s own modelName_chkNNNNN.h5
// files -- built directly at that path without OpenMP (as this test
// itself always runs; see readOutputColumns()'s own usage elsewhere in
// this file), or consolidated there by phase 1's own destructor if
// built with OpenMP, either way leaving the same, single-file-per-
// checkpoint shape restartSetup() reads via its own h5Path branch (see
// its own comment; the alternative, per-thread-directory branch is
// only ever reached with OpenMP, mid-run, before a checkpoint's own
// files have been consolidated -- not exercised by this test). Phase 2
// then restarts: a *new* SimControls (with n_trial back up to the real
// total nTrial), OutputManagerH5(..., restart = true), and
// SimCluster(..., restart = true), built and run exactly the way
// main.cpp itself builds and runs them for a real --restart
// invocation.
//
// Checks (via runEndToEnd()'s own restart-aware assertion, for both
// phases) that each phase's own SimCluster::trialsCompleted() only
// counts the trials it actually ran itself, that pooling every
// checkpoint's own rows together, across both phases, yields every
// trial from 0 to nTrial - 1 exactly once with a unique uid -- i.e.
// that restarting produces the exact same complete, non-overlapping
// coverage a single, uninterrupted run would -- and, deliberately
// clobbering utils::uniqueID() between the two phases to simulate a
// genuinely separate process's own fresh counter, that restartSetup()
// actually restores it from the checkpoint's own "restart_uid"
// attribute rather than either colliding with phase 1's own already-
// used uids or leaving a gap (see closeOutputFile()'s/restartSetup()'s
// own comments for why this matters: phase 1 and phase 2 being two
// calls in the same process, as below, would otherwise mask exactly
// this failure mode, since utils::uniqueID()'s own counter would
// already, coincidentally, carry over correctly on its own).
static auto testSimClusterRestartFromCheckpoint() -> int
{
    const auto outDir = std::filesystem::temp_directory_path() /
        "slugTestSimClusterRestartFromCheckpoint";
    std::filesystem::remove_all(outDir);
    std::filesystem::create_directories(outDir);
    const std::string modelName = "test_sim_cluster_restart";

    constexpr unsigned long checkpointInterval = 5;
    constexpr unsigned long firstPhaseTrials = 15; // 3 whole batches of checkpointInterval
    const unsigned long totalCheckpoints =
        (nTrial + checkpointInterval - 1) / checkpointInterval; // ceiling division

    try
    {
        toml::table deck = makeInputDeck(modelName, outDir);
        deck.at_path("outputs").as_table()->insert(
            "checkpoint_interval", static_cast<int64_t>(checkpointInterval));

        // Phase 1: run only the first firstPhaseTrials trials
        deck.insert_or_assign("n_trial", static_cast<int64_t>(firstPhaseTrials));
        runEndToEnd(deck);

        // The uid utils::uniqueID() would hand out next, right where
        // phase 1 left off -- what phase 2's own restartSetup() is
        // expected to restore, via the checkpoint's own restart_uid
        // attribute, once it is clobbered below
        const auto uidAtPhase1End = utils::uniqueID().read();

        // Deliberately reset the shared counter to a value well below
        // uidAtPhase1End, mimicking what a genuinely separate
        // process's own fresh utils::uniqueID() would look like at the
        // start of phase 2 if restartSetup() did *not* restore it --
        // if restoration were broken, phase 2 would instead just keep
        // counting up from here, handing out uids that collide with
        // phase 1's own already-used ones
        utils::uniqueID().set(0);

        // Phase 2: restart, with n_trial back up to the real total
        deck.insert_or_assign("n_trial", static_cast<int64_t>(nTrial));
        runEndToEnd(deck, /* restart = */ true);

        // Pool every checkpoint's own rows together and check
        // completeness/uniqueness across both phases combined
        TrialMap rowsByTrial;
        for (unsigned long chk = 0; chk < totalCheckpoints; ++chk)
        {
            const auto h5Path = checkpointH5Path(outDir, modelName, chk);
            if (!std::filesystem::exists(h5Path))
            {
                std::cerr << "testSimCluster: restart: missing "
                    << h5Path.string() << "\n";
                return 1;
            }
            const auto cols = readOutputColumns(h5Path);
            for (std::size_t i = 0; i < cols.trial_.size(); ++i)
            { rowsByTrial[cols.trial_.at(i)].push_back(cols.uid_.at(i)); }
        }

        // No checkpoint beyond totalCheckpoints should have been created
        const auto extraPath = checkpointH5Path(outDir, modelName, totalCheckpoints);
        if (std::filesystem::exists(extraPath))
        {
            std::cerr << "testSimCluster: restart: unexpected extra "
                "checkpoint file " << extraPath.string() << "\n";
            return 1;
        }

        if (rowsByTrial.size() != nTrial)
        {
            std::cerr << "testSimCluster: restart: expected " << nTrial
                << " distinct trial numbers across every checkpoint, got "
                << rowsByTrial.size() << "\n";
            return 1;
        }
        if (const int result = checkTrialsAndUids(rowsByTrial); result != 0)
        { return result; }

        // checkTrialsAndUids() above already confirmed every trial has
        // exactly one row and every uid across both phases is unique;
        // the smallest uid among phase 2's own trials (trial >=
        // firstPhaseTrials) -- whichever one happened to run first
        // under dynamic scheduling -- should be exactly uidAtPhase1End,
        // confirming restartSetup() actually restored
        // utils::uniqueID() from the checkpoint's own restart_uid
        // attribute, rather than continuing from the clobbered value
        // set above
        unsigned long minPhase2Uid = rowsByTrial.at(firstPhaseTrials).front();
        for (unsigned long trial = firstPhaseTrials + 1; trial < nTrial; ++trial)
        { minPhase2Uid = std::min(minPhase2Uid, rowsByTrial.at(trial).front()); }
        if (minPhase2Uid != uidAtPhase1End)
        {
            std::cerr << "testSimCluster: restart: expected the smallest uid "
                "among phase 2's own trials to be " << uidAtPhase1End <<
                " (utils::uniqueID()'s own value when phase 1 ended), got " <<
                minPhase2Uid << " -- restartSetup() did not correctly "
                "restore utils::uniqueID() from the checkpoint's own "
                "restart_uid attribute\n";
            return 1;
        }

        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSimCluster: restart test failed: " << error.what() << "\n";
        return 1;
    }
}

// End-to-end check of graceful SIGTERM handling. A background thread
// sleeps briefly then raises SIGTERM against this same process, while
// the main thread is in the middle of a checkpointed SimCluster::run()
// call -- long enough after run() starts that its own SigtermGuard has
// certainly already installed its handler (that happens in
// microseconds), but short relative to localNTrial's own total
// runtime (this deck's own per-trial cost is on the order of 100+ ms,
// so localNTrial trials take at least a couple of seconds run to
// completion) that the run is nowhere near finished when SIGTERM
// actually arrives.
//
// Checks that: run() returns SimCluster::sigtermExitCode rather than
// 0; trialsCompleted() is strictly between 0 and localNTrial (some
// real progress was made, but the run genuinely stopped early rather
// than racing to completion first); the checkpoint that was open when
// SIGTERM arrived -- found by scanning upward from checkpoint 0,
// mirroring restartSetup()'s own discovery, rather than assumed --
// has trialsCompleted() itself, not localNTrial, as its own
// "trials_completed" attribute (see OutputManager::
// notifyEarlyTermination()'s own comment for why this needs checking
// at all: the destructor would otherwise wrongly assume every trial
// had completed); and that no checkpoint beyond it exists (confirming
// checkpoint() -- which would leave a new, empty, dangling one behind
// -- was correctly not called here the way it is for an ordinary,
// mid-run checkpoint rollover). Finally, restarts from there (mirroring
// testSimClusterRestartFromCheckpoint) and checks that pooling every
// checkpoint's own rows together afterward -- across both the
// SIGTERM-truncated first phase and the completed restart -- still
// yields every trial from 0 to localNTrial - 1 exactly once.
static auto testSimClusterSigtermGracefulExit() -> int
{
    const auto outDir = std::filesystem::temp_directory_path() /
        "slugTestSimClusterSigtermGracefulExit";
    std::filesystem::remove_all(outDir);
    std::filesystem::create_directories(outDir);
    const std::string modelName = "test_sim_cluster_sigterm";

    constexpr unsigned long checkpointInterval = 3;
    constexpr unsigned long localNTrial = 15;
    constexpr auto sigtermDelay = std::chrono::milliseconds(400);

    try
    {
        toml::table deck = makeInputDeck(modelName, outDir);
        deck.at_path("outputs").as_table()->insert(
            "checkpoint_interval", static_cast<int64_t>(checkpointInterval));
        deck.insert_or_assign("n_trial", static_cast<int64_t>(localNTrial));

        const io::SimControls simControls(deck);
#ifdef _OPENMP
        omp_set_num_threads(nThreads);
#endif // _OPENMP

        // Scoped so simCluster (and the OutputManagerH5 it owns)
        // destructs -- actually closing and flushing the checkpoint
        // that was open when SIGTERM arrived -- before this reads that
        // file back below; mirrors runEndToEnd()'s own identical
        // implicit scoping (its own simCluster/outputManager are local
        // to that function, destructing when it returns, before its
        // own caller reads anything back).
        int exitCode = 0;
        unsigned long completed = 0;
        {
            auto outputManager = std::make_unique<io::OutputManagerH5>(simControls, deck);
            core::SimCluster simCluster(simControls, std::move(outputManager));

            std::thread sigtermThread([&sigtermDelay]()
            {
                std::this_thread::sleep_for(sigtermDelay);
                std::raise(SIGTERM);
            });
            exitCode = simCluster.run();
            sigtermThread.join();
            completed = simCluster.trialsCompleted();
        }

        if (exitCode != core::SimCluster::sigtermExitCode)
        {
            std::cerr << "testSimCluster: sigterm: expected run() to return "
                << core::SimCluster::sigtermExitCode << ", got " << exitCode << "\n";
            return 1;
        }
        if (completed == 0 || completed >= localNTrial)
        {
            std::cerr << "testSimCluster: sigterm: expected 0 < trialsCompleted() < "
                << localNTrial << " (a real, partial early stop), got " << completed << "\n";
            return 1;
        }

        // Find the highest-numbered checkpoint that actually exists --
        // the one SIGTERM was caught in the middle of -- without
        // assuming which number that is
        unsigned long lastChk = 0;
        while (std::filesystem::exists(checkpointH5Path(outDir, modelName, lastChk + 1)))
        { ++lastChk; }
        const auto lastChkPath = checkpointH5Path(outDir, modelName, lastChk);

        // NOLINTBEGIN(misc-include-cleaner)
        const hid_t file = H5Fopen(lastChkPath.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        const auto lastChkTrialsCompleted = readULongAttr(file, "trials_completed");
        H5Fclose(file);
        // NOLINTEND(misc-include-cleaner)
        if (lastChkTrialsCompleted != completed)
        {
            std::cerr << "testSimCluster: sigterm: expected " << lastChkPath.string()
                << " to have trials_completed=" << completed << ", got "
                << lastChkTrialsCompleted << "\n";
            return 1;
        }

        const auto extraPath = checkpointH5Path(outDir, modelName, lastChk + 1);
        if (std::filesystem::exists(extraPath))
        {
            std::cerr << "testSimCluster: sigterm: unexpected extra "
                "checkpoint file " << extraPath.string() << "\n";
            return 1;
        }

        // Restart, completing the remaining trials
        deck.insert_or_assign("n_trial", static_cast<int64_t>(localNTrial));
        runEndToEnd(deck, /* restart = */ true);

        const unsigned long totalCheckpoints =
            (localNTrial + checkpointInterval - 1) / checkpointInterval;
        TrialMap rowsByTrial;
        for (unsigned long chk = 0; chk < totalCheckpoints; ++chk)
        {
            const auto h5Path = checkpointH5Path(outDir, modelName, chk);
            if (!std::filesystem::exists(h5Path))
            {
                std::cerr << "testSimCluster: sigterm: missing "
                    << h5Path.string() << "\n";
                return 1;
            }
            const auto cols = readOutputColumns(h5Path);
            for (std::size_t i = 0; i < cols.trial_.size(); ++i)
            { rowsByTrial[cols.trial_.at(i)].push_back(cols.uid_.at(i)); }
        }

        if (rowsByTrial.size() != localNTrial)
        {
            std::cerr << "testSimCluster: sigterm: expected " << localNTrial
                << " distinct trial numbers across every checkpoint after "
                "restarting, got " << rowsByTrial.size() << "\n";
            return 1;
        }
        return checkTrialsAndUids(rowsByTrial, localNTrial);
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSimCluster: sigterm test failed: " << error.what() << "\n";
        return 1;
    }
}

auto testSimCluster() -> int
{
    const auto outDir = std::filesystem::temp_directory_path() / "slugTestSimCluster";
    std::filesystem::remove_all(outDir);
    std::filesystem::create_directories(outDir);

    // Fixed [Fe/H] (constFeH() == true): every cluster references the
    // single Tracks2D/Mesh2DGrid/Interpolator1D set SimControls builds
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
    result += testSimClusterPhotH5();
    result += testSimClusterPhotAscii();
    result += testSimClusterCheckpointedH5();
    result += testSimClusterRestartFromCheckpoint();
    result += testSimClusterSigtermGracefulExit();

    return result;
}
