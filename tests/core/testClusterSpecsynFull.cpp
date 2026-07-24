/**
 * @file testClusterSpecsynFull.cpp
 * @author Mark Krumholz
 * @brief Optional full end-to-end test of a cluster simulation with real tracks/spectra.
 * @details
 * This is the first full-scale, end-to-end test of the code including
 * spectral synthesis, using the complete MIST tracks and the complete
 * POWR_WC/POWR_WNE/POWR_WNL/TLUSTY_O/TLUSTY_B/BOSZ/CK04 spectral
 * libraries -- as opposed to every other spectral-synthesis test in
 * this repository, which uses small synthetic or trimmed-down
 * fixtures. Its purpose is to turn up any gaps in coverage between
 * the tracks and the spectral libraries: for example, a star whose
 * (Teff, logg) or (Teff, transformed radius) the tracks produce but
 * none of the spectral libraries cover would surface here as an
 * out-of-bounds error from the final (OOBPolicy::raise) library in
 * the chain. CK04 is listed last, after BOSZ, since it exists
 * specifically to catch stars in one of BOSZ's own gaps (BOSZ has
 * essentially no models for Teff >~ 8000 K combined with log g
 * <~ 1.5) that this test itself found.
 *
 * The data files this test needs are too large to store in the
 * repository (see .gitignore's data/tracks and data/spectra
 * exclusions), so, mirroring tests/tracks/testTracks2D.hpp's own
 * optional-file pattern, this test runs only if every one of them is
 * present locally (i.e. has been fetched separately via
 * data/tools/fetch_mist.py, fetch_powr.py, fetch_tlusty.py,
 * fetch_bosz.py, and fetch_ck04.py); otherwise it is skipped,
 * returning an automatic pass rather than a failure.
 * @date 2026-07-24
 */

#include "../src/io/OutputManager.hpp"
#include "../src/io/OutputManagerH5.hpp"
#include "../src/io/SimControls.hpp"
#include "../src/io/SimPhysics.hpp"
#include "../src/core/SimCluster.hpp"
#include "hdf5.h" // NOLINT(misc-include-cleaner)
#include "testClusterSpecsynFull.hpp"
#include <array>
#include <cmath>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <toml.hpp>
#include <vector>

namespace
{
    // Every file this test needs; both the *.toml registries and the
    // *.h5 data files themselves are gitignored (too large, in the
    // h5 case, to store in the repository), so any of them may be
    // absent depending on whether this machine has fetched them
    const std::array<std::string, 10> requiredDataFiles = {{
        "data/tracks/tracks.toml",
        "data/tracks/mist.h5",
        "data/spectra/spectra.toml",
        "data/spectra/powr_wc.h5",
        "data/spectra/powr_wne.h5",
        "data/spectra/powr_wnl.h5",
        "data/spectra/tlusty_o.h5",
        "data/spectra/tlusty_b.h5",
        "data/spectra/bosz.h5",
        "data/spectra/ck04.h5",
    }};

    // Check that every one of requiredDataFiles is present, printing a
    // message identifying the first missing one if not
    auto allRequiredDataFilesExist() -> bool
    {
        for (const auto& path : requiredDataFiles)
        {
            if (!std::filesystem::exists(path))
            {
                std::cerr << "testClusterSpecsynFull: required data file "
                    << path << " not found; skipping this optional "
                    "full-data end-to-end test\n";
                return false;
            }
        }
        return true;
    }

    // Read an entire 1d extensible dataset of the given HDF5 native
    // type into a vector of the corresponding C++ type -- mirrors
    // testSimCluster.cpp's own identical helper
    template <typename T>
    auto readDataset(const hid_t group, const char* name, const hid_t memType) // NOLINT(misc-include-cleaner)
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

    // Read an entire 2d dataset (row-major) of the given HDF5 native
    // type into a flat vector, along with its (rows, cols) shape
    template <typename T>
    auto readDataset2D(const hid_t group, const char* name, const hid_t memType) // NOLINT(misc-include-cleaner)
        -> std::pair<std::vector<T>, std::pair<hsize_t, hsize_t>>
    {
        // NOLINTBEGIN(misc-include-cleaner)
        const hid_t dset = H5Dopen2(group, name, H5P_DEFAULT);
        const hid_t space = H5Dget_space(dset);
        std::array<hsize_t, 2> dims{};
        H5Sget_simple_extent_dims(space, dims.data(), nullptr);
        std::vector<T> result(static_cast<size_t>(dims.at(0) * dims.at(1)));
        H5Dread(dset, memType, H5S_ALL, H5S_ALL, H5P_DEFAULT, result.data());
        H5Sclose(space);
        H5Dclose(dset);
        // NOLINTEND(misc-include-cleaner)
        return { std::move(result), { dims.at(0), dims.at(1) } };
    }
} // namespace

// Run the full simulation described by
// tests/core/assets/testClusterSpecsynFull.in (MIST tracks; [Fe/H]
// pinned to a delta function at exactly 0.0 for now -- see that
// file's own comments for why -- rather than the flat [-1, 0]
// distribution this test is meant to eventually use; alphaFe = 0,
// v/vcrit = 0.4; a spectra.model chain of POWR_WC, POWR_WNE, POWR_WNL,
// TLUSTY_O, TLUSTY_B, BOSZ, and CK04; a CMF fixed at 10^4 Msun; output at t = 2, 3, 4,
// and 10 Myr, spanning stellar evolution before, during, and after the
// Wolf-Rayet phase; 10 trials) and check that the resulting HDF5
// output has the expected shape and contains only finite, non-trivial
// spectra. There is no independently-computed expected spectrum to
// check against here -- this test's job is to confirm the full
// pipeline runs to completion without an out-of-bounds or other
// error, which is exactly what would happen if the tracks ever
// produced a star outside every spectral library's coverage.
auto testClusterSpecsynFull() -> int
{
    if (!allRequiredDataFilesExist()) { return 0; }

    constexpr unsigned long nTrial = 10;
    constexpr std::size_t nTime = 4; // output_times has 4 entries

    const auto outDir = std::filesystem::temp_directory_path() / "slugTestClusterSpecsynFull";
    std::filesystem::remove_all(outDir);
    std::filesystem::create_directories(outDir);
    const std::string modelName = "test_cluster_specsyn_full";
    const auto h5Path = outDir / (modelName + ".h5");

    try
    {
        toml::table inputDeck = toml::parse_file("tests/core/assets/testClusterSpecsynFull.in");
        inputDeck.insert("output", toml::table{ { "model_name", modelName } });
        inputDeck.at_path("outputs").as_table()->insert("out_dir", outDir.string());

        const io::SimControls simControls(inputDeck);
        const io::SimPhysics simPhysics(inputDeck, simControls.simType());

        std::unique_ptr<io::OutputManager> outputManager =
            std::make_unique<io::OutputManagerH5>(simControls, simPhysics, inputDeck);

        core::SimCluster simCluster(simControls, simPhysics, std::move(outputManager));
        simCluster.run();

        // NOLINTBEGIN(misc-include-cleaner)
        const hid_t file = H5Fopen(h5Path.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        if (file < 0)
        {
            std::cerr << "testClusterSpecsynFull: unable to reopen "
                << h5Path.string() << "\n";
            return 1;
        }

        // Every trial should have written exactly one row of
        // time-invariant cluster properties
        const hid_t clustersGrp = H5Gopen2(file, "clusters", H5P_DEFAULT);
        const auto trialCol = readDataset<unsigned long>(clustersGrp, "trial", H5T_NATIVE_ULONG);
        H5Gclose(clustersGrp);

        if (trialCol.size() != nTrial)
        {
            std::cerr << "testClusterSpecsynFull: expected " << nTrial
                << " rows in clusters/trial, got " << trialCol.size() << "\n";
            H5Fclose(file);
            return 1;
        }
        const std::set<unsigned long> distinctTrials(trialCol.begin(), trialCol.end());
        if (distinctTrials.size() != nTrial)
        {
            std::cerr << "testClusterSpecsynFull: expected " << nTrial
                << " distinct trial numbers in clusters/trial, got "
                << distinctTrials.size() << "\n";
            H5Fclose(file);
            return 1;
        }

        // Every trial should have written one spectrum per output time
        const hid_t specGrp = H5Gopen2(file, "cluster_spectra", H5P_DEFAULT);
        const auto specTrialCol = readDataset<unsigned long>(specGrp, "trial", H5T_NATIVE_ULONG);
        const auto wl = readDataset<double>(specGrp, "wl", H5T_NATIVE_DOUBLE);
        auto [spec, specShape] = readDataset2D<double>(specGrp, "spec", H5T_NATIVE_DOUBLE);
        H5Gclose(specGrp);
        H5Fclose(file);
        // NOLINTEND(misc-include-cleaner)

        const auto expectedRows = static_cast<hsize_t>(nTrial) * static_cast<hsize_t>(nTime);
        if (specTrialCol.size() != expectedRows)
        {
            std::cerr << "testClusterSpecsynFull: expected " << expectedRows
                << " rows in cluster_spectra/trial, got " << specTrialCol.size() << "\n";
            return 1;
        }
        if (specShape.first != expectedRows || specShape.second != wl.size())
        {
            std::cerr << "testClusterSpecsynFull: expected a ("
                << expectedRows << ", " << wl.size()
                << ") spec dataset, got (" << specShape.first << ", "
                << specShape.second << ")\n";
            return 1;
        }
        if (wl.empty())
        {
            std::cerr << "testClusterSpecsynFull: wl dataset is empty\n";
            return 1;
        }

        // Every row's spectrum should be finite everywhere, and have
        // some non-trivial (positive) flux somewhere
        const auto nWl = wl.size();
        for (hsize_t row = 0; row < specShape.first; ++row)
        {
            bool anyPositive = false;
            for (size_t col = 0; col < nWl; ++col)
            {
                const double value = spec.at((static_cast<size_t>(row) * nWl) + col);
                if (!std::isfinite(value))
                {
                    std::cerr << "testClusterSpecsynFull: spec row " << row
                        << ", column " << col << " is not finite (" << value << ")\n";
                    return 1;
                }
                anyPositive = anyPositive || (value > 0.0);
            }
            if (!anyPositive)
            {
                std::cerr << "testClusterSpecsynFull: spec row " << row
                    << " has no positive flux anywhere\n";
                return 1;
            }
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testClusterSpecsynFull: end-to-end run failed: "
            << error.what() << "\n";
        return 1;
    }

    return 0;
}
