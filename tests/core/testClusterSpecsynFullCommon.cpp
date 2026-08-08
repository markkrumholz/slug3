/**
 * @file testClusterSpecsynFullCommon.cpp
 * @author Mark Krumholz
 * @brief Implementation of testClusterSpecsynFullCommon.hpp
 * @date 2026-07-30
 */

#include "testClusterSpecsynFullCommon.hpp"
#include "../src/core/SimCluster.hpp"
#include "../src/io/OutputManager.hpp"
#include "../src/io/OutputManagerH5.hpp"
#include "../src/io/SimControls.hpp"
#include "hdf5.h" // NOLINT(misc-include-cleaner)
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
#include <utility>
#include <vector>

namespace
{
    // Every file the full-scale cluster tests need; both the *.toml
    // registries and the *.h5 data files themselves are gitignored
    // (too large, in the h5 case, to store in the repository), so any
    // of them may be absent depending on whether this machine has
    // fetched them
    const std::array<std::string, 16> requiredDataFiles = { { // NOLINT(bugprone-throwing-static-initialization,cert-err58-cpp) -- built from fixed string literals, so allocation failure aside (which would abort regardless), this can't actually throw
        "data/tracks/tracks.toml",
        "data/tracks/mist.h5",
        "data/spectra/spectra.toml",
        "data/spectra/powr_wc.h5",
        "data/spectra/powr_wne.h5",
        "data/spectra/powr_wnl_h20.h5",
        "data/spectra/powr_wnl_h40.h5",
        "data/spectra/powr_wnl_h60.h5",
        "data/spectra/tlusty_o.h5",
        "data/spectra/tlusty_b.h5",
        "data/spectra/bosz.h5",
        "data/spectra/ck04.h5",
        "data/spectra/tremblay_da.h5",
        "data/spectra/tremblay_elm.h5",
        "data/spectra/rauch.h5",
        "data/spectra/rauch_h07.h5",
    }};

    // The real (gitignored) filter registry and Vega reference
    // spectrum testClusterSpecsynFull's own [phot] section needs --
    // see allPhotDataFilesExist()'s own comment for why this is kept
    // separate from requiredDataFiles above
    const std::array<std::string, 3> requiredPhotDataFiles = { { // NOLINT(bugprone-throwing-static-initialization,cert-err58-cpp) -- see requiredDataFiles above
        "data/filters/filters.toml",
        "data/filters/filters.h5",
        "data/spectra/vega.h5",
    }};

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

    // Read the cluster_phot group's "phot" dataset, if that group
    // exists in the (already-open) HDF5 output file file -- i.e. if
    // the deck that produced it had a [phot] section -- or an empty
    // result (with shape (0, 0)) otherwise
    auto readClusterPhotIfPresent(const hid_t file) // NOLINT(misc-include-cleaner)
        -> std::pair<std::vector<double>, std::pair<hsize_t, hsize_t>>
    {
        // NOLINTBEGIN(misc-include-cleaner)
        if (H5Lexists(file, "cluster_phot", H5P_DEFAULT) <= 0) { return {}; }
        const hid_t photGrp = H5Gopen2(file, "cluster_phot", H5P_DEFAULT);
        auto result = readDataset2D<double>(photGrp, "phot", H5T_NATIVE_DOUBLE);
        H5Gclose(photGrp);
        // NOLINTEND(misc-include-cleaner)
        return result;
    }

    // Check that every row of the cluster_spectra "spec" dataset is
    // finite everywhere, and has some non-trivial (positive) flux
    // somewhere. Factored out of runClusterSpecsynFull to keep its
    // own cognitive complexity down.
    auto validateClusterSpectra(
        const std::vector<double>& spec, const hsize_t nRows, const size_t nWl) -> int
    {
        for (hsize_t row = 0; row < nRows; ++row)
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
        return 0;
    }

    // Check that the cluster_phot dataset (if present at all -- see
    // readClusterPhotIfPresent above) has the expected number of rows
    // and contains no NaNs or infinities (e.g. from a Vega magnitude
    // of a filter with genuinely zero flux). Factored out of
    // runClusterSpecsynFull to keep its own cognitive complexity down.
    auto validateClusterPhot(const bool hasPhot,
        const std::vector<double>& phot,
        const std::pair<hsize_t, hsize_t>& photShape,
        const hsize_t expectedRows) -> int
    {
        if (!hasPhot) { return 0; }

        if (photShape.first != expectedRows)
        {
            std::cerr << "testClusterSpecsynFull: expected " << expectedRows
                << " rows in cluster_phot/phot, got " << photShape.first << "\n";
            return 1;
        }
        for (const double value : phot)
        {
            if (!std::isfinite(value))
            {
                std::cerr << "testClusterSpecsynFull: cluster_phot/phot "
                    "contains a non-finite value (" << value << ")\n";
                return 1;
            }
        }
        return 0;
    }

    // Check that the clusters group's trial column has exactly
    // nTrial rows, one per distinct trial number. Factored out of
    // runClusterSpecsynFull to keep its own cognitive complexity down.
    auto validateClustersTrialColumn(
        const std::vector<unsigned long>& trialCol, const unsigned long nTrial) -> int
    {
        if (trialCol.size() != nTrial)
        {
            std::cerr << "testClusterSpecsynFull: expected " << nTrial
                << " rows in clusters/trial, got " << trialCol.size() << "\n";
            return 1;
        }
        const std::set<unsigned long> distinctTrials(trialCol.begin(), trialCol.end());
        if (distinctTrials.size() != nTrial)
        {
            std::cerr << "testClusterSpecsynFull: expected " << nTrial
                << " distinct trial numbers in clusters/trial, got "
                << distinctTrials.size() << "\n";
            return 1;
        }
        return 0;
    }
} // namespace

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

auto allPhotDataFilesExist() -> bool
{
    for (const auto& path : requiredPhotDataFiles)
    {
        if (!std::filesystem::exists(path))
        {
            std::cerr << "testClusterSpecsynFull: required photometry data file "
                << path << " not found; skipping this optional "
                "full-data end-to-end test\n";
            return false;
        }
    }
    return true;
}

auto runClusterSpecsynFull(const std::string& inputFile, const std::string& modelName) -> int
{
    const auto outDir = std::filesystem::temp_directory_path() / "slugTestClusterSpecsynFull";
    std::filesystem::remove_all(outDir);
    std::filesystem::create_directories(outDir);
    const auto h5Path = outDir / (modelName + ".h5");

    try
    {
        toml::table inputDeck = toml::parse_file(inputFile);
        inputDeck.insert("output", toml::table{ { "model_name", modelName } });
        inputDeck.at_path("outputs").as_table()->insert("out_dir", outDir.string());

        const io::SimControls simControls(inputDeck);
        const auto nTrial = simControls.nTrial(); // read back rather than assumed, since not every caller's deck uses the same n_trial
        const auto nTime = simControls.outTimes().size(); // read back rather than assumed, since not every caller's deck uses the same number of output_times

        std::unique_ptr<io::OutputManager> outputManager =
            std::make_unique<io::OutputManagerH5>(simControls, inputDeck);

        core::SimCluster simCluster(simControls, std::move(outputManager));
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

        if (validateClustersTrialColumn(trialCol, nTrial) != 0)
        {
            H5Fclose(file);
            return 1;
        }

        // Every trial should have written one spectrum per output time
        const hid_t specGrp = H5Gopen2(file, "cluster_spectra", H5P_DEFAULT);
        const auto specTrialCol = readDataset<unsigned long>(specGrp, "trial", H5T_NATIVE_ULONG);
        const auto wl = readDataset<double>(specGrp, "wl", H5T_NATIVE_DOUBLE);
        auto [spec, specShape] = readDataset2D<double>(specGrp, "spec", H5T_NATIVE_DOUBLE);
        H5Gclose(specGrp);

        // Only present if the deck had a [phot] section -- see this
        // function's own comment
        auto [phot, photShape] = readClusterPhotIfPresent(file);
        const bool hasPhot = photShape.first != 0 || photShape.second != 0;

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
        if (validateClusterSpectra(spec, specShape.first, wl.size()) != 0)
        {
            return 1;
        }

        // If the deck also requested photometry, check that every
        // value written is finite. As with the spectrum above, there
        // is no independently-computed expected magnitude/flux to
        // check here, so this just confirms the full filter/
        // photometric-conversion pipeline runs to completion without
        // producing a NaN or infinity.
        if (validateClusterPhot(hasPhot, phot, photShape, expectedRows) != 0)
        {
            return 1;
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
