/**
 * @file testGalaxySpecsynFullCommon.cpp
 * @author Mark Krumholz
 * @brief Implementation of testGalaxySpecsynFullCommon.hpp
 * @date 2026-08-14
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#include "testGalaxySpecsynFullCommon.hpp"
#include "../src/core/SimGalaxy.hpp"
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
#include <string>
#include <toml.hpp>
#include <utility>
#include <vector>

namespace
{
    // Read an entire 1d extensible dataset of the given HDF5 native
    // type into a vector of the corresponding C++ type -- mirrors
    // testClusterSpecsynFullCommon.cpp's own identical helper
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
    // type into a flat vector, along with its (rows, cols) shape --
    // mirrors testClusterSpecsynFullCommon.cpp's own identical helper
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

    // Check that every row of the galaxy_spectra "spec" dataset is
    // finite everywhere, and has some non-trivial (positive) flux
    // somewhere -- mirrors testClusterSpecsynFullCommon.cpp's own
    // validateClusterSpectra(), factored out here for the same reason
    // (keeping runGalaxySpecsynFull's own cognitive complexity down)
    auto validateGalaxySpectra(
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
                    std::cerr << "testGalaxySpecsynFull: spec row " << row
                        << ", column " << col << " is not finite (" << value << ")\n";
                    return 1;
                }
                anyPositive = anyPositive || (value > 0.0);
            }
            if (!anyPositive)
            {
                std::cerr << "testGalaxySpecsynFull: spec row " << row
                    << " has no positive flux anywhere\n";
                return 1;
            }
        }
        return 0;
    }
} // namespace

auto runGalaxySpecsynFull(const std::string& inputFile, const std::string& modelName) -> int
{
    const auto outDir = std::filesystem::temp_directory_path() / "slugTestGalaxySpecsynFull";
    std::filesystem::remove_all(outDir);
    std::filesystem::create_directories(outDir);
    const auto h5Path = outDir / (modelName + ".h5");

    try
    {
        toml::table inputDeck = toml::parse_file(inputFile);
        if (toml::table* outputTbl = inputDeck["output"].as_table())
        { outputTbl->insert("model_name", modelName); }
        else { inputDeck.insert("output", toml::table{ { "model_name", modelName } }); }
        if (toml::table* outputsTbl = inputDeck["outputs"].as_table())
        { outputsTbl->insert("out_dir", outDir.string()); }
        else { inputDeck.insert("outputs", toml::table{ { "out_dir", outDir.string() } }); }

        const io::SimControls simControls(inputDeck);
        const auto nTrial = simControls.nTrial();
        const auto nTime = simControls.outTimes().size();

        std::unique_ptr<io::OutputManager> outputManager =
            std::make_unique<io::OutputManagerH5>(simControls, inputDeck);

        core::SimGalaxy simGalaxy(simControls, std::move(outputManager));
        simGalaxy.run();

        // NOLINTBEGIN(misc-include-cleaner)
        const hid_t file = H5Fopen(h5Path.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        if (file < 0)
        {
            std::cerr << "testGalaxySpecsynFull: unable to reopen "
                << h5Path.string() << "\n";
            return 1;
        }

        const hid_t specGrp = H5Gopen2(file, "galaxy_spectra", H5P_DEFAULT);
        const auto specTrialCol = readDataset<unsigned long>(specGrp, "trial", H5T_NATIVE_ULONG);
        const auto wl = readDataset<double>(specGrp, "wl", H5T_NATIVE_DOUBLE);
        auto [spec, specShape] = readDataset2D<double>(specGrp, "spec", H5T_NATIVE_DOUBLE);
        H5Gclose(specGrp);

        H5Fclose(file);
        // NOLINTEND(misc-include-cleaner)

        const auto expectedRows = static_cast<hsize_t>(nTrial) * static_cast<hsize_t>(nTime);
        if (specTrialCol.size() != expectedRows)
        {
            std::cerr << "testGalaxySpecsynFull: expected " << expectedRows
                << " rows in galaxy_spectra/trial, got " << specTrialCol.size() << "\n";
            return 1;
        }
        if (specShape.first != expectedRows || specShape.second != wl.size())
        {
            std::cerr << "testGalaxySpecsynFull: expected a ("
                << expectedRows << ", " << wl.size()
                << ") spec dataset, got (" << specShape.first << ", "
                << specShape.second << ")\n";
            return 1;
        }
        if (wl.empty())
        {
            std::cerr << "testGalaxySpecsynFull: wl dataset is empty\n";
            return 1;
        }

        if (validateGalaxySpectra(spec, specShape.first, wl.size()) != 0)
        {
            return 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testGalaxySpecsynFull: end-to-end run failed: "
            << error.what() << "\n";
        return 1;
    }

    return 0;
}
