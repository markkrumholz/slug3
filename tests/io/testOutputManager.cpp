/**
 * @file testOutputManager.cpp
 * @author Mark Krumholz
 * @brief Unit tests for the OutputManager class.
 * @date 2026-07-16
 */

#include "../src/io/OutputManager.hpp"
#include "../src/io/SimControls.hpp"
#include "hdf5.h" // NOLINT(misc-include-cleaner)
#include "io/SlugVersion.hpp"
#include "testOutputManager.hpp"
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <toml.hpp>

// Build a minimal, valid input deck for a cluster-type simulation
// with model_name and out_dir set as given, and a single output
// time at t = 0 (the output-time details are irrelevant here;
// OutputManager itself never looks at them).
static auto makeInputDeck(const std::string& modelName,
    const std::filesystem::path& outDir) -> toml::table
{
    const std::string deckText =
        "sim_type = \"cluster\"\n"
        "\n"
        "[output]\n"
        "model_name = \"" + modelName + "\"\n"
        "\n"
        "[outputs]\n"
        "start_time = 0.0\n"
        "end_time = 0.0\n"
        "ntime = 1\n"
        "out_dir = \"" + outDir.string() + "\"\n";
    return toml::parse(deckText);
}

// Verify the ascii OutputManager: opens <model>_summary.txt, writes the
// slug-hash/date/time header followed by the toml input deck, and
// refuses to overwrite an existing file.
static auto testOutputManagerAscii() -> int
{
    const auto outDir = std::filesystem::temp_directory_path() / "slugTestOutputManagerAscii";
    std::filesystem::remove_all(outDir);
    std::filesystem::create_directories(outDir);
    const std::string modelName = "test_model";
    const auto expectedPath = outDir / (modelName + "_summary.txt");
    const toml::table inputDeck = makeInputDeck(modelName, outDir);
    const io::SimControls controls(inputDeck);

    try
    {
        const io::OutputManager<io::SimControls::OutputMode::ascii>
            manager(controls, inputDeck);

        if (!std::filesystem::exists(expectedPath))
        {
            std::cerr << "testOutputManager: ascii: expected output file "
                << expectedPath.string() << " to exist\n";
            return 1;
        }

        const std::ifstream file(expectedPath);
        std::ostringstream contentsStream;
        contentsStream << file.rdbuf();
        const std::string contents = contentsStream.str();

        if (!contents.contains(std::string("slug-hash  ") + io::slugGitHash))
        {
            std::cerr << "testOutputManager: ascii: missing expected slug-hash line\n";
            return 1;
        }
        if (!contents.contains("date  ") || !contents.contains("time  "))
        {
            std::cerr << "testOutputManager: ascii: missing expected date/time lines\n";
            return 1;
        }
        if (!contents.contains("input_deck") || !contents.contains("sim_type"))
        {
            std::cerr << "testOutputManager: ascii: missing dumped input deck\n";
            return 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testOutputManager: ascii test failed: "
            << error.what() << "\n";
        return 1;
    }

    // A second manager pointed at the same (now-existing) file should
    // refuse to overwrite it.
    try
    {
        const io::OutputManager<io::SimControls::OutputMode::ascii>
            manager2(controls, inputDeck);
        std::cerr << "testOutputManager: ascii: expected construction to "
            "throw on an existing output file, but it succeeded\n";
        return 1;
    }
    catch (const std::runtime_error&)
    {
        return 0;
    }
}

// Verify the HDF5 OutputManager: opens <model>.h5, writes the
// slug-hash/date/time header as top-level attributes, dumps the toml
// input deck into an input_deck group, and refuses to overwrite an
// existing file.
static auto testOutputManagerH5() -> int
{
    const auto outDir = std::filesystem::temp_directory_path() / "slugTestOutputManagerH5";
    std::filesystem::remove_all(outDir);
    std::filesystem::create_directories(outDir);
    const std::string modelName = "test_model";
    const auto expectedPath = outDir / (modelName + ".h5");
    const toml::table inputDeck = makeInputDeck(modelName, outDir);
    const io::SimControls controls(inputDeck);

    try
    {
        {
            const io::OutputManager<io::SimControls::OutputMode::h5>
                manager(controls, inputDeck);
        }

        if (!std::filesystem::exists(expectedPath))
        {
            std::cerr << "testOutputManager: h5: expected output file "
                << expectedPath.string() << " to exist\n";
            return 1;
        }

        // NOLINTBEGIN(misc-include-cleaner)
        const hid_t file = H5Fopen(expectedPath.string().c_str(),
            H5F_ACC_RDONLY, H5P_DEFAULT);
        if (file < 0)
        {
            std::cerr << "testOutputManager: h5: unable to reopen "
                << expectedPath.string() << "\n";
            return 1;
        }
        if (H5Aexists(file, "slug-hash") <= 0 ||
            H5Aexists(file, "date") <= 0 ||
            H5Aexists(file, "time") <= 0)
        {
            std::cerr << "testOutputManager: h5: missing expected top-level "
                "attributes\n";
            H5Fclose(file);
            return 1;
        }
        if (H5Lexists(file, "input_deck", H5P_DEFAULT) <= 0)
        {
            std::cerr << "testOutputManager: h5: missing input_deck group\n";
            H5Fclose(file);
            return 1;
        }
        const hid_t grp = H5Gopen2(file, "input_deck", H5P_DEFAULT);
        const bool hasToml = H5Lexists(grp, "toml", H5P_DEFAULT) > 0;
        H5Gclose(grp);
        H5Fclose(file);
        // NOLINTEND(misc-include-cleaner)

        if (!hasToml)
        {
            std::cerr << "testOutputManager: h5: missing toml dataset in "
                "input_deck group\n";
            return 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testOutputManager: h5 test failed: "
            << error.what() << "\n";
        return 1;
    }

    // A second manager pointed at the same (now-existing) file should
    // refuse to overwrite it.
    try
    {
        const io::OutputManager<io::SimControls::OutputMode::h5>
            manager2(controls, inputDeck);
        std::cerr << "testOutputManager: h5: expected construction to "
            "throw on an existing output file, but it succeeded\n";
        return 1;
    }
    catch (const std::runtime_error&)
    {
        return 0;
    }
}

auto testOutputManager() -> int
{
    int result = 0;
    result += testOutputManagerAscii();
    result += testOutputManagerH5();
    return result;
}
