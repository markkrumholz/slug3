/**
 * @file testOutputManager.cpp
 * @author Mark Krumholz
 * @brief Unit tests for the OutputManager class.
 * @date 2026-07-16
 */

#include "../src/core/Cluster.hpp"
#include "../src/io/OutputManager.hpp"
#include "../src/io/SimControls.hpp"
#include "../src/io/SimPhysics.hpp"
#include "../src/utils/RngThread.hpp"
#include "hdf5.h" // NOLINT(misc-include-cleaner)
#include "io/SlugVersion.hpp"
#include "testOutputManager.hpp"
#include <cmath>
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

// Build a valid input deck for a cluster-type simulation that also
// has usable stellar physics (IMF, tracks, CMF, etc.), by reusing
// the deck already exercised by testSimPhysics/testCluster, and
// injecting model_name and out_dir into it as writeCluster's tests
// need an OutputManager pointed at a real Cluster.
static auto makeClusterPhysicsInputDeck(const std::string& modelName,
    const std::filesystem::path& outDir) -> toml::table
{
    toml::table inputDeck = toml::parse_file("tests/core/assets/testCluster.in");
    inputDeck.insert("output", toml::table{ { "model_name", modelName } });
    inputDeck.at_path("outputs").as_table()->insert("out_dir", outDir.string());
    return inputDeck;
}

// Verify that OutputManager::writeCluster writes a fixed-width row
// containing the cluster's trial number, uid, target mass, birth
// mass, formation time, and [Fe/H] to the ascii cluster output file.
static auto testWriteClusterAscii() -> int
{
    const auto outDir = std::filesystem::temp_directory_path() / "slugTestOutputManagerWriteClusterAscii";
    std::filesystem::remove_all(outDir);
    std::filesystem::create_directories(outDir);
    const std::string modelName = "test_model";
    const auto expectedPath = outDir / (modelName + "_clusters.txt");
    const toml::table inputDeck = makeClusterPhysicsInputDeck(modelName, outDir);

    try
    {
        const io::SimControls controls(inputDeck);
        const io::SimPhysics sim(inputDeck, controls.simType());
        utils::rng().seed(42);
        constexpr unsigned long uid = 7;
        constexpr double targetMass = 1e3;
        constexpr double formTime = 0.0;
        const core::Cluster cluster(uid, targetMass, formTime, sim);
        constexpr unsigned long trial = 3;

        {
            io::OutputManager<io::SimControls::OutputMode::ascii>
                manager(controls, inputDeck);
            manager.writeCluster(trial, cluster);
        }

        std::ifstream file(expectedPath);
        std::string headerLine;
        std::string ruleLine;
        std::string dataLine;
        std::getline(file, headerLine);
        std::getline(file, ruleLine);
        std::getline(file, dataLine);

        std::istringstream lineStream(dataLine);
        unsigned long readTrial = 0;
        unsigned long readUid = 0;
        double readTargetMass = 0.0;
        double readBirthMass = 0.0;
        double readFormTime = 0.0;
        double readFeH = 0.0;
        lineStream >> readTrial >> readUid >> readTargetMass >> readBirthMass
            >> readFormTime >> readFeH;

        constexpr double tol = 1e-5;
        if (readTrial != trial || readUid != cluster.uid() ||
            std::abs(readTargetMass - cluster.targetMass()) > tol * cluster.targetMass() ||
            std::abs(readBirthMass - cluster.birthMass()) > tol * cluster.birthMass() ||
            std::abs(readFormTime - cluster.formTime()) > tol ||
            std::abs(readFeH - cluster.feH()) > tol)
        {
            std::cerr << "testOutputManager: ascii: writeCluster produced "
                "unexpected row: " << dataLine << "\n";
            return 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testOutputManager: ascii writeCluster test failed: "
            << error.what() << "\n";
        return 1;
    }
    return 0;
}

// Verify that OutputManager::writeCluster appends the cluster's
// trial number, uid, target mass, birth mass, formation time, and
// [Fe/H] to the corresponding datasets in the HDF5 clusters group.
static auto testWriteClusterH5() -> int
{
    const auto outDir = std::filesystem::temp_directory_path() / "slugTestOutputManagerWriteClusterH5";
    std::filesystem::remove_all(outDir);
    std::filesystem::create_directories(outDir);
    const std::string modelName = "test_model";
    const auto expectedPath = outDir / (modelName + ".h5");
    const toml::table inputDeck = makeClusterPhysicsInputDeck(modelName, outDir);

    try
    {
        const io::SimControls controls(inputDeck);
        const io::SimPhysics sim(inputDeck, controls.simType());
        utils::rng().seed(42);
        constexpr unsigned long uid = 11;
        constexpr double targetMass = 2e3;
        constexpr double formTime = 0.0;
        const core::Cluster cluster(uid, targetMass, formTime, sim);
        constexpr unsigned long trial = 5;

        {
            const io::OutputManager<io::SimControls::OutputMode::h5>
                manager(controls, inputDeck);
            manager.writeCluster(trial, cluster);
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
        const hid_t grp = H5Gopen2(file, "clusters", H5P_DEFAULT);

        unsigned long readTrial = 0;
        unsigned long readUid = 0;
        double readTargetMass = 0.0;
        double readBirthMass = 0.0;
        double readFormTime = 0.0;
        double readFeH = 0.0;

        const auto readScalar = [&grp](const char* name, const hid_t memType, void* dest) -> void
        {
            const hid_t dset = H5Dopen2(grp, name, H5P_DEFAULT);
            H5Dread(dset, memType, H5S_ALL, H5S_ALL, H5P_DEFAULT, dest);
            H5Dclose(dset);
        };

        readScalar("trial", H5T_NATIVE_ULONG, &readTrial);
        readScalar("uid", H5T_NATIVE_ULONG, &readUid);
        readScalar("target_mass", H5T_NATIVE_DOUBLE, &readTargetMass);
        readScalar("birth_mass", H5T_NATIVE_DOUBLE, &readBirthMass);
        readScalar("form_time", H5T_NATIVE_DOUBLE, &readFormTime);
        readScalar("feh", H5T_NATIVE_DOUBLE, &readFeH);

        H5Gclose(grp);
        H5Fclose(file);
        // NOLINTEND(misc-include-cleaner)

        if (readTrial != trial || readUid != cluster.uid() ||
            readTargetMass != cluster.targetMass() ||
            readBirthMass != cluster.birthMass() ||
            readFormTime != cluster.formTime() ||
            readFeH != cluster.feH())
        {
            std::cerr << "testOutputManager: h5: writeCluster produced "
                "unexpected values\n";
            return 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testOutputManager: h5 writeCluster test failed: "
            << error.what() << "\n";
        return 1;
    }
    return 0;
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
    result += testWriteClusterAscii();
    result += testWriteClusterH5();
    return result;
}
