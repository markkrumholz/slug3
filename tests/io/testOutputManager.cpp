/**
 * @file testOutputManager.cpp
 * @author Mark Krumholz
 * @brief Unit tests for the OutputManager class.
 * @date 2026-07-16
 */

#include "../src/core/Cluster.hpp"
#include "../src/io/OutputManagerAscii.hpp"
#include "../src/io/OutputManagerH5.hpp"
#include "../src/io/SimControls.hpp"
#include "../src/utils/RngThread.hpp"
#include "hdf5.h" // NOLINT(misc-include-cleaner)
#include "io/SlugVersion.hpp"
#include "testOutputManager.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <toml.hpp>
#include <vector>

// Build a valid input deck for a cluster-type simulation that also
// has usable stellar physics (IMF, tracks, CMF, etc.), by reusing an
// existing deck (testCluster.in by default, but any deck of the same
// shape can be substituted), and injecting model_name and out_dir
// into it as writeCluster's tests need an OutputManager pointed at a
// real Cluster.
static auto makeClusterPhysicsInputDeck(const std::string& modelName,
    const std::filesystem::path& outDir,
    const std::string& deckPath = "tests/core/assets/testCluster.in") -> toml::table
{
    toml::table inputDeck = toml::parse_file(deckPath);
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
        utils::rng().seed(42);
        constexpr unsigned long uid = 7;
        constexpr double targetMass = 1e3;
        constexpr double formTime = 0.0;
        const core::Cluster cluster(uid, targetMass, formTime, controls);
        constexpr unsigned long trial = 3;

        {
            io::OutputManagerAscii
                manager(controls, inputDeck);
            manager.writeCluster(trial, cluster);
        }

        std::ifstream file(expectedPath);
        std::string headerLine;
        std::string unitsLine;
        std::string ruleLine;
        std::string dataLine;
        std::getline(file, headerLine);
        std::getline(file, unitsLine);
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

// Verify that OutputManagerAscii writes extinction columns to all
// three ascii output files when SimControls::extinct() is set: an
// a_v column (mag) in the cluster file, a spec_ex column (same units
// as spec) in the cluster-spectra file -- reading 0 outside
// extinct()->wlOffset()'s own coverage and cluster.specExtinct()
// inside it -- and one "<filter>_ex" column per real filter (Lbol
// excluded) in the cluster-photometry file. Uses testClusterExtinct.in,
// the same deck testCluster.cpp's own testClusterExtinct() and
// testOutputManager.cpp's own testWriteClusterSpecPhotH5Extinct()
// exercise the underlying Cluster/H5 side of this with.
static auto testWriteClusterAsciiExtinct() -> int
{
    const auto outDir = std::filesystem::temp_directory_path() / "slugTestOutputManagerAsciiExtinct";
    std::filesystem::remove_all(outDir);
    std::filesystem::create_directories(outDir);
    const std::string modelName = "test_model";
    const toml::table inputDeck = makeClusterPhysicsInputDeck(
        modelName, outDir, "tests/core/assets/testClusterExtinct.in");

    try
    {
        const io::SimControls controls(inputDeck);
        if (controls.extinct() == nullptr)
        {
            std::cerr << "testOutputManager: ascii extinct: test bug: "
                "expected SimControls::extinct() to be non-null\n";
            return 1;
        }

        utils::rng().seed(42);
        constexpr unsigned long uid = 23;
        constexpr double targetMass = 1e4;
        constexpr double formTime = 0.0;
        core::Cluster cluster(uid, targetMass, formTime, controls);
        constexpr double ageYr = 1e6;
        cluster.advance(ageYr);
        constexpr unsigned long trial = 4;

        {
            io::OutputManagerAscii manager(controls, inputDeck);
            manager.writeCluster(trial, cluster);
            manager.writeClusterSpec(trial, ageYr, cluster);
            manager.writeClusterPhot(trial, ageYr, cluster);
        }

        constexpr double tol = 1e-5;

        // clusters.txt: header should name an a_v column (units mag),
        // and the data row's a_v field should match cluster.aV()
        {
            std::ifstream file(outDir / (modelName + "_clusters.txt"));
            std::string headerLine;
            std::string unitsLine;
            std::string ruleLine;
            std::string dataLine;
            std::getline(file, headerLine);
            std::getline(file, unitsLine);
            std::getline(file, ruleLine);
            std::getline(file, dataLine);

            if (!headerLine.contains("a_v") || !unitsLine.contains("mag"))
            {
                std::cerr << "testOutputManager: ascii extinct: clusters.txt "
                    "header is missing the expected a_v (mag) column\n";
                return 1;
            }

            std::istringstream lineStream(dataLine);
            unsigned long readTrial = 0;
            unsigned long readUid = 0;
            double readTargetMass = 0.0;
            double readBirthMass = 0.0;
            double readFormTime = 0.0;
            double readFeH = 0.0;
            double readAV = 0.0;
            lineStream >> readTrial >> readUid >> readTargetMass >> readBirthMass
                >> readFormTime >> readFeH >> readAV;
            if (std::abs(readAV - cluster.aV()) > tol)
            {
                std::cerr << "testOutputManager: ascii extinct: clusters.txt "
                    "a_v = " << readAV << " does not match cluster.aV() = "
                    << cluster.aV() << "\n";
                return 1;
            }
        }

        // cluster_spectra.txt: one line per wavelength; spec_ex should
        // read 0 outside extinct()->wl()'s own coverage and
        // cluster.specExtinct() inside it
        {
            std::ifstream file(outDir / (modelName + "_cluster_spectra.txt"));
            std::string headerLine;
            std::string unitsLine;
            std::string ruleLine;
            std::getline(file, headerLine);
            std::getline(file, unitsLine);
            std::getline(file, ruleLine);

            if (!headerLine.contains("spec_ex"))
            {
                std::cerr << "testOutputManager: ascii extinct: cluster_spectra.txt "
                    "header is missing the expected spec_ex column\n";
                return 1;
            }

            const auto& specExtinct = cluster.specExtinct();
            const auto wlOffset = controls.extinct()->wlOffset();

            std::string dataLine;
            std::size_t i = 0;
            while (std::getline(file, dataLine))
            {
                unsigned long readTrial = 0;
                double readTime = 0.0;
                unsigned long readUid = 0;
                double readWl = 0.0;
                double readSpec = 0.0;
                double readSpecEx = 0.0;
                std::istringstream lineStream(dataLine);
                lineStream >> readTrial >> readTime >> readUid >> readWl
                    >> readSpec >> readSpecEx;

                const double expectedSpecEx =
                    (i >= wlOffset && (i - wlOffset) < specExtinct.size()) ?
                    specExtinct.at(i - wlOffset) : 0.0;
                const double denom = std::max(std::abs(expectedSpecEx), 1.0);
                if (std::abs(readSpecEx - expectedSpecEx) > tol * denom)
                {
                    std::cerr << "testOutputManager: ascii extinct: "
                        "cluster_spectra.txt row " << i << ": spec_ex = "
                        << readSpecEx << ", expected " << expectedSpecEx << "\n";
                    return 1;
                }
                ++i;
            }
            if (i != controls.specsyn()->wlObs().size())
            {
                std::cerr << "testOutputManager: ascii extinct: cluster_spectra.txt "
                    "has " << i << " data rows, expected "
                    << controls.specsyn()->wlObs().size() << "\n";
                return 1;
            }
        }

        // cluster_phot.txt: header should carry a "<filter>_ex" column
        // for each real filter (not Lbol), and the data row's trailing
        // columns should match cluster.photExtinct()
        {
            std::ifstream file(outDir / (modelName + "_cluster_phot.txt"));
            std::string headerLine;
            std::string unitsLine;
            std::string ruleLine;
            std::string dataLine;
            std::getline(file, headerLine);
            std::getline(file, unitsLine);
            std::getline(file, ruleLine);
            std::getline(file, dataLine);

            const auto& realFilterNames = controls.filters()->filterNames();
            for (const auto& name : realFilterNames)
            {
                if (!headerLine.contains(name + "_ex"))
                {
                    std::cerr << "testOutputManager: ascii extinct: cluster_phot.txt "
                        "header is missing the expected " << name << "_ex column\n";
                    return 1;
                }
            }

            std::istringstream lineStream(dataLine);
            unsigned long readTrial = 0;
            double readTime = 0.0;
            unsigned long readUid = 0;
            lineStream >> readTrial >> readTime >> readUid;
            // phot columns: real filters + Lbol (in that order, per
            // writeClusterPhot), then the extinct columns
            const std::size_t nPhotCols = realFilterNames.size() +
                (controls.computeLbol() ? 1 : 0);
            std::vector<double> readPhot(nPhotCols);
            for (double& v : readPhot) { lineStream >> v; }
            std::vector<double> readPhotExtinct(cluster.photExtinct().size());
            for (double& v : readPhotExtinct) { lineStream >> v; }

            for (std::size_t i = 0; i < readPhotExtinct.size(); ++i)
            {
                const double expected = cluster.photExtinct().at(i);
                if (std::abs(readPhotExtinct.at(i) - expected) > tol * std::abs(expected))
                {
                    std::cerr << "testOutputManager: ascii extinct: cluster_phot.txt "
                        "extinct column " << i << " = " << readPhotExtinct.at(i)
                        << ", expected " << expected << "\n";
                    return 1;
                }
            }
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testOutputManager: ascii extinct test failed: "
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
        utils::rng().seed(42);
        constexpr unsigned long uid = 11;
        constexpr double targetMass = 2e3;
        constexpr double formTime = 0.0;
        const core::Cluster cluster(uid, targetMass, formTime, controls);
        constexpr unsigned long trial = 5;

        {
            io::OutputManagerH5
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

// Verify that a cluster can be exactly reconstructed from its own
// recorded rng state: write a cluster's rng state to the HDF5
// clusters/rng dataset, read it back, and use it (together with the
// same uid/mass/time/physics) to construct a second Cluster via the
// rngState-accepting constructor. Since that constructor draws its
// stellar masses using exactly the given rng state, the two clusters'
// birthMass() should come out bitwise identical, even though the live
// rng stream has moved on in between (the first cluster's own
// construction, plus the second constructor's own [Fe/H] draw, both
// consume it) -- confirming the round trip does not depend on the live
// rng being in any particular state at reconstruction time.
static auto testWriteReadClusterRngRoundTrip() -> int
{
    const auto outDir = std::filesystem::temp_directory_path() / "slugTestOutputManagerRngRoundTrip";
    std::filesystem::remove_all(outDir);
    std::filesystem::create_directories(outDir);
    const std::string modelName = "test_model";
    const auto expectedPath = outDir / (modelName + ".h5");
    const toml::table inputDeck = makeClusterPhysicsInputDeck(modelName, outDir);

    try
    {
        const io::SimControls controls(inputDeck);
        utils::rng().seed(123);
        constexpr unsigned long uid = 13;
        constexpr double targetMass = 5e3;
        constexpr double formTime = 0.0;
        const core::Cluster cluster(uid, targetMass, formTime, controls);
        constexpr unsigned long trial = 1;

        {
            io::OutputManagerH5 manager(controls, inputDeck);
            manager.writeCluster(trial, cluster);
        }

        // NOLINTBEGIN(misc-include-cleaner)
        const hid_t file = H5Fopen(expectedPath.string().c_str(),
            H5F_ACC_RDONLY, H5P_DEFAULT);
        if (file < 0)
        {
            std::cerr << "testOutputManager: rng round trip: unable to reopen "
                << expectedPath.string() << "\n";
            return 1;
        }
        const hid_t grp = H5Gopen2(file, "clusters", H5P_DEFAULT);
        const hid_t dset = H5Dopen2(grp, "rng", H5P_DEFAULT);
        const hid_t strType = H5Tcopy(H5T_C_S1);
        H5Tset_size(strType, utils::rngStateWidth);
        utils::RngState readState{};
        H5Dread(dset, strType, H5S_ALL, H5S_ALL, H5P_DEFAULT, readState.data());
        H5Tclose(strType);
        H5Dclose(dset);
        H5Gclose(grp);
        H5Fclose(file);
        // NOLINTEND(misc-include-cleaner)

        const core::Cluster rebuilt(uid, targetMass, formTime, controls, readState);

        if (!std::ranges::equal(rebuilt.starMasses(), cluster.starMasses()))
        {
            std::cerr << "testOutputManager: rng round trip: rebuilt "
                "starMasses() does not exactly match the original's\n";
            return 1;
        }
        if (rebuilt.birthMass() != cluster.birthMass())
        {
            std::cerr << "testOutputManager: rng round trip: rebuilt "
                "birthMass() = " << rebuilt.birthMass() << " does not "
                "exactly match original birthMass() = " <<
                cluster.birthMass() << "\n";
            return 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testOutputManager: rng round trip test failed: "
            << error.what() << "\n";
        return 1;
    }
    return 0;
}

// Verify that OutputManager::writeClusterSpec/writeClusterPhot append
// spec_extinct/phot_extinct rows, alongside spec/phot, to the HDF5
// cluster_spectra/cluster_phot groups whenever SimControls::extinct()
// is set, and that writeCluster appends the cluster's own A_V to the
// clusters group's A_V dataset -- using testClusterExtinct.in, the
// same deck testCluster.cpp's own testClusterExtinct() exercises
// Cluster's side of this with. The values written are just
// cluster.aV()/specExtinct()/photExtinct() copied verbatim, so the
// round trip should be bitwise exact.
static auto testWriteClusterSpecPhotH5Extinct() -> int
{
    const auto outDir = std::filesystem::temp_directory_path() / "slugTestOutputManagerSpecPhotExtinct";
    std::filesystem::remove_all(outDir);
    std::filesystem::create_directories(outDir);
    const std::string modelName = "test_model";
    const auto expectedPath = outDir / (modelName + ".h5");
    const toml::table inputDeck = makeClusterPhysicsInputDeck(
        modelName, outDir, "tests/core/assets/testClusterExtinct.in");

    try
    {
        const io::SimControls controls(inputDeck);
        if (controls.extinct() == nullptr)
        {
            std::cerr << "testOutputManager: h5 spec/phot extinct: test bug: "
                "expected SimControls::extinct() to be non-null\n";
            return 1;
        }

        utils::rng().seed(42);
        constexpr unsigned long uid = 17;
        constexpr double targetMass = 1e4;
        constexpr double formTime = 0.0;
        core::Cluster cluster(uid, targetMass, formTime, controls);
        constexpr double ageYr = 1e6;
        cluster.advance(ageYr);
        constexpr unsigned long trial = 2;

        {
            io::OutputManagerH5 manager(controls, inputDeck);
            manager.writeCluster(trial, cluster);
            manager.writeClusterSpec(trial, ageYr, cluster);
            manager.writeClusterPhot(trial, ageYr, cluster);
        }

        // NOLINTBEGIN(misc-include-cleaner)
        const hid_t file = H5Fopen(expectedPath.string().c_str(),
            H5F_ACC_RDONLY, H5P_DEFAULT);
        if (file < 0)
        {
            std::cerr << "testOutputManager: h5 spec/phot extinct: unable to reopen "
                << expectedPath.string() << "\n";
            return 1;
        }

        const auto readRow = [](const hid_t grp, const char* name,
            const std::size_t expectedLen) -> std::vector<double>
        {
            const hid_t dset = H5Dopen2(grp, name, H5P_DEFAULT);
            if (dset < 0)
            {
                throw std::runtime_error(
                    std::string("missing expected dataset ") + name);
            }
            const hid_t space = H5Dget_space(dset);
            std::array<hsize_t, 2> dims{};
            H5Sget_simple_extent_dims(space, dims.data(), nullptr);
            H5Sclose(space);
            if (dims.at(0) != 1 || dims.at(1) != expectedLen)
            {
                H5Dclose(dset);
                throw std::runtime_error(
                    std::string(name) + " has shape (" + std::to_string(dims.at(0)) +
                    ", " + std::to_string(dims.at(1)) + "), expected (1, " +
                    std::to_string(expectedLen) + ")");
            }
            std::vector<double> row(expectedLen);
            H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, row.data());
            H5Dclose(dset);
            return row;
        };

        std::vector<double> readSpecExtinct;
        std::vector<double> readPhotExtinct;
        double readAV = 0.0;
        try
        {
            const hid_t specGrp = H5Gopen2(file, "cluster_spectra", H5P_DEFAULT);
            readSpecExtinct = readRow(specGrp, "spec_extinct", cluster.specExtinct().size());
            H5Gclose(specGrp);

            const hid_t photGrp = H5Gopen2(file, "cluster_phot", H5P_DEFAULT);
            readPhotExtinct = readRow(photGrp, "phot_extinct", cluster.photExtinct().size());
            H5Gclose(photGrp);

            const hid_t clustersGrp = H5Gopen2(file, "clusters", H5P_DEFAULT);
            const hid_t aVDset = H5Dopen2(clustersGrp, "A_V", H5P_DEFAULT);
            if (aVDset < 0)
            {
                H5Gclose(clustersGrp);
                throw std::runtime_error("missing expected dataset A_V");
            }
            H5Dread(aVDset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &readAV);
            H5Dclose(aVDset);
            H5Gclose(clustersGrp);
        }
        catch (const std::runtime_error& error)
        {
            H5Fclose(file);
            std::cerr << "testOutputManager: h5 spec/phot extinct: " << error.what() << "\n";
            return 1;
        }
        H5Fclose(file);
        // NOLINTEND(misc-include-cleaner)

        if (!std::ranges::equal(readSpecExtinct, cluster.specExtinct()))
        {
            std::cerr << "testOutputManager: h5 spec/phot extinct: spec_extinct "
                "row does not match cluster.specExtinct()\n";
            return 1;
        }
        if (!std::ranges::equal(readPhotExtinct, cluster.photExtinct()))
        {
            std::cerr << "testOutputManager: h5 spec/phot extinct: phot_extinct "
                "row does not match cluster.photExtinct()\n";
            return 1;
        }
        if (readAV != cluster.aV())
        {
            std::cerr << "testOutputManager: h5 spec/phot extinct: A_V = " << readAV
                << " does not match cluster.aV() = " << cluster.aV() << "\n";
            return 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testOutputManager: h5 spec/phot extinct test failed: "
            << error.what() << "\n";
        return 1;
    }
    return 0;
}

// Verify the ascii OutputManager: opens <model>_summary.txt, writes the
// slug-hash/date/time/rng_state header followed by the toml input
// deck, and refuses to overwrite an existing file.
static auto testOutputManagerAscii() -> int
{
    const auto outDir = std::filesystem::temp_directory_path() / "slugTestOutputManagerAscii";
    std::filesystem::remove_all(outDir);
    std::filesystem::create_directories(outDir);
    const std::string modelName = "test_model";
    const auto expectedPath = outDir / (modelName + "_summary.txt");
    const toml::table inputDeck = makeClusterPhysicsInputDeck(modelName, outDir);
    const io::SimControls controls(inputDeck);

    try
    {
        const io::OutputManagerAscii
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
        if (!contents.contains("rng_state  "))
        {
            std::cerr << "testOutputManager: ascii: missing expected rng_state line\n";
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
        const io::OutputManagerAscii
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
// slug-hash/date/time/rng_state header as top-level attributes,
// dumps the toml input deck into an input_deck group, and refuses
// to overwrite an existing file.
static auto testOutputManagerH5() -> int
{
    const auto outDir = std::filesystem::temp_directory_path() / "slugTestOutputManagerH5";
    std::filesystem::remove_all(outDir);
    std::filesystem::create_directories(outDir);
    const std::string modelName = "test_model";
    const auto expectedPath = outDir / (modelName + ".h5");
    const toml::table inputDeck = makeClusterPhysicsInputDeck(modelName, outDir);
    const io::SimControls controls(inputDeck);

    try
    {
        {
            const io::OutputManagerH5
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
            H5Aexists(file, "time") <= 0 ||
            H5Aexists(file, "rng_state") <= 0)
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
        const io::OutputManagerH5
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
    result += testWriteClusterAsciiExtinct();
    result += testWriteClusterH5();
    result += testWriteReadClusterRngRoundTrip();
    result += testWriteClusterSpecPhotH5Extinct();
    return result;
}
