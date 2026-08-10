/**
 * @file testOutputManager.cpp
 * @author Mark Krumholz
 * @brief Unit tests for the OutputManager class.
 * @date 2026-07-16
 */

#include "../src/core/Cluster.hpp"
#include "../src/core/Galaxy.hpp"
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
#include <iterator>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <toml.hpp>
#include <utility>
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
    if (toml::table* outputTbl = inputDeck["output"].as_table())
    { outputTbl->insert("model_name", modelName); }
    else { inputDeck.insert("output", toml::table{ { "model_name", modelName } }); }
    if (toml::table* outputsTbl = inputDeck["outputs"].as_table())
    { outputsTbl->insert("out_dir", outDir.string()); }
    else { inputDeck.insert("outputs", toml::table{ { "out_dir", outDir.string() } }); }
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
        std::vector<double> readWlExtinct;
        std::vector<double> readPhotExtinct;
        double readAV = 0.0;
        try
        {
            const hid_t specGrp = H5Gopen2(file, "cluster_spectra", H5P_DEFAULT);
            readSpecExtinct = readRow(specGrp, "spec_extinct", cluster.specExtinct().size());

            // Regression check: cluster_spectra used to write "wl"
            // (the full, unextincted grid) but no separate "wl_extinct"
            // dataset at all, leaving spec_extinct's own wavelength
            // grid unrecorded
            const hid_t wlExtinctDset = H5Dopen2(specGrp, "wl_extinct", H5P_DEFAULT);
            if (wlExtinctDset < 0)
            {
                H5Gclose(specGrp);
                throw std::runtime_error("missing expected dataset wl_extinct");
            }
            const hid_t wlExtinctSpace = H5Dget_space(wlExtinctDset);
            hsize_t wlExtinctLen = 0;
            H5Sget_simple_extent_dims(wlExtinctSpace, &wlExtinctLen, nullptr);
            H5Sclose(wlExtinctSpace);
            readWlExtinct.resize(wlExtinctLen);
            H5Dread(wlExtinctDset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, readWlExtinct.data());
            H5Dclose(wlExtinctDset);
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
        if (!std::ranges::equal(readWlExtinct, controls.extinct()->wlObs()))
        {
            std::cerr << "testOutputManager: h5 spec/phot extinct: wl_extinct "
                "does not match extinct()->wlObs()\n";
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

// Verify that output.write_cluster_spec = false (an optional key,
// defaulting to true) suppresses the cluster_spectra HDF5 group and
// the ascii cluster_spectra.txt file, even though a spectral
// synthesizer was requested -- letting a simulation compute cluster
// spectra (needed as an intermediate for photometry) without also
// paying the disk space to write them out.
static auto testOptOutClusterSpecOutput() -> int
{
    const auto outDir = std::filesystem::temp_directory_path() / "slugTestOutputManagerNoSpecOutput";
    std::filesystem::remove_all(outDir);
    std::filesystem::create_directories(outDir);
    const std::string modelName = "test_model";
    toml::table inputDeck = makeClusterPhysicsInputDeck(modelName, outDir);
    inputDeck.at_path("output").as_table()->insert("write_cluster_spec", false);

    try
    {
        const io::SimControls controls(inputDeck);
        if (controls.specsyn() == nullptr)
        {
            std::cerr << "testOutputManager: opt-out cluster spec: test bug: "
                "expected SimControls::specsyn() to be non-null\n";
            return 1;
        }

        {
            const io::OutputManagerAscii manager(controls, inputDeck);
        }
        {
            const io::OutputManagerH5 manager(controls, inputDeck);
        }

        const auto asciiSpecPath = outDir / (modelName + "_cluster_spectra.txt");
        if (std::filesystem::exists(asciiSpecPath))
        {
            std::cerr << "testOutputManager: opt-out cluster spec: ascii "
                "unexpectedly wrote " << asciiSpecPath.string() << "\n";
            return 1;
        }

        const auto h5Path = outDir / (modelName + ".h5");
        // NOLINTBEGIN(misc-include-cleaner)
        const hid_t file = H5Fopen(h5Path.string().c_str(),
            H5F_ACC_RDONLY, H5P_DEFAULT);
        if (file < 0)
        {
            std::cerr << "testOutputManager: opt-out cluster spec: unable to reopen "
                << h5Path.string() << "\n";
            return 1;
        }
        const bool hasSpecGroup = H5Lexists(file, "cluster_spectra", H5P_DEFAULT) > 0;
        H5Fclose(file);
        // NOLINTEND(misc-include-cleaner)

        if (hasSpecGroup)
        {
            std::cerr << "testOutputManager: opt-out cluster spec: h5 "
                "unexpectedly created a cluster_spectra group\n";
            return 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testOutputManager: opt-out cluster spec test failed: "
            << error.what() << "\n";
        return 1;
    }
    return 0;
}

// Build a valid input deck for a galaxy-type simulation that also has
// usable stellar physics, spectral synthesis, photometry, and
// extinction, by reusing an existing deck (testGalaxyDynamics.in by
// default -- the same one testGalaxy.cpp's own Galaxy-class tests use,
// chosen over the simpler testGalaxy.in, which has neither photometry
// nor extinction) and injecting model_name/out_dir into it, mirroring
// makeClusterPhysicsInputDeck's own approach.
static auto makeGalaxyPhysicsInputDeck(const std::string& modelName,
    const std::filesystem::path& outDir,
    const std::string& deckPath = "tests/core/assets/testGalaxyDynamics.in") -> toml::table
{
    toml::table inputDeck = toml::parse_file(deckPath);
    if (toml::table* outputTbl = inputDeck["output"].as_table())
    { outputTbl->insert("model_name", modelName); }
    else { inputDeck.insert("output", toml::table{ { "model_name", modelName } }); }
    if (toml::table* outputsTbl = inputDeck["outputs"].as_table())
    { outputsTbl->insert("out_dir", outDir.string()); }
    else { inputDeck.insert("outputs", toml::table{ { "out_dir", outDir.string() } }); }
    return inputDeck;
}

// Read a scalar string "units" attribute off an HDF5 dataset
static auto readUnitsAttr(const hid_t dset) -> std::string // NOLINT(misc-include-cleaner)
{
    // NOLINTBEGIN(misc-include-cleaner)
    const hid_t attr = H5Aopen(dset, "units", H5P_DEFAULT);
    if (attr < 0) { throw std::runtime_error("missing expected units attribute"); }
    const hid_t strType = H5Aget_type(attr);
    char* cstr = nullptr;
    H5Aread(attr, strType, &cstr); // NOLINT(bugprone-multi-level-implicit-pointer-conversion)
    std::string result(cstr != nullptr ? cstr : "");
    if (cstr != nullptr)
    {
        const hid_t space = H5Aget_space(attr);
        H5Dvlen_reclaim(strType, space, H5P_DEFAULT, &cstr); // NOLINT(bugprone-multi-level-implicit-pointer-conversion)
        H5Sclose(space);
    }
    H5Tclose(strType);
    H5Aclose(attr);
    return result;
    // NOLINTEND(misc-include-cleaner)
}

// Read a 1d array-of-strings attribute called name on the HDF5 object loc
static auto readStringArrayAttr(const hid_t loc, const char* name) -> std::vector<std::string> // NOLINT(misc-include-cleaner)
{
    // NOLINTBEGIN(misc-include-cleaner)
    const hid_t attr = H5Aopen(loc, name, H5P_DEFAULT);
    if (attr < 0)
    {
        throw std::runtime_error(std::string("missing expected attribute ") + name);
    }
    const hid_t space = H5Aget_space(attr);
    const auto npoints = static_cast<std::size_t>(H5Sget_simple_extent_npoints(space));
    const hid_t memType = H5Aget_type(attr);

    std::vector<char*> buf(npoints);
    H5Aread(attr, memType, static_cast<void*>(buf.data())); // NOLINT(bugprone-multi-level-implicit-pointer-conversion)
    std::vector<std::string> values;
    values.reserve(npoints);
    for (const auto* s : buf) { values.emplace_back(s); }

    H5Dvlen_reclaim(memType, space, H5P_DEFAULT, static_cast<void*>(buf.data())); // NOLINT(bugprone-multi-level-implicit-pointer-conversion)
    H5Tclose(memType);
    H5Sclose(space);
    H5Aclose(attr);
    return values;
    // NOLINTEND(misc-include-cleaner)
}

// Get the current length of a 1d dataset, or the (rows, cols) extent
// of a 2d dataset, in loc
static auto readExtent1d(const hid_t loc, const char* name) -> hsize_t // NOLINT(misc-include-cleaner)
{
    // NOLINTBEGIN(misc-include-cleaner)
    const hid_t dset = H5Dopen2(loc, name, H5P_DEFAULT);
    if (dset < 0) { throw std::runtime_error(std::string("missing expected dataset ") + name); }
    const hid_t space = H5Dget_space(dset);
    hsize_t len = 0;
    H5Sget_simple_extent_dims(space, &len, nullptr);
    H5Sclose(space);
    H5Dclose(dset);
    return len;
    // NOLINTEND(misc-include-cleaner)
}
static auto readExtent2d(const hid_t loc, const char* name) -> std::pair<hsize_t, hsize_t> // NOLINT(misc-include-cleaner)
{
    // NOLINTBEGIN(misc-include-cleaner)
    const hid_t dset = H5Dopen2(loc, name, H5P_DEFAULT);
    if (dset < 0) { throw std::runtime_error(std::string("missing expected dataset ") + name); }
    const hid_t space = H5Dget_space(dset);
    std::array<hsize_t, 2> dims{};
    H5Sget_simple_extent_dims(space, dims.data(), nullptr);
    H5Sclose(space);
    H5Dclose(dset);
    return { dims.at(0), dims.at(1) };
    // NOLINTEND(misc-include-cleaner)
}

// Read an entire 1d dataset of doubles, or of unsigned longs, from loc
static auto readColumnDouble(const hid_t loc, const char* name) -> std::vector<double> // NOLINT(misc-include-cleaner)
{
    // NOLINTBEGIN(misc-include-cleaner)
    std::vector<double> result(readExtent1d(loc, name));
    const hid_t dset = H5Dopen2(loc, name, H5P_DEFAULT);
    H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, result.data());
    H5Dclose(dset);
    return result;
    // NOLINTEND(misc-include-cleaner)
}
static auto readColumnULong(const hid_t loc, const char* name) -> std::vector<unsigned long> // NOLINT(misc-include-cleaner)
{
    // NOLINTBEGIN(misc-include-cleaner)
    std::vector<unsigned long> result(readExtent1d(loc, name));
    const hid_t dset = H5Dopen2(loc, name, H5P_DEFAULT);
    H5Dread(dset, H5T_NATIVE_ULONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, result.data());
    H5Dclose(dset);
    return result;
    // NOLINTEND(misc-include-cleaner)
}

// Read one row (of nCols doubles) of a 2d dataset in loc
static auto readRow2d(const hid_t loc, const char* name,
    const hsize_t row, const hsize_t nCols) -> std::vector<double> // NOLINT(misc-include-cleaner)
{
    // NOLINTBEGIN(misc-include-cleaner)
    const hid_t dset = H5Dopen2(loc, name, H5P_DEFAULT);
    if (dset < 0) { throw std::runtime_error(std::string("missing expected dataset ") + name); }
    const hid_t fileSpace = H5Dget_space(dset);
    const std::array<hsize_t, 2> start = { row, 0 };
    const std::array<hsize_t, 2> count = { 1, nCols };
    H5Sselect_hyperslab(fileSpace, H5S_SELECT_SET, start.data(), nullptr, count.data(), nullptr);
    const hid_t memSpace = H5Screate_simple(2, count.data(), nullptr);
    std::vector<double> result(nCols);
    H5Dread(dset, H5T_NATIVE_DOUBLE, memSpace, fileSpace, H5P_DEFAULT, result.data());
    H5Sclose(memSpace);
    H5Sclose(fileSpace);
    H5Dclose(dset);
    return result;
    // NOLINTEND(misc-include-cleaner)
}

// Find value's index in column, throwing std::runtime_error (caught
// by the caller) if it is not present
static auto findIndex(const std::vector<unsigned long>& column,
    const unsigned long value) -> hsize_t
{
    const auto it = std::ranges::find(column, value);
    if (it == column.end())
    {
        throw std::runtime_error("uid " + std::to_string(value) + " not found");
    }
    return static_cast<hsize_t>(std::distance(column.begin(), it));
}

// Check the galaxy group: empty trial/time/target_mass/actual_mass
// datasets, each with the expected units attribute -- see
// testGalaxyGroupsH5()'s own docstring. Throws std::runtime_error
// (caught by the caller) describing the first mismatch found.
static void checkGalaxyGroup(const hid_t file) // NOLINT(misc-include-cleaner)
{
    // NOLINTBEGIN(misc-include-cleaner)
    const hid_t galaxyGrp = H5Gopen2(file, "galaxy", H5P_DEFAULT);
    if (galaxyGrp < 0) { throw std::runtime_error("missing galaxy group"); }
    for (const auto& [name, units] : { std::pair{"trial", ""},
        std::pair{"time", "yr"}, std::pair{"target_mass", "Msun"},
        std::pair{"actual_mass", "Msun"} })
    {
        if (readExtent1d(galaxyGrp, name) != 0)
        {
            H5Gclose(galaxyGrp);
            throw std::runtime_error(std::string(name) + " is not empty");
        }
        const hid_t dset = H5Dopen2(galaxyGrp, name, H5P_DEFAULT);
        const auto actualUnits = readUnitsAttr(dset);
        H5Dclose(dset);
        if (actualUnits != units)
        {
            H5Gclose(galaxyGrp);
            throw std::runtime_error(std::string(name) + " has units '" +
                actualUnits + "', expected '" + units + "'");
        }
    }
    H5Gclose(galaxyGrp);
    // NOLINTEND(misc-include-cleaner)
}

// Check the galaxy_spectra group: fixed wl/wl_extinct datasets
// (matching SimControls::specsyn()/extinct()'s own wlObs(), each with
// units "Angstrom") plus empty trial/time/spec/spec_extinct -- see
// testGalaxyGroupsH5()'s own docstring. Throws std::runtime_error
// (caught by the caller) describing the first mismatch found.
static void checkGalaxySpectraGroup(const hid_t file, const io::SimControls& controls) // NOLINT(misc-include-cleaner)
{
    // NOLINTBEGIN(misc-include-cleaner)
    const hid_t specGrp = H5Gopen2(file, "galaxy_spectra", H5P_DEFAULT);
    if (specGrp < 0) { throw std::runtime_error("missing galaxy_spectra group"); }

    const auto& wlObs = controls.specsyn()->wlObs();
    const auto& wlExtinctObs = controls.extinct()->wlObs();

    const auto checkFixed = [&](const char* name, const std::vector<double>& expected)
    {
        if (readExtent1d(specGrp, name) != expected.size())
        {
            throw std::runtime_error(std::string(name) + " has unexpected length");
        }
        const hid_t dset = H5Dopen2(specGrp, name, H5P_DEFAULT);
        std::vector<double> actual(expected.size());
        H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, actual.data());
        const auto units = readUnitsAttr(dset);
        H5Dclose(dset);
        if (!std::ranges::equal(actual, expected) || units != "Angstrom")
        {
            throw std::runtime_error(std::string(name) + " does not match the expected wlObs()/Angstrom");
        }
    };

    try
    {
        checkFixed("wl", wlObs);
        checkFixed("wl_extinct", wlExtinctObs);
        if (readExtent1d(specGrp, "trial") != 0 || readExtent1d(specGrp, "time") != 0)
        {
            throw std::runtime_error("galaxy_spectra trial/time are not empty");
        }
        const auto [specRows, specCols] = readExtent2d(specGrp, "spec");
        if (specRows != 0 || specCols != wlObs.size())
        {
            throw std::runtime_error("spec has unexpected shape");
        }
        const auto [specExRows, specExCols] = readExtent2d(specGrp, "spec_extinct");
        if (specExRows != 0 || specExCols != wlExtinctObs.size())
        {
            throw std::runtime_error("spec_extinct has unexpected shape");
        }
    }
    catch (...)
    {
        H5Gclose(specGrp);
        throw;
    }
    H5Gclose(specGrp);
    // NOLINTEND(misc-include-cleaner)
}

// Check the galaxy_phot group: a "filters" attribute matching
// filterNames() (with "Lbol" appended) plus empty trial/time/phot/
// phot_extinct -- see testGalaxyGroupsH5()'s own docstring. Throws
// std::runtime_error (caught by the caller) describing the first
// mismatch found.
static void checkGalaxyPhotGroup(const hid_t file, const io::SimControls& controls) // NOLINT(misc-include-cleaner)
{
    // NOLINTBEGIN(misc-include-cleaner)
    const hid_t photGrp = H5Gopen2(file, "galaxy_phot", H5P_DEFAULT);
    if (photGrp < 0) { throw std::runtime_error("missing galaxy_phot group"); }

    try
    {
        std::vector<std::string> expectedFilters = controls.filters()->filterNames();
        expectedFilters.emplace_back("Lbol");
        const auto actualFilters = readStringArrayAttr(photGrp, "filters");
        if (actualFilters != expectedFilters)
        {
            throw std::runtime_error("filters attribute does not match expected filter list");
        }
        if (readExtent1d(photGrp, "trial") != 0 || readExtent1d(photGrp, "time") != 0)
        {
            throw std::runtime_error("galaxy_phot trial/time are not empty");
        }
        const auto [photRows, photCols] = readExtent2d(photGrp, "phot");
        if (photRows != 0 || photCols != expectedFilters.size())
        {
            throw std::runtime_error("phot has unexpected shape");
        }
        const auto [photExRows, photExCols] = readExtent2d(photGrp, "phot_extinct");
        if (photExRows != 0 || photExCols != controls.filters()->filterNames().size())
        {
            throw std::runtime_error("phot_extinct has unexpected shape");
        }
    }
    catch (...)
    {
        H5Gclose(photGrp);
        throw;
    }
    H5Gclose(photGrp);
    // NOLINTEND(misc-include-cleaner)
}

// Verify that OutputManagerH5's constructor creates the galaxy,
// galaxy_spectra, and galaxy_phot groups (and only those -- no rows
// have been written yet) for a galaxy-type simulation with spectral
// synthesis, photometry, and extinction all requested -- see
// checkGalaxyGroup()/checkGalaxySpectraGroup()/checkGalaxyPhotGroup()'s
// own docstrings for exactly what each group is expected to hold. See
// testGalaxyGroupsAbsentH5() for the complementary check that none of
// these groups exist for a cluster-type simulation.
static auto testGalaxyGroupsH5() -> int
{
    const auto outDir = std::filesystem::temp_directory_path() / "slugTestOutputManagerGalaxyGroupsH5";
    std::filesystem::remove_all(outDir);
    std::filesystem::create_directories(outDir);
    const std::string modelName = "test_model";
    const auto expectedPath = outDir / (modelName + ".h5");
    const toml::table inputDeck = makeGalaxyPhysicsInputDeck(modelName, outDir);

    try
    {
        const io::SimControls controls(inputDeck);
        if (controls.specsyn() == nullptr || controls.filters() == nullptr ||
            controls.extinct() == nullptr)
        {
            std::cerr << "testOutputManager: galaxy groups h5: test bug: "
                "expected specsyn()/filters()/extinct() to all be non-null\n";
            return 1;
        }

        {
            const io::OutputManagerH5 manager(controls, inputDeck);
        }

        // NOLINTBEGIN(misc-include-cleaner)
        const hid_t file = H5Fopen(expectedPath.string().c_str(),
            H5F_ACC_RDONLY, H5P_DEFAULT);
        if (file < 0)
        {
            std::cerr << "testOutputManager: galaxy groups h5: unable to reopen "
                << expectedPath.string() << "\n";
            return 1;
        }

        try
        {
            checkGalaxyGroup(file);
            checkGalaxySpectraGroup(file, controls);
            checkGalaxyPhotGroup(file, controls);
        }
        catch (const std::runtime_error& error)
        {
            H5Fclose(file);
            std::cerr << "testOutputManager: galaxy groups h5: " << error.what() << "\n";
            return 1;
        }
        H5Fclose(file);
        // NOLINTEND(misc-include-cleaner)
    }
    catch (const std::exception& error)
    {
        std::cerr << "testOutputManager: galaxy groups h5 test failed: "
            << error.what() << "\n";
        return 1;
    }
    return 0;
}

// Verify that OutputManagerH5's constructor creates none of the
// galaxy/galaxy_spectra/galaxy_phot groups for a cluster-type
// simulation, even with spectral synthesis, photometry, and extinction
// all requested -- a cluster-type simulation has no Galaxy object at
// all, so there is nothing for these groups to hold.
static auto testGalaxyGroupsAbsentH5() -> int
{
    const auto outDir = std::filesystem::temp_directory_path() / "slugTestOutputManagerGalaxyGroupsAbsentH5";
    std::filesystem::remove_all(outDir);
    std::filesystem::create_directories(outDir);
    const std::string modelName = "test_model";
    const auto expectedPath = outDir / (modelName + ".h5");
    const toml::table inputDeck = makeClusterPhysicsInputDeck(
        modelName, outDir, "tests/core/assets/testClusterExtinct.in");

    try
    {
        const io::SimControls controls(inputDeck);
        if (controls.simType() != io::SimControls::SimType::cluster)
        {
            std::cerr << "testOutputManager: galaxy groups absent h5: test bug: "
                "expected simType() == cluster\n";
            return 1;
        }

        {
            const io::OutputManagerH5 manager(controls, inputDeck);
        }

        // NOLINTBEGIN(misc-include-cleaner)
        const hid_t file = H5Fopen(expectedPath.string().c_str(),
            H5F_ACC_RDONLY, H5P_DEFAULT);
        if (file < 0)
        {
            std::cerr << "testOutputManager: galaxy groups absent h5: unable to reopen "
                << expectedPath.string() << "\n";
            return 1;
        }
        const bool anyGalaxyGroup =
            H5Lexists(file, "galaxy", H5P_DEFAULT) > 0 ||
            H5Lexists(file, "galaxy_spectra", H5P_DEFAULT) > 0 ||
            H5Lexists(file, "galaxy_phot", H5P_DEFAULT) > 0;
        H5Fclose(file);
        // NOLINTEND(misc-include-cleaner)

        if (anyGalaxyGroup)
        {
            std::cerr << "testOutputManager: galaxy groups absent h5: "
                "unexpectedly created a galaxy/galaxy_spectra/galaxy_phot group "
                "for a cluster-type simulation\n";
            return 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testOutputManager: galaxy groups absent h5 test failed: "
            << error.what() << "\n";
        return 1;
    }
    return 0;
}

// Verify that OutputManagerAscii's constructor writes the expected
// column-name/units header lines to _galaxy.txt, _galaxy_spectra.txt,
// and _galaxy_phot.txt for a galaxy-type simulation with spectral
// synthesis, photometry, and extinction all requested -- mirroring
// testGalaxyGroupsH5()'s own checks, but for the ascii column layout
// (no "uid" column, since a galaxy has no individual identity -- see
// writeGalaxyHeader/writeGalaxySpectraHeader/writeGalaxyPhotHeader's
// own comments).
static auto testGalaxyFilesAscii() -> int
{
    const auto outDir = std::filesystem::temp_directory_path() / "slugTestOutputManagerGalaxyFilesAscii";
    std::filesystem::remove_all(outDir);
    std::filesystem::create_directories(outDir);
    const std::string modelName = "test_model";
    const toml::table inputDeck = makeGalaxyPhysicsInputDeck(modelName, outDir);

    try
    {
        const io::SimControls controls(inputDeck);
        if (controls.specsyn() == nullptr || controls.filters() == nullptr ||
            controls.extinct() == nullptr)
        {
            std::cerr << "testOutputManager: galaxy files ascii: test bug: "
                "expected specsyn()/filters()/extinct() to all be non-null\n";
            return 1;
        }

        {
            const io::OutputManagerAscii manager(controls, inputDeck);
        }

        // _galaxy.txt: trial/time/target_mass/actual_mass, no uid
        {
            std::ifstream file(outDir / (modelName + "_galaxy.txt"));
            std::string headerLine;
            std::string unitsLine;
            std::getline(file, headerLine);
            std::getline(file, unitsLine);
            if (!headerLine.contains("trial") || !headerLine.contains("time") ||
                !headerLine.contains("target_mass") || !headerLine.contains("actual_mass") ||
                headerLine.contains("uid"))
            {
                std::cerr << "testOutputManager: galaxy files ascii: _galaxy.txt "
                    "header does not match the expected trial/time/target_mass/"
                    "actual_mass columns (and no uid): " << headerLine << "\n";
                return 1;
            }
            if (!unitsLine.contains("yr") || !unitsLine.contains("Msun"))
            {
                std::cerr << "testOutputManager: galaxy files ascii: _galaxy.txt "
                    "units line does not match the expected yr/Msun units: "
                    << unitsLine << "\n";
                return 1;
            }
        }

        // _galaxy_spectra.txt: trial/time/wl/spec/spec_ex, no uid
        {
            std::ifstream file(outDir / (modelName + "_galaxy_spectra.txt"));
            std::string headerLine;
            std::string unitsLine;
            std::getline(file, headerLine);
            std::getline(file, unitsLine);
            if (!headerLine.contains("trial") || !headerLine.contains("time") ||
                !headerLine.contains("wl") || !headerLine.contains("spec") ||
                !headerLine.contains("spec_ex") || headerLine.contains("uid"))
            {
                std::cerr << "testOutputManager: galaxy files ascii: "
                    "_galaxy_spectra.txt header does not match the expected "
                    "trial/time/wl/spec/spec_ex columns (and no uid): "
                    << headerLine << "\n";
                return 1;
            }
            if (!unitsLine.contains("Angstrom") || !unitsLine.contains("erg/s/Angstrom"))
            {
                std::cerr << "testOutputManager: galaxy files ascii: "
                    "_galaxy_spectra.txt units line does not match the expected "
                    "Angstrom/erg-s-Angstrom units: " << unitsLine << "\n";
                return 1;
            }
        }

        // _galaxy_phot.txt: trial/time, one column per filter (+ Lbol),
        // then one "<filter>_ex" column per real filter, no uid
        {
            std::ifstream file(outDir / (modelName + "_galaxy_phot.txt"));
            std::string headerLine;
            std::getline(file, headerLine);
            if (!headerLine.contains("trial") || !headerLine.contains("time") ||
                headerLine.contains("uid") || !headerLine.contains("Lbol"))
            {
                std::cerr << "testOutputManager: galaxy files ascii: "
                    "_galaxy_phot.txt header does not match the expected "
                    "trial/time/.../Lbol columns (and no uid): " << headerLine << "\n";
                return 1;
            }
            for (const auto& name : controls.filters()->filterNames())
            {
                if (!headerLine.contains(name) || !headerLine.contains(name + "_ex"))
                {
                    std::cerr << "testOutputManager: galaxy files ascii: "
                        "_galaxy_phot.txt header is missing the expected " << name
                        << "/" << name << "_ex columns\n";
                    return 1;
                }
            }
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testOutputManager: galaxy files ascii test failed: "
            << error.what() << "\n";
        return 1;
    }
    return 0;
}

// Verify that OutputManagerAscii's constructor writes none of
// _galaxy.txt, _galaxy_spectra.txt, or _galaxy_phot.txt for a
// cluster-type simulation, even with spectral synthesis, photometry,
// and extinction all requested -- mirrors testGalaxyGroupsAbsentH5()'s
// own check for the ascii side.
static auto testGalaxyFilesAbsentAscii() -> int
{
    const auto outDir = std::filesystem::temp_directory_path() / "slugTestOutputManagerGalaxyFilesAbsentAscii";
    std::filesystem::remove_all(outDir);
    std::filesystem::create_directories(outDir);
    const std::string modelName = "test_model";
    const toml::table inputDeck = makeClusterPhysicsInputDeck(
        modelName, outDir, "tests/core/assets/testClusterExtinct.in");

    try
    {
        const io::SimControls controls(inputDeck);
        if (controls.simType() != io::SimControls::SimType::cluster)
        {
            std::cerr << "testOutputManager: galaxy files absent ascii: test bug: "
                "expected simType() == cluster\n";
            return 1;
        }

        {
            const io::OutputManagerAscii manager(controls, inputDeck);
        }

        for (const std::string& suffix : { "_galaxy.txt", "_galaxy_spectra.txt", "_galaxy_phot.txt" })
        {
            const auto path = outDir / (modelName + suffix);
            if (std::filesystem::exists(path))
            {
                std::cerr << "testOutputManager: galaxy files absent ascii: "
                    "unexpectedly wrote " << path.string()
                    << " for a cluster-type simulation\n";
                return 1;
            }
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testOutputManager: galaxy files absent ascii test failed: "
            << error.what() << "\n";
        return 1;
    }
    return 0;
}

// Advance time used by testWriteGalaxyH5()/testWriteGalaxyAscii(): a
// single advance() call to this time, from testGalaxyDynamics.in's
// own CMF/sfr, forms a manageable number of clusters (a few dozen --
// see that deck's own comment on why it bounds pcubature's cost) with
// clusters.CLF (5e5) long enough that none have disrupted yet, so
// galaxy.clusters() holds every cluster formed.
static constexpr double galaxyWriteTime = 3e5;

// Check the galaxy group's own row (written by writeGalaxy) matches
// trial/time/galaxy.targetMass()/galaxy.actualMass(). Throws
// std::runtime_error (caught by the caller) describing any mismatch.
static void checkGalaxyRowH5(const hid_t file, const unsigned long trial,
    const double time, core::Galaxy& galaxy)
{
    // NOLINTBEGIN(misc-include-cleaner)
    const hid_t grp = H5Gopen2(file, "galaxy", H5P_DEFAULT);
    if (grp < 0) { throw std::runtime_error("missing galaxy group"); }
    const auto readTrial = readColumnULong(grp, "trial");
    const auto readTime = readColumnDouble(grp, "time");
    const auto readTargetMass = readColumnDouble(grp, "target_mass");
    const auto readActualMass = readColumnDouble(grp, "actual_mass");
    H5Gclose(grp);
    // NOLINTEND(misc-include-cleaner)

    if (readTrial.size() != 1 || readTrial.at(0) != trial ||
        readTime.at(0) != time ||
        readTargetMass.at(0) != galaxy.targetMass() ||
        readActualMass.at(0) != galaxy.actualMass())
    {
        throw std::runtime_error("galaxy group row does not match trial/time/"
            "targetMass()/actualMass()");
    }
}

// Check that the clusters group holds exactly one row -- matching
// target_mass and birth_mass -- for every cluster in galaxy.clusters(),
// written by writeGalaxy()'s own per-cluster writeCluster() calls.
// Throws std::runtime_error (caught by the caller) describing any
// mismatch.
static void checkClustersPassthroughH5(const hid_t file, core::Galaxy& galaxy)
{
    // NOLINTBEGIN(misc-include-cleaner)
    const hid_t grp = H5Gopen2(file, "clusters", H5P_DEFAULT);
    if (grp < 0) { throw std::runtime_error("missing clusters group"); }
    const auto uidCol = readColumnULong(grp, "uid");
    const auto targetMassCol = readColumnDouble(grp, "target_mass");
    const auto birthMassCol = readColumnDouble(grp, "birth_mass");
    H5Gclose(grp);
    // NOLINTEND(misc-include-cleaner)

    if (uidCol.size() != galaxy.clusters().size())
    {
        throw std::runtime_error("clusters group has " + std::to_string(uidCol.size()) +
            " rows, expected " + std::to_string(galaxy.clusters().size()));
    }
    for (const auto& cluster : galaxy.clusters())
    {
        const auto i = findIndex(uidCol, cluster.uid());
        if (targetMassCol.at(i) != cluster.targetMass() ||
            birthMassCol.at(i) != cluster.birthMass())
        {
            throw std::runtime_error("clusters group row for uid " +
                std::to_string(cluster.uid()) + " does not match");
        }
    }
}

// Check the galaxy_spectra group's own row (written by
// writeGalaxySpec) matches galaxy.spec()/specExtinct(). Throws
// std::runtime_error (caught by the caller) describing any mismatch.
static void checkGalaxySpecRowH5(const hid_t file, core::Galaxy& galaxy)
{
    // NOLINTBEGIN(misc-include-cleaner)
    const hid_t grp = H5Gopen2(file, "galaxy_spectra", H5P_DEFAULT);
    if (grp < 0) { throw std::runtime_error("missing galaxy_spectra group"); }
    const auto readSpec = readRow2d(grp, "spec", 0, galaxy.spec().size());
    const auto readSpecExtinct = readRow2d(grp, "spec_extinct", 0, galaxy.specExtinct().size());
    H5Gclose(grp);
    // NOLINTEND(misc-include-cleaner)

    if (!std::ranges::equal(readSpec, galaxy.spec()))
    {
        throw std::runtime_error("galaxy_spectra spec row does not match galaxy.spec()");
    }
    if (!std::ranges::equal(readSpecExtinct, galaxy.specExtinct()))
    {
        throw std::runtime_error("galaxy_spectra spec_extinct row does not match "
            "galaxy.specExtinct()");
    }
}

// Check that the cluster_spectra group holds exactly one row --
// matching spec() -- for every cluster in galaxy.clusters(), written
// by writeGalaxySpec()'s own per-cluster writeClusterSpec() calls.
// Throws std::runtime_error (caught by the caller) describing any
// mismatch.
static void checkClusterSpectraPassthroughH5(const hid_t file, core::Galaxy& galaxy)
{
    // NOLINTBEGIN(misc-include-cleaner)
    const hid_t grp = H5Gopen2(file, "cluster_spectra", H5P_DEFAULT);
    if (grp < 0) { throw std::runtime_error("missing cluster_spectra group"); }
    const auto uidCol = readColumnULong(grp, "uid");
    H5Gclose(grp);
    // NOLINTEND(misc-include-cleaner)

    if (uidCol.size() != galaxy.clusters().size())
    {
        throw std::runtime_error("cluster_spectra group has " +
            std::to_string(uidCol.size()) + " rows, expected " +
            std::to_string(galaxy.clusters().size()));
    }
    for (auto& cluster : galaxy.clusters())
    {
        const auto i = findIndex(uidCol, cluster.uid());
        // NOLINTBEGIN(misc-include-cleaner)
        const hid_t grp2 = H5Gopen2(file, "cluster_spectra", H5P_DEFAULT);
        const auto readSpec = readRow2d(grp2, "spec", i, cluster.spec().size());
        H5Gclose(grp2);
        // NOLINTEND(misc-include-cleaner)
        if (!std::ranges::equal(readSpec, cluster.spec()))
        {
            throw std::runtime_error("cluster_spectra row for uid " +
                std::to_string(cluster.uid()) + " does not match spec()");
        }
    }
}

// Check the galaxy_phot group's own row (written by writeGalaxyPhot)
// matches galaxy.phot() (with lbol() appended, if requested) and
// galaxy.photExtinct(). Throws std::runtime_error (caught by the
// caller) describing any mismatch.
static void checkGalaxyPhotRowH5(const hid_t file, const io::SimControls& controls,
    core::Galaxy& galaxy)
{
    auto expectedPhot = galaxy.phot();
    if (controls.computeLbol()) { expectedPhot.push_back(galaxy.lbol()); }

    // NOLINTBEGIN(misc-include-cleaner)
    const hid_t grp = H5Gopen2(file, "galaxy_phot", H5P_DEFAULT);
    if (grp < 0) { throw std::runtime_error("missing galaxy_phot group"); }
    const auto readPhot = readRow2d(grp, "phot", 0, expectedPhot.size());
    const auto readPhotExtinct = readRow2d(grp, "phot_extinct", 0, galaxy.photExtinct().size());
    H5Gclose(grp);
    // NOLINTEND(misc-include-cleaner)

    if (!std::ranges::equal(readPhot, expectedPhot))
    {
        throw std::runtime_error("galaxy_phot phot row does not match galaxy.phot()/lbol()");
    }
    if (!std::ranges::equal(readPhotExtinct, galaxy.photExtinct()))
    {
        throw std::runtime_error("galaxy_phot phot_extinct row does not match "
            "galaxy.photExtinct()");
    }
}

// Check that the cluster_phot group holds exactly one row -- matching
// phot() (with lbol() appended, if requested) -- for every cluster in
// galaxy.clusters(), written by writeGalaxyPhot()'s own per-cluster
// writeClusterPhot() calls. Throws std::runtime_error (caught by the
// caller) describing any mismatch.
static void checkClusterPhotPassthroughH5(const hid_t file,
    const io::SimControls& controls, core::Galaxy& galaxy)
{
    // NOLINTBEGIN(misc-include-cleaner)
    const hid_t grp = H5Gopen2(file, "cluster_phot", H5P_DEFAULT);
    if (grp < 0) { throw std::runtime_error("missing cluster_phot group"); }
    const auto uidCol = readColumnULong(grp, "uid");
    H5Gclose(grp);
    // NOLINTEND(misc-include-cleaner)

    if (uidCol.size() != galaxy.clusters().size())
    {
        throw std::runtime_error("cluster_phot group has " +
            std::to_string(uidCol.size()) + " rows, expected " +
            std::to_string(galaxy.clusters().size()));
    }
    for (auto& cluster : galaxy.clusters())
    {
        const auto i = findIndex(uidCol, cluster.uid());
        auto expectedPhot = cluster.phot();
        if (controls.computeLbol()) { expectedPhot.push_back(cluster.lbol()); }
        // NOLINTBEGIN(misc-include-cleaner)
        const hid_t grp2 = H5Gopen2(file, "cluster_phot", H5P_DEFAULT);
        const auto readPhot = readRow2d(grp2, "phot", i, expectedPhot.size());
        H5Gclose(grp2);
        // NOLINTEND(misc-include-cleaner)
        if (!std::ranges::equal(readPhot, expectedPhot))
        {
            throw std::runtime_error("cluster_phot row for uid " +
                std::to_string(cluster.uid()) + " does not match phot()/lbol()");
        }
    }
}

// Verify that OutputManagerH5::writeGalaxy/writeGalaxySpec/
// writeGalaxyPhot write correct galaxy-level rows (matching the
// Galaxy object's own targetMass()/actualMass()/spec()/specExtinct()/
// phot()/photExtinct()), and that each also writes a matching
// passthrough row, via its own internal writeCluster()/
// writeClusterSpec()/writeClusterPhot() calls, for every currently-
// alive cluster in the galaxy -- see checkGalaxyRowH5()/
// checkClustersPassthroughH5()/etc.'s own docstrings for exactly what
// is checked.
static auto testWriteGalaxyH5() -> int
{
    const auto outDir = std::filesystem::temp_directory_path() / "slugTestOutputManagerWriteGalaxyH5";
    std::filesystem::remove_all(outDir);
    std::filesystem::create_directories(outDir);
    const std::string modelName = "test_model";
    const auto expectedPath = outDir / (modelName + ".h5");
    const toml::table inputDeck = makeGalaxyPhysicsInputDeck(modelName, outDir);

    try
    {
        const io::SimControls controls(inputDeck);
        if (controls.specsyn() == nullptr || controls.filters() == nullptr ||
            controls.extinct() == nullptr)
        {
            std::cerr << "testOutputManager: write galaxy h5: test bug: "
                "expected specsyn()/filters()/extinct() to all be non-null\n";
            return 1;
        }

        utils::rng().seed(42);
        core::Galaxy galaxy(controls);
        galaxy.advance(galaxyWriteTime);
        if (galaxy.clusters().empty())
        {
            std::cerr << "testOutputManager: write galaxy h5: test bug: "
                "expected at least one cluster to have formed\n";
            return 1;
        }
        constexpr unsigned long trial = 6;

        {
            io::OutputManagerH5 manager(controls, inputDeck);
            manager.writeGalaxy(trial, galaxyWriteTime, galaxy);
            manager.writeGalaxySpec(trial, galaxyWriteTime, galaxy);
            manager.writeGalaxyPhot(trial, galaxyWriteTime, galaxy);
        }

        // NOLINTBEGIN(misc-include-cleaner)
        const hid_t file = H5Fopen(expectedPath.string().c_str(),
            H5F_ACC_RDONLY, H5P_DEFAULT);
        if (file < 0)
        {
            std::cerr << "testOutputManager: write galaxy h5: unable to reopen "
                << expectedPath.string() << "\n";
            return 1;
        }

        try
        {
            checkGalaxyRowH5(file, trial, galaxyWriteTime, galaxy);
            checkClustersPassthroughH5(file, galaxy);
            checkGalaxySpecRowH5(file, galaxy);
            checkClusterSpectraPassthroughH5(file, galaxy);
            checkGalaxyPhotRowH5(file, controls, galaxy);
            checkClusterPhotPassthroughH5(file, controls, galaxy);
        }
        catch (const std::runtime_error& error)
        {
            H5Fclose(file);
            std::cerr << "testOutputManager: write galaxy h5: " << error.what() << "\n";
            return 1;
        }
        H5Fclose(file);
        // NOLINTEND(misc-include-cleaner)
    }
    catch (const std::exception& error)
    {
        std::cerr << "testOutputManager: write galaxy h5 test failed: "
            << error.what() << "\n";
        return 1;
    }
    return 0;
}

// Skip an ascii output file's 3-line header (column names, units,
// dashed rule), leaving file positioned at its first data line
static void skipAsciiHeader(std::ifstream& file)
{
    std::string line;
    std::getline(file, line);
    std::getline(file, line);
    std::getline(file, line);
}

// Check the _galaxy.txt file's own single data row (trial, time,
// target_mass, actual_mass) matches trial/time/galaxy.targetMass()/
// galaxy.actualMass(), within tol. Throws std::runtime_error (caught
// by the caller) describing any mismatch.
static void checkGalaxyRowAscii(const std::filesystem::path& path,
    const unsigned long trial, const double time, core::Galaxy& galaxy,
    const double tol)
{
    std::ifstream file(path);
    skipAsciiHeader(file);
    std::string line;
    std::getline(file, line);

    std::istringstream lineStream(line);
    unsigned long readTrial = 0;
    double readTime = 0.0;
    double readTargetMass = 0.0;
    double readActualMass = 0.0;
    lineStream >> readTrial >> readTime >> readTargetMass >> readActualMass;

    if (readTrial != trial || std::abs(readTime - time) > tol ||
        std::abs(readTargetMass - galaxy.targetMass()) > tol * galaxy.targetMass() ||
        std::abs(readActualMass - galaxy.actualMass()) > tol * galaxy.actualMass())
    {
        throw std::runtime_error("_galaxy.txt row does not match: " + line);
    }
}

// Check that _clusters.txt holds exactly one row -- matching
// target_mass -- for every cluster in galaxy.clusters(), found by
// uid. Throws std::runtime_error (caught by the caller) describing
// any mismatch.
static void checkClustersPassthroughAscii(const std::filesystem::path& path,
    core::Galaxy& galaxy, const double tol)
{
    std::ifstream file(path);
    skipAsciiHeader(file);

    std::map<unsigned long, double> targetMassByUid;
    std::string line;
    while (std::getline(file, line))
    {
        if (line.empty()) { continue; }
        std::istringstream lineStream(line);
        unsigned long readTrial = 0;
        unsigned long readUid = 0;
        double readTargetMass = 0.0;
        lineStream >> readTrial >> readUid >> readTargetMass;
        targetMassByUid[readUid] = readTargetMass;
    }

    if (targetMassByUid.size() != galaxy.clusters().size())
    {
        throw std::runtime_error("_clusters.txt has " +
            std::to_string(targetMassByUid.size()) + " rows, expected " +
            std::to_string(galaxy.clusters().size()));
    }
    for (const auto& cluster : galaxy.clusters())
    {
        const auto it = targetMassByUid.find(cluster.uid());
        if (it == targetMassByUid.end() ||
            std::abs(it->second - cluster.targetMass()) > tol * cluster.targetMass())
        {
            throw std::runtime_error("_clusters.txt row for uid " +
                std::to_string(cluster.uid()) + " missing or does not match target_mass");
        }
    }
}

// Check that _galaxy_spectra.txt holds exactly wlObs.size() data rows,
// each one's wl/spec columns matching wlObs/galaxy.spec() in order (a
// single galaxy/time combination lays out sequentially, unlike the
// per-cluster files below, so no need to match rows by uid). Throws
// std::runtime_error (caught by the caller) describing any mismatch.
static void checkGalaxySpecRowsAscii(const std::filesystem::path& path,
    const std::vector<double>& wlObs, core::Galaxy& galaxy, const double tol)
{
    std::ifstream file(path);
    skipAsciiHeader(file);

    const auto& spec = galaxy.spec();
    std::string line;
    std::size_t i = 0;
    while (std::getline(file, line))
    {
        if (line.empty()) { continue; }
        std::istringstream lineStream(line);
        unsigned long readTrial = 0;
        double readTime = 0.0;
        double readWl = 0.0;
        double readSpec = 0.0;
        lineStream >> readTrial >> readTime >> readWl >> readSpec;

        if (std::abs(readWl - wlObs.at(i)) > tol * wlObs.at(i) ||
            std::abs(readSpec - spec.at(i)) > tol * std::max(std::abs(spec.at(i)), 1.0))
        {
            throw std::runtime_error("_galaxy_spectra.txt row " + std::to_string(i) +
                " does not match: " + line);
        }
        ++i;
    }
    if (i != wlObs.size())
    {
        throw std::runtime_error("_galaxy_spectra.txt has " + std::to_string(i) +
            " rows, expected " + std::to_string(wlObs.size()));
    }
}

// Check that _cluster_spectra.txt holds exactly
// clusters().size() * wlObs.size() data rows -- one block of
// wlObs.size() consecutive rows per cluster in galaxy.clusters(),
// written by writeGalaxySpec()'s own per-cluster writeClusterSpec()
// calls. Only checks the row count (the per-cluster ascii spectrum
// layout itself is already covered by testWriteClusterAsciiExtinct());
// throws std::runtime_error (caught by the caller) if it does not
// match.
static void checkClusterSpectraPassthroughAscii(const std::filesystem::path& path,
    const std::vector<double>& wlObs, core::Galaxy& galaxy)
{
    std::ifstream file(path);
    skipAsciiHeader(file);
    std::size_t nRows = 0;
    std::string line;
    while (std::getline(file, line)) { if (!line.empty()) { ++nRows; } }

    const auto expected = galaxy.clusters().size() * wlObs.size();
    if (nRows != expected)
    {
        throw std::runtime_error("_cluster_spectra.txt has " + std::to_string(nRows) +
            " rows, expected " + std::to_string(expected));
    }
}

// Check the _galaxy_phot.txt file's own single data row matches
// galaxy.phot() (with lbol() appended, if requested), within tol.
// Throws std::runtime_error (caught by the caller) describing any
// mismatch.
static void checkGalaxyPhotRowAscii(const std::filesystem::path& path,
    const io::SimControls& controls, core::Galaxy& galaxy, const double tol)
{
    auto expectedPhot = galaxy.phot();
    if (controls.computeLbol()) { expectedPhot.push_back(galaxy.lbol()); }

    std::ifstream file(path);
    skipAsciiHeader(file);
    std::string line;
    std::getline(file, line);

    std::istringstream lineStream(line);
    unsigned long readTrial = 0;
    double readTime = 0.0;
    lineStream >> readTrial >> readTime;
    for (const double expected : expectedPhot)
    {
        double readValue = 0.0;
        lineStream >> readValue;
        if (std::abs(readValue - expected) > tol * std::max(std::abs(expected), 1.0))
        {
            throw std::runtime_error("_galaxy_phot.txt row does not match: " + line);
        }
    }
}

// Check that _cluster_phot.txt holds exactly one data row per cluster
// in galaxy.clusters(), written by writeGalaxyPhot()'s own per-cluster
// writeClusterPhot() calls. Only checks the row count (the per-cluster
// ascii photometry layout itself is already covered by
// testWriteClusterAsciiExtinct()); throws std::runtime_error (caught
// by the caller) if it does not match.
static void checkClusterPhotPassthroughAscii(const std::filesystem::path& path,
    core::Galaxy& galaxy)
{
    std::ifstream file(path);
    skipAsciiHeader(file);
    std::size_t nRows = 0;
    std::string line;
    while (std::getline(file, line)) { if (!line.empty()) { ++nRows; } }

    if (nRows != galaxy.clusters().size())
    {
        throw std::runtime_error("_cluster_phot.txt has " + std::to_string(nRows) +
            " rows, expected " + std::to_string(galaxy.clusters().size()));
    }
}

// Verify that OutputManagerAscii::writeGalaxy/writeGalaxySpec/
// writeGalaxyPhot write correct galaxy-level rows (matching the
// Galaxy object's own targetMass()/actualMass()/spec()/phot()), and
// that each also writes a passthrough row (or, for the two spectra/
// phot files, the right total row count) via its own internal
// writeCluster()/writeClusterSpec()/writeClusterPhot() calls, for
// every currently-alive cluster in the galaxy -- mirrors
// testWriteGalaxyH5()'s own checks for the ascii text layout; see
// checkGalaxyRowAscii()/checkClustersPassthroughAscii()/etc.'s own
// docstrings for exactly what each checks.
static auto testWriteGalaxyAscii() -> int
{
    const auto outDir = std::filesystem::temp_directory_path() / "slugTestOutputManagerWriteGalaxyAscii";
    std::filesystem::remove_all(outDir);
    std::filesystem::create_directories(outDir);
    const std::string modelName = "test_model";
    const toml::table inputDeck = makeGalaxyPhysicsInputDeck(modelName, outDir);
    constexpr double tol = 1e-5;

    try
    {
        const io::SimControls controls(inputDeck);
        if (controls.specsyn() == nullptr || controls.filters() == nullptr ||
            controls.extinct() == nullptr)
        {
            std::cerr << "testOutputManager: write galaxy ascii: test bug: "
                "expected specsyn()/filters()/extinct() to all be non-null\n";
            return 1;
        }

        utils::rng().seed(42);
        core::Galaxy galaxy(controls);
        galaxy.advance(galaxyWriteTime);
        if (galaxy.clusters().empty())
        {
            std::cerr << "testOutputManager: write galaxy ascii: test bug: "
                "expected at least one cluster to have formed\n";
            return 1;
        }
        constexpr unsigned long trial = 9;

        {
            io::OutputManagerAscii manager(controls, inputDeck);
            manager.writeGalaxy(trial, galaxyWriteTime, galaxy);
            manager.writeGalaxySpec(trial, galaxyWriteTime, galaxy);
            manager.writeGalaxyPhot(trial, galaxyWriteTime, galaxy);
        }

        const auto wlObs = controls.specsyn()->wlObs();

        try
        {
            checkGalaxyRowAscii(outDir / (modelName + "_galaxy.txt"),
                trial, galaxyWriteTime, galaxy, tol);
            checkClustersPassthroughAscii(outDir / (modelName + "_clusters.txt"), galaxy, tol);
            checkGalaxySpecRowsAscii(outDir / (modelName + "_galaxy_spectra.txt"),
                wlObs, galaxy, tol);
            checkClusterSpectraPassthroughAscii(
                outDir / (modelName + "_cluster_spectra.txt"), wlObs, galaxy);
            checkGalaxyPhotRowAscii(outDir / (modelName + "_galaxy_phot.txt"),
                controls, galaxy, tol);
            checkClusterPhotPassthroughAscii(
                outDir / (modelName + "_cluster_phot.txt"), galaxy);
        }
        catch (const std::runtime_error& error)
        {
            std::cerr << "testOutputManager: write galaxy ascii: " << error.what() << "\n";
            return 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testOutputManager: write galaxy ascii test failed: "
            << error.what() << "\n";
        return 1;
    }
    return 0;
}

// ---------------------------------------------------------------------
// Output-control tests: output.write_cluster/write_cluster_phot/
// write_galaxy/write_galaxy_spec/write_galaxy_phot (write_cluster_spec
// is covered above by testOptOutClusterSpecOutput) and the two sanity
// checks OutputManager's own constructor runs on the combination of
// all six -- see checkOptOutSuppressesGroup()'s own docstring for the
// shared implementation these opt-out tests build on.
// ---------------------------------------------------------------------

// Verify that setting output.<key> to false in inputDeck, then
// constructing both an OutputManagerAscii and an OutputManagerH5 from
// it, creates neither the ascii <modelName><asciiSuffix> file nor the
// H5 <h5GroupName> group -- mirrors testOptOutClusterSpecOutput()'s
// own logic, factored out so each individual key can be checked with a
// one-line test function below. Returns 0 on success, 1 (after
// printing a diagnostic prefixed with label) on any mismatch or
// unexpected exception.
static auto checkOptOutSuppressesGroup(const toml::table& inputDeck,
    const std::filesystem::path& outDir, const std::string& modelName,
    const std::string& h5GroupName, const std::string& asciiSuffix,
    const std::string& label) -> int
{
    try
    {
        const io::SimControls controls(inputDeck);
        {
            const io::OutputManagerAscii manager(controls, inputDeck);
        }
        {
            const io::OutputManagerH5 manager(controls, inputDeck);
        }

        const auto asciiPath = outDir / (modelName + asciiSuffix);
        if (std::filesystem::exists(asciiPath))
        {
            std::cerr << "testOutputManager: " << label << ": ascii "
                "unexpectedly wrote " << asciiPath.string() << "\n";
            return 1;
        }

        const auto h5Path = outDir / (modelName + ".h5");
        // NOLINTBEGIN(misc-include-cleaner)
        const hid_t file = H5Fopen(h5Path.string().c_str(),
            H5F_ACC_RDONLY, H5P_DEFAULT);
        if (file < 0)
        {
            std::cerr << "testOutputManager: " << label << ": unable to reopen "
                << h5Path.string() << "\n";
            return 1;
        }
        const bool hasGroup = H5Lexists(file, h5GroupName.c_str(), H5P_DEFAULT) > 0;
        H5Fclose(file);
        // NOLINTEND(misc-include-cleaner)

        if (hasGroup)
        {
            std::cerr << "testOutputManager: " << label << ": h5 unexpectedly "
                "created a " << h5GroupName << " group\n";
            return 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testOutputManager: " << label << " test failed: "
            << error.what() << "\n";
        return 1;
    }
    return 0;
}

// Verify that output.write_cluster = false suppresses the clusters H5
// group and the ascii _clusters.txt file, for a cluster-type
// simulation with spectral synthesis still enabled (so write_cluster
// being the only output disabled doesn't also trip the "every relevant
// output is false" sanity check).
static auto testOptOutClusterOutput() -> int
{
    const auto outDir = std::filesystem::temp_directory_path() / "slugTestOutputManagerNoClusterOutput";
    std::filesystem::remove_all(outDir);
    std::filesystem::create_directories(outDir);
    const std::string modelName = "test_model";
    toml::table inputDeck = makeClusterPhysicsInputDeck(modelName, outDir);
    inputDeck.at_path("output").as_table()->insert("write_cluster", false);
    return checkOptOutSuppressesGroup(inputDeck, outDir, modelName,
        "clusters", "_clusters.txt", "opt-out cluster output");
}

// Verify that output.write_cluster_phot = false suppresses the
// cluster_phot H5 group and the ascii _cluster_phot.txt file, for a
// cluster-type simulation with a filter collection requested (so there
// would be something to write if the key were not set to false).
static auto testOptOutClusterPhotOutput() -> int
{
    const auto outDir = std::filesystem::temp_directory_path() / "slugTestOutputManagerNoClusterPhotOutput";
    std::filesystem::remove_all(outDir);
    std::filesystem::create_directories(outDir);
    const std::string modelName = "test_model";
    toml::table inputDeck = makeClusterPhysicsInputDeck(
        modelName, outDir, "tests/core/assets/testClusterPhot.in");
    inputDeck.at_path("output").as_table()->insert("write_cluster_phot", false);
    return checkOptOutSuppressesGroup(inputDeck, outDir, modelName,
        "cluster_phot", "_cluster_phot.txt", "opt-out cluster phot output");
}

// Verify that output.write_galaxy = false suppresses the galaxy H5
// group and the ascii _galaxy.txt file, for a galaxy-type simulation.
static auto testOptOutGalaxyOutput() -> int
{
    const auto outDir = std::filesystem::temp_directory_path() / "slugTestOutputManagerNoGalaxyOutput";
    std::filesystem::remove_all(outDir);
    std::filesystem::create_directories(outDir);
    const std::string modelName = "test_model";
    toml::table inputDeck = makeGalaxyPhysicsInputDeck(modelName, outDir);
    inputDeck.at_path("output").as_table()->insert("write_galaxy", false);
    return checkOptOutSuppressesGroup(inputDeck, outDir, modelName,
        "galaxy", "_galaxy.txt", "opt-out galaxy output");
}

// Verify that output.write_galaxy_spec = false suppresses the
// galaxy_spectra H5 group and the ascii _galaxy_spectra.txt file, for
// a galaxy-type simulation with spectral synthesis requested.
static auto testOptOutGalaxySpecOutput() -> int
{
    const auto outDir = std::filesystem::temp_directory_path() / "slugTestOutputManagerNoGalaxySpecOutput";
    std::filesystem::remove_all(outDir);
    std::filesystem::create_directories(outDir);
    const std::string modelName = "test_model";
    toml::table inputDeck = makeGalaxyPhysicsInputDeck(modelName, outDir);
    inputDeck.at_path("output").as_table()->insert("write_galaxy_spec", false);
    return checkOptOutSuppressesGroup(inputDeck, outDir, modelName,
        "galaxy_spectra", "_galaxy_spectra.txt", "opt-out galaxy spec output");
}

// Verify that output.write_galaxy_phot = false suppresses the
// galaxy_phot H5 group and the ascii _galaxy_phot.txt file, for a
// galaxy-type simulation with a filter collection requested.
static auto testOptOutGalaxyPhotOutput() -> int
{
    const auto outDir = std::filesystem::temp_directory_path() / "slugTestOutputManagerNoGalaxyPhotOutput";
    std::filesystem::remove_all(outDir);
    std::filesystem::create_directories(outDir);
    const std::string modelName = "test_model";
    toml::table inputDeck = makeGalaxyPhysicsInputDeck(modelName, outDir);
    inputDeck.at_path("output").as_table()->insert("write_galaxy_phot", false);
    return checkOptOutSuppressesGroup(inputDeck, outDir, modelName,
        "galaxy_phot", "_galaxy_phot.txt", "opt-out galaxy phot output");
}

// Verify that constructing an OutputManager (via either subclass) from
// a galaxy-type deck with all six output.write_* keys set to false
// throws std::runtime_error, since nothing at all would ever be
// written -- sanity check 1 in OutputManager's own constructor.
static auto testAllOutputsFalseThrows() -> int
{
    const auto outDir = std::filesystem::temp_directory_path() / "slugTestOutputManagerAllOutputsFalse";
    std::filesystem::remove_all(outDir);
    std::filesystem::create_directories(outDir);
    const std::string modelName = "test_model";
    toml::table inputDeck = makeGalaxyPhysicsInputDeck(modelName, outDir);
    toml::table* outputTbl = inputDeck.at_path("output").as_table();
    for (const std::string& key : { "write_cluster", "write_cluster_spec",
        "write_cluster_phot", "write_galaxy", "write_galaxy_spec", "write_galaxy_phot" })
    {
        outputTbl->insert(key, false);
    }

    try
    {
        const io::SimControls controls(inputDeck);
        const io::OutputManagerH5 manager(controls, inputDeck);
        std::cerr << "testOutputManager: all outputs false: expected "
            "construction to throw, but it succeeded\n";
        return 1;
    }
    catch (const std::runtime_error&)
    {
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "testOutputManager: all outputs false test failed "
            "with an unexpected exception type: " << error.what() << "\n";
        return 1;
    }
}

// Verify that constructing an OutputManager from a cluster-type deck
// with write_cluster/write_cluster_spec/write_cluster_phot all set to
// false also throws -- confirming sanity check 1 correctly excludes
// the (irrelevant, still-defaulted-true) write_galaxy* keys when
// deciding whether "every output relevant to this simulation" is
// false, rather than only ever checking the literal six regardless of
// SimType.
static auto testAllClusterOutputsFalseThrows() -> int
{
    const auto outDir = std::filesystem::temp_directory_path() / "slugTestOutputManagerAllClusterOutputsFalse";
    std::filesystem::remove_all(outDir);
    std::filesystem::create_directories(outDir);
    const std::string modelName = "test_model";
    toml::table inputDeck = makeClusterPhysicsInputDeck(modelName, outDir);
    toml::table* outputTbl = inputDeck.at_path("output").as_table();
    for (const std::string& key : { "write_cluster", "write_cluster_spec", "write_cluster_phot" })
    {
        outputTbl->insert(key, false);
    }

    try
    {
        const io::SimControls controls(inputDeck);
        const io::OutputManagerAscii manager(controls, inputDeck);
        std::cerr << "testOutputManager: all cluster outputs false: expected "
            "construction to throw, but it succeeded\n";
        return 1;
    }
    catch (const std::runtime_error&)
    {
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "testOutputManager: all cluster outputs false test failed "
            "with an unexpected exception type: " << error.what() << "\n";
        return 1;
    }
}

// Verify that constructing an OutputManager from a deck with
// phot.filters given, but both write_cluster_phot and write_galaxy_phot
// set to false, throws -- sanity check 2 in OutputManager's own
// constructor, catching the case where photometry was requested but
// would never be written anywhere.
static auto testPhotWithoutOutputThrows() -> int
{
    const auto outDir = std::filesystem::temp_directory_path() / "slugTestOutputManagerPhotWithoutOutput";
    std::filesystem::remove_all(outDir);
    std::filesystem::create_directories(outDir);
    const std::string modelName = "test_model";
    toml::table inputDeck = makeGalaxyPhysicsInputDeck(modelName, outDir);
    toml::table* outputTbl = inputDeck.at_path("output").as_table();
    outputTbl->insert("write_cluster_phot", false);
    outputTbl->insert("write_galaxy_phot", false);

    try
    {
        const io::SimControls controls(inputDeck);
        if (controls.filters() == nullptr)
        {
            std::cerr << "testOutputManager: phot without output: test bug: "
                "expected SimControls::filters() to be non-null\n";
            return 1;
        }
        const io::OutputManagerH5 manager(controls, inputDeck);
        std::cerr << "testOutputManager: phot without output: expected "
            "construction to throw, but it succeeded\n";
        return 1;
    }
    catch (const std::runtime_error&)
    {
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "testOutputManager: phot without output test failed "
            "with an unexpected exception type: " << error.what() << "\n";
        return 1;
    }
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
    result += testOptOutClusterSpecOutput();
    result += testGalaxyGroupsH5();
    result += testGalaxyGroupsAbsentH5();
    result += testGalaxyFilesAscii();
    result += testGalaxyFilesAbsentAscii();
    result += testWriteGalaxyH5();
    result += testWriteGalaxyAscii();
    result += testOptOutClusterOutput();
    result += testOptOutClusterPhotOutput();
    result += testOptOutGalaxyOutput();
    result += testOptOutGalaxySpecOutput();
    result += testOptOutGalaxyPhotOutput();
    result += testAllOutputsFalseThrows();
    result += testAllClusterOutputsFalseThrows();
    result += testPhotWithoutOutputThrows();
    return result;
}
