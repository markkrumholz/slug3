/**
 * @file testTracks2D.hpp
 * @author Mark Krumholz
 * @brief Unit tests for the Tracks2D class.
 * @details
 * This file contains unit tests for the Tracks2D class, which reads a
 * group of stellar evolutionary tracks from an HDF5 file. The tests
 * cover construction of a Tracks2D object from one group of each of
 * the supported track files.
 * @date 2024-07-09
 */

#ifndef TESTTRACKS2D_HPP
#define TESTTRACKS2D_HPP

#include "../../src/tracks/Tracks2D.hpp"
#include "trackFieldFixture.hpp"
#include "hdf5.h"  // NOLINT(misc-include-cleaner)
#include <array>
#include <cmath>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

/**
 * @brief Outcome of attempting to test a single track file
 */
enum class TestTracks2DOutcome
{
    notFound, /**< The file does not exist, so the test was skipped */
    failed,   /**< The file exists, but the test failed */
    passed    /**< The file exists, and the test passed */
};

/**
 * @brief Identifies a track set to test, and the registry it lives in
 */
struct TestTracks2DCase
{
    std::string h5Path;       /**< Path to the HDF5 file, used only to check existence */
    std::string trackName;    /**< Name of the track set within the registry */
    std::string registryName; /**< Registry file containing trackName */
};

// Suppress clang-tidy warnings iun this namespace caused by just including
// hdf5.h, instead of the individual HDF5 headers, since this is the paradigm
// that HDF5 wants
// NOLINTBEGIN(misc-include-cleaner)

/**
 * @brief Attempt to construct a Tracks2D object from one group of a track file
 * @param testCase Identifies the track file, track set, and registry to test
 * @return The outcome of the attempt
 * @details
 * If the file does not exist, this function returns notFound without
 * attempting to open it. Otherwise, it opens the file directly
 * (independently of Tracks2D) to find the first group at its root
 * level and read that group's feh, vvcrit, and afe attributes (the
 * choice of group does not matter for this test, and any attribute
 * that is absent is read as 0.0, since findTrack does not use absent
 * attributes to constrain the match anyway), then uses those values to
 * construct a Tracks2D object via the trackName/registryName-based
 * constructor, returning passed or failed depending on whether
 * construction succeeds.
 */
inline auto testTracks2DFile(const TestTracks2DCase& testCase) -> TestTracks2DOutcome
{
    if (!std::filesystem::exists(testCase.h5Path))
    {
        std::cerr << "testTracks2D: file " << testCase.h5Path
            << " not found, skipping\n";
        return TestTracks2DOutcome::notFound;
    }

    const hid_t file = H5Fopen(testCase.h5Path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0)
    {
        std::cerr << "testTracks2D: unable to open file " << testCase.h5Path << "\n";
        return TestTracks2DOutcome::failed;
    }

    // Find the name of the first group at the root of the file
    const auto nameLen = H5Lget_name_by_idx(file, ".", H5_INDEX_NAME,
        H5_ITER_INC, 0, nullptr, 0, H5P_DEFAULT);
    if (nameLen < 0)
    {
        std::cerr << "testTracks2D: unable to find a group in "
            << testCase.h5Path << "\n";
        H5Fclose(file);
        return TestTracks2DOutcome::failed;
    }
    std::vector<char> nameBuf(static_cast<size_t>(nameLen) + 1);
    H5Lget_name_by_idx(file, ".", H5_INDEX_NAME, H5_ITER_INC, 0,
        nameBuf.data(), nameBuf.size(), H5P_DEFAULT);
    const std::string groupName(nameBuf.data());

    const hid_t grp = H5Gopen2(file, groupName.c_str(), H5P_DEFAULT);
    if (grp < 0)
    {
        std::cerr << "testTracks2D: unable to open group " << groupName
            << " in " << testCase.h5Path << "\n";
        H5Fclose(file);
        return TestTracks2DOutcome::failed;
    }

    // Read the group's feh, vvcrit, and afe attributes, defaulting
    // absent ones to 0.0 (a value findTrack ignores when the
    // corresponding attribute isn't present in the group it inspects)
    const auto readAttr = [grp](const char* name) -> double
    {
        if (H5Aexists(grp, name) <= 0) { return 0.0; }
        const hid_t attr = H5Aopen(grp, name, H5P_DEFAULT);
        if (attr < 0) { return 0.0; }
        double value = 0.0;
        H5Aread(attr, H5T_NATIVE_DOUBLE, &value);
        H5Aclose(attr);
        return value;
    };
    const double feh = readAttr("feh");
    const double vvcrit = readAttr("vvcrit");
    const double afe = readAttr("afe");

    H5Gclose(grp);
    H5Fclose(file);

    auto outcome = TestTracks2DOutcome::passed;
    try
    {
        const tracks::Tracks2D tracks2d(
            testCase.trackName, feh, vvcrit, afe, testCase.registryName);
    }
    catch (const std::exception& e)
    {
        std::cerr << "testTracks2D: failed to construct Tracks2D from "
            << testCase.h5Path << ", group " << groupName << ": "
            << e.what() << "\n";
        outcome = TestTracks2DOutcome::failed;
    }

    return outcome;
}

/**
 * @brief Unit test for the Tracks2D feH(), aFe(), and vVcrit() getters.
 * @return 0 if the test passes, 1 if it fails.
 * @details
 * This function constructs a Tracks2D object from the
 * feh_-0.25_afe_-0.2_vvcrit_0.00 group of the MIST_test track set in
 * tests/tracks/assets/tracks.toml, whose [Fe/H], [alpha/Fe], and
 * v/vcrit values are known exactly from the group name, and checks
 * that feH(), aFe(), and vVcrit() report those values.
 */
inline auto testTracks2DGetters() -> int
{
    const std::string registryName = "tests/tracks/assets/tracks.toml";
    const std::string trackName = "MIST_test";

    int result = 0;
    try
    {
        const tracks::Tracks2D tracks2d(trackName, -0.25, 0.0, -0.2, registryName);
        if (tracks2d.feH() != -0.25 || tracks2d.aFe() != -0.2 ||
            tracks2d.vVcrit() != 0.0)
        {
            std::cerr << "testTracks2DGetters: expected feH=-0.25, "
                "aFe=-0.2, vVcrit=0.0, got feH=" << tracks2d.feH()
                << ", aFe=" << tracks2d.aFe() << ", vVcrit="
                << tracks2d.vVcrit() << "\n";
            result = 1;
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "testTracks2DGetters: failed to construct Tracks2D "
            "from track set " << trackName << ": " << e.what() << "\n";
        result = 1;
    }

    return result;
}

/**
 * @brief Regression test that getTrack() returns fields in the
 *   correct order.
 * @return 0 if the test passes, 1 if it fails.
 * @details
 * This is a regression test for a bug in which the columns of field
 * data read from an HDF5 track file were mismatched against their
 * canonical tracks::FieldIdx order, because the age column (present
 * in every track dataset, but not one of the nQty tracked
 * quantities) was not properly accounted for when mapping canonical
 * field index to on-disk column. It constructs a Tracks2D object from
 * the feh_0.00_afe_-0.2_vvcrit_0.00 group of the MIST_test track set
 * in tests/tracks/assets/tracks.toml, calls getTrack() for mass = 5.0
 * (an exact point on that group's mass grid), and evaluates the
 * result at the exact age of an arbitrary interior row of the raw
 * track_m5.000 dataset. Since mass = 5.0 is on the mesh's mass grid
 * and the query age is on that mass's own age grid, the interpolated
 * result should reproduce the raw row exactly (up to floating-point
 * round-off); the raw row itself is read independently of the
 * Tracks2D field-mapping logic via trackFieldFixture.hpp, so this
 * test does not depend on that logic being correct to establish its
 * expectations.
 */
inline auto testTracks2DFieldOrder() -> int
{
    const std::string registryName = "tests/tracks/assets/tracks.toml";
    const std::string trackName = "MIST_test";
    const std::string h5Path = "tests/tracks/assets/MIST_test.h5";
    const std::string groupName = "feh_0.00_afe_-0.2_vvcrit_0.00";
    constexpr double feh = 0.0;
    constexpr double vvcrit = 0.0;
    constexpr double afe = -0.2;
    constexpr double mass = 5.0;
    constexpr size_t rowIdx = 500;

    const hid_t file = H5Fopen(h5Path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0)
    {
        std::cerr << "testTracks2DFieldOrder: unable to open file "
            << h5Path << "\n";
        return 1;
    }
    const hid_t grp = H5Gopen2(file, groupName.c_str(), H5P_DEFAULT);
    if (grp < 0)
    {
        std::cerr << "testTracks2DFieldOrder: unable to open group "
            << groupName << " in " << h5Path << "\n";
        H5Fclose(file);
        return 1;
    }

    int result = 0;
    try
    {
        const auto [age, expected] = testutil::readRawFields(grp, mass, rowIdx);

        const tracks::Tracks2D tracks2d(trackName, feh, vvcrit, afe, registryName);
        const auto track = tracks2d.getTrack(mass);
        if (!track)
        {
            std::cerr << "testTracks2DFieldOrder: getTrack(" << mass
                << ") returned null\n";
            result = 1;
        }
        else
        {
            const auto actual = (*track)(std::log10(age));
            for (size_t k = 0; k < testutil::nQty; ++k)
            {
                if (!testutil::fieldsMatch(actual.at(k), expected.at(k)))
                {
                    std::cerr << "testTracks2DFieldOrder: field "
                        << tracks::fieldStr.at(k) << " (index " << k
                        << ") mismatch: expected " << expected.at(k)
                        << ", got " << actual.at(k) << "\n";
                    result = 1;
                }
            }
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "testTracks2DFieldOrder: unexpected exception: "
            << e.what() << "\n";
        result = 1;
    }

    H5Gclose(grp);
    H5Fclose(file);
    return result;
}

/**
 * @brief Unit test for the Tracks2D class.
 * @return 0 if the test passes, 1 if it fails.
 * @details
 * This function tests that a Tracks2D object can be successfully
 * constructed from one group of each of a set of track files. The
 * large source track files under data/tracks are not stored in the
 * repository, so most of them will typically be absent except on a
 * machine where they have been fetched separately; a file that is
 * absent is skipped rather than treated as a failure. The small
 * tests/tracks/assets/MIST_test.h5 file, which contains a single
 * group extracted from data/tracks/mist.h5, is stored in the
 * repository and so should always be present. The test fails if none
 * of the files are found (since then nothing was actually tested), or
 * if any file that is found fails to produce a valid Tracks2D object.
 * It also tests the feH(), aFe(), and vVcrit() getters against the
 * known metadata of the MIST_test.h5 group, and that getTrack()
 * returns fields in the correct order (see testTracks2DFieldOrder()).
 */
inline auto testTracks2D() -> int
{
    const std::array<TestTracks2DCase, 5> files = {{
        { "tests/tracks/assets/MIST_test.h5", "MIST_test",
            "tests/tracks/assets/tracks.toml" },
        { "data/tracks/mist.h5", "MIST", tracks::defaultRegistry },
        { "data/tracks/parsec_rot.h5", "PARSEC_rot", tracks::defaultRegistry },
        { "data/tracks/parsec_vms.h5", "PARSEC_vms", tracks::defaultRegistry },
        { "data/tracks/stromlo.h5", "Stromlo", tracks::defaultRegistry }
    }};

    bool anyFound = false;
    bool anyFailed = false;
    for (const auto& testCase : files)
    {
        switch (testTracks2DFile(testCase))
        {
            case TestTracks2DOutcome::notFound:
                break;
            case TestTracks2DOutcome::failed:
                anyFound = true;
                anyFailed = true;
                break;
            case TestTracks2DOutcome::passed:
                anyFound = true;
                break;
        }
    }

    if (!anyFound)
    {
        std::cerr << "testTracks2D: none of the expected track files "
            "were found; unable to run this test\n";
        return 1;
    }

    int result = anyFailed ? 1 : 0;
    if (testTracks2DGetters() != 0) { result = 1; }
    if (testTracks2DFieldOrder() != 0) { result = 1; }
    return result;
}

// NOLINTEND(misc-include-cleaner)

#endif // TESTTRACKS2D_HPP
