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
#include "hdf5.h"  // NOLINT(misc-include-cleaner)
#include <array>
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

// Suppress clang-tidy warnings iun this namespace caused by just including
// hdf5.h, instead of the individual HDF5 headers, since this is the paradigm
// that HDF5 wants
// NOLINTBEGIN(misc-include-cleaner)

/**
 * @brief Attempt to construct a Tracks2D object from one group of a track file
 * @param path Path to the HDF5 track file
 * @return The outcome of the attempt
 * @details
 * If the file does not exist, this function returns notFound without
 * attempting to open it. Otherwise, it opens the file, finds the
 * first group at its root level (the choice of group does not matter
 * for this test), and attempts to use it to construct a Tracks2D
 * object with the default value of ntMin, returning passed or failed
 * depending on whether construction succeeds.
 */
inline auto testTracks2DFile(const std::string& path) -> TestTracks2DOutcome
{
    if (!std::filesystem::exists(path))
    {
        std::cerr << "testTracks2D: file " << path
            << " not found, skipping\n";
        return TestTracks2DOutcome::notFound;
    }

    const hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0)
    {
        std::cerr << "testTracks2D: unable to open file " << path << "\n";
        return TestTracks2DOutcome::failed;
    }

    // Find the name of the first group at the root of the file
    const auto nameLen = H5Lget_name_by_idx(file, ".", H5_INDEX_NAME,
        H5_ITER_INC, 0, nullptr, 0, H5P_DEFAULT);
    if (nameLen < 0)
    {
        std::cerr << "testTracks2D: unable to find a group in "
            << path << "\n";
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
            << " in " << path << "\n";
        H5Fclose(file);
        return TestTracks2DOutcome::failed;
    }

    auto outcome = TestTracks2DOutcome::passed;
    try
    {
        const tracks::Tracks2D tracks2d(grp);
    }
    catch (const std::exception& e)
    {
        std::cerr << "testTracks2D: failed to construct Tracks2D from "
            << path << ", group " << groupName << ": " << e.what() << "\n";
        outcome = TestTracks2DOutcome::failed;
    }

    H5Gclose(grp);
    H5Fclose(file);
    return outcome;
}

/**
 * @brief Unit test for the Tracks2D feH(), aFe(), and vVcrit() getters.
 * @return 0 if the test passes, 1 if it fails.
 * @details
 * This function constructs a Tracks2D object from the
 * feh_-0.25_afe_-0.2_vvcrit_0.00 group of tests/tracks/assets/MIST_test.h5,
 * whose [Fe/H], [alpha/Fe], and v/vcrit values are known exactly from
 * the group name, and checks that feH(), aFe(), and vVcrit() report
 * those values.
 */
inline auto testTracks2DGetters() -> int
{
    const std::string path = "tests/tracks/assets/MIST_test.h5";
    const std::string groupName = "feh_-0.25_afe_-0.2_vvcrit_0.00";

    const hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0)
    {
        std::cerr << "testTracks2DGetters: unable to open file "
            << path << "\n";
        return 1;
    }
    const hid_t grp = H5Gopen2(file, groupName.c_str(), H5P_DEFAULT);
    if (grp < 0)
    {
        std::cerr << "testTracks2DGetters: unable to open group "
            << groupName << " in " << path << "\n";
        H5Fclose(file);
        return 1;
    }

    int result = 0;
    try
    {
        const tracks::Tracks2D tracks2d(grp);
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
            "from " << path << ", group " << groupName << ": "
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
 * known metadata of the MIST_test.h5 group.
 */
inline auto testTracks2D() -> int
{
    const std::array<std::string, 5> files = {
        "tests/tracks/assets/MIST_test.h5",
        "data/tracks/mist.h5",
        "data/tracks/parsec_rot.h5",
        "data/tracks/parsec_vms.h5",
        "data/tracks/stromlo.h5"
    };

    bool anyFound = false;
    bool anyFailed = false;
    for (const auto& f : files)
    {
        switch (testTracks2DFile(f))
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
    return result;
}

// NOLINTEND(misc-include-cleaner)

#endif // TESTTRACKS2D_HPP
