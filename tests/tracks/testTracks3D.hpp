/**
 * @file testTracks3D.hpp
 * @author Mark Krumholz
 * @brief Unit tests for the Tracks3D class.
 * @details
 * This file contains unit tests for the Tracks3D class, which reads a
 * grid of stellar evolutionary tracks spanning a range of [Fe/H]
 * values from an HDF5 file. Unlike the Tracks2D tests, these tests act
 * only on the small, reduced test track set in tests/tracks/assets;
 * building a Tracks3D from the full-size track files in data/tracks
 * would be far too slow for a unit test.
 * @date 2024-07-10
 */

#ifndef TESTTRACKS3D_HPP
#define TESTTRACKS3D_HPP

#include "../../src/tracks/Tracks3D.hpp"
#include "trackFieldFixture.hpp"
#include "hdf5.h" // NOLINT(misc-include-cleaner)
#include <exception>
#include <iostream>
#include <string>

/**
 * @brief Unit test for the Tracks3D class.
 * @return 0 if the test passes, 1 if it fails.
 * @details
 * This function constructs a Tracks3D object from the MIST_test track
 * set in tests/tracks/assets/tracks.toml, which contains 5 groups at
 * afe = -0.2, vvcrit = 0.0, and feh = -1.0, -0.5, -0.25, 0.0, and 0.5,
 * and verifies that construction succeeds without error.
 */
inline auto testTracks3D() -> int
{
    const std::string registryName = "tests/tracks/assets/tracks.toml";
    const std::string trackName = "MIST_test";

    try
    {
        const tracks::Tracks3D tracks3d(
            trackName, -0.5, 0.0, 0.0, -0.2, registryName);
    }
    catch (const std::exception& e)
    {
        std::cerr << "testTracks3D: failed to construct Tracks3D from "
            << registryName << ", track set " << trackName << ": "
            << e.what() << "\n";
        return 1;
    }

    return 0;
}

// Suppress clang-tidy warnings iun this namespace caused by just including
// hdf5.h, instead of the individual HDF5 headers, since this is the paradigm
// that HDF5 wants
// NOLINTBEGIN(misc-include-cleaner)

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
 * field index to on-disk column. It constructs a Tracks3D object with
 * fehMin = fehMax = 0.0, an exact point on the MIST_test set's [Fe/H]
 * grid, which exercises the single-slice code path (a mesh one
 * element wide in the feh direction). It then calls getTrack() for
 * mass = 5.0 (an exact point on the mass grid) and evaluates the
 * result at the exact age of an arbitrary interior row of the raw
 * track_m5.000 dataset in the feh_0.00_afe_-0.2_vvcrit_0.00 group of
 * tests/tracks/assets/MIST_test.h5. Since mass = 5.0 is on the mesh's
 * mass grid, feh = 0.0 is the mesh's only feh point, and the query age
 * is on that mass's own age grid, the interpolated result should
 * reproduce the raw row exactly (up to floating-point round-off); the
 * raw row itself is read independently of the Tracks3D field-mapping
 * logic via trackFieldFixture.hpp, so this test does not depend on
 * that logic being correct to establish its expectations.
 */
inline auto testTracks3DFieldOrder() -> int
{
    const std::string registryName = "tests/tracks/assets/tracks.toml";
    const std::string trackName = "MIST_test";
    const std::string h5Path = "tests/tracks/assets/MIST_test.h5";
    const std::string groupName = "feh_0.00_afe_-0.2_vvcrit_0.00";
    constexpr double feh = 0.0;
    constexpr double mass = 5.0;
    constexpr size_t rowIdx = 500;

    const hid_t file = H5Fopen(h5Path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0)
    {
        std::cerr << "testTracks3DFieldOrder: unable to open file "
            << h5Path << "\n";
        return 1;
    }
    const hid_t grp = H5Gopen2(file, groupName.c_str(), H5P_DEFAULT);
    if (grp < 0)
    {
        std::cerr << "testTracks3DFieldOrder: unable to open group "
            << groupName << " in " << h5Path << "\n";
        H5Fclose(file);
        return 1;
    }

    int result = 0;
    try
    {
        const auto [age, expected] = testutil::readRawFields(grp, mass, rowIdx);

        const tracks::Tracks3D tracks3d(
            trackName, feh, feh, 0.0, -0.2, registryName);
        const auto track = tracks3d.getTrack(mass, feh);
        if (!track)
        {
            std::cerr << "testTracks3DFieldOrder: getTrack(" << mass
                << ", " << feh << ") returned null\n";
            result = 1;
        }
        else
        {
            const auto actual = (*track)(age);
            for (size_t k = 0; k < testutil::nQty; ++k)
            {
                if (!testutil::fieldsMatch(actual.at(k), expected.at(k)))
                {
                    std::cerr << "testTracks3DFieldOrder: field "
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
        std::cerr << "testTracks3DFieldOrder: unexpected exception: "
            << e.what() << "\n";
        result = 1;
    }

    H5Gclose(grp);
    H5Fclose(file);
    return result;
}

// NOLINTEND(misc-include-cleaner)

#endif // TESTTRACKS3D_HPP
