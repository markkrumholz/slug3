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
            registryName, trackName, -0.5, 0.0, 0.0, -0.2);
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

#endif // TESTTRACKS3D_HPP
