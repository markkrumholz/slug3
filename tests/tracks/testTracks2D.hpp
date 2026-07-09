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
#include "hdf5.h"
#include <array>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

/**
 * @brief Attempt to construct a Tracks2D object from one group of a track file
 * @param path Path to the HDF5 track file
 * @return true if a Tracks2D object was successfully constructed, false otherwise
 * @details
 * This function opens the given HDF5 file, finds the first group at
 * its root level (the choice of group does not matter for this test),
 * and attempts to use it to construct a Tracks2D object with the
 * default value of ntMin.
 */
inline auto testTracks2DFile(const std::string& path) -> bool
{
    const hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0)
    {
        std::cerr << "testTracks2D: unable to open file " << path << "\n";
        return false;
    }

    // Find the name of the first group at the root of the file
    const auto nameLen = H5Lget_name_by_idx(file, ".", H5_INDEX_NAME,
        H5_ITER_INC, 0, nullptr, 0, H5P_DEFAULT);
    if (nameLen < 0)
    {
        std::cerr << "testTracks2D: unable to find a group in "
            << path << "\n";
        H5Fclose(file);
        return false;
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
        return false;
    }

    bool success = true;
    try
    {
        const tracks::Tracks2D tracks2d(grp);
    }
    catch (const std::exception& e)
    {
        std::cerr << "testTracks2D: failed to construct Tracks2D from "
            << path << ", group " << groupName << ": " << e.what() << "\n";
        success = false;
    }

    H5Gclose(grp);
    H5Fclose(file);
    return success;
}

/**
 * @brief Unit test for the Tracks2D class.
 * @return 0 if the test passes, 1 if it fails.
 * @details
 * This function tests that a Tracks2D object can be successfully
 * constructed from one group of each of the mist.h5, parsec_rot.h5,
 * parsec_vms.h5, and stromlo.h5 track files.
 */
inline auto testTracks2D() -> int
{
    const std::array<std::string, 4> files = {
        "data/tracks/mist.h5",
        "data/tracks/parsec_rot.h5",
        "data/tracks/parsec_vms.h5",
        "data/tracks/stromlo.h5"
    };

    int result = 0;
    for (const auto& f : files)
    {
        if (!testTracks2DFile(f)) { result = 1; }
    }
    return result;
}

#endif // TESTTRACKS2D_HPP
