/**
 * @file testTracksAll.cpp
 * @author Mark Krumholz
 * @brief Unit tests for the tracks classes.
 * @details
 * This file runs unit tests for all the classes in src/tracks.
 * @date 2024-07-09
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#include "testTrackUtils.hpp"
#include "testTracks2D.hpp"
#include "testTracks3D.hpp"
#include <exception>
#include <iostream>

auto main() -> int {
    try
    {
        int result = 0;
        result += testTracks2D();
        result += testTracks3D();
        result += testTracks3DFieldOrder();
        result += testTracks3DGetStar();
        result += testTrackUtils();
        return result;
    }
    catch (const std::exception& error)
    {
        std::cerr << "testTracksAll: uncaught exception: "
            << error.what() << "\n";
        return 1;
    }
}
