/**
 * @file testTracksFullAll.cpp
 * @author Mark Krumholz
 * @brief Slow tests for the classes in src/tracks, run against full track data.
 * @details
 * Counterpart to testTracksAll.cpp, for tests that run against the
 * real, gitignored track files in data/tracks rather than the small
 * test fixtures in tests/tracks/assets. Built into its own executable
 * and run via its own CTest entry tagged LABELS "slow" (see
 * CMakeLists.txt), for the same reasons as testCoreFullAll.cpp. Add
 * future full-data track tests here.
 * @date 2026-09-24
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#include "testTracksSB99Full.hpp"
#include <exception>
#include <iostream>

auto main() -> int {
    try
    {
        int result = 0;
        result += testTracksSB99Full();
        return result;
    }
    catch (const std::exception& error)
    {
        std::cerr << "testTracksFullAll: uncaught exception: "
            << error.what() << "\n";
        return 1;
    }
}
