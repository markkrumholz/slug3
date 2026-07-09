/**
 * @file testTracksAll.cpp
 * @author Mark Krumholz
 * @brief Unit tests for the tracks classes.
 * @details
 * This file runs unit tests for all the classes in src/tracks.
 * @date 2024-07-09
 */

#include "testTracks2D.hpp"

auto main() -> int {
    int result = 0;
    result += testTracks2D();
    return result;
}
