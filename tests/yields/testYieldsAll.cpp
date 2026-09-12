/**
 * @file testYieldsAll.cpp
 * @author Mark Krumholz
 * @brief Unit tests for the classes in src/yields.
 * @details
 * This file runs unit tests for all the classes in src/yields.
 * @date 2026-09-12
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#include "testYieldChannel.hpp"
#include <exception>
#include <iostream>

auto main() -> int {
    try
    {
        int result = 0;
        result += testYieldChannelCcsn();
        result += testYieldChannelMassiveStarWinds();
        result += testYieldChannelFeHRangeGuard();
        result += testYieldChannelUnknownModel();
        result += testYieldChannelCopyMoveSafety();
        return result;
    }
    catch (const std::exception& error)
    {
        std::cerr << "testYieldsAll: uncaught exception: "
            << error.what() << "\n";
        return 1;
    }
}
