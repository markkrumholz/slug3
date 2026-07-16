/**
 * @file testCoreAll.cpp
 * @author Mark Krumholz
 * @brief Unit tests for the classes in src/core.
 * @details
 * This file runs unit tests for all the classes in src/core.
 * @date 2026-07-13
 */

#include "testCluster.hpp"
#include <exception>
#include <iostream>

auto main() -> int {
    try
    {
        int result = 0;
        result += testCluster();
        return result;
    }
    catch (const std::exception& error)
    {
        std::cerr << "testCoreAll: uncaught exception: "
            << error.what() << "\n";
        return 1;
    }
}
