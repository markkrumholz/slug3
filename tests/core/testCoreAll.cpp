/**
 * @file testCoreAll.cpp
 * @author Mark Krumholz
 * @brief Unit tests for the classes in src/core.
 * @details
 * This file runs the fast unit tests for the classes in src/core; see
 * testCoreFullAll.cpp for the slow, full-scale end-to-end counterpart
 * (e.g. testClusterSpecsynFull), kept in a separate executable/CTest
 * entry specifically so it can be skipped independently of this one.
 * @date 2026-07-13
 */

#include "testCluster.hpp"
#include "testSimCluster.hpp"
#include "testSpecsynChain.hpp"
#include <exception>
#include <iostream>

auto main() -> int {
    try
    {
        int result = 0;
        result += testCluster();
        result += testSimCluster();
        result += testSpecsynChain();
        return result;
    }
    catch (const std::exception& error)
    {
        std::cerr << "testCoreAll: uncaught exception: "
            << error.what() << "\n";
        return 1;
    }
}
