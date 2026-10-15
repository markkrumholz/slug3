/**
 * @file testKdenAll.cpp
 * @author Mark Krumholz
 * @brief Unit tests for the kd tree and kernel density code in src/kden
 * @date 2026-10-15
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#include "testBandwidth.hpp"
#include "testKDTree.hpp"
#include <exception>
#include <iostream>

auto main() -> int {
    try
    {
        int result = 0;
        result += testBandwidthDist2();
        result += testBandwidthReduce();
        result += testBandwidthBoxBound();
        result += testKDTreeTraversal();
        result += testKDTreeConstruction();
        result += testKDTreeNeighbors();
        result += testKDTreeNeighborsBatch();
        return result;
    }
    catch (const std::exception& error)
    {
        std::cerr << "testKdenAll: uncaught exception: " << error.what() << "\n";
        return 1;
    }
}
