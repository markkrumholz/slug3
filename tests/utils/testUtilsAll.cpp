/**
 * @file testUtilsAll.cpp
 * @author Mark Krumholz
 * @brief Unit tests for the utility functions.
 * @details
 * This file contains unit tests for the utility functions defined in the src/utils directory.
 * @date 2024-07-03
 */

#include "testRngThread.hpp"

auto main() -> int {
    int result = 0;
    result += testRngThread();
    return result;
}