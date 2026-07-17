/**
 * @file testUtilsAll.cpp
 * @author Mark Krumholz
 * @brief Unit tests for the utility functions.
 * @details
 * This file contains unit tests for the utility functions defined in the src/utils directory.
 * @date 2024-07-03
 */

#include "testRngThread.hpp"
#include "testUniqueIDManager.hpp"
#include <exception>
#include <iostream>

auto main() -> int {
    try
    {
        int result = 0;
        result += testRngThread();
        result += testUniqueIDManager();
        return result;
    }
    catch (const std::exception& error)
    {
        std::cerr << "testUtilsAll: uncaught exception: "
            << error.what() << "\n";
        return 1;
    }
}
