/**
 * @file testNebularAll.cpp
 * @author Mark Krumholz
 * @brief Unit tests for the classes in src/nebular.
 * @details
 * This file runs unit tests for all the classes in src/nebular.
 * @date 2026-08-30
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#include "testNebular.hpp"
#include <exception>
#include <iostream>

auto main() -> int {
    try
    {
        return testNebular();
    }
    catch (const std::exception& error)
    {
        std::cerr << "testNebularAll: uncaught exception: "
            << error.what() << "\n";
        return 1;
    }
}
