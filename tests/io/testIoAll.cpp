/**
 * @file testIoAll.cpp
 * @author Mark Krumholz
 * @brief Unit tests for the classes in src/io.
 * @details
 * This file runs unit tests for all the classes in src/io.
 * @date 2026-07-16
 */

#include "testSimControls.hpp"
#include "testSimPhysics.hpp"
#include <exception>
#include <iostream>

auto main() -> int {
    try
    {
        int result = 0;
        result += testSimPhysics();
        result += testSimControls();
        return result;
    }
    catch (const std::exception& error)
    {
        std::cerr << "testIoAll: uncaught exception: "
            << error.what() << "\n";
        return 1;
    }
}
