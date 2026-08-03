/**
 * @file testExtinctAll.cpp
 * @author Mark Krumholz
 * @brief Unit tests for the classes in src/extinct.
 * @details
 * This file runs unit tests for all the classes in src/extinct.
 * @date 2026-08-03
 */

#include "testExtinct.hpp"
#include <exception>
#include <iostream>

auto main() -> int {
    try
    {
        int result = 0;
        result += testExtinct();
        return result;
    }
    catch (const std::exception& error)
    {
        std::cerr << "testExtinctAll: uncaught exception: "
            << error.what() << "\n";
        return 1;
    }
}
