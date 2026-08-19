/**
 * @file testExtinctAll.cpp
 * @author Mark Krumholz
 * @brief Unit tests for the classes in src/extinct.
 * @details
 * This file runs unit tests for all the classes in src/extinct.
 * @date 2026-08-03
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#include "testExtinct.hpp"
#include <exception>
#include <iostream>

auto main() -> int {
    try
    {
        int result = 0;
        result += testExtinct();
        result += testExtinctNormalization();
        result += testExtinctApplyExtinctionCtsInvalid();
        return result;
    }
    catch (const std::exception& error)
    {
        std::cerr << "testExtinctAll: uncaught exception: "
            << error.what() << "\n";
        return 1;
    }
}
