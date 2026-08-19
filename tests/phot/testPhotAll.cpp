/**
 * @file testPhotAll.cpp
 * @author Mark Krumholz
 * @brief Unit tests for the classes in src/phot.
 * @details
 * This file runs unit tests for all the classes in src/phot.
 * @date 2026-07-26
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#include "testFilterCollection.hpp"
#include "testFilterIdeal.hpp"
#include "testFilterTabulated.hpp"
#include "testPhotCommons.hpp"
#include <exception>
#include <iostream>

auto main() -> int {
    try
    {
        int result = 0;
        result += testPhotCommons();
        result += testFilterTabulated();
        result += testFilterIdeal();
        result += testFilterCollection();
        return result;
    }
    catch (const std::exception& error)
    {
        std::cerr << "testPhotAll: uncaught exception: "
            << error.what() << "\n";
        return 1;
    }
}
