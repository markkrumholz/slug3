/**
 * @file testElemAll.cpp
 * @author Mark Krumholz
 * @brief Unit tests for the classes in src/elem.
 * @details
 * This file runs unit tests for all the classes in src/elem.
 * @date 2026-08-31
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#include "testElemData.hpp"
#include "testIonizationData.hpp"
#include "testIsotopeData.hpp"
#include "testIsotopeTable.hpp"
#include <exception>
#include <iostream>

auto main() -> int {
    try
    {
        int result = 0;
        result += testElemData();
        result += testIonizationData();
        result += testIsotopeData();
        result += testIsotopeTable();
        return result;
    }
    catch (const std::exception& error)
    {
        std::cerr << "testElemAll: uncaught exception: "
            << error.what() << "\n";
        return 1;
    }
}
