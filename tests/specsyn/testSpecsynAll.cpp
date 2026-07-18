/**
 * @file testSpecsynAll.cpp
 * @author Mark Krumholz
 * @brief Unit tests for the classes in src/specsyn.
 * @details
 * This file runs unit tests for all the classes in src/specsyn.
 * @date 2026-07-18
 */

#include "testSpecsynBlackbody.hpp"
#include <exception>
#include <iostream>

auto main() -> int {
    try
    {
        int result = 0;
        result += testSpecsynBlackbody();
        return result;
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSpecsynAll: uncaught exception: "
            << error.what() << "\n";
        return 1;
    }
}
