/**
 * @file testSpecsynAll.cpp
 * @author Mark Krumholz
 * @brief Unit tests for the classes in src/specsyn.
 * @details
 * This file runs unit tests for all the classes in src/specsyn.
 * @date 2026-07-18
 */

#include "testSpecsynBlackbody.hpp"
#include "testSpecsynLib.hpp"
#include "testSpecsynLibChained.hpp"
#include "testSpecsynLibWR.hpp"
#include "testSpecsynUtils.hpp"
#include <exception>
#include <iostream>

auto main() -> int {
    try
    {
        int result = 0;
        result += testSpecsynBlackbody();
        result += testSpecsynUtils();
        result += testSpecsynLib();
        result += testSpecsynLibWR();
        result += testSpecsynLibChained();
        return result;
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSpecsynAll: uncaught exception: "
            << error.what() << "\n";
        return 1;
    }
}
