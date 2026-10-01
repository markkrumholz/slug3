/**
 * @file testFeedbackAll.cpp
 * @author Mark Krumholz
 * @brief Unit tests for the classes in src/feedback.
 * @details
 * This file runs unit tests for all the classes in src/feedback.
 * @date 2026-09-30
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#include "testWinds.hpp"
#include <exception>
#include <iostream>

auto main() -> int {
    try
    {
        int result = 0;
        result += testWinds();
        return result;
    }
    catch (const std::exception& error)
    {
        std::cerr << "testFeedbackAll: uncaught exception: "
            << error.what() << "\n";
        return 1;
    }
}
