/**
 * @file testUniqueIDManager.hpp
 * @author Mark Krumholz
 * @brief Unit tests for the UniqueIDManager class.
 * @date 2026-07-17
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#ifndef TESTUNIQUEIDMANAGER_HPP
#define TESTUNIQUEIDMANAGER_HPP

#include "../../src/utils/UniqueIDManager.hpp"
#ifdef _OPENMP
#   include <omp.h>
#endif // _OPENMP
#include <algorithm>
#include <cstddef>
#include <iostream>
#include <vector>

auto testUniqueIDManager() -> int
{
#ifdef _OPENMP
    constexpr int nThreads = 4;
#else
    constexpr int nThreads = 1;
#endif // _OPENMP
    constexpr int nRepeat = 1000;

    // Each thread writes its own nRepeat-element slice of ids, so
    // there is no need for any additional synchronization beyond
    // uniqueID().get() itself
    std::vector<unsigned long> ids(static_cast<size_t>(nThreads) * nRepeat);
    #pragma omp parallel num_threads(nThreads)
    {
#ifdef _OPENMP
        const int threadNum = omp_get_thread_num();
#else
        const int threadNum = 0;
#endif // _OPENMP
        for (int r = 0; r < nRepeat; ++r)
        {
            ids[static_cast<size_t>((threadNum * nRepeat) + r)] = utils::uniqueID().get();
        }
    }

    // Every value from 0 to nThreads*nRepeat - 1 should appear exactly
    // once; sorting and checking ids[i] == i confirms both that no
    // value was skipped and that no value was handed out twice
    std::ranges::sort(ids);
    for (size_t i = 0; i < ids.size(); ++i)
    {
        if (ids[i] != i)
        {
            std::cerr << "testUniqueIDManager: expected sorted IDs to be "
                "0.." << (ids.size() - 1) << " each exactly once, but "
                "found " << ids[i] << " at sorted position " << i << "\n";
            return 1;
        }
    }

    return 0; // Success
}

// Checks set()/read() in isolation from get()'s own concurrent-use
// behavior above: set() unconditionally overwrites the shared
// counter, so this is self-contained regardless of whatever value
// testUniqueIDManager() itself left it at.
auto testUniqueIDManagerSetRead() -> int
{
    constexpr unsigned long restoredValue = 424242;
    utils::uniqueID().set(restoredValue);

    // read() reports exactly the value just set, and -- called twice
    // in a row -- does not itself advance the counter the way get()
    // does
    if (utils::uniqueID().read() != restoredValue)
    {
        std::cerr << "testUniqueIDManagerSetRead: expected read() == "
            << restoredValue << " right after set(), got "
            << utils::uniqueID().read() << "\n";
        return 1;
    }
    if (utils::uniqueID().read() != restoredValue)
    {
        std::cerr << "testUniqueIDManagerSetRead: a second read() in a row "
            "should still be " << restoredValue << ", got "
            << utils::uniqueID().read() << "\n";
        return 1;
    }

    // get() returns exactly what read() just reported, and (unlike
    // read()) consumes it, advancing the counter by one
    const auto got = utils::uniqueID().get();
    if (got != restoredValue)
    {
        std::cerr << "testUniqueIDManagerSetRead: expected get() == "
            << restoredValue << ", got " << got << "\n";
        return 1;
    }
    if (utils::uniqueID().read() != restoredValue + 1)
    {
        std::cerr << "testUniqueIDManagerSetRead: expected read() == "
            << (restoredValue + 1) << " after that get(), got "
            << utils::uniqueID().read() << "\n";
        return 1;
    }

    return 0; // Success
}

#endif // TESTUNIQUEIDMANAGER_HPP
