/**
 * @file testUniqueIDManager.hpp
 * @author Mark Krumholz
 * @brief Unit tests for the UniqueIDManager class.
 * @date 2026-07-17
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
    // getID() itself
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
            ids[static_cast<size_t>((threadNum * nRepeat) + r)] = utils::getID();
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

#endif // TESTUNIQUEIDMANAGER_HPP
