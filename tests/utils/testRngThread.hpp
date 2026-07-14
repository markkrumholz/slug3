/**
 * @file testRngThread.hpp
 * @author Mark Krumholz
 * @brief Unit tests for the RngThread class.
 * @details
 * This file contains unit tests for the RngThread class, which implements
 * thread-safe random number generation. The tests cover the construction of
 * the RngThread class, the generation of random numbers, and the seeding of
 * the random number generator.
 * @date 2024-07-03
 */

#ifndef TESTRNGTHREAD_HPP
#define TESTRNGTHREAD_HPP

#include "../../src/utils/RngThread.hpp"
#ifdef _OPENMP
#   include <omp.h>
#endif // _OPENMP
#include <random>

auto testRngThread() -> int
{
    // Test generation of random numbers
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    const double rand1 = dist(utils::rng()());
    const double rand2 = dist(utils::rng()());
    if (rand1 == rand2) {
        std::cerr << "testRngThread: Random number generation failed: "
                  << "two consecutive calls returned the same value: "
                  << rand1 << "\n";
        return 1;
    }

    // Test seeding of random number generator
    const uint64_t seed = 42;
    utils::rng().seed(seed);
    const double randSeeded1 = dist(utils::rng()());
    utils::rng().seed(seed);
    const double randSeeded2 = dist(utils::rng()());
    if (randSeeded1 != randSeeded2) {
        std::cerr << "testRngThread: Random number generation with seed failed: "
                  << "two calls with the same seed returned different values: "
                  << randSeeded1 << " and " << randSeeded2 << "\n";
        return 1;
    }

#ifdef _OPENMP
    // Test that different threads produce different random numbers
    const int nThreads = 4;
    std::vector<double> randThread(nThreads);
    #pragma omp parallel num_threads(nThreads)
    {
        const int threadNum = omp_get_thread_num();
        randThread[threadNum] = dist(utils::rng()());
    }
    for (int i = 0; i < nThreads; ++i) {
        for (int j = i + 1; j < nThreads; ++j) {
            if (randThread[i] == randThread[j]) {
                std::cerr << "testRngThread: Random number generation in threads failed: "
                          << "threads " << i << " and " << j << " returned the same value: "
                          << randThread[i] << "\n";
                return 1;
            }
        }
    }
#endif // _OPENMP

    // Test that getState() changes after drawing a random number
    utils::rng().seed(seed);
    const utils::RngState stateBefore = utils::rng().getState();
    dist(utils::rng()());
    const utils::RngState stateAfter = utils::rng().getState();
    if (stateBefore == stateAfter) {
        std::cerr << "testRngThread: getState() did not change after "
                  << "drawing a random number\n";
        return 1;
    }

    // Test that getState()/setState() round-trip: saving the state,
    // drawing some numbers, then restoring the saved state should
    // reproduce exactly the same sequence of numbers
    utils::rng().seed(seed);
    const utils::RngState savedState = utils::rng().getState();
    const double origDraw1 = dist(utils::rng()());
    const double origDraw2 = dist(utils::rng()());
    utils::rng().setState(savedState);
    const double restoredDraw1 = dist(utils::rng()());
    const double restoredDraw2 = dist(utils::rng()());
    if (origDraw1 != restoredDraw1 || origDraw2 != restoredDraw2) {
        std::cerr << "testRngThread: getState()/setState() round-trip failed: "
                  << "restoring a saved state did not reproduce the same "
                  << "sequence of random numbers (expected " << origDraw1
                  << ", " << origDraw2 << ", got " << restoredDraw1 << ", "
                  << restoredDraw2 << ")\n";
        return 1;
    }

#ifdef _OPENMP
    // Test that getState() is private to the calling thread, like the
    // rng engine itself
    std::vector<utils::RngState> stateThread(nThreads);
    #pragma omp parallel num_threads(nThreads)
    {
        const int threadNum = omp_get_thread_num();
        stateThread[threadNum] = utils::rng().getState();
    }
    for (int i = 0; i < nThreads; ++i) {
        for (int j = i + 1; j < nThreads; ++j) {
            if (stateThread[i] == stateThread[j]) {
                std::cerr << "testRngThread: getState() in threads failed: "
                          << "threads " << i << " and " << j
                          << " returned the same state\n";
                return 1;
            }
        }
    }
#endif // _OPENMP

    return 0; // Passed
}

#endif // TESTRNGTHREAD_HPP