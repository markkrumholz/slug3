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
#include <random>

auto testRngThread() -> int
{
    // Test generation of random numbers
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    const double rand1 = dist(utils::rng());
    const double rand2 = dist(utils::rng());
    if (rand1 == rand2) {
        std::cerr << "testRngThread: Random number generation failed: "
                  << "two consecutive calls returned the same value: "
                  << rand1 << "\n";
        return 1;
    }

    // Test seeding of random number generator
    const uint64_t seed = 42;
    utils::rng.seed(seed);
    const double randSeeded1 = dist(utils::rng());
    utils::rng.seed(seed);
    const double randSeeded2 = dist(utils::rng());
    if (randSeeded1 != randSeeded2) {
        std::cerr << "testRngThread: Random number generation with seed failed: "
                  << "two calls with the same seed returned different values: "
                  << randSeeded1 << " and " << randSeeded2 << "\n";
        return 1;
    }   

    return 0; // Passed
}

#endif // TESTRNGTHREAD_HPP