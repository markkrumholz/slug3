/**
 * @file testCluster.cpp
 * @author Mark Krumholz
 * @brief Unit tests for the Cluster class.
 * @date 2026-07-15
 */

#include "../src/core/Cluster.hpp"
#include "../src/core/SimPhysics.hpp"
#include "../src/extern/tomlplusplus/toml.hpp"
#include "../src/utils/RngThread.hpp"
#include "testCluster.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <string>

namespace
{
    static const std::string inputFile = "tests/core/assets/testCluster.in";
    static constexpr unsigned int rngSeed = 42;

    // Verify that Cluster::starMasses() sums to within 5% of the target mass.
    auto testClusterConstruction() -> int
    {
        try
        {
            const toml::table inputDeck = toml::parse_file(inputFile);
            const core::SimPhysics sim(inputDeck);

            utils::rng().seed(rngSeed);
            const core::Cluster cluster(0, 1e4, 0.0, sim);

            const auto& masses = cluster.starMasses();
            const double totalMass = std::reduce(masses.begin(), masses.end(), 0.0);
            constexpr double targetMass = 1e4;
            constexpr double tolerance = 0.05;

            if (std::abs(totalMass - targetMass) / targetMass > tolerance)
            {
                std::cerr << "testCluster: construction: total mass " << totalMass
                    << " deviates from target " << targetMass
                    << " by more than " << tolerance * 100 << "%\n";
                return 1;
            }
        }
        catch (const std::exception& error)
        {
            std::cerr << "testCluster: construction test failed: "
                << error.what() << "\n";
            return 1;
        }
        return 0;
    }

    // Verify that after advancing to 5 Myr, stars above the live mass range
    // upper limit have been moved to deadStarMasses, and no stars below the
    // live mass range lower limit have been incorrectly killed.
    auto testClusterAdvance() -> int
    {
        constexpr double ageYr = 5e6;
        const double logAge = std::log10(ageYr);

        try
        {
            const toml::table inputDeck = toml::parse_file(inputFile);
            const core::SimPhysics sim(inputDeck);

            utils::rng().seed(rngSeed);
            core::Cluster cluster(0, 1e4, 0.0, sim);

            cluster.advance(ageYr);

            // Obtain the expected live mass range from SimPhysics
            const auto lmr = sim.tracks().liveMassRange(logAge, 0.0);
            if (lmr.empty())
            {
                std::cerr << "testCluster: advance: liveMassRange is empty at age "
                    << ageYr << " yr; cannot proceed\n";
                return 1;
            }

            // Identify the extremes of the live mass range
            double mMaxAlive = lmr.front().second;
            double mMinAlive = lmr.front().first;
            for (const auto& [lo, hi] : lmr)
            {
                mMaxAlive = std::max(mMaxAlive, hi);
                mMinAlive = std::min(mMinAlive, lo);
            }

            // Verify that starMasses() contains no mass above mMaxAlive.
            // The list is sorted, so checking the last element is sufficient.
            const auto& alive = cluster.starMasses();
            if (!alive.empty() && alive.back() > mMaxAlive)
            {
                std::cerr << "testCluster: advance: starMasses() contains mass "
                    << alive.back() << " > mMaxAlive " << mMaxAlive
                    << " at age " << ageYr << " yr\n";
                return 1;
            }

            // Verify that at least one star has died
            const auto& dead = cluster.deadStarMasses();
            if (dead.empty())
            {
                std::cerr << "testCluster: advance: deadStarMasses() is empty after "
                    << ageYr << " yr; expected some massive stars to have died\n";
                return 1;
            }

            // Verify that every dead star is above mMaxAlive (so no star
            // below mMinAlive has been incorrectly killed)
            for (const double m : dead)
            {
                if (m <= mMaxAlive)
                {
                    std::cerr << "testCluster: advance: deadStarMasses() contains mass "
                        << m << " <= mMaxAlive " << mMaxAlive
                        << " at age " << ageYr << " yr\n";
                    return 1;
                }
            }
        }
        catch (const std::exception& error)
        {
            std::cerr << "testCluster: advance test failed: "
                << error.what() << "\n";
            return 1;
        }
        return 0;
    }

} // namespace

auto testCluster() -> int
{
    int result = 0;
    result += testClusterConstruction();
    result += testClusterAdvance();
    return result;
}
