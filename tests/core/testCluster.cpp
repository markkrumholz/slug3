/**
 * @file testCluster.cpp
 * @author Mark Krumholz
 * @brief Unit tests for the Cluster class.
 * @date 2026-07-15
 */

#include "../src/core/Cluster.hpp"
#include "../src/io/SimControls.hpp"
#include "../src/io/SimPhysics.hpp"
#include "../src/utils/RngThread.hpp"
#include "testCluster.hpp"
#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <numeric>
#include <string_view>
#include <toml.hpp>

static constexpr std::string_view inputFile = "tests/core/assets/testCluster.in";
static constexpr std::string_view inputFileMinStochMass =
    "tests/core/assets/testClusterMinStochMass.in";
static constexpr unsigned int rngSeed = 42;

// Verify that Cluster::starMasses() sums to within 5% of the target mass.
static auto testClusterConstruction() -> int
{
    try
    {
        const toml::table inputDeck = toml::parse_file(inputFile);
        const io::SimControls controls(inputDeck);
        const io::SimPhysics sim(inputDeck, controls.simType());

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
static auto testClusterAdvance() -> int
{
    constexpr double ageYr = 5e6;
    const double logAge = std::log10(ageYr);

    try
    {
        const toml::table inputDeck = toml::parse_file(inputFile);
        const io::SimControls controls(inputDeck);
        const io::SimPhysics sim(inputDeck, controls.simType());

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

// Verify min_stoch_mass behaviour: starMasses() should contain only stars at
// or above min_stoch_mass, and their total mass should be within 10% of the
// stochastic fraction of the target cluster mass.
static auto testClusterMinStochMass() -> int
{
    constexpr double targetMass = 1e4;
    constexpr double tolerance = 0.15;

    try
    {
        const toml::table inputDeck = toml::parse_file(inputFileMinStochMass);
        const io::SimControls controls(inputDeck);
        const io::SimPhysics sim(inputDeck, controls.simType());

        utils::rng().seed(rngSeed);
        const core::Cluster cluster(0, targetMass, 0.0, sim);

        const double minStochMass = sim.minStochMass();
        const auto& masses = cluster.starMasses();

        // Every returned star must be at or above min_stoch_mass.
        // The list is sorted, so checking the first element is sufficient.
        if (!masses.empty() && masses.front() < minStochMass)
        {
            std::cerr << "testCluster: minStochMass: starMasses() contains mass "
                << masses.front() << " < minStochMass " << minStochMass << "\n";
            return 1;
        }

        // The total stochastic mass should be within tolerance of
        // fracStochMass * targetMass.
        const double stochTarget = sim.fracStochMass() * targetMass;
        const double totalMass = std::reduce(masses.begin(), masses.end(), 0.0);
        if (std::abs(totalMass - stochTarget) / stochTarget > tolerance)
        {
            std::cerr << "testCluster: minStochMass: stochastic mass " << totalMass
                << " deviates from stochastic target " << stochTarget
                << " by more than " << tolerance * 100 << "%\n";
            return 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testCluster: minStochMass test failed: "
            << error.what() << "\n";
        return 1;
    }
    return 0;
}

auto testCluster() -> int
{
    int result = 0;
    result += testClusterConstruction();
    result += testClusterAdvance();
    result += testClusterMinStochMass();
    return result;
}
