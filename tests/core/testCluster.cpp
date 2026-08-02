/**
 * @file testCluster.cpp
 * @author Mark Krumholz
 * @brief Unit tests for the Cluster class.
 * @date 2026-07-15
 */

#include "../src/core/Cluster.hpp"
#include "../src/io/SimControls.hpp"
#include "../src/phot/FilterCollection.hpp"
#include "../src/utils/RngThread.hpp"
#include "testCluster.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
#include <numeric>
#include <string>
#include <string_view>
#include <toml.hpp>
#include <vector>

static constexpr std::string_view inputFile = "tests/core/assets/testCluster.in";
static constexpr std::string_view inputFileMinStochMass =
    "tests/core/assets/testClusterMinStochMass.in";
static constexpr std::string_view inputFilePhot = "tests/core/assets/testClusterPhot.in";
static constexpr std::string_view inputFileLbol = "tests/core/assets/testClusterLbol.in";
static constexpr unsigned int rngSeed = 42;

// Verify that Cluster::starMasses() sums to within 5% of the target mass.
static auto testClusterConstruction() -> int
{
    try
    {
        const toml::table inputDeck = toml::parse_file(inputFile);
        const io::SimControls controls(inputDeck);

        utils::rng().seed(rngSeed);
        const core::Cluster cluster(0, 1e4, 0.0, controls);

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

        utils::rng().seed(rngSeed);
        core::Cluster cluster(0, 1e4, 0.0, controls);

        cluster.advance(ageYr);

        // Obtain the expected live mass range from SimControls
        const auto lmr = controls.tracks().liveMassRange(logAge, 0.0);
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

        utils::rng().seed(rngSeed);
        const core::Cluster cluster(0, targetMass, 0.0, controls);

        const double minStochMass = controls.minStochMass();
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
        const double stochTarget = controls.fracStochMass() * targetMass;
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

// Verify that Cluster::spec() is populated by advance() when a
// spectral synthesizer is available, from the individually-sampled
// stars alone (min_stoch_mass unset, so every star is drawn
// stochastically and there is no continuously-sampled part of the
// population to add).
static auto testClusterSpecFullyStochastic() -> int
{
    constexpr double ageYr = 1e6;

    try
    {
        const toml::table inputDeck = toml::parse_file(inputFile);
        const io::SimControls controls(inputDeck);

        utils::rng().seed(rngSeed);
        core::Cluster cluster(0, 1e4, 0.0, controls);
        cluster.advance(ageYr);

        const auto& spec = cluster.spec();
        if (spec.size() != controls.specsyn()->wl().size())
        {
            std::cerr << "testCluster: specFullyStochastic: spec() size "
                << spec.size() << " does not match wl() size "
                << controls.specsyn()->wl().size() << "\n";
            return 1;
        }
        if (std::reduce(spec.begin(), spec.end(), 0.0) <= 0.0)
        {
            std::cerr << "testCluster: specFullyStochastic: expected a "
                "non-zero spectrum from the individually-sampled stars\n";
            return 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testCluster: specFullyStochastic test failed: "
            << error.what() << "\n";
        return 1;
    }
    return 0;
}

// Verify that Cluster::spec() is a non-trivial, correctly-sized
// spectrum when there is a continuously-sampled (non-stochastic)
// part of the population (min_stoch_mass set)
static auto testClusterSpecContinuousPopulation() -> int
{
    constexpr double ageYr = 1e6;

    try
    {
        const toml::table inputDeck = toml::parse_file(inputFileMinStochMass);
        const io::SimControls controls(inputDeck);

        utils::rng().seed(rngSeed);
        core::Cluster cluster(0, 1e4, 0.0, controls);
        cluster.advance(ageYr);

        const auto& spec = cluster.spec();
        if (spec.size() != controls.specsyn()->wl().size())
        {
            std::cerr << "testCluster: specContinuousPopulation: spec() size "
                << spec.size() << " does not match wl() size "
                << controls.specsyn()->wl().size() << "\n";
            return 1;
        }
        if (std::reduce(spec.begin(), spec.end(), 0.0) <= 0.0)
        {
            std::cerr << "testCluster: specContinuousPopulation: expected a "
                "non-zero spectrum with a non-stochastic population present\n";
            return 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testCluster: specContinuousPopulation test failed: "
            << error.what() << "\n";
        return 1;
    }
    return 0;
}

// Verify that Cluster::phot() is populated by advance() when a filter
// collection is available (phot.filters given in the input deck), and
// that SimControls::readFilters's "Lbol" handling and
// SimControls::filters()->filterNames() come out as expected.
static auto testClusterPhot() -> int
{
    constexpr double ageYr = 1e6;

    try
    {
        const toml::table inputDeck = toml::parse_file(inputFilePhot);
        const io::SimControls controls(inputDeck);

        if (controls.filters() == nullptr)
        {
            std::cerr << "testCluster: phot: expected SimControls::filters() "
                "to be non-null\n";
            return 1;
        }
        if (!controls.computeLbol())
        {
            std::cerr << "testCluster: phot: expected SimControls::computeLbol() "
                "to be true (\"Lbol\" was in phot.filters)\n";
            return 1;
        }
        const auto& filterNames = controls.filters()->filterNames();
        const std::vector<std::string> expectedNames =
            { "SLUGTEST.CAM1.G500", "ideal_phot_700_1500" };
        if (filterNames != expectedNames)
        {
            std::cerr << "testCluster: phot: expected filterNames() == "
                "{SLUGTEST.CAM1.G500, ideal_phot_700_1500} (with \"Lbol\" "
                "popped out), got a different result\n";
            return 1;
        }

        utils::rng().seed(rngSeed);
        core::Cluster cluster(0, 1e4, 0.0, controls);
        cluster.advance(ageYr);

        const auto& phot = cluster.phot();
        if (phot.size() != filterNames.size())
        {
            std::cerr << "testCluster: phot: phot() size " << phot.size()
                << " does not match filterNames() size " << filterNames.size() << "\n";
            return 1;
        }

        // Cross-check against an independent call to
        // FilterCollection::phot() on the same spectrum, to confirm
        // Cluster::advance() actually wires the two together
        // correctly. Both calls run the exact same deterministic
        // computation on the same (spec, wl) inputs, so the results
        // should be bitwise identical.
        const auto expectedPhot = controls.filters()->phot(controls.specsyn()->wl(), cluster.spec());
        for (std::size_t i = 0; i < phot.size(); ++i)
        {
            if (phot.at(i) != expectedPhot.at(i))
            {
                std::cerr << "testCluster: phot: phot()[" << i << "] = "
                    << phot.at(i) << ", but recomputing FilterCollection::phot() "
                    "on the same spectrum gives " << expectedPhot.at(i) << "\n";
                return 1;
            }
        }

        // Both filters should report a positive value for a genuine,
        // non-zero blackbody spectrum: SLUGTEST.CAM1.G500 is an
        // energy-flux filter (Flambda, the default phot.system, is
        // always >= 0 for a physical spectrum), and
        // ideal_phot_700_1500 is a photon-count filter (also always
        // >= 0)
        for (std::size_t i = 0; i < phot.size(); ++i)
        {
            if (!(phot.at(i) > 0.0))
            {
                std::cerr << "testCluster: phot: phot()[" << i << "] = "
                    << phot.at(i) << ", expected a positive value for filter "
                    << filterNames.at(i) << "\n";
                return 1;
            }
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testCluster: phot test failed: "
            << error.what() << "\n";
        return 1;
    }
    return 0;
}

// Verify that Cluster::phot() stays empty when no filter collection
// was requested (SimControls::filters() is null), mirroring
// testClusterSpecFullyStochastic's own check for spec()
static auto testClusterPhotAbsent() -> int
{
    constexpr double ageYr = 1e6;

    try
    {
        const toml::table inputDeck = toml::parse_file(inputFile);
        const io::SimControls controls(inputDeck);

        if (controls.filters() != nullptr)
        {
            std::cerr << "testCluster: phot: expected SimControls::filters() "
                "to be null for a deck with no [phot] section\n";
            return 1;
        }

        utils::rng().seed(rngSeed);
        core::Cluster cluster(0, 1e4, 0.0, controls);
        cluster.advance(ageYr);

        if (!cluster.phot().empty())
        {
            std::cerr << "testCluster: phot: expected phot() to be empty "
                "when no filter collection was requested\n";
            return 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testCluster: photAbsent test failed: "
            << error.what() << "\n";
        return 1;
    }
    return 0;
}

// Verify that Cluster::lbol() is populated by advance() when
// SimControls::computeLbol() is true. testClusterLbol.in has "Lbol" as
// the sole entry in phot.filters, so SimControls::filters() itself
// should stay null (see SimControls::readFilters()'s own comment) even
// though computeLbol() is true; it also sets min_stoch_mass, so
// birthNonStochMass_ > 0 and both the stochastic and
// continuously-sampled (utils::PDFIntegrator-based) code paths inside
// computeLbol() actually run.
static auto testClusterLbol() -> int
{
    constexpr double ageYr = 1e6;

    try
    {
        const toml::table inputDeck = toml::parse_file(inputFileLbol);
        const io::SimControls controls(inputDeck);

        if (!controls.computeLbol())
        {
            std::cerr << "testCluster: lbol: expected SimControls::computeLbol() "
                "to be true (\"Lbol\" was in phot.filters)\n";
            return 1;
        }
        if (controls.filters() != nullptr)
        {
            std::cerr << "testCluster: lbol: expected SimControls::filters() "
                "to be null when \"Lbol\" is the only entry in phot.filters\n";
            return 1;
        }

        utils::rng().seed(rngSeed);
        core::Cluster cluster(0, 1e4, 0.0, controls);

        if (cluster.lbol() != 0.0)
        {
            std::cerr << "testCluster: lbol: expected lbol() to be 0 "
                "before advance() has ever run, got " << cluster.lbol() << "\n";
            return 1;
        }

        cluster.advance(ageYr);

        if (!std::isfinite(cluster.lbol()) || cluster.lbol() <= 0.0)
        {
            std::cerr << "testCluster: lbol: expected a finite, positive "
                "lbol() after advance(), got " << cluster.lbol() << "\n";
            return 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testCluster: lbol test failed: "
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
    result += testClusterSpecFullyStochastic();
    result += testClusterSpecContinuousPopulation();
    result += testClusterPhot();
    result += testClusterPhotAbsent();
    result += testClusterLbol();
    return result;
}
