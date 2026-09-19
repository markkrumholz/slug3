/**
 * @file testCluster.cpp
 * @author Mark Krumholz
 * @brief Unit tests for the Cluster class.
 * @date 2026-07-15
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#include "../src/core/Cluster.hpp"
#include "../src/interpolation/Interpolator1D.hpp"
#include "../src/io/SimControls.hpp"
#include "../src/phot/FilterCollection.hpp"
#include "../src/utils/HDF5Utils.hpp"
#include "../src/utils/MiscUtils.hpp"
#include "../src/utils/PDFIntegrator.hpp"
#include "../src/utils/RngThread.hpp"
#include "../src/yields/Yields.hpp"
#include "hdf5.h" // NOLINT(misc-include-cleaner) -- see HDF5Utils.hpp's own comment on including hdf5.h wholesale
#include "testCluster.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <functional>
#include <iostream>
#include <iterator>
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
static constexpr std::string_view inputFileExtinct = "tests/core/assets/testClusterExtinct.in";
static constexpr std::string_view yieldsRegistry = "tests/yields/assets/yields.toml";
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
        const auto lmr = controls.tracks()->liveMassRange(logAge, 0.0);
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

// Verify that Cluster::aV(), Cluster::specExtinct() and
// Cluster::photExtinct() are populated by advance() when extinction is
// requested (extinct.AV/extinct.model given in the input deck). There's
// no reference output to check against yet (that comes with a later,
// dedicated commit) -- this just exercises the whole extinction code
// path end to end and checks the results are physically sane: aV()
// matches the deck's fixed AV, specExtinct() and photExtinct() are the
// right sizes, and extinction makes both dimmer than their unextincted
// counterparts.
static auto testClusterExtinct() -> int
{
    constexpr double ageYr = 1e6;
    constexpr double expectedAV = 1.0;

    try
    {
        const toml::table inputDeck = toml::parse_file(inputFileExtinct);
        const io::SimControls controls(inputDeck);

        if (controls.extinct() == nullptr)
        {
            std::cerr << "testCluster: extinct: expected SimControls::extinct() "
                "to be non-null\n";
            return 1;
        }

        utils::rng().seed(rngSeed);
        core::Cluster cluster(0, 1e4, 0.0, controls);

        if (!utils::approxEqual(cluster.aV(), expectedAV))
        {
            std::cerr << "testCluster: extinct: expected aV() == " << expectedAV
                << ", got " << cluster.aV() << "\n";
            return 1;
        }

        cluster.advance(ageYr);

        // specExtinct() lives on extinct()->wl(), which is spec()'s own
        // wavelength grid (controls.specsyn()->wl()) clipped to the
        // extinction curve's native coverage -- generally narrower than
        // spec() itself, so align by locating where extinct()->wl()
        // starts within spec()'s own grid rather than assuming the two
        // are the same size
        const auto& spec = cluster.spec();
        const auto& specExtinct = cluster.specExtinct();
        const auto& fullWl = controls.specsyn()->wl();
        const auto& extWl = controls.extinct()->wl();
        if (specExtinct.size() != extWl.size())
        {
            std::cerr << "testCluster: extinct: specExtinct() size "
                << specExtinct.size() << " does not match extinct()->wl() size "
                << extWl.size() << "\n";
            return 1;
        }
        const auto offsetIt = std::ranges::find(fullWl, extWl.front());
        if (offsetIt == fullWl.end())
        {
            std::cerr << "testCluster: extinct: test bug: extinct()->wl() "
                "front is not on spec()'s own wavelength grid\n";
            return 1;
        }
        const auto offset = static_cast<std::size_t>(std::distance(fullWl.begin(), offsetIt));
        for (std::size_t i = 0; i < specExtinct.size(); ++i)
        {
            if (!(specExtinct.at(i) <= spec.at(offset + i)))
            {
                std::cerr << "testCluster: extinct: specExtinct()[" << i
                    << "] = " << specExtinct.at(i) << " should be <= spec()["
                    << offset + i << "] = " << spec.at(offset + i) << "\n";
                return 1;
            }
        }

        const auto& phot = cluster.phot();
        const auto& photExtinct = cluster.photExtinct();
        if (photExtinct.size() != phot.size())
        {
            std::cerr << "testCluster: extinct: photExtinct() size "
                << photExtinct.size() << " does not match phot() size "
                << phot.size() << "\n";
            return 1;
        }
        for (std::size_t i = 0; i < phot.size(); ++i)
        {
            if (!(photExtinct.at(i) <= phot.at(i)))
            {
                std::cerr << "testCluster: extinct: photExtinct()[" << i
                    << "] = " << photExtinct.at(i) << " should be <= phot()["
                    << i << "] = " << phot.at(i) << "\n";
                return 1;
            }
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testCluster: extinct test failed: "
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

// Verify that Cluster's own specNeb()/specNebExtinct()/lineLum()/
// lineLumExtinct()/photNeb()/photNebExtinct() agree, bit-for-bit, with an independent
// recomputation via SimControls::nebular()'s own public getCluster(),
// applied to cluster.spec() at cluster.feH() and this test's own known
// age -- mirrors testClusterExtinct()'s own general shape, but for
// nebular emission rather than dust extinction. Reuses
// testClusterExtinct.in (already combining phot + extinct), overriding
// its own [nebular] table to point at tests/nebular/assets/
// nebular_test.h5 (see that fixture's own generator,
// data/tools/cloudy/make_nebular_test_fixture.py, for its schema),
// exactly as testGalaxy.cpp's own testFieldStarsExtinct() overrides
// [extinct] on its own base deck.
static auto testClusterNebular() -> int
{
    constexpr double ageYr = 1e7; // an exact hit on the fixture's own tabulated cluster ages

    try
    {
        toml::table inputDeck = toml::parse_file(inputFileExtinct);
        inputDeck.at_path("nebular").as_table()->insert_or_assign("compute_neb", true);
        inputDeck.at_path("nebular").as_table()->insert_or_assign(
            "table", std::string("tests/nebular/assets/nebular_test.h5"));
        const io::SimControls controls(inputDeck);

        if (controls.nebular() == nullptr)
        {
            std::cerr << "testCluster: nebular: expected SimControls::nebular() "
                "to be non-null\n";
            return 1;
        }

        utils::rng().seed(rngSeed);
        core::Cluster cluster(0, 1e4, 0.0, controls);
        cluster.advance(ageYr);

        // Force spec()/specExtinct() current before independently
        // recomputing from them, exactly as Cluster::computeSpec()
        // itself does internally (spec_ before specNeb_/specNebExtinct_)
        const auto& spec = cluster.spec();
        const auto [expectedSpecNeb, expectedLineLum] =
            controls.nebular()->getCluster(spec, cluster.feH(), ageYr);

        if (cluster.specNeb() != expectedSpecNeb)
        {
            std::cerr << "testCluster: nebular: specNeb() does not match the "
                "independently-recomputed getCluster() result\n";
            return 1;
        }
        if (cluster.lineLum() != expectedLineLum)
        {
            std::cerr << "testCluster: nebular: lineLum() does not match the "
                "independently-recomputed getCluster() result\n";
            return 1;
        }
        if (std::reduce(expectedSpecNeb.begin(), expectedSpecNeb.end(), 0.0) <= 0.0)
        {
            std::cerr << "testCluster: nebular: expected a non-zero specNeb()\n";
            return 1;
        }

        const auto ext = controls.extinct();
        if (ext == nullptr)
        {
            std::cerr << "testCluster: nebular: test bug: expected "
                "extinct() non-null\n";
            return 1;
        }
        const auto expectedSpecNebExtinct = ext->applyExtinction(cluster.aV(), cluster.specNeb());
        if (cluster.specNebExtinct() != expectedSpecNebExtinct)
        {
            std::cerr << "testCluster: nebular: specNebExtinct() does not match "
                "ext->applyExtinction(aV(), specNeb())\n";
            return 1;
        }
        const auto expectedLineLumExtinct = ext->applyExtinctionLines(cluster.aV(), cluster.lineLum());
        if (cluster.lineLumExtinct() != expectedLineLumExtinct)
        {
            std::cerr << "testCluster: nebular: lineLumExtinct() does not match "
                "ext->applyExtinctionLines(aV(), lineLum())\n";
            return 1;
        }

        if (controls.filters() == nullptr)
        {
            std::cerr << "testCluster: nebular: test bug: expected "
                "filters() non-null\n";
            return 1;
        }
        const auto expectedPhotNeb =
            controls.filters()->phot(controls.specsyn()->wlObs(), cluster.specNeb());
        if (cluster.photNeb() != expectedPhotNeb)
        {
            std::cerr << "testCluster: nebular: photNeb() does not match "
                "filters->phot(wlObs(), specNeb())\n";
            return 1;
        }
        const auto expectedPhotNebExtinct =
            controls.filters()->phot(ext->wlObs(), cluster.specNebExtinct());
        if (cluster.photNebExtinct() != expectedPhotNebExtinct)
        {
            std::cerr << "testCluster: nebular: photNebExtinct() does not match "
                "filters->phot(ext->wlObs(), specNebExtinct())\n";
            return 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testCluster: nebular test failed: "
            << error.what() << "\n";
        return 1;
    }
    return 0;
}

// Unit test for Extinct's own line-luminosity extinction support
// (extinctLines_/applyExtinctionLines()/applyExtinctionCtsLines()),
// exercised through controls.extinct() directly rather than through
// Cluster (which does not yet apply extinction to line luminosities
// itself). Reuses the same testClusterExtinct.in + nebular-fixture-
// override combination as testClusterNebular() above, so both
// extinct() and nebular() are non-null. Independently re-derives the
// expected per-line extinction from the extinction curve's own raw
// HDF5 data (bypassing Extinct entirely, mirroring
// tests/extinct/testExtinct.hpp's own ground-truth strategy): both
// Extinct's internal interpolator and this test's own rawInterp are
// the exact same Interpolator1D<1> built from the exact same
// (wlRaw, kappaRaw), so they agree at any shared query point, not
// just the native grid's own nodes -- letting norm (the V-band
// normalization factor Extinct's own normalize() applies internally,
// otherwise unobservable) be recovered from a single reference point
// on controls.extinct()->wl()/extinct(), with no need to hit an exact
// native-grid wavelength.
static auto testClusterExtinctLines() -> int
{
    try
    {
        toml::table inputDeck = toml::parse_file(inputFileExtinct);
        inputDeck.at_path("nebular").as_table()->insert_or_assign("compute_neb", true);
        inputDeck.at_path("nebular").as_table()->insert_or_assign(
            "table", std::string("tests/nebular/assets/nebular_test.h5"));
        const io::SimControls controls(inputDeck);

        const auto ext = controls.extinct();
        if (ext == nullptr)
        {
            std::cerr << "testCluster: extinct lines: test bug: expected "
                "SimControls::extinct() to be non-null\n";
            return 1;
        }
        if (controls.nebular() == nullptr)
        {
            std::cerr << "testCluster: extinct lines: test bug: expected "
                "SimControls::nebular() to be non-null\n";
            return 1;
        }

        // Ground truth: the same curve's raw (wavelength, kappa) data,
        // read directly from the HDF5 file -- see
        // tests/extinct/testExtinct.hpp's own identical read, for the
        // same "Calzetti_starburst" curve testClusterExtinct.in itself
        // names as extinct.model
        // NOLINTBEGIN(misc-include-cleaner) -- see HDF5Utils.hpp's own comment
        const hid_t file = H5Fopen("data/extinct/extinct.h5", H5F_ACC_RDONLY, H5P_DEFAULT);
        const hid_t grp = H5Gopen2(file, "Calzetti_starburst", H5P_DEFAULT);
        const auto wlRaw = utils::readDataset1D(grp, "wavelength", "testCluster");
        const auto kappaRaw = utils::readDataset1D(grp, "kappa", "testCluster");
        H5Gclose(grp);
        H5Fclose(file);
        // NOLINTEND(misc-include-cleaner)
        const interp::Interpolator1D<1> rawInterp(wlRaw, kappaRaw);

        // Recover normalize()'s own scale factor from a single
        // reference point already on ext.wl()/ext.extinct() -- see
        // this function's own docstring for why any point works
        const double refWl = ext->wl().front();
        const double norm = ext->extinct().front() / rawInterp(refWl);

        const auto& lineWl = controls.nebular()->lineWl();
        std::vector<double> expectedExtinctLines(lineWl.size());
        std::vector<double> lineLum(lineWl.size());
        for (std::size_t ell = 0; ell < lineWl.size(); ++ell)
        {
            expectedExtinctLines.at(ell) =
                (lineWl.at(ell) >= rawInterp.xMin() && lineWl.at(ell) <= rawInterp.xMax()) ?
                norm * rawInterp(lineWl.at(ell)) : 0.0;
            lineLum.at(ell) = 1.0 + static_cast<double>(ell); // varied, easy-to-check values
        }

        // applyExtinctionLines(): each line multiplied by
        // exp(-A_V * expectedExtinctLines) at a known, single A_V
        constexpr double AV = 1.3; // NOLINT(readability-identifier-naming) -- see Extinct::applyExtinction()'s own identical NOLINT
        const auto result = ext->applyExtinctionLines(AV, lineLum);
        if (result.size() != lineWl.size())
        {
            std::cerr << "testCluster: extinct lines: applyExtinctionLines() "
                "returned " << result.size() << " lines, expected "
                << lineWl.size() << "\n";
            return 1;
        }
        for (std::size_t ell = 0; ell < lineWl.size(); ++ell)
        {
            const double expected = lineLum.at(ell) *
                std::exp(-AV * expectedExtinctLines.at(ell));
            if (!utils::approxEqual(result.at(ell), expected))
            {
                std::cerr << "testCluster: extinct lines: applyExtinctionLines() "
                    "line " << ell << " = " << result.at(ell) << ", expected "
                    << expected << "\n";
                return 1;
            }
        }

        // applyExtinctionCtsLines(): testClusterExtinct.in sets
        // extinct.AV but not extinct.AV_field, so avDistField() is a
        // valid delta at A_V = 0 (see SimControls::readExtinct()'s own
        // comment) -- computeExtinctionFacCtsLines() treats this as
        // the degenerate case A_V = 0, so every line should come back
        // completely unattenuated (exp(0) = 1), exactly like
        // testExtinctApplyExtinctionCtsInvalid()'s own analogous
        // finding for the spectral (non-line) case
        const auto ctsResult = ext->applyExtinctionCtsLines(lineLum);
        if (ctsResult.size() != lineWl.size())
        {
            std::cerr << "testCluster: extinct lines: applyExtinctionCtsLines() "
                "returned " << ctsResult.size() << " lines, expected "
                << lineWl.size() << "\n";
            return 1;
        }
        for (std::size_t ell = 0; ell < lineWl.size(); ++ell)
        {
            if (!utils::approxEqual(ctsResult.at(ell), lineLum.at(ell)))
            {
                std::cerr << "testCluster: extinct lines: applyExtinctionCtsLines() "
                    "line " << ell << " = " << ctsResult.at(ell) << ", expected "
                    "unattenuated " << lineLum.at(ell) << "\n";
                return 1;
            }
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testCluster: extinct lines test failed: "
            << error.what() << "\n";
        return 1;
    }
    return 0;
}

// Verify Cluster::yields()'s stochastic (individually-sampled)
// contribution by comparing against an independent recomputation from
// deadStarMasses() and SimControls::yields()'s own yieldSum() --
// mirrors testClusterExtinctLines()'s own "bit-for-bit independent
// recomputation" style. Uses testCluster.in unmodified -- its own
// min_stoch_mass defaults to 0, so the whole population is stochastic,
// same as testClusterSpecFullyStochastic()'s own setup -- with a
// single ccsn/sukhbold_test yield channel added; at ageYr = 5e6,
// testClusterAdvance() (same base deck/mass/age) already confirms
// some stars have died, and the turnoff mass at that age (~70 Msun)
// lies within sukhbold_test's own native range ([18.2, 100]).
static auto testClusterYieldsStochastic() -> int
{
    constexpr double ageYr = 5e6;
    constexpr double clusterMass = 1e4;
    constexpr double tol = 1e-9;

    try
    {
        toml::table inputDeck = toml::parse_file(inputFile);
        inputDeck.insert("yields", toml::table{
            { "channel1", toml::table{ { "channel", "ccsn" }, { "model", "sukhbold_test" } } },
            { "registry", std::string(yieldsRegistry) },
        });
        const io::SimControls controls(inputDeck);

        if (controls.yields() == nullptr)
        {
            std::cerr << "testCluster: yieldsStochastic: expected yields() non-null\n";
            return 1;
        }

        utils::rng().seed(rngSeed);
        core::Cluster cluster(0, clusterMass, 0.0, controls);

        if (std::reduce(cluster.yields().begin(), cluster.yields().end(), 0.0) != 0.0)
        {
            std::cerr << "testCluster: yieldsStochastic: expected yields() all zero "
                "before advance() has ever run\n";
            return 1;
        }

        cluster.advance(ageYr);

        const auto& dead = cluster.deadStarMasses();
        if (dead.empty())
        {
            std::cerr << "testCluster: yieldsStochastic: expected some dead stars at "
                << ageYr << " yr\n";
            return 1;
        }

        // Independently recompute the expected total by summing
        // controls.yields()->yieldSum() over every dead star
        const std::size_t niso = controls.yields()->isotopes().size();
        std::vector<double> expected(niso, 0.0);
        for (const double m : dead)
        {
            const auto sum = controls.yields()->yieldSum(m, cluster.feH());
            for (std::size_t j = 0; j < niso; ++j) { expected[j] += sum[j]; }
        }

        const auto& actual = cluster.yields();
        if (actual.size() != expected.size())
        {
            std::cerr << "testCluster: yieldsStochastic: yields() has size " <<
                actual.size() << ", expected " << expected.size() << "\n";
            return 1;
        }
        for (std::size_t j = 0; j < niso; ++j)
        {
            if (std::abs(actual[j] - expected[j]) > tol * std::max(1.0, std::abs(expected[j])))
            {
                std::cerr << "testCluster: yieldsStochastic: yields()[" << j << "] = " <<
                    actual[j] << ", expected " << expected[j] << "\n";
                return 1;
            }
        }
        if (std::reduce(expected.begin(), expected.end(), 0.0) <= 0.0)
        {
            std::cerr << "testCluster: yieldsStochastic: expected a positive total yield\n";
            return 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testCluster: yieldsStochastic test failed: " << error.what() << "\n";
        return 1;
    }
    return 0;
}

// Verify Cluster::yields()'s continuously-sampled (non-stochastic)
// contribution by comparing against an independent recomputation using
// the same live-mass-range-difference and utils::PDFIntegrator
// machinery Cluster::computeYields() itself uses internally. Sets
// min_stoch_mass to chabrier.toml's own maximum mass (120), exactly as
// tests/core/assets/testClusterSpecsynFullNonStoch.in's own comment
// describes, to force fracStochMass_ to exactly 0 -- the whole
// population ends up continuously-sampled, starMasses()/
// deadStarMasses() both staying empty throughout, isolating this path
// from testClusterYieldsStochastic()'s own.
static auto testClusterYieldsNonStochastic() -> int
{
    constexpr double ageYr = 5e6;
    constexpr double clusterMass = 1e4;
    constexpr double tol = 1e-9;

    try
    {
        toml::table inputDeck = toml::parse_file(inputFile);
        inputDeck.at_path("stars").as_table()->insert_or_assign("min_stoch_mass", 120.0);
        inputDeck.insert("yields", toml::table{
            { "channel1", toml::table{ { "channel", "ccsn" }, { "model", "sukhbold_test" } } },
            { "registry", std::string(yieldsRegistry) },
        });
        const io::SimControls controls(inputDeck);

        utils::rng().seed(rngSeed);
        core::Cluster cluster(0, clusterMass, 0.0, controls);

        if (!cluster.starMasses().empty())
        {
            std::cerr << "testCluster: yieldsNonStochastic: expected an empty "
                "starMasses() with min_stoch_mass at the IMF's own maximum\n";
            return 1;
        }

        cluster.advance(ageYr);

        if (!cluster.deadStarMasses().empty())
        {
            std::cerr << "testCluster: yieldsNonStochastic: expected an empty "
                "deadStarMasses(): no star should ever be individually sampled\n";
            return 1;
        }

        // Live mass range at birth and at ageYr, read from this
        // cluster's own tracks() -- exactly what computeYields()
        // itself queries (see its own comment). Both are single
        // intervals sharing the same lower bound for this track
        // fixture at these ages (only the upper "turnoff" bound
        // shrinks with age); verified below, rather than assumed,
        // since that is what lets a plain (turnoffNow, turnoffBirth]
        // stand in for the general (possibly multi-segment) set
        // difference computeYields() itself computes.
        const auto& tr = cluster.tracks();
        const auto birthRange = tr.liveMassRange(tr.logTMin());
        const auto nowRange = tr.liveMassRange(
            std::max(std::log10(ageYr), tr.logTMin()));
        if (birthRange.size() != 1 || nowRange.size() != 1 ||
            birthRange.front().first != nowRange.front().first)
        {
            std::cerr << "testCluster: yieldsNonStochastic: live mass range shape "
                "assumption violated -- test needs updating\n";
            return 1;
        }
        const double m0 = nowRange.front().second;
        const double m1 = std::min(birthRange.front().second, controls.minStochMass());
        if (m0 >= m1)
        {
            std::cerr << "testCluster: yieldsNonStochastic: expected a non-empty "
                "died-since-birth mass range at " << ageYr << " yr\n";
            return 1;
        }

        const std::size_t niso = controls.yields()->isotopes().size();
        const std::function<std::vector<double>(double)> integrand =
            [&controls, &cluster](const double m) -> std::vector<double>
            { return controls.yields()->yieldSum(m, cluster.feH()); };
        const utils::PDFIntegrator<std::function<std::vector<double>(double)>> integrator(
            controls.imf(), integrand, niso,
            false, controls.intMaxIter(), controls.intAbsTol(), controls.intRelTol());
        const auto integral = integrator.integrate(m0, m1);

        const auto& actual = cluster.yields();
        if (actual.size() != niso)
        {
            std::cerr << "testCluster: yieldsNonStochastic: yields() has size " <<
                actual.size() << ", expected " << niso << "\n";
            return 1;
        }
        for (std::size_t j = 0; j < niso; ++j)
        {
            const double expected = integral[j] * clusterMass;
            if (std::abs(actual[j] - expected) > tol * std::max(1.0, std::abs(expected)))
            {
                std::cerr << "testCluster: yieldsNonStochastic: yields()[" << j << "] = " <<
                    actual[j] << ", expected " << expected << "\n";
                return 1;
            }
        }
        if (std::reduce(actual.begin(), actual.end(), 0.0) <= 0.0)
        {
            std::cerr << "testCluster: yieldsNonStochastic: expected a positive total yield\n";
            return 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testCluster: yieldsNonStochastic test failed: " << error.what() << "\n";
        return 1;
    }
    return 0;
}

// Regression test: verify that stochastic deaths from an earlier
// advance() call are not silently lost when yields() is never called
// in between two advance() calls. Before computeYields() was moved to
// run eagerly at the end of advance() itself (see lastYieldTime_'s own
// comment), mDead_ -- which only ever holds the deaths from the single
// most recently advance() call, see updateLivingStars()'s own comment
// -- would be overwritten by the second advance() call's own
// updateLivingStars() before ever being consumed into yields_, since
// nothing but a lazy yields() call used to trigger that consumption.
// Same base setup as testClusterYieldsStochastic(), but splits its own
// single advance(ageYr) into several steps (rather than guessing a
// single split point likely to straddle a death, which proved fragile
// against this rngSeed's own draw), deliberately never calling
// yields() until after all of them.
static auto testClusterYieldsMultipleAdvanceCalls() -> int
{
    constexpr double ageYr = 5e6;
    constexpr std::size_t nSteps = 5;
    // 10x testClusterYieldsStochastic()'s own clusterMass: at this
    // rngSeed, that smaller cluster's own most massive stars turn out
    // to all die within the last of these steps (empirically, none
    // before it) -- a bigger cluster samples further into the IMF's
    // own tail, giving some stars massive (and so short-lived) enough
    // to have already died in an earlier step too.
    constexpr double clusterMass = 1e5;
    constexpr double tol = 1e-9;

    try
    {
        toml::table inputDeck = toml::parse_file(inputFile);
        inputDeck.insert("yields", toml::table{
            { "channel1", toml::table{ { "channel", "ccsn" }, { "model", "sukhbold_test" } } },
            { "registry", std::string(yieldsRegistry) },
        });
        const io::SimControls controls(inputDeck);

        utils::rng().seed(rngSeed);
        core::Cluster cluster(0, clusterMass, 0.0, controls);

        const std::size_t niso = controls.yields()->isotopes().size();
        std::vector<double> expected(niso, 0.0);
        bool sawEarlyDeath = false;

        // Advance in nSteps equal steps, deliberately not calling
        // yields() until after the last one -- every step but the
        // last, having at least one death, is exactly the scenario
        // that used to lose that step's own mDead_ once yields() was
        // only ever computed lazily (see lastYieldTime_'s own comment)
        for (std::size_t s = 1; s <= nSteps; ++s)
        {
            cluster.advance(ageYr * static_cast<double>(s) / static_cast<double>(nSteps));
            const auto dead = cluster.deadStarMasses();
            if (!dead.empty() && s < nSteps) { sawEarlyDeath = true; }
            for (const double m : dead)
            {
                const auto sum = controls.yields()->yieldSum(m, cluster.feH());
                for (std::size_t j = 0; j < niso; ++j) { expected[j] += sum[j]; }
            }
        }
        if (!sawEarlyDeath)
        {
            std::cerr << "testCluster: yieldsMultipleAdvanceCalls: test bug: expected "
                "some dead stars before the last of " << nSteps << " advance() calls\n";
            return 1;
        }

        const auto& actual = cluster.yields();
        if (actual.size() != expected.size())
        {
            std::cerr << "testCluster: yieldsMultipleAdvanceCalls: yields() has size " <<
                actual.size() << ", expected " << expected.size() << "\n";
            return 1;
        }
        for (std::size_t j = 0; j < niso; ++j)
        {
            if (std::abs(actual[j] - expected[j]) > tol * std::max(1.0, std::abs(expected[j])))
            {
                std::cerr << "testCluster: yieldsMultipleAdvanceCalls: yields()[" << j <<
                    "] = " << actual[j] << ", expected " << expected[j] <<
                    " -- deaths from the first advance() call may have been lost\n";
                return 1;
            }
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testCluster: yieldsMultipleAdvanceCalls test failed: " << error.what() << "\n";
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
    result += testClusterExtinct();
    result += testClusterLbol();
    result += testClusterNebular();
    result += testClusterExtinctLines();
    result += testClusterYieldsStochastic();
    result += testClusterYieldsNonStochastic();
    result += testClusterYieldsMultipleAdvanceCalls();
    return result;
}
