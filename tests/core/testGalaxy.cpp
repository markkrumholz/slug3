/**
 * @file testGalaxy.cpp
 * @author Mark Krumholz
 * @brief Unit tests for the Galaxy class.
 * @date 2026-08-10
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#include "../src/core/Cluster.hpp"
#include "../src/core/Galaxy.hpp"
#include "../src/elem/IsotopeTable.hpp"
#include "../src/interpolation/Interpolator1D.hpp"
#include "../src/io/SimControls.hpp"
#include "../src/phot/FilterCollection.hpp"
#include "../src/utils/Constants.hpp"
#include "../src/utils/MiscUtils.hpp"
#include "../src/utils/RngThread.hpp"
#include "../src/utils/UniqueIDManager.hpp"
#include "testGalaxy.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
#include <iterator>
#include <limits>
#include <numeric>
#include <string>
#include <string_view>
#include <toml.hpp>
#include <vector>

// Note: distinct from tests/core/assets/testGalaxy.in, which
// testSimControls.cpp uses for its own input-deck-parsing checks --
// see testGalaxyDynamics.in's own comment.
static constexpr std::string_view inputFile = "tests/core/assets/testGalaxyDynamics.in";
static constexpr std::string_view inputFileBasics = "tests/core/assets/testGalaxyDynamicsBasics.in";
static constexpr unsigned int rngSeed = 42;

// Two advance() times, chosen relative to testGalaxyDynamics.in's clusters.CLF
// (a fixed 5e5 yr): every cluster formed in [0, t1] disrupts well
// before t2 (formTime + 5e5 <= t1 + 5e5 = 8e5 < t2), so the second
// advance() call is guaranteed to move at least the whole first-step
// population from clusters() to disruptedClusters(), giving both a
// non-empty disruptedClusters() and a mix of old/new clusters in
// clusters() to check.
static constexpr double t1 = 3e5;
static constexpr double t2 = 1.2e6;

// Loose enough to tolerate genuine Poisson-like scatter from a handful
// of stochastically-drawn cluster masses (testGalaxyDynamicsCMF.toml
// is a real power-law CMF, not a delta function -- see its own
// comment), tight enough to catch a real bug in how mNew/newMasses/
// formTime are computed.
static constexpr double massTolerance = 0.3;

// Verify that a freshly-constructed Galaxy starts with curTime() == 0,
// every list/vector empty, and lbol() == 0, before advance() has ever run.
static auto checkConstruction(const io::SimControls& controls) -> int
{
    core::Galaxy galaxy(controls);

    if (galaxy.curTime() != 0.0)
    {
        std::cerr << "testGalaxy: construction: expected curTime() == 0, got "
            << galaxy.curTime() << "\n";
        return 1;
    }
    if (!galaxy.clusters().empty() || !galaxy.disruptedClusters().empty())
    {
        std::cerr << "testGalaxy: construction: expected clusters() and "
            "disruptedClusters() to both be empty\n";
        return 1;
    }
    if (!galaxy.spec().empty() || !galaxy.specExtinct().empty() ||
        !galaxy.phot().empty() || !galaxy.photExtinct().empty())
    {
        std::cerr << "testGalaxy: construction: expected spec()/"
            "specExtinct()/phot()/photExtinct() to all be empty\n";
        return 1;
    }
    if (galaxy.lbol() != 0.0)
    {
        std::cerr << "testGalaxy: construction: expected lbol() == 0, got "
            << galaxy.lbol() << "\n";
        return 1;
    }
    return 0;
}

// Advance galaxy from prevTime to t, and verify that (1) the total
// target mass of clusters formed during this step is close to
// sfr().integral() over the step, and (2) every cluster formed during
// this step has a formTime() lying within the step's own time range.
// "Formed during this step" is identified by uid: uid values are
// handed out in strictly increasing order by the shared
// utils::uniqueID() service, so any cluster whose uid exceeds a marker
// captured immediately before advance() must have been created by it.
static auto advanceAndCheckMassAge(core::Galaxy& galaxy,
    const io::SimControls& controls, const double prevTime, const double t) -> int
{
    const auto marker = utils::uniqueID().get();
    galaxy.advance(t);

    std::vector<const core::Cluster*> newClusters;
    for (const auto& c : galaxy.clusters())
    { if (c.uid() > marker) { newClusters.push_back(&c); } }
    for (const auto& c : galaxy.disruptedClusters())
    { if (c.uid() > marker) { newClusters.push_back(&c); } }

    for (const auto* c : newClusters)
    {
        if (c->formTime() < prevTime || c->formTime() > t)
        {
            std::cerr << "testGalaxy: massAndAge: cluster uid " << c->uid()
                << " formed at " << c->formTime() << ", expected in ["
                << prevTime << ", " << t << "]\n";
            return 1;
        }
    }

    const double totalMass = std::accumulate(newClusters.begin(), newClusters.end(),
        0.0, [](const double sum, const core::Cluster* c)
        { return sum + c->targetMass(); });
    const double expectedMass = controls.sfr().integral(prevTime, t);
    if (std::abs(totalMass - expectedMass) > massTolerance * expectedMass)
    {
        std::cerr << "testGalaxy: massAndAge: total formed mass "
            << totalMass << " over [" << prevTime << ", " << t
            << "] deviates from target " << expectedMass
            << " by more than " << massTolerance * 100 << "%\n";
        return 1;
    }
    return 0;
}

// Verify that Galaxy::spec()/specExtinct() are exactly the sum of
// spec()/specExtinct() over every cluster in clusters() and
// disruptedClusters() -- this independently performs the identical
// summation, in the identical (clusters() then disruptedClusters())
// order, so the result should match bit for bit, not just approximately.
static auto checkSpecSum(core::Galaxy& galaxy, const io::SimControls& controls) -> int
{
    std::vector<double> expectedSpec(controls.specsyn()->wl().size(), 0.0);
    std::vector<double> expectedSpecExtinct(controls.extinct()->wl().size(), 0.0);
    const auto accumulate = [&](std::vector<core::Cluster>& list)
    {
        for (auto& c : list)
        {
            const auto& s = c.spec();
            for (std::size_t i = 0; i < expectedSpec.size(); ++i)
            { expectedSpec.at(i) += s.at(i); }
            const auto& se = c.specExtinct();
            for (std::size_t i = 0; i < expectedSpecExtinct.size(); ++i)
            { expectedSpecExtinct.at(i) += se.at(i); }
        }
    };
    accumulate(galaxy.clusters());
    accumulate(galaxy.disruptedClusters());

    if (galaxy.spec() != expectedSpec)
    {
        std::cerr << "testGalaxy: specSum: spec() does not equal the "
            "independently-summed per-cluster spec()\n";
        return 1;
    }
    if (galaxy.specExtinct() != expectedSpecExtinct)
    {
        std::cerr << "testGalaxy: specSum: specExtinct() does not equal the "
            "independently-summed per-cluster specExtinct()\n";
        return 1;
    }
    if (std::reduce(expectedSpec.begin(), expectedSpec.end(), 0.0) <= 0.0)
    {
        std::cerr << "testGalaxy: specSum: expected a non-zero galaxy "
            "spectrum after forming clusters\n";
        return 1;
    }
    return 0;
}

// Verify that Galaxy::phot()/photExtinct() -- computed from the summed
// spec()/specExtinct() -- match the sum of phot()/photExtinct() over
// every individual cluster, within a small tolerance. Unlike spec()/
// specExtinct() (a plain elementwise sum, checked exactly above),
// phot() applies a filter convolution (an integral against each
// filter's response) to the summed spectrum, which need not commute
// exactly with summation at the level of floating-point rounding, even
// though phot.system = Flambda (the default) is additive in principle:
// filters.phot() interpolates each spectrum before integrating, and
// interpolating-then-integrating the sum need not round identically to
// integrating each cluster's own spectrum and summing the results.
static auto checkPhotSum(core::Galaxy& galaxy, const io::SimControls& controls) -> int
{
    constexpr double photTolerance = 1e-6;

    const auto nFilt = controls.filters()->filterNames().size();
    std::vector<double> expectedPhot(nFilt, 0.0);
    std::vector<double> expectedPhotExtinct(nFilt, 0.0);
    const auto accumulate = [&](std::vector<core::Cluster>& list)
    {
        for (auto& c : list)
        {
            const auto& p = c.phot();
            for (std::size_t i = 0; i < nFilt; ++i) { expectedPhot.at(i) += p.at(i); }
            const auto& pe = c.photExtinct();
            for (std::size_t i = 0; i < nFilt; ++i) { expectedPhotExtinct.at(i) += pe.at(i); }
        }
    };
    accumulate(galaxy.clusters());
    accumulate(galaxy.disruptedClusters());

    const auto& phot = galaxy.phot();
    const auto& photExtinct = galaxy.photExtinct();
    if (phot.size() != nFilt || photExtinct.size() != nFilt)
    {
        std::cerr << "testGalaxy: photSum: phot()/photExtinct() size does "
            "not match the number of filters\n";
        return 1;
    }
    for (std::size_t i = 0; i < nFilt; ++i)
    {
        if (std::abs(phot.at(i) - expectedPhot.at(i)) >
            photTolerance * std::abs(expectedPhot.at(i)))
        {
            std::cerr << "testGalaxy: photSum: phot()[" << i << "] = "
                << phot.at(i) << ", but summing individual clusters' "
                "phot() gives " << expectedPhot.at(i) << "\n";
            return 1;
        }
        if (std::abs(photExtinct.at(i) - expectedPhotExtinct.at(i)) >
            photTolerance * std::abs(expectedPhotExtinct.at(i)))
        {
            std::cerr << "testGalaxy: photSum: photExtinct()[" << i << "] = "
                << photExtinct.at(i) << ", but summing individual clusters' "
                "photExtinct() gives " << expectedPhotExtinct.at(i) << "\n";
            return 1;
        }
    }
    return 0;
}

// Verify that Galaxy::lbol() equals the sum of lbol() over every
// cluster in clusters() and disruptedClusters()
static auto checkLbolSum(core::Galaxy& galaxy) -> int
{
    double expectedLbol = 0.0;
    for (auto& c : galaxy.clusters()) { expectedLbol += c.lbol(); }
    for (auto& c : galaxy.disruptedClusters()) { expectedLbol += c.lbol(); }

    if (galaxy.lbol() != expectedLbol)
    {
        std::cerr << "testGalaxy: lbolSum: lbol() = " << galaxy.lbol()
            << ", but summing individual clusters' lbol() gives "
            << expectedLbol << "\n";
        return 1;
    }
    if (!(galaxy.lbol() > 0.0))
    {
        std::cerr << "testGalaxy: lbolSum: expected a positive lbol() "
            "after forming clusters, got " << galaxy.lbol() << "\n";
        return 1;
    }
    return 0;
}

// Verify that, after the two-step advance (chosen so every first-step
// cluster's disruption time falls before t2 -- see t1/t2's own
// comment), disruptedClusters() is non-empty, every cluster in it
// reports isDisrupted() == true, and every cluster still in clusters()
// reports isDisrupted() == false
static auto checkDisruption(core::Galaxy& galaxy) -> int
{
    if (galaxy.disruptedClusters().empty())
    {
        std::cerr << "testGalaxy: disruption: expected a non-empty "
            "disruptedClusters() after advancing well past every "
            "first-step cluster's disruption time\n";
        return 1;
    }
    for (const auto& c : galaxy.disruptedClusters())
    {
        if (!c.isDisrupted())
        {
            std::cerr << "testGalaxy: disruption: cluster uid " << c.uid()
                << " is in disruptedClusters() but isDisrupted() is false\n";
            return 1;
        }
    }
    for (const auto& c : galaxy.clusters())
    {
        if (c.isDisrupted())
        {
            std::cerr << "testGalaxy: disruption: cluster uid " << c.uid()
                << " is in clusters() but isDisrupted() is true\n";
            return 1;
        }
    }
    return 0;
}

// Fast diagnostic: no spectra/phot/extinct in play (testGalaxyDynamicsBasics.in
// has no [spectra] section, so SimControls::specsyn() is null and
// Cluster::computeSpec()/computePhot() are no-ops), isolating
// Galaxy::advance()'s cluster/star-drawing machinery from the far more
// expensive spectral synthesis path. Prints cluster/star counts and
// age ranges to stderr so they can be eyeballed for sanity, in
// addition to the same mass/age checks advanceAndCheckMassAge() runs
// for the full (spectra-enabled) test.
static auto testGalaxyBasics() -> int
{
    try
    {
        const toml::table inputDeck = toml::parse_file(inputFileBasics);
        const io::SimControls controls(inputDeck);

        utils::rng().seed(rngSeed);
        core::Galaxy galaxy(controls);
        int result = advanceAndCheckMassAge(galaxy, controls, 0.0, t1);
        result += advanceAndCheckMassAge(galaxy, controls, t1, t2);

        const auto nClusters = galaxy.clusters().size() + galaxy.disruptedClusters().size();
        std::size_t nStarsTotal = 0;
        double minAge = std::numeric_limits<double>::max();
        double maxAge = std::numeric_limits<double>::lowest();
        const auto scan = [&](const std::vector<core::Cluster>& list)
        {
            for (const auto& c : list)
            {
                nStarsTotal += c.starMasses().size() + c.deadStarMasses().size();
                const double age = galaxy.curTime() - c.formTime();
                minAge = std::min(minAge, age);
                maxAge = std::max(maxAge, age);
            }
        };
        scan(galaxy.clusters());
        scan(galaxy.disruptedClusters());

        std::cerr << "testGalaxy: basics: " << nClusters << " clusters, "
            << nStarsTotal << " stars total, ages in ["
            << minAge << ", " << maxAge << "] yr (curTime = "
            << galaxy.curTime() << ")\n";

        if (nClusters == 0)
        {
            std::cerr << "testGalaxy: basics: expected at least one cluster "
                "to have formed\n";
            return 1;
        }
        if (nStarsTotal == 0)
        {
            std::cerr << "testGalaxy: basics: expected at least one star "
                "across all clusters\n";
            return 1;
        }
        if (minAge < 0.0 || maxAge > galaxy.curTime())
        {
            std::cerr << "testGalaxy: basics: cluster ages fall outside "
                "[0, curTime()]\n";
            return 1;
        }
        return result;
    }
    catch (const std::exception& error)
    {
        std::cerr << "testGalaxy: basics test failed: " << error.what() << "\n";
        return 1;
    }
}

// Verify that Galaxy::computeSpec() correctly adds the continuously-
// treated population's own spectral contribution (via
// Specsyn::specCts()'s continuous-population overload) when
// fCluster() < 1. Uses fCluster = 0.0 exactly (entirely continuous, no
// stochastic clusters at all) so galaxy.spec() is guaranteed to come
// *entirely* from the new code path, with no cluster-summed
// contribution to confound it -- clusters()/disruptedClusters() are
// both checked empty as a sanity check on that premise. inputFile's
// own stars.FeH = 0.0 (a constant) exercises specCts()'s 2D (time,
// mass) code path; testContinuousPopSpecMultiFeh below instead
// overrides it with a real distribution, exercising the 3D (time,
// feh, mass) path. See testContinuousPopSpecReferenceCheck for a
// stronger, independent correctness check of the same 2D case (this
// test only checks the result is finite, non-negative, and non-zero).
static auto testContinuousPopSpecSingleFeh() -> int
{
    try
    {
        toml::table inputDeck = toml::parse_file(inputFile);
        inputDeck.at_path("clusters").as_table()->insert("f_cluster", 0.0);
        // Overrides inputFile's own stars.min_stoch_mass = 10 up to
        // chabrier.toml's own IMF max mass, so that every non-clustered
        // star is treated as fully continuous, with no individual
        // FieldStar of its own -- see Galaxy::FieldStar's own comment
        // -- exactly preserving this test's own pre-field-star behavior.
        inputDeck.at_path("stars").as_table()->insert_or_assign("min_stoch_mass", 120.0);
        const io::SimControls controls(inputDeck);

        utils::rng().seed(rngSeed);
        core::Galaxy galaxy(controls);
        galaxy.advance(t1);

        if (!galaxy.clusters().empty() || !galaxy.disruptedClusters().empty())
        {
            std::cerr << "testGalaxy: continuousPopSpecSingleFeh: test bug: "
                "expected no stochastic clusters at all with f_cluster = 0\n";
            return 1;
        }

        const auto& spec = galaxy.spec();
        if (spec.size() != controls.specsyn()->wl().size())
        {
            std::cerr << "testGalaxy: continuousPopSpecSingleFeh: spec() size "
                << spec.size() << " does not match wl() size "
                << controls.specsyn()->wl().size() << "\n";
            return 1;
        }
        double total = 0.0;
        for (const double v : spec)
        {
            if (!std::isfinite(v) || v < 0.0)
            {
                std::cerr << "testGalaxy: continuousPopSpecSingleFeh: spec() "
                    "contains a non-finite or negative value\n";
                return 1;
            }
            total += v;
        }
        if (!(total > 0.0))
        {
            std::cerr << "testGalaxy: continuousPopSpecSingleFeh: expected a "
                "non-trivial spectrum from the continuous population, got "
                "all zeros\n";
            return 1;
        }

        // With clusters()/disruptedClusters() both empty, specExtinct()
        // comes entirely from the continuous population's own
        // (unattenuated) contribution -- see Galaxy::computeSpec()'s own
        // comment for why -- so specExtinct()[i] should exactly equal
        // spec()[wlOffset + i], the same underlying contSpec value at the
        // same wavelength, just offset by however many of spec()'s own
        // leading elements fall outside the extinction curve's own
        // narrower coverage (see Extinct::wlOffset()'s own comment).
        const auto wlOffset = controls.extinct()->wlOffset();
        const auto& specExtinct = galaxy.specExtinct();
        if (specExtinct.size() != controls.extinct()->wl().size())
        {
            std::cerr << "testGalaxy: continuousPopSpecSingleFeh: specExtinct() "
                "size does not match extinct()->wl() size\n";
            return 1;
        }
        for (std::size_t i = 0; i < specExtinct.size(); ++i)
        {
            if (specExtinct.at(i) != spec.at(wlOffset + i))
            {
                std::cerr << "testGalaxy: continuousPopSpecSingleFeh: "
                    "specExtinct()[" << i << "] does not exactly equal "
                    "spec()[" << (wlOffset + i) << "]\n";
                return 1;
            }
        }
        if (galaxy.phot().empty() || galaxy.photExtinct().empty())
        {
            std::cerr << "testGalaxy: continuousPopSpecSingleFeh: expected "
                "non-empty phot() and photExtinct()\n";
            return 1;
        }

        // inputFile's own phot.filters includes "Lbol" (see
        // io::SimControls::computeLbol()'s own comment), so this
        // exercises Specsyn::specAndLbolCts() rather than plain
        // specCts() -- see Galaxy::computeSpec()'s own comment. Lbol
        // must be finite, positive, and at least as large as the
        // luminosity already visible within spec()'s own covered
        // wavelength range (a trapezoidal integral of spec() over
        // wl(), converted from erg/s/Angstrom to erg/s): light outside
        // that range can only add to the true bolometric total, never
        // subtract from it, so this one-directional bound holds
        // regardless of the exact SED shape, while still catching a
        // badly broken Lbol calculation (e.g. wrong units, or Lbol not
        // really being computed at all).
        const double lbol = galaxy.lbol();
        if (!std::isfinite(lbol) || !(lbol > 0.0))
        {
            std::cerr << "testGalaxy: continuousPopSpecSingleFeh: expected a "
                "finite, positive lbol(), got " << lbol << "\n";
            return 1;
        }
        const auto& wl = controls.specsyn()->wl();
        double trapzErgS = 0.0;
        for (std::size_t k = 0; k + 1 < spec.size(); ++k)
        {
            trapzErgS += 0.5 * (spec.at(k) + spec.at(k + 1)) * (wl.at(k + 1) - wl.at(k));
        }
        constexpr double lbolTolerance = 1e-6; // floating-point/quadrature-level slop only
        if (lbol * utils::Lsun < trapzErgS * (1.0 - lbolTolerance))
        {
            std::cerr << "testGalaxy: continuousPopSpecSingleFeh: lbol() = "
                << lbol << " Lsun (" << (lbol * utils::Lsun) << " erg/s) is "
                "less than the luminosity already visible in spec()'s own "
                "covered wavelength range (" << trapzErgS << " erg/s)\n";
            return 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testGalaxy: continuousPopSpecSingleFeh test failed: "
            << error.what() << "\n";
        return 1;
    }
    return 0;
}

// Same as testContinuousPopSpecSingleFeh, but overrides stars.FeH with
// a real (non-degenerate) distribution -- flat over [-1, 0], well
// within MIST_test's own [-1, 0.5] grid (see
// tests/tracks/assets/tracks.toml) -- so specCts()'s 3D (time, feh,
// mass) code path is exercised instead of its 2D one.
static auto testContinuousPopSpecMultiFeh() -> int
{
    try
    {
        toml::table inputDeck = toml::parse_file(inputFile);
        inputDeck.at_path("clusters").as_table()->insert("f_cluster", 0.0);
        // Overrides inputFile's own stars.min_stoch_mass = 10 up to
        // chabrier.toml's own IMF max mass, so that every non-clustered
        // star is treated as fully continuous, with no individual
        // FieldStar of its own -- see Galaxy::FieldStar's own comment
        // -- exactly preserving this test's own pre-field-star behavior.
        inputDeck.at_path("stars").as_table()->insert_or_assign("min_stoch_mass", 120.0);
        inputDeck.at_path("stars").as_table()->insert_or_assign(
            "FeH", "tests/core/assets/testClusterSpecsynFullFeHDist.toml");
        const io::SimControls controls(inputDeck);

        utils::rng().seed(rngSeed);
        core::Galaxy galaxy(controls);
        galaxy.advance(t1);

        if (!galaxy.clusters().empty() || !galaxy.disruptedClusters().empty())
        {
            std::cerr << "testGalaxy: continuousPopSpecMultiFeh: test bug: "
                "expected no stochastic clusters at all with f_cluster = 0\n";
            return 1;
        }

        const auto& spec = galaxy.spec();
        if (spec.size() != controls.specsyn()->wl().size())
        {
            std::cerr << "testGalaxy: continuousPopSpecMultiFeh: spec() size "
                << spec.size() << " does not match wl() size "
                << controls.specsyn()->wl().size() << "\n";
            return 1;
        }
        double total = 0.0;
        for (const double v : spec)
        {
            if (!std::isfinite(v) || v < 0.0)
            {
                std::cerr << "testGalaxy: continuousPopSpecMultiFeh: spec() "
                    "contains a non-finite or negative value\n";
                return 1;
            }
            total += v;
        }
        if (!(total > 0.0))
        {
            std::cerr << "testGalaxy: continuousPopSpecMultiFeh: expected a "
                "non-trivial spectrum from the continuous population, got "
                "all zeros\n";
            return 1;
        }

        // See testContinuousPopSpecSingleFeh's own identical check for why
        // specExtinct()[i] should exactly equal spec()[wlOffset + i] here.
        const auto wlOffset = controls.extinct()->wlOffset();
        const auto& specExtinct = galaxy.specExtinct();
        if (specExtinct.size() != controls.extinct()->wl().size())
        {
            std::cerr << "testGalaxy: continuousPopSpecMultiFeh: specExtinct() "
                "size does not match extinct()->wl() size\n";
            return 1;
        }
        for (std::size_t i = 0; i < specExtinct.size(); ++i)
        {
            if (specExtinct.at(i) != spec.at(wlOffset + i))
            {
                std::cerr << "testGalaxy: continuousPopSpecMultiFeh: "
                    "specExtinct()[" << i << "] does not exactly equal "
                    "spec()[" << (wlOffset + i) << "]\n";
                return 1;
            }
        }
        if (galaxy.phot().empty() || galaxy.photExtinct().empty())
        {
            std::cerr << "testGalaxy: continuousPopSpecMultiFeh: expected "
                "non-empty phot() and photExtinct()\n";
            return 1;
        }

        // See testContinuousPopSpecSingleFeh's own identical check for
        // the rationale.
        const double lbol = galaxy.lbol();
        if (!std::isfinite(lbol) || !(lbol > 0.0))
        {
            std::cerr << "testGalaxy: continuousPopSpecMultiFeh: expected a "
                "finite, positive lbol(), got " << lbol << "\n";
            return 1;
        }
        const auto& wl = controls.specsyn()->wl();
        double trapzErgS = 0.0;
        for (std::size_t k = 0; k + 1 < spec.size(); ++k)
        {
            trapzErgS += 0.5 * (spec.at(k) + spec.at(k + 1)) * (wl.at(k + 1) - wl.at(k));
        }
        constexpr double lbolTolerance = 1e-6; // floating-point/quadrature-level slop only
        if (lbol * utils::Lsun < trapzErgS * (1.0 - lbolTolerance))
        {
            std::cerr << "testGalaxy: continuousPopSpecMultiFeh: lbol() = "
                << lbol << " Lsun (" << (lbol * utils::Lsun) << " erg/s) is "
                "less than the luminosity already visible in spec()'s own "
                "covered wavelength range (" << trapzErgS << " erg/s)\n";
            return 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testGalaxy: continuousPopSpecMultiFeh test failed: "
            << error.what() << "\n";
        return 1;
    }
    return 0;
}

// Cross-checks the new joint (time, mass) integral -- exercised via
// testContinuousPopSpecSingleFeh's own scenario (fCluster = 0,
// constant FeH) -- against an independent brute-force reference: a
// fixed-step Riemann sum over age, at each step calling the *other*,
// already-established specCts() overload (single fixed-age isochrone,
// integrated over mass alone) on that step's own isochrone, weighted
// by the star-forming mass in that step (sfr().integral() over it),
// exactly mirroring how a real continuous stellar population's light
// is built up from stars of many different ages. This doesn't rely on
// any of the new PDFIntegratorND/isochroneCache_ machinery at all, so
// close agreement between the two is strong evidence the new code
// path's own age/mass weighting and isochrone lookups are correct,
// not just that it runs without crashing (which is all
// testContinuousPopSpecSingleFeh itself checks).
static auto testContinuousPopSpecReferenceCheck() -> int
{
    constexpr double relTolerance = 0.05; // the brute-force reference has its own O(1/nSteps) discretization error
    constexpr int nSteps = 400;
    constexpr double feh = 0.0; // matches inputFile's own stars.FeH

    try
    {
        toml::table inputDeck = toml::parse_file(inputFile);
        inputDeck.at_path("clusters").as_table()->insert("f_cluster", 0.0);
        // Overrides inputFile's own stars.min_stoch_mass = 10 up to
        // chabrier.toml's own IMF max mass, so that every non-clustered
        // star is treated as fully continuous, with no individual
        // FieldStar of its own -- see Galaxy::FieldStar's own comment
        // -- exactly preserving this test's own pre-field-star behavior.
        inputDeck.at_path("stars").as_table()->insert_or_assign("min_stoch_mass", 120.0);
        const io::SimControls controls(inputDeck);

        utils::rng().seed(rngSeed);
        core::Galaxy galaxy(controls);
        galaxy.advance(t1);
        const auto& newSpec = galaxy.spec();

        const auto synth = controls.specsyn();
        const auto tracks2D = controls.tracks2D();
        const auto& imf = controls.imf();
        const auto& sfr = controls.sfr();

        std::vector<double> refSpec(synth->wl().size(), 0.0);
        const double dt = t1 / nSteps;
        for (int i = 0; i < nSteps; ++i)
        {
            const double tLo = i * dt;
            const double tHi = (i + 1) * dt;
            const double mStep = sfr.integral(tLo, tHi);
            if (mStep <= 0.0) { continue; }

            const double age = t1 - (0.5 * (tLo + tHi));
            const double logAge = std::max(std::log10(age), tracks2D->logTMin());
            const auto isochrone = tracks2D->getIsochrone(logAge);

            const auto stepSpec = synth->specCts(
                isochrone, imf, mStep, imf.getMin(), imf.getMax(), feh);
            for (std::size_t k = 0; k < refSpec.size(); ++k) { refSpec.at(k) += stepSpec.at(k); }
        }

        double newTotal = 0.0;
        double refTotal = 0.0;
        for (std::size_t k = 0; k < newSpec.size(); ++k)
        {
            newTotal += newSpec.at(k);
            refTotal += refSpec.at(k);
        }
        if (!(refTotal > 0.0))
        {
            std::cerr << "testGalaxy: continuousPopSpecReferenceCheck: test bug: "
                "brute-force reference spectrum is all zeros\n";
            return 1;
        }
        if (std::abs(newTotal - refTotal) > relTolerance * refTotal)
        {
            std::cerr << "testGalaxy: continuousPopSpecReferenceCheck: total "
                "flux from the new joint integral (" << newTotal << ") deviates "
                "from the brute-force per-age reference (" << refTotal <<
                ") by more than " << relTolerance * 100 << "%\n";
            return 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testGalaxy: continuousPopSpecReferenceCheck test failed: "
            << error.what() << "\n";
        return 1;
    }
    return 0;
}

// Verify that Galaxy::lbol()'s two different paths to the continuous
// population's own Lbol -- as a byproduct of computing a spectrum
// (Specsyn::specAndLbolCts(), via Galaxy::computeSpec()) versus
// standalone (Galaxy::computeLbolCts()), used respectively when spec()
// has or hasn't already been computed this step -- agree with each
// other. Builds two separate Galaxy instances from the same input
// deck and rng seed (so both draw an identical population), one where
// spec() is called before lbol() (forcing the spec-computed path) and
// one where lbol() alone is ever called (forcing the standalone path),
// then compares the two lbol() values. fCluster = 0.0, as in
// testContinuousPopSpecSingleFeh, so clusters()/disruptedClusters()
// are empty in both cases and lbol() comes entirely from whichever of
// the two continuous-population code paths actually ran, isolating
// the comparison to exactly the two paths this test means to check
// against each other.
static auto testContinuousPopLbolStandaloneMatchesSpec() -> int
{
    constexpr double relTolerance = 0.02; // both paths target the same SimControls::intRelTol() independently

    try
    {
        toml::table inputDeck = toml::parse_file(inputFile);
        inputDeck.at_path("clusters").as_table()->insert("f_cluster", 0.0);
        // Overrides inputFile's own stars.min_stoch_mass = 10 up to
        // chabrier.toml's own IMF max mass, so that every non-clustered
        // star is treated as fully continuous, with no individual
        // FieldStar of its own -- see Galaxy::FieldStar's own comment
        // -- exactly preserving this test's own pre-field-star behavior.
        inputDeck.at_path("stars").as_table()->insert_or_assign("min_stoch_mass", 120.0);
        const io::SimControls controls(inputDeck);

        utils::rng().seed(rngSeed);
        core::Galaxy galaxyViaSpec(controls);
        galaxyViaSpec.advance(t1);
        static_cast<void>(galaxyViaSpec.spec()); // forces computeSpec(), which sets lbolCts_ as a byproduct
        const double lbolViaSpec = galaxyViaSpec.lbol();

        utils::rng().seed(rngSeed);
        core::Galaxy galaxyStandalone(controls);
        galaxyStandalone.advance(t1);
        const double lbolStandalone = galaxyStandalone.lbol(); // spec() never called: forces the standalone computeLbolCts() path

        if (!std::isfinite(lbolViaSpec) || !(lbolViaSpec > 0.0))
        {
            std::cerr << "testGalaxy: continuousPopLbolStandaloneMatchesSpec: "
                "expected a finite, positive lbol() via the spec-computed path, "
                "got " << lbolViaSpec << "\n";
            return 1;
        }
        if (!std::isfinite(lbolStandalone) || !(lbolStandalone > 0.0))
        {
            std::cerr << "testGalaxy: continuousPopLbolStandaloneMatchesSpec: "
                "expected a finite, positive lbol() via the standalone path, "
                "got " << lbolStandalone << "\n";
            return 1;
        }
        if (std::abs(lbolViaSpec - lbolStandalone) > relTolerance * lbolViaSpec)
        {
            std::cerr << "testGalaxy: continuousPopLbolStandaloneMatchesSpec: "
                "lbol() via the spec-computed path (" << lbolViaSpec <<
                " Lsun) deviates from the standalone path (" << lbolStandalone <<
                " Lsun) by more than " << relTolerance * 100 << "%\n";
            return 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testGalaxy: continuousPopLbolStandaloneMatchesSpec test "
            "failed: " << error.what() << "\n";
        return 1;
    }
    return 0;
}

// Verify that Galaxy::advance() correctly incorporates
// SimControls::fCluster() into its own mass accounting: with
// fCluster < 1, only fCluster of the target stellar mass should be
// drawn into stochastic clusters (clusters.CMF), with the remaining
// (1 - fCluster) folded directly into actualMass() as the continuous,
// non-clustered population's own share -- see Galaxy::advance()'s own
// comment for the exact formula this checks.
static auto testFCluster() -> int
{
    constexpr double fClusterValue = 0.5;
    constexpr double tightTol = 1e-9;

    try
    {
        toml::table inputDeck = toml::parse_file(inputFile);
        inputDeck.at_path("clusters").as_table()->insert("f_cluster", fClusterValue);
        // See testContinuousPopSpecSingleFeh's own identical override
        // for why: keeps this test's own expectedActualMass formula
        // below exactly (1 - fCluster) * targetMass() + clusterMass,
        // with no separate field-star mass term to account for.
        inputDeck.at_path("stars").as_table()->insert_or_assign("min_stoch_mass", 120.0);
        const io::SimControls controls(inputDeck);
        if (std::abs(controls.fCluster() - fClusterValue) > tightTol)
        {
            std::cerr << "testGalaxy: fCluster: test bug: expected "
                "SimControls::fCluster() == " << fClusterValue << ", got "
                << controls.fCluster() << "\n";
            return 1;
        }

        utils::rng().seed(rngSeed);
        core::Galaxy galaxy(controls);
        galaxy.advance(t1);

        // targetMass() should be unaffected by fCluster -- always the
        // full SFR-integrated target, clustered and continuous together
        const double expectedTargetMass = controls.sfr().integral(0.0, t1);
        if (std::abs(galaxy.targetMass() - expectedTargetMass) >
            tightTol * expectedTargetMass)
        {
            std::cerr << "testGalaxy: fCluster: targetMass() = " << galaxy.targetMass()
                << ", expected sfr().integral(0, t1) = " << expectedTargetMass << "\n";
            return 1;
        }

        // actualMass() should equal (1 - fCluster) * targetMass() (the
        // continuous population's own share) plus the sum of every
        // newly-formed cluster's own targetMass() -- an exact identity
        // (not just approximate), since this is the very first (and
        // only) advance() call
        double clusterMass = 0.0;
        for (const auto& c : galaxy.clusters()) { clusterMass += c.targetMass(); }
        for (const auto& c : galaxy.disruptedClusters()) { clusterMass += c.targetMass(); }
        const double expectedActualMass =
            ((1.0 - fClusterValue) * galaxy.targetMass()) + clusterMass;
        if (std::abs(galaxy.actualMass() - expectedActualMass) >
            tightTol * expectedActualMass)
        {
            std::cerr << "testGalaxy: fCluster: actualMass() = " << galaxy.actualMass()
                << ", expected (1 - fCluster) * targetMass() + cluster mass = "
                << expectedActualMass << "\n";
            return 1;
        }

        // The stochastic clusters' own total target mass should be
        // close to fCluster * targetMass() -- loose tolerance, since
        // CMF sampling is stochastic (see massTolerance's own comment)
        const double expectedClusterMass = fClusterValue * galaxy.targetMass();
        if (std::abs(clusterMass - expectedClusterMass) > massTolerance * expectedClusterMass)
        {
            std::cerr << "testGalaxy: fCluster: stochastic cluster mass "
                << clusterMass << " deviates from fCluster * targetMass() = "
                << expectedClusterMass << " by more than " << massTolerance * 100 << "%\n";
            return 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testGalaxy: fCluster test failed: " << error.what() << "\n";
        return 1;
    }
    return 0;
}

// Verify that Galaxy::advance()'s field-star mass accounting is exact:
// with fCluster < 1 and inputFile's own (non-degenerate)
// stars.min_stoch_mass left as-is (unlike testFCluster's own
// min_stoch_mass = 120 override), actualMass() should equal
// (1 - fCluster) * (1 - fracStochMass()) * targetMass() (the purely
// continuous share) plus the sum of every newly-formed cluster's own
// targetMass() plus the sum of every newly-drawn field star's own
// mass -- an exact identity, since this is the very first (and only)
// advance() call, so fieldStars() + deadFieldStars() together hold
// every field star drawn this step (deadFieldStars() only ever holds
// *this* step's own deaths -- see Galaxy::advance()'s own comment).
static auto testFieldStarsMassBudget() -> int
{
    constexpr double fClusterValue = 0.5;
    constexpr double tightTol = 1e-9;

    try
    {
        toml::table inputDeck = toml::parse_file(inputFile);
        inputDeck.at_path("clusters").as_table()->insert("f_cluster", fClusterValue);
        // Overrides inputFile's own galaxy.sfr = 1e-2 up to 1.0: at the
        // default rate, the expected number of field stars drawn by t1
        // is only O(1) (comparable to a single star's own typical
        // mass), so whether at least one actually gets drawn is a
        // near-coin-flip -- sensitive enough to platform-dependent
        // floating-point differences in the RNG/distribution chain
        // (e.g. GCC/libstdc++ vs Clang/libc++) that this test was
        // observed to flake on CI (ubuntu-latest/gcc) despite a fixed
        // rng seed. A 100x higher sfr pushes the expected count into
        // the dozens, comfortably far from that boundary.
        inputDeck.at_path("galaxy").as_table()->insert_or_assign("sfr", 1.0);
        const io::SimControls controls(inputDeck);

        utils::rng().seed(rngSeed);
        core::Galaxy galaxy(controls);
        galaxy.advance(t1);

        if (galaxy.fieldStars().empty() && galaxy.deadFieldStars().empty())
        {
            std::cerr << "testGalaxy: fieldStarsMassBudget: test bug: expected "
                "at least one field star to have been drawn\n";
            return 1;
        }
        for (const auto& fs : galaxy.fieldStars())
        {
            if (fs.mass_ < controls.minStochMass())
            {
                std::cerr << "testGalaxy: fieldStarsMassBudget: field star mass "
                    << fs.mass_ << " is below minStochMass() = "
                    << controls.minStochMass() << "\n";
                return 1;
            }
        }

        double clusterMass = 0.0;
        for (const auto& c : galaxy.clusters()) { clusterMass += c.targetMass(); }
        for (const auto& c : galaxy.disruptedClusters()) { clusterMass += c.targetMass(); }

        double fieldStarMass = 0.0;
        for (const auto& fs : galaxy.fieldStars()) { fieldStarMass += fs.mass_; }
        for (const auto& fs : galaxy.deadFieldStars()) { fieldStarMass += fs.mass_; }

        const double expectedActualMass =
            ((1.0 - fClusterValue) * (1.0 - controls.fracStochMass()) * galaxy.targetMass()) +
            clusterMass + fieldStarMass;
        if (std::abs(galaxy.actualMass() - expectedActualMass) >
            tightTol * expectedActualMass)
        {
            std::cerr << "testGalaxy: fieldStarsMassBudget: actualMass() = "
                << galaxy.actualMass() << ", expected (1 - fCluster) * "
                "(1 - fracStochMass()) * targetMass() + cluster mass + field "
                "star mass = " << expectedActualMass << "\n";
            return 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testGalaxy: fieldStarsMassBudget test failed: "
            << error.what() << "\n";
        return 1;
    }
    return 0;
}

// Verify that field stars with masses outside the tracks' own mass grid
// (0.1-300 Msun for MIST_test) get clamped lifetimes rather than
// Tracks3D::starLifetime() being called out of range (which asserts,
// or with assertions disabled returns garbage): with f_cluster = 0 and
// stars.min_stoch_mass at the IMF's own 0.08 Msun minimum, every field
// star below 0.1 Msun should have an infinite death time and never
// die, while every other one has a finite death time; and with a
// delta-function IMF at 400 Msun, every field star should have a death
// time of -infinity, and so be dead after the first advance().
static auto testGalaxyFieldStarLifetimeClamped() -> int
{
    int result = 0;
    try
    {
        toml::table inputDeck = toml::parse_file(inputFile);
        inputDeck.at_path("clusters").as_table()->insert("f_cluster", 0.0);
        inputDeck.at_path("stars").as_table()->insert_or_assign("min_stoch_mass", 0.08);
        const io::SimControls controls(inputDeck);
        const double tracksMin = controls.tracks()->mMin();

        utils::rng().seed(rngSeed);
        core::Galaxy galaxy(controls);
        galaxy.advance(t1);
        galaxy.advance(1e6);

        std::size_t nBelow = 0;
        for (const auto& fs : galaxy.fieldStars())
        {
            if (fs.mass_ < tracksMin)
            {
                ++nBelow;
                if (fs.deathTime_ != std::numeric_limits<double>::infinity())
                {
                    std::cerr << "testGalaxy: fieldStarLifetimeClamped: star of mass " << fs.mass_
                        << " below the tracks' minimum has deathTime_ " << fs.deathTime_ << "\n";
                    result = 1;
                }
            }
            else if (!std::isfinite(fs.deathTime_))
            {
                std::cerr << "testGalaxy: fieldStarLifetimeClamped: star of mass " << fs.mass_
                    << " within the tracks' range has non-finite deathTime_\n";
                result = 1;
            }
        }
        if (nBelow == 0)
        {
            std::cerr << "testGalaxy: fieldStarLifetimeClamped: test bug: expected some field "
                "stars below the tracks' minimum mass\n";
            result = 1;
        }
        for (const auto& fs : galaxy.deadFieldStars())
        {
            if (fs.mass_ < tracksMin)
            {
                std::cerr << "testGalaxy: fieldStarLifetimeClamped: star of mass " << fs.mass_
                    << " below the tracks' minimum has died\n";
                result = 1;
            }
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testGalaxy: fieldStarLifetimeClamped: low-mass case threw: "
            << error.what() << "\n";
        return 1;
    }

    try
    {
        toml::table inputDeck = toml::parse_file(inputFile);
        inputDeck.at_path("clusters").as_table()->insert("f_cluster", 0.0);
        inputDeck.at_path("stars").as_table()->insert_or_assign("IMF", 400.0);
        const io::SimControls controls(inputDeck);
        if (controls.tracks()->mMax() >= 400.0)
        {
            std::cerr << "testGalaxy: fieldStarLifetimeClamped: test bug: expected the tracks' "
                "maximum mass to be below 400 Msun\n";
            return 1;
        }

        utils::rng().seed(rngSeed);
        core::Galaxy galaxy(controls);
        galaxy.advance(t1);
        if (!galaxy.fieldStars().empty() || galaxy.deadFieldStars().empty())
        {
            std::cerr << "testGalaxy: fieldStarLifetimeClamped: expected every 400 Msun field "
                "star to be dead after the first advance(), got " << galaxy.fieldStars().size()
                << " alive and " << galaxy.deadFieldStars().size() << " dead\n";
            result = 1;
        }
        for (const auto& fs : galaxy.deadFieldStars())
        {
            if (fs.deathTime_ != -std::numeric_limits<double>::infinity())
            {
                std::cerr << "testGalaxy: fieldStarLifetimeClamped: 400 Msun star has deathTime_ "
                    << fs.deathTime_ << ", expected -infinity\n";
                result = 1;
                break;
            }
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testGalaxy: fieldStarLifetimeClamped: high-mass case threw: "
            << error.what() << "\n";
        return 1;
    }
    return result;
}

// Verify that field stars are both created and, given enough time,
// destroyed: fieldStars() should be non-empty (every entry alive,
// with mass_ >= minStochMass() and deathTime_ >= curTime()) shortly
// after formation, and deadFieldStars() should be non-empty (every
// entry with deathTime_ < curTime(), and every entry still in
// fieldStars() with deathTime_ >= curTime()) once curTime() has
// advanced far enough past every field star's own main-sequence
// lifetime.
static auto testFieldStarsCreationAndDeath() -> int
{
    // Empirically, at this deck/seed/sfr combination, the shortest-lived
    // field star among those formed by t1 (the most massive one drawn)
    // dies around 3.5e6 yr -- 4e6 gives a safe margin past that while
    // keeping the additional star-forming mass over (t1, tDeath] (at
    // the boosted sfr below) modest, rather than the much larger figure
    // an earlier, far larger tDeath (1e8) produced, which made this
    // step draw so many new stars/clusters that the test hung.
    constexpr double tDeath = 4e6;

    try
    {
        toml::table inputDeck = toml::parse_file(inputFile);
        inputDeck.at_path("clusters").as_table()->insert("f_cluster", 0.5);
        // See testFieldStarsMassBudget's own identical override for why.
        inputDeck.at_path("galaxy").as_table()->insert_or_assign("sfr", 1.0);
        const io::SimControls controls(inputDeck);

        utils::rng().seed(rngSeed);
        core::Galaxy galaxy(controls);
        galaxy.advance(t1);

        if (galaxy.fieldStars().empty())
        {
            std::cerr << "testGalaxy: fieldStarsCreationAndDeath: expected at "
                "least one field star to have formed by t1\n";
            return 1;
        }
        for (const auto& fs : galaxy.fieldStars())
        {
            if (fs.mass_ < controls.minStochMass())
            {
                std::cerr << "testGalaxy: fieldStarsCreationAndDeath: field "
                    "star mass " << fs.mass_ << " is below minStochMass()\n";
                return 1;
            }
            if (fs.formTime_ < 0.0 || fs.formTime_ > t1)
            {
                std::cerr << "testGalaxy: fieldStarsCreationAndDeath: field "
                    "star formTime_ " << fs.formTime_ << " outside [0, t1]\n";
                return 1;
            }
            if (fs.deathTime_ < galaxy.curTime())
            {
                std::cerr << "testGalaxy: fieldStarsCreationAndDeath: a star "
                    "in fieldStars() has deathTime_ " << fs.deathTime_ <<
                    " < curTime() " << galaxy.curTime() << "\n";
                return 1;
            }
        }

        galaxy.advance(tDeath);
        if (galaxy.deadFieldStars().empty())
        {
            std::cerr << "testGalaxy: fieldStarsCreationAndDeath: expected at "
                "least one field star to have died by tDeath = " << tDeath << " yr\n";
            return 1;
        }
        for (const auto& fs : galaxy.deadFieldStars())
        {
            if (fs.deathTime_ >= galaxy.curTime())
            {
                std::cerr << "testGalaxy: fieldStarsCreationAndDeath: a star "
                    "in deadFieldStars() has deathTime_ " << fs.deathTime_ <<
                    " >= curTime() " << galaxy.curTime() << "\n";
                return 1;
            }
        }
        for (const auto& fs : galaxy.fieldStars())
        {
            if (fs.deathTime_ < galaxy.curTime())
            {
                std::cerr << "testGalaxy: fieldStarsCreationAndDeath: a star "
                    "in fieldStars() has deathTime_ " << fs.deathTime_ <<
                    " < curTime() " << galaxy.curTime() << "\n";
                return 1;
            }
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testGalaxy: fieldStarsCreationAndDeath test failed: "
            << error.what() << "\n";
        return 1;
    }
    return 0;
}

// Verify that Galaxy::computeSpec() correctly adds every currently-
// alive field star's own spectrum: independently recomputes the
// expected spectrum as the sum of (1) Specsyn::specCts()'s own
// continuous-population contribution (over [imf().getMin(),
// minStochMass()], exactly as Galaxy::computeSpec() itself calls it)
// and (2) each field star's own spectrum, evaluated via the same
// tracks2D().getStar() + Specsyn::spec() path Galaxy::
// getFieldStarProps()/computeSpec() use internally -- then checks
// that galaxy.spec() matches bit for bit, mirroring checkSpecSum()'s
// own exact-recomputation approach for clusters. fCluster = 0
// isolates this from any cluster contribution, and inputFile's own
// stars.FeH = 0.0 (constFeH()) makes tracks2D() valid to call
// directly.
static auto testFieldStarsSpec() -> int
{
    try
    {
        toml::table inputDeck = toml::parse_file(inputFile);
        inputDeck.at_path("clusters").as_table()->insert("f_cluster", 0.0);
        // See testFieldStarsMassBudget's own identical override for why.
        inputDeck.at_path("galaxy").as_table()->insert_or_assign("sfr", 1.0);
        const io::SimControls controls(inputDeck);

        utils::rng().seed(rngSeed);
        core::Galaxy galaxy(controls);
        galaxy.advance(t1);

        if (!galaxy.clusters().empty() || !galaxy.disruptedClusters().empty())
        {
            std::cerr << "testGalaxy: fieldStarsSpec: test bug: expected no "
                "stochastic clusters at all with f_cluster = 0\n";
            return 1;
        }
        if (galaxy.fieldStars().empty())
        {
            std::cerr << "testGalaxy: fieldStarsSpec: expected at least one "
                "field star to have formed\n";
            return 1;
        }

        // Mirrors Galaxy::computeSpec()'s own branch on computeLbol()
        // exactly (see its own comment): specAndLbolCts() and specCts()
        // are not just related by dropping an extra return value --
        // computeLbol changes specCtsHelper()'s own nInt (the number
        // of quantities the adaptive age integrator tracks error
        // against), which can change its own subdivision decisions
        // and hence the converged result at the bit level, so this
        // must call whichever overload Galaxy::computeSpec() itself
        // would for an exact match below.
        const auto synth = controls.specsyn();
        std::vector<double> expectedSpec;
        if (controls.computeLbol())
        {
            expectedSpec = synth->specAndLbolCts(controls.sfr(), controls.imf(),
                controls.fehDist(), galaxy.curTime(), controls.fCluster(),
                controls.imf().getMin(), controls.minStochMass()).first;
        }
        else
        {
            expectedSpec = synth->specCts(controls.sfr(), controls.imf(),
                controls.fehDist(), galaxy.curTime(), controls.fCluster(),
                controls.imf().getMin(), controls.minStochMass());
        }

        const auto tracks2D = controls.tracks2D();
        for (const auto& fs : galaxy.fieldStars())
        {
            const double logT = std::max(
                std::log10(galaxy.curTime() - fs.formTime_), tracks2D->logTMin());
            const auto props = tracks2D->getStar(fs.mass_, logT);
            const auto starSpec = synth->spec(props, fs.feh_);
            for (std::size_t i = 0; i < expectedSpec.size(); ++i)
            { expectedSpec.at(i) += starSpec.at(i); }
        }

        if (galaxy.spec() != expectedSpec)
        {
            std::cerr << "testGalaxy: fieldStarsSpec: spec() does not equal "
                "the independently-recomputed continuous + field-star "
                "spectrum\n";
            return 1;
        }
        if (std::reduce(expectedSpec.begin(), expectedSpec.end(), 0.0) <= 0.0)
        {
            std::cerr << "testGalaxy: fieldStarsSpec: expected a non-zero "
                "spectrum\n";
            return 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testGalaxy: fieldStarsSpec test failed: "
            << error.what() << "\n";
        return 1;
    }
    return 0;
}

// Verify that Galaxy::addContinuousSpec() correctly extinguishes the
// combined field-star + continuous-population contribution to
// specExtinct(): both together, by the expectation value of
// exp(-A_V * extinct()) over avDistField() (via Extinct::
// applyExtinctionCts()) -- field stars are folded directly into the
// same contSpec the continuous population's own share occupies before
// either is extinguished, so a field star's own individually-drawn
// aV_ is not read for this purpose at all (see FieldStar::aV_'s own
// comment) -- rather than the pre-field-star-extinction behavior of
// passing both through unattenuated (still exercised, unchanged, by
// testFieldStarsSpec's own use of inputFile's default
// extinct.AV_field, which is absent and so defaults to a delta at 0 --
// see io::SimControls::avDistField()'s own comment). fCluster = 0
// isolates this from any cluster contribution (which uses its own,
// unrelated avDist()); overrides galaxy.sfr for the same reason
// testFieldStarsMassBudget's own identical override does, and sets
// extinct.AV_field to a fixed value distinct from inputFile's own
// clusters.AV, so field stars draw a genuinely different, independent
// A_V than clusters would -- exercised here only to confirm it plays
// no role in specExtinct(), via the exact-aV_ check below.
static auto testFieldStarsExtinct() -> int
{
    constexpr double avFieldValue = 0.5; // NOLINT(readability-identifier-naming) -- see Extinct::applyExtinction()'s own identical NOLINT

    try
    {
        toml::table inputDeck = toml::parse_file(inputFile);
        inputDeck.at_path("clusters").as_table()->insert("f_cluster", 0.0);
        // See testFieldStarsMassBudget's own identical override for why.
        inputDeck.at_path("galaxy").as_table()->insert_or_assign("sfr", 1.0);
        inputDeck.at_path("extinct").as_table()->insert_or_assign("AV_field", avFieldValue);
        const io::SimControls controls(inputDeck);

        utils::rng().seed(rngSeed);
        core::Galaxy galaxy(controls);
        galaxy.advance(t1);

        if (!galaxy.clusters().empty() || !galaxy.disruptedClusters().empty())
        {
            std::cerr << "testGalaxy: fieldStarsExtinct: test bug: expected no "
                "stochastic clusters at all with f_cluster = 0\n";
            return 1;
        }
        if (galaxy.fieldStars().empty())
        {
            std::cerr << "testGalaxy: fieldStarsExtinct: expected at least one "
                "field star to have formed\n";
            return 1;
        }

        const auto ext = controls.extinct();
        if (ext == nullptr)
        {
            std::cerr << "testGalaxy: fieldStarsExtinct: test bug: expected "
                "extinct() non-null\n";
            return 1;
        }

        // Every field star should have drawn aV_ == avFieldValue exactly,
        // since avDistField() is a delta function at that value
        for (const auto& fs : galaxy.fieldStars())
        {
            if (!utils::approxEqual(fs.aV_, avFieldValue))
            {
                std::cerr << "testGalaxy: fieldStarsExtinct: field star aV_ = "
                    << fs.aV_ << ", expected " << avFieldValue << "\n";
                return 1;
            }
        }

        // Independently recompute the expected specExtinct(): builds
        // contSpec exactly as Galaxy::addContinuousSpec() now does --
        // the continuous population's own contribution, then every
        // field star's own spectrum added directly into it, in that
        // same order -- then extinguishes the combined result via
        // applyExtinctionCts() once, matching the real implementation
        // bit-for-bit (rather than extinguishing each piece separately
        // and summing the results, which would not generally agree at
        // the bit level, since floating-point addition is not
        // associative).
        const auto synth = controls.specsyn();

        std::vector<double> contSpec;
        if (controls.computeLbol())
        {
            contSpec = synth->specAndLbolCts(controls.sfr(), controls.imf(),
                controls.fehDist(), galaxy.curTime(), controls.fCluster(),
                controls.imf().getMin(), controls.minStochMass()).first;
        }
        else
        {
            contSpec = synth->specCts(controls.sfr(), controls.imf(),
                controls.fehDist(), galaxy.curTime(), controls.fCluster(),
                controls.imf().getMin(), controls.minStochMass());
        }

        const auto tracks2D = controls.tracks2D();
        for (const auto& fs : galaxy.fieldStars())
        {
            const double logT = std::max(
                std::log10(galaxy.curTime() - fs.formTime_), tracks2D->logTMin());
            const auto props = tracks2D->getStar(fs.mass_, logT);
            const auto starSpec = synth->spec(props, fs.feh_);
            for (std::size_t i = 0; i < contSpec.size(); ++i)
            { contSpec.at(i) += starSpec.at(i); }
        }

        const auto expectedSpecExtinct = ext->applyExtinctionCts(contSpec);

        if (galaxy.specExtinct() != expectedSpecExtinct)
        {
            std::cerr << "testGalaxy: fieldStarsExtinct: specExtinct() does "
                "not equal the independently-recomputed extinguished "
                "continuous + field-star spectrum\n";
            return 1;
        }
        if (std::reduce(expectedSpecExtinct.begin(), expectedSpecExtinct.end(), 0.0) <= 0.0)
        {
            std::cerr << "testGalaxy: fieldStarsExtinct: expected a non-zero "
                "specExtinct()\n";
            return 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testGalaxy: fieldStarsExtinct test failed: "
            << error.what() << "\n";
        return 1;
    }
    return 0;
}

// Verify that Extinct::applyExtinctionCts() exactly reduces to
// applyExtinction() at a fixed A_V when avDistField() is a delta
// function (no variation to average over) -- extinct.AV_field given as
// a plain number in the input deck creates exactly this. Lives here,
// alongside the rest of this file's own field-star/extinction tests,
// rather than in tests/extinct/testExtinct.hpp: unlike that file's own
// tests, this one needs a full, TOML-deck-driven SimControls (to reach
// a real, non-default avDistField() at all), which tests/extinct's own
// slugTestExtinct target does not link -- mirroring testCluster.cpp's
// own testClusterExtinct() for the identical reason.
static auto testExtinctApplyExtinctionCtsDegenerate() -> int
{
    constexpr double avFieldValue = 1.5; // NOLINT(readability-identifier-naming) -- see Extinct::applyExtinction()'s own identical NOLINT

    try
    {
        toml::table inputDeck = toml::parse_file(inputFile);
        inputDeck.at_path("extinct").as_table()->insert_or_assign("AV_field", avFieldValue);
        const io::SimControls controls(inputDeck);
        const auto ext = controls.extinct();
        if (ext == nullptr)
        {
            std::cerr << "testExtinctApplyExtinctionCtsDegenerate: test bug: "
                "expected extinct() non-null\n";
            return 1;
        }

        const std::vector<double> spec(controls.specsyn()->wl().size(), 1.0);
        const auto expected = ext->applyExtinction(avFieldValue, spec);
        const auto actual = ext->applyExtinctionCts(spec);
        for (std::size_t i = 0; i < expected.size(); i++)
        {
            if (!utils::approxEqual(actual.at(i), expected.at(i)))
            {
                std::cerr << "testExtinctApplyExtinctionCtsDegenerate: "
                    "applyExtinctionCts()[" << i << "] = " << actual.at(i)
                    << ", expected applyExtinction(" << avFieldValue << ", ...)["
                    << i << "] = " << expected.at(i) << "\n";
                return 1;
            }
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testExtinctApplyExtinctionCtsDegenerate test failed: "
            << error.what() << "\n";
        return 1;
    }
    return 0;
}

// Verify that Extinct::applyExtinctionCts() matches the closed-form
// integral a uniform avDistField() admits: int_0^X exp(-A_V * k) / X
// dA_V = (1 - exp(-k * X)) / (k * X), for k = extinct() at each
// wavelength, computed independently here rather than via
// utils::PDFIntegrator (the k = 0 limit of that expression is 1,
// matching exp(0) = 1 -- no wavelength should ever actually hit that
// in practice, since Extinct::normalize() only guarantees a
// V-band-weighted average of 1, not a pointwise floor, but the limit
// is included for robustness). See
// testExtinctApplyExtinctionCtsDegenerate's own comment for why this
// lives here rather than in tests/extinct/testExtinct.hpp.
// tests/extinct/assets/testExtinctAVFieldUniform.toml gives
// avDistField() a uniform distribution over [0, 2].
static auto testExtinctApplyExtinctionCtsUniform() -> int
{
    constexpr double relTol = 1e-6; // GKIntegrator's own default relTol, at GK15
    constexpr double avFieldMax = 2.0; // NOLINT(readability-identifier-naming) -- see Extinct::applyExtinction()'s own identical NOLINT

    try
    {
        toml::table inputDeck = toml::parse_file(inputFile);
        inputDeck.at_path("extinct").as_table()->insert_or_assign(
            "AV_field", "tests/extinct/assets/testExtinctAVFieldUniform.toml");
        const io::SimControls controls(inputDeck);
        const auto ext = controls.extinct();
        if (ext == nullptr)
        {
            std::cerr << "testExtinctApplyExtinctionCtsUniform: test bug: "
                "expected extinct() non-null\n";
            return 1;
        }
        if (!utils::approxEqual(controls.avDistField().getMin(), 0.0) ||
            !utils::approxEqual(controls.avDistField().getMax(), avFieldMax))
        {
            std::cerr << "testExtinctApplyExtinctionCtsUniform: test bug: "
                "expected avDistField() to span [0, " << avFieldMax << "]\n";
            return 1;
        }

        const std::vector<double> spec(controls.specsyn()->wl().size(), 1.0);
        const auto result = ext->applyExtinctionCts(spec);
        for (std::size_t i = 0; i < ext->wl().size(); i++)
        {
            const double k = ext->extinct().at(i);
            const double expected = (k == 0.0) ? 1.0 :
                (1.0 - std::exp(-k * avFieldMax)) / (k * avFieldMax);
            if (std::abs(result.at(i) - expected) > relTol * std::abs(expected))
            {
                std::cerr << "testExtinctApplyExtinctionCtsUniform: at wl "
                    "index " << i << ", applyExtinctionCts() gives "
                    << result.at(i) << ", expected closed-form " << expected << "\n";
                return 1;
            }
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testExtinctApplyExtinctionCtsUniform test failed: "
            << error.what() << "\n";
        return 1;
    }
    return 0;
}

// Verify that Galaxy's own specNeb()/specNebExtinct()/lineLum()/
// lineLumExtinct() equal the independently-summed per-cluster
// specNeb()/specNebExtinct()/lineLum()/lineLumExtinct() -- mirrors
// checkSpecSum()'s own exact style, but for
// nebular emission rather than plain spec()/specExtinct() -- and that
// photNeb()/photNebExtinct() equal filters->phot() applied directly
// to specNeb()/specNebExtinct(), exactly as Galaxy::computePhot()
// itself computes them (unlike checkPhotSum()'s own phot()/
// photExtinct() check, which cross-checks two genuinely different
// computation paths -- summed-then-photometered vs photometered-then-
// summed -- only approximately, this is the same single path
// Galaxy::computePhot() takes, so an exact match is expected).
// inputFile (testGalaxyDynamics.in) defaults fCluster() to 1.0 (not
// overridden here), so every star forms in a stochastic cluster and
// Galaxy::addContinuousSpec() never runs, letting this test check
// pure cluster summation without also needing to reproduce that
// function's own continuous-population/field-star contribution.
// Reuses testGalaxyDynamics.in, overriding its own [nebular] table to
// point at tests/nebular/assets/nebular_test.h5 (see that fixture's
// own generator, data/tools/cloudy/make_nebular_test_fixture.py, for
// its schema), exactly as testFieldStarsExtinct() overrides [extinct]
// on the same base deck.
// Independently sum specNeb()/lineLum()/specNebExtinct()/
// lineLumExtinct() over every cluster in galaxy's own clusters()/
// disruptedClusters() and compare against galaxy's own summed values
// -- factored out of testGalaxyNebular() to keep it within its
// cognitive-complexity budget.
static auto checkGalaxyNebularSums(const io::SimControls& controls, core::Galaxy& galaxy) -> int
{
    std::vector<double> expectedSpecNeb(controls.specsyn()->wl().size(), 0.0);
    std::vector<double> expectedLineLum(controls.nebular()->lineWl().size(), 0.0);
    std::vector<double> expectedSpecNebExtinct(controls.extinct()->wl().size(), 0.0);
    std::vector<double> expectedLineLumExtinct(controls.nebular()->lineWl().size(), 0.0);
    const auto accumulate = [&](std::vector<core::Cluster>& list)
    {
        for (auto& c : list)
        {
            const auto& s = c.specNeb();
            for (std::size_t i = 0; i < expectedSpecNeb.size(); ++i) { expectedSpecNeb.at(i) += s.at(i); }
            const auto& l = c.lineLum();
            for (std::size_t i = 0; i < expectedLineLum.size(); ++i) { expectedLineLum.at(i) += l.at(i); }
            const auto& se = c.specNebExtinct();
            for (std::size_t i = 0; i < expectedSpecNebExtinct.size(); ++i) { expectedSpecNebExtinct.at(i) += se.at(i); }
            const auto& le = c.lineLumExtinct();
            for (std::size_t i = 0; i < expectedLineLumExtinct.size(); ++i) { expectedLineLumExtinct.at(i) += le.at(i); }
        }
    };
    accumulate(galaxy.clusters());
    accumulate(galaxy.disruptedClusters());

    if (galaxy.specNeb() != expectedSpecNeb)
    {
        std::cerr << "testGalaxy: nebular: specNeb() does not equal the "
            "independently-summed per-cluster specNeb()\n";
        return 1;
    }
    if (galaxy.lineLum() != expectedLineLum)
    {
        std::cerr << "testGalaxy: nebular: lineLum() does not equal the "
            "independently-summed per-cluster lineLum()\n";
        return 1;
    }
    if (galaxy.specNebExtinct() != expectedSpecNebExtinct)
    {
        std::cerr << "testGalaxy: nebular: specNebExtinct() does not equal "
            "the independently-summed per-cluster specNebExtinct()\n";
        return 1;
    }
    if (galaxy.lineLumExtinct() != expectedLineLumExtinct)
    {
        std::cerr << "testGalaxy: nebular: lineLumExtinct() does not equal "
            "the independently-summed per-cluster lineLumExtinct()\n";
        return 1;
    }
    if (std::reduce(expectedSpecNeb.begin(), expectedSpecNeb.end(), 0.0) <= 0.0)
    {
        std::cerr << "testGalaxy: nebular: expected a non-zero specNeb()\n";
        return 1;
    }
    return 0;
}

static auto testGalaxyNebular() -> int
{
    try
    {
        toml::table inputDeck = toml::parse_file(inputFile);
        inputDeck.at_path("nebular").as_table()->insert_or_assign("compute_neb", true);
        inputDeck.at_path("nebular").as_table()->insert_or_assign(
            "table", std::string("tests/nebular/assets/nebular_test.h5"));
        const io::SimControls controls(inputDeck);

        if (controls.nebular() == nullptr)
        {
            std::cerr << "testGalaxy: nebular: expected SimControls::nebular() "
                "to be non-null\n";
            return 1;
        }
        if (!utils::approxEqual(controls.fCluster(), 1.0))
        {
            std::cerr << "testGalaxy: nebular: test bug: expected fCluster() == 1\n";
            return 1;
        }

        utils::rng().seed(rngSeed);
        core::Galaxy galaxy(controls);
        galaxy.advance(t1);

        if (galaxy.clusters().empty() && galaxy.disruptedClusters().empty())
        {
            std::cerr << "testGalaxy: nebular: test bug: expected at least one "
                "cluster to have formed\n";
            return 1;
        }

        if (checkGalaxyNebularSums(controls, galaxy) != 0) { return 1; }

        const auto expectedPhotNeb = controls.filters()->phot(controls.specsyn()->wlObs(), galaxy.specNeb());
        if (galaxy.photNeb() != expectedPhotNeb)
        {
            std::cerr << "testGalaxy: nebular: photNeb() does not match "
                "filters->phot(wlObs(), specNeb())\n";
            return 1;
        }
        const auto expectedPhotNebExtinct =
            controls.filters()->phot(controls.extinct()->wlObs(), galaxy.specNebExtinct());
        if (galaxy.photNebExtinct() != expectedPhotNebExtinct)
        {
            std::cerr << "testGalaxy: nebular: photNebExtinct() does not match "
                "filters->phot(ext->wlObs(), specNebExtinct())\n";
            return 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testGalaxy: nebular test failed: " << error.what() << "\n";
        return 1;
    }
    return 0;
}

// Verify that Galaxy::addContinuousSpec() correctly extinguishes the
// continuous population's own nebular-reprocessed line luminosities
// into lineLumExtinct() -- the code path testGalaxyNebular() above
// deliberately does not exercise, since it keeps fCluster() == 1
// (no continuous population at all) to isolate pure cluster summation
// instead. Mirrors testContinuousPopSpecSingleFeh's own f_cluster = 0,
// min_stoch_mass = 120.0 combination (entirely continuous, no
// stochastic clusters and no individual field stars either) so
// contSpec is guaranteed to come *only* from Specsyn::specCts()'s/
// specAndLbolCts()'s own continuous-population overload, with no
// cluster- or field-star-summed contribution to confound it --
// clusters()/disruptedClusters()/fieldStars() are all checked empty as
// a sanity check on that premise. Independently rebuilds contSpec the
// same way, passes it through Nebular::getGalaxy() at
// SimControls::fehDist()'s own expectationValue() (exactly as
// addContinuousSpec() itself does) to get the expected nebular line
// luminosities, then extinguishes those via Extinct::
// applyExtinctionCtsLines() -- since there are no clusters at all here,
// this alone should exactly equal galaxy.lineLumExtinct(), with no
// cluster-summed term added on top. Reuses testGalaxyDynamics.in,
// overriding its own [nebular] table to point at tests/nebular/assets/
// nebular_test.h5, exactly as testGalaxyNebular() above does.
static auto testContinuousPopNebularExtinct() -> int
{
    try
    {
        toml::table inputDeck = toml::parse_file(inputFile);
        inputDeck.at_path("clusters").as_table()->insert("f_cluster", 0.0);
        inputDeck.at_path("stars").as_table()->insert_or_assign("min_stoch_mass", 120.0);
        inputDeck.at_path("nebular").as_table()->insert_or_assign("compute_neb", true);
        inputDeck.at_path("nebular").as_table()->insert_or_assign(
            "table", std::string("tests/nebular/assets/nebular_test.h5"));
        const io::SimControls controls(inputDeck);

        if (controls.nebular() == nullptr)
        {
            std::cerr << "testGalaxy: continuousPopNebularExtinct: expected "
                "SimControls::nebular() to be non-null\n";
            return 1;
        }
        const auto ext = controls.extinct();
        if (ext == nullptr)
        {
            std::cerr << "testGalaxy: continuousPopNebularExtinct: test bug: "
                "expected extinct() non-null\n";
            return 1;
        }

        utils::rng().seed(rngSeed);
        core::Galaxy galaxy(controls);
        galaxy.advance(t1);

        if (!galaxy.clusters().empty() || !galaxy.disruptedClusters().empty() ||
            !galaxy.fieldStars().empty())
        {
            std::cerr << "testGalaxy: continuousPopNebularExtinct: test bug: "
                "expected no stochastic clusters or field stars at all with "
                "f_cluster = 0 and min_stoch_mass = 120\n";
            return 1;
        }

        // Independently rebuild contSpec exactly as
        // Galaxy::addContinuousSpec() itself does -- see
        // testFieldStarsExtinct's own identical construction
        const auto synth = controls.specsyn();
        std::vector<double> contSpec;
        if (controls.computeLbol())
        {
            contSpec = synth->specAndLbolCts(controls.sfr(), controls.imf(),
                controls.fehDist(), galaxy.curTime(), controls.fCluster(),
                controls.imf().getMin(), controls.minStochMass()).first;
        }
        else
        {
            contSpec = synth->specCts(controls.sfr(), controls.imf(),
                controls.fehDist(), galaxy.curTime(), controls.fCluster(),
                controls.imf().getMin(), controls.minStochMass());
        }

        const auto [nebContSpec, nebContLineLum] =
            controls.nebular()->getGalaxy(contSpec, controls.fehDist().expectationValue());
        const auto expectedLineLumExtinct = ext->applyExtinctionCtsLines(nebContLineLum);

        if (galaxy.lineLumExtinct() != expectedLineLumExtinct)
        {
            std::cerr << "testGalaxy: continuousPopNebularExtinct: "
                "lineLumExtinct() does not equal the independently-recomputed "
                "extinguished continuous-population line luminosities\n";
            return 1;
        }
        if (std::reduce(expectedLineLumExtinct.begin(), expectedLineLumExtinct.end(), 0.0) <= 0.0)
        {
            std::cerr << "testGalaxy: continuousPopNebularExtinct: expected a "
                "non-zero lineLumExtinct()\n";
            return 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testGalaxy: continuousPopNebularExtinct test failed: "
            << error.what() << "\n";
        return 1;
    }
    return 0;
}

// Verify Galaxy::sfr_'s own resolution logic (see SFRVar's header
// comment): when controls.sfr() is valid (galaxy.sfr was given), it
// should be used directly, as a live reference -- checked here via
// pointer identity, so a mutation through controls would be visible
// through galaxy.sfr() too. When controls.sfr() is invalid instead
// (galaxy.sfr_dist was given), Galaxy::Galaxy() should draw once from
// controls.sfrDist() and build an owned constant-in-time PDF from that
// draw via SimControls::buildConstantSFR() -- exercised here with a
// delta-function sfr_dist so the draw is deterministic, letting this
// compare galaxy.sfr() bit-for-bit against an independently-built
// buildConstantSFR() reference.
static auto testGalaxySFRDistResolution() -> int
{
    // galaxy.sfr given: sfr() should be a live reference to
    // controls.sfr() itself, not a copy
    try
    {
        const toml::table inputDeck = toml::parse_file(inputFile);
        const io::SimControls controls(inputDeck);
        const core::Galaxy galaxy(controls);
        if (&galaxy.sfr() != &controls.sfr())
        {
            std::cerr << "testGalaxy: sfrDistResolution: expected sfr() to "
                "be a live reference to controls.sfr() when galaxy.sfr "
                "was given\n";
            return 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testGalaxy: sfrDistResolution: sfr-given case "
            "failed: " << error.what() << "\n";
        return 1;
    }

    // galaxy.sfr_dist given instead: sfr() should be an owned PDF, built
    // by drawing once from sfrDist() and passing that draw through
    // SimControls::buildConstantSFR() -- matched here against an
    // independently-built reference, since sfrDist() is a delta
    // function and so draws deterministically
    constexpr double sfrDistValue = 5e-3;
    try
    {
        toml::table inputDeck = toml::parse_file(inputFile);
        auto* galaxyTbl = inputDeck.at_path("galaxy").as_table();
        galaxyTbl->erase("sfr");
        galaxyTbl->insert("sfr_dist", sfrDistValue);
        const io::SimControls controls(inputDeck);

        if (controls.sfr().valid())
        {
            std::cerr << "testGalaxy: sfrDistResolution: test bug: "
                "expected controls.sfr() invalid when galaxy.sfr_dist "
                "is given\n";
            return 1;
        }

        const core::Galaxy galaxy(controls);
        const auto expected = io::SimControls::buildConstantSFR(sfrDistValue);
        if (!galaxy.sfr().valid() ||
            galaxy.sfr().getMin() != expected.getMin() ||
            galaxy.sfr().getMax() != expected.getMax() ||
            !utils::approxEqual(
                galaxy.sfr().integral(0.0, t1), expected.integral(0.0, t1)))
        {
            std::cerr << "testGalaxy: sfrDistResolution: expected sfr() to "
                "match buildConstantSFR(sfrDist().draw()) when "
                "galaxy.sfr_dist was given\n";
            return 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testGalaxy: sfrDistResolution: sfr_dist-given case "
            "failed: " << error.what() << "\n";
        return 1;
    }

    return 0;
}

// Verify Galaxy::yieldsRate()'s order of magnitude: for a population
// forming with a constant SFR, the instantaneous hydrogen return rate
// at age 100 Myr should be of order 10% of the SFR itself, following
// the standard rule of thumb that roughly 10% of the mass that forms
// in a stellar population is eventually returned to the ISM within
// about 100 Myr (dominated by the most massive stars' winds and
// supernovae). Forces min_stoch_mass to chabrier.toml's own maximum
// mass and f_cluster to 0 (mirroring
// testContinuousPopLbolStandaloneMatchesSpec()'s own identical
// technique) so the entire population -- including every mass whose
// lifetime is under 100 Myr -- is continuously sampled, letting
// yieldsRate() (which only covers that population) account for all of
// it. Loads both ccsn models (kobayashi_test, covering [13, 18] Msun,
// and sukhbold_test, covering [18.2, 100]) plus
// massive_star_winds/sukhbold_test, between them covering the whole
// relevant mass range down to the ~15 Msun turnoff at 100 Myr -- a
// single, narrower channel underestimates the ratio by roughly an
// order of magnitude, since most of the return over the full [0, 100
// Myr] integration window then comes from masses no yield channel
// tabulates. yields.channel_decomposed = false collapses the two ccsn
// entries (a deliberately duplicated channel type, hence the expected
// "slug: warning" on construction -- see
// testSimControlsYieldsDuplicateChannelWarning()'s own identical
// pattern) and the wind channel into one combined per-isotope total,
// so hydrogen's own index can be read directly. Although hydrogen
// itself is stable (and so unaffected by decay either way), the
// galaxy is still advance()d to age first: Yields::applyDecay()
// processes every isotope internally, including unstable ones (Ni56,
// also in this fixture), so leaving curTime_ at its initial 0 while
// evaluating yieldsRate(age, ...) -- t > curTime_ -- would still throw
// from DecayChain::applyDecay()'s own negative-dt rejection.
static auto testYieldsRateHydrogenOrderOfMagnitude() -> int
{
    constexpr double age = 1e8; // 100 Myr
    constexpr double ratioMin = 0.01; // one order of magnitude below 10%
    constexpr double ratioMax = 1.0; // one order of magnitude above 10%

    try
    {
        toml::table inputDeck = toml::parse_file(inputFile);
        inputDeck.at_path("clusters").as_table()->insert("f_cluster", 0.0);
        inputDeck.at_path("stars").as_table()->insert_or_assign("min_stoch_mass", 120.0);
        inputDeck.insert("yields", toml::table{
            { "channel1", toml::table{ { "channel", "ccsn" }, { "model", "kobayashi_test" } } },
            { "channel2", toml::table{ { "channel", "ccsn" }, { "model", "sukhbold_test" } } },
            { "channel3", toml::table{ { "channel", "massive_star_winds" }, { "model", "sukhbold_test" } } },
            { "channel_decomposed", false },
            { "registry", std::string("tests/yields/assets/yields.toml") },
        });
        const io::SimControls controls(inputDeck);

        if (controls.yields() == nullptr)
        {
            std::cerr << "testGalaxy: yieldsRateHydrogenOrderOfMagnitude: "
                "expected yields() non-null\n";
            return 1;
        }

        // Find hydrogen's own index (Z = 1, A = 1) in isotopes()
        const auto& isotopes = controls.yields()->isotopes();
        const auto h1It = std::ranges::find_if(isotopes,
            [](const auto& iso) { return iso.get().Z() == 1 && iso.get().A() == 1; });
        if (h1It == isotopes.end())
        {
            std::cerr << "testGalaxy: yieldsRateHydrogenOrderOfMagnitude: "
                "expected hydrogen (Z=1, A=1) among isotopes()\n";
            return 1;
        }
        const auto h1Idx = static_cast<std::size_t>(std::distance(isotopes.begin(), h1It));

        utils::rng().seed(rngSeed);
        core::Galaxy galaxy(controls);
        galaxy.advance(age);

        const auto rate = galaxy.yieldsRate(age, 0.0);
        const double sfrVal = galaxy.sfr()(age);
        if (!(sfrVal > 0.0))
        {
            std::cerr << "testGalaxy: yieldsRateHydrogenOrderOfMagnitude: "
                "expected a positive sfr(), got " << sfrVal << "\n";
            return 1;
        }
        if (h1Idx >= rate.size() || !std::isfinite(rate[h1Idx]) || rate[h1Idx] <= 0.0)
        {
            std::cerr << "testGalaxy: yieldsRateHydrogenOrderOfMagnitude: "
                "expected a finite, positive hydrogen return rate, got " <<
                (h1Idx < rate.size() ? std::to_string(rate[h1Idx]) : "out of range") << "\n";
            return 1;
        }

        const double ratio = rate[h1Idx] / sfrVal;
        if (ratio < ratioMin || ratio > ratioMax)
        {
            std::cerr << "testGalaxy: yieldsRateHydrogenOrderOfMagnitude: "
                "hydrogen return rate / sfr = " << ratio << " at age " << age <<
                " yr, expected order of magnitude 0.1 (i.e. in [" << ratioMin <<
                ", " << ratioMax << "])\n";
            return 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testGalaxy: yieldsRateHydrogenOrderOfMagnitude test failed: "
            << error.what() << "\n";
        return 1;
    }
    return 0;
}

// Verify that Galaxy::yieldsRate(t) (no feh argument) exactly
// delegates to yieldsRate(t, feh) when SimControls::fehDist() is
// degenerate (a single value) -- testGalaxyDynamics.in's own stars.FeH
// = 0.0 is exactly this case. Reuses
// testYieldsRateHydrogenOrderOfMagnitude()'s own yields/min_stoch_mass/
// f_cluster setup, and, like that test (see its own updated comment),
// advance()s the galaxy to age first: yieldsRate(t, feh) itself decays
// its own raw rate forward by curTime_ - t (see its own comment), and
// evaluating it here with t == age > curTime_ == 0 (had this not
// advance()d first) would compute decay over a *negative* elapsed
// time for every unstable isotope in the fixture (Ni56), which
// DecayChain::applyDecay() itself now rejects outright.
static auto testYieldsRateSingleFehDelegates() -> int
{
    constexpr double age = 1e8;

    try
    {
        toml::table inputDeck = toml::parse_file(inputFile);
        inputDeck.at_path("clusters").as_table()->insert("f_cluster", 0.0);
        inputDeck.at_path("stars").as_table()->insert_or_assign("min_stoch_mass", 120.0);
        inputDeck.insert("yields", toml::table{
            { "channel1", toml::table{ { "channel", "ccsn" }, { "model", "kobayashi_test" } } },
            { "channel2", toml::table{ { "channel", "ccsn" }, { "model", "sukhbold_test" } } },
            { "channel3", toml::table{ { "channel", "massive_star_winds" }, { "model", "sukhbold_test" } } },
            { "channel_decomposed", false },
            { "registry", std::string("tests/yields/assets/yields.toml") },
        });
        const io::SimControls controls(inputDeck);

        if (controls.fehDist().getMin() != controls.fehDist().getMax())
        {
            std::cerr << "testGalaxy: yieldsRateSingleFehDelegates: test bug: "
                "expected a degenerate (single-value) fehDist()\n";
            return 1;
        }

        utils::rng().seed(rngSeed);
        core::Galaxy galaxy(controls);
        galaxy.advance(age);

        const auto rateNoFeh = galaxy.yieldsRate(age);
        const auto rateWithFeh = galaxy.yieldsRate(age, controls.fehDist().getMin());
        if (rateNoFeh.size() != rateWithFeh.size())
        {
            std::cerr << "testGalaxy: yieldsRateSingleFehDelegates: size mismatch: "
                << rateNoFeh.size() << " vs " << rateWithFeh.size() << "\n";
            return 1;
        }
        for (std::size_t k = 0; k < rateNoFeh.size(); ++k)
        {
            if (rateNoFeh.at(k) != rateWithFeh.at(k))
            {
                std::cerr << "testGalaxy: yieldsRateSingleFehDelegates: entry " << k <<
                    ": yieldsRate(t) = " << rateNoFeh.at(k) << ", yieldsRate(t, feh) = " <<
                    rateWithFeh.at(k) << " -- expected bit-for-bit equality\n";
                return 1;
            }
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testGalaxy: yieldsRateSingleFehDelegates test failed: "
            << error.what() << "\n";
        return 1;
    }
    return 0;
}

// Verify Galaxy::yieldsRate(t)'s multi-feh averaging against an
// independent recomputation using the same public API the
// implementation itself is built from: evaluates yieldsRate(t, feh) at
// every SimControls::tracks()->feH() grid point directly, weights each
// by SimControls::fehDist()'s own density there via a hand-built
// interp::Interpolator1D<1> pair (one for the weights, one per
// isotope for the weighted values), and compares the resulting
// weighted average against yieldsRate(t)'s own result -- mirrors this
// test suite's general "independent recomputation" style (e.g.
// testContinuousPopSpecMultiFeh() and its own reference-check sibling)
// rather than merely checking the result is finite. Reuses
// testContinuousPopSpecMultiFeh()'s own [-1, 0] fixture, combined with
// sukhbold_test/massive_star_winds (whose own [-1.0, 0.0] tabulated
// range exactly matches it, unlike kobayashi_test's own single-point
// Fe/H = 0.0 -- SimControls itself rejects a yield channel whose own
// range doesn't cover fehDist's, at construction, so kobayashi_test
// can't be used here at all). Tracks3D's own one-point padding beyond
// fehDist's own range still reaches Fe/H = 0.5 (see
// tests/tracks/assets/tracks.toml's own MIST_test grid), outside even
// sukhbold_test/massive_star_winds's own [-1.0, 0.0] -- exercising
// Yields::yield()'s own [Fe/H] range check (mirroring its existing
// hasYield(mass) check) that makes evaluating yieldsRate(t, feh) at
// that padding point return zero from every channel rather than
// hitting YieldChannel::yield()'s own out-of-range assert. Like
// testYieldsRateSingleFehDelegates() (see its own comment), this
// checks every isotope including unstable ones, so the galaxy is
// advance()d to age first, keeping every yieldsRate(t, feh) call's own
// internal decay (curTime_ - t) non-negative.
static auto testYieldsRateMultiFeh() -> int
{
    constexpr double age = 1e8;
    constexpr double relTol = 1e-9; // both sides use the exact same yieldsRate(t, feh) calls

    try
    {
        toml::table inputDeck = toml::parse_file(inputFile);
        inputDeck.at_path("clusters").as_table()->insert("f_cluster", 0.0);
        inputDeck.at_path("stars").as_table()->insert_or_assign("min_stoch_mass", 120.0);
        inputDeck.at_path("stars").as_table()->insert_or_assign(
            "FeH", "tests/core/assets/testClusterSpecsynFullFeHDist.toml");
        inputDeck.insert("yields", toml::table{
            { "channel1", toml::table{ { "channel", "ccsn" }, { "model", "sukhbold_test" } } },
            { "channel2", toml::table{ { "channel", "massive_star_winds" }, { "model", "sukhbold_test" } } },
            { "channel_decomposed", false },
            { "registry", std::string("tests/yields/assets/yields.toml") },
        });
        const io::SimControls controls(inputDeck);

        if (controls.fehDist().getMin() == controls.fehDist().getMax())
        {
            std::cerr << "testGalaxy: yieldsRateMultiFeh: test bug: "
                "expected a non-degenerate fehDist()\n";
            return 1;
        }

        utils::rng().seed(rngSeed);
        core::Galaxy galaxy(controls);
        galaxy.advance(age);

        // Independent recomputation, mirroring Galaxy::yieldsRate(t)'s
        // own multi-feh implementation exactly, but built here from
        // scratch via the public API alone
        const auto& fehDist = controls.fehDist();
        const auto& fehGrid = controls.tracks()->feH();
        const std::size_t nFeh = fehGrid.size();

        std::vector<std::vector<double>> rateAtFeh(nFeh);
        std::vector<double> fehWeight(nFeh);
        for (std::size_t f = 0; f < nFeh; ++f)
        {
            rateAtFeh.at(f) = galaxy.yieldsRate(age, fehGrid.at(f));
            fehWeight.at(f) = fehDist(fehGrid.at(f));
        }

        const interp::Interpolator1D<1> weightInterp(fehGrid, fehWeight);
        const double weightIntegral = weightInterp.integ(fehDist.getMin(), fehDist.getMax());

        const std::size_t n = rateAtFeh.front().size();
        std::vector<double> expected(n, 0.0);
        std::vector<double> quantityAtFeh(nFeh);
        for (std::size_t k = 0; k < n; ++k)
        {
            for (std::size_t f = 0; f < nFeh; ++f)
            {
                quantityAtFeh.at(f) = rateAtFeh.at(f).at(k) * fehWeight.at(f);
            }
            const interp::Interpolator1D<1> quantityInterp(fehGrid, quantityAtFeh);
            expected.at(k) = quantityInterp.integ(fehDist.getMin(), fehDist.getMax()) / weightIntegral;
        }

        const auto actual = galaxy.yieldsRate(age);
        if (actual.size() != expected.size())
        {
            std::cerr << "testGalaxy: yieldsRateMultiFeh: size mismatch: "
                << actual.size() << " vs " << expected.size() << "\n";
            return 1;
        }
        for (std::size_t k = 0; k < expected.size(); ++k)
        {
            if (std::abs(actual.at(k) - expected.at(k)) >
                relTol * std::max(1.0, std::abs(expected.at(k))))
            {
                std::cerr << "testGalaxy: yieldsRateMultiFeh: entry " << k <<
                    ": yieldsRate(t) = " << actual.at(k) << ", expected " <<
                    expected.at(k) << "\n";
                return 1;
            }
        }
        if (std::reduce(expected.begin(), expected.end(), 0.0) <= 0.0)
        {
            std::cerr << "testGalaxy: yieldsRateMultiFeh: expected a positive "
                "total yield rate\n";
            return 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testGalaxy: yieldsRateMultiFeh test failed: " << error.what() << "\n";
        return 1;
    }
    return 0;
}

// Verify Galaxy::computeYields()'s clustered-population share: with
// f_cluster = 1 (every star clustered, so fieldYields_ and the
// continuous-population integral both stay exactly zero -- see
// computeYields()'s own header comment), galaxy.yields() should equal
// the sum of Cluster::yields()'s own cumulative total over every
// cluster in clusters() and disruptedClusters(), bit for bit --
// calling cluster.yields() again here after galaxy.yields() already
// has is safe/idempotent (its own lastYieldTime_ is already at
// curTime_, so it just returns the cached total, not recomputing it).
// Reuses testYieldsRateHydrogenOrderOfMagnitude()'s own yields setup.
static auto testGalaxyYieldsClusteredOnly() -> int
{
    constexpr double age = 1e8;

    try
    {
        toml::table inputDeck = toml::parse_file(inputFile);
        inputDeck.at_path("clusters").as_table()->insert_or_assign("f_cluster", 1.0);
        // A fixed 1e4 Msun cluster mass (a delta-function CMF, like
        // testCluster.cpp's own testClusterYieldsStochastic()) rather
        // than testGalaxyDynamicsCMF.toml's own 100-1000 Msun power law
        // -- clusters that small essentially never sample a star above
        // the yield channels' own 13 Msun floor at all (verified: with
        // the small CMF, the single most massive star drawn across
        // every cluster this test formed topped out at 11.4 Msun), so
        // every cluster needs to be big enough to reliably sample some.
        inputDeck.at_path("clusters").as_table()->insert_or_assign("CMF", 1e4);
        inputDeck.insert("yields", toml::table{
            { "channel1", toml::table{ { "channel", "ccsn" }, { "model", "kobayashi_test" } } },
            { "channel2", toml::table{ { "channel", "ccsn" }, { "model", "sukhbold_test" } } },
            { "channel3", toml::table{ { "channel", "massive_star_winds" }, { "model", "sukhbold_test" } } },
            { "channel_decomposed", false },
            { "registry", std::string("tests/yields/assets/yields.toml") },
        });
        const io::SimControls controls(inputDeck);

        utils::rng().seed(rngSeed);
        core::Galaxy galaxy(controls);
        galaxy.advance(age);

        if (galaxy.clusters().empty() && galaxy.disruptedClusters().empty())
        {
            std::cerr << "testGalaxy: yieldsRateClusteredOnly: expected at "
                "least one cluster to have formed by age " << age << "\n";
            return 1;
        }

        const auto& actual = galaxy.yields();

        std::vector<double> expected(actual.size(), 0.0);
        for (auto& cluster : galaxy.clusters())
        {
            const auto& cy = cluster.yields();
            for (std::size_t k = 0; k < cy.size(); ++k) { expected.at(k) += cy.at(k); }
        }
        for (auto& cluster : galaxy.disruptedClusters())
        {
            const auto& cy = cluster.yields();
            for (std::size_t k = 0; k < cy.size(); ++k) { expected.at(k) += cy.at(k); }
        }

        if (actual.size() != expected.size())
        {
            std::cerr << "testGalaxy: yieldsRateClusteredOnly: size mismatch: "
                << actual.size() << " vs " << expected.size() << "\n";
            return 1;
        }
        for (std::size_t k = 0; k < expected.size(); ++k)
        {
            if (actual.at(k) != expected.at(k))
            {
                std::cerr << "testGalaxy: yieldsRateClusteredOnly: entry " << k <<
                    ": galaxy.yields() = " << actual.at(k) << ", expected " <<
                    expected.at(k) << "\n";
                return 1;
            }
        }
        if (std::reduce(expected.begin(), expected.end(), 0.0) <= 0.0)
        {
            std::cerr << "testGalaxy: yieldsRateClusteredOnly: expected a "
                "positive total yield\n";
            return 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testGalaxy: yieldsRateClusteredOnly test failed: " << error.what() << "\n";
        return 1;
    }
    return 0;
}

// Verify Galaxy::computeYields()'s field-star and purely-continuous
// shares together (f_cluster = 0, so clusters()/disruptedClusters()
// stay empty and yields_ == fieldYields_ exactly): independently
// recomputes the field-star share as the sum of
// controls.yields()->yieldSum(mass_, feh_) over every star in
// deadFieldStars() (the same per-star call computeYields() itself
// makes), and the continuous share as a composite Simpson's-rule
// integral of yieldsRate(t) -- a different quadrature from
// computeYields()'s own adaptive Gauss-Kronrod utils::GKIntegrator,
// so this is a genuine independent check, not a call into the same
// code path. min_stoch_mass = 50 (unlike
// testYieldsRateHydrogenOrderOfMagnitude()'s 120) is chosen to fall
// strictly inside the yield channels' own covered mass range
// ([13, 100], see tests/yields/assets/yields.toml), splitting it
// between the continuous population ([imf().getMin(), 50], including
// the yield-covered [13, 50]) and field stars ([50, imf().getMax()]
// = [50, 120], including the yield-covered [50, 100]) -- so both
// shares are actually exercised with nonzero contributions, not just
// the zero one either would trivially get called with no yield
// coverage at all.
static auto testGalaxyYieldsFieldAndContinuous() -> int
{
    constexpr double age = 1e8;
    constexpr std::size_t nSimpson = 2000; // even, for composite Simpson's rule
    // Composite Simpson's rule vs. adaptive Gauss-Kronrod -- looser than
    // a typical cross-check because yieldsRate(t) is not smooth here: as
    // t increases, the stellar mass corresponding to that lifetime
    // (massAndDerivFromLifetime()) sweeps across the yield channels' own
    // tabulated mass boundaries (13/18/18.2/100 Msun, see
    // tests/yields/assets/yields.toml), each a small jump in
    // hasYield()'s own coverage and so a kink in yieldsRate(t) itself --
    // composite Simpson only converges linearly (not its usual 4th
    // order) across a kink, so 0.1%-level agreement isn't reachable at a
    // sample count that still runs in reasonable time; empirically, the
    // observed disagreement dropped from ~1.2% at nSimpson = 200 to
    // ~0.27% at 2000 (i.e. shrinking, not a fixed offset -- consistent
    // with discretization error, not a real bug), so 0.5% comfortably
    // covers it with margin.
    constexpr double relTol = 5e-3;

    try
    {
        toml::table inputDeck = toml::parse_file(inputFile);
        inputDeck.at_path("clusters").as_table()->insert_or_assign("f_cluster", 0.0);
        inputDeck.at_path("stars").as_table()->insert_or_assign("min_stoch_mass", 50.0);
        inputDeck.insert("yields", toml::table{
            { "channel1", toml::table{ { "channel", "ccsn" }, { "model", "kobayashi_test" } } },
            { "channel2", toml::table{ { "channel", "ccsn" }, { "model", "sukhbold_test" } } },
            { "channel3", toml::table{ { "channel", "massive_star_winds" }, { "model", "sukhbold_test" } } },
            { "channel_decomposed", false },
            { "registry", std::string("tests/yields/assets/yields.toml") },
        });
        const io::SimControls controls(inputDeck);

        if (controls.yields() == nullptr)
        {
            std::cerr << "testGalaxy: yieldsRateFieldAndContinuous: expected "
                "yields() non-null\n";
            return 1;
        }
        const auto& isotopes = controls.yields()->isotopes();
        const auto h1It = std::ranges::find_if(isotopes,
            [](const auto& iso) { return iso.get().Z() == 1 && iso.get().A() == 1; });
        if (h1It == isotopes.end())
        {
            std::cerr << "testGalaxy: yieldsRateFieldAndContinuous: expected "
                "hydrogen (Z=1, A=1) among isotopes()\n";
            return 1;
        }
        const auto h1Idx = static_cast<std::size_t>(std::distance(isotopes.begin(), h1It));

        utils::rng().seed(rngSeed);
        core::Galaxy galaxy(controls);
        galaxy.advance(age);

        if (galaxy.deadFieldStars().empty())
        {
            std::cerr << "testGalaxy: yieldsRateFieldAndContinuous: expected "
                "at least one dead field star by age " << age << "\n";
            return 1;
        }

        double fieldH1 = 0.0;
        for (const auto& fs : galaxy.deadFieldStars())
        {
            fieldH1 += controls.yields()->yieldSum(fs.mass_, fs.feh_).at(h1Idx);
        }
        if (!(fieldH1 > 0.0))
        {
            std::cerr << "testGalaxy: yieldsRateFieldAndContinuous: expected "
                "a positive field-star hydrogen yield, got " << fieldH1 << "\n";
            return 1;
        }

        // Composite Simpson's rule for int_0^age yieldsRate(t)[h1Idx] dt,
        // independent of computeYields()'s own adaptive quadrature. The
        // t = 0 endpoint is taken as exactly 0 rather than calling
        // yieldsRate(0.0) -- analytically correct (no star has had time
        // to die yet), and avoids yieldsRate(t, feh)'s own internal
        // integrator.integrate(0.0, 0.0, ...) zero-width-interval edge
        // case, whose relative-error check divides by a zero quadrature
        // estimate.
        const double h = age / static_cast<double>(nSimpson);
        double contH1 = 0.0 + galaxy.yieldsRate(age).at(h1Idx);
        for (std::size_t i = 1; i < nSimpson; ++i)
        {
            const double t = static_cast<double>(i) * h;
            const double weight = (i % 2 == 0) ? 2.0 : 4.0;
            contH1 += weight * galaxy.yieldsRate(t).at(h1Idx);
        }
        contH1 *= h / 3.0;
        if (!(contH1 > 0.0))
        {
            std::cerr << "testGalaxy: yieldsRateFieldAndContinuous: expected "
                "a positive continuous-population hydrogen yield, got " <<
                contH1 << "\n";
            return 1;
        }

        const double expectedH1 = fieldH1 + contH1;
        const double actualH1 = galaxy.yields().at(h1Idx);
        if (std::abs(actualH1 - expectedH1) > relTol * std::abs(expectedH1))
        {
            std::cerr << "testGalaxy: yieldsRateFieldAndContinuous: "
                "galaxy.yields()[h1] = " << actualH1 << ", expected " <<
                expectedH1 << " (field " << fieldH1 << " + continuous " <<
                contH1 << ")\n";
            return 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testGalaxy: yieldsRateFieldAndContinuous test failed: "
            << error.what() << "\n";
        return 1;
    }
    return 0;
}

// Regression test: verify that field-star deaths from an earlier
// advance() call are not silently lost when yields() is never called
// in between two advance() calls -- mirrors testCluster.cpp's own
// testClusterYieldsMultipleAdvanceCalls(), one level up in Galaxy:
// before computeYields() was moved to run eagerly at the end of
// advance() itself (see lastYieldTime_'s own comment), deadFieldStars_
// -- which only ever holds the deaths from the single most recently
// advance() call, see advance()'s own step 6 -- would be overwritten
// by the second advance() call's own step 6 before ever being consumed
// into fieldYields_. Reuses testGalaxyYieldsFieldAndContinuous()'s own
// setup and age exactly (min_stoch_mass = 50, default sfr, age = 1e8),
// splitting its single advance(age) into two steps at age / 2 with no
// yields() call in between, so expected is built the same way that
// test's own is: the direct field-star sum (now from both steps'
// deadFieldStars(), rather than just one) plus a composite Simpson's
// rule integral of yieldsRate(t) -- still over the whole [0, age], by
// additivity of integration, regardless of the intermediate step.
static auto testGalaxyYieldsMultipleAdvanceCalls() -> int
{
    constexpr double age = 1e8;
    constexpr std::size_t nSimpson = 2000; // even, for composite Simpson's rule
    constexpr double relTol = 5e-3; // see testGalaxyYieldsFieldAndContinuous()'s own comment

    try
    {
        toml::table inputDeck = toml::parse_file(inputFile);
        inputDeck.at_path("clusters").as_table()->insert_or_assign("f_cluster", 0.0);
        inputDeck.at_path("stars").as_table()->insert_or_assign("min_stoch_mass", 50.0);
        inputDeck.insert("yields", toml::table{
            { "channel1", toml::table{ { "channel", "ccsn" }, { "model", "kobayashi_test" } } },
            { "channel2", toml::table{ { "channel", "ccsn" }, { "model", "sukhbold_test" } } },
            { "channel3", toml::table{ { "channel", "massive_star_winds" }, { "model", "sukhbold_test" } } },
            { "channel_decomposed", false },
            { "registry", std::string("tests/yields/assets/yields.toml") },
        });
        const io::SimControls controls(inputDeck);

        if (controls.yields() == nullptr)
        {
            std::cerr << "testGalaxy: yieldsMultipleAdvanceCalls: expected "
                "yields() non-null\n";
            return 1;
        }
        const auto& isotopes = controls.yields()->isotopes();
        const auto h1It = std::ranges::find_if(isotopes,
            [](const auto& iso) { return iso.get().Z() == 1 && iso.get().A() == 1; });
        if (h1It == isotopes.end())
        {
            std::cerr << "testGalaxy: yieldsMultipleAdvanceCalls: expected "
                "hydrogen (Z=1, A=1) among isotopes()\n";
            return 1;
        }
        const auto h1Idx = static_cast<std::size_t>(std::distance(isotopes.begin(), h1It));

        utils::rng().seed(rngSeed);
        core::Galaxy galaxy(controls);

        double fieldH1 = 0.0;

        // First advance, to the halfway point -- deliberately not
        // calling yields() afterward
        galaxy.advance(age / 2.0);
        const auto deadFirst = galaxy.deadFieldStars();
        if (deadFirst.empty())
        {
            std::cerr << "testGalaxy: yieldsMultipleAdvanceCalls: test bug: "
                "expected some dead field stars by " << (age / 2.0) <<
                " yr, the first of two advance() calls\n";
            return 1;
        }
        for (const auto& fs : deadFirst)
        {
            fieldH1 += controls.yields()->yieldSum(fs.mass_, fs.feh_).at(h1Idx);
        }

        // Second advance, to the final age -- again not calling
        // yields() in between
        galaxy.advance(age);
        for (const auto& fs : galaxy.deadFieldStars())
        {
            fieldH1 += controls.yields()->yieldSum(fs.mass_, fs.feh_).at(h1Idx);
        }
        if (!(fieldH1 > 0.0))
        {
            std::cerr << "testGalaxy: yieldsMultipleAdvanceCalls: expected a "
                "positive field-star hydrogen yield, got " << fieldH1 << "\n";
            return 1;
        }

        // Composite Simpson's rule for int_0^age yieldsRate(t)[h1Idx]
        // dt -- see testGalaxyYieldsFieldAndContinuous()'s own comment
        const double h = age / static_cast<double>(nSimpson);
        double contH1 = 0.0 + galaxy.yieldsRate(age).at(h1Idx);
        for (std::size_t i = 1; i < nSimpson; ++i)
        {
            const double t = static_cast<double>(i) * h;
            const double weight = (i % 2 == 0) ? 2.0 : 4.0;
            contH1 += weight * galaxy.yieldsRate(t).at(h1Idx);
        }
        contH1 *= h / 3.0;

        const double expectedH1 = fieldH1 + contH1;
        const double actualH1 = galaxy.yields().at(h1Idx);
        if (std::abs(actualH1 - expectedH1) > relTol * std::abs(expectedH1))
        {
            std::cerr << "testGalaxy: yieldsMultipleAdvanceCalls: "
                "galaxy.yields()[h1] = " << actualH1 << ", expected " <<
                expectedH1 << " (field " << fieldH1 << " + continuous " <<
                contH1 << ") -- deaths from the first advance() call may "
                "have been lost\n";
            return 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testGalaxy: yieldsMultipleAdvanceCalls test failed: "
            << error.what() << "\n";
        return 1;
    }
    return 0;
}

// Verify Galaxy::computeYields()'s field-star share of Ni56 (unstable,
// real lifetime ~0.024 yr -- vs. hydrogen, stable and so unaffected by
// decay either way, which every non-decay yields test in this file
// checks instead), with radioactive decay enabled (yields.no_decay
// left at its default, false).
//
// Unlike testGalaxyYieldsFieldAndContinuous() (which cross-checks
// hydrogen's own *undecayed* continuous share against a composite
// Simpson's rule integral of yieldsRate(t)), this does not attempt the
// analogous check for Ni56's *decayed* continuous share: with Ni56's
// own lifetime many orders of magnitude shorter than age, the decayed
// integrand galaxy.yieldsRate(t)[Ni56] is an extremely narrow spike
// concentrated within a lifetime or so of t = curTime_ -- far too
// narrow for a low-order quadrature (composite Simpson at any sample
// count that still runs quickly, or, for that matter, Yields's own
// internal adaptive Gauss-Kronrod) to resolve reliably. Physically,
// this narrowness means the true continuous contribution is
// necessarily tiny (bounded above by roughly the raw, undecayed
// instantaneous rate at age times Ni56's own lifetime -- see
// continuousUpperBound below), which is exactly why this test can
// safely ignore it and still get a meaningful check: the field-star
// share, by contrast, needs no integration at all (each field star's
// own dtDecay = age - deathTime_ is an exact, known number, so
// yieldSum() evaluates a closed-form expression, not a quadrature), so
// galaxy.yields()[Ni56] is checked against fieldNi (the exact,
// independently-recomputed field-star-only sum) to within a tolerance
// set by that same physical bound on the continuous share.
static auto testGalaxyYieldsFieldAndContinuousDecay() -> int
{
    constexpr double age = 1e8;

    try
    {
        toml::table inputDeck = toml::parse_file(inputFile);
        inputDeck.at_path("clusters").as_table()->insert_or_assign("f_cluster", 0.0);
        inputDeck.at_path("stars").as_table()->insert_or_assign("min_stoch_mass", 50.0);
        inputDeck.insert("yields", toml::table{
            { "channel1", toml::table{ { "channel", "ccsn" }, { "model", "kobayashi_test" } } },
            { "channel2", toml::table{ { "channel", "ccsn" }, { "model", "sukhbold_test" } } },
            { "channel3", toml::table{ { "channel", "massive_star_winds" }, { "model", "sukhbold_test" } } },
            { "channel_decomposed", false },
            { "registry", std::string("tests/yields/assets/yields.toml") },
        });
        const io::SimControls controls(inputDeck);

        if (controls.yields() == nullptr)
        {
            std::cerr << "testGalaxy: yieldsFieldAndContinuousDecay: expected "
                "yields() non-null\n";
            return 1;
        }
        const auto& isotopes = controls.yields()->isotopes();
        const auto niIt = std::ranges::find_if(isotopes,
            [](const auto& iso) { return iso.get().Z() == 28 && iso.get().A() == 56; });
        if (niIt == isotopes.end())
        {
            std::cerr << "testGalaxy: yieldsFieldAndContinuousDecay: expected "
                "Ni56 (Z=28, A=56) among isotopes()\n";
            return 1;
        }
        const auto niIdx = static_cast<std::size_t>(std::distance(isotopes.begin(), niIt));

        utils::rng().seed(rngSeed);
        core::Galaxy galaxy(controls);
        galaxy.advance(age);

        if (galaxy.deadFieldStars().empty())
        {
            std::cerr << "testGalaxy: yieldsFieldAndContinuousDecay: expected "
                "at least one dead field star by age " << age << "\n";
            return 1;
        }

        double fieldNi = 0.0;
        double fieldNiUndecayed = 0.0;
        for (const auto& fs : galaxy.deadFieldStars())
        {
            fieldNi += controls.yields()->yieldSum(fs.mass_, fs.feh_, age - fs.deathTime_).at(niIdx);
            fieldNiUndecayed += controls.yields()->yieldSum(fs.mass_, fs.feh_, 0.0).at(niIdx);
        }
        if (!(fieldNiUndecayed > 0.0))
        {
            std::cerr << "testGalaxy: yieldsFieldAndContinuousDecay: test bug: "
                "expected a positive undecayed field-star Ni56 yield\n";
            return 1;
        }

        // Physical upper bound on the continuous population's own
        // present-day Ni56 contribution -- see this test's own header
        // comment. galaxy.yieldsRate(age)[Ni56] here is the *decayed*
        // rate (dtDecay = curTime_ - age = 0, so this is also the raw,
        // undecayed instantaneous rate), and ni56.lifetime() is Ni56's
        // own mean lifetime, in yr.
        const auto& ni56 = elem::isotopeTable(28U, 56U);
        const double continuousUpperBound = galaxy.yieldsRate(age).at(niIdx) * ni56.lifetime();

        const double actualNi = galaxy.yields().at(niIdx);
        const double tolerance = (10.0 * continuousUpperBound) + (1e-9 * fieldNiUndecayed);
        if (std::abs(actualNi - fieldNi) > tolerance)
        {
            std::cerr << "testGalaxy: yieldsFieldAndContinuousDecay: "
                "galaxy.yields()[Ni56] = " << actualNi << ", expected " <<
                fieldNi << " (exact field-star share) within tolerance " <<
                tolerance << " (continuous upper bound " <<
                continuousUpperBound << ")\n";
            return 1;
        }

        // Sanity check that decay actually changed something relative
        // to the undecayed field-star total -- otherwise this test
        // could pass vacuously if dtDecay were silently always 0. With
        // age (1e8 yr) so many orders of magnitude beyond Ni56's own
        // lifetime (~0.024 yr), essentially none should survive.
        if (actualNi > 0.1 * fieldNiUndecayed)
        {
            std::cerr << "testGalaxy: yieldsFieldAndContinuousDecay: "
                "galaxy.yields()[Ni56] = " << actualNi << " is not much "
                "smaller than the undecayed field-star total " <<
                fieldNiUndecayed << " -- decay does not appear to be applied\n";
            return 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testGalaxy: yieldsFieldAndContinuousDecay test failed: "
            << error.what() << "\n";
        return 1;
    }
    return 0;
}

// Verify that fieldYields_'s own already-accumulated total is
// correctly aged forward (via Yields::applyDecay()) across multiple
// advance() calls -- see Galaxy::computeYields()'s own comment for why
// this incremental aging step exists at all (fieldYields_, unlike
// Cluster::yields_, keeps no memory of which isotope came from which
// star or when). Otherwise mirrors
// testGalaxyYieldsFieldAndContinuousDecay() (same reasoning for why
// only the field-star share, not the continuous one, is checked
// precisely), but splits the single advance(age) into two steps at
// age / 2, as testGalaxyYieldsMultipleAdvanceCalls() does for
// hydrogen. Field-star contributions from *either* step are recomputed
// with dtDecay = age - deathTime_ (decayed all the way to the final
// age, regardless of which step the star actually died in) -- if the
// aging step were missing entirely, the first step's own field stars
// would be under-decayed relative to this (decayed only to age / 2,
// not all the way to age), and, since Ni56's field-star contribution
// is not negligible the way the continuous share is (see
// testGalaxyYieldsFieldAndContinuousDecay()'s own comment), this test
// would catch that.
static auto testGalaxyYieldsMultipleAdvanceCallsDecay() -> int
{
    constexpr double age = 1e8;

    try
    {
        toml::table inputDeck = toml::parse_file(inputFile);
        inputDeck.at_path("clusters").as_table()->insert_or_assign("f_cluster", 0.0);
        inputDeck.at_path("stars").as_table()->insert_or_assign("min_stoch_mass", 50.0);
        inputDeck.insert("yields", toml::table{
            { "channel1", toml::table{ { "channel", "ccsn" }, { "model", "kobayashi_test" } } },
            { "channel2", toml::table{ { "channel", "ccsn" }, { "model", "sukhbold_test" } } },
            { "channel3", toml::table{ { "channel", "massive_star_winds" }, { "model", "sukhbold_test" } } },
            { "channel_decomposed", false },
            { "registry", std::string("tests/yields/assets/yields.toml") },
        });
        const io::SimControls controls(inputDeck);

        if (controls.yields() == nullptr)
        {
            std::cerr << "testGalaxy: yieldsMultipleAdvanceCallsDecay: expected "
                "yields() non-null\n";
            return 1;
        }
        const auto& isotopes = controls.yields()->isotopes();
        const auto niIt = std::ranges::find_if(isotopes,
            [](const auto& iso) { return iso.get().Z() == 28 && iso.get().A() == 56; });
        if (niIt == isotopes.end())
        {
            std::cerr << "testGalaxy: yieldsMultipleAdvanceCallsDecay: expected "
                "Ni56 (Z=28, A=56) among isotopes()\n";
            return 1;
        }
        const auto niIdx = static_cast<std::size_t>(std::distance(isotopes.begin(), niIt));

        utils::rng().seed(rngSeed);
        core::Galaxy galaxy(controls);

        double fieldNi = 0.0;
        double fieldNiUndecayed = 0.0;

        galaxy.advance(age / 2.0);
        const auto deadFirst = galaxy.deadFieldStars();
        if (deadFirst.empty())
        {
            std::cerr << "testGalaxy: yieldsMultipleAdvanceCallsDecay: test bug: "
                "expected some dead field stars by " << (age / 2.0) <<
                " yr, the first of two advance() calls\n";
            return 1;
        }
        for (const auto& fs : deadFirst)
        {
            fieldNi += controls.yields()->yieldSum(fs.mass_, fs.feh_, age - fs.deathTime_).at(niIdx);
            fieldNiUndecayed += controls.yields()->yieldSum(fs.mass_, fs.feh_, 0.0).at(niIdx);
        }

        galaxy.advance(age);
        for (const auto& fs : galaxy.deadFieldStars())
        {
            fieldNi += controls.yields()->yieldSum(fs.mass_, fs.feh_, age - fs.deathTime_).at(niIdx);
            fieldNiUndecayed += controls.yields()->yieldSum(fs.mass_, fs.feh_, 0.0).at(niIdx);
        }
        if (!(fieldNiUndecayed > 0.0))
        {
            std::cerr << "testGalaxy: yieldsMultipleAdvanceCallsDecay: test bug: "
                "expected a positive undecayed field-star Ni56 yield\n";
            return 1;
        }

        const auto& ni56 = elem::isotopeTable(28U, 56U);
        const double continuousUpperBound = galaxy.yieldsRate(age).at(niIdx) * ni56.lifetime();

        const double actualNi = galaxy.yields().at(niIdx);
        const double tolerance = (10.0 * continuousUpperBound) + (1e-9 * fieldNiUndecayed);
        if (std::abs(actualNi - fieldNi) > tolerance)
        {
            std::cerr << "testGalaxy: yieldsMultipleAdvanceCallsDecay: "
                "galaxy.yields()[Ni56] = " << actualNi << ", expected " <<
                fieldNi << " (exact field-star share) within tolerance " <<
                tolerance << " (continuous upper bound " <<
                continuousUpperBound << ") -- fieldYields_ may not be aged "
                "forward correctly across advance() calls\n";
            return 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testGalaxy: yieldsMultipleAdvanceCallsDecay test failed: "
            << error.what() << "\n";
        return 1;
    }
    return 0;
}

auto testGalaxy() -> int
{
    int result = testGalaxyBasics();
    result += testGalaxySFRDistResolution();
    result += testFCluster();
    result += testContinuousPopSpecSingleFeh();
    result += testContinuousPopSpecMultiFeh();
    result += testContinuousPopSpecReferenceCheck();
    result += testContinuousPopLbolStandaloneMatchesSpec();
    result += testFieldStarsMassBudget();
    result += testFieldStarsCreationAndDeath();
    result += testGalaxyFieldStarLifetimeClamped();
    result += testFieldStarsSpec();
    result += testFieldStarsExtinct();
    result += testExtinctApplyExtinctionCtsDegenerate();
    result += testExtinctApplyExtinctionCtsUniform();
    result += testGalaxyNebular();
    result += testContinuousPopNebularExtinct();
    result += testYieldsRateHydrogenOrderOfMagnitude();
    result += testYieldsRateSingleFehDelegates();
    result += testYieldsRateMultiFeh();
    result += testGalaxyYieldsClusteredOnly();
    result += testGalaxyYieldsFieldAndContinuous();
    result += testGalaxyYieldsMultipleAdvanceCalls();
    result += testGalaxyYieldsFieldAndContinuousDecay();
    result += testGalaxyYieldsMultipleAdvanceCallsDecay();

    try
    {
        const toml::table inputDeck = toml::parse_file(inputFile);
        const io::SimControls controls(inputDeck);

        result += checkConstruction(controls);

        // Build a single galaxy and advance it through both t1 and t2,
        // checking mass/age conservation at each step as it happens,
        // then run every other check against the resulting state --
        // one build serves every check below, rather than each
        // rebuilding (and re-forming every cluster) independently.
        utils::rng().seed(rngSeed);
        core::Galaxy galaxy(controls);
        result += advanceAndCheckMassAge(galaxy, controls, 0.0, t1);
        result += advanceAndCheckMassAge(galaxy, controls, t1, t2);

        result += checkSpecSum(galaxy, controls);
        result += checkPhotSum(galaxy, controls);
        result += checkLbolSum(galaxy);
        result += checkDisruption(galaxy);

        return result;
    }
    catch (const std::exception& error)
    {
        std::cerr << "testGalaxy: test failed: " << error.what() << "\n";
        return 1;
    }
}
