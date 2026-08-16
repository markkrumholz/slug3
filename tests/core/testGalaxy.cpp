/**
 * @file testGalaxy.cpp
 * @author Mark Krumholz
 * @brief Unit tests for the Galaxy class.
 * @date 2026-08-10
 */

#include "../src/core/Cluster.hpp"
#include "../src/core/Galaxy.hpp"
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
#include <limits>
#include <numeric>
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
// utils::getID() service, so any cluster whose uid exceeds a marker
// captured immediately before advance() must have been created by it.
static auto advanceAndCheckMassAge(core::Galaxy& galaxy,
    const io::SimControls& controls, const double prevTime, const double t) -> int
{
    const auto marker = utils::getID();
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

        const auto* synth = controls.specsyn();
        const auto& tracks2D = controls.tracks2D();
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
            const double logAge = std::max(std::log10(age), tracks2D.logTMin());
            const auto isochrone = tracks2D.getIsochrone(logAge);

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
        const auto* synth = controls.specsyn();
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

        const auto& tracks2D = controls.tracks2D();
        for (const auto& fs : galaxy.fieldStars())
        {
            const double logT = std::max(
                std::log10(galaxy.curTime() - fs.formTime_), tracks2D.logTMin());
            const auto props = tracks2D.getStar(fs.mass_, logT);
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

// Verify that Galaxy::addFieldStarSpec()/addContinuousSpec() correctly
// extinguish the field-star/continuous-population contributions to
// specExtinct(): each field star by its own aV_ (drawn independently
// from SimControls::avDistField(), via Extinct::applyExtinction()),
// and the continuous population by the expectation value of
// exp(-A_V * extinct()) over avDistField() (via Extinct::
// applyExtinctionCts()) -- rather than the pre-field-star-extinction
// behavior of passing both through unattenuated (still exercised,
// unchanged, by testFieldStarsSpec's own use of inputFile's default
// extinct.AV_field, which is absent and so defaults to a delta at 0 --
// see io::SimControls::avDistField()'s own comment). fCluster = 0
// isolates this from any cluster contribution (which uses its own,
// unrelated avDist()); overrides galaxy.sfr for the same reason
// testFieldStarsMassBudget's own identical override does, and sets
// extinct.AV_field to a fixed value distinct from inputFile's own
// clusters.AV, so field stars draw a genuinely different, independent
// A_V than clusters would.
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

        const auto* ext = controls.extinct();
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

        // Independently recompute the expected specExtinct(): the
        // continuous population's own contribution (mirroring
        // Galaxy::addContinuousSpec()) extinguished via
        // applyExtinctionCts(), plus each field star's own spectrum
        // (mirroring Galaxy::addFieldStarSpec()) extinguished via
        // applyExtinction() at that star's own aV_
        const auto* synth = controls.specsyn();
        std::vector<double> expectedSpecExtinct(ext->wl().size(), 0.0);

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
        const auto contSpecExtinct = ext->applyExtinctionCts(contSpec);
        for (std::size_t i = 0; i < expectedSpecExtinct.size(); ++i)
        { expectedSpecExtinct.at(i) += contSpecExtinct.at(i); }

        const auto& tracks2D = controls.tracks2D();
        for (const auto& fs : galaxy.fieldStars())
        {
            const double logT = std::max(
                std::log10(galaxy.curTime() - fs.formTime_), tracks2D.logTMin());
            const auto props = tracks2D.getStar(fs.mass_, logT);
            const auto starSpec = synth->spec(props, fs.feh_);
            const auto starSpecExtinct = ext->applyExtinction(fs.aV_, starSpec);
            for (std::size_t i = 0; i < expectedSpecExtinct.size(); ++i)
            { expectedSpecExtinct.at(i) += starSpecExtinct.at(i); }
        }

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
        const auto* ext = controls.extinct();
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
        const auto* ext = controls.extinct();
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
        core::Galaxy galaxy(controls);
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

        core::Galaxy galaxy(controls);
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
    result += testFieldStarsSpec();
    result += testFieldStarsExtinct();
    result += testExtinctApplyExtinctionCtsDegenerate();
    result += testExtinctApplyExtinctionCtsUniform();

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
