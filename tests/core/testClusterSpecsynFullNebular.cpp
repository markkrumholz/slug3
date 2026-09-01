/**
 * @file testClusterSpecsynFullNebular.cpp
 * @author Mark Krumholz
 * @brief Optional full end-to-end test of nebular emission against the real cloudy grid.
 * @details
 * Unlike every other nebular-emission test in this repository (which
 * uses the small synthetic fixture built by
 * data/tools/cloudy/make_nebular_test_fixture.py), this one exercises
 * the real, gitignored data/nebular/nebular.h5 cloudy grid together
 * with the real MIST tracks and full chained spectral synthesis (see
 * testClusterSpecsynFull's own comment for that shared rationale), so
 * it runs only if both sets of data files are present locally --
 * otherwise it is skipped, returning an automatic pass. The real grid
 * now fully covers every [Fe/H]/v_vcrit combination the tracks it's
 * built against actually offer, so SimControls construction failing
 * here -- Nebular's own constructor eagerly loads every [Fe/H] group
 * under the requested track, not just the one this test's own deck
 * asks for (see Nebular.cpp's own constructor) -- is a real test
 * failure (a genuine coverage regression in the grid, or a defect in
 * Nebular's own loading logic), not an expected, tolerated gap.
 *
 * There is no independently-computed expected spectrum or line
 * luminosity to check bit-for-bit against here (that is exactly what
 * the small-fixture tests in tests/nebular/ and tests/core/testCluster.cpp
 * already do). This test instead checks that the real grid's output is
 * *physically reasonable*: the stellar+nebular spectrum should be the
 * same order of magnitude as the stellar-only spectrum (nebular
 * reprocessing redistributes energy, it does not manufacture or
 * destroy an order of magnitude of it) while still differing from it
 * somewhere (nebular continuum/line emission actually did something),
 * and the total line luminosity should be positive, finite, and not
 * wildly exceed the ionizing photon power budget (Q(HI) times a
 * representative photon energy) that ultimately powers it.
 * @date 2026-08-30
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#include "testClusterSpecsynFullNebular.hpp"
#include "../src/core/Cluster.hpp"
#include "../src/io/SimControls.hpp"
#include "../src/phot/FilterIdeal.hpp"
#include "../src/specsyn/Specsyn.hpp"
#include "../src/utils/RngThread.hpp"
#include "testClusterSpecsynFullCommon.hpp"
#include <cmath>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <iostream>
#include <toml.hpp>
#include <vector>

// Check that specNeb's integrated (trapezoidal) luminosity over wl is
// within an order of magnitude of spec's own, while still differing
// from it somewhere pointwise -- see this file's own header comment
// for why that is the "reasonable" bar for real-grid nebular
// reprocessing rather than a bit-exact comparison. Prints its own
// diagnostic and returns false on any violation; true otherwise.
// Factored out of testClusterSpecsynFullNebular() to keep it within
// its own cognitive-complexity budget.
auto checkSpecVsSpecNebReasonable(const std::vector<double>& spec,
    const std::vector<double>& specNeb, const std::vector<double>& wl) -> bool
{
    double trapzStellar = 0.0;
    double trapzNeb = 0.0;
    bool anyDifferent = false;
    for (std::size_t i = 0; i + 1 < wl.size(); ++i)
    {
        trapzStellar += 0.5 * (spec.at(i) + spec.at(i + 1)) * (wl.at(i + 1) - wl.at(i));
        trapzNeb += 0.5 * (specNeb.at(i) + specNeb.at(i + 1)) * (wl.at(i + 1) - wl.at(i));
    }
    for (std::size_t i = 0; i < spec.size(); ++i)
    {
        if (spec.at(i) != specNeb.at(i)) { anyDifferent = true; break; }
    }

    if (!std::isfinite(trapzStellar) || !std::isfinite(trapzNeb) ||
        trapzStellar <= 0.0 || trapzNeb <= 0.0)
    {
        std::cerr << "testClusterSpecsynFullNebular: expected finite, "
            "positive integrated stellar and stellar+nebular luminosities, "
            "got " << trapzStellar << " and " << trapzNeb << "\n";
        return false;
    }
    if (!anyDifferent)
    {
        std::cerr << "testClusterSpecsynFullNebular: expected specNeb() to "
            "differ from spec() somewhere, but they are identical\n";
        return false;
    }
    constexpr double orderOfMagnitudeFactor = 10.0;
    const double ratio = trapzNeb / trapzStellar;
    if (ratio < (1.0 / orderOfMagnitudeFactor) || ratio > orderOfMagnitudeFactor)
    {
        std::cerr << "testClusterSpecsynFullNebular: integrated "
            "stellar+nebular luminosity (" << trapzNeb <<
            ") is not within an order of magnitude of the integrated "
            "stellar-only luminosity (" << trapzStellar << ")\n";
        return false;
    }
    return true;
}

// Run the full simulation described by
// tests/core/assets/testClusterSpecsynFullNebular.in (real MIST
// tracks, v_vcrit = 0.4, alphaFe = 0, fixed FeH = 0; the full "default"
// chained spectral synthesis model; a single 10^5 Msun cluster at
// 1 Myr; the real data/nebular/nebular.h5 cloudy grid) and check that
// the resulting spectrum/line luminosities are physically reasonable
// -- see this file's own header comment for exactly what "reasonable"
// means here.
auto testClusterSpecsynFullNebular() -> int
{
    if (!allRequiredDataFilesExist()) { return 0; }
    if (!std::filesystem::exists("data/nebular/nebular.h5")) { return 0; }

    try
    {
        const toml::table inputDeck =
            toml::parse_file("tests/core/assets/testClusterSpecsynFullNebular.in");

        // Nebular's own constructor eagerly loads spec/line_lum data
        // for *every* FeH group under the requested track (not just
        // the one [Fe/H] value this test's own deck actually asks
        // for -- see Nebular.cpp's own constructor, which loops over
        // every childrenByAttr(trackGrp, "FeH", ...) result), so a
        // hole anywhere in MIST's own [Fe/H]/v_vcrit coverage would
        // throw right here, at SimControls construction -- not lazily
        // later, when getCluster() is actually called for this deck's
        // own [Fe/H] = 0. The real cloudy grid now fully covers every
        // such combination (see this file's own header comment), so
        // this is allowed to throw straight out to the catch below,
        // like any other genuine test failure.
        const io::SimControls controls(inputDeck);

        if (controls.nebular() == nullptr)
        {
            std::cerr << "testClusterSpecsynFullNebular: test bug: expected "
                "SimControls::nebular() to be non-null\n";
            return 1;
        }

        utils::rng().seed(42);
        core::Cluster cluster(0, 1e5, 0.0, controls);
        constexpr double ageYr = 1e6;
        cluster.advance(ageYr);

        const auto& spec = cluster.spec();
        const auto& specNeb = cluster.specNeb();
        if (spec.empty() || specNeb.empty() || spec.size() != specNeb.size())
        {
            std::cerr << "testClusterSpecsynFullNebular: expected non-empty, "
                "equal-sized spec()/specNeb()\n";
            return 1;
        }

        // "Same order of magnitude" is checked on the bolometric-like
        // integrated luminosity rather than pointwise, since nebular
        // reprocessing can locally swing a narrow bin (e.g. a strong
        // line-deposit window) by orders of magnitude while leaving
        // the overall SED comparable
        const auto& wl = controls.specsyn()->wl();
        if (!checkSpecVsSpecNebReasonable(spec, specNeb, wl)) { return 1; }

        // Cross-check the total line luminosity against the ionizing
        // photon power budget: Q(HI) (independently computed here via
        // phot::FilterIdeal, mirroring tests/nebular/testNebular.hpp's
        // own cross-check, rather than reusing any of Nebular's own
        // internal machinery) times a representative ~13.6 eV photon
        // energy bounds how much power is actually available to power
        // case-B recombination line emission -- Nebular's own
        // covFac_ (see its comment) discounts this further, well
        // inside the generous safety factor used below.
        const phot::FilterIdeal qhiFilter("Q(HI)");
        const double qhi = qhiFilter.phot(wl, spec);
        if (!std::isfinite(qhi) || qhi <= 0.0)
        {
            std::cerr << "testClusterSpecsynFullNebular: test bug: expected a "
                "finite, positive Q(HI) from this massive-star population, got "
                << qhi << "\n";
            return 1;
        }
        constexpr double hIonizingEnergyErg = 2.179e-11; // ~13.6 eV, in erg
        const double ionizingPowerErgS = qhi * hIonizingEnergyErg;

        const auto& lineLum = cluster.lineLum();
        if (lineLum.empty())
        {
            std::cerr << "testClusterSpecsynFullNebular: expected a non-empty "
                "lineLum()\n";
            return 1;
        }
        double totalLineLum = 0.0;
        for (const double l : lineLum)
        {
            if (!std::isfinite(l) || l < 0.0)
            {
                std::cerr << "testClusterSpecsynFullNebular: lineLum() contains "
                    "a non-finite or negative value (" << l << ")\n";
                return 1;
            }
            totalLineLum += l;
        }
        if (!(totalLineLum > 0.0))
        {
            std::cerr << "testClusterSpecsynFullNebular: expected a non-zero "
                "total line luminosity\n";
            return 1;
        }
        constexpr double lineLumSafetyFactor = 10.0; // generous margin -- a sanity bound, not precision physics
        if (totalLineLum > lineLumSafetyFactor * ionizingPowerErgS)
        {
            std::cerr << "testClusterSpecsynFullNebular: total line luminosity ("
                << totalLineLum << " erg/s) exceeds " << lineLumSafetyFactor
                << "x the ionizing photon power budget (" << ionizingPowerErgS
                << " erg/s, from Q(HI) = " << qhi << " photon/s)\n";
            return 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testClusterSpecsynFullNebular test failed: "
            << error.what() << "\n";
        return 1;
    }
    return 0;
}
