/**
 * @file testSpecsynLibChained.cpp
 * @author Mark Krumholz
 * @brief Unit tests for the SpecsynLibChained class.
 * @details
 * This file contains end-to-end unit tests for
 * SpecsynLibChained::spec, chaining the same BOSZ_test.h5/spectra.toml
 * and TLUSTY_test.h5 fixtures used by testSpecsynLib.cpp together: one
 * covering a cool, solar-parameter star and the other a hot, massive
 * (OB-like) star. These fixtures require different microTurb values
 * (BOSZ_test was fetched at micro = 0, TLUSTY_test at micro = 10), so
 * these tests all construct the chain with its microTurb argument left
 * at the default (an empty vector), relying on each library's own
 * micro_default registry entry (also 0 and 10, respectively) to
 * resolve the right value per library automatically -- rather than
 * having to spell out {0.0, 10.0} explicitly, which was the only
 * option before SpecsynLib gained per-library registry defaults. The
 * tests check that a star handled by the second library in the chain
 * falls through correctly after the first returns an empty (out-of-
 * bounds) result, that this holds regardless of which library is
 * listed first, and that a star outside every chained library's grid
 * still throws (every chained library uses OOBPolicy::coerce, so this
 * only happens once every library in the star's own GridType-specific
 * chain -- see below -- has failed it via ordinary spec(), and the
 * last one has also failed to force a match via specForce()).
 *
 * Also tests classifyGridType's per-GridType routing (see
 * SpecsynLibChained::spec()'s own comment): a star's GridType (WR,
 * WD, or normal) decides which of three separate chains
 * (wrLibs_/wdLibs_/normalLibs_) is even tried at all, rather than
 * only how much of a pre-dispatch clamp to apply before trying every
 * chained library regardless of type. testClassifyWD checks a star
 * whose properties exceed even WD_test's own log(g) ceiling is still
 * rescued (via SpecsynLibWD::specForce() moving log(g) to WD_test's
 * own nearest populated value) when tClamp = true routes it to
 * GridType::wdGrid, but genuinely fails when tClamp = false forces
 * GridType::normalGrid instead. testClassifyWRRescue checks that
 * GridType::wrGrid classification is unconditional on tClamp (a
 * Wolf-Rayet star far outside every chained library's own log(Teff)
 * range is now rescued the same way regardless of tClamp, via
 * SpecsynLibWR::specForce() searching for the nearest populated
 * (log(R_t), log(Teff)) point); testClassifyWRFehOutOfRangeClamps
 * checks that feh itself, entirely outside every chained WR library's
 * own range, is clamped to that range rather than thrown on (both
 * sub- and super-solar), since spec() clamps feh before dispatching
 * for a WR-classified star specifically -- unlike a normal-grid star
 * with an equally out-of-range feh, which testChainOOBThrows confirms
 * still throws. testNormalLoggSpecForce checks that a
 * star well within a normal library's own log(Teff) range, but with a
 * log(g) far outside its range, is still rescued by
 * SpecsynLibNoWind::specForce(). testChainWithSparseWD checks that a
 * WD_grid entry backed by a partially-filled grid (RAUCH_test)
 * dispatches correctly through the full chain, not just standalone
 * (see testSpecsynLibWD.cpp's own sparse-grid tests for that).
 *
 * It also tests SpecsynLibChained::makeCommonWlGrid directly, both
 * against a fully controlled synthetic scenario (so the window-by-
 * window point-count logic can be checked exactly) and against
 * BOSZ_test/TLUSTY_test's own native grids (to confirm the
 * constructor actually resamples every chained library onto a single
 * common grid, rather than each keeping its own), and covers the
 * constructor's other two wavelength-grid cases directly -- wlMin/
 * wlMax/nWl all supplied (an exact utils::logspace grid, independent
 * of either library's own native range) and nWl supplied alone (an
 * utils::logspace grid spanning the combined native range of every
 * library in the chain).
 * @date 2026-07-21
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#include "../../src/io/SimControls.hpp"
#include "../../src/specsyn/SpecsynLibNoWind.hpp"
#include "../../src/specsyn/SpecsynLibChained.hpp"
#include "../../src/tracks/TrackCommons.hpp"
#include "../../src/utils/Constants.hpp"
#include "../../src/utils/MiscUtils.hpp"
#include "testSpecsynLibChained.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    const std::string registryName = "tests/specsyn/assets/spectra.toml";

    constexpr double solarLuminosity = utils::Lsun;
    constexpr double obLuminosity = 189859.68762747623 * solarLuminosity;

    // Every SpecsynLibNoWind/SpecsynLibChained constructor call in
    // this file now needs an explicit controls argument -- see
    // testSpecsynLib.cpp's own identical comment on testControls for
    // why.
    const io::SimControls testControls;

    /**
     * @brief Build a StarData with the given mass, log(L), and log(Teff)
     */
    auto makeStarData(const double mass, const double logL, const double logTeff)
        -> specsyn::Specsyn::StarData
    {
        specsyn::Specsyn::StarData props{};
        props.at(static_cast<std::size_t>(tracks::FieldIdx::mass)) = mass;
        props.at(static_cast<std::size_t>(tracks::FieldIdx::logL)) = logL;
        props.at(static_cast<std::size_t>(tracks::FieldIdx::logTe)) = logTeff;
        return props;
    }

    /**
     * @brief The solar-parameter star used by testSpecsynLib.cpp's BOSZ_test check
     * @details M = 1 Msun, Teff = 5772 K, L = 1 Lsun, feh = 0.1 --
     *   inside BOSZ_test's grid, outside TLUSTY_test's (Teff far below
     *   its 27500-30000 K range).
     */
    auto solarStar() -> specsyn::Specsyn::StarData
    {
        return makeStarData(1.0, 0.0, std::log10(5772.0));
    }
    constexpr double solarFeh = 0.1;

    /**
     * @brief The OB-star parameters used by testSpecsynLib.cpp's TLUSTY_test check
     * @details M = 15 Msun, Teff = 28750 K, giving log(g) = 3.125 --
     *   inside TLUSTY_test's grid, outside BOSZ_test's (Teff far above
     *   its 5750-6000 K range).
     */
    auto obStar() -> specsyn::Specsyn::StarData
    {
        return makeStarData(15.0, 5.278432762001573, std::log10(28750.0));
    }
    constexpr double obFeh = -1.2;

    /**
     * @brief A star far outside both BOSZ_test's and TLUSTY_test's grids
     * @details Same out-of-bounds Teff (15000 K) used by
     *   testSpecsynLib.cpp's OOB checks -- above BOSZ_test's range,
     *   below TLUSTY_test's.
     */
    auto oobStar() -> specsyn::Specsyn::StarData
    {
        return makeStarData(1.0, 0.0, std::log10(15000.0));
    }
    constexpr double oobFeh = 0.1;

    /**
     * @brief A compact, hot ("white dwarf-like") star
     * @details M = 0.0592346591493545 Msun, Teff = 28750 K (the same
     *   Teff as obStar()), log(L) = -2.0, giving log(g) = 8.0 -- inside
     *   WD_test's grid (logg 7.0-9.0, log(Teff) 4.0-4.6), but far
     *   outside TLUSTY_test's own logg range (3.0-3.25, nowhere near a
     *   compact star's), so a chain listing TLUSTY_test before WD_test
     *   must fall through to WD_test for this star -- TLUSTY_test's
     *   own (feh, logTeff) bounds check passes first (this Teff and
     *   wdFeh both sit inside its range), so it's specifically the
     *   logg mismatch, not an earlier check, that forces the fallthrough.
     */
    auto wdStar() -> specsyn::Specsyn::StarData
    {
        return makeStarData(0.0592346591493545, -2.0, std::log10(28750.0));
    }
    constexpr double wdFeh = 0.0;

    // Expected peak(wl * spec) for wdStar() against WD_test's own
    // amplitude(logg, logTeff) = 1 + 2*logg + 3*logTeff formula (see
    // data/tools/spectra/make_wd_test_fixture.py), evaluated at logg = 8.0,
    // log(Teff) = log10(28750), times the shape/wl values at WD_test's
    // last (largest) grid point, times the surface area implied by
    // wdStar()'s own (mass, logL, logTeff) -- see checkSpectrum's own
    // generous (three-orders-of-magnitude) tolerance for why this
    // doesn't need to be more precise than an order-of-magnitude estimate.
    constexpr double wdLuminosity = 2.4e24;

    /**
     * @brief Check that result is non-empty and that peak(wl * result) is near expectedLuminosity
     * @return 0 on success, 1 (after printing a diagnostic) on failure
     */
    auto checkSpectrum(
        const std::vector<double>& result,
        const std::vector<double>& wl,
        const double expectedLuminosity,
        const std::string& label) -> int
    {
        if (result.empty())
        {
            std::cerr << "testSpecsynLibChained: expected a non-empty spectrum for "
                << label << ", got an empty one\n";
            return 1;
        }
        if (result.size() != wl.size())
        {
            std::cerr << "testSpecsynLibChained: " << label << " spec() returned "
                << result.size() << " values, expected " << wl.size() << "\n";
            return 1;
        }

        double maxWlSpec = 0.0;
        for (std::size_t i = 0; i < result.size(); ++i)
        {
            maxWlSpec = std::max(maxWlSpec, result.at(i) * wl.at(i));
        }
        if (maxWlSpec < 1e-3 * expectedLuminosity || maxWlSpec > 1e3 * expectedLuminosity)
        {
            std::cerr << "testSpecsynLibChained: " << label << " max(wl * spec) = "
                << maxWlSpec << " erg/s is unreasonably far from the expected "
                << expectedLuminosity << " erg/s\n";
            return 1;
        }
        return 0;
    }

    /**
     * @brief Build an evenly-spaced grid [lo, hi] with the given step
     */
    auto linspaceStep(const double lo, const double hi, const double step) -> std::vector<double>
    {
        std::vector<double> grid;
        for (double x = lo; x <= hi + 1e-9; x += step) { grid.push_back(x); }
        return grid;
    }

    /**
     * @brief Build a StarData with WNE-classifying surface composition, plus the given mass/logL/logTeff/mdot
     * @details
     * heSurf/cSurf/nSurf mirror testSpecsynLibWR.cpp's own
     * makeWRStarData defaults, which classify as WRType::WNE via
     * getWRType (heSurf > 0.9, cSurf < nSurf) -- matching
     * POWR_WNE_test's own type.
     */
    auto makeWRStarData(const double mass, const double logL, const double logTeff, const double mdot)
        -> specsyn::Specsyn::StarData
    {
        specsyn::Specsyn::StarData props{};
        props.at(static_cast<std::size_t>(tracks::FieldIdx::mass)) = mass;
        props.at(static_cast<std::size_t>(tracks::FieldIdx::mdot)) = mdot;
        props.at(static_cast<std::size_t>(tracks::FieldIdx::logL)) = logL;
        props.at(static_cast<std::size_t>(tracks::FieldIdx::logTe)) = logTeff;
        props.at(static_cast<std::size_t>(tracks::FieldIdx::heSurf)) = 0.98;
        props.at(static_cast<std::size_t>(tracks::FieldIdx::cSurf)) = 0.0;
        props.at(static_cast<std::size_t>(tracks::FieldIdx::nSurf)) = 0.01;
        props.at(static_cast<std::size_t>(tracks::FieldIdx::hSurf)) = 0.0;
        return props;
    }
} // namespace

// Check that, with TLUSTY_test listed first, an OB star is handled
// immediately by TLUSTY_test, and a solar star -- out of bounds for
// TLUSTY_test -- falls through to BOSZ_test. Since the chain's
// constructor resamples every library onto a common grid, both
// results must match the chain's own wl() -- regardless of which
// underlying library actually produced them.
static auto testChainTlustyFirst() -> int
{
    const specsyn::SpecsynLibChained chain(
        { "TLUSTY_test", "BOSZ_test" }, -3.0, 1.0, 0.0, 0.0,
        {}, specsyn::defaultR, registryName, 0.0, 0.0, 0, true, testControls);

    int result = 0;
    try
    {
        const auto obResult = chain.spec(obStar(), obFeh);
        result += checkSpectrum(obResult, chain.wl(), obLuminosity,
            "an OB star with TLUSTY_test first");
    }
    catch (const std::exception& e)
    {
        std::cerr << "testSpecsynLibChained: unexpected exception for an "
            "in-bounds OB star: " << e.what() << "\n";
        result += 1;
    }

    try
    {
        const auto solarResult = chain.spec(solarStar(), solarFeh);
        result += checkSpectrum(solarResult, chain.wl(), solarLuminosity,
            "a solar star falling through to BOSZ_test");
    }
    catch (const std::exception& e)
    {
        std::cerr << "testSpecsynLibChained: unexpected exception for a "
            "solar star that should fall through to BOSZ_test: "
            << e.what() << "\n";
        result += 1;
    }

    return result;
}

// Check the same two stars with the chain order reversed (BOSZ_test
// first): the solar star is now handled immediately, and the OB star
// -- out of bounds for BOSZ_test -- falls through to TLUSTY_test
static auto testChainBoszFirst() -> int
{
    const specsyn::SpecsynLibChained chain(
        { "BOSZ_test", "TLUSTY_test" }, -3.0, 1.0, 0.0, 0.0,
        {}, specsyn::defaultR, registryName, 0.0, 0.0, 0, true, testControls);

    int result = 0;
    try
    {
        const auto solarResult = chain.spec(solarStar(), solarFeh);
        result += checkSpectrum(solarResult, chain.wl(), solarLuminosity,
            "a solar star with BOSZ_test first");
    }
    catch (const std::exception& e)
    {
        std::cerr << "testSpecsynLibChained: unexpected exception for an "
            "in-bounds solar star: " << e.what() << "\n";
        result += 1;
    }

    try
    {
        const auto obResult = chain.spec(obStar(), obFeh);
        result += checkSpectrum(obResult, chain.wl(), obLuminosity,
            "an OB star falling through to TLUSTY_test");
    }
    catch (const std::exception& e)
    {
        std::cerr << "testSpecsynLibChained: unexpected exception for an "
            "OB star that should fall through to TLUSTY_test: "
            << e.what() << "\n";
        result += 1;
    }

    return result;
}

// Check that a chain including a WD_grid entry (WD_test) dispatches to
// a SpecsynLibWD for it: a normal OB star is still handled by
// TLUSTY_test as usual, and a compact, high-log(g) star -- out of
// bounds for TLUSTY_test's own logg range, but inside WD_test's --
// falls through to WD_test, exactly as an ordinary NoWind-to-NoWind
// fallthrough would (see testChainTlustyFirst/testChainBoszFirst)
static auto testChainWithWD() -> int
{
    const specsyn::SpecsynLibChained chain(
        { "TLUSTY_test", "WD_test" }, -3.0, 1.0, 0.0, 0.0,
        {}, specsyn::defaultR, registryName, 0.0, 0.0, 0, true, testControls);

    int result = 0;
    try
    {
        const auto obResult = chain.spec(obStar(), obFeh);
        result += checkSpectrum(obResult, chain.wl(), obLuminosity,
            "an OB star with TLUSTY_test first, WD_test chained after it");
    }
    catch (const std::exception& e)
    {
        std::cerr << "testSpecsynLibChained: unexpected exception for an "
            "in-bounds OB star (WD chain): " << e.what() << "\n";
        result += 1;
    }

    try
    {
        const auto wdResult = chain.spec(wdStar(), wdFeh);
        result += checkSpectrum(wdResult, chain.wl(), wdLuminosity,
            "a compact star falling through to WD_test");
    }
    catch (const std::exception& e)
    {
        std::cerr << "testSpecsynLibChained: unexpected exception for a "
            "compact star that should fall through to WD_test: "
            << e.what() << "\n";
        result += 1;
    }

    return result;
}

// Check that a chain including a WD_grid entry backed by a partially-
// filled grid (RAUCH_test, see data/tools/spectra/make_rauch_test_fixture.py)
// dispatches to it correctly through the full SpecsynLibChained
// machinery -- not just standalone via SpecsynLibWD directly (see
// testSpecsynLibWD.cpp's own testSparseGridExactPoint/
// testSparseGridMissingCorner) -- for a star landing on one of its
// populated grid points, exactly as testChainWithWD does for the
// filled-tensor WD_test fixture. tClamp = true here (unlike before this
// class's refactor to per-type chains, where it had to be left false to
// avoid a pre-dispatch clamp defeating the point of landing exactly on
// RAUCH_test's own grid point -- no such clamp exists any more, and
// GridType::wdGrid classification, which is what actually routes this
// star to RAUCH_test's own chain at all, requires tClamp = true).
static auto testChainWithSparseWD() -> int
{
    const specsyn::SpecsynLibChained chain(
        { "TLUSTY_test", "RAUCH_test" }, -3.0, 1.0, 0.0, 0.0,
        {}, specsyn::defaultR, registryName, 0.0, 0.0, 0, true, testControls);

    // (logg, log_Teff) = (8.0, 4.5): a populated corner of RAUCH_test's
    // own grid (see data/tools/spectra/make_rauch_test_fixture.py),
    // where flux = 4.0, chosen (rather than its own axis-minimum
    // corner (7.0, 4.0)) specifically so that classifyGridType's own
    // strict-inequality floor check (log(g)/log(Teff) > loggMin[wd]/
    // logTeffMin[wd], not >=) still classifies this star
    // GridType::wdGrid: RAUCH_test's own (7.0, 4.0) sits exactly on
    // that floor in both dimensions at once, which this strict
    // inequality would otherwise exclude. mass is chosen (offline) so
    // that (logL, logTeff) below give exactly this log(g); log(Teff)
    // = 4.5 (~31600 K) is also above TLUSTY_test's own 27500-30000 K
    // range, and log(g) = 8.0 far above TLUSTY_test's own 3.0-3.25
    // ceiling, so this star classifies GridType::wdGrid (aboveNormal
    // via both axes) and falls straight through to RAUCH_test.
    constexpr double logL = -2.0;
    constexpr double logTeff = 4.5;
    constexpr double mass = 0.04046848123819107;
    constexpr double area = 6.750715544104499e17; // surface area implied by (logL, logTeff) above
    constexpr double rauchFluxAtCorner = 4.0;
    constexpr double rauchWlMax = 10000.0; // RAUCH_test's own largest wavelength point

    int result = 0;
    try
    {
        const auto spec = chain.spec(makeStarData(mass, logL, logTeff), 0.0);
        result += checkSpectrum(spec, chain.wl(), area * rauchFluxAtCorner * rauchWlMax,
            "a star landing on RAUCH_test's own populated grid point via the chain");
    }
    catch (const std::exception& e)
    {
        std::cerr << "testSpecsynLibChained: unexpected exception for a star "
            "that should fall through to RAUCH_test: " << e.what() << "\n";
        result += 1;
    }
    return result;
}

// Check that a WR-classified star far outside POWR_WNE_test's own
// log(Teff) range (log(Teff) = 4.6-4.8, i.e. ~40000-63000 K) is still
// rescued to a real spectrum, regardless of tClamp -- classifyGridType
// routes it to GridType::wrGrid purely from its surface composition
// (per SpecsynLibWR::getWRType), independent of tClamp entirely
// (unlike GridType::wdGrid -- see testClassifyWD), so it is always
// tried against wrLibs_ = [POWR_WNE_test] alone, regardless of
// TLUSTY_test's/WD_test's own (much different) ranges also present in
// this chain. SpecsynLibWR::specForce() searches for the nearest
// populated (log(R_t), log(Teff)) grid point regardless of how far
// out of range log(Teff) itself starts (see its own comment for why),
// so this now succeeds identically whether tClamp is true or false,
// confirming tClamp remains irrelevant to WR dispatch even after that
// rescue was added.
static auto testClassifyWRRescue() -> int
{
    // log(Teff) = 6.0 (~1e6 K) is far past every grid in this chain,
    // including POWR_WNE_test's own -- specForce() must fall back on
    // its nearest-populated-point search rather than any ordinary
    // interpolation
    const auto props = makeWRStarData(20.0, 5.7, 6.0, 3e-5);
    constexpr double wrFeh = -0.5; // inside POWR_WNE_test's own Fe_H = [-1.0, 0.0]
    constexpr double wrLuminosity = 501187.23362727246 * solarLuminosity; // 10^5.7 Lsun

    int result = 0;
    for (const bool tClamp : { true, false })
    {
        const specsyn::SpecsynLibChained chain(
            { "POWR_WNE_test", "TLUSTY_test", "WD_test" }, -3.0, 1.0, 0.0, 0.0,
            {}, specsyn::defaultR, registryName, 0.0, 0.0, 0, tClamp, testControls);
        try
        {
            const auto spec = chain.spec(props, wrFeh);
            result += checkSpectrum(spec, chain.wl(), wrLuminosity,
                "a WR star far outside POWR_WNE_test's own log(Teff) range, "
                "rescued via specForce()");
        }
        catch (const std::exception& e)
        {
            std::cerr << "testSpecsynLibChained: unexpected exception for a WR "
                "star that specForce() should have rescued (tClamp = " <<
                tClamp << "): " << e.what() << "\n";
            result += 1;
        }
    }
    return result;
}

// Check that a WR-classified star whose feh is entirely outside every
// wrLibs_ library's own Fe_H range is clamped to that range (both
// sub- and super-solar), rather than throwing -- SpecsynLibChained::
// spec() now clamps feh to fehMin_[wrGrid]/fehMax_[wrGrid] before
// calling into the chain (see its own comment), specifically because
// no super-solar (or very sub-solar) WR atmosphere grids currently
// exist, so a WR star at either extreme should still get the nearest
// available WR spectrum rather than fail outright. Only WR stars are
// clamped this way -- testChainOOBThrows checks a normal-grid star
// with an equally out-of-range feh still throws, unchanged.
static auto testClassifyWRFehOutOfRangeClamps() -> int
{
    const auto props = makeWRStarData(20.0, 5.7, 4.7, 3e-5);
    constexpr double wrLuminosity = 501187.23362727246 * solarLuminosity; // 10^5.7 Lsun

    const specsyn::SpecsynLibChained chain(
        { "POWR_WNE_test", "TLUSTY_test", "WD_test" }, -3.0, 1.0, 0.0, 0.0,
        {}, specsyn::defaultR, registryName, 0.0, 0.0, 0, true, testControls);

    int result = 0;
    // feh = -5.0 and feh = 5.0 are both far outside POWR_WNE_test's
    // own [-1.0, 0.0] range, on opposite sides of it
    for (const double outOfRangeFeh : { -5.0, 5.0 })
    {
        try
        {
            const auto spec = chain.spec(props, outOfRangeFeh);
            result += checkSpectrum(spec, chain.wl(), wrLuminosity,
                "a WR star with feh = " + std::to_string(outOfRangeFeh) +
                ", entirely outside POWR_WNE_test's own range, clamped rather than rescued");
        }
        catch (const std::exception& e)
        {
            std::cerr << "testSpecsynLibChained: unexpected exception for a WR star "
                "with feh = " << outOfRangeFeh << " (should have been clamped, not "
                "thrown): " << e.what() << "\n";
            result += 1;
        }
    }
    return result;
}

// Check that a star classified GridType::wdGrid by classifyGridType
// (not WR; log(g) above TLUSTY_test's own 3.25 ceiling; both log(g)
// and log(Teff) above WD_test's own floor) -- even one exceeding
// WD_test's own log(g) ceiling too (log(g) = 12, versus WD_test's own
// 9.0 maximum) -- is rescued by tClamp = true (clamped into
// WD_test's own range, not TLUSTY_test's), and genuinely fails
// without it.
static auto testClassifyWD() -> int
{
    constexpr double logL = -2.0;
    constexpr double logTeff = 4.3; // within WD_test's own 4.0-4.6 range
    // mass chosen (offline) to give log(g) = 12.0 at (logL, logTeff) above
    constexpr double mass = 2553.9353133421105;
    const auto props = makeStarData(mass, logL, logTeff);

    int result = 0;
    {
        const specsyn::SpecsynLibChained chain(
            { "TLUSTY_test", "WD_test" }, -3.0, 1.0, 0.0, 0.0,
            {}, specsyn::defaultR, registryName, 0.0, 0.0, 0, true, testControls);
        try
        {
            const auto spec = chain.spec(props, wdFeh);
            if (spec.empty())
            {
                std::cerr << "testSpecsynLibChained: expected a non-empty spectrum "
                    "for a WD-classified star clamped to WD_test's own range, got "
                    "an empty one\n";
                result += 1;
            }
        }
        catch (const std::exception& e)
        {
            std::cerr << "testSpecsynLibChained: unexpected exception for a "
                "WD-classified star that tClamp should have rescued: "
                << e.what() << "\n";
            result += 1;
        }
    }
    {
        const specsyn::SpecsynLibChained chain(
            { "TLUSTY_test", "WD_test" }, -3.0, 1.0, 0.0, 0.0,
            {}, specsyn::defaultR, registryName, 0.0, 0.0, 0, false, testControls);
        try
        {
            [[maybe_unused]] const auto spec = chain.spec(props, wdFeh);
            std::cerr << "testSpecsynLibChained: expected a star with log(g) = 12 "
                "(beyond even WD_test's own ceiling) to throw with tClamp = false, "
                "but it did not\n";
            result += 1;
        }
        catch (const std::runtime_error&) { /* expected */ }
    }
    return result;
}

// Check that a star with a log(Teff) safely within TLUSTY_test's own
// range, but a log(g) enormously outside its range (or any real
// star's), is still rescued via SpecsynLibNoWind::specForce() moving
// log(g) to TLUSTY_test's own nearest populated value -- the
// GridType::normalGrid counterpart to testClassifyWD's own
// GridType::wdGrid case. Only TLUSTY_test is chained here (no WD_test
// at all), so classifyGridType's withinWDFloor check is unconditionally
// false (logTeffMin[wd]/loggMin[wd] both stay quiet_NaN() with no
// GridType::wdGrid library present at all -- see classifyGridType's
// own comment), making this star's own classification
// GridType::normalGrid regardless of tClamp, and so exercising
// specForce()'s own rescue in isolation from classification concerns.
static auto testNormalLoggSpecForce() -> int
{
    constexpr double logL = 0.0;
    constexpr double logTeff = 4.46; // strictly between TLUSTY_test's own 27500/30000 K (4.4393/4.4771) grid points
    // mass chosen (offline) to give log(g) = 20.0 at (logL, logTeff)
    // above -- far past TLUSTY_test's own 3.0-3.25 ceiling
    constexpr double mass = 5849475224336.677;
    const auto props = makeStarData(mass, logL, logTeff);

    const specsyn::SpecsynLibChained chain(
        { "TLUSTY_test" }, -3.0, 1.0, 0.0, 0.0,
        {}, specsyn::defaultR, registryName, 0.0, 0.0, 0, true, testControls);
    try
    {
        const auto spec = chain.spec(props, wdFeh);
        if (spec.empty())
        {
            std::cerr << "testSpecsynLibChained: expected a non-empty spectrum "
                "for a star with log(g) forced to TLUSTY_test's own nearest "
                "populated value, got an empty one\n";
            return 1;
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "testSpecsynLibChained: unexpected exception for a star "
            "specForce() should have rescued via log(g): " << e.what() << "\n";
        return 1;
    }
    return 0;
}

// Check that a star outside every chained library's grid throws,
// since the last library in the chain is always constructed with
// OOBPolicy::raise regardless of chain order
static auto testChainOOBThrows() -> int
{
    const specsyn::SpecsynLibChained chain(
        { "TLUSTY_test", "BOSZ_test" }, -3.0, 1.0, 0.0, 0.0,
        {}, specsyn::defaultR, registryName, 0.0, 0.0, 0, true, testControls);

    try
    {
        [[maybe_unused]] const auto result = chain.spec(oobStar(), oobFeh);
        std::cerr << "testSpecsynLibChained: expected spec() to throw for a "
            "star outside every chained library's grid, but it did not\n";
        return 1;
    }
    catch (const std::runtime_error&) { /* this is the expected outcome */ }

    return 0;
}

// Check that the constructor rejects an empty spectraName list and a
// microTurb vector whose size does not match spectraName's
static auto testChainConstructorValidation() -> int
{
    int result = 0;

    try
    {
        [[maybe_unused]] const specsyn::SpecsynLibChained chain(
            {}, -3.0, 1.0, 0.0, 0.0, {}, specsyn::defaultR, registryName, 0.0, 0.0, 0, true, testControls);
        std::cerr << "testSpecsynLibChained: expected the constructor to throw "
            "for an empty spectraName, but it did not\n";
        result += 1;
    }
    catch (const std::runtime_error&) { /* expected */ }

    try
    {
        [[maybe_unused]] const specsyn::SpecsynLibChained chain(
            { "TLUSTY_test", "BOSZ_test" }, -3.0, 1.0, 0.0, 0.0,
            { 10.0 }, specsyn::defaultR, registryName, 0.0, 0.0, 0, true, testControls);
        std::cerr << "testSpecsynLibChained: expected the constructor to throw "
            "for a mismatched microTurb size, but it did not\n";
        result += 1;
    }
    catch (const std::runtime_error&) { /* expected */ }

    return result;
}

// Check that SpecsynLibChained's constructor actually builds and uses
// a single common grid spanning both BOSZ_test's and TLUSTY_test's own
// native ranges, rather than just adopting one library's grid
// verbatim: the chain's wl() must span the full union of both native
// ranges, be strictly increasing, and have exactly as many points as
// calling makeCommonWlGrid directly on the two native grids.
static auto testChainUsesCommonGrid() -> int
{
    const specsyn::SpecsynLibNoWind<specsyn::OOBPolicy::raise> boszRef(
        "BOSZ_test", -3.0, 1.0, 0.0, 0.0, 0.0, specsyn::defaultR, registryName, 0.0, 0.0, 0, testControls);
    const specsyn::SpecsynLibNoWind<specsyn::OOBPolicy::raise> tlustyRef(
        "TLUSTY_test", -3.0, 1.0, 0.0, 0.0, 10.0, specsyn::defaultR, registryName, 0.0, 0.0, 0, testControls);

    const auto expected = specsyn::SpecsynLibChained::makeCommonWlGrid(
        { boszRef.wl(), tlustyRef.wl() });

    const specsyn::SpecsynLibChained chain(
        { "BOSZ_test", "TLUSTY_test" }, -3.0, 1.0, 0.0, 0.0,
        {}, specsyn::defaultR, registryName, 0.0, 0.0, 0, true, testControls);

    if (chain.wl() != expected)
    {
        std::cerr << "testSpecsynLibChained: chain.wl() (" << chain.wl().size()
            << " points) does not match makeCommonWlGrid's own result ("
            << expected.size() << " points) for the same two native grids\n";
        return 1;
    }

    const double expectedMin = std::min(boszRef.wl().front(), tlustyRef.wl().front());
    const double expectedMax = std::max(boszRef.wl().back(), tlustyRef.wl().back());
    if (chain.wl().front() != expectedMin || chain.wl().back() != expectedMax)
    {
        std::cerr << "testSpecsynLibChained: chain.wl() does not span the full "
            "union [" << expectedMin << ", " << expectedMax << "] of both "
            "libraries' native ranges: got [" << chain.wl().front() << ", "
            << chain.wl().back() << "]\n";
        return 1;
    }

    if (!std::ranges::is_sorted(chain.wl(), std::less<>{}))
    {
        std::cerr << "testSpecsynLibChained: chain.wl() is not strictly "
            "increasing\n";
        return 1;
    }

    return 0;
}

// Check that supplying wlMin/wlMax/nWl to the constructor (case 1 of
// its three-way common-grid selection) uses that grid directly --
// exactly utils::logspace(wlMin, wlMax, nWl) -- rather than deriving
// anything from the individual libraries' own native grids, and that
// every chained library still produces spectra on it correctly
static auto testChainWlMinMaxSpecified() -> int
{
    constexpr double wlMin = 1000.0;
    constexpr double wlMax = 20000.0;
    constexpr std::size_t nWl = 50;

    const specsyn::SpecsynLibChained chain(
        { "BOSZ_test", "TLUSTY_test" }, -3.0, 1.0, 0.0, 0.0,
        {}, specsyn::defaultR, registryName, wlMin, wlMax, nWl, true, testControls);

    const auto expected = utils::logspace(wlMin, wlMax, nWl);
    if (chain.wl() != expected)
    {
        std::cerr << "testSpecsynLibChained: chain.wl() (" << chain.wl().size()
            << " points, [" << chain.wl().front() << ", " << chain.wl().back()
            << "]) does not match utils::logspace(wlMin, wlMax, nWl) ("
            << expected.size() << " points, [" << expected.front() << ", "
            << expected.back() << "]) when wlMin/wlMax/nWl are all supplied\n";
        return 1;
    }

    int result = 0;
    try
    {
        const auto solarResult = chain.spec(solarStar(), solarFeh);
        result += checkSpectrum(solarResult, chain.wl(), solarLuminosity,
            "a solar star with an explicit wlMin/wlMax/nWl grid");
    }
    catch (const std::exception& e)
    {
        std::cerr << "testSpecsynLibChained: unexpected exception for a "
            "solar star with an explicit wlMin/wlMax/nWl grid: " << e.what() << "\n";
        result += 1;
    }
    try
    {
        const auto obResult = chain.spec(obStar(), obFeh);
        result += checkSpectrum(obResult, chain.wl(), obLuminosity,
            "an OB star with an explicit wlMin/wlMax/nWl grid");
    }
    catch (const std::exception& e)
    {
        std::cerr << "testSpecsynLibChained: unexpected exception for an "
            "OB star with an explicit wlMin/wlMax/nWl grid: " << e.what() << "\n";
        result += 1;
    }

    return result;
}

// Check that supplying nWl alone (wlMin = wlMax = 0, the "not
// supplied" sentinel; case 2 of the constructor's three-way common-
// grid selection) spans the combined native range of every library in
// the chain -- not just the first, or just whichever library's native
// range happens to be requested -- at exactly nWl points
static auto testChainNWlOnly() -> int
{
    const specsyn::SpecsynLibNoWind<specsyn::OOBPolicy::raise> boszRef(
        "BOSZ_test", -3.0, 1.0, 0.0, 0.0, 0.0, specsyn::defaultR, registryName, 0.0, 0.0, 0, testControls);
    const specsyn::SpecsynLibNoWind<specsyn::OOBPolicy::raise> tlustyRef(
        "TLUSTY_test", -3.0, 1.0, 0.0, 0.0, 10.0, specsyn::defaultR, registryName, 0.0, 0.0, 0, testControls);
    const double globalWlMin = std::min(boszRef.wl().front(), tlustyRef.wl().front());
    const double globalWlMax = std::max(boszRef.wl().back(), tlustyRef.wl().back());

    constexpr std::size_t nWl = 40;
    const specsyn::SpecsynLibChained chain(
        { "BOSZ_test", "TLUSTY_test" }, -3.0, 1.0, 0.0, 0.0,
        {}, specsyn::defaultR, registryName, 0.0, 0.0, nWl, true, testControls);

    const auto expected = utils::logspace(globalWlMin, globalWlMax, nWl);
    if (chain.wl() != expected)
    {
        std::cerr << "testSpecsynLibChained: chain.wl() (" << chain.wl().size()
            << " points, [" << chain.wl().front() << ", " << chain.wl().back()
            << "]) does not match utils::logspace over the combined native "
            "range [" << globalWlMin << ", " << globalWlMax << "] ("
            << expected.size() << " points) when only nWl is supplied\n";
        return 1;
    }

    int result = 0;
    try
    {
        const auto solarResult = chain.spec(solarStar(), solarFeh);
        result += checkSpectrum(solarResult, chain.wl(), solarLuminosity,
            "a solar star with nWl alone");
    }
    catch (const std::exception& e)
    {
        std::cerr << "testSpecsynLibChained: unexpected exception for a "
            "solar star with nWl alone: " << e.what() << "\n";
        result += 1;
    }

    return result;
}

// Check makeCommonWlGrid's window-selection logic against a fully
// controlled synthetic scenario: a coarse grid spanning [200, 2000]
// Angstrom every 10 Angstrom (181 points) and a fine grid spanning
// [1500, 3000] Angstrom every 5 Angstrom (301 points). This splits
// wavelength space into three windows -- [200, 1500), [1500, 2000),
// and [2000, 3000] -- and the fine grid should win the overlap window
// [1500, 2000) (100 of its points there vs. the coarse grid's 50), so
// the merged grid should equal: the coarse grid's points below 1500,
// followed by the fine grid's points from 1500 to 3000.
static auto testMakeCommonWlGridWindows() -> int
{
    const auto coarse = linspaceStep(200.0, 2000.0, 10.0);
    const auto fine = linspaceStep(1500.0, 3000.0, 5.0);

    const auto merged = specsyn::SpecsynLibChained::makeCommonWlGrid({ coarse, fine });

    std::vector<double> expected;
    for (const double x : coarse) { if (x < 1500.0) { expected.push_back(x); } }
    for (const double x : fine) { expected.push_back(x); }

    if (merged != expected)
    {
        std::cerr << "testSpecsynLibChained: makeCommonWlGrid produced "
            << merged.size() << " points, expected " << expected.size()
            << " for the coarse/fine overlap scenario\n";
        return 1;
    }

    return 0;
}

// Check that makeCommonWlGrid leaves a genuine coverage gap between
// two non-overlapping, non-touching grids empty, rather than
// inventing samples there
static auto testMakeCommonWlGridGap() -> int
{
    const std::vector<double> low = { 100.0, 150.0, 200.0 };
    const std::vector<double> high = { 500.0, 550.0, 600.0 };

    const auto merged = specsyn::SpecsynLibChained::makeCommonWlGrid({ low, high });

    std::vector<double> expected = low;
    expected.insert(expected.end(), high.begin(), high.end());

    if (merged != expected)
    {
        std::cerr << "testSpecsynLibChained: makeCommonWlGrid did not skip the "
            "coverage gap between two disjoint grids as expected\n";
        return 1;
    }

    return 0;
}

// Check that makeCommonWlGrid rejects an empty list of grids and a
// list containing an empty grid
static auto testMakeCommonWlGridErrors() -> int
{
    int result = 0;

    try
    {
        [[maybe_unused]] const auto merged =
            specsyn::SpecsynLibChained::makeCommonWlGrid({});
        std::cerr << "testSpecsynLibChained: expected makeCommonWlGrid to throw "
            "for an empty list of grids, but it did not\n";
        result += 1;
    }
    catch (const std::runtime_error&) { /* expected */ }

    try
    {
        [[maybe_unused]] const auto merged = specsyn::SpecsynLibChained::makeCommonWlGrid(
            { { 100.0, 200.0 }, {} });
        std::cerr << "testSpecsynLibChained: expected makeCommonWlGrid to throw "
            "for a list containing an empty grid, but it did not\n";
        result += 1;
    }
    catch (const std::runtime_error&) { /* expected */ }

    return result;
}

auto testSpecsynLibChained() -> int
{
    int result = 0;
    result += testChainTlustyFirst();
    result += testChainBoszFirst();
    result += testChainWithWD();
    result += testChainWithSparseWD();
    result += testClassifyWRRescue();
    result += testClassifyWRFehOutOfRangeClamps();
    result += testClassifyWD();
    result += testNormalLoggSpecForce();
    result += testChainOOBThrows();
    result += testChainConstructorValidation();
    result += testChainUsesCommonGrid();
    result += testChainWlMinMaxSpecified();
    result += testChainNWlOnly();
    result += testMakeCommonWlGridWindows();
    result += testMakeCommonWlGridGap();
    result += testMakeCommonWlGridErrors();
    return result;
}
