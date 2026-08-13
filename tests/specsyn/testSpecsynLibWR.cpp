/**
 * @file testSpecsynLibWR.cpp
 * @author Mark Krumholz
 * @brief Unit tests for the SpecsynLibWR class.
 * @details
 * This file contains end-to-end unit tests for SpecsynLibWR::spec,
 * against the small POWR_test.h5/spectra.toml fixture stored under
 * tests/specsyn/assets. That fixture is entirely synthetic (not real
 * PoWR data, which is far too large to store in the repository): it
 * has two [Fe/H] groups (-1.0 and 0.0), each with a (log_teff, log_rt)
 * grid of exactly {4.6, 4.8} x {0.5, 1.0}, every one of whose 4 grid
 * points holds the same Gaussian SED (centered at 5000 Angstrom) and
 * the same logl = 5.5 and dinf = 1.0 -- see
 * data/tools/spectra/make_powr_test_fixture.py for exactly how it was built.
 * Sharing one SED and logl across every point removes any dependence
 * on the interpolation weights themselves from the expected result,
 * isolating what these tests actually care about: that spec() derives
 * a (FeH, logRt, logTeff) point that lands inside this small grid for
 * a plausible WNE star, classifies WRType and out-of-range feh/logTeff
 * as out of bounds, clamps an out-of-range logRt into the grid rather
 * than rejecting it, and rescales the result to the star's own
 * luminosity rather than the grid point's. The three POWR_WNL_H20/H40/
 * H60_test.h5 fixtures share this same grid shape but each has its own
 * distinct SED center (see make_powr_test_fixture.py), letting
 * testSpecWNLSuccess confirm a WNL star's own hSurf-derived H-bucket
 * (see getWRType) actually selects that bucket's own file.
 * @date 2026-07-23
 */

#include "../../src/io/SimControls.hpp"
#include "../../src/specsyn/SpecsynLibWR.hpp"
#include "../../src/tracks/TrackCommons.hpp"
#include "../../src/utils/Constants.hpp"
#include "testSpecsynLibWR.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{
    const std::string registryName = "tests/specsyn/assets/spectra.toml";
    const std::string spectraName = "POWR_WNE_test";
    const std::string wnlH20SpectraName = "POWR_WNL_H20_test";
    const std::string wnlH40SpectraName = "POWR_WNL_H40_test";
    const std::string wnlH60SpectraName = "POWR_WNL_H60_test";
    constexpr double solarLuminosity = utils::Lsun;

    // Every SpecsynLibWR constructor call in this file now needs an
    // explicit controls argument -- see testSpecsynLib.cpp's own
    // identical comment on testControls for why.
    const io::SimControls testControls;

    /**
     * @brief Build a StarData for a Wolf-Rayet star
     * @details
     * heSurf, cSurf, and nSurf default to values that classify as
     * WRType::WNE (heSurf > 0.9, cSurf < nSurf) via getWRType,
     * matching POWR_WNE_test's own type_. hSurf defaults to 0.0
     * (WRType::WNLH20's own range), which only matters for a star
     * whose heSurf/logTeff otherwise puts it in one of the WNL
     * subtypes.
     */
    auto makeWRStarData(
        const double mass,
        const double logL,
        const double logTeff,
        const double mdot,
        const double heSurf = 0.98,
        const double cSurf = 0.0,
        const double nSurf = 0.01,
        const double hSurf = 0.0) -> specsyn::Specsyn::StarData
    {
        specsyn::Specsyn::StarData props{};
        props.at(static_cast<std::size_t>(tracks::FieldIdx::mass)) = mass;
        props.at(static_cast<std::size_t>(tracks::FieldIdx::mdot)) = mdot;
        props.at(static_cast<std::size_t>(tracks::FieldIdx::logL)) = logL;
        props.at(static_cast<std::size_t>(tracks::FieldIdx::logTe)) = logTeff;
        props.at(static_cast<std::size_t>(tracks::FieldIdx::heSurf)) = heSurf;
        props.at(static_cast<std::size_t>(tracks::FieldIdx::cSurf)) = cSurf;
        props.at(static_cast<std::size_t>(tracks::FieldIdx::nSurf)) = nSurf;
        props.at(static_cast<std::size_t>(tracks::FieldIdx::hSurf)) = hSurf;
        return props;
    }
} // namespace

// Check that spec() successfully interpolates a spectrum for a
// plausible WNE star (M = 20 Msun, L = 10^5.7 Lsun, Teff = 10^4.7 K,
// Mdot = 3e-5 Msun/yr) that works out (see
// data/tools/spectra/make_powr_test_fixture.py's own derivation notes) to
// logRt ~= 0.74, comfortably inside POWR_WNE_test's [0.5, 1.0] log_rt
// range; logTeff = 4.7 and feh = -0.5 similarly fall strictly between
// this fixture's grid points, so this exercises genuine (non-
// degenerate) trilinear interpolation along all three axes at once.
static auto testSpecWNESuccess() -> int
{
    const specsyn::SpecsynLibWR<specsyn::OOBPolicy::raise> lib(
        spectraName, -3.0, 1.0, registryName, 0.0, 0.0, 0, testControls);

    const auto props = makeWRStarData(20.0, 5.7, 4.7, 3e-5);

    std::vector<double> result;
    try
    {
        result = lib.spec(props, -0.5);
    }
    catch (const std::exception& e)
    {
        std::cerr << "testSpecsynLibWR: unexpected exception from spec() "
            "for an in-bounds WNE star: " << e.what() << "\n";
        return 1;
    }

    if (result.size() != lib.wl().size())
    {
        std::cerr << "testSpecsynLibWR: spec() returned " << result.size()
            << " values, expected " << lib.wl().size() << "\n";
        return 1;
    }

    // As with SpecsynLibNoWind's own tests, there is no independently-
    // computed expected spectrum to check against, so this is a sanity
    // check only: the peak of wl * spec(wl) should be within a few
    // orders of magnitude of the star's own luminosity (10^5.7 Lsun) --
    // not the fixture's own logl = 5.5, which spec() must rescale away.
    constexpr double starLuminosity = 501187.23362727246 * solarLuminosity; // 10^5.7 Lsun
    double maxWlSpec = 0.0;
    for (std::size_t i = 0; i < result.size(); ++i)
    {
        maxWlSpec = std::max(maxWlSpec, result.at(i) * lib.wl().at(i));
    }
    if (maxWlSpec < 1e-3 * starLuminosity || maxWlSpec > 1e3 * starLuminosity)
    {
        std::cerr << "testSpecsynLibWR: max(wl * spec) = " << maxWlSpec
            << " erg/s is unreasonably far from the test star's own "
            "luminosity = " << starLuminosity << " erg/s\n";
        return 1;
    }

    return 0;
}

// Check that spec() treats a WRType mismatch as out of bounds: a star
// with heSurf = 0.2 (below the 0.4 threshold in getWRType) and
// logTeff = 4.0 (below getWRType's log10(50000) None/WNL boundary, so
// it isn't instead pulled into WRType::WNL for being hot enough to
// look WC/WO-like -- see getWRType's own comment) classifies as
// WRType::None, which can never match POWR_WNE_test's own
// WRType::WNE, regardless of any other property
static auto testSpecTypeMismatchThrow() -> int
{
    const specsyn::SpecsynLibWR<specsyn::OOBPolicy::raise> lib(
        spectraName, -3.0, 1.0, registryName, 0.0, 0.0, 0, testControls);

    const auto props = makeWRStarData(20.0, 5.7, 4.0, 3e-5, 0.2);

    try
    {
        [[maybe_unused]] const auto result = lib.spec(props, -0.5);
        std::cerr << "testSpecsynLibWR: expected spec() to throw for a "
            "WRType-mismatched star under OOBPolicy::raise, but it did not\n";
        return 1;
    }
    catch (const std::runtime_error&) { /* this is the expected outcome */ }

    return 0;
}

// Check that spec() silently returns an empty spectrum for the same
// WRType-mismatched star when instantiated with OOBPolicy::silent instead
static auto testSpecTypeMismatchSilent() -> int
{
    const specsyn::SpecsynLibWR<specsyn::OOBPolicy::silent> lib(
        spectraName, -3.0, 1.0, registryName, 0.0, 0.0, 0, testControls);

    const auto props = makeWRStarData(20.0, 5.7, 4.0, 3e-5, 0.2);

    std::vector<double> result;
    try
    {
        result = lib.spec(props, -0.5);
    }
    catch (const std::exception& e)
    {
        std::cerr << "testSpecsynLibWR: unexpected exception from spec() "
            "under OOBPolicy::silent: " << e.what() << "\n";
        return 1;
    }

    if (!result.empty())
    {
        std::cerr << "testSpecsynLibWR: expected spec() to return an empty "
            "spectrum for a WRType-mismatched star under OOBPolicy::silent, "
            "got " << result.size() << " values\n";
        return 1;
    }

    return 0;
}

// Check that spec() clamps a derived logRt outside the grid to the
// grid's own edge, rather than treating it as out of bounds. M = 20
// Msun, L = 10^5.0 Lsun, Teff = 10^4.7 K, Mdot = 1e-4 Msun/yr works out
// to logRt ~= -0.77 (see data/tools/spectra/make_powr_test_fixture.py's own
// derivation notes), well outside POWR_WNE_test's [0.5, 1.0] log_rt
// range -- logRt is a single-scattering (mdot * vWind = L / c)
// estimate that a handful of real high-mdot/L Wolf-Rayet points push
// out of range this way (see SpecsynLibWR::spec()'s own comment on
// this), so this case is expected to occur for genuine stars and
// should still produce a spectrum rather than an out-of-bounds result.
static auto testSpecLogRtClamped() -> int
{
    const specsyn::SpecsynLibWR<specsyn::OOBPolicy::raise> lib(
        spectraName, -3.0, 1.0, registryName, 0.0, 0.0, 0, testControls);

    const auto props = makeWRStarData(20.0, 5.0, 4.7, 1e-4);

    std::vector<double> result;
    try
    {
        result = lib.spec(props, -0.5);
    }
    catch (const std::exception& e)
    {
        std::cerr << "testSpecsynLibWR: unexpected exception from spec() "
            "for a star whose logRt should be clamped into range: " <<
            e.what() << "\n";
        return 1;
    }

    if (result.size() != lib.wl().size())
    {
        std::cerr << "testSpecsynLibWR: spec() returned " << result.size()
            << " values, expected " << lib.wl().size() << "\n";
        return 1;
    }

    constexpr double starLuminosity = 1e5 * solarLuminosity; // 10^5.0 Lsun
    double maxWlSpec = 0.0;
    for (std::size_t i = 0; i < result.size(); ++i)
    {
        maxWlSpec = std::max(maxWlSpec, result.at(i) * lib.wl().at(i));
    }
    if (maxWlSpec < 1e-3 * starLuminosity || maxWlSpec > 1e3 * starLuminosity)
    {
        std::cerr << "testSpecsynLibWR: max(wl * spec) = " << maxWlSpec
            << " erg/s is unreasonably far from the test star's own "
            "luminosity = " << starLuminosity << " erg/s\n";
        return 1;
    }

    return 0;
}

// Check that spec() still treats a derived logTeff outside the grid as
// out of bounds -- unlike logRt, logTeff is not clamped, since the
// crude-wind-velocity issue that motivates clamping logRt does not
// apply to it. Teff = 10^6.0 K falls well outside POWR_WNE_test's
// {4.6, 4.8} log_teff grid.
static auto testSpecTeffGridBoundsThrow() -> int
{
    const specsyn::SpecsynLibWR<specsyn::OOBPolicy::raise> lib(
        spectraName, -3.0, 1.0, registryName, 0.0, 0.0, 0, testControls);

    const auto props = makeWRStarData(20.0, 5.0, 6.0, 1e-4);

    try
    {
        [[maybe_unused]] const auto result = lib.spec(props, -0.5);
        std::cerr << "testSpecsynLibWR: expected spec() to throw for a star "
            "whose logTeff falls outside the grid under OOBPolicy::raise, "
            "but it did not\n";
        return 1;
    }
    catch (const std::runtime_error&) { /* this is the expected outcome */ }

    return 0;
}

// Check that spec() silently returns an empty spectrum for the same
// out-of-grid-range star when instantiated with OOBPolicy::silent instead
static auto testSpecTeffGridBoundsSilent() -> int
{
    const specsyn::SpecsynLibWR<specsyn::OOBPolicy::silent> lib(
        spectraName, -3.0, 1.0, registryName, 0.0, 0.0, 0, testControls);

    const auto props = makeWRStarData(20.0, 5.0, 6.0, 1e-4);

    std::vector<double> result;
    try
    {
        result = lib.spec(props, -0.5);
    }
    catch (const std::exception& e)
    {
        std::cerr << "testSpecsynLibWR: unexpected exception from spec() "
            "under OOBPolicy::silent: " << e.what() << "\n";
        return 1;
    }

    if (!result.empty())
    {
        std::cerr << "testSpecsynLibWR: expected spec() to return an empty "
            "spectrum for an out-of-grid-range star under OOBPolicy::silent, "
            "got " << result.size() << " values\n";
        return 1;
    }

    return 0;
}

// Check that spec() successfully interpolates a spectrum for a
// plausible WNL star of a given H-bucket (same mass/L/Teff/Mdot as
// testSpecWNESuccess; heSurf = 0.7 puts it in a WNL subtype, and
// hSurf -- via getWRType's own H20/H40/H60 subdivision -- selects
// which one), loading from that bucket's own POWR_WNL_H*_test
// fixture. Beyond the same in-range/luminosity sanity checks as
// testSpecWNESuccess, this also checks that the peak of wl * spec(wl)
// actually lands near that bucket's own fixture SED center (each of
// the three POWR_WNL_H*_test fixtures has a different one -- see
// data/tools/spectra/make_powr_test_fixture.py), confirming spec() loaded the
// H-bucket file matching this star's own hSurf, not some other one.
static auto testSpecWNLSuccess(
    const std::string& wnlSpectraName, const double hSurf,
    const double expectedPeakWl, const std::string& label) -> int
{
    const specsyn::SpecsynLibWR<specsyn::OOBPolicy::raise> lib(
        wnlSpectraName, -3.0, 1.0, registryName, 0.0, 0.0, 0, testControls);

    const auto props = makeWRStarData(20.0, 5.7, 4.7, 3e-5, 0.7, 0.0, 0.01, hSurf);

    std::vector<double> result;
    try
    {
        result = lib.spec(props, -0.5);
    }
    catch (const std::exception& e)
    {
        std::cerr << "testSpecsynLibWR: unexpected exception from spec() "
            "for an in-bounds " << label << " star: " << e.what() << "\n";
        return 1;
    }

    if (result.size() != lib.wl().size())
    {
        std::cerr << "testSpecsynLibWR: spec() returned " << result.size()
            << " values, expected " << lib.wl().size() << "\n";
        return 1;
    }

    constexpr double starLuminosity = 501187.23362727246 * solarLuminosity; // 10^5.7 Lsun
    double maxWlSpec = 0.0;
    double peakWl = 0.0;
    for (std::size_t i = 0; i < result.size(); ++i)
    {
        const double wl = lib.wl().at(i);
        const double wlSpec = result.at(i) * wl;
        if (wlSpec > maxWlSpec)
        {
            maxWlSpec = wlSpec;
            peakWl = wl;
        }
    }
    if (maxWlSpec < 1e-3 * starLuminosity || maxWlSpec > 1e3 * starLuminosity)
    {
        std::cerr << "testSpecsynLibWR: " << label << ": max(wl * spec) = " <<
            maxWlSpec << " erg/s is unreasonably far from the test star's "
            "own luminosity = " << starLuminosity << " erg/s\n";
        return 1;
    }
    if (std::abs(peakWl - expectedPeakWl) > 2000.0)
    {
        std::cerr << "testSpecsynLibWR: " << label << ": peak of wl * spec() "
            "is at " << peakWl << " Angstrom, expected near " <<
            expectedPeakWl << " Angstrom -- did spec() load the wrong "
            "H-bucket fixture?\n";
        return 1;
    }

    return 0;
}

// Check that spec() treats a WRType mismatch as out of bounds for a
// WNL library too: a star with heSurf = 0.2 and logTeff = 4.0 (see
// testSpecTypeMismatchThrow's own comment for why logTeff must be
// below getWRType's log10(50000) None/WNL boundary here) classifies
// as WRType::None, which can never match wnlSpectraName's own
// WRType::WNLH20/WNLH40/WNLH60, regardless of any other property
static auto testSpecWNLTypeMismatchThrow(const std::string& wnlSpectraName) -> int
{
    const specsyn::SpecsynLibWR<specsyn::OOBPolicy::raise> lib(
        wnlSpectraName, -3.0, 1.0, registryName, 0.0, 0.0, 0, testControls);

    const auto props = makeWRStarData(20.0, 5.7, 4.0, 3e-5, 0.2);

    try
    {
        [[maybe_unused]] const auto result = lib.spec(props, -0.5);
        std::cerr << "testSpecsynLibWR: expected spec() to throw for a "
            "WRType-mismatched star under OOBPolicy::raise, but it did not\n";
        return 1;
    }
    catch (const std::runtime_error&) { /* this is the expected outcome */ }

    return 0;
}

// Check getWRType's WNL surface-H subdivision directly, at and around
// its 0.3/0.5 boundaries (see its own doc comment): hSurf < 0.3 gives
// WRType::WNLH20, hSurf in [0.3, 0.5] gives WRType::WNLH40, and
// hSurf > 0.5 gives WRType::WNLH60. heSurf = 0.7 and logTeff = 4.7
// (comfortably inside the WNL heSurf/logTeff window, unrelated to
// hSurf) are held fixed throughout, so only hSurf varies. wnlRanges
// below stands in for what SpecsynLibChained would normally compute
// from every chained WNL library's own logTeff() -- {4.6, 4.8} is the
// actual log_teff range shared by every POWR_WNL_H*_test fixture (see
// this file's own top-level comment), so this exercises the same gate
// getWRType applies in real use, not a no-op stand-in.
static auto testGetWRTypeWNLHBuckets() -> int
{
    using WRType = specsyn::SpecsynLibWR<specsyn::OOBPolicy::raise>::WRType;
    const std::array<std::pair<double, WRType>, 6> cases = {{
        { 0.0,  WRType::WNLH20 },
        { 0.29, WRType::WNLH20 },
        { 0.3,  WRType::WNLH40 },
        { 0.5,  WRType::WNLH40 },
        { 0.51, WRType::WNLH60 },
        { 1.0,  WRType::WNLH60 },
    }};
    const std::array<std::pair<double, double>, 3> wnlRanges = {{
        { 4.6, 4.8 }, { 4.6, 4.8 }, { 4.6, 4.8 },
    }};

    int result = 0;
    for (const auto& [hSurf, expected] : cases)
    {
        const auto props = makeWRStarData(20.0, 5.7, 4.7, 3e-5, 0.7, 0.0, 0.01, hSurf);
        const auto actual =
            specsyn::SpecsynLibWR<specsyn::OOBPolicy::raise>::getWRType(props, wnlRanges);
        if (actual != expected)
        {
            std::cerr << "testSpecsynLibWR: getWRType(hSurf = " << hSurf
                << ") returned the wrong WNL subtype\n";
            result = 1;
        }
    }
    return result;
}

// Check that a star whose surface composition matches a WNL bucket
// (heSurf = 0.7, hSurf = 0.4 -> WNLH40) but whose own log(Teff) = 4.08
// (~12000 K) falls outside that bucket's real grid coverage
// ({4.6, 4.8} here, standing in for a real WNL grid's own much hotter
// coverage) correctly falls through to WRType::None rather than being
// misclassified as WNLH40 -- this is the exact scenario that used to
// misclassify an ordinary, merely He-enriched evolved star as
// Wolf-Rayet purely because its composition happened to match, even
// though no real WNL spectrum exists anywhere near this cool. Also
// checks that an entirely unknown bucket range (NaN, e.g. because no
// WNLH20 library was ever chained) has the same effect for an
// otherwise-in-range star.
static auto testGetWRTypeOutOfRangeFallsThroughToNone() -> int
{
    using WRType = specsyn::SpecsynLibWR<specsyn::OOBPolicy::raise>::WRType;
    constexpr double nan = std::numeric_limits<double>::quiet_NaN();
    const std::array<std::pair<double, double>, 3> wnlRanges = {{
        { 4.6, 4.8 }, { 4.6, 4.8 }, { 4.6, 4.8 },
    }};
    const std::array<std::pair<double, double>, 3> unknownH20Range = {{
        { nan, nan }, { 4.6, 4.8 }, { 4.6, 4.8 },
    }};

    int result = 0;
    {
        const auto props = makeWRStarData(20.0, 5.0, 4.08, 3e-5, 0.7, 0.0, 0.01, 0.4);
        const auto actual =
            specsyn::SpecsynLibWR<specsyn::OOBPolicy::raise>::getWRType(props, wnlRanges);
        if (actual != WRType::None)
        {
            std::cerr << "testSpecsynLibWR: getWRType for a WNL-composition star "
                "whose logTeff falls outside its bucket's own grid should return "
                "WRType::None, not force a WNL classification\n";
            result = 1;
        }
    }
    {
        // logTeff = 4.65 here (not 4.7, as elsewhere in this file):
        // must stay below the log10(50000) = 4.69897 WNE/WC hot
        // threshold, or this star would legitimately fall through to
        // WRType::WNE via that branch instead, defeating the point of
        // checking that an unknown WNL range alone is what's forcing
        // WRType::None here.
        const auto props = makeWRStarData(20.0, 5.7, 4.65, 3e-5, 0.7, 0.0, 0.01, 0.1);
        const auto actual =
            specsyn::SpecsynLibWR<specsyn::OOBPolicy::raise>::getWRType(props, unknownH20Range);
        if (actual != WRType::None)
        {
            std::cerr << "testSpecsynLibWR: getWRType for a WNLH20-composition star "
                "with an unknown (NaN) WNLH20 range should return WRType::None\n";
            result = 1;
        }
    }
    return result;
}

// Check that requesting nWl alone (wlMin = wlMax = 0, the "not
// supplied" sentinel) falls back to the global wavelength range
// spanned by every populated grid point's own native grid, at the
// requested point count. makeCommonWlGrid's own result spans the full
// union of its input grids' ranges (see its own doc), so a default
// (nWl = 0) construction's wl() front/back already is that same
// global range, giving an independent value to check against without
// duplicating the sweep this test is meant to exercise.
static auto testNWlOnly() -> int
{
    const specsyn::SpecsynLibWR<specsyn::OOBPolicy::raise> libNative(
        spectraName, -3.0, 1.0, registryName, 0.0, 0.0, 0, testControls);
    const auto& wlNative = libNative.wl();

    constexpr std::size_t nWlRequested = 37;
    const specsyn::SpecsynLibWR<specsyn::OOBPolicy::raise> libNWlOnly(
        spectraName, -3.0, 1.0, registryName, 0.0, 0.0, nWlRequested, testControls);
    const auto& wlNWlOnly = libNWlOnly.wl();

    if (wlNWlOnly.size() != nWlRequested)
    {
        std::cerr << "testSpecsynLibWR: nWl-only construction expected "
            << nWlRequested << " wavelength points, got " << wlNWlOnly.size() << "\n";
        return 1;
    }
    if (wlNWlOnly.front() != wlNative.front() || wlNWlOnly.back() != wlNative.back())
    {
        std::cerr << "testSpecsynLibWR: nWl-only construction expected the "
            "native-derived wavelength range [" << wlNative.front() << ", "
            << wlNative.back() << "], got [" << wlNWlOnly.front() << ", "
            << wlNWlOnly.back() << "]\n";
        return 1;
    }

    return 0;
}

auto testSpecsynLibWR() -> int
{
    int result = 0;
    result += testSpecWNESuccess();
    result += testSpecTypeMismatchThrow();
    result += testSpecTypeMismatchSilent();
    result += testSpecLogRtClamped();
    result += testSpecTeffGridBoundsThrow();
    result += testSpecTeffGridBoundsSilent();
    result += testSpecWNLSuccess(wnlH20SpectraName, 0.1, 5000.0, "WNL-H20");
    result += testSpecWNLSuccess(wnlH40SpectraName, 0.4, 10000.0, "WNL-H40");
    result += testSpecWNLSuccess(wnlH60SpectraName, 0.6, 15000.0, "WNL-H60");
    result += testSpecWNLTypeMismatchThrow(wnlH20SpectraName);
    result += testGetWRTypeWNLHBuckets();
    result += testGetWRTypeOutOfRangeFallsThroughToNone();
    result += testNWlOnly();
    return result;
}
