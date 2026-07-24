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
 * data/tools/make_powr_test_fixture.py for exactly how it was built.
 * Sharing one SED and logl across every point removes any dependence
 * on the interpolation weights themselves from the expected result,
 * isolating what these tests actually care about: that spec() derives
 * a (FeH, logRt, logTeff) point that lands inside this small grid for
 * a plausible WNE star, classifies WRType and grid-range mismatches as
 * out of bounds, and rescales the result to the star's own luminosity
 * rather than the grid point's.
 * @date 2026-07-23
 */

#include "../../src/specsyn/SpecsynLibWR.hpp"
#include "../../src/tracks/TrackCommons.hpp"
#include "testSpecsynLibWR.hpp"
#include <algorithm>
#include <cstddef>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    const std::string registryName = "tests/specsyn/assets/spectra.toml";
    const std::string spectraName = "POWR_WNE_test";
    const std::string wnlSpectraName = "POWR_WNL_test";
    constexpr double solarLuminosity = 3.828e33; // erg/s, IAU 2015 nominal value

    /**
     * @brief Build a StarData for a Wolf-Rayet star
     * @details
     * hSurf, cSurf, and nSurf default to values that classify as
     * WRType::WNE (hSurf <= 1e-5, cSurf < nSurf) via getWRType,
     * matching POWR_WNE_test's own type_.
     */
    auto makeWRStarData(
        const double mass,
        const double logL,
        const double logTeff,
        const double mdot,
        const double hSurf = 1e-6,
        const double cSurf = 0.0,
        const double nSurf = 0.01) -> specsyn::Specsyn::StarData
    {
        specsyn::Specsyn::StarData props{};
        props.at(static_cast<std::size_t>(tracks::FieldIdx::mass)) = mass;
        props.at(static_cast<std::size_t>(tracks::FieldIdx::mdot)) = mdot;
        props.at(static_cast<std::size_t>(tracks::FieldIdx::logL)) = logL;
        props.at(static_cast<std::size_t>(tracks::FieldIdx::logTe)) = logTeff;
        props.at(static_cast<std::size_t>(tracks::FieldIdx::hSurf)) = hSurf;
        props.at(static_cast<std::size_t>(tracks::FieldIdx::cSurf)) = cSurf;
        props.at(static_cast<std::size_t>(tracks::FieldIdx::nSurf)) = nSurf;
        return props;
    }
} // namespace

// Check that spec() successfully interpolates a spectrum for a
// plausible WNE star (M = 20 Msun, L = 10^5.7 Lsun, Teff = 10^4.7 K,
// Mdot = 3e-5 Msun/yr) that works out (see
// data/tools/make_powr_test_fixture.py's own derivation notes) to
// logRt ~= 0.74, comfortably inside POWR_WNE_test's [0.5, 1.0] log_rt
// range; logTeff = 4.7 and feh = -0.5 similarly fall strictly between
// this fixture's grid points, so this exercises genuine (non-
// degenerate) trilinear interpolation along all three axes at once.
static auto testSpecWNESuccess() -> int
{
    const specsyn::SpecsynLibWR<specsyn::OOBPolicy::Throw> lib(
        spectraName, -3.0, 1.0, registryName);

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
// with hSurf = 0.5 (well above the 0.3 threshold in getWRType)
// classifies as WRType::None, which can never match POWR_WNE_test's
// own WRType::WNE, regardless of any other property
static auto testSpecTypeMismatchThrow() -> int
{
    const specsyn::SpecsynLibWR<specsyn::OOBPolicy::Throw> lib(
        spectraName, -3.0, 1.0, registryName);

    const auto props = makeWRStarData(20.0, 5.7, 4.7, 3e-5, 0.5);

    try
    {
        [[maybe_unused]] const auto result = lib.spec(props, -0.5);
        std::cerr << "testSpecsynLibWR: expected spec() to throw for a "
            "WRType-mismatched star under OOBPolicy::Throw, but it did not\n";
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
        spectraName, -3.0, 1.0, registryName);

    const auto props = makeWRStarData(20.0, 5.7, 4.7, 3e-5, 0.5);

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

// Check that spec() treats a derived (FeH, logRt, logTeff) point
// outside the grid as out of bounds, even though the star's own
// WRType matches this library's. M = 20 Msun, L = 10^5.0 Lsun,
// Teff = 10^4.7 K, Mdot = 1e-4 Msun/yr works out to logRt ~= -0.77 (see
// data/tools/make_powr_test_fixture.py's own derivation notes), well
// outside POWR_WNE_test's [0.5, 1.0] log_rt range.
static auto testSpecGridBoundsThrow() -> int
{
    const specsyn::SpecsynLibWR<specsyn::OOBPolicy::Throw> lib(
        spectraName, -3.0, 1.0, registryName);

    const auto props = makeWRStarData(20.0, 5.0, 4.7, 1e-4);

    try
    {
        [[maybe_unused]] const auto result = lib.spec(props, -0.5);
        std::cerr << "testSpecsynLibWR: expected spec() to throw for a star "
            "whose logRt falls outside the grid under OOBPolicy::Throw, "
            "but it did not\n";
        return 1;
    }
    catch (const std::runtime_error&) { /* this is the expected outcome */ }

    return 0;
}

// Check that spec() silently returns an empty spectrum for the same
// out-of-grid-range star when instantiated with OOBPolicy::silent instead
static auto testSpecGridBoundsSilent() -> int
{
    const specsyn::SpecsynLibWR<specsyn::OOBPolicy::silent> lib(
        spectraName, -3.0, 1.0, registryName);

    const auto props = makeWRStarData(20.0, 5.0, 4.7, 1e-4);

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
// plausible WNL star (same mass/L/Teff/Mdot as testSpecWNESuccess,
// just with hSurf = 0.1 so it classifies as WRType::WNL instead of
// WRType::WNE), loading from POWR_WNL_test -- a fixture whose
// constructor comments (see make_powr_test_fixture.py) describe two
// groups per [Fe/H]: an H20 (xh = 0.20) group with a Gaussian SED
// peaked at 5000 Angstrom (width 500 Angstrom, so utterly negligible
// by 15000 Angstrom -- 20 sigma out), and a decoy (xh = 0.50) group
// with an equally tall Gaussian peaked at 15000 Angstrom instead.
// Beyond the same in-range/luminosity sanity checks as
// testSpecWNESuccess, this also checks that flux near 15000 Angstrom
// is negligible relative to the peak: if SpecsynLibWR's WNL H20-only
// filter (see its constructor) failed to discard the decoy group, the
// decoy's own grid point would sit exactly at this star's feh = -0.5
// query bracket alongside the H20 one (verified against this exact
// fixture by hand: without the filter, spec() blends the two 50/50,
// making the decoy's peak at 15000 Angstrom come out about as tall as
// the real H20 peak at 5000 -- not just "present but small"), so this
// check would fail loudly rather than by a hard-to-notice margin.
static auto testSpecWNLSuccess() -> int
{
    const specsyn::SpecsynLibWR<specsyn::OOBPolicy::Throw> lib(
        wnlSpectraName, -3.0, 1.0, registryName);

    const auto props = makeWRStarData(20.0, 5.7, 4.7, 3e-5, 0.1);

    std::vector<double> result;
    try
    {
        result = lib.spec(props, -0.5);
    }
    catch (const std::exception& e)
    {
        std::cerr << "testSpecsynLibWR: unexpected exception from spec() "
            "for an in-bounds WNL star: " << e.what() << "\n";
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
    double decoyWlSpec = 0.0; // wl * spec() nearest 15000 Angstrom, the decoy group's own peak
    double bestDecoyDist = std::numeric_limits<double>::max();
    for (std::size_t i = 0; i < result.size(); ++i)
    {
        const double wl = lib.wl().at(i);
        const double wlSpec = result.at(i) * wl;
        maxWlSpec = std::max(maxWlSpec, wlSpec);

        const double decoyDist = std::abs(wl - 15000.0);
        if (decoyDist < bestDecoyDist)
        {
            bestDecoyDist = decoyDist;
            decoyWlSpec = wlSpec;
        }
    }
    if (maxWlSpec < 1e-3 * starLuminosity || maxWlSpec > 1e3 * starLuminosity)
    {
        std::cerr << "testSpecsynLibWR: max(wl * spec) = " << maxWlSpec
            << " erg/s is unreasonably far from the test star's own "
            "luminosity = " << starLuminosity << " erg/s\n";
        return 1;
    }
    if (decoyWlSpec > 1e-6 * maxWlSpec)
    {
        std::cerr << "testSpecsynLibWR: wl * spec() near 15000 Angstrom = " <<
            decoyWlSpec << " erg/s is a non-negligible fraction of the peak "
            "(" << maxWlSpec << " erg/s) -- consistent with POWR_WNL_test's "
            "decoy (xh = 0.50) group leaking into the result, meaning the "
            "WNL H20-only filter does not appear to have excluded it\n";
        return 1;
    }

    return 0;
}

// Check that spec() treats a WRType mismatch as out of bounds for a
// WNL library too: a star with hSurf = 0.5 (above the 0.3 threshold in
// getWRType) classifies as WRType::None, which can never match
// POWR_WNL_test's own WRType::WNL, regardless of any other property
static auto testSpecWNLTypeMismatchThrow() -> int
{
    const specsyn::SpecsynLibWR<specsyn::OOBPolicy::Throw> lib(
        wnlSpectraName, -3.0, 1.0, registryName);

    const auto props = makeWRStarData(20.0, 5.7, 4.7, 3e-5, 0.5);

    try
    {
        [[maybe_unused]] const auto result = lib.spec(props, -0.5);
        std::cerr << "testSpecsynLibWR: expected spec() to throw for a "
            "WRType-mismatched star under OOBPolicy::Throw, but it did not\n";
        return 1;
    }
    catch (const std::runtime_error&) { /* this is the expected outcome */ }

    return 0;
}

auto testSpecsynLibWR() -> int
{
    int result = 0;
    result += testSpecWNESuccess();
    result += testSpecTypeMismatchThrow();
    result += testSpecTypeMismatchSilent();
    result += testSpecGridBoundsThrow();
    result += testSpecGridBoundsSilent();
    result += testSpecWNLSuccess();
    result += testSpecWNLTypeMismatchThrow();
    return result;
}
