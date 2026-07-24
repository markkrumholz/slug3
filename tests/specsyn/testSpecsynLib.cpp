/**
 * @file testSpecsynLib.cpp
 * @author Mark Krumholz
 * @brief Unit tests for the SpecsynLibNoWind class.
 * @details
 * This file contains end-to-end unit tests for SpecsynLibNoWind::spec,
 * exercising all three of its possible outcomes: a successfully
 * interpolated spectrum, a silently empty spectrum (OOBPolicy::silent),
 * and a thrown runtime error (OOBPolicy::raise), against the small
 * BOSZ_test.h5/spectra.toml fixture stored under tests/specsyn/assets
 * (see SpecsynUtils' own tests for how that fixture is derived from
 * the full-size BOSZ library). It also checks a successful
 * interpolation against TLUSTY_test.h5, a second library with a
 * genuinely irregular [Fe/H] grid and a single, non-"r"-named
 * wavelength grid, to cover both of those (BOSZ doesn't exercise
 * either one). It also tests SpecsynLib::resample, which resamples
 * every stored spectrum onto a new wavelength grid, and the NaN-
 * default handling of the microTurb constructor argument, which
 * resolves to each library's own micro_default registry entry (0 for
 * BOSZ_test, 10 for TLUSTY_test) rather than one shared constant.
 * @date 2026-07-20
 */

#include "../../src/specsyn/SpecsynLibNoWind.hpp"
#include "../../src/tracks/TrackCommons.hpp"
#include "testSpecsynLib.hpp"
#include <algorithm>
#include <array>
#include <cmath>
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
    const std::string spectraName = "BOSZ_test";

    // BOSZ_test.h5's (Teff, logg) grid has exactly one populated cell:
    // Teff in [5750, 6000] K, logg in [4.0, 4.5], present at every one
    // of its 14 feh slices (see the fixture-generation notes in
    // testSpecsynUtils.cpp for how the fixture was built)
    constexpr double solarLuminosity = 3.828e33; // erg/s, IAU 2015 nominal value

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
} // namespace

// Check that spec() successfully interpolates a spectrum for a
// solar-parameter star (M = 1 Msun, Teff = 5772 K, L = 1 Lsun) that
// falls inside BOSZ_test.h5's single populated grid cell -- log(g)
// for these parameters works out to ~4.44 (essentially the Sun's
// actual surface gravity), safely inside the cell's [4.0, 4.5] range,
// since getSAandLogg uses the very same physics used to construct
// this test point. feh = 0.1 lies strictly between two grid points
// (0.0 and 0.25), so this exercises genuine (non-degenerate)
// trilinear interpolation along all three axes at once.
static auto testSpecSuccess() -> int
{
    const specsyn::SpecsynLibNoWind<specsyn::OOBPolicy::raise> lib(
        spectraName, -3.0, 1.0, 0.0, 0.0, 0.0, 500, registryName);

    const double logTeff = std::log10(5772.0);
    const auto props = makeStarData(1.0, 0.0, logTeff);

    std::vector<double> result;
    try
    {
        result = lib.spec(props, 0.1);
    }
    catch (const std::exception& e)
    {
        std::cerr << "testSpecsynLib: unexpected exception from spec() "
            "for an in-bounds star: " << e.what() << "\n";
        return 1;
    }

    if (result.size() != lib.wl().size())
    {
        std::cerr << "testSpecsynLib: spec() returned " << result.size()
            << " values, expected " << lib.wl().size() << "\n";
        return 1;
    }

    // We have no independently-computed expected spectrum to check
    // against (these are real BOSZ model fluxes, not a simple
    // formula), so this is a sanity check only: the peak of
    // wl * spec(wl) -- roughly the luminosity carried per logarithmic
    // wavelength interval -- should be within a few orders of
    // magnitude of the Sun's actual bolometric luminosity for a
    // solar-parameter star, not wildly (many orders of magnitude) off.
    double maxWlSpec = 0.0;
    for (std::size_t i = 0; i < result.size(); ++i)
    {
        maxWlSpec = std::max(maxWlSpec, result.at(i) * lib.wl().at(i));
    }
    if (maxWlSpec < 1e-3 * solarLuminosity || maxWlSpec > 1e3 * solarLuminosity)
    {
        std::cerr << "testSpecsynLib: max(wl * spec) = " << maxWlSpec
            << " erg/s is unreasonably far from Lsun = "
            << solarLuminosity << " erg/s\n";
        return 1;
    }

    return 0;
}

// Check that spec() throws for a star far outside BOSZ_test.h5's grid
// (Teff = 15000 K, well above the fixture's 6000 K maximum) when
// instantiated with OOBPolicy::raise
static auto testSpecOOBThrow() -> int
{
    const specsyn::SpecsynLibNoWind<specsyn::OOBPolicy::raise> lib(
        spectraName, -3.0, 1.0, 0.0, 0.0, 0.0, 500, registryName);

    const double logTeff = std::log10(15000.0);
    const auto props = makeStarData(1.0, 0.0, logTeff);

    try
    {
        [[maybe_unused]] const auto result = lib.spec(props, 0.1);
        std::cerr << "testSpecsynLib: expected spec() to throw for an "
            "out-of-bounds star under OOBPolicy::raise, but it did not\n";
        return 1;
    }
    catch (const std::runtime_error&) { /* this is the expected outcome */ }

    return 0;
}

// Check that spec() silently returns an empty spectrum for the same
// out-of-bounds star when instantiated with OOBPolicy::silent instead
static auto testSpecOOBSilent() -> int
{
    const specsyn::SpecsynLibNoWind<specsyn::OOBPolicy::silent> lib(
        spectraName, -3.0, 1.0, 0.0, 0.0, 0.0, 500, registryName);

    const double logTeff = std::log10(15000.0);
    const auto props = makeStarData(1.0, 0.0, logTeff);

    std::vector<double> result;
    try
    {
        result = lib.spec(props, 0.1);
    }
    catch (const std::exception& e)
    {
        std::cerr << "testSpecsynLib: unexpected exception from spec() "
            "under OOBPolicy::silent: " << e.what() << "\n";
        return 1;
    }

    if (!result.empty())
    {
        std::cerr << "testSpecsynLib: expected spec() to return an empty "
            "spectrum for an out-of-bounds star under OOBPolicy::silent, "
            "got " << result.size() << " values\n";
        return 1;
    }

    return 0;
}

// Check that spec() successfully interpolates a spectrum against
// TLUSTY_test.h5, a second library whose [Fe/H] grid is genuinely
// irregular (log10 of a fixed set of archival Z values, not evenly
// spaced -- unlike BOSZ's) and whose sole wavelength grid is stored
// under a non-"r"-named key ("native", since TLUSTY's downsampling
// means no single r value is meaningful; see fetch_tlusty.py),
// reached at the default r via SpecsynLibNoWind's single-entry fallback.
// The test star (M = 15 Msun, Teff = 28750 K, giving log(g) = 3.125)
// falls inside TLUSTY_test.h5's single populated grid cell: Teff in
// [27500, 30000] K, logg in [3.0, 3.25]. feh = -1.2 lies strictly
// between two of TLUSTY's irregularly-spaced grid points (-1.481 and
// -1.0), so this exercises real interpolation across that irregular
// axis -- the very case findRegularBracket used to get wrong.
static auto testSpecTlustySuccess() -> int
{
    const specsyn::SpecsynLibNoWind<specsyn::OOBPolicy::raise> lib(
        "TLUSTY_test", -3.0, 1.0, 0.0, 0.0, 10.0, specsyn::defaultR, registryName);

    const double logTeff = std::log10(28750.0);
    const auto props = makeStarData(15.0, 5.278432762001573, logTeff);

    std::vector<double> result;
    try
    {
        result = lib.spec(props, -1.2);
    }
    catch (const std::exception& e)
    {
        std::cerr << "testSpecsynLib: unexpected exception from spec() "
            "for an in-bounds TLUSTY star: " << e.what() << "\n";
        return 1;
    }

    if (result.size() != lib.wl().size())
    {
        std::cerr << "testSpecsynLib: TLUSTY spec() returned " << result.size()
            << " values, expected " << lib.wl().size() << "\n";
        return 1;
    }

    // As with testSpecSuccess, no independently-computed expected
    // spectrum exists to check against, so this is a sanity check
    // only: the peak of wl * spec(wl) should be within a few orders
    // of magnitude of the test star's own bolometric luminosity
    // (~1.9e5 Lsun for these parameters), not wildly off.
    constexpr double solarLuminosity = 3.828e33; // erg/s, IAU 2015 nominal value
    constexpr double starLuminosity = 189859.68762747623 * solarLuminosity;
    double maxWlSpec = 0.0;
    for (std::size_t i = 0; i < result.size(); ++i)
    {
        maxWlSpec = std::max(maxWlSpec, result.at(i) * lib.wl().at(i));
    }
    if (maxWlSpec < 1e-3 * starLuminosity || maxWlSpec > 1e3 * starLuminosity)
    {
        std::cerr << "testSpecsynLib: TLUSTY max(wl * spec) = " << maxWlSpec
            << " erg/s is unreasonably far from the test star's own "
            "luminosity = " << starLuminosity << " erg/s\n";
        return 1;
    }

    return 0;
}

// Check that resample() reproduces spec()'s pre-resample values
// exactly at wavelengths carried over unchanged from the original
// grid, and assigns exactly zero flux at wavelengths outside the
// original grid's range. Since spec() sums each grid corner's own
// flux at a given wavelength index with weights that depend only on
// (feh, logg, Teff) -- never on wl_ or the flux values themselves --
// resample()'s per-corner Interpolator1D reproducing each corner's
// flux exactly at any wavelength that coincides with an original grid
// point means spec()'s weighted sum at that same wavelength must also
// be reproduced exactly, regardless of how many corners are actually
// blended together for this particular star.
static auto testResampleExactAndOOB() -> int
{
    specsyn::SpecsynLibNoWind<specsyn::OOBPolicy::raise> lib(
        spectraName, -3.0, 1.0, 0.0, 0.0, 0.0, 500, registryName);

    const double logTeff = std::log10(5772.0);
    const auto props = makeStarData(1.0, 0.0, logTeff);

    const auto wlOrig = lib.wl();
    std::vector<double> before;
    try
    {
        before = lib.spec(props, 0.1);
    }
    catch (const std::exception& e)
    {
        std::cerr << "testSpecsynLib: unexpected exception from spec() "
            "before resample(): " << e.what() << "\n";
        return 1;
    }

    // A new grid mixing: one wavelength below wlOrig's range, three
    // wavelengths copied verbatim from early/middle/late in wlOrig
    // (so resample() must reproduce their flux exactly), and one
    // wavelength above wlOrig's range
    const std::size_t iEarly = 10;
    const std::size_t iMid = wlOrig.size() / 2;
    const std::size_t iLate = wlOrig.size() - 10;
    const std::vector<double> wlNew = {
        wlOrig.front() - 100.0,
        wlOrig.at(iEarly),
        wlOrig.at(iMid),
        wlOrig.at(iLate),
        wlOrig.back() + 100.0
    };

    lib.resample(wlNew);

    if (lib.wl() != wlNew)
    {
        std::cerr << "testSpecsynLib: resample() did not replace wl() with "
            "the new grid\n";
        return 1;
    }

    std::vector<double> after;
    try
    {
        after = lib.spec(props, 0.1);
    }
    catch (const std::exception& e)
    {
        std::cerr << "testSpecsynLib: unexpected exception from spec() "
            "after resample(): " << e.what() << "\n";
        return 1;
    }

    if (after.size() != wlNew.size())
    {
        std::cerr << "testSpecsynLib: spec() returned " << after.size()
            << " values after resample(), expected " << wlNew.size() << "\n";
        return 1;
    }

    // Index 0 (below wlOrig's range) and index 4 (above it) must be
    // exactly zero
    if (after.at(0) != 0.0 || after.at(4) != 0.0)
    {
        std::cerr << "testSpecsynLib: expected exactly zero flux outside "
            "the original wavelength range after resample(), got "
            << after.at(0) << " and " << after.at(4) << "\n";
        return 1;
    }

    // Indices 1-3 (copied verbatim from wlOrig) must reproduce
    // before's value at the corresponding original index
    constexpr double relTol = 1e-8;
    const std::array<std::pair<std::size_t, std::size_t>, 3> matches = {{
        { 1, iEarly }, { 2, iMid }, { 3, iLate }
    }};
    for (const auto& [jNew, iOld] : matches)
    {
        const double expected = before.at(iOld);
        const double got = after.at(jNew);
        if (std::abs(got - expected) > relTol * std::abs(expected))
        {
            std::cerr << "testSpecsynLib: resample() did not reproduce the "
                "original flux at wavelength " << wlNew.at(jNew) << " -- "
                "expected " << expected << ", got " << got << "\n";
            return 1;
        }
    }

    return 0;
}

// Check that resampling onto a grid entirely outside the library's
// original wavelength range leaves every populated grid point's
// spectrum non-empty (still the right length -- OOBPolicy only
// governs (feh, logg, Teff) bounds, never wavelength content), but
// with every one of its fluxes set to zero
static auto testResampleAllOutOfRange() -> int
{
    specsyn::SpecsynLibNoWind<specsyn::OOBPolicy::raise> lib(
        spectraName, -3.0, 1.0, 0.0, 0.0, 0.0, 500, registryName);

    const double logTeff = std::log10(5772.0);
    const auto props = makeStarData(1.0, 0.0, logTeff);

    const auto wlOrig = lib.wl();
    const std::vector<double> wlNew = {
        wlOrig.back() + 100.0, wlOrig.back() + 200.0, wlOrig.back() + 300.0
    };
    lib.resample(wlNew);

    std::vector<double> after;
    try
    {
        after = lib.spec(props, 0.1);
    }
    catch (const std::exception& e)
    {
        std::cerr << "testSpecsynLib: unexpected exception from spec() "
            "after an all-out-of-range resample(): " << e.what() << "\n";
        return 1;
    }

    if (after.size() != wlNew.size())
    {
        std::cerr << "testSpecsynLib: spec() returned " << after.size()
            << " values after an all-out-of-range resample(), expected "
            << wlNew.size() << "\n";
        return 1;
    }
    for (const double v : after)
    {
        if (v != 0.0)
        {
            std::cerr << "testSpecsynLib: expected every flux to be zero "
                "after an all-out-of-range resample(), got " << v << "\n";
            return 1;
        }
    }

    return 0;
}

// Check that OOBPolicy::coerce interpolates a query point that falls
// in a gap (one of its 8 bracketing corners is unpopulated) using only
// its valid neighbors, renormalized by their combined weight, rather
// than treating it as out of bounds -- and that the very same query
// still fails under OOBPolicy::raise/::silent, confirming this really
// is a gap under the old semantics rather than something coerce merely
// papers over regardless of policy. Uses COERCE_test.h5 (see
// data/tools/make_coerce_test_fixture.py), whose only populated
// (Teff, logg) cell is missing exactly one of its four corners
// (Teff = 6000 K, logg = 4.5), each holding a constant (wavelength-
// independent) flux: 1.0, 2.0, and 3.0 at the three populated corners.
// A query at the cell's exact center (Teff = 5500 K, logg = 4.25) sits
// at equal (0.25) weight from all four corners, so under coerce the
// missing corner's weight is simply dropped and the remaining three
// renormalized, working out to the plain average of their flux values
// -- (1.0 + 2.0 + 3.0) / 3 = 2.0 -- scaled by the star's own surface
// area, giving an exact expected result to check against rather than
// only a sanity range.
static auto testSpecCoerce() -> int
{
    const std::string coerceRegistryName = "tests/specsyn/assets/spectra.toml";
    const std::string coerceSpectraName = "COERCE_test";

    // mass = 0.7866094058795904 Msun and area = 7.3774697762410635e+22
    // cm^2 are the two quantities getSAandLogg derives from
    // (mass, logL, Teff) below; mass was chosen (see
    // data/tools/make_coerce_test_fixture.py's own derivation notes)
    // so that log(g) works out to exactly 4.25 -- the center of the
    // fixture's populated cell -- and area follows from L and Teff
    // alone via the Stefan-Boltzmann law, independent of mass.
    constexpr double mass = 0.7866094058795904; // Msun
    constexpr double logL = 0.0;                // log10(L / Lsun)
    constexpr double area = 7.3774697762410635e+22; // cm^2
    constexpr double expectedFlux = ((1.0 + 2.0 + 3.0) / 3.0) * area;
    constexpr double feh = 0.0;
    const double logTeff = std::log10(5500.0);
    const auto props = makeStarData(mass, logL, logTeff);

    // Under coerce: spec() succeeds, interpolating from only the
    // three populated corners
    {
        const specsyn::SpecsynLibNoWind<specsyn::OOBPolicy::coerce> lib(
            coerceSpectraName, 0.0, 0.0, 0.0, 0.0, 0.0, specsyn::defaultR, coerceRegistryName);

        std::vector<double> result;
        try
        {
            result = lib.spec(props, feh);
        }
        catch (const std::exception& e)
        {
            std::cerr << "testSpecsynLib: coerce: unexpected exception from "
                "spec(): " << e.what() << "\n";
            return 1;
        }

        if (result.size() != lib.wl().size())
        {
            std::cerr << "testSpecsynLib: coerce: spec() returned "
                << result.size() << " values, expected " << lib.wl().size() << "\n";
            return 1;
        }

        constexpr double relTol = 1e-8;
        for (std::size_t i = 0; i < result.size(); ++i)
        {
            if (std::abs(result.at(i) - expectedFlux) > relTol * expectedFlux)
            {
                std::cerr << "testSpecsynLib: coerce: spec()[" << i << "] = "
                    << result.at(i) << ", expected " << expectedFlux << "\n";
                return 1;
            }
        }
    }

    // Under raise: the same query still throws, confirming this is a
    // genuine gap
    {
        const specsyn::SpecsynLibNoWind<specsyn::OOBPolicy::raise> lib(
            coerceSpectraName, 0.0, 0.0, 0.0, 0.0, 0.0, specsyn::defaultR, coerceRegistryName);
        try
        {
            [[maybe_unused]] const auto result = lib.spec(props, feh);
            std::cerr << "testSpecsynLib: coerce: expected spec() to throw "
                "under OOBPolicy::raise for the same gap query, but it did not\n";
            return 1;
        }
        catch (const std::runtime_error&) { /* expected */ }
    }

    // Under silent: the same query returns an empty spectrum
    {
        const specsyn::SpecsynLibNoWind<specsyn::OOBPolicy::silent> lib(
            coerceSpectraName, 0.0, 0.0, 0.0, 0.0, 0.0, specsyn::defaultR, coerceRegistryName);
        std::vector<double> result;
        try
        {
            result = lib.spec(props, feh);
        }
        catch (const std::exception& e)
        {
            std::cerr << "testSpecsynLib: coerce: unexpected exception from "
                "spec() under OOBPolicy::silent: " << e.what() << "\n";
            return 1;
        }
        if (!result.empty())
        {
            std::cerr << "testSpecsynLib: coerce: expected an empty spectrum "
                "under OOBPolicy::silent for the same gap query, got "
                << result.size() << " values\n";
            return 1;
        }
    }

    return 0;
}

// Check that leaving microTurb at its default (NaN) resolves to each
// library's own micro_default registry entry, rather than one shared
// default: BOSZ_test's is 0 and TLUSTY_test's is 10, so a NaN-default
// construction of each must produce the exact same spectrum as an
// explicit construction at that library's own value, and (since the
// two libraries' defaults really do differ) explicitly requesting the
// wrong one for TLUSTY_test must fail to find any matching spectra.
static auto testMicroTurbDefault() -> int
{
    constexpr double nan = std::numeric_limits<double>::quiet_NaN();

    try
    {
        const specsyn::SpecsynLibNoWind<specsyn::OOBPolicy::raise> boszDefault(
            spectraName, -3.0, 1.0, 0.0, 0.0, nan, 500, registryName);
        const specsyn::SpecsynLibNoWind<specsyn::OOBPolicy::raise> boszExplicit(
            spectraName, -3.0, 1.0, 0.0, 0.0, 0.0, 500, registryName);

        const auto props = makeStarData(1.0, 0.0, std::log10(5772.0));
        if (boszDefault.spec(props, 0.1) != boszExplicit.spec(props, 0.1))
        {
            std::cerr << "testSpecsynLib: leaving microTurb at its default did "
                "not resolve to BOSZ_test's own micro_default (0)\n";
            return 1;
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "testSpecsynLib: unexpected exception resolving "
            "BOSZ_test's default microTurb: " << e.what() << "\n";
        return 1;
    }

    try
    {
        const specsyn::SpecsynLibNoWind<specsyn::OOBPolicy::raise> tlustyDefault(
            "TLUSTY_test", -3.0, 1.0, 0.0, 0.0, nan, specsyn::defaultR, registryName);
        const specsyn::SpecsynLibNoWind<specsyn::OOBPolicy::raise> tlustyExplicit(
            "TLUSTY_test", -3.0, 1.0, 0.0, 0.0, 10.0, specsyn::defaultR, registryName);

        const auto props = makeStarData(15.0, 5.278432762001573, std::log10(28750.0));
        if (tlustyDefault.spec(props, -1.2) != tlustyExplicit.spec(props, -1.2))
        {
            std::cerr << "testSpecsynLib: leaving microTurb at its default did "
                "not resolve to TLUSTY_test's own micro_default (10)\n";
            return 1;
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "testSpecsynLib: unexpected exception resolving "
            "TLUSTY_test's default microTurb: " << e.what() << "\n";
        return 1;
    }

    try
    {
        [[maybe_unused]] const specsyn::SpecsynLibNoWind<specsyn::OOBPolicy::raise> tlustyWrongMicro(
            "TLUSTY_test", -3.0, 1.0, 0.0, 0.0, 0.0, specsyn::defaultR, registryName);
        std::cerr << "testSpecsynLib: expected constructing TLUSTY_test with "
            "microTurb = 0 (BOSZ_test's default, not its own) to fail, but it "
            "did not\n";
        return 1;
    }
    catch (const std::runtime_error&) { /* expected */ }

    return 0;
}

auto testSpecsynLib() -> int
{
    int result = 0;
    result += testSpecSuccess();
    result += testSpecOOBThrow();
    result += testSpecOOBSilent();
    result += testSpecTlustySuccess();
    result += testResampleExactAndOOB();
    result += testResampleAllOutOfRange();
    result += testMicroTurbDefault();
    result += testSpecCoerce();
    return result;
}
