/**
 * @file testSpecsynLib.cpp
 * @author Mark Krumholz
 * @brief Unit tests for the SpecsynLib class.
 * @details
 * This file contains end-to-end unit tests for SpecsynLib::spec,
 * exercising all three of its possible outcomes: a successfully
 * interpolated spectrum, a silently empty spectrum (OOBPolicy::silent),
 * and a thrown runtime error (OOBPolicy::Throw), against the small
 * BOSZ_test.h5/spectra.toml fixture stored under tests/specsyn/assets
 * (see SpecsynUtils' own tests for how that fixture is derived from
 * the full-size BOSZ library). It also checks a successful
 * interpolation against TLUSTY_test.h5, a second library with a
 * genuinely irregular [Fe/H] grid and a single, non-"r"-named
 * wavelength grid, to cover both of those (BOSZ doesn't exercise
 * either one).
 * @date 2026-07-20
 */

#include "../../src/specsyn/SpecsynLib.hpp"
#include "../../src/tracks/TrackCommons.hpp"
#include "testSpecsynLib.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
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
    const specsyn::SpecsynLib<specsyn::OOBPolicy::Throw> lib(
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
// instantiated with OOBPolicy::Throw
static auto testSpecOOBThrow() -> int
{
    const specsyn::SpecsynLib<specsyn::OOBPolicy::Throw> lib(
        spectraName, -3.0, 1.0, 0.0, 0.0, 0.0, 500, registryName);

    const double logTeff = std::log10(15000.0);
    const auto props = makeStarData(1.0, 0.0, logTeff);

    try
    {
        [[maybe_unused]] const auto result = lib.spec(props, 0.1);
        std::cerr << "testSpecsynLib: expected spec() to throw for an "
            "out-of-bounds star under OOBPolicy::Throw, but it did not\n";
        return 1;
    }
    catch (const std::runtime_error&) { /* this is the expected outcome */ }

    return 0;
}

// Check that spec() silently returns an empty spectrum for the same
// out-of-bounds star when instantiated with OOBPolicy::silent instead
static auto testSpecOOBSilent() -> int
{
    const specsyn::SpecsynLib<specsyn::OOBPolicy::silent> lib(
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
// reached at the default r via SpecsynLib's single-entry fallback.
// The test star (M = 15 Msun, Teff = 28750 K, giving log(g) = 3.125)
// falls inside TLUSTY_test.h5's single populated grid cell: Teff in
// [27500, 30000] K, logg in [3.0, 3.25]. feh = -1.2 lies strictly
// between two of TLUSTY's irregularly-spaced grid points (-1.481 and
// -1.0), so this exercises real interpolation across that irregular
// axis -- the very case findRegularBracket used to get wrong.
static auto testSpecTlustySuccess() -> int
{
    const specsyn::SpecsynLib<specsyn::OOBPolicy::Throw> lib(
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

auto testSpecsynLib() -> int
{
    int result = 0;
    result += testSpecSuccess();
    result += testSpecOOBThrow();
    result += testSpecOOBSilent();
    result += testSpecTlustySuccess();
    return result;
}
