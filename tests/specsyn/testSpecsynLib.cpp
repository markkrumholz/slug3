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

#include "../../src/io/SimControls.hpp"
#include "../../src/specsyn/SpecsynLibNoWind.hpp"
#include "../../src/tracks/TrackCommons.hpp"
#include "../../src/utils/Constants.hpp"
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

    // Every SpecsynLibNoWind constructor call in this file now needs
    // an explicit controls argument -- see Specsyn's own controls_
    // member, which stores a live reference rather than a snapshot,
    // so a default-constructed temporary is no longer safe to bind
    // to it. This file only ever exercises spec() (never specCts()),
    // so the tolerances this holds are never actually read; a plain,
    // all-defaults SimControls is enough.
    const io::SimControls testControls;

    // BOSZ_test.h5's (Teff, logg) grid has exactly one populated cell:
    // Teff in [5750, 6000] K, logg in [4.0, 4.5], present at every one
    // of its 14 feh slices (see the fixture-generation notes in
    // testSpecsynUtils.cpp for how the fixture was built)
    constexpr double solarLuminosity = utils::Lsun;

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
        spectraName, -3.0, 1.0, 0.0, 0.0, 0.0, 500, registryName, 0.0, 0.0, 0, 0.0, testControls);

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
        spectraName, -3.0, 1.0, 0.0, 0.0, 0.0, 500, registryName, 0.0, 0.0, 0, 0.0, testControls);

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
        spectraName, -3.0, 1.0, 0.0, 0.0, 0.0, 500, registryName, 0.0, 0.0, 0, 0.0, testControls);

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
        "TLUSTY_test", -3.0, 1.0, 0.0, 0.0, 10.0, specsyn::defaultR, registryName, 0.0, 0.0, 0, 0.0, testControls);

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
    constexpr double solarLuminosity = utils::Lsun;
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
        spectraName, -3.0, 1.0, 0.0, 0.0, 0.0, 500, registryName, 0.0, 0.0, 0, 0.0, testControls);

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

    // A new grid consisting of wlOrig's own points verbatim, plus one
    // wavelength an order of magnitude below wlOrig's range and one
    // an order of magnitude above it. resample() now integrates each
    // new grid point's flux over a bin bounded by the geometric
    // (log-space) midpoints to its own neighbors in the new grid (see
    // SpecsynLib::resample's own comment), rather than point-sampling,
    // so reproducing the original flux exactly is no longer expected
    // -- but keeping wlOrig's own points (rather than some coarser
    // grid) means each interior bin is only as wide as wlOrig's own
    // native spacing, which the library's own resolution should
    // already adequately resolve, so the result should still stay
    // close to the original point value. The two added endpoints, by
    // contrast, sit far enough outside wlOrig's range that even their
    // own (much wider, since their only neighbor in the new grid is
    // wlOrig's own first/last point) bins cannot reach back into it,
    // so they still exercise the "genuinely outside the library's
    // range" case cleanly.
    std::vector<double> wlNew;
    wlNew.reserve(wlOrig.size() + 2);
    wlNew.push_back(wlOrig.front() / 10.0);
    wlNew.insert(wlNew.end(), wlOrig.begin(), wlOrig.end());
    wlNew.push_back(wlOrig.back() * 10.0);

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

    // The two added endpoints (genuinely outside wlOrig's range) must
    // be exactly zero
    if (after.front() != 0.0 || after.back() != 0.0)
    {
        std::cerr << "testSpecsynLib: expected exactly zero flux outside "
            "the original wavelength range after resample(), got "
            << after.front() << " and " << after.back() << "\n";
        return 1;
    }

    // Every point copied verbatim from wlOrig should closely -- though,
    // per the comment above, no longer exactly -- reproduce before's
    // value at the corresponding original index (relative error stays
    // under 5% everywhere, in practice, and well under that for all
    // but a handful of points right at the steepest edges of an
    // absorption line, where even wlOrig's own native resolution
    // doesn't quite make the interpolant featureless across one bin).
    // Points where the original flux is a tiny fraction of the
    // spectrum's peak (e.g. deep in a strong absorption line, or off
    // the Wien tail) are skipped: relative error is not a meaningful
    // measure there, and a bin average that pulls in a neighboring,
    // much brighter point can legitimately move such a tiny value by
    // a large relative amount without indicating anything actually
    // wrong.
    const double peak = *std::ranges::max_element(before);
    constexpr double relTol = 0.05;
    constexpr double peakFrac = 1e-3;
    for (std::size_t i = 0; i < wlOrig.size(); ++i)
    {
        const double expected = before.at(i);
        if (std::abs(expected) < peakFrac * peak) { continue; }
        const double got = after.at(i + 1); // +1 for the leading below-range endpoint
        if (std::abs(got - expected) > relTol * std::abs(expected))
        {
            std::cerr << "testSpecsynLib: resample() did not approximately "
                "reproduce the original flux at wavelength "
                << wlNew.at(i + 1) << " -- expected " << expected
                << ", got " << got << "\n";
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
        spectraName, -3.0, 1.0, 0.0, 0.0, 0.0, 500, registryName, 0.0, 0.0, 0, 0.0, testControls);

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
// SpecsynLibNoWind interpolates in log(Teff) rather than Teff itself,
// so the query needs to sit at the cell's exact center in log(Teff)
// space -- log10(Teff) = (log10(5000) + log10(6000)) / 2, i.e.
// Teff = sqrt(5000 * 6000) =~ 5477.2256 K, the geometric (not
// arithmetic) mean of the two grid points -- to land at equal (0.25)
// weight from all four corners. There, under coerce the missing
// corner's weight is simply dropped and the remaining three
// renormalized, working out to the plain average of their flux values
// -- (1.0 + 2.0 + 3.0) / 3 = 2.0 -- scaled by the star's own surface
// area, giving an exact expected result to check against rather than
// only a sanity range.
static auto testSpecCoerce() -> int
{
    const std::string coerceRegistryName = "tests/specsyn/assets/spectra.toml";
    const std::string coerceSpectraName = "COERCE_test";

    // logTeff is the exact log-space midpoint between the fixture's
    // two Teff grid points (see above). mass = 0.7997741882974353
    // Msun and area = 7.500939930135089e+22 cm^2 are the two
    // quantities getSAandLogg derives from (mass, logL, logTeff)
    // below; mass was chosen so that log(g) works out to exactly 4.25
    // -- the center of the fixture's populated cell on the logg axis
    // -- and area follows from L and Teff alone via the
    // Stefan-Boltzmann law, independent of mass.
    const double logTeff = (std::log10(5000.0) + std::log10(6000.0)) / 2.0;
    constexpr double mass = 0.7997741882974353; // Msun
    constexpr double logL = 0.0;                // log10(L / Lsun)
    constexpr double area = 7.500939930135089e+22; // cm^2
    constexpr double expectedFlux = ((1.0 + 2.0 + 3.0) / 3.0) * area;
    constexpr double feh = 0.0;
    const auto props = makeStarData(mass, logL, logTeff);

    // Under coerce: spec() succeeds, interpolating from only the
    // three populated corners
    {
        const specsyn::SpecsynLibNoWind<specsyn::OOBPolicy::coerce> lib(
            coerceSpectraName, 0.0, 0.0, 0.0, 0.0, 0.0, specsyn::defaultR, coerceRegistryName, 0.0, 0.0, 0, 0.0, testControls);

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
            coerceSpectraName, 0.0, 0.0, 0.0, 0.0, 0.0, specsyn::defaultR, coerceRegistryName, 0.0, 0.0, 0, 0.0, testControls);
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
            coerceSpectraName, 0.0, 0.0, 0.0, 0.0, 0.0, specsyn::defaultR, coerceRegistryName, 0.0, 0.0, 0, 0.0, testControls);
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

// Check that OOBPolicy::coerce reports a query as out of bounds rather
// than dividing by zero when a query lands exactly on a grid line
// along one axis, so that every corner with nonzero interpolation
// weight is unpopulated even though some (zero-weight) corner is
// populated. Uses COERCE_ZERO_test.h5 (see
// data/tools/make_coerce_zero_weight_test_fixture.py), whose Teff axis
// is {1000, 10000} K and whose only populated points are (1000, 4.0),
// (1000, 4.5), and (10000, 5.0). A query at exactly Teff = 10000 K
// (which -- unlike an arbitrary Teff -- round-trips exactly through
// spec()'s log10()/pow(10, .) conversion, since 10000 is a power of
// ten) with logg strictly between 4.0 and 4.5 sits at this library's
// upper Teff edge: the Teff = 1000 corners get exactly zero weight
// (skipped before spec() ever checks whether they're populated), while
// the Teff = 10000, logg in {4.0, 4.5} corners -- the only ones
// actually used -- are both unpopulated. hasValidNeighbor is still
// true (the zero-weight Teff = 1000 corners are populated), so before
// the wSum <= 0 guard was added to SpecsynLib::spec(), this divided by
// zero and returned a NaN/Inf "spectrum" instead of a clean out-of-
// bounds result -- exactly what testClusterSpecsynFull's full-scale
// run against real data turned up. mass = 0.07197967694676924 Msun and
// logL = 0 are chosen (mirroring testSpecCoerce's own derivation) so
// that log(g) works out to 4.25 -- strictly between this fixture's two
// populated logg values -- at Teff = 10000 K.
static auto testSpecCoerceZeroWeight() -> int
{
    const std::string coerceRegistryName = "tests/specsyn/assets/spectra.toml";
    const std::string coerceSpectraName = "COERCE_ZERO_test";

    constexpr double mass = 0.07197967694676924; // Msun
    constexpr double logL = 0.0;                 // log10(L / Lsun)
    constexpr double feh = 0.0;
    const double logTeff = std::log10(10000.0);
    const auto props = makeStarData(mass, logL, logTeff);

    const specsyn::SpecsynLibNoWind<specsyn::OOBPolicy::coerce> lib(
        coerceSpectraName, 0.0, 0.0, 0.0, 0.0, 0.0, specsyn::defaultR, coerceRegistryName, 0.0, 0.0, 0, 0.0, testControls);

    std::vector<double> result;
    try
    {
        result = lib.spec(props, feh);
    }
    catch (const std::exception& e)
    {
        std::cerr << "testSpecsynLib: coerceZeroWeight: unexpected "
            "exception from spec(): " << e.what() << "\n";
        return 1;
    }

    if (!result.empty())
    {
        std::cerr << "testSpecsynLib: coerceZeroWeight: expected an empty "
            "(out-of-bounds) spectrum for a query landing exactly on the "
            "grid's missing corner, got " << result.size() <<
            " values -- possibly a NaN/Inf divide-by-zero regression\n";
        return 1;
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
            spectraName, -3.0, 1.0, 0.0, 0.0, nan, 500, registryName, 0.0, 0.0, 0, 0.0, testControls);
        const specsyn::SpecsynLibNoWind<specsyn::OOBPolicy::raise> boszExplicit(
            spectraName, -3.0, 1.0, 0.0, 0.0, 0.0, 500, registryName, 0.0, 0.0, 0, 0.0, testControls);

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
            "TLUSTY_test", -3.0, 1.0, 0.0, 0.0, nan, specsyn::defaultR, registryName, 0.0, 0.0, 0, 0.0, testControls);
        const specsyn::SpecsynLibNoWind<specsyn::OOBPolicy::raise> tlustyExplicit(
            "TLUSTY_test", -3.0, 1.0, 0.0, 0.0, 10.0, specsyn::defaultR, registryName, 0.0, 0.0, 0, 0.0, testControls);

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
            "TLUSTY_test", -3.0, 1.0, 0.0, 0.0, 0.0, specsyn::defaultR, registryName, 0.0, 0.0, 0, 0.0, testControls);
        std::cerr << "testSpecsynLib: expected constructing TLUSTY_test with "
            "microTurb = 0 (BOSZ_test's default, not its own) to fail, but it "
            "did not\n";
        return 1;
    }
    catch (const std::runtime_error&) { /* expected */ }

    return 0;
}

// Check that resample()'s bin-integration approach (see
// SpecsynLib::resample's own comment) is approximately
// flux-conserving when heavily downsampling a library's native
// wavelength grid onto a much coarser one -- unlike simple
// point-sampling, which can badly misrepresent this comparison
// whenever downsampling happens to skip over a narrow spectral
// feature entirely. Builds two copies of the same library: one at
// its native resolution, and one resampled (via the wlMin/wlMax/nWl
// constructor arguments) onto just 64 points spanning that same
// wavelength range. Computes the same star's spectrum from both, and
// compares their own (trapezoidal-rule) integrated flux -- which
// should stay close between the two despite the 64-point grid having
// far too few points to resolve any of the native spectrum's actual
// narrow features.
static auto testResampleFluxConservation() -> int
{
    specsyn::SpecsynLibNoWind<specsyn::OOBPolicy::raise> libNative(
        spectraName, -3.0, 1.0, 0.0, 0.0, 0.0, 500, registryName, 0.0, 0.0, 0, 0.0, testControls);
    const auto wlNative = libNative.wl();

    constexpr std::size_t nWlCoarse = 64;
    specsyn::SpecsynLibNoWind<specsyn::OOBPolicy::raise> libCoarse(
        spectraName, -3.0, 1.0, 0.0, 0.0, 0.0, 500, registryName,
        wlNative.front(), wlNative.back(), nWlCoarse, 0.0, testControls);

    const double logTeff = std::log10(5772.0);
    const auto props = makeStarData(1.0, 0.0, logTeff);

    std::vector<double> specNative;
    std::vector<double> specCoarse;
    try
    {
        specNative = libNative.spec(props, 0.1);
        specCoarse = libCoarse.spec(props, 0.1);
    }
    catch (const std::exception& e)
    {
        std::cerr << "testSpecsynLib: unexpected exception from spec() in "
            "the flux conservation test: " << e.what() << "\n";
        return 1;
    }

    // Trapezoidal-rule integral of a spectrum over its own wavelength grid
    auto trapzIntegral = [](const std::vector<double>& wl, const std::vector<double>& flux) -> double
    {
        double total = 0.0;
        for (std::size_t i = 0; i + 1 < wl.size(); ++i)
        {
            total += 0.5 * (flux.at(i) + flux.at(i + 1)) * (wl.at(i + 1) - wl.at(i));
        }
        return total;
    };

    const double integralNative = trapzIntegral(wlNative, specNative);
    const double integralCoarse = trapzIntegral(libCoarse.wl(), specCoarse);

    constexpr double relTol = 0.01;
    const double relErr = std::abs(integralCoarse - integralNative) / std::abs(integralNative);
    if (relErr > relTol)
    {
        std::cerr << "testSpecsynLib: flux conservation check failed -- "
            "native integral = " << integralNative << ", coarse (nWl = "
            << nWlCoarse << ") integral = " << integralCoarse
            << ", relative difference = " << relErr << ", tolerance = "
            << relTol << "\n";
        return 1;
    }

    return 0;
}

// Check that requesting nWl alone (wlMin = wlMax = 0, the "not
// supplied" sentinel) falls back to this library's own native
// wavelength range -- the same range a default (nWl = 0) construction
// reads from disk -- at the requested point count
static auto testResampleNWlOnly() -> int
{
    const specsyn::SpecsynLibNoWind<specsyn::OOBPolicy::raise> libNative(
        spectraName, -3.0, 1.0, 0.0, 0.0, 0.0, 500, registryName, 0.0, 0.0, 0, 0.0, testControls);
    const auto& wlNative = libNative.wl();

    constexpr std::size_t nWlRequested = 37;
    const specsyn::SpecsynLibNoWind<specsyn::OOBPolicy::raise> libNWlOnly(
        spectraName, -3.0, 1.0, 0.0, 0.0, 0.0, 500, registryName,
        0.0, 0.0, nWlRequested, 0.0, testControls);
    const auto& wlNWlOnly = libNWlOnly.wl();

    if (wlNWlOnly.size() != nWlRequested)
    {
        std::cerr << "testSpecsynLib: nWl-only resample expected "
            << nWlRequested << " wavelength points, got " << wlNWlOnly.size() << "\n";
        return 1;
    }
    if (wlNWlOnly.front() != wlNative.front() || wlNWlOnly.back() != wlNative.back())
    {
        std::cerr << "testSpecsynLib: nWl-only resample expected the native "
            "wavelength range [" << wlNative.front() << ", " << wlNative.back()
            << "], got [" << wlNWlOnly.front() << ", " << wlNWlOnly.back() << "]\n";
        return 1;
    }

    return 0;
}

// Check that SpecsynLibNoWind correctly interpolates between two
// bracketing [alpha/Fe] values when no exact-match groups exist for
// the requested afe. Uses the BOSZ_test.h5 fixture's two synthetic
// groups at feh=0.0: afe=0.25 (constant flux = 1.0) and afe=0.50
// (constant flux = 3.0). Requesting afe=0.375 gives alpha=0.5, so
// the interpolated flux is (1-0.5)*1.0 + 0.5*3.0 = 2.0 at every
// wavelength. A constant-flux grid always interpolates to exactly
// that constant regardless of star position inside the cell, so the
// returned spectrum is 2.0 * area at every wavelength, where area is
// Lsun / (sigma_SB * 5772^4). Also checks that:
//   - exact-afe construction still works (backward-compat check)
//   - requesting afe outside the available range throws
static auto testSpecAfeInterp() -> int
{
    // area = Lsun / (sigma_SB * Teff^4) for (logL=0, Teff=5772 K)
    // Using SLUG's constants: Lsun=3.828e33, sigma_SB=GSL value
    constexpr double areaExpected = 6.0820909091272306e+22; // cm^2
    constexpr double loFlux = 1.0;
    constexpr double hiFlux = 3.0;
    // alpha = (0.375 - 0.25) / (0.50 - 0.25) = 0.5
    constexpr double expectedFluxPerWl = (1.0 - 0.5) * loFlux + 0.5 * hiFlux; // = 2.0
    constexpr double expectedSpec = expectedFluxPerWl * areaExpected;
    constexpr double relTol = 1e-8;

    const double logTeff = std::log10(5772.0);
    const auto props = makeStarData(1.0, 0.0, logTeff);
    constexpr double feh = 0.0;
    // fehMin == fehMax == 0.0 → only the feh=0.0 group is loaded
    constexpr double fehMin = 0.0;
    constexpr double fehMax = 0.0;

    // --- interpolating path ---
    {
        const specsyn::SpecsynLibNoWind<specsyn::OOBPolicy::raise> lib(
            spectraName, fehMin, fehMax, 0.375, 0.0, 0.0, 500, registryName, 0.0, 0.0, 0, 0.0, testControls);

        std::vector<double> result;
        try
        {
            result = lib.spec(props, feh);
        }
        catch (const std::exception& e)
        {
            std::cerr << "testSpecsynLib: afeInterp: unexpected exception from "
                "spec() at afe=0.375: " << e.what() << "\n";
            return 1;
        }

        if (result.size() != lib.wl().size())
        {
            std::cerr << "testSpecsynLib: afeInterp: spec() returned "
                << result.size() << " values, expected " << lib.wl().size() << "\n";
            return 1;
        }

        for (std::size_t i = 0; i < result.size(); ++i)
        {
            if (std::abs(result.at(i) - expectedSpec) > relTol * expectedSpec)
            {
                std::cerr << "testSpecsynLib: afeInterp: spec()[" << i << "] = "
                    << result.at(i) << ", expected " << expectedSpec
                    << " (afe=0.375 interpolated)\n";
                return 1;
            }
        }
    }

    // --- exact-match path still works ---
    {
        try
        {
            const specsyn::SpecsynLibNoWind<specsyn::OOBPolicy::raise> lib(
                spectraName, fehMin, fehMax, 0.25, 0.0, 0.0, 500, registryName, 0.0, 0.0, 0, 0.0, testControls);
            const auto result = lib.spec(props, feh);
            if (result.size() != lib.wl().size())
            {
                std::cerr << "testSpecsynLib: afeInterp: exact-match at afe=0.25 "
                    "returned wrong size\n";
                return 1;
            }
            for (std::size_t i = 0; i < result.size(); ++i)
            {
                const double expected = loFlux * areaExpected;
                if (std::abs(result.at(i) - expected) > relTol * expected)
                {
                    std::cerr << "testSpecsynLib: afeInterp: exact-match spec()["
                        << i << "] = " << result.at(i) << ", expected "
                        << expected << " (afe=0.25 exact)\n";
                    return 1;
                }
            }
        }
        catch (const std::exception& e)
        {
            std::cerr << "testSpecsynLib: afeInterp: unexpected exception for "
                "exact afe=0.25 match: " << e.what() << "\n";
            return 1;
        }
    }

    // --- afe outside available range throws ---
    {
        try
        {
            [[maybe_unused]] const specsyn::SpecsynLibNoWind<specsyn::OOBPolicy::raise> lib(
                spectraName, fehMin, fehMax, 0.75, 0.0, 0.0, 500, registryName, 0.0, 0.0, 0, 0.0, testControls);
            std::cerr << "testSpecsynLib: afeInterp: expected constructor to throw "
                "for afe=0.75 (above library's maximum 0.50), but it did not\n";
            return 1;
        }
        catch (const std::runtime_error&) { /* expected */ }
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
    result += testResampleExactAndOOB();
    result += testResampleAllOutOfRange();
    result += testResampleFluxConservation();
    result += testResampleNWlOnly();
    result += testMicroTurbDefault();
    result += testSpecCoerce();
    result += testSpecCoerceZeroWeight();
    result += testSpecAfeInterp();
    return result;
}
