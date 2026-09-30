/**
 * @file testWinds.cpp
 * @author Mark Krumholz
 * @brief Unit tests for the Winds class.
 * @details
 * Every expected value below was computed independently, by a small
 * Python script implementing the exact same documented formula (Nugis
 * & Lamers 2000, A&A, 360, 227) against the same GSL cgs constants
 * (GSL_CONST_CGSM_GRAVITATIONAL_CONSTANT, GSL_CONST_CGSM_SOLAR_MASS,
 * GSL_CONST_CGSM_STEFAN_BOLTZMANN_CONSTANT) and utils::Lsun that
 * Winds.cpp itself uses -- not by calling Winds::vWindWR() and
 * checking it against itself. The unclamped-velocity cases (mass,
 * logL, logTeff, hSurf, heSurf, cSurf, nSurf) are real snapshots taken
 * directly from data/tracks/mist.h5, found by sweeping its Wolf-Rayet-
 * classified points for representative WNE/WC/WNL-composition cases
 * whose raw (pre-clamp) velocity happens to already fall inside
 * [740, 5500] km/s, so those tests also exercise std::clamp's own
 * pass-through branch, not just its two boundary branches.
 * @date 2026-09-30
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#include "../../src/feedback/Winds.hpp"
#include "../../src/io/SimControls.hpp"
#include "../../src/tracks/TrackCommons.hpp"
#include "../../src/utils/MiscUtils.hpp"
#include "testWinds.hpp"
#include <cstddef>
#include <iostream>
#include <string_view>

namespace
{
    // Default-constructed: Winds does not yet read anything from
    // controls_ (see Winds.hpp's own comment), so this is sufficient
    // for every test in this file.
    const io::SimControls testControls;

    /**
     * @brief Build a StarData for a Wolf-Rayet wind velocity test
     */
    auto makeStarData(
        const double mass,
        const double logL,
        const double logTeff,
        const double hSurf,
        const double heSurf,
        const double cSurf,
        const double nSurf) -> specsyn::Specsyn::StarData
    {
        specsyn::Specsyn::StarData props{};
        props.at(static_cast<std::size_t>(tracks::FieldIdx::mass)) = mass;
        props.at(static_cast<std::size_t>(tracks::FieldIdx::logL)) = logL;
        props.at(static_cast<std::size_t>(tracks::FieldIdx::logTe)) = logTeff;
        props.at(static_cast<std::size_t>(tracks::FieldIdx::hSurf)) = hSurf;
        props.at(static_cast<std::size_t>(tracks::FieldIdx::heSurf)) = heSurf;
        props.at(static_cast<std::size_t>(tracks::FieldIdx::cSurf)) = cSurf;
        props.at(static_cast<std::size_t>(tracks::FieldIdx::nSurf)) = nSurf;
        return props;
    }

    // Relative tolerance for the formula-derived (non-clamped) checks --
    // generous enough to absorb any floating-point operation-ordering
    // difference between this file's independently-computed expected
    // values and Winds.cpp's own arithmetic, while still catching a
    // wrong constant, sign, or exponent
    constexpr double relTol = 1e-4;

    auto checkApproxEqual(
        const double actual, const double expected, const std::string_view label) -> int
    {
        if (!utils::approxEqual(actual, expected, relTol * expected))
        {
            std::cerr << "testWinds: " << label << ": vWindWR() = " << actual
                << " cm/s, expected " << expected << " cm/s\n";
            return 1;
        }
        return 0;
    }
} // namespace

// A plausible WNE star (cSurf < nSurf), taken from a real MIST track
// (feh=0, afe=0, vvcrit=0, initial mass 48 Msun) whose predicted raw
// velocity already falls inside [740, 5500] km/s, so vWindWR() should
// return the unclamped formula result.
static auto testVWindWRWNEUnclamped() -> int
{
    const feedback::Winds winds(testControls);
    const auto props = makeStarData(
        19.2440, 5.90241, 5.33139, 0.0, 0.98412906, 0.00017037, 0.01056297);

    constexpr double expectedCgs = 1.0003508346e8; // 1000.35 km/s
    return checkApproxEqual(winds.vWindWR(props), expectedCgs, "testVWindWRWNEUnclamped");
}

// A plausible WC star (cSurf >= nSurf), taken from a real MIST track
// (feh=0, afe=0, vvcrit=0, initial mass 50 Msun), again unclamped.
static auto testVWindWRWCUnclamped() -> int
{
    const feedback::Winds winds(testControls);
    const auto props = makeStarData(
        19.8360, 5.81330, 5.17038, 0.0, 0.23631528, 0.43523672, 0.00075809);

    constexpr double expectedCgs = 2.0027169059e8; // 2002.72 km/s
    return checkApproxEqual(winds.vWindWR(props), expectedCgs, "testVWindWRWCUnclamped");
}

// A WNL-composition star (heSurf = 0.87, inside getWRType's [0.4, 0.9]
// WNL window) but with cSurf < nSurf, taken from a real MIST track
// (feh=-0.75, afe=0, vvcrit=0, initial mass 180 Msun). Nugis & Lamers
// provide no separate WNL calibration, so vWindWR() must apply the WN
// formula here -- the WC formula would give a markedly different,
// still-unclamped answer (1857.96 km/s, computed by the same
// independent script) for these same inputs, so this test would catch
// a wrong formula selection, not just a wrong constant.
static auto testVWindWRWNLCompositionUsesWNFormula() -> int
{
    const feedback::Winds winds(testControls);
    const auto props = makeStarData(
        21.2593, 5.74500, 5.10231, 0.00000002, 0.87078103, 0.00177230, 0.11921254);

    constexpr double expectedCgs = 8.9454507500e7; // 894.55 km/s (WN formula)
    return checkApproxEqual(
        winds.vWindWR(props), expectedCgs, "testVWindWRWNLCompositionUsesWNFormula");
}

// Gamma_e > 1 (super-Eddington) makes v_esc's own radicand negative --
// here mass = 20 Msun and logL = 7.0 together give Gamma_e ~= 7.68,
// comfortably over the boundary. Must return the clamp's own lower
// bound exactly, rather than propagating a NaN through the final
// std::clamp (see Winds.hpp's own comment).
static auto testVWindWRGammaEGreaterThanOne() -> int
{
    const feedback::Winds winds(testControls);
    const auto props = makeStarData(20.0, 7.0, 4.7, 0.0, 1.0, 0.0, 0.01);

    constexpr double expectedCgs = 7.4e7; // 740 km/s, exactly
    const double actual = winds.vWindWR(props);
    if (actual != expectedCgs)
    {
        std::cerr << "testVWindWRGammaEGreaterThanOne: vWindWR() = " << actual
            << " cm/s, expected exactly " << expectedCgs << " cm/s\n";
        return 1;
    }
    return 0;
}

// A near-Eddington (but Gamma_e < 1) WNE star -- mass = 200 Msun,
// logL = 7.1 -- whose small (1 - Gamma_e) collapses v_esc, and
// therefore the raw formula velocity (48.06 km/s), far below the
// calibration sample's own 740 km/s floor. Must be clamped up to it.
static auto testVWindWRClampToLower() -> int
{
    const feedback::Winds winds(testControls);
    const auto props = makeStarData(200.0, 7.1, 4.7, 0.033, 0.944, 0.0, 0.01);

    constexpr double expectedCgs = 7.4e7; // 740 km/s, exactly
    const double actual = winds.vWindWR(props);
    if (actual != expectedCgs)
    {
        std::cerr << "testVWindWRClampToLower: vWindWR() = " << actual
            << " cm/s, expected exactly " << expectedCgs << " cm/s\n";
        return 1;
    }
    return 0;
}

// A very luminous, very metal-poor (in the sense of a small surface Z)
// WC star -- mass = 280 Msun, logL = 7.06, heSurf = 0.07 -- whose raw
// formula velocity (26079 km/s) is pushed far above the calibration
// sample's own 5500 km/s ceiling by the WC formula's steep L
// dependence at extreme luminosity, well outside the (much lower)
// luminosity range Nugis & Lamers actually calibrated against. Must be
// clamped down to the ceiling.
static auto testVWindWRClampToUpper() -> int
{
    const feedback::Winds winds(testControls);
    const auto props = makeStarData(280.0, 7.06, 5.5, 0.0, 0.07, 0.5, 0.01);

    constexpr double expectedCgs = 5.5e8; // 5500 km/s, exactly
    const double actual = winds.vWindWR(props);
    if (actual != expectedCgs)
    {
        std::cerr << "testVWindWRClampToUpper: vWindWR() = " << actual
            << " cm/s, expected exactly " << expectedCgs << " cm/s\n";
        return 1;
    }
    return 0;
}

// Winds retains a live reference to the SimControls it was
// constructed with, matching Specsyn's/Extinct's/Yields's own
// identical pattern -- checked here the same way
// testSpecsynLib.cpp checks Specsyn's own controls().
static auto testWindsControlsAccessor() -> int
{
    const feedback::Winds winds(testControls);
    if (&winds.controls() != &testControls)
    {
        std::cerr << "testWindsControlsAccessor: controls() did not return "
            "a reference to the SimControls this Winds was constructed with\n";
        return 1;
    }
    return 0;
}

auto testWinds() -> int
{
    int result = 0;
    result += testVWindWRWNEUnclamped();
    result += testVWindWRWCUnclamped();
    result += testVWindWRWNLCompositionUsesWNFormula();
    result += testVWindWRGammaEGreaterThanOne();
    result += testVWindWRClampToLower();
    result += testVWindWRClampToUpper();
    result += testWindsControlsAccessor();
    return result;
}
