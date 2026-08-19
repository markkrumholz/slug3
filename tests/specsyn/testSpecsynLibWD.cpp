/**
 * @file testSpecsynLibWD.cpp
 * @author Mark Krumholz
 * @brief Unit tests for the SpecsynLibWD class.
 * @details
 * This file contains end-to-end unit tests for SpecsynLibWD::spec,
 * against two small synthetic fixtures stored under
 * tests/specsyn/assets (neither is real data, which is far too large
 * to store in the repository):
 *
 *   - WD_test.h5, a completely filled (logg, log_Teff) = {7.0, 8.0,
 *     9.0} x {4.0, 4.3, 4.6} tensor grid (mirroring fetch_tremblay.py's
 *     own schema), with flux(logg, logTeff, wl) = amplitude(logg,
 *     logTeff) * shape(wl), where amplitude is an exactly linear
 *     function of (logg, logTeff) -- see
 *     data/tools/spectra/make_wd_test_fixture.py for the precise coefficients
 *     and shape values used. Bilinear interpolation of a function that
 *     is linear (not bilinear) in its two variables is exact
 *     everywhere within the grid, not just at grid points, so
 *     testInterpOffGrid below checks an off-grid point exactly, rather
 *     than only approximately or only at grid points.
 *   - RAUCH_test.h5, a partially-filled (logg, log_Teff) grid
 *     (mirroring fetch_rauch.py's own schema, and its real data's own
 *     partially-filled coverage), with one corner of an otherwise
 *     complete 2x2 cell deliberately left unpopulated -- see
 *     data/tools/spectra/make_rauch_test_fixture.py -- used by
 *     testSparseGridExactPoint/testSparseGridMissingCorner below to
 *     exercise SpecsynLib2D::spec()'s unpopulated-neighbor handling
 *     under all three OOBPolicy variants.
 * @date 2026-08-07
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#include "../../src/io/SimControls.hpp"
#include "../../src/specsyn/SpecsynLibWD.hpp"
#include "../../src/tracks/TrackCommons.hpp"
#include "../../src/utils/Constants.hpp"
#include "testSpecsynLibWD.hpp"
#include <cmath>
#include <cstddef>
#include <iostream>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    const std::string registryName = "tests/specsyn/assets/spectra.toml";
    const std::string spectraName = "WD_test";
    const std::string sparseSpectraName = "RAUCH_test";

    // Mirrors data/tools/spectra/make_rauch_test_fixture.py's own constants
    // exactly: a partially-filled 3x2 (logg, log_Teff) grid, missing
    // only the (logg=9.0, log_Teff=4.5) corner, with a constant
    // (wavelength-independent) flux value at each populated point.
    // This leaves the [7.0, 8.0] logg cell completely populated (used
    // by testSparseGridExactPoint) and the [8.0, 9.0] logg cell with
    // one missing corner (used by testSparseGridMissingCorner).
    constexpr double sparseLoggFull = 7.0;  // corner of the fully-populated cell
    constexpr double sparseLoggGapLo = 8.0; // corners of the cell with a missing corner
    constexpr double sparseLoggGapHi = 9.0;
    constexpr double sparseLogTeffLo = 4.0;
    constexpr double sparseLogTeffHi = 4.5;
    constexpr double sparseFluxFullLo = 1.0;  // (logg=7.0, log_Teff=4.0)
    constexpr double sparseFluxGapLoLo = 3.0; // (logg=8.0, log_Teff=4.0)
    constexpr double sparseFluxGapLoHi = 4.0; // (logg=8.0, log_Teff=4.5)
    constexpr double sparseFluxGapHiLo = 5.0; // (logg=9.0, log_Teff=4.0)
    // (logg=9.0, log_Teff=4.5) is deliberately missing

    // Every SpecsynLibWD constructor call in this file needs an
    // explicit controls argument -- see testSpecsynLib.cpp's own
    // identical comment on testControls for why.
    const io::SimControls testControls;

    // Mirrors data/tools/spectra/make_wd_test_fixture.py's own constants exactly
    constexpr double amp0 = 1.0;
    constexpr double ampLogg = 2.0;
    constexpr double ampLogTeff = 3.0;
    const std::vector<double> shape = { 1.0, 2.0, 3.0, 4.0, 5.0 };

    auto amplitude(const double logg, const double logTeff) -> double
    {
        return amp0 + (ampLogg * logg) + (ampLogTeff * logTeff);
    }

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
     * @brief Surface area implied by (logL, logTeff), mirroring Specsyn::getSAandLogg
     */
    auto surfaceArea(const double logL, const double logTeff) -> double
    {
        constexpr double pi = std::numbers::pi_v<double>;
        const double temperature = std::pow(10.0, logTeff);
        const double luminosity = std::pow(10.0, logL) * utils::Lsun;
        const double temperature4 = temperature * temperature * temperature * temperature;
        const double radius = std::sqrt(luminosity / (4.0 * pi * utils::sigmaSB * temperature4));
        return 4.0 * pi * radius * radius;
    }

    /**
     * @brief Mass that lands exactly at targetLogg, given (logL, logTeff)
     * @details
     * log(g) = log10(G * mass * Msun / R^2), and R is fixed by (logL,
     * logTeff) alone (see surfaceArea's own radius calculation) -- so
     * solving for mass at a target log(g) is a direct algebraic
     * inversion, letting tests place a star at an exact grid point (or
     * a precisely known off-grid point) rather than guessing.
     */
    auto massForLogg(const double targetLogg, const double logL, const double logTeff) -> double
    {
        const double area = surfaceArea(logL, logTeff);
        const double radiusSq = area / (4.0 * std::numbers::pi_v<double>);
        const double g = std::pow(10.0, targetLogg);
        return g * radiusSq / (utils::G * utils::Msun);
    }

    /**
     * @brief Check result against amplitude(logg, logTeff) * shape * surfaceArea(logL, logTeff), exactly
     */
    auto checkResult(const std::vector<double>& result, const double logg,
        const double logL, const double logTeff, const std::string& label) -> int
    {
        if (result.size() != shape.size())
        {
            std::cerr << "testSpecsynLibWD: " << label << ": result has " << result.size()
                << " elements, expected " << shape.size() << "\n";
            return 1;
        }
        const double area = surfaceArea(logL, logTeff);
        const double amp = amplitude(logg, logTeff);
        for (std::size_t i = 0; i < shape.size(); ++i)
        {
            const double expected = area * amp * shape.at(i);
            const double relErr = std::abs(result.at(i) - expected) / std::abs(expected);
            if (relErr > 1e-10)
            {
                std::cerr << "testSpecsynLibWD: " << label << ": result[" << i << "] = "
                    << result.at(i) << ", expected " << expected
                    << " (relative error " << relErr << ")\n";
                return 1;
            }
        }
        return 0;
    }

    /**
     * @brief Check result against expectedFlux * surfaceArea(logL, logTeff), exactly, at every wavelength
     * @details
     * Used for the RAUCH_test fixture, whose populated points each
     * hold a constant (wavelength-independent) flux value rather than
     * WD_test's amplitude(logg, logTeff) * shape(wl) form -- see this
     * file's own sparseFlux* constants and
     * data/tools/spectra/make_rauch_test_fixture.py.
     */
    auto checkConstantResult(const std::vector<double>& result, const double expectedFlux,
        const double logL, const double logTeff, const std::string& label) -> int
    {
        if (result.size() != 3)
        {
            std::cerr << "testSpecsynLibWD: " << label << ": result has " << result.size()
                << " elements, expected 3\n";
            return 1;
        }
        const double area = surfaceArea(logL, logTeff);
        const double expected = area * expectedFlux;
        for (std::size_t i = 0; i < result.size(); ++i)
        {
            const double relErr = std::abs(result.at(i) - expected) / std::abs(expected);
            if (relErr > 1e-10)
            {
                std::cerr << "testSpecsynLibWD: " << label << ": result[" << i << "] = "
                    << result.at(i) << ", expected " << expected
                    << " (relative error " << relErr << ")\n";
                return 1;
            }
        }
        return 0;
    }
} // namespace

// Check that a star at an exact (logg, log_Teff) grid point reproduces
// the fixture's own stored value exactly (up to the area scaling
// spec() itself applies)
static auto testExactGridPoint() -> int
{
    const specsyn::SpecsynLibWD<specsyn::OOBPolicy::raise> wd(
        spectraName, registryName, 0.0, 0.0, 0, testControls);

    constexpr double logL = 0.0;
    constexpr double logTeff = 4.3; // exact grid point
    constexpr double targetLogg = 8.0; // exact grid point
    const double mass = massForLogg(targetLogg, logL, logTeff);

    try
    {
        const auto result = wd.spec(makeStarData(mass, logL, logTeff), 0.0);
        return checkResult(result, targetLogg, logL, logTeff, "exact grid point");
    }
    catch (const std::exception& e)
    {
        std::cerr << "testSpecsynLibWD: exact grid point: unexpected exception: "
            << e.what() << "\n";
        return 1;
    }
}

// Check that an off-grid (logg, log_Teff) point -- strictly between
// grid points on both axes -- is bilinearly interpolated exactly,
// exploiting the fixture's amplitude being an exactly linear (not
// just bilinear) function of (logg, logTeff)
static auto testInterpOffGrid() -> int
{
    const specsyn::SpecsynLibWD<specsyn::OOBPolicy::raise> wd(
        spectraName, registryName, 0.0, 0.0, 0, testControls);

    constexpr double logL = 0.3;
    constexpr double logTeff = 4.15; // strictly between 4.0 and 4.3
    constexpr double targetLogg = 7.65; // strictly between 7.0 and 8.0
    const double mass = massForLogg(targetLogg, logL, logTeff);

    try
    {
        const auto result = wd.spec(makeStarData(mass, logL, logTeff), 0.0);
        return checkResult(result, targetLogg, logL, logTeff, "off-grid interpolation");
    }
    catch (const std::exception& e)
    {
        std::cerr << "testSpecsynLibWD: off-grid interpolation: unexpected exception: "
            << e.what() << "\n";
        return 1;
    }
}

// Check that a star outside log_Teff's own range is out of bounds:
// throws for OOBPolicy::raise, returns empty for OOBPolicy::silent
static auto testOutOfBoundsLogTeff() -> int
{
    int result = 0;
    constexpr double logL = 0.0;
    constexpr double logTeff = 5.0; // outside {4.0, ..., 4.6}
    const double mass = massForLogg(8.0, logL, logTeff);
    const auto props = makeStarData(mass, logL, logTeff);

    const specsyn::SpecsynLibWD<specsyn::OOBPolicy::raise> raiseWd(
        spectraName, registryName, 0.0, 0.0, 0, testControls);
    try
    {
        [[maybe_unused]] const auto r = raiseWd.spec(props, 0.0);
        std::cerr << "testSpecsynLibWD: out-of-bounds log(Teff): expected a throw, got none\n";
        result += 1;
    }
    catch (const std::runtime_error&) { /* expected */ }

    const specsyn::SpecsynLibWD<specsyn::OOBPolicy::silent> silentWd(
        spectraName, registryName, 0.0, 0.0, 0, testControls);
    const auto silentResult = silentWd.spec(props, 0.0);
    if (!silentResult.empty())
    {
        std::cerr << "testSpecsynLibWD: out-of-bounds log(Teff): expected an empty "
            "result for OOBPolicy::silent, got " << silentResult.size() << " elements\n";
        result += 1;
    }
    return result;
}

// Check that a star outside logg's own range is out of bounds: throws
// for OOBPolicy::raise
static auto testOutOfBoundsLogg() -> int
{
    constexpr double logL = 5.0; // pushes logg well below the grid's own floor at fixed Teff
    constexpr double logTeff = 4.3;
    const auto props = makeStarData(1.0, logL, logTeff);

    const specsyn::SpecsynLibWD<specsyn::OOBPolicy::raise> wd(
        spectraName, registryName, 0.0, 0.0, 0, testControls);
    try
    {
        [[maybe_unused]] const auto r = wd.spec(props, 0.0);
        std::cerr << "testSpecsynLibWD: out-of-bounds log(g): expected a throw, got none\n";
        return 1;
    }
    catch (const std::runtime_error&) { /* expected */ }
    return 0;
}

// Check that an exact, populated grid point in the fully-populated
// cell of the RAUCH_test fixture (see
// data/tools/spectra/make_rauch_test_fixture.py) reproduces its own stored
// constant flux value exactly under OOBPolicy::raise -- confirms
// SpecsynLibWD's sparse-grid loading path itself is correct,
// independent of the missing-corner handling testSparseGridMissingCorner
// below exercises. This deliberately queries the [7.0, 8.0] logg cell,
// which never touches the fixture's one missing corner at
// (logg=9.0, log_Teff=4.5) -- unlike SpecsynLib::spec(double, double,
// double), a raise/silent query anywhere in a cell requires all 4 of
// that cell's corners to be populated, not just the ones with nonzero
// interpolation weight, so this test would fail even at this exact
// corner if it were placed in the gapped cell instead.
static auto testSparseGridExactPoint() -> int
{
    const specsyn::SpecsynLibWD<specsyn::OOBPolicy::raise> wd(
        sparseSpectraName, registryName, 0.0, 0.0, 0, testControls);

    constexpr double logL = 0.0;
    constexpr double logTeff = sparseLogTeffLo; // exact grid point
    constexpr double targetLogg = sparseLoggFull; // exact grid point
    const double mass = massForLogg(targetLogg, logL, logTeff);

    try
    {
        const auto result = wd.spec(makeStarData(mass, logL, logTeff), 0.0);
        return checkConstantResult(result, sparseFluxFullLo, logL, logTeff, "sparse exact grid point");
    }
    catch (const std::exception& e)
    {
        std::cerr << "testSpecsynLibWD: sparse exact grid point: unexpected exception: "
            << e.what() << "\n";
        return 1;
    }
}

// Check that a query point whose bracketing cell has one missing
// corner (see data/tools/spectra/make_rauch_test_fixture.py) is handled
// per-OOBPolicy exactly as SpecsynLib::spec(double, double, double)
// handles the analogous 3D case: raise throws, silent returns empty,
// and coerce interpolates from the 3 populated corners alone,
// renormalized by their combined weight
static auto testSparseGridMissingCorner() -> int
{
    int result = 0;
    constexpr double logL = 0.0;
    constexpr double logTeff = 0.5 * (sparseLogTeffLo + sparseLogTeffHi); // 4.25: center of the cell
    constexpr double targetLogg = 0.5 * (sparseLoggGapLo + sparseLoggGapHi); // 8.5: center of the cell
    const double mass = massForLogg(targetLogg, logL, logTeff);
    const auto props = makeStarData(mass, logL, logTeff);

    const specsyn::SpecsynLibWD<specsyn::OOBPolicy::raise> raiseWd(
        sparseSpectraName, registryName, 0.0, 0.0, 0, testControls);
    try
    {
        [[maybe_unused]] const auto r = raiseWd.spec(props, 0.0);
        std::cerr << "testSpecsynLibWD: sparse missing corner: expected a throw "
            "for OOBPolicy::raise, got none\n";
        result += 1;
    }
    catch (const std::runtime_error&) { /* expected */ }

    const specsyn::SpecsynLibWD<specsyn::OOBPolicy::silent> silentWd(
        sparseSpectraName, registryName, 0.0, 0.0, 0, testControls);
    const auto silentResult = silentWd.spec(props, 0.0);
    if (!silentResult.empty())
    {
        std::cerr << "testSpecsynLibWD: sparse missing corner: expected an empty "
            "result for OOBPolicy::silent, got " << silentResult.size() << " elements\n";
        result += 1;
    }

    const specsyn::SpecsynLibWD<specsyn::OOBPolicy::coerce> coerceWd(
        sparseSpectraName, registryName, 0.0, 0.0, 0, testControls);
    const auto coerceResult = coerceWd.spec(props, 0.0);
    // All 4 corners share equal (0.25) weight at this exact center
    // query point, so coercing over the 3 populated ones renormalizes
    // to their plain average -- mirrors make_coerce_test_fixture.py's
    // own identical reasoning
    constexpr double expectedFlux =
        (sparseFluxGapLoLo + sparseFluxGapLoHi + sparseFluxGapHiLo) / 3.0;
    result += checkConstantResult(coerceResult, expectedFlux, logL, logTeff,
        "sparse missing corner (coerce)");
    return result;
}

auto testSpecsynLibWD() -> int
{
    int result = 0;
    result += testExactGridPoint();
    result += testInterpOffGrid();
    result += testOutOfBoundsLogTeff();
    result += testOutOfBoundsLogg();
    result += testSparseGridExactPoint();
    result += testSparseGridMissingCorner();
    return result;
}
