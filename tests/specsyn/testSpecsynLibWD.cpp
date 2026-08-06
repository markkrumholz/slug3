/**
 * @file testSpecsynLibWD.cpp
 * @author Mark Krumholz
 * @brief Unit tests for the SpecsynLibWD class.
 * @details
 * This file contains end-to-end unit tests for SpecsynLibWD::spec,
 * against the small WD_test.h5/spectra.toml fixture stored under
 * tests/specsyn/assets. That fixture is entirely synthetic (not real
 * Tremblay et al. data, which is far too large to store in the
 * repository): a (logg, log_Teff) = {7.0, 8.0, 9.0} x {4.0, 4.3, 4.6}
 * grid, with flux(logg, logTeff, wl) = amplitude(logg, logTeff) *
 * shape(wl), where amplitude is an exactly linear function of (logg,
 * logTeff) -- see data/tools/make_wd_test_fixture.py for the precise
 * coefficients and shape values used. Bilinear interpolation of a
 * function that is linear (not bilinear) in its two variables is
 * exact everywhere within the grid, not just at grid points, so
 * testInterpOffGrid below checks an off-grid point exactly, rather
 * than only approximately or only at grid points.
 * @date 2026-08-07
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

    // Every SpecsynLibWD constructor call in this file needs an
    // explicit controls argument -- see testSpecsynLib.cpp's own
    // identical comment on testControls for why.
    const io::SimControls testControls;

    // Mirrors data/tools/make_wd_test_fixture.py's own constants exactly
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

auto testSpecsynLibWD() -> int
{
    int result = 0;
    result += testExactGridPoint();
    result += testInterpOffGrid();
    result += testOutOfBoundsLogTeff();
    result += testOutOfBoundsLogg();
    return result;
}
