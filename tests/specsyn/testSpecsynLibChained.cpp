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
 * (BOSZ_test was fetched at micro = 0, TLUSTY_test at micro = 10),
 * which is exactly the scenario the per-library microTurb vector is
 * for. The tests check that a star handled by the second library in
 * the chain falls through correctly after the first returns an empty
 * (out-of-bounds) result, that this holds regardless of which library
 * is listed first, and that a star outside every chained library's
 * grid still throws (since the last library in the chain always uses
 * OOBPolicy::Throw).
 *
 * BOSZ_test.h5 and TLUSTY_test.h5 have different wavelength grids
 * (different lengths and spacing), and SpecsynLibChained::wl()
 * currently just exposes its first chained library's grid without
 * resampling (a known limitation, to be addressed separately), so
 * these tests build standalone reference SpecsynLib instances of
 * their own purely to obtain each fixture's own wl() for the sanity
 * checks below, rather than relying on the chain's wl().
 * @date 2026-07-21
 */

#include "../../src/specsyn/SpecsynLib.hpp"
#include "../../src/specsyn/SpecsynLibChained.hpp"
#include "../../src/tracks/TrackCommons.hpp"
#include "testSpecsynLibChained.hpp"
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

    constexpr double solarLuminosity = 3.828e33; // erg/s, IAU 2015 nominal value
    constexpr double obLuminosity = 189859.68762747623 * solarLuminosity;

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
     * @brief Reference BOSZ_test library, used only to obtain its own wl() for sanity checks
     */
    auto boszRef() -> specsyn::SpecsynLib<specsyn::OOBPolicy::Throw>
    {
        return { "BOSZ_test", -3.0, 1.0, 0.0, 0.0, 0.0, specsyn::defaultR, registryName };
    }

    /**
     * @brief Reference TLUSTY_test library, used only to obtain its own wl() for sanity checks
     */
    auto tlustyRef() -> specsyn::SpecsynLib<specsyn::OOBPolicy::Throw>
    {
        return { "TLUSTY_test", -3.0, 1.0, 0.0, 0.0, 10.0, specsyn::defaultR, registryName };
    }

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
} // namespace

// Check that, with TLUSTY_test listed first, an OB star is handled
// immediately by TLUSTY_test, and a solar star -- out of bounds for
// TLUSTY_test -- falls through to BOSZ_test
static auto testChainTlustyFirst() -> int
{
    const specsyn::SpecsynLibChained chain(
        { "TLUSTY_test", "BOSZ_test" }, -3.0, 1.0, 0.0, 0.0,
        { 10.0, 0.0 }, specsyn::defaultR, registryName);
    const auto tlusty = tlustyRef();
    const auto bosz = boszRef();

    int result = 0;
    try
    {
        const auto obResult = chain.spec(obStar(), obFeh);
        result += checkSpectrum(obResult, tlusty.wl(), obLuminosity,
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
        result += checkSpectrum(solarResult, bosz.wl(), solarLuminosity,
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
        { 0.0, 10.0 }, specsyn::defaultR, registryName);
    const auto tlusty = tlustyRef();
    const auto bosz = boszRef();

    int result = 0;
    try
    {
        const auto solarResult = chain.spec(solarStar(), solarFeh);
        result += checkSpectrum(solarResult, bosz.wl(), solarLuminosity,
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
        result += checkSpectrum(obResult, tlusty.wl(), obLuminosity,
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

// Check that a star outside every chained library's grid throws,
// since the last library in the chain is always constructed with
// OOBPolicy::Throw regardless of chain order
static auto testChainOOBThrows() -> int
{
    const specsyn::SpecsynLibChained chain(
        { "TLUSTY_test", "BOSZ_test" }, -3.0, 1.0, 0.0, 0.0,
        { 10.0, 0.0 }, specsyn::defaultR, registryName);

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
            {}, -3.0, 1.0, 0.0, 0.0, {}, specsyn::defaultR, registryName);
        std::cerr << "testSpecsynLibChained: expected the constructor to throw "
            "for an empty spectraName, but it did not\n";
        result += 1;
    }
    catch (const std::runtime_error&) { /* expected */ }

    try
    {
        [[maybe_unused]] const specsyn::SpecsynLibChained chain(
            { "TLUSTY_test", "BOSZ_test" }, -3.0, 1.0, 0.0, 0.0,
            { 10.0 }, specsyn::defaultR, registryName);
        std::cerr << "testSpecsynLibChained: expected the constructor to throw "
            "for a mismatched microTurb size, but it did not\n";
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
    result += testChainOOBThrows();
    result += testChainConstructorValidation();
    return result;
}
