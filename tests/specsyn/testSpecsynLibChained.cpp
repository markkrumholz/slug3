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
 * It also tests SpecsynLibChained::makeCommonWlGrid directly, both
 * against a fully controlled synthetic scenario (so the window-by-
 * window point-count logic can be checked exactly) and against
 * BOSZ_test/TLUSTY_test's own native grids (to confirm the
 * constructor actually resamples every chained library onto a single
 * common grid, rather than each keeping its own).
 * @date 2026-07-21
 */

#include "../../src/specsyn/SpecsynLib.hpp"
#include "../../src/specsyn/SpecsynLibChained.hpp"
#include "../../src/tracks/TrackCommons.hpp"
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
        { 10.0, 0.0 }, specsyn::defaultR, registryName);

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
        { 0.0, 10.0 }, specsyn::defaultR, registryName);

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

// Check that SpecsynLibChained's constructor actually builds and uses
// a single common grid spanning both BOSZ_test's and TLUSTY_test's own
// native ranges, rather than just adopting one library's grid
// verbatim: the chain's wl() must span the full union of both native
// ranges, be strictly increasing, and have exactly as many points as
// calling makeCommonWlGrid directly on the two native grids.
static auto testChainUsesCommonGrid() -> int
{
    const specsyn::SpecsynLib<specsyn::OOBPolicy::Throw> boszRef(
        "BOSZ_test", -3.0, 1.0, 0.0, 0.0, 0.0, specsyn::defaultR, registryName);
    const specsyn::SpecsynLib<specsyn::OOBPolicy::Throw> tlustyRef(
        "TLUSTY_test", -3.0, 1.0, 0.0, 0.0, 10.0, specsyn::defaultR, registryName);

    const auto expected = specsyn::SpecsynLibChained::makeCommonWlGrid(
        { boszRef.wl(), tlustyRef.wl() });

    const specsyn::SpecsynLibChained chain(
        { "BOSZ_test", "TLUSTY_test" }, -3.0, 1.0, 0.0, 0.0,
        { 0.0, 10.0 }, specsyn::defaultR, registryName);

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
    result += testChainOOBThrows();
    result += testChainConstructorValidation();
    result += testChainUsesCommonGrid();
    result += testMakeCommonWlGridWindows();
    result += testMakeCommonWlGridGap();
    result += testMakeCommonWlGridErrors();
    return result;
}
