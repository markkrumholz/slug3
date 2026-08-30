/**
 * @file testNebular.hpp
 * @author Mark Krumholz
 * @brief Unit tests for the Nebular class.
 * @date 2026-08-30
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#ifndef TESTNEBULAR_HPP
#define TESTNEBULAR_HPP

#include "../src/io/SimControls.hpp"
#include "../src/nebular/Nebular.hpp"
#include "../src/phot/FilterIdeal.hpp"
#include "../src/utils/MiscUtils.hpp"
#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <toml.hpp>
#include <vector>

// tests/nebular/assets/nebular_test.h5 (built by
// data/tools/cloudy/make_nebular_test_fixture.py) tabulates, for track
// "MIST_test", [Fe/H] in {-0.5, 0.0, 0.5}, v/vcrit = 0, a single
// log(U) point exactly at nebular::defaultLogU (so no log(U)
// interpolation ever happens -- see the generator's own module
// docstring for why), cluster ages in {1e6, 3e6, 1e7, 3e7, 1e8} yr,
// and 3 lines at {4000, 6000, 9000} Angstrom labeled LINE1/LINE2/LINE3.
// Every continuum/line value is exactly linear in [Fe/H] (galaxy) or
// [Fe/H] and age (cluster), constant across wavelength, so any
// getGalaxy()/getCluster() call's expected line luminosities (and,
// away from a line's own deposit window, its expected continuum) can
// be computed directly from these same coefficients rather than
// needing real cloudy physics to check against.
constexpr std::string_view testNebularInputFile = "tests/nebular/assets/testNebular.in";

constexpr double testNebularCtmGal0 = 1.0e-20;
constexpr double testNebularLineGal0 = 1.0e-18;
constexpr double testNebularCtmClus0 = 1.0e-21;
constexpr double testNebularLineClus0 = 1.0e-19;
constexpr std::size_t testNebularNLine = 3;
constexpr double testNebularWlNativeMin = 500.0;
constexpr double testNebularWlNativeMax = 20000.0;

/**
 * @brief Unit test: Nebular constructs from the fixture without throwing
 * @returns 0 if the test passes, 1 if it fails
 */
inline auto testNebularConstruct() -> int
{
    try
    {
        const toml::table inputDeck = toml::parse_file(testNebularInputFile);
        const io::SimControls controls(inputDeck);
        if (controls.nebular() == nullptr)
        {
            std::cerr << "testNebular: construct: expected SimControls::nebular() "
                "to be non-null\n";
            return 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testNebular: construct test failed: " << error.what() << "\n";
        return 1;
    }
    return 0;
}

/**
 * @brief Unit test: Nebular::lineLabel()/lineWl() match the fixture exactly
 * @returns 0 if the test passes, 1 if it fails
 */
inline auto testNebularLineLabelWl() -> int
{
    try
    {
        const toml::table inputDeck = toml::parse_file(testNebularInputFile);
        const io::SimControls controls(inputDeck);
        const auto* neb = controls.nebular();

        const std::vector<std::string> expectedLabel{"LINE1", "LINE2", "LINE3"};
        const std::vector<double> expectedWl{4000.0, 6000.0, 9000.0};

        if (neb->lineLabel() != expectedLabel)
        {
            std::cerr << "testNebular: lineLabelWl: lineLabel() does not match "
                "the fixture's own line_label\n";
            return 1;
        }
        if (neb->lineWl().size() != expectedWl.size())
        {
            std::cerr << "testNebular: lineLabelWl: lineWl() size "
                << neb->lineWl().size() << " does not match expected "
                << expectedWl.size() << "\n";
            return 1;
        }
        for (std::size_t ell = 0; ell < expectedWl.size(); ++ell)
        {
            if (!utils::approxEqual(neb->lineWl().at(ell), expectedWl.at(ell)))
            {
                std::cerr << "testNebular: lineLabelWl: lineWl()[" << ell << "] = "
                    << neb->lineWl().at(ell) << ", expected " << expectedWl.at(ell) << "\n";
                return 1;
            }
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testNebular: lineLabelWl test failed: " << error.what() << "\n";
        return 1;
    }
    return 0;
}

/**
 * @brief Unit test: Nebular::getGalaxy() returns the expected line luminosities and continuum
 * @returns 0 if the test passes, 1 if it fails
 * @details
 * Uses a constant input spectrum (1 erg/s/Angstrom at every
 * wavelength) purely so Q(HI) -- and hence every expected output
 * below, which all scale linearly with it -- can be computed
 * independently via a second, directly-constructed
 * phot::FilterIdeal("Q(HI)") (the same public class Nebular itself
 * uses internally for this), without needing to reimplement its own
 * integration. Checks lineLum() exactly (a plain formula, with no
 * deposit-window geometry involved), and the returned spectrum's
 * continuum away from every line's own deposit window (near a line
 * center, depositLines() adds extra power there too, which this test
 * does not independently reproduce -- see computeLineDepositWindows()'s
 * own comment for why that geometry depends on the local wavelength
 * grid spacing).
 */
inline auto testNebularGetGalaxy() -> int
{
    try
    {
        const toml::table inputDeck = toml::parse_file(testNebularInputFile);
        const io::SimControls controls(inputDeck);
        const auto* neb = controls.nebular();
        const auto& wl = controls.specsyn()->wl();

        const std::vector<double> spec(wl.size(), 1.0);
        const phot::FilterIdeal qhiFilter("Q(HI)");
        const double covFac = controls.nebControls().covFac_;
        const double qhi = covFac * qhiFilter.phot(wl, spec);
        if (!(qhi > 0.0))
        {
            std::cerr << "testNebular: getGalaxy: test bug: expected qhi > 0, got "
                << qhi << "\n";
            return 1;
        }

        constexpr double feh = 0.0; // exact grid hit -- no [Fe/H] interpolation
        const auto [outSpec, lineLum] = neb->getGalaxy(spec, feh);

        if (lineLum.size() != testNebularNLine)
        {
            std::cerr << "testNebular: getGalaxy: lineLum() size " << lineLum.size()
                << " does not match expected " << testNebularNLine << "\n";
            return 1;
        }
        for (std::size_t ell = 0; ell < testNebularNLine; ++ell)
        {
            const double expected =
                qhi * testNebularLineGal0 * static_cast<double>(ell + 1) * (1.0 + feh);
            if (!utils::approxEqual(lineLum.at(ell), expected, 1e-4))
            {
                std::cerr << "testNebular: getGalaxy: lineLum()[" << ell << "] = "
                    << lineLum.at(ell) << ", expected " << expected << "\n";
                return 1;
            }
        }

        if (outSpec.size() != wl.size())
        {
            std::cerr << "testNebular: getGalaxy: output spectrum size "
                << outSpec.size() << " does not match wl size " << wl.size() << "\n";
            return 1;
        }

        const double edgeWl = qhiFilter.wlMax();
        const double expectedCtm = qhi * testNebularCtmGal0 * (1.0 + feh);
        constexpr double lineHalfWidthWl = 200.0; // stay well clear of every line's own deposit window
        const std::vector<double> lineCenters{4000.0, 6000.0, 9000.0};
        for (std::size_t k = 0; k < wl.size(); ++k)
        {
            bool nearLine = false;
            for (const double lineWl : lineCenters)
            {
                if (std::abs(wl.at(k) - lineWl) < lineHalfWidthWl) { nearLine = true; }
            }
            if (nearLine) { continue; }

            // The nebular continuum is added unconditionally across the
            // whole grid (within the fixture's own native wl range) --
            // stellarAboveEdge() only zeroes the *input stellar*
            // contribution at or below the edge, not the nebula's own
            // continuum there (real nebular continua, e.g. two-photon
            // or free-bound emission from species other than H, are not
            // confined to wavelengths above the H-ionization edge).
            const double stellarPart = (wl.at(k) > edgeWl) ? spec.at(k) : 0.0;
            const bool inNative =
                wl.at(k) >= testNebularWlNativeMin && wl.at(k) <= testNebularWlNativeMax;
            const double ctmPart = inNative ? expectedCtm : 0.0;
            const double expected = stellarPart + ctmPart;
            if (!utils::approxEqual(outSpec.at(k), expected, 1e-4))
            {
                std::cerr << "testNebular: getGalaxy: outSpec[" << k
                    << "] at wl = " << wl.at(k) << " = " << outSpec.at(k)
                    << ", expected " << expected << "\n";
                return 1;
            }
        }

        // Out-of-range [Fe/H] must throw
        bool threw = false;
        try { (void)neb->getGalaxy(spec, 10.0); }
        catch (const std::runtime_error&) { threw = true; }
        if (!threw)
        {
            std::cerr << "testNebular: getGalaxy: expected an out-of-range [Fe/H] "
                "to throw\n";
            return 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testNebular: getGalaxy test failed: " << error.what() << "\n";
        return 1;
    }
    return 0;
}

/**
 * @brief Unit test: Nebular::getCluster() at an exact [Fe/H]/age grid hit
 * @returns 0 if the test passes, 1 if it fails
 * @details
 * Mirrors testNebularGetGalaxy()'s own independent-Q(HI) approach.
 * age = 1e7 yr is one of clusterAge_'s own tabulated entries, [Fe/H] =
 * 0 is one of feH_'s own, so neither axis needs to interpolate here --
 * see testNebularGetClusterOffGrid() for that. Split out of
 * testNebularGetCluster() purely to keep that dispatcher's own
 * cognitive complexity down.
 */
inline auto testNebularGetClusterExactHit() -> int
{
    try
    {
        const toml::table inputDeck = toml::parse_file(testNebularInputFile);
        const io::SimControls controls(inputDeck);
        const auto* neb = controls.nebular();
        const auto& wl = controls.specsyn()->wl();

        const std::vector<double> spec(wl.size(), 1.0);
        const phot::FilterIdeal qhiFilter("Q(HI)");
        const double qhi = controls.nebControls().covFac_ * qhiFilter.phot(wl, spec);

        constexpr double feh = 0.0;
        constexpr double age = 1.0e7;
        const auto [outSpec, lineLum] = neb->getCluster(spec, feh, age);
        if (outSpec.size() != wl.size())
        {
            std::cerr << "testNebular: getClusterExactHit: output spectrum size "
                << outSpec.size() << " does not match wl size " << wl.size() << "\n";
            return 1;
        }
        for (std::size_t ell = 0; ell < testNebularNLine; ++ell)
        {
            const double expected = qhi * testNebularLineClus0 *
                static_cast<double>(ell + 1) * (1.0 + feh) * (age / 1.0e6);
            if (!utils::approxEqual(lineLum.at(ell), expected, 1e-4))
            {
                std::cerr << "testNebular: getClusterExactHit: lineLum()[" << ell
                    << "] = " << lineLum.at(ell) << ", expected " << expected << "\n";
                return 1;
            }
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testNebular: getClusterExactHit test failed: " << error.what() << "\n";
        return 1;
    }
    return 0;
}

/**
 * @brief Unit test: Nebular::getCluster() bilinearly interpolates off-grid in both [Fe/H] and age together
 * @returns 0 if the test passes, 1 if it fails
 * @details
 * feh = 0.25, age = 5e6 yr are both off their own tabulated grids
 * (see testNebular.hpp's own module comment for feH_/clusterAge_),
 * exercising the bilinear interpolation itself, not just a
 * single-axis bracket. Also checks the returned spectrum's continuum
 * away from every line's own deposit window -- see
 * testNebularGetGalaxy()'s own comment for why the continuum is added
 * everywhere in-range regardless of the H-ionization edge, while the
 * stellar contribution itself is kept only above it. Split out of
 * testNebularGetCluster() purely to keep that dispatcher's own
 * cognitive complexity down.
 */
inline auto testNebularGetClusterOffGrid() -> int
{
    try
    {
        const toml::table inputDeck = toml::parse_file(testNebularInputFile);
        const io::SimControls controls(inputDeck);
        const auto* neb = controls.nebular();
        const auto& wl = controls.specsyn()->wl();

        const std::vector<double> spec(wl.size(), 1.0);
        const phot::FilterIdeal qhiFilter("Q(HI)");
        const double qhi = controls.nebControls().covFac_ * qhiFilter.phot(wl, spec);

        constexpr double feh = 0.25;
        constexpr double age = 5.0e6;
        const auto [outSpec, lineLum] = neb->getCluster(spec, feh, age);
        for (std::size_t ell = 0; ell < testNebularNLine; ++ell)
        {
            const double expected = qhi * testNebularLineClus0 *
                static_cast<double>(ell + 1) * (1.0 + feh) * (age / 1.0e6);
            if (!utils::approxEqual(lineLum.at(ell), expected, 1e-4))
            {
                std::cerr << "testNebular: getClusterOffGrid: lineLum()[" << ell
                    << "] = " << lineLum.at(ell) << ", expected " << expected << "\n";
                return 1;
            }
        }

        const double edgeWl = qhiFilter.wlMax();
        const double expectedCtm =
            qhi * testNebularCtmClus0 * (1.0 + feh) * (age / 1.0e6);
        constexpr double lineHalfWidthWl = 200.0;
        const std::vector<double> lineCenters{4000.0, 6000.0, 9000.0};
        for (std::size_t k = 0; k < wl.size(); ++k)
        {
            bool nearLine = false;
            for (const double lineWl : lineCenters)
            {
                if (std::abs(wl.at(k) - lineWl) < lineHalfWidthWl) { nearLine = true; }
            }
            if (nearLine) { continue; }

            const double stellarPart = (wl.at(k) > edgeWl) ? spec.at(k) : 0.0;
            const bool inNative =
                wl.at(k) >= testNebularWlNativeMin && wl.at(k) <= testNebularWlNativeMax;
            const double ctmPart = inNative ? expectedCtm : 0.0;
            const double expected = stellarPart + ctmPart;
            if (!utils::approxEqual(outSpec.at(k), expected, 1e-4))
            {
                std::cerr << "testNebular: getClusterOffGrid: outSpec[" << k
                    << "] at wl = " << wl.at(k) << " = " << outSpec.at(k)
                    << ", expected " << expected << "\n";
                return 1;
            }
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testNebular: getClusterOffGrid test failed: " << error.what() << "\n";
        return 1;
    }
    return 0;
}

/**
 * @brief Unit test: Nebular::getCluster() falls back cleanly above the tabulated age range
 * @returns 0 if the test passes, 1 if it fails
 * @details
 * age = 1e9 yr exceeds clusterAge_.back() -- see getCluster()'s own
 * comment for the documented fallback this exercises: spec passed
 * through with only the H-ionization edge zeroed (no continuum or
 * line emission added at all), lineLum all zero. Split out of
 * testNebularGetCluster() purely to keep that dispatcher's own
 * cognitive complexity down.
 */
inline auto testNebularGetClusterAboveAgeRange() -> int
{
    try
    {
        const toml::table inputDeck = toml::parse_file(testNebularInputFile);
        const io::SimControls controls(inputDeck);
        const auto* neb = controls.nebular();
        const auto& wl = controls.specsyn()->wl();

        const std::vector<double> spec(wl.size(), 1.0);
        const phot::FilterIdeal qhiFilter("Q(HI)");

        constexpr double feh = 0.0;
        constexpr double veryOldAge = 1.0e9;
        const auto [outSpec, lineLum] = neb->getCluster(spec, feh, veryOldAge);
        for (const double lum : lineLum)
        {
            if (lum != 0.0)
            {
                std::cerr << "testNebular: getClusterAboveAgeRange: expected all-zero "
                    "lineLum above the tabulated age range, got " << lum << "\n";
                return 1;
            }
        }
        const double edgeWl = qhiFilter.wlMax();
        for (std::size_t k = 0; k < wl.size(); ++k)
        {
            const double expected = (wl.at(k) <= edgeWl) ? 0.0 : spec.at(k);
            if (!utils::approxEqual(outSpec.at(k), expected, 1e-6))
            {
                std::cerr << "testNebular: getClusterAboveAgeRange: outSpec[" << k
                    << "] = " << outSpec.at(k) << ", expected " << expected << "\n";
                return 1;
            }
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testNebular: getClusterAboveAgeRange test failed: "
            << error.what() << "\n";
        return 1;
    }
    return 0;
}

/**
 * @brief Unit test: Nebular::getCluster(), covering an exact grid hit, off-grid bilinear interpolation, and the above-age-range fallback
 * @returns 0 if every sub-test passes, 1 if any fails
 */
inline auto testNebularGetCluster() -> int
{
    int result = 0;
    result += testNebularGetClusterExactHit();
    result += testNebularGetClusterOffGrid();
    result += testNebularGetClusterAboveAgeRange();
    return result;
}

/**
 * @brief Run every Nebular unit test
 * @returns 0 if every test passes, 1 if any fails
 */
inline auto testNebular() -> int
{
    int result = 0;
    result += testNebularConstruct();
    result += testNebularLineLabelWl();
    result += testNebularGetGalaxy();
    result += testNebularGetCluster();
    return result;
}

#endif // TESTNEBULAR_HPP
