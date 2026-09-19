/**
 * @file testYieldChannel.hpp
 * @author Mark Krumholz
 * @brief Unit tests for the YieldChannel class.
 * @details
 * This file contains unit tests for the YieldChannel class, which
 * reads nucleosynthetic yield data for one channel from an HDF5 file.
 * Like the Tracks3D tests, these act only on a small, reduced test
 * fixture in tests/yields/assets (see
 * data/tools/yields/make_yields_test_fixture.py), extracted directly
 * from data/yields/sukhbold16.h5 -- the expected values checked here
 * were independently verified by hand against the original
 * s18.2.yield_table/s100.yield_table text files (see
 * import_yield_tables.py's own docstring for that source format). The
 * mass-grid extrapolation/interpolation tests further down instead use
 * the fixture's own "kobayashi_test" model (extracted from the real
 * data/yields/kobayashi06_11.h5, which -- unlike sukhbold_test's 2-mass
 * grid -- has 3 masses, enough to exercise "some requested masses are
 * native grid points, some are interpolated between two of them, some
 * are extrapolated beyond either end" all in the same YieldChannel).
 *
 * YieldChannel::YieldChannel() no longer builds masses()/isotopes()/
 * yield() itself -- every test below must call rebuildYieldGrid()
 * explicitly (see its own comment) before using any of those; a
 * dedicated test (testYieldChannelYieldBeforeRebuild()) checks that
 * calling yield() before doing so throws, rather than silently
 * misbehaving.
 * @date 2026-09-12
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#ifndef TESTYIELDCHANNEL_HPP
#define TESTYIELDCHANNEL_HPP

#include "../../src/elem/IsotopeTable.hpp"
#include "../../src/yields/YieldChannel.hpp"
#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace
{
    constexpr double tol = 1e-10;

    /**
     * @brief Check one (mass, isotope) entry of a YieldChannel's yield data
     * @return 0 if the test passes, 1 if it fails
     */
    auto checkYield(const yields::YieldChannel& yc, const std::string& label,
        const std::size_t massIdx, const std::size_t isoIdx, const double expected) -> int
    {
        const double actual = yc.yld()[0, massIdx, isoIdx];
        if (std::abs(actual - expected) > tol)
        {
            std::cerr << label << ": expected " << expected <<
                " at (mass index " << massIdx << ", isotope index " <<
                isoIdx << "), got " << actual << "\n";
            return 1;
        }
        return 0;
    }

    /**
     * @brief Check one yield() result vector against expected values
     * @return 0 if the test passes, 1 if it fails
     * @details
     * Shared by every test below that checks yield()'s own return
     * value; rejects a non-finite actual value explicitly, since
     * std::abs(NaN - expected) > tol is false, which would otherwise
     * let a NaN regression pass silently.
     */
    auto checkVec(const std::string& label, const std::vector<double>& actual,
        const std::vector<double>& expected) -> int
    {
        if (actual.size() != expected.size())
        {
            std::cerr << label << ": expected a vector of size " << expected.size() <<
                ", got " << actual.size() << "\n";
            return 1;
        }
        int result = 0;
        for (std::size_t i = 0; i < expected.size(); ++i)
        {
            if (!std::isfinite(actual[i]) || std::abs(actual[i] - expected[i]) > tol)
            {
                std::cerr << label << ": isotope index " << i << ": expected " <<
                    expected[i] << ", got " << actual[i] << "\n";
                result = 1;
            }
        }
        return result;
    }
} // namespace

/**
 * @brief Unit test for YieldChannel's ccsn (ejecta) values
 * @return 0 if the test passes, 1 if it fails
 * @details
 * tests/yields/assets/yields.toml's "sukhbold_test" model holds
 * masses [18.2, 100.0] and isotopes [h1 (Z=1,A=1), fe56 (Z=26,A=56),
 * ni56 (Z=28,A=56)] at Fe_H = 0.0. Mass 100.0 is a failed supernova
 * (no [ejecta] column in the original source file -- see
 * import_yield_tables.py's own comment), so every isotope's own ccsn
 * yield there must be exactly zero, not merely small. Also checks that
 * isotopesOrig() -- unlike masses()/isotopes()/yield() -- is already
 * populated right after construction, with no rebuildYieldGrid() call
 * needed first.
 */
inline auto testYieldChannelCcsn() -> int
{
    const std::string registryName = "tests/yields/assets/yields.toml";
    int result = 0;

    try
    {
        yields::YieldChannel yc(
            yields::YieldChannelDescriptor{ yields::Channel::ccsn_, "sukhbold_test" },
            0.0, 0.0, registryName);

        if (yc.channel() != yields::Channel::ccsn_)
        {
            std::cerr << "testYieldChannelCcsn: channel() does not match constructor argument\n";
            result = 1;
        }

        // isotopesOrig() is populated by the constructor itself -- no
        // rebuildYieldGrid() call needed first, unlike isotopes() below
        if (yc.isotopesOrig().size() != 3 ||
            yc.isotopesOrig()[0].get().Z() != 1 || yc.isotopesOrig()[0].get().A() != 1 ||
            yc.isotopesOrig()[1].get().Z() != 26 || yc.isotopesOrig()[1].get().A() != 56 ||
            yc.isotopesOrig()[2].get().Z() != 28 || yc.isotopesOrig()[2].get().A() != 56)
        {
            std::cerr << "testYieldChannelCcsn: unexpected isotopesOrig()\n";
            result = 1;
        }

        yc.rebuildYieldGrid();

        if (yc.masses().size() != 2 || yc.masses()[0] != 18.2 || yc.masses()[1] != 100.0)
        {
            std::cerr << "testYieldChannelCcsn: unexpected masses()\n";
            result = 1;
        }
        if (yc.isotopes().size() != 3 ||
            yc.isotopes()[0].get().Z() != 1 || yc.isotopes()[0].get().A() != 1 ||
            yc.isotopes()[1].get().Z() != 26 || yc.isotopes()[1].get().A() != 56 ||
            yc.isotopes()[2].get().Z() != 28 || yc.isotopes()[2].get().A() != 56)
        {
            std::cerr << "testYieldChannelCcsn: unexpected isotopes()\n";
            result = 1;
        }
        if (yc.feH().size() != 1 || yc.feH()[0] != 0.0)
        {
            std::cerr << "testYieldChannelCcsn: unexpected feH()\n";
            result = 1;
        }

        // mass = 18.2 (index 0): h1, fe56, ni56 ejecta
        result += checkYield(yc, "testYieldChannelCcsn", 0, 0, 5.93);
        result += checkYield(yc, "testYieldChannelCcsn", 0, 1, 8.46e-2);
        result += checkYield(yc, "testYieldChannelCcsn", 0, 2, 7.02e-2);
        // mass = 100.0 (index 1), a failed supernova: every ccsn yield is exactly zero
        result += checkYield(yc, "testYieldChannelCcsn", 1, 0, 0.0);
        result += checkYield(yc, "testYieldChannelCcsn", 1, 1, 0.0);
        result += checkYield(yc, "testYieldChannelCcsn", 1, 2, 0.0);
    }
    catch (const std::exception& e)
    {
        std::cerr << "testYieldChannelCcsn: failed to construct YieldChannel from "
            << registryName << ": " << e.what() << "\n";
        return 1;
    }

    return result;
}

/**
 * @brief Unit test for YieldChannel's massive_star_winds values
 * @return 0 if the test passes, 1 if it fails
 * @details
 * Same fixture as testYieldChannelCcsn(), but the wind channel: unlike
 * ccsn, mass 100.0 (a failed supernova) still has real, nonzero wind
 * yields, since a failed supernova still loses mass via winds before
 * collapsing.
 */
inline auto testYieldChannelMassiveStarWinds() -> int
{
    const std::string registryName = "tests/yields/assets/yields.toml";
    int result = 0;

    try
    {
        yields::YieldChannel yc(
            yields::YieldChannelDescriptor{ yields::Channel::massiveStarWinds_, "sukhbold_test" },
            0.0, 0.0, registryName);
        yc.rebuildYieldGrid();

        if (yc.channel() != yields::Channel::massiveStarWinds_)
        {
            std::cerr << "testYieldChannelMassiveStarWinds: channel() does not "
                "match constructor argument\n";
            result = 1;
        }

        // mass = 18.2 (index 0): h1, fe56, ni56 wind yields
        result += checkYield(yc, "testYieldChannelMassiveStarWinds", 0, 0, 2.21);
        result += checkYield(yc, "testYieldChannelMassiveStarWinds", 0, 1, 4.00e-3);
        result += checkYield(yc, "testYieldChannelMassiveStarWinds", 0, 2, 0.0);
        // mass = 100.0 (index 1), a failed supernova: wind yields are still real
        result += checkYield(yc, "testYieldChannelMassiveStarWinds", 1, 0, 30.6);
        result += checkYield(yc, "testYieldChannelMassiveStarWinds", 1, 1, 1.16e-1);
        result += checkYield(yc, "testYieldChannelMassiveStarWinds", 1, 2, 0.0);
    }
    catch (const std::exception& e)
    {
        std::cerr << "testYieldChannelMassiveStarWinds: failed to construct "
            "YieldChannel from " << registryName << ": " << e.what() << "\n";
        return 1;
    }

    return result;
}

/**
 * @brief Unit test that an out-of-range [Fe/H] request is rejected
 * @return 0 if the test passes, 1 if it fails
 * @details
 * tests/yields/assets/yields.toml's "sukhbold_test" model provides
 * Fe_H = -1.0 (synthetic -- see make_yields_test_fixture.py's own
 * docstring) and 0.0 (real). This checks that requesting a fehMin
 * below -1.0 or a fehMax above 0.0 throws, mirroring
 * testTracks3DFeHRangeGuard()'s identical check for Tracks3D. Thrown
 * by the constructor itself, before rebuildYieldGrid() would even be
 * relevant, so this test needs no such call.
 */
inline auto testYieldChannelFeHRangeGuard() -> int
{
    const std::string registryName = "tests/yields/assets/yields.toml";
    int result = 0;

    try
    {
        const yields::YieldChannel yc(
            yields::YieldChannelDescriptor{ yields::Channel::ccsn_, "sukhbold_test" },
            -1.1, 0.0, registryName);
        std::cerr << "testYieldChannelFeHRangeGuard: construction with "
            "fehMin = -1.1 (below the available minimum of -1.0) "
            "should have thrown, but did not\n";
        result = 1;
    }
    catch (const std::exception&) { /* expected */ }

    try
    {
        const yields::YieldChannel yc(
            yields::YieldChannelDescriptor{ yields::Channel::ccsn_, "sukhbold_test" },
            -1.0, 0.1, registryName);
        std::cerr << "testYieldChannelFeHRangeGuard: construction with "
            "fehMax = 0.1 (above the available maximum of 0.0) "
            "should have thrown, but did not\n";
        result = 1;
    }
    catch (const std::exception&) { /* expected */ }

    return result;
}

/**
 * @brief Unit test that an unknown model name is rejected
 * @return 0 if the test passes, 1 if it fails
 */
inline auto testYieldChannelUnknownModel() -> int
{
    const std::string registryName = "tests/yields/assets/yields.toml";

    try
    {
        const yields::YieldChannel yc(
            yields::YieldChannelDescriptor{ yields::Channel::ccsn_, "no_such_model" },
            0.0, 0.0, registryName);
        std::cerr << "testYieldChannelUnknownModel: construction with an "
            "unknown model name should have thrown, but did not\n";
        return 1;
    }
    catch (const std::exception&) { return 0; /* expected */ }
}

/**
 * @brief Unit test that yield() throws before rebuildYieldGrid() has ever been called
 * @return 0 if the test passes, 1 if it fails
 * @details
 * YieldChannel::YieldChannel() deliberately leaves masses_/isotopes_/
 * yieldData_ empty (see its own comment) -- this checks the guard
 * yield() uses to report that clearly (std::runtime_error) rather than
 * asserting (debug-only) or reading past the end of an empty
 * yieldData_ (release builds). hasYield()/masses() are not similarly
 * guarded (see hasYield()'s own comment), so this only checks yield().
 */
inline auto testYieldChannelYieldBeforeRebuild() -> int
{
    const std::string registryName = "tests/yields/assets/yields.toml";

    try
    {
        const yields::YieldChannel yc(
            yields::YieldChannelDescriptor{ yields::Channel::ccsn_, "sukhbold_test" },
            0.0, 0.0, registryName);
        if (!yc.masses().empty())
        {
            std::cerr << "testYieldChannelYieldBeforeRebuild: expected masses() empty "
                "before rebuildYieldGrid() is ever called\n";
            return 1;
        }

        try
        {
            (void)yc.yield(18.2, 0.0);
        }
        catch (const std::runtime_error&) { return 0; /* expected */ }
        std::cerr << "testYieldChannelYieldBeforeRebuild: expected yield() to throw "
            "before rebuildYieldGrid() is ever called, but it did not\n";
        return 1;
    }
    catch (const std::exception& e)
    {
        std::cerr << "testYieldChannelYieldBeforeRebuild: failed to construct "
            "YieldChannel from " << registryName << ": " << e.what() << "\n";
        return 1;
    }
}

/**
 * @brief Unit test for YieldChannel::hasYield()
 * @return 0 if the test passes, 1 if it fails
 * @details
 * tests/yields/assets/yields.toml's "sukhbold_test" model holds masses
 * [18.2, 100.0]; checks both endpoints (inclusive), a mass strictly
 * between them, and masses strictly below/above the range.
 */
inline auto testYieldChannelHasYield() -> int
{
    const std::string registryName = "tests/yields/assets/yields.toml";
    int result = 0;

    try
    {
        yields::YieldChannel yc(
            yields::YieldChannelDescriptor{ yields::Channel::ccsn_, "sukhbold_test" },
            0.0, 0.0, registryName);
        yc.rebuildYieldGrid();

        const std::vector<std::pair<double, bool>> cases{
            { 18.2, true }, { 100.0, true }, { 59.1, true },
            { 18.199, false }, { 100.001, false }, { 5.0, false }, { 500.0, false },
        };
        for (const auto& [mass, expected] : cases)
        {
            if (yc.hasYield(mass) != expected)
            {
                std::cerr << "testYieldChannelHasYield: hasYield(" << mass <<
                    ") returned " << yc.hasYield(mass) << ", expected " << expected << "\n";
                result = 1;
            }
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "testYieldChannelHasYield: failed to construct YieldChannel from "
            << registryName << ": " << e.what() << "\n";
        return 1;
    }

    return result;
}

/**
 * @brief Unit test for YieldChannel::yield()'s interpolation
 * @return 0 if the test passes, 1 if it fails
 * @details
 * Checks three things against tests/yields/assets/yields.toml's
 * "sukhbold_test" model (masses [18.2, 100.0], a single Fe_H = 0.0):
 *
 * - yield() at an exact grid mass (18.2) reproduces the same values
 *   checkYield() already verified directly against yld() in
 *   testYieldChannelCcsn()/testYieldChannelMassiveStarWinds(), showing
 *   the bracket-and-interpolate machinery collapses correctly to an
 *   exact hit rather than perturbing it.
 * - yield() at mass = 59.1, the exact arithmetic midpoint of [18.2,
 *   100.0], gives exactly the unweighted average of the two grid
 *   masses' own values, for both the ccsn (where mass 100.0 is a
 *   failed supernova with an all-zero yield) and massive_star_winds
 *   channels.
 * - Every one of the checks above also exercises feH_'s own singular
 *   (size-1) axis, at Fe_H = 0.0 -- the exact scenario the real
 *   Sukhbold et al. (2016) data itself presents, Solar-only --
 *   confirming utils::findBracket's own degenerate-axis handling
 *   (lo_ == hi_, t_ == 0) needs no special-casing in yield() itself.
 * - A separate YieldChannel, loaded over [-1.0, 0.0], covers
 *   interpolation *across* Fe_H groups instead: the real data alone
 *   can't exercise this (it is Solar-only), so
 *   make_yields_test_fixture.py adds one synthetic Fe_H = -1.0 group
 *   whose yield array is exactly 2x the real Fe_H = 0.0 array (see its
 *   own docstring) -- making the Fe_H = -0.5 midpoint exactly 1.5x the
 *   real value, at a fixed, exact-grid mass.
 */
inline auto testYieldChannelInterpolation() -> int
{
    const std::string registryName = "tests/yields/assets/yields.toml";
    int result = 0;

    try
    {
        yields::YieldChannel ccsn(
            yields::YieldChannelDescriptor{ yields::Channel::ccsn_, "sukhbold_test" },
            0.0, 0.0, registryName);
        ccsn.rebuildYieldGrid();
        yields::YieldChannel wind(
            yields::YieldChannelDescriptor{ yields::Channel::massiveStarWinds_, "sukhbold_test" },
            0.0, 0.0, registryName);
        wind.rebuildYieldGrid();

        // Exact grid hit at mass = 18.2 -- same values as
        // testYieldChannelCcsn()/testYieldChannelMassiveStarWinds()
        result += checkVec("testYieldChannelInterpolation (ccsn, exact 18.2)",
            ccsn.yield(18.2, 0.0), { 5.93, 8.46e-2, 7.02e-2 });
        result += checkVec("testYieldChannelInterpolation (winds, exact 18.2)",
            wind.yield(18.2, 0.0), { 2.21, 4.00e-3, 0.0 });

        // Midpoint mass = 59.1 -- exactly halfway between the two grid
        // masses, so the interpolated result is exactly their average
        result += checkVec("testYieldChannelInterpolation (ccsn, midpoint)",
            ccsn.yield(59.1, 0.0), { 2.965, 4.23e-2, 3.51e-2 });
        result += checkVec("testYieldChannelInterpolation (winds, midpoint)",
            wind.yield(59.1, 0.0), { 16.405, 6.0e-2, 0.0 });

        // Multi-metallicity interpolation, across Fe_H groups rather
        // than across masses -- see this function's own comment on
        // why a synthetic Fe_H = -1.0 group is needed for this
        yields::YieldChannel ccsnMultiFeH(
            yields::YieldChannelDescriptor{ yields::Channel::ccsn_, "sukhbold_test" },
            -1.0, 0.0, registryName);
        ccsnMultiFeH.rebuildYieldGrid();
        yields::YieldChannel windMultiFeH(
            yields::YieldChannelDescriptor{ yields::Channel::massiveStarWinds_, "sukhbold_test" },
            -1.0, 0.0, registryName);
        windMultiFeH.rebuildYieldGrid();

        if (ccsnMultiFeH.feH().size() != 2 || ccsnMultiFeH.feH()[0] != -1.0 ||
            ccsnMultiFeH.feH()[1] != 0.0)
        {
            std::cerr << "testYieldChannelInterpolation: unexpected feH() on the "
                "[-1.0, 0.0] channel\n";
            result = 1;
        }

        // Exact hit on the synthetic Fe_H = -1.0 group: exactly 2x the
        // real Fe_H = 0.0 values, at the exact grid mass 18.2
        result += checkVec("testYieldChannelInterpolation (ccsn, exact Fe_H=-1.0)",
            ccsnMultiFeH.yield(18.2, -1.0), { 11.86, 1.692e-1, 1.404e-1 });
        result += checkVec("testYieldChannelInterpolation (winds, exact Fe_H=-1.0)",
            windMultiFeH.yield(18.2, -1.0), { 4.42, 8.00e-3, 0.0 });

        // Fe_H = -0.5, the exact arithmetic midpoint of [-1.0, 0.0]:
        // exactly 1.5x the real Fe_H = 0.0 value, at the exact grid mass 18.2
        result += checkVec("testYieldChannelInterpolation (ccsn, Fe_H midpoint)",
            ccsnMultiFeH.yield(18.2, -0.5), { 8.895, 1.269e-1, 1.053e-1 });
        result += checkVec("testYieldChannelInterpolation (winds, Fe_H midpoint)",
            windMultiFeH.yield(18.2, -0.5), { 3.315, 6.00e-3, 0.0 });
    }
    catch (const std::exception& e)
    {
        std::cerr << "testYieldChannelInterpolation: failed to construct YieldChannel from "
            << registryName << ": " << e.what() << "\n";
        return 1;
    }

    return result;
}

/**
 * @brief Unit test for YieldChannel's mass-grid extrapolation and interpolation
 * @return 0 if the test passes, 1 if it fails
 * @details
 * tests/yields/assets/yields.toml's "kobayashi_test" model (ccsn only)
 * holds masses [13.0, 15.0, 18.0] and isotopes [h1, fe56, ni58] at
 * Fe_H = 0.0 -- see make_yields_test_fixture.py's own comment for why
 * this fixture, rather than sukhbold_test's own 2-mass one, is used
 * here. rebuildYieldGrid(8.0, 25.0) is called with mMin = 8.0,
 * mMax = 25.0 (both outside massesOrig()'s own [13.0, 18.0] range), so
 * masses() becomes exactly [8.0, 13.0, 15.0, 18.0, 25.0] --
 * massesOrig()'s own 3 points, unchanged, plus the two new extrapolated
 * endpoints. This exercises all three of rebuildYieldGrid()'s own mass-
 * axis rules at once:
 * - masses() entries 13.0/15.0/18.0 are exact massesOrig() hits, so
 *   yield() there must reproduce the real, hand-verified native values
 *   exactly (rule 1);
 * - mass = 14.0, strictly between two massesOrig() values, is linearly
 *   interpolated between them (rule 2);
 * - mass = 8.0 (below massesOrig().front() = 13.0) and mass = 25.0
 *   (above massesOrig().back() = 18.0) are each extrapolated by
 *   scaling the nearest native column by the ratio of the requested
 *   mass to that column's own mass (rule 3) -- e.g. every isotope's
 *   13.0 Msun yield, times 8.0/13.0, for mass = 8.0.
 */
inline auto testYieldChannelMassGridExtrapolation() -> int
{
    const std::string registryName = "tests/yields/assets/yields.toml";
    int result = 0;

    try
    {
        yields::YieldChannel yc(
            yields::YieldChannelDescriptor{ yields::Channel::ccsn_, "kobayashi_test" },
            0.0, 0.0, registryName);
        yc.rebuildYieldGrid(8.0, 25.0);

        if (yc.massesOrig().size() != 3 || yc.massesOrig()[0] != 13.0 ||
            yc.massesOrig()[1] != 15.0 || yc.massesOrig()[2] != 18.0)
        {
            std::cerr << "testYieldChannelMassGridExtrapolation: unexpected massesOrig()\n";
            result = 1;
        }
        const std::vector<double> expectedMasses{ 8.0, 13.0, 15.0, 18.0, 25.0 };
        if (yc.masses() != expectedMasses)
        {
            std::cerr << "testYieldChannelMassGridExtrapolation: unexpected masses()\n";
            result = 1;
        }

        // kobayashi_test's own isotopes are h1, fe56, ni58 -- distinct
        // from sukhbold_test's h1, fe56, ni56 (testYieldChannelCcsn's
        // own check), confirming isotopes() resolves the (Z, A) pairs
        // this specific model's file actually holds, not some other
        // model's
        if (yc.isotopes().size() != 3 ||
            yc.isotopes()[0].get().Z() != 1 || yc.isotopes()[0].get().A() != 1 ||
            yc.isotopes()[1].get().Z() != 26 || yc.isotopes()[1].get().A() != 56 ||
            yc.isotopes()[2].get().Z() != 28 || yc.isotopes()[2].get().A() != 58)
        {
            std::cerr << "testYieldChannelMassGridExtrapolation: unexpected isotopes()\n";
            result = 1;
        }

        // Exact massesOrig() hits: unchanged from the real, native values
        result += checkVec("testYieldChannelMassGridExtrapolation (exact 13.0)",
            yc.yield(13.0, 0.0), { 6.16, 8.32e-2, 2.23e-3 });
        result += checkVec("testYieldChannelMassGridExtrapolation (exact 15.0)",
            yc.yield(15.0, 0.0), { 6.79, 8.52e-2, 1.15e-3 });
        result += checkVec("testYieldChannelMassGridExtrapolation (exact 18.0)",
            yc.yield(18.0, 0.0), { 7.53, 8.72e-2, 2.70e-3 });

        // Interior interpolation, strictly between two native masses
        result += checkVec("testYieldChannelMassGridExtrapolation (interpolated 14.0)",
            yc.yield(14.0, 0.0), { 6.475, 8.42e-2, 1.69e-3 });

        // Extrapolation below/above massesOrig()'s own range
        result += checkVec("testYieldChannelMassGridExtrapolation (extrapolated 8.0)",
            yc.yield(8.0, 0.0), { 3.790769230769231, 5.12e-2, 1.3723076923076924e-3 });
        result += checkVec("testYieldChannelMassGridExtrapolation (extrapolated 25.0)",
            yc.yield(25.0, 0.0), { 10.458333333333334, 1.211111111111111e-1, 3.75e-3 });
    }
    catch (const std::exception& e)
    {
        std::cerr << "testYieldChannelMassGridExtrapolation: failed to construct "
            "YieldChannel from " << registryName << ": " << e.what() << "\n";
        return 1;
    }

    return result;
}

/**
 * @brief Unit test that mMin/mMax can also narrow masses() to a sub-range of massesOrig()
 * @return 0 if the test passes, 1 if it fails
 * @details
 * rebuildYieldGrid(14.0, 17.0): both bounds lie strictly inside
 * massesOrig()'s own [13.0, 18.0] range, so masses() becomes
 * [14.0, 15.0, 17.0]: the one massesOrig() value inside (14.0, 17.0) --
 * 15.0 -- kept unchanged, plus the two new endpoints, each interpolated
 * (not extrapolated, since both lie inside massesOrig()'s own range).
 * hasYield() no longer accepts 13.0 or 18.0 once the range has been
 * narrowed away from them this way.
 */
inline auto testYieldChannelMassGridNarrowing() -> int
{
    const std::string registryName = "tests/yields/assets/yields.toml";
    int result = 0;

    try
    {
        yields::YieldChannel yc(
            yields::YieldChannelDescriptor{ yields::Channel::ccsn_, "kobayashi_test" },
            0.0, 0.0, registryName);
        yc.rebuildYieldGrid(14.0, 17.0);

        const std::vector<double> expectedMasses{ 14.0, 15.0, 17.0 };
        if (yc.masses() != expectedMasses)
        {
            std::cerr << "testYieldChannelMassGridNarrowing: unexpected masses()\n";
            result = 1;
        }
        if (yc.hasYield(13.0) || yc.hasYield(18.0))
        {
            std::cerr << "testYieldChannelMassGridNarrowing: hasYield() should reject "
                "masses outside the narrowed [14.0, 17.0] range\n";
            result = 1;
        }

        result += checkVec("testYieldChannelMassGridNarrowing (interpolated 14.0)",
            yc.yield(14.0, 0.0), { 6.475, 8.42e-2, 1.69e-3 });
        result += checkVec("testYieldChannelMassGridNarrowing (exact 15.0)",
            yc.yield(15.0, 0.0), { 6.79, 8.52e-2, 1.15e-3 });
        result += checkVec("testYieldChannelMassGridNarrowing (interpolated 17.0)",
            yc.yield(17.0, 0.0), { 7.283333333333333, 8.653333333333332e-2, 2.183333333333333e-3 });
    }
    catch (const std::exception& e)
    {
        std::cerr << "testYieldChannelMassGridNarrowing: failed to construct "
            "YieldChannel from " << registryName << ": " << e.what() << "\n";
        return 1;
    }

    return result;
}

/**
 * @brief Unit test for YieldChannel::rebuildYieldGrid() called repeatedly, after construction
 * @return 0 if the test passes, 1 if it fails
 * @details
 * Constructs a "kobayashi_test" channel, checks that masses() starts
 * out empty (see testYieldChannelYieldBeforeRebuild() for the
 * analogous yield() check), then calls rebuildYieldGrid() with no
 * arguments (masses() becomes massesOrig() itself), then
 * rebuildYieldGrid(8.0, 25.0) (masses() becomes identical to
 * testYieldChannelMassGridExtrapolation()'s own equivalent), and
 * finally checks that rebuildYieldGrid(20.0, 10.0) (mMin >= mMax)
 * throws std::invalid_argument without disturbing the grid from the
 * last successful call.
 */
inline auto testYieldChannelRebuildYieldGrid() -> int
{
    const std::string registryName = "tests/yields/assets/yields.toml";
    int result = 0;

    try
    {
        yields::YieldChannel yc(
            yields::YieldChannelDescriptor{ yields::Channel::ccsn_, "kobayashi_test" },
            0.0, 0.0, registryName);

        if (!yc.masses().empty())
        {
            std::cerr << "testYieldChannelRebuildYieldGrid: expected masses() empty "
                "before rebuildYieldGrid() is ever called\n";
            result = 1;
        }

        const std::vector<double> nativeMasses{ 13.0, 15.0, 18.0 };
        yc.rebuildYieldGrid();
        if (yc.masses() != nativeMasses)
        {
            std::cerr << "testYieldChannelRebuildYieldGrid: expected masses() == "
                "massesOrig() after rebuildYieldGrid() with no arguments\n";
            result = 1;
        }

        yc.rebuildYieldGrid(8.0, 25.0);
        const std::vector<double> expectedMasses{ 8.0, 13.0, 15.0, 18.0, 25.0 };
        if (yc.masses() != expectedMasses)
        {
            std::cerr << "testYieldChannelRebuildYieldGrid: unexpected masses() "
                "after rebuildYieldGrid(8.0, 25.0)\n";
            result = 1;
        }
        if (yc.massesOrig() != nativeMasses)
        {
            std::cerr << "testYieldChannelRebuildYieldGrid: massesOrig() must not "
                "change when rebuildYieldGrid() is called\n";
            result = 1;
        }

        bool threw = false;
        try { yc.rebuildYieldGrid(20.0, 10.0); }
        catch (const std::invalid_argument&) { threw = true; }
        if (!threw)
        {
            std::cerr << "testYieldChannelRebuildYieldGrid: rebuildYieldGrid(20.0, 10.0) "
                "(mMin >= mMax) should have thrown std::invalid_argument, but did not\n";
            result = 1;
        }
        // The failed call above must not have disturbed the grid from
        // the last successful rebuildYieldGrid(8.0, 25.0) call
        if (yc.masses() != expectedMasses)
        {
            std::cerr << "testYieldChannelRebuildYieldGrid: masses() changed after a "
                "rebuildYieldGrid() call that threw\n";
            result = 1;
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "testYieldChannelRebuildYieldGrid: failed to construct "
            "YieldChannel from " << registryName << ": " << e.what() << "\n";
        return 1;
    }

    return result;
}

/**
 * @brief Unit test for rebuildYieldGrid()'s own isotope-remapping argument
 * @return 0 if the test passes, 1 if it fails
 * @details
 * "kobayashi_test"'s own native isotopes are h1 (Z=1,A=1), fe56
 * (Z=26,A=56), ni58 (Z=28,A=58) (see testYieldChannelMassGridExtrapolation()'s
 * own check) -- this passes rebuildYieldGrid() a hand-built isotope
 * list of [fe56, ni56 (Z=28,A=56), h1], in that order, exercising all
 * three things a caller-supplied isotope list can do at once, relative
 * to isotopesOrig():
 * - reordering: fe56 first and h1 last, the reverse of isotopesOrig()'s
 *   own [h1, fe56, ni58] order -- isotopes() and yld()'s own third axis
 *   must follow the requested order, not isotopesOrig()'s;
 * - a "foreign" isotope not in isotopesOrig() at all: ni56, present in
 *   sukhbold_test but not kobayashi_test, must appear in isotopes()
 *   (so every YieldChannel synchronized this way exposes the same
 *   isotope list) with an all-zero yield rather than being dropped or
 *   causing an error;
 * - subsetting: ni58, one of kobayashi_test's own 3 native isotopes,
 *   is not requested at all, and so must not appear in isotopes() or
 *   yld() -- confirming a caller-supplied list, not isotopesOrig(),
 *   fully determines isotopes() once given.
 *
 * The mass axis is left at its default (no mMin/mMax override) to
 * isolate the isotope-remapping behavior from mass extrapolation/
 * interpolation, already covered by
 * testYieldChannelMassGridExtrapolation()/Narrowing().
 */
inline auto testYieldChannelIsotopeRemap() -> int
{
    const std::string registryName = "tests/yields/assets/yields.toml";
    int result = 0;

    try
    {
        yields::YieldChannel yc(
            yields::YieldChannelDescriptor{ yields::Channel::ccsn_, "kobayashi_test" },
            0.0, 0.0, registryName);

        const elem::IsotopeList target{
            elem::isotopeTable(26, 56), // fe56 -- native to kobayashi_test
            elem::isotopeTable(28, 56), // ni56 -- NOT native to kobayashi_test
            elem::isotopeTable(1, 1),   // h1 -- native to kobayashi_test
        };
        yc.rebuildYieldGrid(std::nullopt, std::nullopt, target);

        // masses() is unaffected by the isotope remap: still exactly massesOrig()
        const std::vector<double> nativeMasses{ 13.0, 15.0, 18.0 };
        if (yc.masses() != nativeMasses)
        {
            std::cerr << "testYieldChannelIsotopeRemap: masses() should be unaffected "
                "by rebuildYieldGrid()'s own isotopes argument\n";
            result = 1;
        }

        if (yc.isotopes().size() != 3 ||
            yc.isotopes()[0].get().Z() != 26 || yc.isotopes()[0].get().A() != 56 ||
            yc.isotopes()[1].get().Z() != 28 || yc.isotopes()[1].get().A() != 56 ||
            yc.isotopes()[2].get().Z() != 1 || yc.isotopes()[2].get().A() != 1)
        {
            std::cerr << "testYieldChannelIsotopeRemap: isotopes() does not match "
                "the requested [fe56, ni56, h1] order\n";
            result = 1;
        }

        // At the exact grid mass 13.0: fe56 = 8.32e-2 and h1 = 6.16
        // (the same real values testYieldChannelMassGridExtrapolation()
        // checks, just reordered), and ni56 = 0.0 (not tabulated by
        // kobayashi_test at all)
        result += checkVec("testYieldChannelIsotopeRemap (exact 13.0)",
            yc.yield(13.0, 0.0), { 8.32e-2, 0.0, 6.16 });

        // isotopesOrig() must remain kobayashi_test's own native list,
        // unaffected by the isotope remap -- only isotopes() changes
        if (yc.isotopesOrig().size() != 3 ||
            yc.isotopesOrig()[0].get().Z() != 1 || yc.isotopesOrig()[0].get().A() != 1 ||
            yc.isotopesOrig()[1].get().Z() != 26 || yc.isotopesOrig()[1].get().A() != 56 ||
            yc.isotopesOrig()[2].get().Z() != 28 || yc.isotopesOrig()[2].get().A() != 58)
        {
            std::cerr << "testYieldChannelIsotopeRemap: isotopesOrig() must not "
                "change when rebuildYieldGrid() is given an isotopes argument\n";
            result = 1;
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "testYieldChannelIsotopeRemap: failed to construct "
            "YieldChannel from " << registryName << ": " << e.what() << "\n";
        return 1;
    }

    return result;
}

// YieldChannel is deliberately neither copyable nor movable -- see its
// own constructor comment for why (massCache_/fehCache_ are
// utils::ThreadVec, itself neither, the same reason SpecsynLib is
// neither). A static_assert, not a runtime test, since this is a
// compile-time property; checked here so a future change that
// accidentally made YieldChannel copyable/movable again (e.g. by
// switching massCache_/fehCache_ to a different cache type) would fail
// the build with a clear message, rather than silently reintroducing
// the per-thread-cache-sharing hazard the deleted special members
// exist to prevent.
static_assert(!std::is_copy_constructible_v<yields::YieldChannel>,
    "YieldChannel must not be copy-constructible (see its own constructor comment)");
static_assert(!std::is_move_constructible_v<yields::YieldChannel>,
    "YieldChannel must not be move-constructible (see its own constructor comment)");

#endif // TESTYIELDCHANNEL_HPP
