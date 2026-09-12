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
 * import_yield_tables.py's own docstring for that source format).
 * @date 2026-09-12
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#ifndef TESTYIELDCHANNEL_HPP
#define TESTYIELDCHANNEL_HPP

#include "../../src/yields/YieldChannel.hpp"
#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
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
 * yield there must be exactly zero, not merely small.
 */
inline auto testYieldChannelCcsn() -> int
{
    const std::string registryName = "tests/yields/assets/yields.toml";
    int result = 0;

    try
    {
        const yields::YieldChannel yc(
            yields::Channel::ccsn_, "sukhbold_test", 0.0, 0.0, registryName);

        if (yc.channel() != yields::Channel::ccsn_)
        {
            std::cerr << "testYieldChannelCcsn: channel() does not match constructor argument\n";
            result = 1;
        }
        if (yc.masses().size() != 2 || yc.masses()[0] != 18.2 || yc.masses()[1] != 100.0)
        {
            std::cerr << "testYieldChannelCcsn: unexpected masses()\n";
            result = 1;
        }
        if (yc.yldZ().size() != 3 || yc.yldZ()[0] != 1 || yc.yldZ()[1] != 26 || yc.yldZ()[2] != 28)
        {
            std::cerr << "testYieldChannelCcsn: unexpected yldZ()\n";
            result = 1;
        }
        if (yc.yldA().size() != 3 || yc.yldA()[0] != 1 || yc.yldA()[1] != 56 || yc.yldA()[2] != 56)
        {
            std::cerr << "testYieldChannelCcsn: unexpected yldA()\n";
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
        const yields::YieldChannel yc(
            yields::Channel::massiveStarWinds_, "sukhbold_test", 0.0, 0.0, registryName);

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
 * testTracks3DFeHRangeGuard()'s identical check for Tracks3D.
 */
inline auto testYieldChannelFeHRangeGuard() -> int
{
    const std::string registryName = "tests/yields/assets/yields.toml";
    int result = 0;

    try
    {
        const yields::YieldChannel yc(
            yields::Channel::ccsn_, "sukhbold_test", -1.1, 0.0, registryName);
        std::cerr << "testYieldChannelFeHRangeGuard: construction with "
            "fehMin = -1.1 (below the available minimum of -1.0) "
            "should have thrown, but did not\n";
        result = 1;
    }
    catch (const std::exception&) { /* expected */ }

    try
    {
        const yields::YieldChannel yc(
            yields::Channel::ccsn_, "sukhbold_test", -1.0, 0.1, registryName);
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
            yields::Channel::ccsn_, "no_such_model", 0.0, 0.0, registryName);
        std::cerr << "testYieldChannelUnknownModel: construction with an "
            "unknown model name should have thrown, but did not\n";
        return 1;
    }
    catch (const std::exception&) { return 0; /* expected */ }
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
        const yields::YieldChannel yc(
            yields::Channel::ccsn_, "sukhbold_test", 0.0, 0.0, registryName);

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

    auto checkVec = [&result](const std::string& label, const std::vector<double>& actual,
        const std::vector<double>& expected) {
        if (actual.size() != expected.size())
        {
            std::cerr << label << ": expected a vector of size " << expected.size() <<
                ", got " << actual.size() << "\n";
            result = 1;
            return;
        }
        for (std::size_t i = 0; i < expected.size(); ++i)
        {
            if (std::abs(actual[i] - expected[i]) > tol)
            {
                std::cerr << label << ": isotope index " << i << ": expected " <<
                    expected[i] << ", got " << actual[i] << "\n";
                result = 1;
            }
        }
    };

    try
    {
        const yields::YieldChannel ccsn(
            yields::Channel::ccsn_, "sukhbold_test", 0.0, 0.0, registryName);
        const yields::YieldChannel wind(
            yields::Channel::massiveStarWinds_, "sukhbold_test", 0.0, 0.0, registryName);

        // Exact grid hit at mass = 18.2 -- same values as
        // testYieldChannelCcsn()/testYieldChannelMassiveStarWinds()
        checkVec("testYieldChannelInterpolation (ccsn, exact 18.2)",
            ccsn.yield(18.2, 0.0), { 5.93, 8.46e-2, 7.02e-2 });
        checkVec("testYieldChannelInterpolation (winds, exact 18.2)",
            wind.yield(18.2, 0.0), { 2.21, 4.00e-3, 0.0 });

        // Midpoint mass = 59.1 -- exactly halfway between the two grid
        // masses, so the interpolated result is exactly their average
        checkVec("testYieldChannelInterpolation (ccsn, midpoint)",
            ccsn.yield(59.1, 0.0), { 2.965, 4.23e-2, 3.51e-2 });
        checkVec("testYieldChannelInterpolation (winds, midpoint)",
            wind.yield(59.1, 0.0), { 16.405, 6.0e-2, 0.0 });

        // Multi-metallicity interpolation, across Fe_H groups rather
        // than across masses -- see this function's own comment on
        // why a synthetic Fe_H = -1.0 group is needed for this
        const yields::YieldChannel ccsnMultiFeH(
            yields::Channel::ccsn_, "sukhbold_test", -1.0, 0.0, registryName);
        const yields::YieldChannel windMultiFeH(
            yields::Channel::massiveStarWinds_, "sukhbold_test", -1.0, 0.0, registryName);

        if (ccsnMultiFeH.feH().size() != 2 || ccsnMultiFeH.feH()[0] != -1.0 ||
            ccsnMultiFeH.feH()[1] != 0.0)
        {
            std::cerr << "testYieldChannelInterpolation: unexpected feH() on the "
                "[-1.0, 0.0] channel\n";
            result = 1;
        }

        // Exact hit on the synthetic Fe_H = -1.0 group: exactly 2x the
        // real Fe_H = 0.0 values, at the exact grid mass 18.2
        checkVec("testYieldChannelInterpolation (ccsn, exact Fe_H=-1.0)",
            ccsnMultiFeH.yield(18.2, -1.0), { 11.86, 1.692e-1, 1.404e-1 });
        checkVec("testYieldChannelInterpolation (winds, exact Fe_H=-1.0)",
            windMultiFeH.yield(18.2, -1.0), { 4.42, 8.00e-3, 0.0 });

        // Fe_H = -0.5, the exact arithmetic midpoint of [-1.0, 0.0]:
        // exactly 1.5x the real Fe_H = 0.0 value, at the exact grid mass 18.2
        checkVec("testYieldChannelInterpolation (ccsn, Fe_H midpoint)",
            ccsnMultiFeH.yield(18.2, -0.5), { 8.895, 1.269e-1, 1.053e-1 });
        checkVec("testYieldChannelInterpolation (winds, Fe_H midpoint)",
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
