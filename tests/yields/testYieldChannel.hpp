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
#include <exception>
#include <iostream>
#include <string>
#include <utility>

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
 * tests/yields/assets/yields.toml's "sukhbold_test" model only
 * provides Fe_H = 0.0. This checks that requesting a fehMin below or
 * a fehMax above 0.0 throws, mirroring
 * testTracks3DFeHRangeGuard()'s identical check for Tracks3D.
 */
inline auto testYieldChannelFeHRangeGuard() -> int
{
    const std::string registryName = "tests/yields/assets/yields.toml";
    int result = 0;

    try
    {
        const yields::YieldChannel yc(
            yields::Channel::ccsn_, "sukhbold_test", -0.1, 0.0, registryName);
        std::cerr << "testYieldChannelFeHRangeGuard: construction with "
            "fehMin = -0.1 (below the available minimum of 0.0) "
            "should have thrown, but did not\n";
        result = 1;
    }
    catch (const std::exception&) { /* expected */ }

    try
    {
        const yields::YieldChannel yc(
            yields::Channel::ccsn_, "sukhbold_test", 0.0, 0.1, registryName);
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
 * @brief Unit test that copying/moving a YieldChannel keeps yld() valid
 * @return 0 if the test passes, 1 if it fails
 * @details
 * yld()'s own comment explains why it builds its mdspan view fresh
 * from yieldData_ on every call rather than caching one as a sibling
 * member: a cached view would keep pointing at whichever object's
 * yieldData_ happened to own the memory at the time it was built,
 * which a subsequent copy or move can invalidate or orphan. This
 * constructs one YieldChannel, copies it, move-constructs a second
 * object from a third, and checks that every one of the resulting
 * three objects' own yld() still reads back the correct value --
 * this would fail (or, worse, read freed memory) if yld() were ever
 * changed back to a plain stored member instead.
 */
inline auto testYieldChannelCopyMoveSafety() -> int
{
    const std::string registryName = "tests/yields/assets/yields.toml";
    int result = 0;

    try
    {
        const yields::YieldChannel original(
            yields::Channel::ccsn_, "sukhbold_test", 0.0, 0.0, registryName);

        const yields::YieldChannel copy(original); // NOLINT(performance-unnecessary-copy-initialization) -- deliberately exercising the copy constructor itself, not just its result

        yields::YieldChannel toMoveFrom(original);
        const yields::YieldChannel moved(std::move(toMoveFrom));

        result += checkYield(original, "testYieldChannelCopyMoveSafety (original)", 0, 0, 5.93);
        result += checkYield(copy, "testYieldChannelCopyMoveSafety (copy)", 0, 0, 5.93);
        result += checkYield(moved, "testYieldChannelCopyMoveSafety (moved)", 0, 0, 5.93);
    }
    catch (const std::exception& e)
    {
        std::cerr << "testYieldChannelCopyMoveSafety: failed to construct YieldChannel from "
            << registryName << ": " << e.what() << "\n";
        return 1;
    }

    return result;
}

#endif // TESTYIELDCHANNEL_HPP
