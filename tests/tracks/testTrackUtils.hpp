/**
 * @file testTrackUtils.hpp
 * @author Mark Krumholz
 * @brief Unit tests for the TrackUtils functions.
 * @details
 * This file contains unit tests for the parseRegistry and
 * findMatchingTracks functions declared in TrackUtils.hpp. The tests
 * use a small track registry and HDF5 file stored under
 * tests/tracks/assets so that they can run without access to the
 * full-size track files under data/tracks, which are too large to
 * store in the repository.
 * @date 2024-07-10
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#ifndef TESTTRACKUTILS_HPP
#define TESTTRACKUTILS_HPP

#include "../../src/tracks/TrackCommons.hpp"
#include "../../src/tracks/TrackUtils.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

/**
 * @brief Unit test for the parseRegistry function.
 * @return 0 if the test passes, 1 if it fails.
 * @details
 * This function tests that parseRegistry can successfully parse the
 * test registry tests/tracks/assets/tracks.toml, and that it throws
 * an exception for a registry file that does not exist.
 */
inline auto testParseRegistry() -> int
{
    const std::string registryName = "tests/tracks/assets/tracks.toml";

    try
    {
        auto [registry, registryPath] = tracks::parseRegistry(registryName);

        const toml::array* trackSets = registry["track_sets"].as_array();
        if (!trackSets || trackSets->size() != 2 ||
            !registry.contains("MIST_test") ||
            !registry.contains("MIST_test_lowmass"))
        {
            std::cerr << "testParseRegistry: parsed registry "
                << registryName << " does not have the expected "
                "[track_sets] contents\n";
            return 1;
        }
        if (registryPath.filename() != "tracks.toml")
        {
            std::cerr << "testParseRegistry: unexpected registry path "
                << registryPath.string() << "\n";
            return 1;
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "testParseRegistry: failed to parse " << registryName
            << ": " << e.what() << "\n";
        return 1;
    }

    // A registry file that does not exist should cause parseRegistry
    // to throw
    try
    {
        tracks::parseRegistry("tests/tracks/assets/no_such_registry.toml");
        std::cerr << "testParseRegistry: expected an exception for a "
            "nonexistent registry file, but none was thrown\n";
        return 1;
    }
    catch (const std::exception&) { /* this is the expected outcome */ }

    return 0;
}

/**
 * @brief Unit test for the findMatchingTracks function.
 * @return 0 if the test passes, 1 if it fails.
 * @details
 * This function tests findMatchingTracks against the test track set
 * MIST_test in tests/tracks/assets/tracks.toml, which contains 5
 * groups at afe = -0.2, vvcrit = 0.0, and feh = -1.0, -0.5, -0.25,
 * 0.0, and 0.5. It checks that the full feh range returns all 5
 * groups; that a bracketing feh range returns the minimal set of
 * groups whose feh values encompass that range, rather than only the
 * groups strictly inside it; that nExpand widens that bracketing range
 * by the requested number of groups on each side; that nExpand is
 * silently clamped when it would run off the edge of the available
 * groups; that a mismatched afe value returns no groups; and that
 * requesting an unknown track set throws an exception.
 */
inline auto testFindMatchingTracks() -> int
{
    const std::string registryName = "tests/tracks/assets/tracks.toml";
    const std::string trackName = "MIST_test";

    try
    {
        // Full feh range, with vvcrit and afe matching every group:
        // expect all 5 groups, sorted from lowest to highest feh
        auto [feh, names] = tracks::findMatchingTracks(
            trackName, -2.0, 2.0, 0.0, -0.2, 0, registryName);
        const std::vector<double> expectedFeh =
            { -1.0, -0.5, -0.25, 0.0, 0.5 };
        if (feh != expectedFeh || names.size() != expectedFeh.size())
        {
            std::cerr << "testFindMatchingTracks: full feh range did "
                "not return the expected 5 groups\n";
            return 1;
        }
        if (!std::ranges::is_sorted(feh))
        {
            std::cerr << "testFindMatchingTracks: returned feh values "
                "are not sorted\n";
            return 1;
        }

        // A bracketing feh range should return the minimal set of
        // groups whose feh values encompass it: [-0.4, -0.1] falls
        // strictly between the -0.5/-0.25 and -0.25/0.0 grid points
        // respectively, so the minimal enclosing set is -0.5, -0.25,
        // and 0.0 (not just -0.25, which is the only value that would
        // fall strictly inside [-0.4, -0.1])
        auto [fehBracket, namesBracket] = tracks::findMatchingTracks(
            trackName, -0.4, -0.1, 0.0, -0.2, 0, registryName);
        const std::vector<double> expectedBracket = { -0.5, -0.25, 0.0 };
        if (fehBracket != expectedBracket ||
            namesBracket.size() != expectedBracket.size())
        {
            std::cerr << "testFindMatchingTracks: bracketing feh range "
                "did not return the expected minimal enclosing set of "
                "groups\n";
            return 1;
        }

        // nExpand = 1 should widen the above bracketing range by one
        // group on each side, to -1.0, -0.5, -0.25, 0.0, 0.5 (i.e. all
        // 5 groups, since there is exactly one group beyond each edge
        // of the unexpanded bracket)
        auto [fehExpand1, namesExpand1] = tracks::findMatchingTracks(
            trackName, -0.4, -0.1, 0.0, -0.2, 1, registryName);
        if (fehExpand1 != expectedFeh ||
            namesExpand1.size() != expectedFeh.size())
        {
            std::cerr << "testFindMatchingTracks: nExpand = 1 did not "
                "widen the bracketing range by one group on each side\n";
            return 1;
        }

        // A large nExpand should be silently clamped to the available
        // range rather than erroring out
        auto [fehExpandBig, namesExpandBig] = tracks::findMatchingTracks(
            trackName, -0.4, -0.1, 0.0, -0.2, 100, registryName);
        if (fehExpandBig != expectedFeh ||
            namesExpandBig.size() != expectedFeh.size())
        {
            std::cerr << "testFindMatchingTracks: a large nExpand was "
                "not silently clamped to the available range\n";
            return 1;
        }

        // Every group in this file has an afe attribute, so an afe
        // value that matches none of them should return no groups
        auto [fehNoMatch, namesNoMatch] = tracks::findMatchingTracks(
            trackName, -2.0, 2.0, 0.0, 0.0, 0, registryName);
        if (!namesNoMatch.empty() || !fehNoMatch.empty())
        {
            std::cerr << "testFindMatchingTracks: expected no matches "
                "for a nonexistent afe value, got "
                << namesNoMatch.size() << "\n";
            return 1;
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "testFindMatchingTracks: unexpected exception: "
            << e.what() << "\n";
        return 1;
    }

    // Requesting an unknown track set should throw
    try
    {
        tracks::findMatchingTracks(
            "NoSuchTrackSet", -2.0, 2.0, 0.0, 0.0, 0, registryName);
        std::cerr << "testFindMatchingTracks: expected an exception for "
            "an unknown track set, but none was thrown\n";
        return 1;
    }
    catch (const std::exception&) { /* this is the expected outcome */ }

    return 0;
}

/**
 * @brief Unit test for the findTrack function.
 * @return 0 if the test passes, 1 if it fails.
 * @details
 * This function tests findTrack against the test track set MIST_test
 * in tests/tracks/assets/tracks.toml, which contains 5 groups at
 * afe = -0.2, vvcrit = 0.0, and feh = -1.0, -0.5, -0.25, 0.0, and 0.5.
 * It checks that a feh/vvcrit/afe combination matching one of those
 * groups returns that group's name; that a feh value not on the grid
 * returns an empty string; that a mismatched afe value returns an
 * empty string; and that requesting an unknown track set throws an
 * exception.
 */
inline auto testFindTrack() -> int
{
    const std::string registryName = "tests/tracks/assets/tracks.toml";
    const std::string trackName = "MIST_test";

    try
    {
        // feh = 0.0, vvcrit = 0.0, afe = -0.2 matches one of the 5 groups
        const auto name = tracks::findTrack(
            trackName, 0.0, 0.0, -0.2, registryName);
        if (name != "feh_0.00_afe_-0.2_vvcrit_0.00")
        {
            std::cerr << "testFindTrack: expected group "
                "feh_0.00_afe_-0.2_vvcrit_0.00, got " << name << "\n";
            return 1;
        }

        // feh = -0.25 matches a different one of the 5 groups
        const auto nameNeg = tracks::findTrack(
            trackName, -0.25, 0.0, -0.2, registryName);
        if (nameNeg != "feh_-0.25_afe_-0.2_vvcrit_0.00")
        {
            std::cerr << "testFindTrack: expected group "
                "feh_-0.25_afe_-0.2_vvcrit_0.00, got " << nameNeg << "\n";
            return 1;
        }

        // No group in this file has feh = 0.25, so this should return
        // an empty string
        const auto noFehMatch = tracks::findTrack(
            trackName, 0.25, 0.0, -0.2, registryName);
        if (!noFehMatch.empty())
        {
            std::cerr << "testFindTrack: expected no match for a "
                "nonexistent feh value, got " << noFehMatch << "\n";
            return 1;
        }

        // Every group in this file has an afe attribute, so an afe
        // value that matches none of them should return an empty string
        const auto noAfeMatch = tracks::findTrack(
            trackName, 0.0, 0.0, 0.0, registryName);
        if (!noAfeMatch.empty())
        {
            std::cerr << "testFindTrack: expected no match for a "
                "nonexistent afe value, got " << noAfeMatch << "\n";
            return 1;
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "testFindTrack: unexpected exception: "
            << e.what() << "\n";
        return 1;
    }

    // Requesting an unknown track set should throw
    try
    {
        tracks::findTrack("NoSuchTrackSet", 0.0, 0.0, 0.0, registryName);
        std::cerr << "testFindTrack: expected an exception for "
            "an unknown track set, but none was thrown\n";
        return 1;
    }
    catch (const std::exception&) { /* this is the expected outcome */ }

    return 0;
}

/**
 * @brief Unit test for the getTrackSize function.
 * @return 0 if the test passes, 1 if it fails.
 * @details
 * This function tests getTrackSize against tests/tracks/assets/MIST_test.h5,
 * whose groups each contain 6 masses, with the tracks for those masses
 * having differing numbers of time points, the largest of which is
 * 1721. It checks that getTrackSize reports these values correctly for
 * two different groups in the file, and that it throws exceptions for
 * a nonexistent file and a nonexistent group.
 */
inline auto testGetTrackSize() -> int
{
    const std::filesystem::path h5Name = "tests/tracks/assets/MIST_test.h5";
    const std::pair<size_t, size_t> expected = { 6, 1721 };

    try
    {
        for (const auto& groupName :
            { "feh_-0.25_afe_-0.2_vvcrit_0.00", "feh_0.50_afe_-0.2_vvcrit_0.00" })
        {
            const auto size = tracks::getTrackSize(h5Name, groupName);
            if (size != expected)
            {
                std::cerr << "testGetTrackSize: for group " << groupName
                    << " expected (nmass, ntime) = (" << expected.first
                    << ", " << expected.second << "), got ("
                    << size.first << ", " << size.second << ")\n";
                return 1;
            }
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "testGetTrackSize: unexpected exception: "
            << e.what() << "\n";
        return 1;
    }

    // A nonexistent file should cause getTrackSize to throw
    try
    {
        tracks::getTrackSize("tests/tracks/assets/no_such_file.h5",
            "feh_-0.25_afe_-0.2_vvcrit_0.00");
        std::cerr << "testGetTrackSize: expected an exception for a "
            "nonexistent file, but none was thrown\n";
        return 1;
    }
    catch (const std::exception&) { /* this is the expected outcome */ }

    // A nonexistent group should cause getTrackSize to throw
    try
    {
        tracks::getTrackSize(h5Name, "no_such_group");
        std::cerr << "testGetTrackSize: expected an exception for a "
            "nonexistent group, but none was thrown\n";
        return 1;
    }
    catch (const std::exception&) { /* this is the expected outcome */ }

    return 0;
}

/**
 * @brief Unit test for the renormalizeSurfaceAbundances function.
 * @return 0 if the test passes, 1 if it fails.
 * @details
 * Checks that surface mass fractions summing to at most 1 are left
 * untouched; that a sum above 1 -- both the composition of the star
 * that produced NaN photometry in the 3.0.1 validation run,
 * X + Y + C + N + O = 1.0135, and an 8.5% overshoot, the largest any
 * shipped track set's interpolated isochrones produce -- is rescaled to
 * sum to exactly 1, with ratios preserved, Z = 1 - X - Y non-negative,
 * and every non-abundance field untouched; and that a NaN sum is left
 * alone.
 */
inline auto testRenormalizeSurfaceAbundances() -> int
{
    using tracks::FieldIdx;
    constexpr auto nQty = static_cast<std::size_t>(FieldIdx::nTrackQty);
    const auto idx = [](const FieldIdx f) { return static_cast<std::size_t>(f); };
    const auto makeProps = [&idx](const double x, const double y, const double c,
        const double n, const double o) -> std::array<double, nQty>
    {
        std::array<double, nQty> props{};
        props.at(idx(FieldIdx::mass)) = 66.5;
        props.at(idx(FieldIdx::mdot)) = 1.4e-4;
        props.at(idx(FieldIdx::logL)) = 6.45;
        props.at(idx(FieldIdx::logTe)) = 4.9;
        props.at(idx(FieldIdx::hSurf)) = x;
        props.at(idx(FieldIdx::heSurf)) = y;
        props.at(idx(FieldIdx::cSurf)) = c;
        props.at(idx(FieldIdx::nSurf)) = n;
        props.at(idx(FieldIdx::oSurf)) = o;
        return props;
    };
    const auto surfSum = [&idx](const std::array<double, nQty>& p)
    {
        return p.at(idx(FieldIdx::hSurf)) + p.at(idx(FieldIdx::heSurf)) +
            p.at(idx(FieldIdx::cSurf)) + p.at(idx(FieldIdx::nSurf)) + p.at(idx(FieldIdx::oSurf));
    };

    // Sum <= 1: untouched
    const auto valid = makeProps(0.7, 0.28, 0.003, 0.001, 0.008);
    auto validCopy = valid;
    tracks::renormalizeSurfaceAbundances(validCopy);
    if (validCopy != valid)
    {
        std::cerr << "testRenormalizeSurfaceAbundances: a composition summing to "
            "less than 1 was modified\n";
        return 1;
    }

    // Overshoots: rescaled to sum to 1, ratios and other fields kept
    const std::array<std::array<double, nQty>, 2> overshoots = {
        makeProps(0.1538509684580525, 0.847424251704682, 0.0058013157108631344,
            0.005026218138719116, 0.0014471129569203775), // sum = 1.0135
        makeProps(0.4833, 0.5896, 0.003, 0.0061, 0.003), // sum = 1.085
    };
    for (const auto& bad : overshoots)
    {
        auto fixed = bad;
        tracks::renormalizeSurfaceAbundances(fixed);
        const double badSum = surfSum(bad);
        if (std::abs(surfSum(fixed) - 1.0) > 1e-14 ||
            1.0 - fixed.at(idx(FieldIdx::hSurf)) - fixed.at(idx(FieldIdx::heSurf)) < 0.0)
        {
            std::cerr << "testRenormalizeSurfaceAbundances: renormalized sum is "
                << surfSum(fixed) << ", expected 1 with Z = 1 - X - Y >= 0\n";
            return 1;
        }
        for (std::size_t k = 0; k < nQty; ++k)
        {
            const bool isSurf = k >= idx(FieldIdx::hSurf) && k <= idx(FieldIdx::oSurf);
            const double expected = isSurf ? bad.at(k) / badSum : bad.at(k);
            if (std::abs(fixed.at(k) - expected) > 1e-15 * std::abs(expected))
            {
                std::cerr << "testRenormalizeSurfaceAbundances: field " << tracks::fieldStr.at(k)
                    << " is " << fixed.at(k) << " after renormalization, expected "
                    << expected << "\n";
                return 1;
            }
        }
    }

    // NaN sum: left alone, for downstream checks to catch
    auto withNaN = makeProps(std::numeric_limits<double>::quiet_NaN(), 0.9, 0.003, 0.001, 0.008);
    tracks::renormalizeSurfaceAbundances(withNaN);
    if (!std::isnan(withNaN.at(idx(FieldIdx::hSurf))) || withNaN.at(idx(FieldIdx::heSurf)) != 0.9)
    {
        std::cerr << "testRenormalizeSurfaceAbundances: a NaN composition was modified\n";
        return 1;
    }

    return 0;
}

/**
 * @brief Unit test for the checkSurfaceAbundances function.
 * @return 0 if the test passes, 1 if it fails.
 * @details
 * Builds a three-row table whose columns are deliberately not in
 * FieldIdx order (so qtyIdx is exercised), and checks that rows summing
 * to at most 1 + surfaceAbundanceTol -- including a roundoff-level
 * excess of 6e-5, the largest in any shipped track set -- pass, while a
 * row summing to 1.01 throws, naming that row.
 */
inline auto testCheckSurfaceAbundances() -> int
{
    using tracks::FieldIdx;
    constexpr auto nQty = static_cast<std::size_t>(FieldIdx::nTrackQty);
    // On-disk column order: an age column, then the quantities in reverse
    constexpr std::size_t ncol = nQty + 1;
    std::vector<std::size_t> qtyIdx(nQty);
    for (std::size_t k = 0; k < nQty; ++k) { qtyIdx.at(k) = ncol - 1 - k; }
    const auto makeTable = [&qtyIdx](const std::vector<std::array<double, 5>>& rows)
    {
        std::vector<double> data(rows.size() * ncol, 0.0);
        for (std::size_t r = 0; r < rows.size(); ++r)
        {
            for (std::size_t j = 0; j < 5; ++j)
            {
                const auto k = static_cast<std::size_t>(FieldIdx::hSurf) + j;
                data.at((r * ncol) + qtyIdx.at(k)) = rows.at(r).at(j);
            }
        }
        return data;
    };

    const auto good = makeTable({ { 0.7, 0.28, 0.003, 0.001, 0.008 },
        { 0.0, 0.98, 0.01, 0.0, 0.01 }, { 0.5, 0.48006, 0.01, 0.005, 0.005 } });
    try
    {
        tracks::checkSurfaceAbundances(good, 3, ncol, qtyIdx, "good table");
    }
    catch (const std::exception& e)
    {
        std::cerr << "testCheckSurfaceAbundances: valid table rejected: " << e.what() << "\n";
        return 1;
    }

    const auto bad = makeTable({ { 0.7, 0.28, 0.003, 0.001, 0.008 },
        { 0.7, 0.28, 0.003, 0.001, 0.008 }, { 0.5, 0.49, 0.01, 0.005, 0.005 } });
    try
    {
        tracks::checkSurfaceAbundances(bad, 3, ncol, qtyIdx, "bad table");
        std::cerr << "testCheckSurfaceAbundances: expected a row summing to 1.01 to throw\n";
        return 1;
    }
    catch (const std::runtime_error& e)
    {
        const std::string msg = e.what();
        if (msg.find("bad table, row 2") == std::string::npos)
        {
            std::cerr << "testCheckSurfaceAbundances: error does not identify the "
                "offending row: " << msg << "\n";
            return 1;
        }
    }
    return 0;
}

/**
 * @brief Unit test for the TrackUtils functions.
 * @return 0 if the test passes, 1 if it fails.
 */
inline auto testTrackUtils() -> int
{
    int result = 0;
    result += testParseRegistry();
    result += testFindMatchingTracks();
    result += testFindTrack();
    result += testGetTrackSize();
    result += testRenormalizeSurfaceAbundances();
    result += testCheckSurfaceAbundances();
    return result;
}

#endif // TESTTRACKUTILS_HPP
