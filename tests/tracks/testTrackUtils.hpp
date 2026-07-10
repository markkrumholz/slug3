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
 */

#ifndef TESTTRACKUTILS_HPP
#define TESTTRACKUTILS_HPP

#include "../../src/tracks/TrackUtils.hpp"
#include <algorithm>
#include <exception>
#include <iostream>
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
        if (!trackSets || trackSets->size() != 1 ||
            !registry.contains("MIST_test"))
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
            registryName, trackName, -2.0, 2.0, 0.0, -0.2);
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
            registryName, trackName, -0.4, -0.1, 0.0, -0.2);
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
            registryName, trackName, -0.4, -0.1, 0.0, -0.2, 1);
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
            registryName, trackName, -0.4, -0.1, 0.0, -0.2, 100);
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
            registryName, trackName, -2.0, 2.0, 0.0, 0.0);
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
            registryName, "NoSuchTrackSet", -2.0, 2.0);
        std::cerr << "testFindMatchingTracks: expected an exception for "
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
 * @brief Unit test for the TrackUtils functions.
 * @return 0 if the test passes, 1 if it fails.
 */
inline auto testTrackUtils() -> int
{
    int result = 0;
    result += testParseRegistry();
    result += testFindMatchingTracks();
    result += testGetTrackSize();
    return result;
}

#endif // TESTTRACKUTILS_HPP
