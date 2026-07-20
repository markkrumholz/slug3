/**
 * @file testSpectraUtils.cpp
 * @author Mark Krumholz
 * @brief Unit tests for the SpectraUtils functions.
 * @details
 * This file contains unit tests for the parseRegistry and
 * findMatchingSpectra functions declared in SpectraUtils.hpp. The
 * tests use a small spectra registry and HDF5 file stored under
 * tests/specsyn/assets so that they can run without access to the
 * full-size BOSZ library under data/spectra, which is too large to
 * store in the repository.
 * @date 2026-07-20
 */

#include "../../src/specsyn/SpectraUtils.hpp"
#include "testSpectraUtils.hpp"
#include <algorithm>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

namespace
{
    const std::string registryName = "tests/specsyn/assets/spectra.toml";
    const std::string spectraName = "BOSZ_test";

    // The test fixture BOSZ_test.h5 has 14 feh groups, all sharing
    // afe = 0.0, cfe = 0.0, micro = 0.0, and r = 500
    const std::vector<double> allFeh =
        { -2.5, -2.25, -2.0, -1.75, -1.5, -1.25, -1.0, -0.75,
          -0.5, -0.25, 0.0, 0.25, 0.5, 0.75 };
} // namespace

/**
 * @brief Unit test for the parseRegistry function.
 * @return 0 if the test passes, 1 if it fails.
 * @details
 * This function tests that parseRegistry can successfully parse the
 * test registry tests/specsyn/assets/spectra.toml, and that it throws
 * an exception for a registry file that does not exist.
 */
static auto testParseRegistry() -> int
{
    try
    {
        auto [registry, registryPath] = specsyn::parseRegistry(registryName);

        const toml::array* spectraSets = registry["spectra_sets"].as_array();
        if (!spectraSets || spectraSets->size() != 1 ||
            !registry.contains(spectraName))
        {
            std::cerr << "testParseRegistry: parsed registry "
                << registryName << " does not have the expected "
                "[spectra_sets] contents\n";
            return 1;
        }
        if (registryPath.filename() != "spectra.toml")
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
        specsyn::parseRegistry("tests/specsyn/assets/no_such_registry.toml");
        std::cerr << "testParseRegistry: expected an exception for a "
            "nonexistent registry file, but none was thrown\n";
        return 1;
    }
    catch (const std::exception&) { /* this is the expected outcome */ }

    return 0;
}

/**
 * @brief Unit test for the findMatchingSpectra function.
 * @return 0 if the test passes, 1 if it fails.
 * @details
 * This function tests findMatchingSpectra against the test spectra
 * set BOSZ_test in tests/specsyn/assets/spectra.toml, which contains
 * 14 groups, all at afe = cfe = microTurb = 0.0 and r = 500, spanning
 * feh = -2.5 through 0.75 in steps of 0.25. It checks that the full
 * feh range returns all 14 groups, sorted from lowest to highest feh;
 * that a feh range not aligned with the grid returns only the groups
 * strictly inside it (unlike findMatchingTracks, findMatchingSpectra
 * does not bracket the range); that mismatched afe, cfe, microTurb,
 * or r values each return no groups; and that requesting an unknown
 * spectra set throws an exception.
 */
static auto testFindMatchingSpectra() -> int
{
    try
    {
        // Full feh range, with afe/cfe/microTurb/r matching every group:
        // expect all 14 groups, sorted from lowest to highest feh
        auto [feh, names] = specsyn::findMatchingSpectra(
            spectraName, -3.0, 1.0, 0.0, 0.0, 0.0, 500, registryName);
        if (feh != allFeh || names.size() != allFeh.size())
        {
            std::cerr << "testFindMatchingSpectra: full feh range did "
                "not return the expected 14 groups\n";
            return 1;
        }
        if (!std::ranges::is_sorted(feh))
        {
            std::cerr << "testFindMatchingSpectra: returned feh values "
                "are not sorted\n";
            return 1;
        }

        // A feh range not aligned with the grid should return only the
        // groups whose feh actually falls inside [fehMin, fehMax] --
        // not a bracketing superset the way findMatchingTracks would
        auto [fehSub, namesSub] = specsyn::findMatchingSpectra(
            spectraName, -0.9, -0.1, 0.0, 0.0, 0.0, 500, registryName);
        const std::vector<double> expectedSub = { -0.75, -0.5, -0.25 };
        if (fehSub != expectedSub || namesSub.size() != expectedSub.size())
        {
            std::cerr << "testFindMatchingSpectra: off-grid feh range "
                "did not return the expected literal subset of groups\n";
            return 1;
        }

        // Every group in this file has an afe attribute, so an afe
        // value that matches none of them should return no groups
        auto [fehNoAfe, namesNoAfe] = specsyn::findMatchingSpectra(
            spectraName, -3.0, 1.0, 1.0, 0.0, 0.0, 500, registryName);
        if (!namesNoAfe.empty() || !fehNoAfe.empty())
        {
            std::cerr << "testFindMatchingSpectra: expected no matches "
                "for a nonexistent afe value, got "
                << namesNoAfe.size() << "\n";
            return 1;
        }

        // Likewise for a mismatched cfe value
        auto [fehNoCfe, namesNoCfe] = specsyn::findMatchingSpectra(
            spectraName, -3.0, 1.0, 0.0, 1.0, 0.0, 500, registryName);
        if (!namesNoCfe.empty() || !fehNoCfe.empty())
        {
            std::cerr << "testFindMatchingSpectra: expected no matches "
                "for a nonexistent cfe value, got "
                << namesNoCfe.size() << "\n";
            return 1;
        }

        // Likewise for a mismatched microTurb value
        auto [fehNoMicro, namesNoMicro] = specsyn::findMatchingSpectra(
            spectraName, -3.0, 1.0, 0.0, 0.0, 5.0, 500, registryName);
        if (!namesNoMicro.empty() || !fehNoMicro.empty())
        {
            std::cerr << "testFindMatchingSpectra: expected no matches "
                "for a nonexistent microTurb value, got "
                << namesNoMicro.size() << "\n";
            return 1;
        }

        // Likewise for a mismatched r value
        auto [fehNoR, namesNoR] = specsyn::findMatchingSpectra(
            spectraName, -3.0, 1.0, 0.0, 0.0, 0.0, 100, registryName);
        if (!namesNoR.empty() || !fehNoR.empty())
        {
            std::cerr << "testFindMatchingSpectra: expected no matches "
                "for a nonexistent r value, got "
                << namesNoR.size() << "\n";
            return 1;
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "testFindMatchingSpectra: unexpected exception: "
            << e.what() << "\n";
        return 1;
    }

    // Requesting an unknown spectra set should throw
    try
    {
        specsyn::findMatchingSpectra(
            "NoSuchSpectraSet", -3.0, 1.0, 0.0, 0.0, 0.0, 500, registryName);
        std::cerr << "testFindMatchingSpectra: expected an exception for "
            "an unknown spectra set, but none was thrown\n";
        return 1;
    }
    catch (const std::exception&) { /* this is the expected outcome */ }

    return 0;
}

auto testSpectraUtils() -> int
{
    int result = 0;
    result += testParseRegistry();
    result += testFindMatchingSpectra();
    return result;
}
