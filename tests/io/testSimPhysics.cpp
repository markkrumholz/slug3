/**
 * @file testSimPhysics.cpp
 * @author Mark Krumholz
 * @brief Unit tests for the SimPhysics class.
 * @date 2026-07-13
 */

#include "../src/extern/tomlplusplus/toml.hpp"
#include "../src/io/SimControls.hpp"
#include "../src/io/SimPhysics.hpp"
#include "../src/pdfs/PDF.hpp"
#include "../src/pdfs/PDFSegment.hpp"
#include "../src/pdfs/PDFSegmentLognormal.hpp"
#include "../src/pdfs/PDFSegmentPowerlaw.hpp"
#include "testSimPhysics.hpp"
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <numbers>
#include <string>
#include <utility>
#include <vector>

// Build the PDF that a correctly-parsed chabrier.toml
// should produce, and check that imf agrees
// with it. Both files describe the same Chabrier (2005) IMF, so
// this comparison is shared by the cluster and galaxy test decks.
static auto checkChabrierIMF(const pdfs::PDF& imf, const std::string& label) -> int
{
    auto plnCmp = std::make_unique<pdfs::PDFSegmentLognormal>(0.08, 1, 0.2, 0.55*std::numbers::ln10);
    auto pplCmp = std::make_unique<pdfs::PDFSegmentPowerlaw>(1, 120, -2.35);
    const std::vector<double> wgtCompare = { 1.0, (*plnCmp)(1.0) / (*pplCmp)(1.0) };
    std::vector<std::unique_ptr<pdfs::PDFSegment>> segCompare;
    segCompare.push_back(std::move(plnCmp));
    segCompare.push_back(std::move(pplCmp));
    const pdfs::PDF pdfCompare(std::move(segCompare), wgtCompare);

    if (imf.expectationValue() != pdfCompare.expectationValue())
    {
        std::cerr << "testSimPhysics: " << label << ": IMF does not match "
            "expected Chabrier IMF; expectation value = "
            << imf.expectationValue() << ", expected = "
            << pdfCompare.expectationValue() << "\n";
        return 1;
    }
    if (imf.integral(0.5, 20) != pdfCompare.integral(0.5, 20))
    {
        std::cerr << "testSimPhysics: " << label << ": IMF does not match "
            "expected Chabrier IMF; integral over [0.5,20] = "
            << imf.integral(0.5, 20) << ", expected = "
            << pdfCompare.integral(0.5, 20) << "\n";
        return 1;
    }
    return 0;
}

// Verify that SimPhysics read and constructed usable stellar
// tracks. Both test decks specify the MIST_test track set with
// alphaFe = -0.2 and the default vvcrit = 0.0, at FeH = 0.0 (an
// exact point on that track set's own [Fe/H] grid), so this check
// is shared between them. This is deliberately far lighter than
// the Tracks3D unit tests in tests/tracks -- it only needs to
// confirm the tracks were read successfully and are usable, not
// exhaustively verify Tracks3D's own behavior.
static auto checkTracks(const io::SimPhysics& sim, const std::string& label) -> int
{
    const auto& tracks = sim.tracks();

    if (tracks.aFe() != -0.2 || tracks.vVcrit() != 0.0)
    {
        std::cerr << "testSimPhysics: " << label << ": tracks do not have "
            "expected aFe/vVcrit; aFe = " << tracks.aFe()
            << ", vVcrit = " << tracks.vVcrit() << "\n";
        return 1;
    }

    if (tracks.feH().size() != 1 || tracks.feH().front() != 0.0)
    {
        std::cerr << "testSimPhysics: " << label << ": tracks do not have "
            "the expected single [Fe/H] = 0.0 slice\n";
        return 1;
    }

    // Confirm the tracks are actually usable by requesting a
    // track for a mass within their range
    constexpr double mass = 1.0;
    const auto track = tracks.getTrack(mass, 0.0);
    if (!track || track->xMin() >= track->xMax())
    {
        std::cerr << "testSimPhysics: " << label << ": getTrack(" << mass
            << ", 0.0) did not return a usable track\n";
        return 1;
    }

    return 0;
}

// Test parsing of a cluster-type input deck
static auto testSimPhysicsCluster() -> int
{
    const std::string fileName = "tests/core/assets/testCluster.in";
    try
    {
        const toml::table inputDeck = toml::parse_file(fileName);
        const io::SimControls controls(inputDeck);
        const io::SimPhysics sim(inputDeck, controls.simType());

        if (controls.simType() != io::SimControls::SimType::cluster)
        {
            std::cerr << "testSimPhysics: " << fileName
                << ": expected simType() == cluster\n";
            return 1;
        }

        if (checkChabrierIMF(sim.imf(), fileName) != 0) { return 1; }

        if (sim.cmf().getMin() != 1e3 || sim.cmf().getMax() != 1e3 ||
            sim.cmf().expectationValue() != 1e3 || !sim.cmf().normalized())
        {
            std::cerr << "testSimPhysics: " << fileName
                << ": CMF does not match expected delta function at 1e3\n";
            return 1;
        }

        if (sim.fehDist().getMin() != 0.0 || sim.fehDist().getMax() != 0.0 ||
            sim.fehDist().expectationValue() != 0.0 || !sim.fehDist().normalized())
        {
            std::cerr << "testSimPhysics: " << fileName
                << ": [Fe/H] distribution does not match expected delta function at 0.0\n";
            return 1;
        }

        // Galaxy-only physics should not have been initialized
        if (sim.clf().valid() || sim.sfr().valid())
        {
            std::cerr << "testSimPhysics: " << fileName
                << ": CLF and SFR should not be initialized for a cluster simulation\n";
            return 1;
        }

        if (checkTracks(sim, fileName) != 0) { return 1; }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSimPhysics: failed to parse valid cluster input deck "
            << fileName << ": " << error.what() << "\n";
        return 1;
    }
    return 0;
}

// Test parsing of a galaxy-type input deck
static auto testSimPhysicsGalaxy() -> int
{
    const std::string fileName = "tests/core/assets/testGalaxy.in";
    try
    {
        const toml::table inputDeck = toml::parse_file(fileName);
        const io::SimControls controls(inputDeck);
        const io::SimPhysics sim(inputDeck, controls.simType());

        if (controls.simType() != io::SimControls::SimType::galaxy)
        {
            std::cerr << "testSimPhysics: " << fileName
                << ": expected simType() == galaxy\n";
            return 1;
        }

        if (checkChabrierIMF(sim.imf(), fileName) != 0) { return 1; }

        if (sim.cmf().getMin() != 1e3 || sim.cmf().getMax() != 1e3 ||
            sim.cmf().expectationValue() != 1e3 || !sim.cmf().normalized())
        {
            std::cerr << "testSimPhysics: " << fileName
                << ": CMF does not match expected delta function at 1e3\n";
            return 1;
        }

        if (sim.fehDist().getMin() != 0.0 || sim.fehDist().getMax() != 0.0 ||
            sim.fehDist().expectationValue() != 0.0 || !sim.fehDist().normalized())
        {
            std::cerr << "testSimPhysics: " << fileName
                << ": [Fe/H] distribution does not match expected delta function at 0.0\n";
            return 1;
        }

        if (sim.clf().getMin() != 1e300 || sim.clf().getMax() != 1e300 ||
            sim.clf().expectationValue() != 1e300 || !sim.clf().normalized())
        {
            std::cerr << "testSimPhysics: " << fileName
                << ": CLF does not match expected delta function at 1e300\n";
            return 1;
        }

        // The SFR was given as a bare number, so it should have been
        // turned into a non-normalized, constant-in-time PDF spanning
        // [0, DBL_MAX] with weight DBL_MAX (since sfr = 1.0).
        if (sim.sfr().getMin() != 0.0 ||
            sim.sfr().getMax() != std::numeric_limits<double>::max() ||
            sim.sfr().normalized())
        {
            std::cerr << "testSimPhysics: " << fileName
                << ": SFR does not match expected non-normalized constant PDF\n";
            return 1;
        }

        if (checkTracks(sim, fileName) != 0) { return 1; }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSimPhysics: failed to parse valid galaxy input deck "
            << fileName << ": " << error.what() << "\n";
        return 1;
    }
    return 0;
}

auto testSimPhysics() -> int
{
    int result = 0;
    result += testSimPhysicsCluster();
    result += testSimPhysicsGalaxy();
    return result;
}
