/**
 * @file testSimControls.cpp
 * @author Mark Krumholz
 * @brief Unit tests for the SimControls class.
 * @date 2026-07-16
 */

#include "../src/io/SimControls.hpp"
#include "../src/pdfs/PDF.hpp"
#include "../src/pdfs/PDFSegment.hpp"
#include "../src/pdfs/PDFSegmentLognormal.hpp"
#include "../src/pdfs/PDFSegmentPowerlaw.hpp"
#include "../src/utils/MiscUtils.hpp"
#include "testSimControls.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
#include <memory>
#include <numbers>
#include <stdexcept>
#include <string>
#include <toml.hpp>
#include <utility>
#include <vector>

// Compare two vectors of doubles for approximate equality, reporting
// a descriptive error through std::cerr if they don't match.
static auto checkOutTimes(const std::vector<double>& actual,
    const std::vector<double>& expected,
    const std::string& label) -> int
{
    constexpr double tolerance = 1e-9;
    if (actual.size() != expected.size())
    {
        std::cerr << "testSimControls: " << label
            << ": outTimes() has size " << actual.size()
            << ", expected " << expected.size() << "\n";
        return 1;
    }
    for (std::size_t i = 0; i < actual.size(); ++i)
    {
        const double denom = std::max(std::abs(expected.at(i)), 1.0);
        if (std::abs(actual.at(i) - expected.at(i)) / denom > tolerance)
        {
            std::cerr << "testSimControls: " << label
                << ": outTimes()[" << i << "] = " << actual.at(i)
                << ", expected " << expected.at(i) << "\n";
            return 1;
        }
    }
    return 0;
}

// Verify that simType is read correctly, that model_name, verbosity,
// output_mode, out_dir, outputClusters, and n_trial fall back to their
// documented defaults when not specified in the input deck, and that
// the start_time/end_time/ntime output option (linear spacing) is
// correctly expanded.
static auto testSimControlsDefaults() -> int
{
    const std::string fileName = "tests/core/assets/testCluster.in";
    try
    {
        const toml::table inputDeck = toml::parse_file(fileName);
        const io::SimControls controls(inputDeck);

        if (controls.simType() != io::SimControls::SimType::cluster)
        {
            std::cerr << "testSimControls: " << fileName
                << ": expected simType() == SimType::cluster\n";
            return 1;
        }
        if (controls.modelName() != "slug_sim")
        {
            std::cerr << "testSimControls: " << fileName
                << ": expected default modelName() == \"slug_sim\", got "
                << controls.modelName() << "\n";
            return 1;
        }
        if (controls.verbosity() != 0)
        {
            std::cerr << "testSimControls: " << fileName
                << ": expected default verbosity() == 0, got "
                << controls.verbosity() << "\n";
            return 1;
        }
        if (controls.outputMode() != io::SimControls::OutputMode::h5)
        {
            std::cerr << "testSimControls: " << fileName
                << ": expected default outputMode() == OutputMode::h5\n";
            return 1;
        }
        if (!controls.outDir().empty())
        {
            std::cerr << "testSimControls: " << fileName
                << ": expected default outDir() == \"\", got "
                << controls.outDir() << "\n";
            return 1;
        }
        if (!controls.outputClusters())
        {
            std::cerr << "testSimControls: " << fileName
                << ": expected default outputClusters() == true\n";
            return 1;
        }
        if (controls.nTrial() != 1)
        {
            std::cerr << "testSimControls: " << fileName
                << ": expected default nTrial() == 1, got "
                << controls.nTrial() << "\n";
            return 1;
        }

        return checkOutTimes(controls.outTimes(), { 0.0, 5.0, 10.0 }, fileName);
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSimControls: failed to parse valid input deck "
            << fileName << ": " << error.what() << "\n";
        return 1;
    }
}

// Verify that model_name, verbosity, output_mode, out_dir, and n_trial
// are correctly read from the input deck when explicitly specified,
// and that outputs.output_clusters is ignored (outputClusters() stays
// at its default of true) for a cluster-type simulation even though
// this deck sets it to false.
static auto testSimControlsExplicit() -> int
{
    const std::string fileName = "tests/io/assets/testControlsExplicit.in";
    try
    {
        const toml::table inputDeck = toml::parse_file(fileName);
        const io::SimControls controls(inputDeck);

        if (controls.simType() != io::SimControls::SimType::cluster)
        {
            std::cerr << "testSimControls: " << fileName
                << ": expected simType() == SimType::cluster\n";
            return 1;
        }
        if (!controls.outputClusters())
        {
            std::cerr << "testSimControls: " << fileName
                << ": expected outputClusters() == true (output_clusters is "
                   "only read for a galaxy-type simulation)\n";
            return 1;
        }
        if (controls.modelName() != "my_test_model")
        {
            std::cerr << "testSimControls: " << fileName
                << ": expected modelName() == \"my_test_model\", got "
                << controls.modelName() << "\n";
            return 1;
        }
        if (controls.verbosity() != 3)
        {
            std::cerr << "testSimControls: " << fileName
                << ": expected verbosity() == 3, got "
                << controls.verbosity() << "\n";
            return 1;
        }
        if (controls.outputMode() != io::SimControls::OutputMode::ascii)
        {
            std::cerr << "testSimControls: " << fileName
                << ": expected outputMode() == OutputMode::ascii\n";
            return 1;
        }
        if (controls.outDir() != "my_test_outdir")
        {
            std::cerr << "testSimControls: " << fileName
                << ": expected outDir() == \"my_test_outdir\", got "
                << controls.outDir() << "\n";
            return 1;
        }
        if (controls.nTrial() != 7)
        {
            std::cerr << "testSimControls: " << fileName
                << ": expected nTrial() == 7, got "
                << controls.nTrial() << "\n";
            return 1;
        }

        return checkOutTimes(controls.outTimes(), { 0.0, 5.0, 10.0 }, fileName);
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSimControls: failed to parse valid input deck "
            << fileName << ": " << error.what() << "\n";
        return 1;
    }
}

// Verify option 1's PDF interpretation: output.output_times, given a
// single scalar value, is tried first as a PDF (via
// utils::initPDFFromKey) and succeeds without ever falling back to
// readOutputTimesArray -- here the distribution is a delta function at
// 5.0, so the draw is deterministic.
static auto testSimControlsOutputTimeDist() -> int
{
    const std::string fileName = "tests/io/assets/testControlsOutputTimeDist.in";
    try
    {
        const toml::table inputDeck = toml::parse_file(fileName);
        const io::SimControls controls(inputDeck);
        return checkOutTimes(controls.outTimes(), { 5.0 }, fileName);
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSimControls: failed to parse valid input deck "
            << fileName << ": " << error.what() << "\n";
        return 1;
    }
}

// Verify option 1's array-fallback interpretation: output.output_times,
// given an array, fails utils::initPDFFromKey (an array is neither a
// number nor a string) and falls back to readOutputTimesArray, which
// reads it back exactly as an explicit array of output times.
static auto testSimControlsOutputTimesArray() -> int
{
    const std::string fileName = "tests/io/assets/testControlsOutputTimesArray.in";
    try
    {
        const toml::table inputDeck = toml::parse_file(fileName);
        const io::SimControls controls(inputDeck);
        return checkOutTimes(controls.outTimes(), { 1.0, 2.5, 9.0 }, fileName);
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSimControls: failed to parse valid input deck "
            << fileName << ": " << error.what() << "\n";
        return 1;
    }
}

// Verify option 3a: a log-spaced start_time/end_time/ntime grid produces
// a geometric sequence of output times.
static auto testSimControlsOutputTimesLog() -> int
{
    const std::string fileName = "tests/io/assets/testControlsOutputTimesLog.in";
    try
    {
        const toml::table inputDeck = toml::parse_file(fileName);
        const io::SimControls controls(inputDeck);
        return checkOutTimes(controls.outTimes(), { 1.0, 10.0, 100.0 }, fileName);
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSimControls: failed to parse valid input deck "
            << fileName << ": " << error.what() << "\n";
        return 1;
    }
}

// Verify that constructing SimControls from a deck with no output time
// option at all throws.
static auto testSimControlsNoOutputs() -> int
{
    const std::string fileName = "tests/io/assets/testControlsNoOutputs.in";
    const toml::table inputDeck = toml::parse_file(fileName);
    try
    {
        const io::SimControls controls(inputDeck);
        std::cerr << "testSimControls: " << fileName
            << ": expected construction to throw, but it succeeded\n";
        return 1;
    }
    catch (const std::runtime_error&)
    {
        return 0;
    }
}

// Verify that constructing SimControls from a deck specifying both
// output.output_times and the output.start_time/end_time/ntime range
// throws.
static auto testSimControlsOutputTimesConflict() -> int
{
    const std::string fileName = "tests/io/assets/testControlsOutputTimesConflict.in";
    const toml::table inputDeck = toml::parse_file(fileName);
    try
    {
        const io::SimControls controls(inputDeck);
        std::cerr << "testSimControls: " << fileName
            << ": expected construction to throw, but it succeeded\n";
        return 1;
    }
    catch (const std::runtime_error&)
    {
        return 0;
    }
}

// Verify that constructing SimControls from a deck specifying only
// output.start_time and output.end_time, without output.ntime, throws.
static auto testSimControlsOutputTimesPartialRange() -> int
{
    const std::string fileName = "tests/io/assets/testControlsOutputTimesPartialRange.in";
    const toml::table inputDeck = toml::parse_file(fileName);
    try
    {
        const io::SimControls controls(inputDeck);
        std::cerr << "testSimControls: " << fileName
            << ": expected construction to throw, but it succeeded\n";
        return 1;
    }
    catch (const std::runtime_error&)
    {
        return 0;
    }
}

// Verify that constructing SimControls from a deck with an unrecognized
// outputs.output_mode value throws.
static auto testSimControlsInvalidOutputMode() -> int
{
    const std::string fileName = "tests/io/assets/testControlsInvalidOutputMode.in";
    const toml::table inputDeck = toml::parse_file(fileName);
    try
    {
        const io::SimControls controls(inputDeck);
        std::cerr << "testSimControls: " << fileName
            << ": expected construction to throw, but it succeeded\n";
        return 1;
    }
    catch (const std::runtime_error&)
    {
        return 0;
    }
}

// Verify that constructing SimControls from a deck with an unrecognized
// sim_type value throws.
static auto testSimControlsInvalidSimType() -> int
{
    const std::string fileName = "tests/io/assets/testControlsInvalidSimType.in";
    const toml::table inputDeck = toml::parse_file(fileName);
    try
    {
        const io::SimControls controls(inputDeck);
        std::cerr << "testSimControls: " << fileName
            << ": expected construction to throw, but it succeeded\n";
        return 1;
    }
    catch (const std::runtime_error&)
    {
        return 0;
    }
}

// Verify that a galaxy-type simulation reports simType() == galaxy and,
// with outputs.output_clusters unspecified, outputClusters() defaults
// to true.
static auto testSimControlsGalaxy() -> int
{
    const std::string fileName = "tests/io/assets/testControlsGalaxy.in";
    try
    {
        const toml::table inputDeck = toml::parse_file(fileName);
        const io::SimControls controls(inputDeck);

        if (controls.simType() != io::SimControls::SimType::galaxy)
        {
            std::cerr << "testSimControls: " << fileName
                << ": expected simType() == SimType::galaxy\n";
            return 1;
        }
        if (!controls.outputClusters())
        {
            std::cerr << "testSimControls: " << fileName
                << ": expected default outputClusters() == true\n";
            return 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSimControls: failed to parse valid input deck "
            << fileName << ": " << error.what() << "\n";
        return 1;
    }
    return 0;
}

// Verify that a galaxy-type simulation with outputs.output_clusters
// explicitly set to false reports outputClusters() == false.
static auto testSimControlsGalaxyNoClusters() -> int
{
    const std::string fileName = "tests/io/assets/testControlsGalaxyNoClusters.in";
    try
    {
        const toml::table inputDeck = toml::parse_file(fileName);
        const io::SimControls controls(inputDeck);

        if (controls.simType() != io::SimControls::SimType::galaxy)
        {
            std::cerr << "testSimControls: " << fileName
                << ": expected simType() == SimType::galaxy\n";
            return 1;
        }
        if (controls.outputClusters())
        {
            std::cerr << "testSimControls: " << fileName
                << ": expected outputClusters() == false\n";
            return 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSimControls: failed to parse valid input deck "
            << fileName << ": " << error.what() << "\n";
        return 1;
    }
    return 0;
}

// ---------------------------------------------------------------------
// Physics-settings tests (originally SimPhysics, merged into
// SimControls -- see this file's own header comment)
// ---------------------------------------------------------------------

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
        std::cerr << "testSimControls: " << label << ": IMF does not match "
            "expected Chabrier IMF; expectation value = "
            << imf.expectationValue() << ", expected = "
            << pdfCompare.expectationValue() << "\n";
        return 1;
    }
    if (imf.integral(0.5, 20) != pdfCompare.integral(0.5, 20))
    {
        std::cerr << "testSimControls: " << label << ": IMF does not match "
            "expected Chabrier IMF; integral over [0.5,20] = "
            << imf.integral(0.5, 20) << ", expected = "
            << pdfCompare.integral(0.5, 20) << "\n";
        return 1;
    }
    return 0;
}

// Verify that SimControls read and constructed usable stellar
// tracks. Both test decks specify the MIST_test track set with
// alphaFe = -0.2 and the default vvcrit = 0.0, at FeH = 0.0 (an
// exact point on that track set's own [Fe/H] grid), so this check
// is shared between them. This is deliberately far lighter than
// the Tracks3D unit tests in tests/tracks -- it only needs to
// confirm the tracks were read successfully and are usable, not
// exhaustively verify Tracks3D's own behavior.
static auto checkTracks(const io::SimControls& sim, const std::string& label) -> int
{
    const auto& tracks = sim.tracks();

    if (tracks.aFe() != -0.2 || tracks.vVcrit() != 0.0)
    {
        std::cerr << "testSimControls: " << label << ": tracks do not have "
            "expected aFe/vVcrit; aFe = " << tracks.aFe()
            << ", vVcrit = " << tracks.vVcrit() << "\n";
        return 1;
    }

    if (tracks.feH().size() != 1 || tracks.feH().front() != 0.0)
    {
        std::cerr << "testSimControls: " << label << ": tracks do not have "
            "the expected single [Fe/H] = 0.0 slice\n";
        return 1;
    }

    // Confirm the tracks are actually usable by requesting a
    // track for a mass within their range
    constexpr double mass = 1.0;
    const auto track = tracks.getTrack(mass, 0.0);
    if (!track || track->xMin() >= track->xMax())
    {
        std::cerr << "testSimControls: " << label << ": getTrack(" << mass
            << ", 0.0) did not return a usable track\n";
        return 1;
    }

    return 0;
}

// Test parsing of a cluster-type input deck's physics settings
static auto testSimControlsPhysicsCluster() -> int
{
    const std::string fileName = "tests/core/assets/testCluster.in";
    try
    {
        const toml::table inputDeck = toml::parse_file(fileName);
        const io::SimControls sim(inputDeck);

        if (sim.simType() != io::SimControls::SimType::cluster)
        {
            std::cerr << "testSimControls: " << fileName
                << ": expected simType() == cluster\n";
            return 1;
        }

        if (checkChabrierIMF(sim.imf(), fileName) != 0) { return 1; }

        if (sim.cmf().getMin() != 1e3 || sim.cmf().getMax() != 1e3 ||
            sim.cmf().expectationValue() != 1e3 || !sim.cmf().normalized())
        {
            std::cerr << "testSimControls: " << fileName
                << ": CMF does not match expected delta function at 1e3\n";
            return 1;
        }

        if (sim.fehDist().getMin() != 0.0 || sim.fehDist().getMax() != 0.0 ||
            sim.fehDist().expectationValue() != 0.0 || !sim.fehDist().normalized())
        {
            std::cerr << "testSimControls: " << fileName
                << ": [Fe/H] distribution does not match expected delta function at 0.0\n";
            return 1;
        }

        // Galaxy-only physics should not have been initialized
        if (sim.clf().valid() || sim.sfr().valid())
        {
            std::cerr << "testSimControls: " << fileName
                << ": CLF and SFR should not be initialized for a cluster simulation\n";
            return 1;
        }

        if (checkTracks(sim, fileName) != 0) { return 1; }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSimControls: failed to parse valid cluster input deck "
            << fileName << ": " << error.what() << "\n";
        return 1;
    }
    return 0;
}

// Test parsing of a galaxy-type input deck's physics settings
static auto testSimControlsPhysicsGalaxy() -> int
{
    const std::string fileName = "tests/core/assets/testGalaxy.in";
    try
    {
        const toml::table inputDeck = toml::parse_file(fileName);
        const io::SimControls sim(inputDeck);

        if (sim.simType() != io::SimControls::SimType::galaxy)
        {
            std::cerr << "testSimControls: " << fileName
                << ": expected simType() == galaxy\n";
            return 1;
        }

        if (checkChabrierIMF(sim.imf(), fileName) != 0) { return 1; }

        if (sim.cmf().getMin() != 1e3 || sim.cmf().getMax() != 1e3 ||
            sim.cmf().expectationValue() != 1e3 || !sim.cmf().normalized())
        {
            std::cerr << "testSimControls: " << fileName
                << ": CMF does not match expected delta function at 1e3\n";
            return 1;
        }

        if (sim.fehDist().getMin() != 0.0 || sim.fehDist().getMax() != 0.0 ||
            sim.fehDist().expectationValue() != 0.0 || !sim.fehDist().normalized())
        {
            std::cerr << "testSimControls: " << fileName
                << ": [Fe/H] distribution does not match expected delta function at 0.0\n";
            return 1;
        }

        if (sim.clf().getMin() != 1e300 || sim.clf().getMax() != 1e300 ||
            sim.clf().expectationValue() != 1e300 || !sim.clf().normalized())
        {
            std::cerr << "testSimControls: " << fileName
                << ": CLF does not match expected delta function at 1e300\n";
            return 1;
        }

        // The SFR was given as a bare number (1.0), so it should have
        // been turned into a non-normalized, constant-in-time PDF
        // spanning [0, tMax] (see SimControls.cpp's own
        // buildConstantSFR(), whose tMax is fixed at 1e15 yr -- far
        // beyond any realistic simulation timescale).
        constexpr double sfrTMax = 1e15;
        if (sim.sfr().getMin() != 0.0 ||
            sim.sfr().getMax() != sfrTMax ||
            sim.sfr().normalized())
        {
            std::cerr << "testSimControls: " << fileName
                << ": SFR does not match expected non-normalized constant PDF\n";
            return 1;
        }

        // Check the actual physical behavior this construction exists
        // to produce: integral(a, b) should equal sfr * (b - a) for
        // any interval well within [0, tMax]. A previous version of
        // this construction got the weight/tMax scaling backwards,
        // returning (b - a) / sfr instead -- undetected until now
        // because nothing checked the integral's actual value, only
        // the PDF's structural properties checked just above.
        constexpr double sfrValue = 1.0; // matches galaxy.sfr in the deck
        constexpr double intervalStart = 100.0;
        constexpr double intervalEnd = 300.0;
        const double expectedIntegral = sfrValue * (intervalEnd - intervalStart);
        const double actualIntegral = sim.sfr().integral(intervalStart, intervalEnd);
        if (!utils::approxEqual(actualIntegral, expectedIntegral))
        {
            std::cerr << "testSimControls: " << fileName << ": SFR integral("
                << intervalStart << ", " << intervalEnd << ") = " << actualIntegral
                << ", expected sfr * (b - a) = " << expectedIntegral << "\n";
            return 1;
        }

        if (checkTracks(sim, fileName) != 0) { return 1; }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSimControls: failed to parse valid galaxy input deck "
            << fileName << ": " << error.what() << "\n";
        return 1;
    }
    return 0;
}

// Verify that spectra.model is optional: a deck with no [spectra]
// table at all should construct successfully rather than throwing.
static auto testSimControlsNoSpectraModel() -> int
{
    const std::string fileName = "tests/core/assets/testCluster.in";
    try
    {
        toml::table inputDeck = toml::parse_file(fileName);
        inputDeck.erase("spectra");
        const io::SimControls sim(inputDeck);
        (void)sim;
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSimControls: " << fileName
            << ": expected construction with no spectra.model to succeed, but it threw: "
            << error.what() << "\n";
        return 1;
    }
    return 0;
}

// Verify that an unrecognized spectra.model value is rejected
static auto testSimControlsInvalidSpectraModel() -> int
{
    const std::string fileName = "tests/core/assets/testCluster.in";
    toml::table inputDeck = toml::parse_file(fileName);
    inputDeck.at_path("spectra").as_table()->insert_or_assign("model", std::string("not_a_model"));
    try
    {
        const io::SimControls sim(inputDeck);
        std::cerr << "testSimControls: " << fileName
            << ": expected construction with invalid spectra.model to throw, but it succeeded\n";
        return 1;
    }
    catch (const std::runtime_error&)
    {
        return 0;
    }
}

// Verify that a single-string spectra.model resolves to a working
// SpecsynLibNoWind, using a custom spectra.registry (the same test
// fixture tests/specsyn's own SpecsynLib tests use) and its BOSZ_test
// entry -- a non-WR-grid library, so this exercises readSpectra's
// SpecsynLibNoWind branch
static auto testSimControlsSpectraLibrary() -> int
{
    const std::string fileName = "tests/core/assets/testCluster.in";
    toml::table inputDeck = toml::parse_file(fileName);
    auto* spectraTable = inputDeck.at_path("spectra").as_table();
    spectraTable->insert_or_assign("registry", std::string("tests/specsyn/assets/spectra.toml"));
    spectraTable->insert_or_assign("model", std::string("BOSZ_test"));

    try
    {
        const io::SimControls sim(inputDeck);

        if (sim.specsyn() == nullptr)
        {
            std::cerr << "testSimControls: " << fileName
                << ": expected specsyn() to be populated for spectra.model = BOSZ_test\n";
            return 1;
        }
        if (sim.specsyn()->wl().empty())
        {
            std::cerr << "testSimControls: " << fileName
                << ": expected a non-empty wavelength grid for spectra.model = BOSZ_test\n";
            return 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSimControls: " << fileName
            << ": expected spectra.model = BOSZ_test (via a custom spectra.registry) "
            "to construct successfully, but it threw: " << error.what() << "\n";
        return 1;
    }
    return 0;
}

// Verify that an array-valued spectra.model resolves to a working
// SpecsynLibChained -- chaining TLUSTY_test and BOSZ_test, the same
// two non-WR-grid libraries used above, just as a priority-ordered
// list instead of a single name
static auto testSimControlsSpectraChained() -> int
{
    const std::string fileName = "tests/core/assets/testCluster.in";
    toml::table inputDeck = toml::parse_file(fileName);
    auto* spectraTable = inputDeck.at_path("spectra").as_table();
    spectraTable->insert_or_assign("registry", std::string("tests/specsyn/assets/spectra.toml"));
    spectraTable->insert_or_assign("model", toml::array{ "TLUSTY_test", "BOSZ_test" });

    try
    {
        const io::SimControls sim(inputDeck);

        if (sim.specsyn() == nullptr)
        {
            std::cerr << "testSimControls: " << fileName
                << ": expected specsyn() to be populated for a chained spectra.model\n";
            return 1;
        }
        if (sim.specsyn()->wl().empty())
        {
            std::cerr << "testSimControls: " << fileName
                << ": expected a non-empty wavelength grid for a chained spectra.model\n";
            return 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSimControls: " << fileName
            << ": expected an array-valued spectra.model to construct successfully, "
            "but it threw: " << error.what() << "\n";
        return 1;
    }
    return 0;
}

auto testSimControls() -> int
{
    int result = 0;
    result += testSimControlsDefaults();
    result += testSimControlsExplicit();
    result += testSimControlsOutputTimeDist();
    result += testSimControlsOutputTimesArray();
    result += testSimControlsOutputTimesLog();
    result += testSimControlsNoOutputs();
    result += testSimControlsOutputTimesConflict();
    result += testSimControlsOutputTimesPartialRange();
    result += testSimControlsInvalidOutputMode();
    result += testSimControlsInvalidSimType();
    result += testSimControlsGalaxy();
    result += testSimControlsGalaxyNoClusters();
    result += testSimControlsPhysicsCluster();
    result += testSimControlsPhysicsGalaxy();
    result += testSimControlsNoSpectraModel();
    result += testSimControlsInvalidSpectraModel();
    result += testSimControlsSpectraLibrary();
    result += testSimControlsSpectraChained();
    return result;
}
