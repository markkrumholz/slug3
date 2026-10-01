/**
 * @file testSimControls.cpp
 * @author Mark Krumholz
 * @brief Unit tests for the SimControls class.
 * @date 2026-07-16
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#include "../src/elem/IsotopeTable.hpp"
#include "../src/feedback/FeedbackCommons.hpp"
#include "../src/feedback/Winds.hpp"
#include "../src/io/SimControls.hpp"
#include "../src/pdfs/PDF.hpp"
#include "../src/pdfs/PDFSegment.hpp"
#include "../src/pdfs/PDFSegmentLognormal.hpp"
#include "../src/pdfs/PDFSegmentPowerlaw.hpp"
#include "../src/specsyn/SpecsynBlackbody.hpp"
#include "../src/specsyn/SpecsynLibNoWind.hpp"
#include "../src/tracks/Tracks3D.hpp"
#include "../src/utils/MiscUtils.hpp"
#include "../src/yields/YieldChannel.hpp"
#include "../src/yields/YieldCommons.hpp"
#include "../src/yields/Yields.hpp"
#include "testSimControls.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <numbers>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
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
// output_mode, out_dir, n_trial, and checkpoint_interval fall back to
// their documented defaults when not specified in the input deck, and
// that the start_time/end_time/ntime output option (linear spacing)
// is correctly expanded.
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
        if (controls.nTrial() != 1)
        {
            std::cerr << "testSimControls: " << fileName
                << ": expected default nTrial() == 1, got "
                << controls.nTrial() << "\n";
            return 1;
        }
        if (controls.checkpointInterval() != 0)
        {
            std::cerr << "testSimControls: " << fileName
                << ": expected default checkpointInterval() == 0, got "
                << controls.checkpointInterval() << "\n";
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
// are correctly read from the input deck when explicitly specified.
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
// output.output_mode value throws.
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

// Verify that output.checkpoint_interval is read correctly from a
// deck with h5 output (the only output mode checkpointing supports --
// see testSimControlsCheckpointAsciiThrows for the rejected
// combination).
static auto testSimControlsCheckpointInterval() -> int
{
    const std::string fileName = "tests/core/assets/testCluster.in";
    try
    {
        toml::table inputDeck = toml::parse_file(fileName);
        if (toml::table* outputTbl = inputDeck["output"].as_table())
        { outputTbl->insert_or_assign("checkpoint_interval", static_cast<int64_t>(5)); }
        else
        {
            inputDeck.insert("output",
                toml::table{ { "checkpoint_interval", static_cast<int64_t>(5) } });
        }
        const io::SimControls controls(inputDeck);

        if (controls.checkpointInterval() != 5)
        {
            std::cerr << "testSimControls: " << fileName
                << ": expected checkpointInterval() == 5, got "
                << controls.checkpointInterval() << "\n";
            return 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSimControls: checkpointInterval test failed: "
            << error.what() << "\n";
        return 1;
    }
    return 0;
}

// Verify that constructing SimControls from a deck with a non-zero
// output.checkpoint_interval combined with ascii output throws --
// checkpointing is only supported with HDF5 output (see
// OutputManagerH5::checkpoint()'s own comment for why: unlike
// OutputManagerH5 rolling over to a new HDF5 file, OutputManagerAscii
// has no way to reopen/append to an ascii file it has already
// finished writing).
static auto testSimControlsCheckpointAsciiThrows() -> int
{
    const std::string fileName = "tests/io/assets/testControlsCheckpointAsciiConflict.in";
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

// Verify that a galaxy-type simulation reports simType() == galaxy.
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
    const auto tracks = sim.tracks();

    if (tracks->aFe() != -0.2 || tracks->vVcrit() != 0.0)
    {
        std::cerr << "testSimControls: " << label << ": tracks do not have "
            "expected aFe/vVcrit; aFe = " << tracks->aFe()
            << ", vVcrit = " << tracks->vVcrit() << "\n";
        return 1;
    }

    if (tracks->feH().size() != 1 || tracks->feH().front() != 0.0)
    {
        std::cerr << "testSimControls: " << label << ": tracks do not have "
            "the expected single [Fe/H] = 0.0 slice\n";
        return 1;
    }

    // Confirm the tracks are actually usable by requesting a
    // track for a mass within their range
    constexpr double mass = 1.0;
    const auto track = tracks->getTrack(mass, 0.0);
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

        if (sim.fCluster() != 1.0)
        {
            std::cerr << "testSimControls: " << fileName
                << ": expected default fCluster() == 1.0, got " << sim.fCluster() << "\n";
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

        if (sim.fCluster() != 1.0)
        {
            std::cerr << "testSimControls: " << fileName
                << ": expected default fCluster() == 1.0, got " << sim.fCluster() << "\n";
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

// Verify that clusters.f_cluster is read for a galaxy-type simulation
// when given explicitly, and ignored (fCluster() stays at its default
// of 1.0) when the same key is given on a cluster-type deck instead --
// mirrors testSimControlsPhysicsCluster()/testSimControlsPhysicsGalaxy()'s
// own default-value checks, but for an explicitly-set, non-default
// value.
static auto testSimControlsFCluster() -> int
{
    constexpr double fClusterValue = 0.6;

    try
    {
        toml::table galaxyDeck = toml::parse_file("tests/core/assets/testGalaxy.in");
        galaxyDeck.at_path("clusters").as_table()->insert("f_cluster", fClusterValue);
        const io::SimControls galaxySim(galaxyDeck);
        if (!utils::approxEqual(galaxySim.fCluster(), fClusterValue))
        {
            std::cerr << "testSimControls: fCluster: expected galaxy fCluster() == "
                << fClusterValue << ", got " << galaxySim.fCluster() << "\n";
            return 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSimControls: fCluster: failed to parse valid galaxy "
            "input deck: " << error.what() << "\n";
        return 1;
    }

    try
    {
        toml::table clusterDeck = toml::parse_file("tests/core/assets/testCluster.in");
        if (toml::table* clustersTbl = clusterDeck["clusters"].as_table())
        { clustersTbl->insert("f_cluster", fClusterValue); }
        else { clusterDeck.insert("clusters", toml::table{ { "f_cluster", fClusterValue } }); }
        const io::SimControls clusterSim(clusterDeck);
        if (clusterSim.fCluster() != 1.0)
        {
            std::cerr << "testSimControls: fCluster: expected fCluster() == 1.0 "
                "(clusters.f_cluster is only read for a galaxy-type simulation), got "
                << clusterSim.fCluster() << "\n";
            return 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSimControls: fCluster: failed to parse valid cluster "
            "input deck: " << error.what() << "\n";
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

// Verify that nebular.compute_neb defaults to false, rather than
// crashing, when there is no spectral synthesizer to pair a nebular
// emission grid with: a deck with neither a [spectra] nor a [nebular]
// table must construct, with nebular() left null.
static auto testSimControlsNebularDefaultNoSpectra() -> int
{
    const std::string fileName = "tests/core/assets/testCluster.in";
    try
    {
        toml::table inputDeck = toml::parse_file(fileName);
        inputDeck.erase("spectra");
        inputDeck.erase("nebular");
        const io::SimControls sim(inputDeck);
        if (sim.specsyn() != nullptr)
        {
            std::cerr << "testSimControls: nebularDefaultNoSpectra: expected specsyn() "
                "to be null with no spectra.model\n";
            return 1;
        }
        if (sim.nebular() != nullptr || sim.nebControls().computeNeb_)
        {
            std::cerr << "testSimControls: nebularDefaultNoSpectra: expected nebular() "
                "to be null and nebControls().computeNeb_ to be false when "
                "nebular.compute_neb is not given and there is no spectral synthesizer\n";
            return 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSimControls: nebularDefaultNoSpectra: expected construction "
            "with neither [spectra] nor [nebular] to succeed, but it threw: "
            << error.what() << "\n";
        return 1;
    }
    return 0;
}

// Verify that explicitly setting nebular.compute_neb = true with no
// spectral synthesizer is rejected with a clear error, while
// explicitly setting it false is accepted
static auto testSimControlsNebularExplicitNoSpectra() -> int
{
    const std::string fileName = "tests/core/assets/testCluster.in";

    {
        toml::table inputDeck = toml::parse_file(fileName);
        inputDeck.erase("spectra");
        inputDeck.at_path("nebular").as_table()->insert_or_assign("compute_neb", true);
        try
        {
            const io::SimControls sim(inputDeck);
            std::cerr << "testSimControls: nebularExplicitNoSpectra: expected "
                "nebular.compute_neb = true with no spectral synthesizer to throw, "
                "but construction succeeded\n";
            return 1;
        }
        catch (const std::runtime_error& error)
        {
            if (std::string(error.what()).find("compute_neb") == std::string::npos)
            {
                std::cerr << "testSimControls: nebularExplicitNoSpectra: expected the "
                    "error to mention compute_neb, got: " << error.what() << "\n";
                return 1;
            }
        }
    }

    {
        toml::table inputDeck = toml::parse_file(fileName);
        inputDeck.erase("spectra");
        inputDeck.at_path("nebular").as_table()->insert_or_assign("compute_neb", false);
        try
        {
            const io::SimControls sim(inputDeck);
            if (sim.nebular() != nullptr || sim.nebControls().computeNeb_)
            {
                std::cerr << "testSimControls: nebularExplicitNoSpectra: expected "
                    "nebular.compute_neb = false to leave nebular() null\n";
                return 1;
            }
        }
        catch (const std::exception& error)
        {
            std::cerr << "testSimControls: nebularExplicitNoSpectra: expected "
                "nebular.compute_neb = false with no spectral synthesizer to succeed, "
                "but it threw: " << error.what() << "\n";
            return 1;
        }
    }
    return 0;
}

// Verify that nebular.compute_neb still defaults to true when a
// spectral synthesizer is available: a deck with [spectra] but no
// nebular.compute_neb key builds a Nebular (using the small synthetic
// fixture tests/nebular/assets/nebular_test.h5, via nebular.table)
static auto testSimControlsNebularDefaultWithSpectra() -> int
{
    const std::string fileName = "tests/nebular/assets/testNebular.in";
    try
    {
        toml::table inputDeck = toml::parse_file(fileName);
        inputDeck.at_path("nebular").as_table()->erase("compute_neb");
        const io::SimControls sim(inputDeck);
        if (sim.nebular() == nullptr || !sim.nebControls().computeNeb_)
        {
            std::cerr << "testSimControls: nebularDefaultWithSpectra: expected nebular() "
                "to be non-null and nebControls().computeNeb_ to be true when "
                "nebular.compute_neb is not given and a spectral synthesizer is available\n";
            return 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSimControls: nebularDefaultWithSpectra: expected construction "
            "to succeed, but it threw: " << error.what() << "\n";
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

// Verify that a library-based spectral synthesizer is loaded over the
// [Fe/H] range the tracks were requested for, not their padded grid:
// with stars.FeH flat in [-0.5, 0.5], MIST_test's own loaded grid pads
// out to -1 (for the tracks' own interpolation), but BOSZ_test (with
// planes every 0.25 dex) should hold data only for [-0.5, 0.5], the
// only [Fe/H] spectra can ever be requested at.
static auto testSimControlsSpectraSkipTrackPadding() -> int
{
    const std::string fileName = "tests/core/assets/testCluster.in";
    toml::table inputDeck = toml::parse_file(fileName);
    inputDeck.at_path("stars").as_table()->insert_or_assign(
        "FeH", "tests/core/assets/testClusterFeHDist.toml");
    auto* spectraTable = inputDeck.at_path("spectra").as_table();
    spectraTable->insert_or_assign("registry", std::string("tests/specsyn/assets/spectra.toml"));
    spectraTable->insert_or_assign("model", std::string("BOSZ_test"));
    try
    {
        const io::SimControls sim(inputDeck);
        if (!(sim.tracks()->feH().front() < -0.5))
        {
            std::cerr << "testSimControls: spectraSkipTrackPadding: test bug: expected the "
                "tracks' own grid to be padded below -0.5, got " << sim.tracks()->feH().front()
                << "\n";
            return 1;
        }
        if (sim.specsyn()->fehMin() != -0.5 || sim.specsyn()->fehMax() != 0.5)
        {
            std::cerr << "testSimControls: spectraSkipTrackPadding: expected BOSZ_test loaded "
                "over [-0.5, 0.5], got [" << sim.specsyn()->fehMin() << ", "
                << sim.specsyn()->fehMax() << "]\n";
            return 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSimControls: spectraSkipTrackPadding: unexpected exception: "
            << error.what() << "\n";
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

// Verify that SimControls::readExtinct() symmetrically defaults
// whichever of avDist()/avDistField() was not given an explicit
// distribution to a valid delta function PDF at 0, so the two are
// always either both valid or both invalid, never just one -- see
// readExtinct()'s own comment for why.
static auto testSimControlsExtinctField() -> int
{
    constexpr double tightTol = 1e-9;
    constexpr std::string_view baseDeck = "tests/core/assets/testGalaxy.in";

    // Neither extinct.AV nor extinct.AV_field given: both invalid,
    // extinct() null -- unchanged from before extinct.AV_field existed
    try
    {
        const toml::table inputDeck = toml::parse_file(baseDeck);
        const io::SimControls controls(inputDeck);
        if (controls.avDist().valid() || controls.avDistField().valid() ||
            controls.extinct() != nullptr)
        {
            std::cerr << "testSimControls: extinctField: expected avDist()/"
                "avDistField() invalid and extinct() null when neither "
                "extinct.AV nor extinct.AV_field is given\n";
            return 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSimControls: extinctField: neither-given case "
            "failed: " << error.what() << "\n";
        return 1;
    }

    // extinct.AV given, extinct.AV_field not: avDist() the real
    // distribution, avDistField() a valid delta at 0
    try
    {
        toml::table inputDeck = toml::parse_file(baseDeck);
        inputDeck.insert("extinct", toml::table{
            { "AV", 1.0 }, { "model", "Calzetti_starburst" } });
        const io::SimControls controls(inputDeck);

        if (controls.extinct() == nullptr)
        {
            std::cerr << "testSimControls: extinctField: expected extinct() "
                "non-null when extinct.AV is given\n";
            return 1;
        }
        if (!controls.avDist().valid() ||
            std::abs(controls.avDist().draw() - 1.0) > tightTol)
        {
            std::cerr << "testSimControls: extinctField: expected avDist() "
                "to always draw 1.0, got " << controls.avDist().draw() << "\n";
            return 1;
        }
        if (!controls.avDistField().valid() ||
            std::abs(controls.avDistField().draw()) > tightTol)
        {
            std::cerr << "testSimControls: extinctField: expected "
                "avDistField() to be a valid delta at 0 when only "
                "extinct.AV is given, got " << controls.avDistField().draw() << "\n";
            return 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSimControls: extinctField: AV-only case failed: "
            << error.what() << "\n";
        return 1;
    }

    // extinct.AV_field given, extinct.AV not: avDistField() the real
    // distribution, avDist() a valid delta at 0
    try
    {
        toml::table inputDeck = toml::parse_file(baseDeck);
        inputDeck.insert("extinct", toml::table{
            { "AV_field", 2.0 }, { "model", "Calzetti_starburst" } });
        const io::SimControls controls(inputDeck);

        if (controls.extinct() == nullptr)
        {
            std::cerr << "testSimControls: extinctField: expected extinct() "
                "non-null when extinct.AV_field is given\n";
            return 1;
        }
        if (!controls.avDistField().valid() ||
            std::abs(controls.avDistField().draw() - 2.0) > tightTol)
        {
            std::cerr << "testSimControls: extinctField: expected "
                "avDistField() to always draw 2.0, got "
                << controls.avDistField().draw() << "\n";
            return 1;
        }
        if (!controls.avDist().valid() ||
            std::abs(controls.avDist().draw()) > tightTol)
        {
            std::cerr << "testSimControls: extinctField: expected avDist() "
                "to be a valid delta at 0 when only extinct.AV_field is "
                "given, got " << controls.avDist().draw() << "\n";
            return 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSimControls: extinctField: AV_field-only case "
            "failed: " << error.what() << "\n";
        return 1;
    }

    // extinct.AV_field given without extinct.model should throw, exactly
    // as extinct.AV given without extinct.model already does
    try
    {
        toml::table inputDeck = toml::parse_file(baseDeck);
        inputDeck.insert("extinct", toml::table{ { "AV_field", 1.0 } });
        const io::SimControls controls(inputDeck);
        std::cerr << "testSimControls: extinctField: expected an exception "
            "when extinct.AV_field is given without extinct.model\n";
        return 1;
    }
    catch (const std::runtime_error&) { /* expected */ }

    return 0;
}

// Verify SimControls::readYields()'s own parsing of yields.channel1,
// yields.channel2, etc. into yieldChannels(), and of yields.registry
// into the Yields it builds from them.
static auto testSimControlsYields() -> int
{
    constexpr std::string_view baseDeck = "tests/core/assets/testGalaxy.in";
    int result = 0;

    // No yields.channelN table at all: yieldChannels() empty, yields() null
    try
    {
        const toml::table inputDeck = toml::parse_file(baseDeck);
        const io::SimControls controls(inputDeck);
        if (!controls.yieldChannels().empty() || controls.yields() != nullptr)
        {
            std::cerr << "testSimControls: yields: expected yieldChannels() empty "
                "and yields() null when no yields.channelN table is given\n";
            result = 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSimControls: yields: no-channels case failed: "
            << error.what() << "\n";
        result = 1;
    }

    // Two channels (both real models in the default registry, so
    // Yields::Yields() -- called from readYields() -- actually builds
    // both), the second with m_min/m_max extrapolating past
    // sukhbold16's own native [9.0, 120.0] mass range on both ends,
    // plus an explicit (if here, redundant with the default) registry
    try
    {
        toml::table inputDeck = toml::parse_file(baseDeck);
        inputDeck.insert("yields", toml::table{
            { "channel1", toml::table{
                { "channel", "ccsn" }, { "model", "sukhbold16" } } },
            { "channel2", toml::table{
                { "channel", "massive_star_winds" }, { "model", "sukhbold16" },
                { "m_min", 8.0 }, { "m_max", 150.0 } } },
            { "registry", yields::defaultRegistry },
        });
        const io::SimControls controls(inputDeck);

        const auto& channels = controls.yieldChannels();
        if (channels.size() != 2)
        {
            std::cerr << "testSimControls: yields: expected 2 yieldChannels(), got "
                << channels.size() << "\n";
            result = 1;
        }
        else
        {
            if (channels[0].channel_ != yields::Channel::ccsn_ ||
                channels[0].modelName_ != "sukhbold16" ||
                channels[0].mMin_.has_value() || channels[0].mMax_.has_value())
            {
                std::cerr << "testSimControls: yields: unexpected yieldChannels()[0]\n";
                result = 1;
            }
            if (channels[1].channel_ != yields::Channel::massiveStarWinds_ ||
                channels[1].modelName_ != "sukhbold16" ||
                !channels[1].mMin_.has_value() || channels[1].mMin_.value() != 8.0 ||
                !channels[1].mMax_.has_value() || channels[1].mMax_.value() != 150.0)
            {
                std::cerr << "testSimControls: yields: unexpected yieldChannels()[1]\n";
                result = 1;
            }
        }

        if (controls.yields() == nullptr)
        {
            std::cerr << "testSimControls: yields: expected yields() non-null "
                "when yields.channel1 is given\n";
            result = 1;
        }
        else
        {
            if (controls.yields()->registryName() != yields::defaultRegistry)
            {
                std::cerr << "testSimControls: yields: expected yields()->registryName() "
                    "== yields::defaultRegistry, got \"" << controls.yields()->registryName() << "\"\n";
                result = 1;
            }

            const auto& loaded = controls.yields()->yieldChannels();
            if (loaded.size() != 2)
            {
                std::cerr << "testSimControls: yields: expected yields()->yieldChannels() "
                    "to have size 2, got " << loaded.size() << "\n";
                result = 1;
            }
            else
            {
                if (loaded[0]->channel() != yields::Channel::ccsn_)
                {
                    std::cerr << "testSimControls: yields: yields()->yieldChannels()[0] "
                        "has the wrong channel()\n";
                    result = 1;
                }
                if (loaded[1]->channel() != yields::Channel::massiveStarWinds_ ||
                    !loaded[1]->hasYield(8.0) || !loaded[1]->hasYield(150.0))
                {
                    std::cerr << "testSimControls: yields: yields()->yieldChannels()[1] "
                        "has the wrong channel(), or doesn't cover the requested "
                        "[8.0, 150.0] extrapolated mass range\n";
                    result = 1;
                }
            }
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSimControls: yields: two-channels case failed: "
            << error.what() << "\n";
        result = 1;
    }

    // A channel naming a model that doesn't exist in the registry: not
    // checked by readYields() itself (see its own comment), but Yields'
    // constructor builds the real YieldChannel eagerly, so this throws
    // once SimControls actually gets there.
    try
    {
        toml::table inputDeck = toml::parse_file(baseDeck);
        inputDeck.insert("yields", toml::table{
            { "channel1", toml::table{
                { "channel", "ccsn" }, { "model", "not_a_real_model" } } },
        });
        const io::SimControls controls(inputDeck);
        std::cerr << "testSimControls: yields: expected an exception for a "
            "yields.channel1.model that doesn't exist in the registry\n";
        result = 1;
    }
    catch (const std::runtime_error&) { /* expected */ }

    // One channel, no yields.registry given: yields()->registryName()
    // falls back to yields::defaultRegistry
    try
    {
        toml::table inputDeck = toml::parse_file(baseDeck);
        inputDeck.insert("yields", toml::table{
            { "channel1", toml::table{
                { "channel", "ccsn" }, { "model", "sukhbold16" } } },
        });
        const io::SimControls controls(inputDeck);

        if (controls.yields() == nullptr ||
            controls.yields()->registryName() != yields::defaultRegistry)
        {
            std::cerr << "testSimControls: yields: expected yields()->registryName() "
                "to fall back to yields::defaultRegistry when yields.registry is absent\n";
            result = 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSimControls: yields: default-registry case failed: "
            << error.what() << "\n";
        result = 1;
    }

    // yields.channel1.channel not one of channelStr's own entries: throws
    try
    {
        toml::table inputDeck = toml::parse_file(baseDeck);
        inputDeck.insert("yields", toml::table{
            { "channel1", toml::table{
                { "channel", "not_a_real_channel" }, { "model", "sukhbold16" } } },
        });
        const io::SimControls controls(inputDeck);
        std::cerr << "testSimControls: yields: expected an exception for an "
            "unrecognized yields.channel1.channel\n";
        result = 1;
    }
    catch (const std::runtime_error&) { /* expected */ }

    // yields.channel1 missing its required "model" keyword: throws
    try
    {
        toml::table inputDeck = toml::parse_file(baseDeck);
        inputDeck.insert("yields", toml::table{
            { "channel1", toml::table{ { "channel", "ccsn" } } },
        });
        const io::SimControls controls(inputDeck);
        std::cerr << "testSimControls: yields: expected an exception when "
            "yields.channel1.model is missing\n";
        result = 1;
    }
    catch (const std::runtime_error&) { /* expected */ }

    // yields.channel1 missing its required "channel" keyword: throws
    try
    {
        toml::table inputDeck = toml::parse_file(baseDeck);
        inputDeck.insert("yields", toml::table{
            { "channel1", toml::table{ { "model", "sukhbold16" } } },
        });
        const io::SimControls controls(inputDeck);
        std::cerr << "testSimControls: yields: expected an exception when "
            "yields.channel1.channel is missing\n";
        result = 1;
    }
    catch (const std::runtime_error&) { /* expected */ }

    return result;
}

// Verify noDecay() defaults to false, is parsed from the optional
// yields.no_decay key (read regardless of whether yieldChannels_ ends
// up empty, like yields.channel_decomposed), and that setNoDecay()
// updates it directly.
static auto testSimControlsYieldsNoDecay() -> int
{
    constexpr std::string_view baseDeck = "tests/core/assets/testGalaxy.in";
    int result = 0;

    // Default: false, with no yields.channelN table at all
    try
    {
        const toml::table inputDeck = toml::parse_file(baseDeck);
        const io::SimControls controls(inputDeck);
        if (controls.noDecay())
        {
            std::cerr << "testSimControls: yieldsNoDecay: expected noDecay() "
                "== false by default (no [yields] table at all)\n";
            result = 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSimControls: yieldsNoDecay: no-yields-table case "
            "failed: " << error.what() << "\n";
        result = 1;
    }

    // Default: false, with a real [yields] table but no explicit no_decay
    try
    {
        toml::table inputDeck = toml::parse_file(baseDeck);
        inputDeck.insert("yields", toml::table{
            { "channel1", toml::table{
                { "channel", "ccsn" }, { "model", "sukhbold_test" } } },
            { "registry", "tests/yields/assets/yields.toml" },
        });
        const io::SimControls controls(inputDeck);
        if (controls.noDecay())
        {
            std::cerr << "testSimControls: yieldsNoDecay: expected noDecay() "
                "== false by default (yields.no_decay not given)\n";
            result = 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSimControls: yieldsNoDecay: default-with-channels "
            "case failed: " << error.what() << "\n";
        result = 1;
    }

    // Explicit yields.no_decay = true
    try
    {
        toml::table inputDeck = toml::parse_file(baseDeck);
        inputDeck.insert("yields", toml::table{
            { "channel1", toml::table{
                { "channel", "ccsn" }, { "model", "sukhbold_test" } } },
            { "registry", "tests/yields/assets/yields.toml" },
            { "no_decay", true },
        });
        io::SimControls controls(inputDeck);
        if (!controls.noDecay())
        {
            std::cerr << "testSimControls: yieldsNoDecay: expected noDecay() "
                "== true when yields.no_decay = true\n";
            result = 1;
        }

        // setNoDecay() must update it live
        controls.setNoDecay(false);
        if (controls.noDecay())
        {
            std::cerr << "testSimControls: yieldsNoDecay: expected noDecay() "
                "== false after setNoDecay(false)\n";
            result = 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSimControls: yieldsNoDecay: explicit-true case "
            "failed: " << error.what() << "\n";
        result = 1;
    }

    return result;
}

// Verify writeClusterYields()/writeGalaxyYields() default to true, are
// parsed from output.write_cluster_yields/output.write_galaxy_yields
// exactly like the other six output.write_* keys (see readOutput()'s
// own comment), and that readYields() throws if yield channels were
// requested but both are false, or if writeClusterYields() alone is
// false in a cluster-type simulation (where writeGalaxyYields() is
// meaningless and so cannot rescue the yields from going unwritten).
static auto testSimControlsWriteYields() -> int
{
    constexpr std::string_view galaxyDeck = "tests/core/assets/testGalaxy.in";
    constexpr std::string_view clusterDeck = "tests/core/assets/testCluster.in";
    int result = 0;

    // Defaults: both true when no yields.channelN and no write_*_yields
    // key are given at all
    try
    {
        const toml::table inputDeck = toml::parse_file(galaxyDeck);
        const io::SimControls controls(inputDeck);
        if (!controls.writeClusterYields() || !controls.writeGalaxyYields())
        {
            std::cerr << "testSimControls: write yields: expected both "
                "writeClusterYields() and writeGalaxyYields() to default "
                "to true\n";
            result = 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSimControls: write yields: defaults case failed: "
            << error.what() << "\n";
        result = 1;
    }

    // output.write_cluster_yields = false, no yield channels requested:
    // parsed straight through to writeClusterYields(), no throw (the
    // sanity check only fires when yieldChannels() is non-empty)
    try
    {
        toml::table inputDeck = toml::parse_file(galaxyDeck);
        inputDeck.at_path("output").as_table()->insert("write_cluster_yields", false);
        const io::SimControls controls(inputDeck);
        if (controls.writeClusterYields() || !controls.writeGalaxyYields())
        {
            std::cerr << "testSimControls: write yields: expected "
                "writeClusterYields() false, writeGalaxyYields() true, "
                "after setting only output.write_cluster_yields = false\n";
            result = 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSimControls: write yields: no-channels opt-out "
            "case failed: " << error.what() << "\n";
        result = 1;
    }

    // Yield channels requested, output.write_cluster_yields and
    // output.write_galaxy_yields both false: throws
    try
    {
        toml::table inputDeck = toml::parse_file(galaxyDeck);
        inputDeck.insert("yields", toml::table{
            { "channel1", toml::table{
                { "channel", "ccsn" }, { "model", "sukhbold16" } } },
        });
        inputDeck.at_path("output").as_table()->insert("write_cluster_yields", false);
        inputDeck.at_path("output").as_table()->insert("write_galaxy_yields", false);
        const io::SimControls controls(inputDeck);
        std::cerr << "testSimControls: write yields: expected an exception "
            "when yield channels are given but both write_cluster_yields "
            "and write_galaxy_yields are false\n";
        result = 1;
    }
    catch (const std::runtime_error&) { /* expected */ }

    // Yield channels requested, only output.write_cluster_yields false,
    // in a galaxy-type simulation: no throw, since writeGalaxyYields()
    // (still true) can carry the yields
    try
    {
        toml::table inputDeck = toml::parse_file(galaxyDeck);
        inputDeck.insert("yields", toml::table{
            { "channel1", toml::table{
                { "channel", "ccsn" }, { "model", "sukhbold16" } } },
        });
        inputDeck.at_path("output").as_table()->insert("write_cluster_yields", false);
        const io::SimControls controls(inputDeck);
        if (controls.writeClusterYields() || !controls.writeGalaxyYields())
        {
            std::cerr << "testSimControls: write yields: expected "
                "writeClusterYields() false, writeGalaxyYields() true, "
                "after setting only output.write_cluster_yields = false\n";
            result = 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSimControls: write yields: galaxy-sim opt-out "
            "case failed: " << error.what() << "\n";
        result = 1;
    }

    // Yield channels requested, only output.write_cluster_yields false,
    // in a cluster-type simulation: throws, since writeGalaxyYields()
    // is meaningless there (no Galaxy, so no galaxy_yields group/file)
    // and so cannot rescue the yields the way it could for a galaxy-type
    // simulation
    try
    {
        toml::table inputDeck = toml::parse_file(clusterDeck);
        inputDeck.insert("yields", toml::table{
            { "channel1", toml::table{
                { "channel", "ccsn" }, { "model", "sukhbold16" } } },
        });
        inputDeck.at_path("output").as_table()->insert("write_cluster_yields", false);
        const io::SimControls controls(inputDeck);
        std::cerr << "testSimControls: write yields: expected an exception "
            "when yield channels are given and write_cluster_yields is "
            "false in a cluster-type simulation\n";
        result = 1;
    }
    catch (const std::runtime_error&) { /* expected */ }

    return result;
}

// Verify Yields::isotopes() is the deduplicated, sorted union of every
// loaded channel's own isotopesOrig(), and that Yields::rebuildYieldGrid()
// (called by the constructor) actually pushes that same list back down
// into every channel -- both its isotopes() (in the same order) and its
// yield() values (remapped, with 0 for any isotope that channel's own
// model doesn't tabulate). Uses the small yields test fixture
// (tests/yields/assets/yields.toml, see make_yields_test_fixture.py)
// rather than the real registry: its "sukhbold_test" (h1, fe56, ni56)
// and "kobayashi_test" (h1, fe56, ni58) models deliberately share some
// isotopes and differ in one, so this actually exercises
// deduplication and zero-backfill -- two channels of the very same
// real model would share their whole isotope_z/isotope_a datasets
// outright and so could never expose either kind of bug.
static auto testSimControlsYieldsIsotopes() -> int
{
    constexpr std::string_view baseDeck = "tests/core/assets/testGalaxy.in";
    int result = 0;

    try
    {
        toml::table inputDeck = toml::parse_file(baseDeck);
        inputDeck.insert("yields", toml::table{
            { "channel1", toml::table{
                { "channel", "ccsn" }, { "model", "sukhbold_test" } } },
            { "channel2", toml::table{
                { "channel", "ccsn" }, { "model", "kobayashi_test" } } },
            { "registry", "tests/yields/assets/yields.toml" },
            { "min_isotope_lifetime", 0.0 }, // track Co56/Ni56 explicitly rather than skipping them
        });
        const io::SimControls controls(inputDeck);

        if (controls.yields() == nullptr)
        {
            std::cerr << "testSimControls: yieldsIsotopes: expected yields() non-null\n";
            return 1;
        }

        const auto& isotopes = controls.yields()->isotopes();
        // Co56 (27, 56) is not directly tabulated by either model, but
        // Ni56 is unstable and decays into it, so rebuildYieldGrid()'s
        // own force-expansion adds it -- see its own comment.
        const std::vector<std::pair<unsigned int, unsigned int>> expected{
            { 1, 1 }, { 26, 56 }, { 27, 56 }, { 28, 56 }, { 28, 58 } }; // h1, fe56, co56, ni56, ni58
        if (isotopes.size() != expected.size())
        {
            std::cerr << "testSimControls: yieldsIsotopes: expected " << expected.size() <<
                " isotopes (h1, fe56, co56, ni56, ni58), got " << isotopes.size() << "\n";
            result = 1;
        }
        else
        {
            for (std::size_t i = 0; i < expected.size(); ++i)
            {
                if (isotopes[i].get().Z() != expected[i].first ||
                    isotopes[i].get().A() != expected[i].second)
                {
                    std::cerr << "testSimControls: yieldsIsotopes: isotopes()[" << i <<
                        "] expected (Z=" << expected[i].first << ", A=" << expected[i].second <<
                        "), got (Z=" << isotopes[i].get().Z() << ", A=" <<
                        isotopes[i].get().A() << ")\n";
                    result = 1;
                }
            }
        }

        const auto& loaded = controls.yields()->yieldChannels();
        if (loaded.size() != 2)
        {
            std::cerr << "testSimControls: yieldsIsotopes: expected 2 yieldChannels(), got "
                << loaded.size() << "\n";
            return result | 1;
        }

        // Both channels must be synchronized onto the exact same
        // isotopes() -- Yields::isotopes() itself, in the same order
        for (const auto* label : { "sukhbold_test", "kobayashi_test" })
        {
            const auto& channelIsotopes = (label == std::string_view{ "sukhbold_test" }) ?
                loaded[0]->isotopes() : loaded[1]->isotopes();
            if (channelIsotopes.size() != isotopes.size())
            {
                std::cerr << "testSimControls: yieldsIsotopes: " << label <<
                    " channel's own isotopes() was not synchronized to yields()->isotopes()\n";
                result = 1;
                continue;
            }
            for (std::size_t i = 0; i < isotopes.size(); ++i)
            {
                if (channelIsotopes[i].get() != isotopes[i].get())
                {
                    std::cerr << "testSimControls: yieldsIsotopes: " << label <<
                        " channel's own isotopes()[" << i << "] does not match "
                        "yields()->isotopes()[" << i << "]\n";
                    result = 1;
                }
            }
        }

        // sukhbold_test's own native h1/fe56/ni56 values at its own
        // exact grid mass 18.2, remapped onto [h1, fe56, co56, ni56,
        // ni58] -- co56 (index 2, force-expanded) and ni58 (index 4)
        // must both be exactly 0, since sukhbold_test never tabulated
        // either directly
        constexpr double yieldTol = 1e-10;
        const std::vector<double> sukhboldExpected{ 5.93, 8.46e-2, 0.0, 7.02e-2, 0.0 };
        const auto sukhboldActual = loaded[0]->yield(18.2, 0.0);
        if (sukhboldActual.size() != sukhboldExpected.size())
        {
            std::cerr << "testSimControls: yieldsIsotopes: sukhbold_test yield() has "
                "size " << sukhboldActual.size() << ", expected " << sukhboldExpected.size() << "\n";
            result = 1;
        }
        for (std::size_t i = 0; i < std::min(sukhboldActual.size(), sukhboldExpected.size()); ++i)
        {
            if (std::abs(sukhboldActual[i] - sukhboldExpected[i]) > yieldTol)
            {
                std::cerr << "testSimControls: yieldsIsotopes: sukhbold_test yield()[" << i <<
                    "] = " << sukhboldActual[i] << ", expected " << sukhboldExpected[i] << "\n";
                result = 1;
            }
        }

        // kobayashi_test's own native h1/fe56/ni58 values at its own
        // exact grid mass 13.0, remapped onto [h1, fe56, co56, ni56,
        // ni58] -- co56 (index 2, force-expanded) and ni56 (index 3)
        // must both be exactly 0, since kobayashi_test never tabulated
        // either directly
        const std::vector<double> kobayashiExpected{ 6.16, 8.32e-2, 0.0, 0.0, 2.23e-3 };
        const auto kobayashiActual = loaded[1]->yield(13.0, 0.0);
        if (kobayashiActual.size() != kobayashiExpected.size())
        {
            std::cerr << "testSimControls: yieldsIsotopes: kobayashi_test yield() has "
                "size " << kobayashiActual.size() << ", expected " << kobayashiExpected.size() << "\n";
            result = 1;
        }
        for (std::size_t i = 0; i < std::min(kobayashiActual.size(), kobayashiExpected.size()); ++i)
        {
            if (std::abs(kobayashiActual[i] - kobayashiExpected[i]) > yieldTol)
            {
                std::cerr << "testSimControls: yieldsIsotopes: kobayashi_test yield()[" << i <<
                    "] = " << kobayashiActual[i] << ", expected " << kobayashiExpected[i] << "\n";
                result = 1;
            }
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSimControls: yieldsIsotopes: failed: " << error.what() << "\n";
        result = 1;
    }

    return result;
}

// Verify SimControls::readYields()'s own yields.isotopes handling:
// parses each entry as a case-insensitive element symbol immediately
// followed by a mass number, converts the whole list to a
// elem::IsotopeList, and passes it to yields_->rebuildYieldGrid() so
// that yields()->isotopes() ends up narrowed to that list plus its
// decay-chain context (see Yields::rebuildYieldGrid()'s own comment;
// testSimControlsYieldsIsotopesDecayClosure() below covers that part
// in detail). Reuses the same sukhbold_test/kobayashi_test fixture
// (isotope union h1, fe56, ni56, ni58) as testSimControlsYieldsIsotopes()
// above.
static auto testSimControlsYieldsIsotopesKeyword() -> int
{
    constexpr std::string_view baseDeck = "tests/core/assets/testGalaxy.in";
    int result = 0;

    auto buildDeck = [&](const toml::array& isotopesArr) -> toml::table
    {
        toml::table inputDeck = toml::parse_file(baseDeck);
        inputDeck.insert("yields", toml::table{
            { "channel1", toml::table{
                { "channel", "ccsn" }, { "model", "sukhbold_test" } } },
            { "channel2", toml::table{
                { "channel", "ccsn" }, { "model", "kobayashi_test" } } },
            { "registry", "tests/yields/assets/yields.toml" },
            { "min_isotope_lifetime", 0.0 }, // track Co56/Ni56 explicitly rather than skipping them
            { "isotopes", isotopesArr },
        });
        return inputDeck;
    };

    // Mixed-case symbols, one entry ("C12") that matches no loaded
    // channel's own isotopes at all: isotopes() ends up narrowed to
    // exactly {fe56, co56, ni56, ni58} (Z-then-A order) -- the two
    // requested isotopes, plus co56 and ni56, which decay into the
    // requested fe56 (ni56 -> co56 -> fe56) -- dropping only h1, and
    // silently ignoring the C12 entry rather than throwing for it.
    try
    {
        const toml::table inputDeck = buildDeck(toml::array{ "Fe56", "ni58", "C12" });
        const io::SimControls controls(inputDeck);
        if (controls.yields() == nullptr)
        {
            std::cerr << "testSimControls: yieldsIsotopesKeyword: expected yields() non-null\n";
            return 1;
        }

        const auto& isotopes = controls.yields()->isotopes();
        const std::vector<std::pair<unsigned int, unsigned int>> expected{
            { 26, 56 }, { 27, 56 }, { 28, 56 }, { 28, 58 } }; // fe56, co56, ni56, ni58
        if (isotopes.size() != expected.size())
        {
            std::cerr << "testSimControls: yieldsIsotopesKeyword: expected " <<
                expected.size() << " isotopes (fe56, co56, ni56, ni58), got " << isotopes.size() << "\n";
            result = 1;
        }
        else
        {
            for (std::size_t i = 0; i < expected.size(); ++i)
            {
                if (isotopes[i].get().Z() != expected[i].first ||
                    isotopes[i].get().A() != expected[i].second)
                {
                    std::cerr << "testSimControls: yieldsIsotopesKeyword: isotopes()[" << i <<
                        "] expected (Z=" << expected[i].first << ", A=" << expected[i].second <<
                        "), got (Z=" << isotopes[i].get().Z() << ", A=" <<
                        isotopes[i].get().A() << ")\n";
                    result = 1;
                }
            }
        }

        // sukhbold_test's own yield() (Yields::rebuildYieldGrid() ran
        // again with the narrowed isotope list, so YieldChannel::yield()
        // now returns only 4 entries, in the same [fe56, co56, ni56,
        // ni58] order): co56 (force-expanded) and ni58 are never
        // tabulated by sukhbold_test, so are exactly 0
        constexpr double yieldTol = 1e-10;
        const auto& loaded = controls.yields()->yieldChannels();
        const std::vector<double> sukhboldExpected{ 8.46e-2, 0.0, 7.02e-2, 0.0 }; // fe56, co56, ni56, ni58
        const auto sukhboldActual = loaded.at(0)->yield(18.2, 0.0);
        if (sukhboldActual.size() != sukhboldExpected.size())
        {
            std::cerr << "testSimControls: yieldsIsotopesKeyword: sukhbold_test yield() has "
                "size " << sukhboldActual.size() << ", expected " << sukhboldExpected.size() << "\n";
            result = 1;
        }
        else
        {
            for (std::size_t i = 0; i < sukhboldExpected.size(); ++i)
            {
                if (std::abs(sukhboldActual[i] - sukhboldExpected[i]) > yieldTol)
                {
                    std::cerr << "testSimControls: yieldsIsotopesKeyword: sukhbold_test yield()[" <<
                        i << "] = " << sukhboldActual[i] << ", expected " << sukhboldExpected[i] << "\n";
                    result = 1;
                }
            }
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSimControls: yieldsIsotopesKeyword: filter case failed: "
            << error.what() << "\n";
        result = 1;
    }

    // An empty yields.isotopes array is the same as not giving the key
    // at all -- no restriction, every unioned isotope is kept
    try
    {
        const toml::table inputDeck = buildDeck(toml::array{});
        const io::SimControls controls(inputDeck);
        // 5, not the 4-isotope union itself: Ni56 is unstable, so
        // rebuildYieldGrid()'s own force-expansion also pulls in its
        // decay daughter Co56 -- see testSimControlsYieldsIsotopes()'s
        // own identical case.
        if (controls.yields() == nullptr || controls.yields()->isotopes().size() != 5)
        {
            std::cerr << "testSimControls: yieldsIsotopesKeyword: expected an empty "
                "yields.isotopes to leave all 4 unioned isotopes (plus Co56, "
                "force-expanded) in place\n";
            result = 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSimControls: yieldsIsotopesKeyword: empty-array case failed: "
            << error.what() << "\n";
        result = 1;
    }

    // yields.isotopes not an array: throws
    try
    {
        toml::table inputDeck = toml::parse_file(baseDeck);
        inputDeck.insert("yields", toml::table{
            { "channel1", toml::table{
                { "channel", "ccsn" }, { "model", "sukhbold_test" } } },
            { "registry", "tests/yields/assets/yields.toml" },
            { "min_isotope_lifetime", 0.0 }, // track Co56/Ni56 explicitly rather than skipping them
            { "isotopes", "fe56" },
        });
        const io::SimControls controls(inputDeck);
        std::cerr << "testSimControls: yieldsIsotopesKeyword: expected an exception "
            "when yields.isotopes is not an array\n";
        result = 1;
    }
    catch (const std::runtime_error&) { /* expected */ }

    // yields.isotopes entry with an unrecognized element symbol: throws
    try
    {
        const toml::table inputDeck = buildDeck(toml::array{ "a2" });
        const io::SimControls controls(inputDeck);
        std::cerr << "testSimControls: yieldsIsotopesKeyword: expected an exception "
            "for the unrecognized symbol 'a' in 'a2'\n";
        result = 1;
    }
    catch (const std::runtime_error&) { /* expected */ }

    // yields.isotopes entry with no mass number: throws
    try
    {
        const toml::table inputDeck = buildDeck(toml::array{ "Na" });
        const io::SimControls controls(inputDeck);
        std::cerr << "testSimControls: yieldsIsotopesKeyword: expected an exception "
            "for 'Na' having no mass number\n";
        result = 1;
    }
    catch (const std::runtime_error&) { /* expected */ }

    // yields.isotopes entry naming a real element but a mass number
    // with no corresponding entry in the global isotope table: throws
    try
    {
        const toml::table inputDeck = buildDeck(toml::array{ "H999" });
        const io::SimControls controls(inputDeck);
        std::cerr << "testSimControls: yieldsIsotopesKeyword: expected an exception "
            "for 'H999', which does not exist in the isotope table\n";
        result = 1;
    }
    catch (const std::runtime_error&) { /* expected */ }

    // yields.isotopes entirely non-empty but matching no isotope any
    // loaded channel tabulates (unlike the earlier "Fe56"/"ni58"/"C12"
    // case above, where C12 alone was ignored because Fe56/ni58 still
    // matched something): Yields::rebuildYieldGrid() throws rather than
    // silently leaving yields()->isotopes() empty -- see its own
    // comment for why an entirely-empty result can only mean the whole
    // list was wrong
    try
    {
        const toml::table inputDeck = buildDeck(toml::array{ "C12" });
        const io::SimControls controls(inputDeck);
        std::cerr << "testSimControls: yieldsIsotopesKeyword: expected an exception "
            "when yields.isotopes matches nothing any loaded channel tabulates\n";
        result = 1;
    }
    catch (const std::runtime_error&) { /* expected */ }

    return result;
}

// Verify that a user-supplied yields.isotopes list is expanded along
// decay chains rather than simply intersected with what the channels
// tabulate: every isotope that decays into a requested one (so its
// yield is complete), and every decay product of anything kept (so
// the decay network stays closed), is kept too. Uses only the
// sukhbold_test model, whose own isotopes are h1, fe56, ni56 -- with
// co56 force-expanded in as ni56's decay daughter -- so the decay
// chain of interest is ni56 -> co56 -> fe56. In particular, {H1, Ni56}
// is the request that used to fail, because keeping ni56 while
// dropping co56 broke the decay network.
static auto testSimControlsYieldsIsotopesDecayClosure() -> int
{
    constexpr std::string_view baseDeck = "tests/core/assets/testGalaxy.in";
    int result = 0;

    auto buildControls = [&](const toml::array& isotopesArr) -> std::unique_ptr<io::SimControls>
    {
        toml::table inputDeck = toml::parse_file(baseDeck);
        inputDeck.insert("yields", toml::table{
            { "channel1", toml::table{
                { "channel", "ccsn" }, { "model", "sukhbold_test" } } },
            { "registry", "tests/yields/assets/yields.toml" },
            { "min_isotope_lifetime", 0.0 }, // track Co56/Ni56 explicitly rather than skipping them
            { "isotopes", isotopesArr },
        });
        return std::make_unique<io::SimControls>(inputDeck);
    };

    const auto checkIsotopes = [&](const char* label, const toml::array& requested,
        const std::vector<std::string>& expected) -> int
    {
        try
        {
            const auto controls = buildControls(requested);
            std::vector<std::string> actual;
            for (const auto& iso : controls->yields()->isotopes()) { actual.push_back(iso.get().label()); }
            if (actual != expected)
            {
                std::cerr << "testSimControls: yieldsIsotopesDecayClosure: " << label <<
                    ": isotopes() = [";
                for (const auto& name : actual) { std::cerr << " " << name; }
                std::cerr << " ], expected [";
                for (const auto& name : expected) { std::cerr << " " << name; }
                std::cerr << " ]\n";
                return 1;
            }
        }
        catch (const std::exception& error)
        {
            std::cerr << "testSimControls: yieldsIsotopesDecayClosure: " << label <<
                ": threw: " << error.what() << "\n";
            return 1;
        }
        return 0;
    };

    // Requesting a decay product keeps its parents (upstream)...
    result += checkIsotopes("Fe56", toml::array{ "Fe56" }, { "Fe56", "Co56", "Ni56" });
    result += checkIsotopes("Co56", toml::array{ "Co56" }, { "Fe56", "Co56", "Ni56" });
    // ...and requesting a parent keeps its decay products (downstream)
    result += checkIsotopes("Ni56", toml::array{ "Ni56" }, { "Fe56", "Co56", "Ni56" });
    // The request that used to throw: an unstable isotope, alongside an
    // unrelated stable one, with the intermediate left out
    result += checkIsotopes("H1 + Ni56", toml::array{ "H1", "Ni56" }, { "H1", "Fe56", "Co56", "Ni56" });
    // A stable isotope on no decay chain brings nothing else with it
    result += checkIsotopes("H1", toml::array{ "H1" }, { "H1" });

    // The requested Fe56's own yield picks up its parents' decay: with
    // only Fe56 requested, and dtDecay long enough that all the Ni56 and
    // Co56 has decayed, Fe56 holds the model's own Fe56 plus its Ni56
    // (sukhbold_test at its own exact grid mass 18.2: fe56 = 8.46e-2,
    // ni56 = 7.02e-2), and no Ni56 or Co56 is left.
    try
    {
        const auto controls = buildControls(toml::array{ "Fe56" });
        const double dtDecay = 1000.0 * elem::isotopeTable(27U, 56U).lifetime();
        const auto [view, data] = controls->yields()->yield(18.2, 0.0, dtDecay);
        constexpr double tol = 1e-9;
        const std::vector<double> expected{ 8.46e-2 + 7.02e-2, 0.0, 0.0 }; // fe56, co56, ni56
        for (std::size_t j = 0; j < expected.size(); ++j)
        {
            if (std::abs(view[0, j] - expected[j]) > tol) // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- j < expected.size() by loop bound
            {
                std::cerr << "testSimControls: yieldsIsotopesDecayClosure: fully decayed "
                    "yield()[0, " << j << "] = " << view[0, j] << ", expected " << expected[j] << "\n"; // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- see above
                result = 1;
            }
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSimControls: yieldsIsotopesDecayClosure: decayed-yield case threw: "
            << error.what() << "\n";
        result = 1;
    }
    return result;
}

// Verify that emitted protons and alphas (H1, He4) do not count as
// decay links when yields.isotopes pulls in the parents of a requested
// isotope. The isotope data lists He4 as a daughter of every alpha
// emitter (and H1 of every proton emitter), so without this every
// alpha emitter the yield table tabulates would count as a "parent"
// of a requested He4. Uses the real sukhbold16 table, which tabulates
// H1, He4, and several alpha emitters and their decay products, e.g.
// Sm147 -> Nd143 + He4 and Nd144 -> Ce140 + He4.
static auto testSimControlsYieldsIsotopesEmittedParticles() -> int
{
    constexpr std::string_view baseDeck = "tests/core/assets/testGalaxy.in";
    int result = 0;

    const auto isotopeLabels = [&](const char* requested) -> std::vector<std::string>
    {
        toml::table inputDeck = toml::parse_file(baseDeck);
        inputDeck.insert("yields", toml::table{
            { "channel1", toml::table{
                { "channel", "ccsn" }, { "model", "sukhbold16" } } },
            { "isotopes", toml::array{ requested } },
        });
        const io::SimControls controls(inputDeck);
        std::vector<std::string> labels;
        for (const auto& iso : controls.yields()->isotopes()) { labels.push_back(iso.get().label()); }
        return labels;
    };
    const auto contains = [](const std::vector<std::string>& labels, const char* name) -> bool
    {
        return std::ranges::find(labels, std::string(name)) != labels.end();
    };

    try
    {
        // Requesting He4 or H1 must not pull in any proton/alpha emitter
        for (const char* requested : { "He4", "H1" })
        {
            const auto labels = isotopeLabels(requested);
            if (labels != std::vector<std::string>{ requested })
            {
                std::cerr << "testSimControls: yieldsIsotopesEmittedParticles: requesting " <<
                    requested << " gave " << labels.size() << " isotopes, expected just " <<
                    requested << "\n";
                result = 1;
            }
        }

        // Requesting Nd143 (Sm147 -> Nd143 + He4) keeps its real parent
        // Sm147 and, as a decay product of that alpha emitter, He4 -- but
        // not the unrelated Nd144 -> Ce140 chain, nor anything unrelated
        const auto labels = isotopeLabels("Nd143");
        for (const char* wanted : { "Nd143", "Sm147", "He4" })
        {
            if (!contains(labels, wanted))
            {
                std::cerr << "testSimControls: yieldsIsotopesEmittedParticles: requesting "
                    "Nd143 did not keep " << wanted << "\n";
                result = 1;
            }
        }
        for (const char* unwanted : { "Nd144", "Ce140", "Fe56", "H1" })
        {
            if (contains(labels, unwanted))
            {
                std::cerr << "testSimControls: yieldsIsotopesEmittedParticles: requesting "
                    "Nd143 unexpectedly kept " << unwanted << "\n";
                result = 1;
            }
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSimControls: yieldsIsotopesEmittedParticles: threw: " << error.what() << "\n";
        result = 1;
    }
    return result;
}

// Verify Yields::yield()/yieldSum() correctly aggregate every loaded
// channel's own YieldChannel::yield() into a (nchannels,
// isotopes().size()) array (yield()) and its column sums (yieldSum()).
// Uses the same two-model small-fixture setup as
// testSimControlsYieldsIsotopes(), but with sukhbold_test's own m_min
// overridden to 15.0 (extrapolated from its native minimum of 18.2,
// by a plain ratio -- no interpolation ambiguity) so that mass = 15.0
// is a single query both channels can answer at once: their native
// mass ranges (sukhbold_test [18.2, 100.0], kobayashi_test
// [13.0, 15.0, 18.0]) don't otherwise overlap at all, and 15.0 is
// kobayashi_test's own exact native grid point too, so both rows below
// are exact, hand-computable values rather than interpolated ones.
static auto testSimControlsYieldsYieldAndSum() -> int
{
    constexpr std::string_view baseDeck = "tests/core/assets/testGalaxy.in";
    constexpr double tol = 1e-9;
    int result = 0;

    try
    {
        toml::table inputDeck = toml::parse_file(baseDeck);
        inputDeck.insert("yields", toml::table{
            { "channel1", toml::table{
                { "channel", "ccsn" }, { "model", "sukhbold_test" }, { "m_min", 15.0 } } },
            { "channel2", toml::table{
                { "channel", "ccsn" }, { "model", "kobayashi_test" } } },
            { "registry", "tests/yields/assets/yields.toml" },
            { "min_isotope_lifetime", 0.0 }, // track Co56/Ni56 explicitly rather than skipping them
        });
        const io::SimControls controls(inputDeck);

        if (controls.yields() == nullptr)
        {
            std::cerr << "testSimControls: yieldsYieldAndSum: expected yields() non-null\n";
            return 1;
        }

        // Isotope order is [h1, fe56, co56, ni56, ni58] (see
        // testSimControlsYieldsIsotopes()'s own identical check --
        // co56 is force-expanded in as Ni56's own decay daughter, not
        // directly tabulated by either model). Row 0 (sukhbold_test):
        // its native mass-18.2 values (5.93, 8.46e-2, 7.02e-2 for
        // h1/fe56/ni56), each scaled by 15.0/18.2 (extrapolating down
        // to the overridden m_min), with 0 for co56/ni58 (neither
        // tabulated by sukhbold_test). Row 1 (kobayashi_test): its own
        // real, native mass-15.0 values (6.79, 8.52e-2, 1.15e-3 for
        // h1/fe56/ni58), with 0 for co56/ni56 (neither tabulated by
        // kobayashi_test).
        const auto [view, data] = controls.yields()->yield(15.0, 0.0);
        const std::vector<double> row0{
            5.93 * 15.0 / 18.2, 8.46e-2 * 15.0 / 18.2, 0.0, 7.02e-2 * 15.0 / 18.2, 0.0 };
        const std::vector<double> row1{ 6.79, 8.52e-2, 0.0, 0.0, 1.15e-3 };

        if (view.extent(0) != 2 || view.extent(1) != row0.size())
        {
            std::cerr << "testSimControls: yieldsYieldAndSum: expected yield() shape (2, " <<
                row0.size() << "), got (" << view.extent(0) << ", " << view.extent(1) << ")\n";
            return 1;
        }
        for (std::size_t j = 0; j < row0.size(); ++j)
        {
            if (std::abs(view[0, j] - row0[j]) > tol)
            {
                std::cerr << "testSimControls: yieldsYieldAndSum: yield()[0, " << j << "] = " <<
                    view[0, j] << ", expected " << row0[j] << "\n";
                result = 1;
            }
            if (std::abs(view[1, j] - row1[j]) > tol)
            {
                std::cerr << "testSimControls: yieldsYieldAndSum: yield()[1, " << j << "] = " <<
                    view[1, j] << ", expected " << row1[j] << "\n";
                result = 1;
            }
        }

        const auto sum = controls.yields()->yieldSum(15.0, 0.0);
        if (sum.size() != row0.size())
        {
            std::cerr << "testSimControls: yieldsYieldAndSum: yieldSum() has size " <<
                sum.size() << ", expected " << row0.size() << "\n";
            result = 1;
        }
        else
        {
            for (std::size_t j = 0; j < row0.size(); ++j)
            {
                const double expected = row0[j] + row1[j];
                if (std::abs(sum[j] - expected) > tol)
                {
                    std::cerr << "testSimControls: yieldsYieldAndSum: yieldSum()[" << j << "] = " <<
                        sum[j] << ", expected " << expected << "\n";
                    result = 1;
                }
            }
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSimControls: yieldsYieldAndSum: failed: " << error.what() << "\n";
        result = 1;
    }

    return result;
}

// Verify Yields::yield() leaves a channel's own row all zero, rather
// than throwing, when the query mass lies outside that particular
// channel's own hasYield() range -- the two-model fixture's native
// mass ranges (sukhbold_test [18.2, 100.0], kobayashi_test
// [13.0, 15.0, 18.0], no m_min/m_max overrides this time) don't
// overlap at their shared edge, so mass = 18.2 is sukhbold_test's own
// exact native grid point (hasYield(18.2) true, no interpolation) but
// lies just outside kobayashi_test's own range (hasYield(18.2) false,
// its own max mass being 18.0).
static auto testSimControlsYieldsPartialRange() -> int
{
    constexpr std::string_view baseDeck = "tests/core/assets/testGalaxy.in";
    constexpr double tol = 1e-9;
    int result = 0;

    try
    {
        toml::table inputDeck = toml::parse_file(baseDeck);
        inputDeck.insert("yields", toml::table{
            { "channel1", toml::table{ { "channel", "ccsn" }, { "model", "sukhbold_test" } } },
            { "channel2", toml::table{ { "channel", "ccsn" }, { "model", "kobayashi_test" } } },
            { "registry", "tests/yields/assets/yields.toml" },
            { "min_isotope_lifetime", 0.0 }, // track Co56/Ni56 explicitly rather than skipping them
        });
        const io::SimControls controls(inputDeck);

        // Isotope order is [h1, fe56, co56, ni56, ni58] (see
        // testSimControlsYieldsIsotopes()'s own identical check). Row 0
        // (sukhbold_test) is its own real, native mass-18.2 values
        // (0 for co56, force-expanded but not directly tabulated); row
        // 1 (kobayashi_test) is all zero, since 18.2 is outside its own
        // [13.0, 18.0] range.
        const auto [view, data] = controls.yields()->yield(18.2, 0.0);
        const std::vector<double> row0{ 5.93, 8.46e-2, 0.0, 7.02e-2, 0.0 };
        const std::vector<double> row1{ 0.0, 0.0, 0.0, 0.0, 0.0 };

        if (view.extent(0) != 2 || view.extent(1) != row0.size())
        {
            std::cerr << "testSimControls: yieldsPartialRange: expected yield() shape (2, " <<
                row0.size() << "), got (" << view.extent(0) << ", " << view.extent(1) << ")\n";
            return 1;
        }
        for (std::size_t j = 0; j < row0.size(); ++j)
        {
            if (std::abs(view[0, j] - row0[j]) > tol)
            {
                std::cerr << "testSimControls: yieldsPartialRange: yield()[0, " << j << "] = " <<
                    view[0, j] << ", expected " << row0[j] << "\n";
                result = 1;
            }
            if (std::abs(view[1, j] - row1[j]) > tol)
            {
                std::cerr << "testSimControls: yieldsPartialRange: yield()[1, " << j << "] = " <<
                    view[1, j] << ", expected " << row1[j] << " (out of range, should stay 0)\n";
                result = 1;
            }
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSimControls: yieldsPartialRange: failed: " << error.what() << "\n";
        result = 1;
    }

    return result;
}

// Verify Yields::yield()/yieldSum()'s optional dtDecay argument actually
// applies radioactive decay through Yields's own internal DecayChain,
// using the same two-channel fixture as testSimControlsYieldsYieldAndSum()
// (isotope order [h1, fe56, co56, ni56, ni58] -- co56 is force-expanded
// in as Ni56's own decay daughter, not directly tabulated by either
// model). Ni56 -> Co56 -> Fe56 is a real, non-branching,
// branching-ratio-1 chain (see testDecayChain.hpp): co56's own transient
// abundance is tracked explicitly and checked here, while Fe56 still
// receives its own correct, delayed inflow via the closed-form two-step
// Bateman solution. Also checks that controls_->noDecay() == true makes
// dtDecay a pure no-op, and that dtDecay == 0 changes nothing even with
// noDecay() == false.
static auto testSimControlsYieldsDecayApplication() -> int
{
    constexpr std::string_view baseDeck = "tests/core/assets/testGalaxy.in";
    constexpr double tol = 1e-9;
    int result = 0;

    try
    {
        toml::table inputDeck = toml::parse_file(baseDeck);
        inputDeck.insert("yields", toml::table{
            { "channel1", toml::table{
                { "channel", "ccsn" }, { "model", "sukhbold_test" }, { "m_min", 15.0 } } },
            { "channel2", toml::table{ { "channel", "ccsn" }, { "model", "kobayashi_test" } } },
            { "registry", "tests/yields/assets/yields.toml" },
            { "min_isotope_lifetime", 0.0 }, // track Co56/Ni56 explicitly rather than skipping them
        });
        const io::SimControls controls(inputDeck);

        const auto& ni56 = elem::isotopeTable(28U, 56U);
        const auto& co56 = elem::isotopeTable(27U, 56U);
        const double l1 = 1.0 / ni56.lifetime();
        const double l2 = 1.0 / co56.lifetime();
        const double dt = 2.0 * ni56.lifetime();
        const double n1 = std::exp(-l1 * dt);
        const double n2 = l1 / (l2 - l1) * (std::exp(-l1 * dt) - std::exp(-l2 * dt));
        const double n3 = 1.0 - n1 - n2;

        // Undecayed (dtDecay == 0) values, isotope order
        // [h1, fe56, co56, ni56, ni58]
        const std::vector<double> row0{
            5.93 * 15.0 / 18.2, 8.46e-2 * 15.0 / 18.2, 0.0, 7.02e-2 * 15.0 / 18.2, 0.0 };
        const std::vector<double> row1{ 6.79, 8.52e-2, 0.0, 0.0, 1.15e-3 };
        const double m0Ni56 = row0[3]; // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- fixed-size literal above

        // Decayed values: Ni56 shrinks to m0Ni56 * n1, Co56 gains
        // m0Ni56 * n2 (its own transient share, now tracked), Fe56
        // gains m0Ni56 * n3 (row1's own Ni56 is 0, so it contributes
        // nothing to either).
        const std::vector<double> row0Decayed{
            row0[0], row0[1] + (m0Ni56 * n3), m0Ni56 * n2, m0Ni56 * n1, row0[4] // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- fixed-size literal above
        };
        const std::vector<double>& row1Decayed = row1; // unchanged: row1's own Ni56 is 0

        const auto checkYield = [&](const char* label, double dtDecay,
            const std::vector<double>& expected0, const std::vector<double>& expected1) -> int
        {
            int localResult = 0;
            const auto [view, data] = controls.yields()->yield(15.0, 0.0, dtDecay);
            for (std::size_t j = 0; j < expected0.size(); ++j)
            {
                if (std::abs(view[0, j] - expected0[j]) > tol) // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- j < expected0.size() by loop bound
                {
                    std::cerr << "testSimControls: yieldsDecayApplication: " << label <<
                        ": yield()[0, " << j << "] = " << view[0, j] << ", expected " <<
                        expected0[j] << "\n"; // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- see above
                    localResult = 1;
                }
                if (std::abs(view[1, j] - expected1[j]) > tol) // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- see above
                {
                    std::cerr << "testSimControls: yieldsDecayApplication: " << label <<
                        ": yield()[1, " << j << "] = " << view[1, j] << ", expected " <<
                        expected1[j] << "\n"; // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- see above
                    localResult = 1;
                }
            }

            const auto sum = controls.yields()->yieldSum(15.0, 0.0, dtDecay);
            for (std::size_t j = 0; j < expected0.size(); ++j)
            {
                const double expected = expected0[j] + expected1[j]; // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- see above
                if (std::abs(sum[j] - expected) > tol)
                {
                    std::cerr << "testSimControls: yieldsDecayApplication: " << label <<
                        ": yieldSum()[" << j << "] = " << sum[j] << ", expected " <<
                        expected << "\n";
                    localResult = 1;
                }
            }
            return localResult;
        };

        // dtDecay == 0 must be a no-op
        result += checkYield("dtDecay=0", 0.0, row0, row1);
        // dtDecay > 0, noDecay() == false (default): decay applied
        result += checkYield("dtDecay>0, noDecay=false", dt, row0Decayed, row1Decayed);

        // noDecay() == true: dtDecay must be ignored entirely, even
        // though it is nonzero
        io::SimControls noDecayControls(inputDeck);
        noDecayControls.setNoDecay(true);
        {
            const auto [view, data] = noDecayControls.yields()->yield(15.0, 0.0, dt);
            for (std::size_t j = 0; j < row0.size(); ++j)
            {
                if (std::abs(view[0, j] - row0[j]) > tol || std::abs(view[1, j] - row1[j]) > tol) // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- j < row0.size() by loop bound
                {
                    std::cerr << "testSimControls: yieldsDecayApplication: noDecay=true: "
                        "expected dtDecay to be ignored at column " << j << "\n";
                    result = 1;
                }
            }
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSimControls: yieldsDecayApplication: failed: " << error.what() << "\n";
        result = 1;
    }

    return result;
}

// Verify yields.min_isotope_lifetime / SimControls::
// setMinIsotopeLifetime(): the default (1e4 yr) skips Co56 and Ni56
// (lifetimes well under a yr) from the sukhbold_test/kobayashi_test
// isotope list, leaving H1/Fe56/Ni58 and moving Co56/Ni56 into
// skippedIsotopes(); an explicit 0 keeps them. Long after the skipped
// isotopes' lifetimes (1e3 yr here), yieldSum() must agree between the
// two: the skipped list credits the tabulated Ni56 straight to Fe56,
// the full one decays it there explicitly. A negative key throws.
// setMinIsotopeLifetime() rebuilds yields() at once, keeping a
// yields.isotopes restriction; a negative value, or one that would
// skip every requested isotope, throws and leaves the old value and
// isotope list in place.
static auto testSimControlsMinIsotopeLifetime() -> int
{
    constexpr std::string_view baseDeck = "tests/core/assets/testGalaxy.in";
    const auto makeDeck = [&](const std::optional<double> minLifetime,
        const std::optional<std::string>& isotope) -> toml::table
    {
        toml::table inputDeck = toml::parse_file(baseDeck);
        toml::table yieldsTbl{
            { "channel1", toml::table{ { "channel", "ccsn" }, { "model", "sukhbold_test" } } },
            { "channel2", toml::table{ { "channel", "ccsn" }, { "model", "kobayashi_test" } } },
            { "registry", "tests/yields/assets/yields.toml" },
        };
        if (minLifetime.has_value()) { yieldsTbl.insert("min_isotope_lifetime", *minLifetime); }
        if (isotope.has_value()) { yieldsTbl.insert("isotopes", toml::array{ *isotope }); }
        inputDeck.insert("yields", std::move(yieldsTbl));
        return inputDeck;
    };
    const auto labels = [](const elem::IsotopeList& isotopes) -> std::vector<std::string>
    {
        std::vector<std::string> result;
        for (const auto& iso : isotopes) { result.push_back(iso.get().label()); }
        return result;
    };
    const auto join = [](const std::vector<std::string>& names) -> std::string
    {
        std::string result;
        for (const auto& name : names) { result += " " + name; }
        return result;
    };

    try
    {
        // Default: Co56 and Ni56 skipped
        io::SimControls skipping(makeDeck(std::nullopt, std::nullopt));
        io::SimControls full(makeDeck(0.0, std::nullopt));
        if (skipping.minIsotopeLifetime() != yields::defaultMinIsotopeLifetime ||
            yields::defaultMinIsotopeLifetime != 1e4 || full.minIsotopeLifetime() != 0.0)
        {
            std::cerr << "testSimControls: minIsotopeLifetime: expected 1e4 by default and 0 when "
                "set to 0, got " << skipping.minIsotopeLifetime() << " and " <<
                full.minIsotopeLifetime() << "\n";
            return 1;
        }
        const std::vector<std::string> expectedKept{ "H1", "Fe56", "Ni58" };
        const std::vector<std::string> expectedSkipped{ "Co56", "Ni56" };
        const std::vector<std::string> expectedFull{ "H1", "Fe56", "Co56", "Ni56", "Ni58" };
        if (labels(skipping.yields()->isotopes()) != expectedKept ||
            labels(skipping.yields()->skippedIsotopes()) != expectedSkipped ||
            labels(full.yields()->isotopes()) != expectedFull ||
            !full.yields()->skippedIsotopes().empty())
        {
            std::cerr << "testSimControls: minIsotopeLifetime: expected isotopes()/skippedIsotopes() ="
                << join(expectedKept) << " /" << join(expectedSkipped) << " by default and" <<
                join(expectedFull) << " / (none) with 0, got" << join(labels(skipping.yields()->isotopes())) <<
                " /" << join(labels(skipping.yields()->skippedIsotopes())) << " and" <<
                join(labels(full.yields()->isotopes())) << " /" <<
                join(labels(full.yields()->skippedIsotopes())) << "\n";
            return 1;
        }

        // Long after Co56/Ni56 have decayed, both agree on every kept
        // isotope (full's own order is H1, Fe56, Co56, Ni56, Ni58)
        constexpr double dt = 1e3;
        const auto skippedSum = skipping.yields()->yieldSum(15.0, 0.0, dt);
        const auto fullSum = full.yields()->yieldSum(15.0, 0.0, dt);
        const std::array<std::size_t, 3> fullIndex{ 0, 1, 4 };
        for (std::size_t k = 0; k < fullIndex.size(); ++k)
        {
            if (!utils::approxEqual(skippedSum.at(k), fullSum.at(fullIndex.at(k)), 1e-9) ||
                fullSum.at(fullIndex.at(k)) <= 0.0)
            {
                std::cerr << "testSimControls: minIsotopeLifetime: yieldSum() of " <<
                    expectedKept.at(k) << " after " << dt << " yr = " << skippedSum.at(k) <<
                    " with skipping, " << fullSum.at(fullIndex.at(k)) << " without; expected "
                    "them equal and positive\n";
                return 1;
            }
        }

        // setMinIsotopeLifetime(): rebuilds at once; negative rejected
        skipping.setMinIsotopeLifetime(0.0);
        if (labels(skipping.yields()->isotopes()) != expectedFull)
        {
            std::cerr << "testSimControls: minIsotopeLifetime: expected setMinIsotopeLifetime(0) "
                "to rebuild isotopes() to" << join(expectedFull) << ", got" <<
                join(labels(skipping.yields()->isotopes())) << "\n";
            return 1;
        }
        try
        {
            skipping.setMinIsotopeLifetime(-1.0);
            std::cerr << "testSimControls: minIsotopeLifetime: expected a negative value to throw\n";
            return 1;
        }
        catch (const std::invalid_argument&) { /* expected */ }
        if (skipping.minIsotopeLifetime() != 0.0)
        {
            std::cerr << "testSimControls: minIsotopeLifetime: expected a rejected value to leave "
                "minIsotopeLifetime() at 0\n";
            return 1;
        }

        // A yields.isotopes restriction is kept across rebuilds
        io::SimControls restricted(makeDeck(0.0, "fe56"));
        const std::vector<std::string> fe56Full{ "Fe56", "Co56", "Ni56" };
        const std::vector<std::string> fe56Skipped{ "Fe56" };
        if (labels(restricted.yields()->isotopes()) != fe56Full)
        {
            std::cerr << "testSimControls: minIsotopeLifetime: test bug: expected a Fe56 "
                "restriction to keep" << join(fe56Full) << "\n";
            return 1;
        }
        restricted.setMinIsotopeLifetime(1e4);
        if (labels(restricted.yields()->isotopes()) != fe56Skipped)
        {
            std::cerr << "testSimControls: minIsotopeLifetime: expected setMinIsotopeLifetime(1e4) "
                "to keep the Fe56 restriction, giving" << join(fe56Skipped) << ", got" <<
                join(labels(restricted.yields()->isotopes())) << "\n";
            return 1;
        }

        // Skipping every requested isotope throws, restoring the old
        // value and isotope list
        io::SimControls ni56Only(makeDeck(0.0, "ni56"));
        const auto before = labels(ni56Only.yields()->isotopes());
        try
        {
            ni56Only.setMinIsotopeLifetime(1e4);
            std::cerr << "testSimControls: minIsotopeLifetime: expected skipping the only "
                "requested isotope, Ni56, to throw\n";
            return 1;
        }
        catch (const std::runtime_error&) { /* expected */ }
        if (ni56Only.minIsotopeLifetime() != 0.0 || labels(ni56Only.yields()->isotopes()) != before)
        {
            std::cerr << "testSimControls: minIsotopeLifetime: expected a failed rebuild to "
                "restore minIsotopeLifetime() 0 and isotopes()" << join(before) << "\n";
            return 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSimControls: minIsotopeLifetime: unexpected exception: " << error.what() << "\n";
        return 1;
    }

    // A negative yields.min_isotope_lifetime throws
    try
    {
        const io::SimControls controls(makeDeck(-1.0, std::nullopt));
        std::cerr << "testSimControls: minIsotopeLifetime: expected a negative "
            "yields.min_isotope_lifetime to throw\n";
        return 1;
    }
    catch (const std::runtime_error&) { /* expected */ }
    return 0;
}

// Verify Yields::Yields() prints a "slug: warning" (to std::cout) when
// the same channel is requested more than once, but does not forbid
// it -- both channels must still be built normally. Also verifies the
// negative case: two channels of genuinely different types must not
// warn. Captures std::cout via a temporary rdbuf redirect (there's no
// existing precedent for this in the test suite, since nothing else
// prints a warning worth checking the exact text of); always restores
// the original rdbuf before returning, including on the exception path.
static auto testSimControlsYieldsDuplicateChannelWarning() -> int
{
    constexpr std::string_view baseDeck = "tests/core/assets/testGalaxy.in";
    int result = 0;

    // Two channels, both "ccsn": should warn, but still build both
    try
    {
        toml::table inputDeck = toml::parse_file(baseDeck);
        inputDeck.insert("yields", toml::table{
            { "channel1", toml::table{
                { "channel", "ccsn" }, { "model", "sukhbold_test" } } },
            { "channel2", toml::table{
                { "channel", "ccsn" }, { "model", "kobayashi_test" } } },
            { "registry", "tests/yields/assets/yields.toml" },
        });

        std::ostringstream captured;
        std::streambuf* const origBuf = std::cout.rdbuf(captured.rdbuf());
        std::unique_ptr<io::SimControls> controls;
        try
        {
            controls = std::make_unique<io::SimControls>(inputDeck);
        }
        catch (...)
        {
            std::cout.rdbuf(origBuf);
            throw;
        }
        std::cout.rdbuf(origBuf);

        if (captured.str().find("2 yield channels requested for channel 'ccsn'") == std::string::npos)
        {
            std::cerr << "testSimControls: yieldsDuplicateChannelWarning: expected a "
                "duplicate-channel warning for two ccsn channels, got: \"" <<
                captured.str() << "\"\n";
            result = 1;
        }
        if (controls->yields() == nullptr || controls->yields()->yieldChannels().size() != 2)
        {
            std::cerr << "testSimControls: yieldsDuplicateChannelWarning: duplicate "
                "channels should still both be built, not rejected\n";
            result = 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSimControls: yieldsDuplicateChannelWarning: failed to construct "
            "SimControls with two ccsn channels: " << error.what() << "\n";
        result = 1;
    }

    // Two channels, different types ("ccsn"/"massive_star_winds"): should not warn
    try
    {
        toml::table inputDeck = toml::parse_file(baseDeck);
        inputDeck.insert("yields", toml::table{
            { "channel1", toml::table{
                { "channel", "ccsn" }, { "model", "sukhbold_test" } } },
            { "channel2", toml::table{
                { "channel", "massive_star_winds" }, { "model", "sukhbold_test" } } },
            { "registry", "tests/yields/assets/yields.toml" },
        });

        std::ostringstream captured;
        std::streambuf* const origBuf = std::cout.rdbuf(captured.rdbuf());
        try
        {
            const io::SimControls controls(inputDeck);
        }
        catch (...)
        {
            std::cout.rdbuf(origBuf);
            throw;
        }
        std::cout.rdbuf(origBuf);

        // baseDeck itself already triggers an unrelated, pre-existing
        // "slug: warning" (tracks minimum mass vs. IMF minimum mass --
        // see readTracks()'s own caller in initPhysics()), so this
        // checks specifically for the duplicate-channel warning's own
        // text, not just "slug: warning" generically.
        if (captured.str().find("yield channels requested for channel") != std::string::npos)
        {
            std::cerr << "testSimControls: yieldsDuplicateChannelWarning: unexpected "
                "duplicate-channel warning for two channels of different types: \"" <<
                captured.str() << "\"\n";
            result = 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSimControls: yieldsDuplicateChannelWarning: failed to construct "
            "SimControls with distinct channels: " << error.what() << "\n";
        result = 1;
    }

    return result;
}

// Verify galaxy.sfr/galaxy.sfr_dist's exclusive-or requirement and
// galaxy.sfr_dist's own parsing: both given must throw, neither given
// must throw, a plain number for galaxy.sfr_dist must become an
// ordinary delta-function PDF (unlike galaxy.sfr's own special
// buildConstantSFR() handling -- see sfrDist()'s own header comment),
// and a file must be read via the same utils::initPDFFromKey() path
// every other PDF-valued key uses. In every galaxy.sfr_dist case,
// sfr() itself must remain invalid, since exactly one of the two is
// ever meant to be valid at once.
static auto testSimControlsSFRDist() -> int
{
    constexpr std::string_view baseDeck = "tests/core/assets/testGalaxy.in";

    // Both galaxy.sfr and galaxy.sfr_dist given: must throw
    try
    {
        toml::table inputDeck = toml::parse_file(baseDeck);
        inputDeck.at_path("galaxy").as_table()->insert("sfr_dist", 2.0);
        const io::SimControls controls(inputDeck);
        std::cerr << "testSimControls: sfrDist: expected construction to "
            "throw when both galaxy.sfr and galaxy.sfr_dist are given\n";
        return 1;
    }
    catch (const std::runtime_error&) { /* expected */ }

    // Neither galaxy.sfr nor galaxy.sfr_dist given: must throw
    try
    {
        toml::table inputDeck = toml::parse_file(baseDeck);
        inputDeck.at_path("galaxy").as_table()->erase("sfr");
        const io::SimControls controls(inputDeck);
        std::cerr << "testSimControls: sfrDist: expected construction to "
            "throw when neither galaxy.sfr nor galaxy.sfr_dist is given\n";
        return 1;
    }
    catch (const std::runtime_error&) { /* expected */ }

    // galaxy.sfr_dist given as a plain number: unlike galaxy.sfr, this
    // is an ordinary delta function (via utils::initPDFFromKey(), the
    // same convention CMF/CLF/[Fe/H] already use), not a
    // buildConstantSFR()-style non-normalized constant-in-time PDF
    constexpr double sfrDistValue = 2.5;
    try
    {
        toml::table inputDeck = toml::parse_file(baseDeck);
        auto* galaxyTbl = inputDeck.at_path("galaxy").as_table();
        galaxyTbl->erase("sfr");
        galaxyTbl->insert("sfr_dist", sfrDistValue);
        const io::SimControls controls(inputDeck);

        if (controls.sfr().valid())
        {
            std::cerr << "testSimControls: sfrDist: expected sfr() to "
                "remain invalid when galaxy.sfr_dist is given\n";
            return 1;
        }
        if (!controls.sfrDist().valid() ||
            controls.sfrDist().getMin() != sfrDistValue ||
            controls.sfrDist().getMax() != sfrDistValue ||
            !controls.sfrDist().normalized())
        {
            std::cerr << "testSimControls: sfrDist: expected sfrDist() to "
                "be a valid delta function at " << sfrDistValue << "\n";
            return 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSimControls: sfrDist: numeric case failed: "
            << error.what() << "\n";
        return 1;
    }

    // galaxy.sfr_dist given as a PDF descriptor file: reuses
    // testExtinct's own uniform-over-[0,2] fixture, purely as a
    // convenient, already-existing non-delta PDF descriptor
    try
    {
        toml::table inputDeck = toml::parse_file(baseDeck);
        auto* galaxyTbl = inputDeck.at_path("galaxy").as_table();
        galaxyTbl->erase("sfr");
        galaxyTbl->insert("sfr_dist",
            std::string("tests/extinct/assets/testExtinctAVFieldUniform.toml"));
        const io::SimControls controls(inputDeck);

        if (controls.sfr().valid())
        {
            std::cerr << "testSimControls: sfrDist: expected sfr() to "
                "remain invalid when galaxy.sfr_dist is given (file case)\n";
            return 1;
        }
        if (!controls.sfrDist().valid() ||
            !utils::approxEqual(controls.sfrDist().getMin(), 0.0) ||
            !utils::approxEqual(controls.sfrDist().getMax(), 2.0))
        {
            std::cerr << "testSimControls: sfrDist: expected sfrDist() to "
                "span [0, 2] when read from a file\n";
            return 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSimControls: sfrDist: file case failed: "
            << error.what() << "\n";
        return 1;
    }

    // setSFR()/setSFRDist() must each clear the other back to invalid,
    // so only one of sfr_/sfrDist_ is ever valid at once
    constexpr double setSFRDistValue = 3.0;
    constexpr double setSFRValue = 4.0;
    constexpr double sfrTMax = 1e15; // matches buildConstantSFR()'s own tMax
    try
    {
        const toml::table inputDeck = toml::parse_file(baseDeck);
        io::SimControls controls(inputDeck);
        if (!controls.sfr().valid() || controls.sfrDist().valid())
        {
            std::cerr << "testSimControls: sfrDist: test bug: expected "
                "sfr() valid and sfrDist() invalid right after construction\n";
            return 1;
        }

        controls.setSFRDist(std::to_string(setSFRDistValue));
        if (controls.sfr().valid())
        {
            std::cerr << "testSimControls: sfrDist: expected setSFRDist() "
                "to clear sfr() back to invalid\n";
            return 1;
        }
        if (!controls.sfrDist().valid() ||
            !utils::approxEqual(controls.sfrDist().draw(), setSFRDistValue))
        {
            std::cerr << "testSimControls: sfrDist: expected setSFRDist() "
                "to leave sfrDist() drawing " << setSFRDistValue << "\n";
            return 1;
        }

        controls.setSFR(std::to_string(setSFRValue));
        if (controls.sfrDist().valid())
        {
            std::cerr << "testSimControls: sfrDist: expected setSFR() to "
                "clear sfrDist() back to invalid\n";
            return 1;
        }
        if (!controls.sfr().valid() ||
            controls.sfr().getMin() != 0.0 || controls.sfr().getMax() != sfrTMax)
        {
            std::cerr << "testSimControls: sfrDist: expected setSFR() to "
                "leave sfr() a valid constant-in-time PDF\n";
            return 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSimControls: sfrDist: setter mutual-exclusion "
            "case failed: " << error.what() << "\n";
        return 1;
    }

    return 0;
}

// Verify that setFeH() rejects a new [Fe/H] distribution whose own
// [min, max] range is broader than tracks_'s own [fehMin(), fehMax()]
// -- tracks_ is only ever loaded, once, at construction, over that
// range, so accepting a distribution wider than what was actually
// loaded risks interpolating outside the data tracks_ actually holds.
// Critically, the check compares against tracks_'s own construction-
// time range, not against fehDist_'s current one: narrowing once, then
// setting a second, unrelated narrow value that doesn't nest inside
// the first (but is still within what tracks_ covers), and separately
// widening back out to exactly tracks_'s own bounds, must both
// succeed -- only a distribution that actually exceeds tracks_'s own
// range should ever be rejected.
static auto testSimControlsSetFeHRejectsBroadening() -> int
{
    const std::string fileName = "tests/core/assets/testClusterVarFeH.in";
    const toml::table inputDeck = toml::parse_file(fileName);
    io::SimControls sim(inputDeck);

    if (sim.fehDist().getMin() != -0.5 || sim.fehDist().getMax() != 0.5)
    {
        std::cerr << "testSimControls: setFeH: test bug: expected " << fileName
            << "'s own stars.FeH to be [-0.5, 0.5]\n";
        return 1;
    }

    // Narrowing to a fixed value well within [-0.5, 0.5] should succeed
    try
    {
        sim.setFeH("-0.25");
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSimControls: setFeH: expected narrowing to -0.25 "
            "to succeed, but it threw: " << error.what() << "\n";
        return 1;
    }
    if (sim.fehDist().getMin() != -0.25 || sim.fehDist().getMax() != -0.25)
    {
        std::cerr << "testSimControls: setFeH: expected fehDist() == "
            "[-0.25, -0.25] after narrowing\n";
        return 1;
    }

    // Setting a second, unrelated narrow value (0.4) that does not
    // nest inside the first (-0.25) must still succeed, since both are
    // within tracks_'s own [-0.5, 0.5] -- this is exactly the case the
    // old, buggy comparison against fehDist_'s current (already
    // narrowed) value got wrong
    try
    {
        sim.setFeH("0.4");
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSimControls: setFeH: expected setting the "
            "unrelated narrow value 0.4 (after narrowing to -0.25) to "
            "succeed, but it threw: " << error.what() << "\n";
        return 1;
    }
    if (sim.fehDist().getMin() != 0.4 || sim.fehDist().getMax() != 0.4)
    {
        std::cerr << "testSimControls: setFeH: expected fehDist() == "
            "[0.4, 0.4] after setting 0.4\n";
        return 1;
    }

    // Widening back out to exactly [-0.5, 0.5] -- tracks_'s own range,
    // loaded at construction -- must also succeed
    try
    {
        sim.setFeH("tests/core/assets/testClusterFeHDist.toml");
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSimControls: setFeH: expected widening back to "
            "[-0.5, 0.5] (tracks_'s own range) to succeed, but it threw: "
            << error.what() << "\n";
        return 1;
    }
    if (sim.fehDist().getMin() != -0.5 || sim.fehDist().getMax() != 0.5)
    {
        std::cerr << "testSimControls: setFeH: expected fehDist() == "
            "[-0.5, 0.5] after widening back to tracks_'s own range\n";
        return 1;
    }

    // A distribution that actually exceeds tracks_'s own [-0.5, 0.5]
    // (here, a single point at -0.75, still within the MIST_test
    // fixture's real [-1.0, 0.5] availability, but outside what this
    // SimControls' own tracks_ was actually constructed to cover)
    // should still be rejected
    try
    {
        sim.setFeH("-0.75");
        std::cerr << "testSimControls: setFeH: expected -0.75 (outside "
            "tracks_'s own [-0.5, 0.5]) to throw\n";
        return 1;
    }
    catch (const std::runtime_error&) { /* expected */ }

    // ...and fehDist_ itself should be left exactly as it was before
    // the rejected call, not partially updated
    if (sim.fehDist().getMin() != -0.5 || sim.fehDist().getMax() != 0.5)
    {
        std::cerr << "testSimControls: setFeH: expected fehDist() to remain "
            "[-0.5, 0.5] after the rejected out-of-range attempt\n";
        return 1;
    }

    return 0;
}

// Verify that setTracks() rejects tracks whose own [fehMin(), fehMax()]
// does not cover fehDist() -- the counterpart of setFeH()'s own check
// (see testSimControlsSetFeHRejectsBroadening()) -- including a
// default-constructed Tracks3D, whose NaN range must be rejected rather
// than slipping through every comparison; that a rejected call leaves
// tracks() unchanged; and that tracks which do cover fehDist() are
// accepted, including after fehDist() has been narrowed with setFeH().
static auto testSimControlsSetTracksRejectsUncoveredFeH() -> int
{
    const std::string fileName = "tests/core/assets/testClusterVarFeH.in";
    const toml::table inputDeck = toml::parse_file(fileName);
    io::SimControls sim(inputDeck);
    constexpr std::string_view registry = "tests/tracks/assets/tracks.toml";
    auto mist = [&registry](const double fehMin, const double fehMax)
    {
        return std::make_unique<tracks::Tracks3D>(
            "MIST_test", fehMin, fehMax, 0.0, -0.2, std::string(registry));
    };

    if (sim.fehDist().getMin() != -0.5 || sim.fehDist().getMax() != 0.5)
    {
        std::cerr << "testSimControls: setTracks: test bug: expected " << fileName
            << "'s own stars.FeH to be [-0.5, 0.5]\n";
        return 1;
    }

    // Tracks loaded over [-1, 0] do not cover [-0.5, 0.5]; nor does a
    // default-constructed Tracks3D, whose range is NaN. Either must be
    // rejected, leaving tracks() as it was
    const auto* before = sim.tracks().get();
    for (auto* label : { "[-1, 0]", "default-constructed" })
    {
        auto candidate = std::string_view(label) == "[-1, 0]" ?
            mist(-1.0, 0.0) : std::make_unique<tracks::Tracks3D>();
        try
        {
            sim.setTracks(std::move(candidate));
            std::cerr << "testSimControls: setTracks: expected " << label
                << " tracks, not covering fehDist() [-0.5, 0.5], to throw\n";
            return 1;
        }
        catch (const std::invalid_argument&) { /* expected */ }
        if (sim.tracks().get() != before)
        {
            std::cerr << "testSimControls: setTracks: expected tracks() to be "
                "unchanged after rejecting " << label << " tracks\n";
            return 1;
        }
    }

    try
    {
        // Tracks covering [-0.5, 0.5] are accepted
        sim.setTracks(mist(-1.0, 0.5));
        if (sim.tracks()->fehMin() != -1.0 || sim.tracks()->fehMax() != 0.5)
        {
            std::cerr << "testSimControls: setTracks: expected the new [-1, 0.5] "
                "tracks to be installed\n";
            return 1;
        }

        // After narrowing fehDist() to -0.25, the [-1, 0] tracks
        // rejected above now cover it, and are accepted
        sim.setFeH("-0.25");
        sim.setTracks(mist(-1.0, 0.0));
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSimControls: setTracks: expected tracks covering "
            "fehDist() to be accepted, but it threw: " << error.what() << "\n";
        return 1;
    }
    return 0;
}

// Verify that an IMF extending above the stellar tracks' maximum mass
// is rejected wherever the IMF or tracks can change: by the
// constructor, by setIMF() (against the current tracks), and by
// setTracks() (against the current IMF); that a rejected setIMF() or
// setTracks() leaves this SimControls unchanged; and that an IMF
// reaching exactly the tracks' maximum mass is accepted. Uses
// MIST_test (maximum 300 Msun) and MIST_test_lowmass (the same tracks
// truncated at 100 Msun; see
// data/tools/tracks/make_lowmass_track_fixture.py).
static auto testSimControlsIMFWithinTracks() -> int
{
    const std::string fileName = "tests/core/assets/testClusterVarFeH.in";
    constexpr std::string_view registry = "tests/tracks/assets/tracks.toml";

    // Constructor: a delta-function IMF at 400 Msun is above
    // MIST_test's maximum and must throw; one at exactly 300 must not
    for (const double mass : { 400.0, 300.0 })
    {
        toml::table inputDeck = toml::parse_file(fileName);
        inputDeck.at_path("stars").as_table()->insert_or_assign("IMF", mass);
        try
        {
            const io::SimControls sim(inputDeck);
            if (mass > sim.tracks()->mMax())
            {
                std::cerr << "testSimControls: IMFWithinTracks: expected a " << mass
                    << " Msun IMF, above the tracks' maximum mass, to throw\n";
                return 1;
            }
        }
        catch (const std::invalid_argument& error)
        {
            if (mass <= 300.0)
            {
                std::cerr << "testSimControls: IMFWithinTracks: expected a " << mass
                    << " Msun IMF to be accepted, but it threw: " << error.what() << "\n";
                return 1;
            }
        }
    }

    try
    {
        io::SimControls sim(toml::parse_file(fileName));
        const double tracksMax = sim.tracks()->mMax();
        if (tracksMax != 300.0)
        {
            std::cerr << "testSimControls: IMFWithinTracks: test bug: expected MIST_test's "
                "maximum mass to be 300, got " << tracksMax << "\n";
            return 1;
        }

        // setIMF(): 400 Msun is rejected, leaving the IMF unchanged; 300
        // is accepted
        const double imfMaxBefore = sim.imf().getMax();
        try
        {
            sim.setIMF("400.0");
            std::cerr << "testSimControls: IMFWithinTracks: expected setIMF(\"400.0\") to throw\n";
            return 1;
        }
        catch (const std::invalid_argument&) { /* expected */ }
        if (sim.imf().getMax() != imfMaxBefore)
        {
            std::cerr << "testSimControls: IMFWithinTracks: expected a rejected setIMF() to "
                "leave the IMF unchanged\n";
            return 1;
        }
        sim.setIMF("300.0");

        // setTracks(): with the IMF at 300 Msun, tracks ending at 100
        // Msun are rejected, leaving tracks() unchanged
        auto lowmass = [&registry]()
        {
            return std::make_unique<tracks::Tracks3D>(
                "MIST_test_lowmass", -0.5, 0.5, 0.0, -0.2, std::string(registry));
        };
        const auto* before = sim.tracks().get();
        try
        {
            sim.setTracks(lowmass());
            std::cerr << "testSimControls: IMFWithinTracks: expected setTracks() with a "
                "maximum mass of 100 below the IMF's 300 to throw\n";
            return 1;
        }
        catch (const std::invalid_argument&) { /* expected */ }
        if (sim.tracks().get() != before)
        {
            std::cerr << "testSimControls: IMFWithinTracks: expected a rejected setTracks() "
                "to leave tracks() unchanged\n";
            return 1;
        }

        // Lowering the IMF first lets the same tracks through; the IMF
        // then cannot be raised back above their new maximum
        sim.setIMF("50.0");
        sim.setTracks(lowmass());
        if (sim.tracks()->mMax() != 100.0)
        {
            std::cerr << "testSimControls: IMFWithinTracks: expected MIST_test_lowmass's "
                "maximum mass to be 100, got " << sim.tracks()->mMax() << "\n";
            return 1;
        }
        try
        {
            sim.setIMF("300.0");
            std::cerr << "testSimControls: IMFWithinTracks: expected setIMF(\"300.0\") to "
                "throw against tracks ending at 100 Msun\n";
            return 1;
        }
        catch (const std::invalid_argument&) { /* expected */ }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSimControls: IMFWithinTracks: unexpected exception: "
            << error.what() << "\n";
        return 1;
    }
    return 0;
}

// Verify that setFeH() resets tracks2D() (constFeHTracks_) back to
// nullptr when the new [Fe/H] distribution is no longer degenerate --
// otherwise it would keep pointing at a stale slice from whichever
// single value used to apply, silently contradicting tracks2D()'s own
// "nullptr if constFeH() is false" contract (a real bug CodeRabbit
// caught on the PR that switched constFeHTracks_ from a plain Tracks2D
// value to a shared_ptr<Tracks2D>).
static auto testSimControlsSetFeHResetsTracks2DWhenNoLongerFixed() -> int
{
    const std::string fileName = "tests/core/assets/testClusterVarFeH.in";
    const toml::table inputDeck = toml::parse_file(fileName);
    io::SimControls sim(inputDeck);

    if (sim.constFeH() || sim.tracks2D() != nullptr)
    {
        std::cerr << "testSimControls: setFeH: test bug: expected " << fileName
            << "'s own variable stars.FeH to leave constFeH() false and "
            "tracks2D() null at construction\n";
        return 1;
    }

    sim.setFeH("-0.25");
    if (!sim.constFeH() || sim.tracks2D() == nullptr)
    {
        std::cerr << "testSimControls: setFeH: expected narrowing to a fixed "
            "value to make constFeH() true and build tracks2D()\n";
        return 1;
    }

    sim.setFeH("tests/core/assets/testClusterFeHDist.toml");
    if (sim.constFeH())
    {
        std::cerr << "testSimControls: setFeH: expected widening back to the "
            "original variable distribution to make constFeH() false again\n";
        return 1;
    }
    if (sim.tracks2D() != nullptr)
    {
        std::cerr << "testSimControls: setFeH: expected tracks2D() to be reset "
            "to nullptr once constFeH() is false again, not left pointing at "
            "the stale fixed-[Fe/H] slice\n";
        return 1;
    }

    return 0;
}

// Verify that setSpecsyn() rejects a spectral synthesizer constructed
// for a narrower [Fe/H] range than the installed one's own
// requestedFehMin()/requestedFehMax(), leaving specsyn() unchanged;
// accepts one covering at least the same range, including a
// SpecsynBlackbody (no [Fe/H] axis, so an unbounded range); and, with
// no synthesizer installed, requires the replacement to cover
// fehDist() instead.
static auto testSimControlsSetSpecsynRejectsNarrowerFeh() -> int
{
    const std::string fileName = "tests/core/assets/testCluster.in";
    const std::string registry = "tests/specsyn/assets/spectra.toml";
    toml::table inputDeck = toml::parse_file(fileName);
    inputDeck.at_path("stars").as_table()->insert_or_assign(
        "FeH", "tests/core/assets/testClusterFeHDist.toml");
    auto* spectraTable = inputDeck.at_path("spectra").as_table();
    spectraTable->insert_or_assign("registry", registry);
    spectraTable->insert_or_assign("model", std::string("BOSZ_test"));
    try
    {
        io::SimControls sim(inputDeck);
        const double curMin = sim.specsyn()->requestedFehMin();
        const double curMax = sim.specsyn()->requestedFehMax();
        if (!(curMin <= sim.fehDist().getMin() && sim.fehDist().getMax() <= curMax))
        {
            std::cerr << "testSimControls: setSpecsyn: test bug: expected the deck's own "
                "synthesizer to cover fehDist(), got [" << curMin << ", " << curMax << "]\n";
            return 1;
        }
        auto bosz = [&](const double fehMin, const double fehMax)
        {
            return std::make_unique<specsyn::SpecsynLibNoWind<specsyn::OOBPolicy::raise>>(
                "BOSZ_test", fehMin, fehMax, 0.0, 0.0, 0.0, specsyn::defaultR, registry,
                0.0, 0.0, 0, sim);
        };

        // Narrower than the installed synthesizer: rejected
        const auto* before = sim.specsyn().get();
        try
        {
            sim.setSpecsyn(bosz(-0.25, 0.25));
            std::cerr << "testSimControls: setSpecsyn: expected a synthesizer built for "
                "[-0.25, 0.25] to be rejected, since the installed one covers ["
                << curMin << ", " << curMax << "]\n";
            return 1;
        }
        catch (const std::invalid_argument&) { /* expected */ }
        if (sim.specsyn().get() != before)
        {
            std::cerr << "testSimControls: setSpecsyn: expected specsyn() unchanged after "
                "a rejected replacement\n";
            return 1;
        }

        // The same range, then an unbounded one: accepted
        sim.setSpecsyn(bosz(curMin, curMax));
        sim.setSpecsyn(std::make_unique<specsyn::SpecsynBlackbody>(3000.0, 9000.0, 50, sim));

        // With none installed, a replacement must cover fehDist()
        sim.setSpecsyn(nullptr);
        try
        {
            sim.setSpecsyn(bosz(-0.25, 0.25));
            std::cerr << "testSimControls: setSpecsyn: expected a synthesizer built for "
                "[-0.25, 0.25] to be rejected with none installed, since it does not "
                "cover fehDist() [-0.5, 0.5]\n";
            return 1;
        }
        catch (const std::invalid_argument&) { /* expected */ }
        sim.setSpecsyn(bosz(-0.5, 0.5));
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSimControls: setSpecsyn: unexpected exception: " << error.what() << "\n";
        return 1;
    }
    return 0;
}

// Verify that setYields() rejects a Yields built while fehDist() was
// narrowed, once fehDist() has been broadened again -- the replacement
// covers a narrower [Fe/H] range than the installed Yields -- and,
// with no Yields installed, one not covering fehDist(); that a rejected
// call leaves yields() unchanged; and that a Yields covering the full
// range is accepted. Also checks that Yields::requestedFehMin()/Max()
// report the range its channels were built for.
static auto testSimControlsSetYieldsRejectsNarrowerFeh() -> int
{
    const std::string fileName = "tests/core/assets/testCluster.in";
    const std::string registry = "tests/yields/assets/yields.toml";
    const std::string fehFull = "tests/core/assets/testClusterSpecsynFullFeHDist.toml"; // flat [-1, 0]
    toml::table inputDeck = toml::parse_file(fileName);
    inputDeck.at_path("stars").as_table()->insert_or_assign("FeH", fehFull);
    inputDeck.insert_or_assign("yields", toml::table{
        { "channel1", toml::table{ { "channel", "ccsn" }, { "model", "sukhbold_test" } } },
        { "registry", registry },
    });
    try
    {
        io::SimControls sim(inputDeck);
        if (sim.yields()->requestedFehMin() != -1.0 || sim.yields()->requestedFehMax() != 0.0)
        {
            std::cerr << "testSimControls: setYields: expected the deck's own Yields to "
                "report requestedFeh range [-1, 0], got [" << sim.yields()->requestedFehMin()
                << ", " << sim.yields()->requestedFehMax() << "]\n";
            return 1;
        }

        // Build a Yields while fehDist() is narrowed to -0.5, then
        // broaden fehDist() back out
        sim.setFeH("-0.5");
        auto narrow = std::make_unique<yields::Yields>(sim, registry);
        sim.setFeH(fehFull);

        const auto* before = sim.yields().get();
        try
        {
            sim.setYields(std::move(narrow));
            std::cerr << "testSimControls: setYields: expected a Yields built while "
                "fehDist() was narrowed to -0.5 to be rejected\n";
            return 1;
        }
        catch (const std::invalid_argument&) { /* expected */ }
        if (sim.yields().get() != before)
        {
            std::cerr << "testSimControls: setYields: expected yields() unchanged after a "
                "rejected replacement\n";
            return 1;
        }

        // With none installed, the same narrow Yields must still be
        // rejected, since it does not cover fehDist() [-1, 0]
        sim.setYields(nullptr);
        sim.setFeH("-0.5");
        narrow = std::make_unique<yields::Yields>(sim, registry);
        sim.setFeH(fehFull);
        try
        {
            sim.setYields(std::move(narrow));
            std::cerr << "testSimControls: setYields: expected a Yields built for [-0.5, "
                "-0.5] to be rejected with none installed\n";
            return 1;
        }
        catch (const std::invalid_argument&) { /* expected */ }

        // A Yields built over the full fehDist() is accepted
        auto installed = std::make_unique<yields::Yields>(sim, registry);
        auto* raw = installed.get();
        sim.setYields(std::move(installed));

        // Mutating the installed Yields with a pre-built channel built
        // for [-0.5, 0], narrower than fehDist() [-1, 0], is rejected
        // (via addChannel() and setChannels() alike), leaving its
        // channels unchanged; a standalone Yields accepts it
        const yields::YieldChannelDescriptor desc{
            yields::Channel::ccsn_, "sukhbold_test", std::nullopt, std::nullopt };
        auto narrowChannel = [&]()
        { return std::make_unique<yields::YieldChannel>(desc, -0.5, 0.0, registry); };
        const auto nBefore = raw->yieldChannels().size();
        try
        {
            raw->addChannel(narrowChannel());
            std::cerr << "testSimControls: setYields: expected adding a [-0.5, 0] channel "
                "to the installed Yields to be rejected\n";
            return 1;
        }
        catch (const std::invalid_argument&) { /* expected */ }
        try
        {
            std::vector<std::unique_ptr<yields::YieldChannel>> channels;
            channels.push_back(narrowChannel());
            raw->setChannels(std::move(channels));
            std::cerr << "testSimControls: setYields: expected replacing the installed "
                "Yields' channels with a [-0.5, 0] channel to be rejected\n";
            return 1;
        }
        catch (const std::invalid_argument&) { /* expected */ }
        if (raw->yieldChannels().size() != nBefore)
        {
            std::cerr << "testSimControls: setYields: expected the installed Yields' "
                "channels unchanged after rejected mutations\n";
            return 1;
        }
        yields::Yields standalone(sim, registry);
        standalone.addChannel(narrowChannel());
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSimControls: setYields: unexpected exception: " << error.what() << "\n";
        return 1;
    }
    return 0;
}

// Verify that setFeH() rejects widening fehDist() beyond the [Fe/H]
// range the installed Yields or spectral synthesizer was constructed
// for, even when the tracks would allow it: after setTracks() installs
// tracks loaded over [-1, 0.5], widening fehDist() from [-1, 0] to
// [-0.5, 0.5] passes the tracks check, but must still be rejected while
// the installed Yields covers only [-1, 0], and again while a
// synthesizer built for [-1, 0] is installed, leaving fehDist()
// unchanged each time; once both are removed or replaced by ones
// covering it, the widening succeeds.
static auto testSimControlsSetFeHRejectsBeyondSpecsynYields() -> int
{
    const std::string fileName = "tests/core/assets/testCluster.in";
    const std::string specRegistry = "tests/specsyn/assets/spectra.toml";
    const std::string fehWide = "tests/core/assets/testClusterFeHDist.toml"; // flat [-0.5, 0.5]
    toml::table inputDeck = toml::parse_file(fileName);
    inputDeck.at_path("stars").as_table()->insert_or_assign(
        "FeH", "tests/core/assets/testClusterSpecsynFullFeHDist.toml"); // flat [-1, 0]
    auto* spectraTable = inputDeck.at_path("spectra").as_table();
    spectraTable->insert_or_assign("registry", specRegistry);
    spectraTable->insert_or_assign("model", std::string("BOSZ_test"));
    inputDeck.insert_or_assign("yields", toml::table{
        { "channel1", toml::table{ { "channel", "ccsn" }, { "model", "sukhbold_test" } } },
        { "registry", std::string("tests/yields/assets/yields.toml") },
    });
    try
    {
        io::SimControls sim(inputDeck);
        auto bosz = [&](const double fehMin, const double fehMax)
        {
            return std::make_unique<specsyn::SpecsynLibNoWind<specsyn::OOBPolicy::raise>>(
                "BOSZ_test", fehMin, fehMax, 0.0, 0.0, 0.0, specsyn::defaultR, specRegistry,
                0.0, 0.0, 0, sim);
        };
        auto expectRejected = [&](const char* label) -> bool
        {
            try
            {
                sim.setFeH(fehWide);
                std::cerr << "testSimControls: setFeH: expected widening to [-0.5, 0.5] "
                    "to be rejected " << label << "\n";
                return false;
            }
            catch (const std::runtime_error&) { /* expected */ }
            if (sim.fehDist().getMin() != -1.0 || sim.fehDist().getMax() != 0.0)
            {
                std::cerr << "testSimControls: setFeH: expected fehDist() to remain [-1, 0] "
                    "after the rejected widening " << label << "\n";
                return false;
            }
            return true;
        };

        sim.setTracks(std::make_unique<tracks::Tracks3D>(
            "MIST_test", -1.0, 0.5, 0.0, -0.2, "tests/tracks/assets/tracks.toml"));
        if (!expectRejected("while the installed Yields covers only [-1, 0]")) { return 1; }

        sim.setYields(nullptr);
        sim.setSpecsyn(nullptr);
        sim.setSpecsyn(bosz(-1.0, 0.0));
        if (!expectRejected("while the installed synthesizer covers only [-1, 0]")) { return 1; }

        sim.setSpecsyn(bosz(-1.0, 0.5));
        sim.setFeH(fehWide);
        if (sim.fehDist().getMin() != -0.5 || sim.fehDist().getMax() != 0.5)
        {
            std::cerr << "testSimControls: setFeH: expected widening to [-0.5, 0.5] to "
                "succeed once the synthesizer covers it and no Yields is installed\n";
            return 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSimControls: setFeH: unexpected exception: " << error.what() << "\n";
        return 1;
    }
    return 0;
}

// Verify that setSpecsyn()/setExtinct()/setNebular() each reject an
// object constructed against a different SimControls than the one
// it's being installed on -- each of Specsyn/Extinct/Nebular stores a
// live reference to whichever SimControls it was built against, so
// installing one bound elsewhere would leave it silently reading (or
// describing) the wrong object's own settings.
static auto testSimControlsSettersRejectMismatchedControls() -> int
{
    const std::string fileName = "tests/core/assets/testCluster.in";
    const toml::table inputDeck = toml::parse_file(fileName);
    io::SimControls sim1(inputDeck);
    io::SimControls sim2(inputDeck);

    {
        auto badSpecsyn = std::make_unique<specsyn::SpecsynBlackbody>(3000.0, 9000.0, 50, sim1);
        try
        {
            sim2.setSpecsyn(std::move(badSpecsyn));
            std::cerr << "testSimControls: setSpecsyn: expected a Specsyn built "
                "against sim1 to be rejected when installed on sim2\n";
            return 1;
        }
        catch (const std::invalid_argument&) { /* expected */ }
    }

    {
        auto badExtinct = std::make_unique<extinct::Extinct>("Calzetti_starburst", sim1);
        try
        {
            sim2.setExtinct(std::move(badExtinct));
            std::cerr << "testSimControls: setExtinct: expected an Extinct built "
                "against sim1 to be rejected when installed on sim2\n";
            return 1;
        }
        catch (const std::invalid_argument&) { /* expected */ }
    }

    {
        auto badNebular = std::make_unique<nebular::Nebular>(
            "tests/nebular/assets/nebular_test.h5", "MIST_test", sim1);
        try
        {
            sim2.setNebular(std::move(badNebular));
            std::cerr << "testSimControls: setNebular: expected a Nebular built "
                "against sim1 to be rejected when installed on sim2\n";
            return 1;
        }
        catch (const std::invalid_argument&) { /* expected */ }
    }

    return 0;
}

// Paths of a list of input-deck key reports, in order
static auto deckKeyPaths(const std::vector<utils::DeckKeyReport>& reports) -> std::vector<std::string>
{
    std::vector<std::string> paths;
    for (const auto& report : reports) { paths.push_back(report.path_); }
    return paths;
}

// Whether a list of input-deck key reports contains a key with this path
static auto deckKeyListed(const std::vector<utils::DeckKeyReport>& reports, const std::string& path) -> bool
{
    return std::ranges::any_of(reports, [&path](const auto& report) { return report.path_ == path; });
}

// Verify that an input deck's stray keys are reported: a deck with no
// stray keys reports none, a misplaced key (n_trial given under [output]
// rather than at the top level) and a misspelled one (v_vcirt for
// v_vcrit) are each reported with a hint for what was probably meant, and
// both are still accepted (the run just warns) unless strict_input is set
static auto testSimControlsUnusedKeys() -> int
{
    const std::string fileName = "tests/core/assets/testCluster.in";
    int result = 0;

    try
    {
        const io::SimControls clean(toml::parse_file(fileName));
        if (!clean.unusedKeys().empty() || clean.strictInput())
        {
            std::cerr << "testSimControls: unusedKeys: expected a clean deck to report no unused keys "
                "and strict_input to default to false\n";
            result = 1;
        }

        toml::table inputDeck = toml::parse_file(fileName);
        inputDeck.at_path("output").as_table()->insert_or_assign("n_trial", 5);
        inputDeck.at_path("stars").as_table()->insert_or_assign("v_vcirt", 0.1);
        const io::SimControls sim(inputDeck);
        if (deckKeyPaths(sim.unusedKeys()) != std::vector<std::string>{ "output.n_trial", "stars.v_vcirt" })
        {
            std::cerr << "testSimControls: unusedKeys: expected exactly output.n_trial and "
                "stars.v_vcirt to be reported, got " << sim.unusedKeys().size() << " keys\n";
            return 1;
        }
        for (const auto& report : sim.unusedKeys())
        {
            const std::string expected = report.path_ == "output.n_trial" ?
                "did you mean 'n_trial'?" : "did you mean 'stars.v_vcrit'?";
            if (report.hint_ != expected)
            {
                std::cerr << "testSimControls: unusedKeys: hint for " << report.path_ << " was '" <<
                    report.hint_ << "', expected '" << expected << "'\n";
                result = 1;
            }
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSimControls: unusedKeys: threw: " << error.what() << "\n";
        return 1;
    }
    return result;
}

// Verify strict_input: an unused key then stops the run with an error
// that names it, while a clean deck is unaffected by the flag
static auto testSimControlsStrictInput() -> int
{
    const std::string fileName = "tests/core/assets/testCluster.in";

    {
        toml::table inputDeck = toml::parse_file(fileName);
        inputDeck.insert_or_assign("strict_input", true);
        try
        {
            const io::SimControls sim(inputDeck);
            if (!sim.strictInput() || !sim.unusedKeys().empty())
            {
                std::cerr << "testSimControls: strictInput: expected a clean deck with strict_input "
                    "set to construct, with strictInput() true and no unused keys\n";
                return 1;
            }
        }
        catch (const std::exception& error)
        {
            std::cerr << "testSimControls: strictInput: a clean deck failed under strict_input: " <<
                error.what() << "\n";
            return 1;
        }
    }

    toml::table inputDeck = toml::parse_file(fileName);
    inputDeck.insert_or_assign("strict_input", true);
    inputDeck.at_path("output").as_table()->insert_or_assign("n_trial", 5);
    try
    {
        const io::SimControls sim(inputDeck);
        std::cerr << "testSimControls: strictInput: expected an unused key to throw under strict_input\n";
        return 1;
    }
    catch (const std::runtime_error& error)
    {
        if (std::string(error.what()).find("output.n_trial") == std::string::npos)
        {
            std::cerr << "testSimControls: strictInput: expected the error to name output.n_trial, got: " <<
                error.what() << "\n";
            return 1;
        }
    }
    return 0;
}

// Verify that keys which are valid but not read because of the
// simulation's own settings are reported as ignored, with a reason, and
// not as unused: a nebular key when compute_neb is false, spectral
// synthesis keys when there is no spectra.model, galaxy keys in a
// cluster simulation, and an extinction key when there is no extinct.AV
static auto testSimControlsIgnoredKeys() -> int
{
    const std::string fileName = "tests/core/assets/testCluster.in";
    toml::table inputDeck = toml::parse_file(fileName);
    inputDeck.erase("spectra");
    inputDeck.at_path("nebular").as_table()->insert_or_assign("log_U", -2.0);
    inputDeck.at_path("stars").as_table()->insert_or_assign("CFe", 0.1);
    inputDeck.at_path("clusters").as_table()->insert_or_assign("CLF", 1e6);
    inputDeck.insert_or_assign("galaxy", toml::table{ { "sfr", 1.0 } });
    inputDeck.insert_or_assign("extinct", toml::table{ { "model", "Calzetti_starburst" } });
    inputDeck.insert_or_assign("spectra", toml::table{
        { "wl_min", 1000.0 }, { "registry", "tests/specsyn/assets/spectra.toml" } });

    try
    {
        const io::SimControls sim(inputDeck);
        if (!sim.unusedKeys().empty())
        {
            std::cerr << "testSimControls: ignoredKeys: expected these keys to be ignored, not unused; "
                "first unused key was " << sim.unusedKeys().front().path_ << "\n";
            return 1;
        }
        int result = 0;
        for (const char* path : { "nebular.log_U", "stars.CFe", "clusters.CLF", "galaxy.sfr",
            "extinct.model", "spectra.wl_min", "spectra.registry" })
        {
            if (!deckKeyListed(sim.ignoredKeys(), path))
            {
                std::cerr << "testSimControls: ignoredKeys: expected " << path << " to be ignored\n";
                result = 1;
            }
        }
        for (const auto& report : sim.ignoredKeys())
        {
            if (report.reason_.empty())
            {
                std::cerr << "testSimControls: ignoredKeys: " << report.path_ << " has no reason\n";
                result = 1;
            }
        }
        return result;
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSimControls: ignoredKeys: threw: " << error.what() << "\n";
        return 1;
    }
}

// Verify fracStochMass() and nonStochIMFMass() against a direct
// numerical integration of m * imf(m) (trapezoidal, in log mass) for
// the Chabrier IMF with stars.min_stoch_mass = 10: fracStochMass()
// must be the fraction of the IMF's total *mass* above 10 Msun (not
// the fraction by number, imf().integral(10, max), which is ~35 times
// smaller), and nonStochIMFMass() the integral of m * imf(m) below 10
// Msun. Also checks the edge cases min_stoch_mass >= the IMF's maximum
// (fracStochMass() == 0) and <= its minimum (fracStochMass() == 1,
// nonStochIMFMass() == 0), via setMinStochMass().
static auto testSimControlsFracStochMass() -> int
{
    constexpr std::string_view baseDeck = "tests/core/assets/testGalaxy.in";
    constexpr double mStoch = 10.0;
    constexpr double tol = 1e-4;
    try
    {
        toml::table inputDeck = toml::parse_file(baseDeck);
        inputDeck.at_path("stars").as_table()->insert_or_assign("min_stoch_mass", mStoch);
        io::SimControls controls(inputDeck);
        const auto& imf = controls.imf();

        // Integral of m * imf(m) dm over [a, b], by the trapezoidal rule
        // in ln m (m * imf(m) dm = m^2 * imf(m) dln m)
        auto massIntegral = [&imf](const double a, const double b) -> double {
            constexpr int n = 200000;
            const double la = std::log(a);
            const double dl = (std::log(b) - la) / n;
            double sum = 0.0;
            for (int i = 0; i <= n; ++i)
            {
                const double m = std::exp(la + (i * dl));
                const double w = (i == 0 || i == n) ? 0.5 : 1.0;
                sum += w * m * m * imf(m);
            }
            return sum * dl;
        };
        const double massBelow = massIntegral(imf.getMin(), mStoch);
        const double massAbove = massIntegral(mStoch, imf.getMax());
        const double expectedFrac = massAbove / (massBelow + massAbove);

        int result = 0;
        if (std::abs(controls.fracStochMass() - expectedFrac) > tol * expectedFrac)
        {
            std::cerr << "testSimControls: fracStochMass: fracStochMass() is "
                << controls.fracStochMass() << ", expected " << expectedFrac << "\n";
            result = 1;
        }
        if (std::abs(controls.nonStochIMFMass() - massBelow) > tol * massBelow)
        {
            std::cerr << "testSimControls: fracStochMass: nonStochIMFMass() is "
                << controls.nonStochIMFMass() << ", expected " << massBelow << "\n";
            result = 1;
        }

        controls.setMinStochMass(2.0 * imf.getMax());
        if (controls.fracStochMass() != 0.0)
        {
            std::cerr << "testSimControls: fracStochMass: expected 0 with min_stoch_mass "
                "above the IMF's maximum, got " << controls.fracStochMass() << "\n";
            result = 1;
        }
        controls.setMinStochMass(0.5 * imf.getMin());
        if (controls.fracStochMass() != 1.0 || controls.nonStochIMFMass() != 0.0)
        {
            std::cerr << "testSimControls: fracStochMass: expected fracStochMass() 1 and "
                "nonStochIMFMass() 0 with min_stoch_mass below the IMF's minimum, got "
                << controls.fracStochMass() << " and " << controls.nonStochIMFMass() << "\n";
            result = 1;
        }

        // A delta-function IMF at 20 Msun: entirely non-stochastic, with
        // nonStochIMFMass() equal to its own point mass, if
        // min_stoch_mass is above 20 Msun, and entirely stochastic if
        // below it. (Also checks that setIMF() recomputes
        // fracStochMass().)
        constexpr double deltaMass = 20.0;
        controls.setMinStochMass(30.0);
        controls.setIMF("20.0");
        if (controls.fracStochMass() != 0.0 ||
            std::abs(controls.nonStochIMFMass() - deltaMass) > tol * deltaMass)
        {
            std::cerr << "testSimControls: fracStochMass: expected fracStochMass() 0 and "
                "nonStochIMFMass() " << deltaMass << " for a delta-function IMF at "
                << deltaMass << " Msun with min_stoch_mass = 30, got " << controls.fracStochMass()
                << " and " << controls.nonStochIMFMass() << "\n";
            result = 1;
        }
        controls.setMinStochMass(10.0);
        if (controls.fracStochMass() != 1.0 || controls.nonStochIMFMass() != 0.0)
        {
            std::cerr << "testSimControls: fracStochMass: expected fracStochMass() 1 and "
                "nonStochIMFMass() 0 for a delta-function IMF at " << deltaMass
                << " Msun with min_stoch_mass = 10, got " << controls.fracStochMass()
                << " and " << controls.nonStochIMFMass() << "\n";
            result = 1;
        }
        return result;
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSimControls: fracStochMass: threw: " << error.what() << "\n";
        return 1;
    }
}

// Verify that a yields.channelN table after a gap in the numbering (a
// channel3 with no channel2), which SLUG has never read, is reported
// rather than silently dropped, and that yields keys read only once a
// channel exists (registry, isotopes) are merely ignored when there is
// none
static auto testSimControlsYieldsChannelGap() -> int
{
    constexpr std::string_view baseDeck = "tests/core/assets/testGalaxy.in";
    toml::table inputDeck = toml::parse_file(baseDeck);
    inputDeck.insert("yields", toml::table{
        { "channel1", toml::table{ { "channel", "ccsn" }, { "model", "sukhbold_test" } } },
        { "channel3", toml::table{ { "channel", "ccsn" }, { "model", "kobayashi_test" } } },
        { "registry", "tests/yields/assets/yields.toml" },
    });
    try
    {
        const io::SimControls sim(inputDeck);
        if (deckKeyPaths(sim.unusedKeys()) !=
            std::vector<std::string>{ "yields.channel3.channel", "yields.channel3.model" })
        {
            std::cerr << "testSimControls: yieldsChannelGap: expected exactly the channel3 keys to be "
                "reported unused, got " << sim.unusedKeys().size() << " keys\n";
            return 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSimControls: yieldsChannelGap: threw: " << error.what() << "\n";
        return 1;
    }

    // With no channel1 at all, registry and isotopes go unread but are not mistakes...
    toml::table noChannels = toml::parse_file(baseDeck);
    noChannels.insert("yields", toml::table{
        { "registry", "tests/yields/assets/yields.toml" }, { "isotopes", toml::array{ "Fe56" } } });
    try
    {
        const io::SimControls sim(noChannels);
        if (!sim.unusedKeys().empty() || !deckKeyListed(sim.ignoredKeys(), "yields.isotopes") ||
            !deckKeyListed(sim.ignoredKeys(), "yields.registry"))
        {
            std::cerr << "testSimControls: yieldsChannelGap: expected yields.registry and "
                "yields.isotopes to be ignored, and nothing unused, when there is no channel1\n";
            return 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSimControls: yieldsChannelGap: no-channel case threw: " << error.what() << "\n";
        return 1;
    }
    return 0;
}

// Verify Yields::hasYield(): with two ccsn channels covering disjoint
// mass ranges (kobayashi_test, 13-18 Msun; sukhbold_test, 18.2-100
// Msun), a massive_star_winds channel whose mass range is extended to
// 10-150 Msun via m_min/m_max, and no agb channel, hasYield() should
// be true for a given channel iff the mass lies within the range of
// at least one channel of that type
static auto testSimControlsYieldsHasYield() -> int
{
    constexpr std::string_view baseDeck = "tests/core/assets/testGalaxy.in";
    toml::table inputDeck = toml::parse_file(baseDeck);
    inputDeck.insert("yields", toml::table{
        { "channel1", toml::table{ { "channel", "ccsn" }, { "model", "sukhbold_test" } } },
        { "channel2", toml::table{ { "channel", "ccsn" }, { "model", "kobayashi_test" } } },
        { "channel3", toml::table{
            { "channel", "massive_star_winds" }, { "model", "sukhbold_test" },
            { "m_min", 10.0 }, { "m_max", 150.0 } } },
        { "registry", "tests/yields/assets/yields.toml" },
    });

    struct Case
    {
        double mass_;
        yields::Channel channel_;
        bool expected_;
    };
    constexpr std::array<Case, 11> cases = { {
        { 15.0, yields::Channel::ccsn_, true },    // kobayashi_test only
        { 50.0, yields::Channel::ccsn_, true },    // sukhbold_test only
        { 18.0, yields::Channel::ccsn_, true },    // kobayashi_test upper edge
        { 18.1, yields::Channel::ccsn_, false },   // gap between the two ccsn models
        { 10.0, yields::Channel::ccsn_, false },   // below both ccsn models
        { 120.0, yields::Channel::ccsn_, false },  // above both ccsn models
        { 10.0, yields::Channel::massiveStarWinds_, true },  // extended lower edge
        { 150.0, yields::Channel::massiveStarWinds_, true }, // extended upper edge
        { 5.0, yields::Channel::massiveStarWinds_, false },
        { 15.0, yields::Channel::agb_, false },    // no agb channel at all
        { 50.0, yields::Channel::agb_, false },
    } };

    try
    {
        const io::SimControls controls(inputDeck);
        if (controls.yields() == nullptr)
        {
            std::cerr << "testSimControls: yieldsHasYield: expected yields() non-null\n";
            return 1;
        }
        int result = 0;
        for (const auto& c : cases)
        {
            if (controls.yields()->hasYield(c.mass_, c.channel_) != c.expected_)
            {
                std::cerr << "testSimControls: yieldsHasYield: hasYield(" << c.mass_ << ", "
                    << yields::channelStr.at(static_cast<std::size_t>(c.channel_))
                    << ") returned " << !c.expected_ << ", expected " << c.expected_ << "\n";
                result = 1;
            }
        }
        return result;
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSimControls: yieldsHasYield: threw: " << error.what() << "\n";
        return 1;
    }
}

// Check hasSN() against a list of (mass, expected) cases, printing a
// diagnostic for each mismatch; returns the number of mismatches
static auto checkHasSN(const io::SimControls& controls, const std::string_view label,
    const std::vector<std::pair<double, bool>>& cases) -> int
{
    int nFail = 0;
    for (const auto& [mass, expected] : cases)
    {
        if (controls.hasSN(mass) != expected)
        {
            std::cerr << "testSimControls: feedback: " << label << ": hasSN(" << mass
                << ") returned " << !expected << ", expected " << expected << "\n";
            ++nFail;
        }
    }
    return nFail;
}

// Verify readFeedback()'s parsing of feedback.sn_mass_range,
// setSNMassLimits()'s validation, and hasSN()'s choice between
// explicit mass limits and the ccsn yield channels
static auto testSimControlsFeedback() -> int
{
    constexpr std::string_view baseDeck = "tests/core/assets/testGalaxy.in";
    int result = 0;

    // Build a deck from baseDeck, optionally adding the test fixture's
    // two ccsn yield channels (kobayashi_test, 13-18 Msun;
    // sukhbold_test, 18.2-100 Msun) and/or a [feedback] table
    auto makeDeck = [baseDeck](const bool withYields, std::optional<toml::table> feedback)
        -> toml::table {
        toml::table deck = toml::parse_file(baseDeck);
        if (withYields)
        {
            deck.insert("yields", toml::table{
                { "channel1", toml::table{ { "channel", "ccsn" }, { "model", "sukhbold_test" } } },
                { "channel2", toml::table{ { "channel", "ccsn" }, { "model", "kobayashi_test" } } },
                { "registry", "tests/yields/assets/yields.toml" },
            });
        }
        if (feedback.has_value()) { deck.insert("feedback", std::move(feedback.value())); }
        return deck;
    };

    try
    {
        // No feedback.sn_mass_range and no yields: no star has a SN
        {
            const io::SimControls controls(makeDeck(false, std::nullopt));
            if (!controls.snMassLimits().empty())
            {
                std::cerr << "testSimControls: feedback: expected empty snMassLimits() by default\n";
                result = 1;
            }
            result += checkHasSN(controls, "no limits, no yields",
                { { 5.0, false }, { 20.0, false }, { 100.0, false } });
        }

        // One explicit interval, including both of its edges
        {
            const io::SimControls controls(makeDeck(false,
                toml::table{ { "sn_mass_range", toml::array{ 8.0, 40.0 } } }));
            if (controls.snMassLimits() != std::vector<double>{ 8.0, 40.0 })
            {
                std::cerr << "testSimControls: feedback: unexpected snMassLimits() for one interval\n";
                result = 1;
            }
            if (!controls.unusedKeys().empty())
            {
                std::cerr << "testSimControls: feedback: feedback.sn_mass_range reported unused\n";
                result = 1;
            }
            result += checkHasSN(controls, "one interval",
                { { 7.9, false }, { 8.0, true }, { 20.0, true }, { 40.0, true }, { 40.1, false } });
        }

        // Two disjoint explicit intervals, given partly as integers
        {
            const io::SimControls controls(makeDeck(false,
                toml::table{ { "sn_mass_range", toml::array{ 8, 20.0, 25.0, 40 } } }));
            result += checkHasSN(controls, "two intervals",
                { { 5.0, false }, { 10.0, true }, { 22.0, false }, { 30.0, true }, { 50.0, false } });
        }

        // No explicit limits, but ccsn yields: defer to their mass ranges
        {
            const io::SimControls controls(makeDeck(true, std::nullopt));
            result += checkHasSN(controls, "yields only",
                { { 10.0, false }, { 15.0, true }, { 18.1, false }, { 50.0, true }, { 120.0, false } });
        }

        // Explicit limits and ccsn yields: explicit limits take precedence
        {
            const io::SimControls controls(makeDeck(true,
                toml::table{ { "sn_mass_range", toml::array{ 8.0, 12.0 } } }));
            result += checkHasSN(controls, "limits override yields",
                { { 10.0, true }, { 15.0, false }, { 50.0, false } });
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSimControls: feedback: valid-deck case threw: " << error.what() << "\n";
        return 1;
    }

    // Invalid feedback.sn_mass_range values must be rejected
    const std::vector<std::pair<std::string_view, toml::table>> badDecks = {
        { "odd number of elements", toml::table{ { "sn_mass_range", toml::array{ 8.0, 20.0, 25.0 } } } },
        { "decreasing", toml::table{ { "sn_mass_range", toml::array{ 20.0, 8.0 } } } },
        { "repeated value", toml::table{ { "sn_mass_range", toml::array{ 8.0, 20.0, 20.0, 40.0 } } } },
        { "non-numeric element", toml::table{ { "sn_mass_range", toml::array{ 8.0, "twenty" } } } },
        { "not an array", toml::table{ { "sn_mass_range", 8.0 } } },
    };
    for (const auto& [label, feedback] : badDecks)
    {
        try
        {
            const io::SimControls controls(makeDeck(false, feedback));
            std::cerr << "testSimControls: feedback: expected feedback.sn_mass_range ("
                << label << ") to throw, but it did not\n";
            result = 1;
        }
        catch (const std::exception&) { /* expected */ } // NOLINT(bugprone-empty-catch) -- the throw is the expected outcome
    }

    // setSNMassLimits() directly: an invalid vector throws and leaves
    // the existing limits unchanged; an empty vector is valid and
    // makes hasSN() defer to yields again
    try
    {
        io::SimControls controls(makeDeck(true, std::nullopt));
        controls.setSNMassLimits({ 8.0, 12.0 });
        // A NaN, non-positive, or infinite limit is rejected
        const std::vector<std::vector<double>> badLimits{
            { 8.0, std::numeric_limits<double>::quiet_NaN() },
            { -5.0, 40.0 },
            { 0.0, 40.0 },
            { 8.0, std::numeric_limits<double>::infinity() } };
        for (const auto& bad : badLimits)
        {
            try
            {
                controls.setSNMassLimits(bad);
                std::cerr << "testSimControls: feedback: setSNMassLimits accepted invalid limits ["
                    << bad.at(0) << ", " << bad.at(1) << "]\n";
                result = 1;
            }
            catch (const std::invalid_argument&) { /* expected */ } // NOLINT(bugprone-empty-catch) -- the throw is the expected outcome
        }
        if (controls.snMassLimits() != std::vector<double>{ 8.0, 12.0 })
        {
            std::cerr << "testSimControls: feedback: rejected setSNMassLimits call changed snMassLimits()\n";
            result = 1;
        }
        controls.setSNMassLimits({});
        result += checkHasSN(controls, "setter cleared",
            { { 10.0, false }, { 15.0, true } });
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSimControls: feedback: setter case threw: " << error.what() << "\n";
        result = 1;
    }

    return result > 0 ? 1 : 0;
}

// Verify readFeedback()'s parsing of feedback.wr_winds, that it always
// builds winds() against this same SimControls, and setWinds()'s
// ownership check
static auto testSimControlsFeedbackWRWinds() -> int
{
    constexpr std::string_view baseDeck = "tests/core/assets/testGalaxy.in";
    int result = 0;

    auto makeDeck = [baseDeck](std::optional<toml::table> feedback) -> toml::table {
        toml::table deck = toml::parse_file(baseDeck);
        if (feedback.has_value()) { deck.insert("feedback", std::move(feedback.value())); }
        return deck;
    };

    // winds() must be built, against *this, whether or not a
    // [feedback] table was given at all
    auto checkWinds = [&result](const io::SimControls& controls, const std::string_view label) {
        if (controls.winds() == nullptr)
        {
            std::cerr << "testSimControls: feedbackWRWinds: " << label << ": winds() is null\n";
            result = 1;
        }
        else if (&controls.winds()->controls() != &controls)
        {
            std::cerr << "testSimControls: feedbackWRWinds: " << label
                << ": winds() was not built against this SimControls\n";
            result = 1;
        }
    };

    try
    {
        // No feedback.wr_winds: defaults to nugis_lamers_00
        {
            const io::SimControls controls(makeDeck(std::nullopt));
            if (controls.wrWindModel() != feedback::WRwindModel::nugisLamers00_)
            {
                std::cerr << "testSimControls: feedbackWRWinds: expected nugis_lamers_00 by default, got '"
                    << feedback::wrWindModelToString(controls.wrWindModel()) << "'\n";
                result = 1;
            }
            checkWinds(controls, "no [feedback] table");
        }

        // Each valid name selects its own model, is not reported
        // unused, and still builds winds() -- including alongside
        // feedback.sn_mass_range, which must not stop wr_winds being read
        const std::array<std::pair<std::string_view, feedback::WRwindModel>, 3> valid{ {
            { "none", feedback::WRwindModel::none_ },
            { "l_over_c", feedback::WRwindModel::lOverc_ },
            { "nugis_lamers_00", feedback::WRwindModel::nugisLamers00_ },
        } };
        for (const bool withSNRange : { false, true })
        {
            for (const auto& [name, model] : valid)
            {
                toml::table feedbackTable{ { "wr_winds", name } };
                if (withSNRange) { feedbackTable.insert("sn_mass_range", toml::array{ 8.0, 40.0 }); }
                const io::SimControls controls(makeDeck(feedbackTable));
                if (controls.wrWindModel() != model)
                {
                    std::cerr << "testSimControls: feedbackWRWinds: wr_winds = '" << name
                        << "' gave wrWindModel() = '"
                        << feedback::wrWindModelToString(controls.wrWindModel()) << "'\n";
                    result = 1;
                }
                if (!controls.unusedKeys().empty())
                {
                    std::cerr << "testSimControls: feedbackWRWinds: wr_winds = '" << name
                        << "' reported unused\n";
                    result = 1;
                }
                checkWinds(controls, name);
            }
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSimControls: feedbackWRWinds: valid-deck case threw: " << error.what() << "\n";
        return 1;
    }

    // An unrecognized name, or a value that is not a string, must be
    // rejected
    const std::vector<std::pair<std::string_view, toml::table>> badDecks = {
        { "unrecognized name", toml::table{ { "wr_winds", "nugis_lamers" } } },
        { "wrong case", toml::table{ { "wr_winds", "None" } } },
        { "not a string", toml::table{ { "wr_winds", 1.0 } } },
    };
    for (const auto& [label, feedbackTable] : badDecks)
    {
        try
        {
            const io::SimControls controls(makeDeck(feedbackTable));
            std::cerr << "testSimControls: feedbackWRWinds: expected feedback.wr_winds ("
                << label << ") to throw, but it did not\n";
            result = 1;
        }
        catch (const std::exception&) { /* expected */ } // NOLINT(bugprone-empty-catch) -- the throw is the expected outcome
    }

    // setWinds(): rejects a Winds built against a different SimControls
    // (leaving winds() unchanged), accepts one built against this one,
    // and accepts nullptr to remove it
    try
    {
        io::SimControls controls(makeDeck(std::nullopt));
        const io::SimControls otherControls(makeDeck(std::nullopt));
        const auto* const original = controls.winds().get();
        try
        {
            controls.setWinds(std::make_unique<feedback::Winds>(otherControls));
            std::cerr << "testSimControls: feedbackWRWinds: setWinds accepted a foreign Winds\n";
            result = 1;
        }
        catch (const std::invalid_argument&) { /* expected */ } // NOLINT(bugprone-empty-catch) -- the throw is the expected outcome
        if (controls.winds().get() != original)
        {
            std::cerr << "testSimControls: feedbackWRWinds: rejected setWinds call changed winds()\n";
            result = 1;
        }

        auto replacement = std::make_unique<feedback::Winds>(controls);
        const auto* const replacementPtr = replacement.get();
        controls.setWinds(std::move(replacement));
        if (controls.winds().get() != replacementPtr)
        {
            std::cerr << "testSimControls: feedbackWRWinds: setWinds did not install the new Winds\n";
            result = 1;
        }

        controls.setWinds(nullptr);
        if (controls.winds() != nullptr)
        {
            std::cerr << "testSimControls: feedbackWRWinds: setWinds(nullptr) did not remove winds()\n";
            result = 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSimControls: feedbackWRWinds: setWinds case threw: " << error.what() << "\n";
        result = 1;
    }

    return result > 0 ? 1 : 0;
}

// Verify readFeedback()'s parsing of feedback.ob_winds, independently
// of feedback.wr_winds (setWinds() itself is covered by
// testSimControlsFeedbackWRWinds, and does not depend on either model)
static auto testSimControlsFeedbackOBWinds() -> int
{
    constexpr std::string_view baseDeck = "tests/core/assets/testGalaxy.in";
    int result = 0;

    auto makeDeck = [baseDeck](std::optional<toml::table> feedback) -> toml::table {
        toml::table deck = toml::parse_file(baseDeck);
        if (feedback.has_value()) { deck.insert("feedback", std::move(feedback.value())); }
        return deck;
    };

    try
    {
        // No feedback.ob_winds: defaults to vink_sander_21
        {
            const io::SimControls controls(makeDeck(std::nullopt));
            if (controls.obWindModel() != feedback::OBwindModel::vinkSander21_)
            {
                std::cerr << "testSimControls: feedbackOBWinds: expected vink_sander_21 by default, got '"
                    << feedback::obWindModelToString(controls.obWindModel()) << "'\n";
                result = 1;
            }
        }

        // Each valid name selects its own model, is not reported
        // unused, and still builds winds() -- both alone and alongside
        // feedback.wr_winds, which must each be read independently of
        // the other
        const std::array<std::pair<std::string_view, feedback::OBwindModel>, 3> valid{ {
            { "none", feedback::OBwindModel::none_ },
            { "vink_01", feedback::OBwindModel::vink01_ },
            { "vink_sander_21", feedback::OBwindModel::vinkSander21_ },
        } };
        for (const bool withWRWinds : { false, true })
        {
            for (const auto& [name, model] : valid)
            {
                toml::table feedbackTable{ { "ob_winds", name } };
                if (withWRWinds) { feedbackTable.insert("wr_winds", "l_over_c"); }
                const io::SimControls controls(makeDeck(feedbackTable));
                if (controls.obWindModel() != model)
                {
                    std::cerr << "testSimControls: feedbackOBWinds: ob_winds = '" << name
                        << "' gave obWindModel() = '"
                        << feedback::obWindModelToString(controls.obWindModel()) << "'\n";
                    result = 1;
                }
                const auto expectedWR = withWRWinds ?
                    feedback::WRwindModel::lOverc_ : feedback::WRwindModel::nugisLamers00_;
                if (controls.wrWindModel() != expectedWR)
                {
                    std::cerr << "testSimControls: feedbackOBWinds: ob_winds = '" << name
                        << "' changed wrWindModel() to '"
                        << feedback::wrWindModelToString(controls.wrWindModel()) << "'\n";
                    result = 1;
                }
                if (!controls.unusedKeys().empty())
                {
                    std::cerr << "testSimControls: feedbackOBWinds: ob_winds = '" << name
                        << "' reported unused\n";
                    result = 1;
                }
                if (controls.winds() == nullptr)
                {
                    std::cerr << "testSimControls: feedbackOBWinds: ob_winds = '" << name
                        << "': winds() is null\n";
                    result = 1;
                }
            }
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSimControls: feedbackOBWinds: valid-deck case threw: " << error.what() << "\n";
        return 1;
    }

    // An unrecognized name, or a value that is not a string, must be
    // rejected
    const std::vector<std::pair<std::string_view, toml::table>> badDecks = {
        { "unrecognized name", toml::table{ { "ob_winds", "vink01" } } },
        { "wrong case", toml::table{ { "ob_winds", "None" } } },
        { "WR model name", toml::table{ { "ob_winds", "nugis_lamers_00" } } },
        { "not a string", toml::table{ { "ob_winds", 1.0 } } },
    };
    for (const auto& [label, feedbackTable] : badDecks)
    {
        try
        {
            const io::SimControls controls(makeDeck(feedbackTable));
            std::cerr << "testSimControls: feedbackOBWinds: expected feedback.ob_winds ("
                << label << ") to throw, but it did not\n";
            result = 1;
        }
        catch (const std::exception&) { /* expected */ } // NOLINT(bugprone-empty-catch) -- the throw is the expected outcome
    }

    return result > 0 ? 1 : 0;
}

// Verify Yields::hasYield(mass, feH, channel) and SimControls::hasSN(mass,
// feH) against the synthetic gap_test ccsn model (masses 10-40 Msun,
// whose 20 and 30 Msun yields are all zero at [Fe/H] = 0 -- see
// make_yields_test_fixture.py): at [Fe/H] = 0, a 25 Msun star lies
// within the model's mass range, so the mass-only hasYield()/hasSN()
// report a yield/SN, but it lies in a failed-supernova gap, so the
// [Fe/H]-aware versions do not. Also checks that explicit SN mass limits
// still take precedence over yields in hasSN(mass, feH), and that
// hasSN(mass, feH) is false with neither.
static auto testSimControlsHasSNFeH() -> int
{
    constexpr std::string_view baseDeck = "tests/core/assets/testGalaxy.in";
    const auto ccsn = yields::Channel::ccsn_;
    int result = 0;
    try
    {
        toml::table inputDeck = toml::parse_file(baseDeck);
        inputDeck.insert("yields", toml::table{
            { "channel1", toml::table{ { "channel", "ccsn" }, { "model", "gap_test" } } },
            { "registry", "tests/yields/assets/yields.toml" },
        });
        io::SimControls controls(inputDeck);
        const auto yields = controls.yields();
        if (yields == nullptr)
        {
            std::cerr << "testSimControls: hasSNFeH: expected yields() non-null\n";
            return 1;
        }

        struct Case
        {
            std::string_view label_;
            bool actual_;
            bool expected_;
        };
        const std::array<Case, 9> cases = { {
            { "yields()->hasYield(25, ccsn)", yields->hasYield(25.0, ccsn), true },
            { "yields()->hasYield(25, 0, ccsn)", yields->hasYield(25.0, 0.0, ccsn), false },
            { "yields()->hasYield(15, 0, ccsn)", yields->hasYield(15.0, 0.0, ccsn), true },
            { "yields()->hasYield(35, 0, ccsn)", yields->hasYield(35.0, 0.0, ccsn), true },
            { "yields()->hasYield(25, 0, massive_star_winds)",
                yields->hasYield(25.0, 0.0, yields::Channel::massiveStarWinds_), false },
            { "hasSN(25)", controls.hasSN(25.0), true },
            { "hasSN(25, 0)", controls.hasSN(25.0, 0.0), false },
            { "hasSN(15, 0)", controls.hasSN(15.0, 0.0), true },
            { "hasSN(45, 0)", controls.hasSN(45.0, 0.0), false },
        } };
        for (const auto& c : cases)
        {
            if (c.actual_ != c.expected_)
            {
                std::cerr << "testSimControls: hasSNFeH: " << c.label_ << " returned "
                    << c.actual_ << ", expected " << c.expected_ << "\n";
                result = 1;
            }
        }

        // Explicit limits take precedence over yields, [Fe/H] or not
        controls.setSNMassLimits({ 20.0, 30.0 });
        if (!controls.hasSN(25.0, 0.0) || controls.hasSN(15.0, 0.0))
        {
            std::cerr << "testSimControls: hasSNFeH: explicit limits [20, 30] did not "
                "override yields in hasSN(mass, feH)\n";
            result = 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSimControls: hasSNFeH: threw: " << error.what() << "\n";
        return 1;
    }

    // No limits and no yields: no SNe
    try
    {
        const io::SimControls controls(toml::parse_file(baseDeck));
        if (controls.hasSN(25.0, 0.0))
        {
            std::cerr << "testSimControls: hasSNFeH: expected hasSN(25, 0) false "
                "with no limits and no yields\n";
            result = 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSimControls: hasSNFeH: no-yields case threw: " << error.what() << "\n";
        return 1;
    }
    return result;
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
    result += testSimControlsCheckpointInterval();
    result += testSimControlsCheckpointAsciiThrows();
    result += testSimControlsGalaxy();
    result += testSimControlsPhysicsCluster();
    result += testSimControlsPhysicsGalaxy();
    result += testSimControlsFCluster();
    result += testSimControlsNoSpectraModel();
    result += testSimControlsNebularDefaultNoSpectra();
    result += testSimControlsNebularExplicitNoSpectra();
    result += testSimControlsNebularDefaultWithSpectra();
    result += testSimControlsInvalidSpectraModel();
    result += testSimControlsSpectraLibrary();
    result += testSimControlsSpectraSkipTrackPadding();
    result += testSimControlsSpectraChained();
    result += testSimControlsExtinctField();
    result += testSimControlsYields();
    result += testSimControlsYieldsNoDecay();
    result += testSimControlsWriteYields();
    result += testSimControlsYieldsIsotopes();
    result += testSimControlsYieldsIsotopesKeyword();
    result += testSimControlsYieldsIsotopesDecayClosure();
    result += testSimControlsYieldsIsotopesEmittedParticles();
    result += testSimControlsYieldsYieldAndSum();
    result += testSimControlsYieldsPartialRange();
    result += testSimControlsYieldsDecayApplication();
    result += testSimControlsMinIsotopeLifetime();
    result += testSimControlsYieldsDuplicateChannelWarning();
    result += testSimControlsSFRDist();
    result += testSimControlsSetFeHRejectsBroadening();
    result += testSimControlsSetTracksRejectsUncoveredFeH();
    result += testSimControlsIMFWithinTracks();
    result += testSimControlsSetFeHResetsTracks2DWhenNoLongerFixed();
    result += testSimControlsSettersRejectMismatchedControls();
    result += testSimControlsSetSpecsynRejectsNarrowerFeh();
    result += testSimControlsSetYieldsRejectsNarrowerFeh();
    result += testSimControlsSetFeHRejectsBeyondSpecsynYields();
    result += testSimControlsUnusedKeys();
    result += testSimControlsStrictInput();
    result += testSimControlsIgnoredKeys();
    result += testSimControlsYieldsChannelGap();
    result += testSimControlsFracStochMass();
    result += testSimControlsYieldsHasYield();
    result += testSimControlsFeedback();
    result += testSimControlsFeedbackWRWinds();
    result += testSimControlsFeedbackOBWinds();
    result += testSimControlsHasSNFeH();
    return result;
}
