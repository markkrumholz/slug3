/**
 * @file SimControls.cpp
 * @author Mark Krumholz
 * @date 2026-07-16
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 * @brief Implementation of SimControls
 */

#include "SimControls.hpp"
#include "../elem/ElemCommons.hpp"
#include "../elem/IonizationData.hpp"
#include "../elem/IsotopeData.hpp"
#include "../elem/IsotopeTable.hpp"
#include "../extinct/Extinct.hpp"
#include "../feedback/FeedbackCommons.hpp"
#include "../feedback/Winds.hpp"
#include "../nebular/Nebular.hpp"
#include "../nebular/NebularCommons.hpp"
#include "../pdfs/PDF.hpp"
#include "../pdfs/PDFFileParser.hpp"
#include "../pdfs/PDFSegmentDelta.hpp"
#include "../pdfs/PDFSegmentPowerlaw.hpp"
#include "../phot/FilterCollection.hpp"
#include "../phot/FilterCommons.hpp"
#include "../phot/PhotCommons.hpp"
#include "../phot/VegaSpectrum.hpp"
#include "../specsyn/Specsyn.hpp"
#include "../specsyn/SpecsynBlackbody.hpp"
#include "../specsyn/SpecsynCommons.hpp"
#include "../specsyn/SpecsynLibChained.hpp"
#include "../specsyn/SpecsynLibNoWind.hpp"
#include "../specsyn/SpecsynLibWR.hpp"
#include "../specsyn/SpecsynUtils.hpp"
#include "../tracks/TrackCommons.hpp"
#include "../tracks/Tracks2D.hpp"
#include "../tracks/Tracks3D.hpp"
#include "../utils/MPIUtils.hpp"
#include "../utils/ParseUtils.hpp"
#include "../utils/RngThread.hpp"
#include "../utils/TOMLUtils.hpp"
#include "../utils/TrackedDeck.hpp"
#include "../yields/YieldCommons.hpp"
#include "../yields/Yields.hpp"
#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <toml.hpp>
#include <utility>
#include <vector>

// Build a constant-in-time PDF representing a fixed star formation
// rate, used by both the constructor's own galaxy.sfr handling and
// setSFR() below (and, via its own public static declaration, by
// Galaxy::Galaxy() -- see buildConstantSFR()'s own header comment).
// Represented as a single flat (alpha = 0) powerlaw segment spanning
// [0, tMax], with tMax fixed far larger than any realistic simulation
// timescale so that every query [a, b] this PDF ever sees (cluster
// formation times, sfr().integral() over an advance() step) falls
// well inside the flat region, never clamped against tMax. A flat
// segment's own normalized integral over [a, b] is (b - a) / tMax
// (see PDFSegmentPowerlaw::integral()), so setting the segment's
// PDF-level weight to sfr * tMax makes PDF::integral(a, b) -- weight
// times that normalized integral -- come out to exactly sfr * (b - a),
// matching the physical definition of a star formation rate.
auto io::SimControls::buildConstantSFR(const double sfr) -> pdfs::PDF
{
    constexpr double tMax = 1e15; // yr; far beyond any realistic simulation time
    const double wgt = sfr * tMax;
    auto pl = std::make_unique<pdfs::PDFSegmentPowerlaw>(0.0, tMax, 0.0);
    return pdfs::PDF(std::move(pl), wgt);
}

// Read an explicit array of output times from output.output_times
static auto readOutputTimesArray(const utils::TrackedDeck& inputDeck) -> std::vector<double>
{
    const toml::array* arr = inputDeck.atPath("output.output_times").as_array();
    if (arr == nullptr)
    {
        throw std::runtime_error(
            "SimControls: output.output_times must be an array of numbers");
    }
    std::vector<double> times;
    times.reserve(arr->size());
    for (const auto& elem : *arr)
    {
        const auto val = elem.value<double>();
        if (!val.has_value())
        {
            throw std::runtime_error(
                "SimControls: output.output_times must be an array of numbers");
        }
        times.push_back(val.value());
    }
    std::ranges::sort(times); // Times must be sorted
    return times;
}

// Generate a uniformly- or log-spaced grid of output times from
// output.start_time, output.end_time, output.ntime, and the
// optional output.log_time
static auto generateOutputTimesRange(const utils::TrackedDeck& inputDeck) -> std::vector<double>
{
    const auto startTimeInput = inputDeck.value<double>("output.start_time", true);
    if (!startTimeInput.has_value())
    {
        throw std::runtime_error("SimControls: output.start_time not found");
    }
    const double startTime = startTimeInput.value();

    const auto endTimeInput = inputDeck.value<double>("output.end_time", true);
    if (!endTimeInput.has_value())
    {
        throw std::runtime_error("SimControls: output.end_time not found");
    }
    const double endTime = endTimeInput.value();

    const auto nTimeInput = inputDeck.value<unsigned long>("output.ntime", true);
    if (!nTimeInput.has_value())
    {
        throw std::runtime_error("SimControls: output.ntime not found");
    }
    const unsigned long nTime = nTimeInput.value();

    const bool logTime = inputDeck.value<bool>("output.log_time").value_or(false);

    if (startTime < 0.0 || endTime < 0.0)
    {
        throw std::runtime_error(
            "SimControls: output.start_time and output.end_time must be >= 0");
    }
    if (nTime == 0)
    {
        throw std::runtime_error("SimControls: output.ntime must be > 0");
    }
    if ((startTime == endTime) != (nTime == 1))
    {
        throw std::runtime_error(
            "SimControls: output.start_time == output.end_time is allowed "
            "only if output.ntime == 1, and output.ntime == 1 is allowed "
            "only if output.start_time == output.end_time");
    }
    if (logTime && startTime <= 0.0)
    {
        throw std::runtime_error(
            "SimControls: output.start_time must be > 0 when output.log_time is set");
    }

    std::vector<double> times(nTime, 0.0);
    if (nTime == 1)
    {
        times.at(0) = startTime;
    }
    else if (logTime)
    {
        const double logStart = std::log(startTime);
        const double logEnd = std::log(endTime);
        const double dLog = (logEnd - logStart) / static_cast<double>(nTime - 1);
        for (unsigned long i = 0; i < nTime; ++i)
        {
            times.at(i) = std::exp(logStart + (static_cast<double>(i) * dLog));
        }
    }
    else
    {
        const double dt = (endTime - startTime) / static_cast<double>(nTime - 1);
        for (unsigned long i = 0; i < nTime; ++i)
        {
            times.at(i) = startTime + (static_cast<double>(i) * dt);
        }
    }
    return times;
}

io::SimControls::SimControls(const toml::table& inputDeck)
{
    // Cache the deck's own text before parsing anything out of it, so
    // OutputManagerH5/OutputManagerAscii can later record exactly what
    // was actually used, without needing a toml::table of their own --
    // see inputDeckStr()'s own comment.
    std::ostringstream inputDeckStream;
    inputDeckStream << inputDeck;
    inputDeckStr_ = inputDeckStream.str();

    // From here on every read of the deck goes through a TrackedDeck,
    // which records which keys are used so that any that never are can
    // be reported at the end -- see reportUnusedKeys()
    const utils::TrackedDeck trackedDeck(inputDeck);
    initControlFlow(trackedDeck);
    initPhysics(trackedDeck);
    reportUnusedKeys(trackedDeck);
}

// Text identifying an input-deck key in a message: its path, and its
// line if the deck was parsed from text
static auto describeDeckKey(const utils::DeckKeyReport& key) -> std::string
{
    std::string text = "'" + key.path_ + "'";
    if (key.line_ != 0) { text += " (line " + std::to_string(key.line_) + ")"; }
    return text;
}

// Message for a key that was never used: where it is, and what it might
// have been meant to be
static auto describeUnusedDeckKey(const utils::DeckKeyReport& key) -> std::string
{
    std::string text = "input deck key " + describeDeckKey(key) + " was never used";
    if (!key.hint_.empty()) { text += "; " + key.hint_; }
    return text;
}

// Record which input-deck keys were never read, and report them. A key
// that is deliberately not read for a legitimate reason (e.g.
// nebular.log_U when nebular.compute_neb is false) was marked ignored
// where that decision was made, and is only mentioned at verbosity >= 1.
// An unused key is a warning, or an error if strict_input is set.
void io::SimControls::reportUnusedKeys(const utils::TrackedDeck& deck)
{
    unusedKeys_ = deck.unused();
    ignoredKeys_ = deck.ignored();

    if (strictInput_ && !unusedKeys_.empty())
    {
        std::string message = "SimControls: strict_input is set, but the input deck has keys that were never used:";
        for (const auto& key : unusedKeys_) { message += "\n  " + describeUnusedDeckKey(key); }
        throw std::runtime_error(message);
    }
    // The deck is the same on every MPI rank, so every rank has the
    // same keys to report: print them once, from the I/O rank (the
    // stored lists above are still populated on every rank, and the
    // strict_input throw above still happens on every rank)
    if (!utils::isIORank()) { return; }
    for (const auto& key : unusedKeys_)
    {
        std::cout << "slug: warning: " << describeUnusedDeckKey(key) << "\n";
    }
    if (verbosity_ < 1) { return; }
    for (const auto& key : ignoredKeys_)
    {
        std::cout << "slug: note: input deck key " << describeDeckKey(key) << " was ignored: " << key.reason_ << "\n";
    }
}

void io::SimControls::initControlFlow(const utils::TrackedDeck& inputDeck)
{
    // Determine simulation type
    const auto simTypeInput = inputDeck.value<std::string>("sim_type", true);
    if (!simTypeInput.has_value())
    {
        throw std::runtime_error("SimControls: sim_type not found");
    }
    if (simTypeInput.value() == "galaxy") { simType_ = SimType::galaxy; }
    else if (simTypeInput.value() == "cluster") { simType_ = SimType::cluster; }
    else
    {
        throw std::runtime_error("SimControls: sim_type must be 'galaxy' or 'cluster'");
    }

    // Read verbosity
    const auto verbosityInput =
        inputDeck.value<unsigned int>("verbosity");
    if (verbosityInput.has_value()) { verbosity_ = verbosityInput.value(); }

    // Read whether an input-deck key that is never used is an error,
    // rather than just a warning -- see reportUnusedKeys()
    strictInput_ = inputDeck.value<bool>("strict_input").value_or(false);

    // Read output mode
    const auto outputMode = inputDeck.value<std::string>("output.output_mode");
    if (outputMode.has_value())
    {
        if (outputMode.value() == "h5" || outputMode.value() == "hdf5")
        { outputMode_ = OutputMode::h5; }
        else if (outputMode.value() == "h5divided" || outputMode.value() == "hdf5divided")
        { outputMode_ = OutputMode::h5divided; }
        else if (outputMode.value() == "ascii" || outputMode.value() == "txt")
        { outputMode_ = OutputMode::ascii; }
        else
        {
            throw std::runtime_error("SimControls: unknown output_mode "
                 + outputMode.value());
        }
    }

    // Read model name
    const auto modelNameInput =
        inputDeck.value<std::string>("output.model_name");
    if (modelNameInput.has_value()) { modelName_ = modelNameInput.value(); }

    // Read output directory
    const auto outDirInput =
        inputDeck.value<std::string>("output.out_dir");
    if (outDirInput.has_value()) { outDir_ = outDirInput.value(); }

    // Read number of trials
    const auto nTrialInput =
        inputDeck.value<unsigned long>("n_trial");
    if (nTrialInput.has_value())
    {
        nTrial_ = nTrialInput.value();
    }

    // Read the checkpoint interval (optional; 0 = disabled, the
    // default). Checkpointing is only supported with HDF5 output --
    // OutputManagerAscii has no way to reopen/append to an ascii file
    // it has already finished writing, unlike OutputManagerH5 rolling
    // over to a new HDF5 file (see OutputManagerH5::checkpoint()'s own
    // comment) -- so a non-zero interval combined with ascii output is
    // rejected here, at construction, rather than left to fail later,
    // mid-run, the first time a checkpoint would actually be attempted.
    const auto checkpointIntervalInput = inputDeck.value<unsigned long>("output.checkpoint_interval");
    if (checkpointIntervalInput.has_value())
    {
        checkpointInterval_ = checkpointIntervalInput.value();
    }
    if (checkpointInterval_ != 0 && outputMode_ == OutputMode::ascii)
    {
        throw std::runtime_error(
            "SimControls: output.checkpoint_interval is non-zero, but "
            "checkpointing is only supported with HDF5 output "
            "(output.output_mode = \"h5\" or \"h5divided\"), not ascii");
    }

    // Handle output time generation
    setOutputTimes(inputDeck);

    // Read the output content flags (output.write_cluster/etc.)
    readOutput(inputDeck);

    // If we have been given a specific RNG seed, use it
    const auto rngSeed =
        inputDeck.value<unsigned long>("rng_seed");
    if (rngSeed.has_value()) { utils::rng().seed(rngSeed.value()); }

    // Read optional integrator tolerance controls; defaults match
    // PDFIntegrator's own defaults so omitting them is a no-op
    const auto relTolInput = inputDeck.value<double>("integrator.rel_tol");
    if (relTolInput.has_value()) { intRelTol_ = relTolInput.value(); }
    const auto absTolInput = inputDeck.value<double>("integrator.abs_tol");
    if (absTolInput.has_value()) { intAbsTol_ = absTolInput.value(); }
    const auto maxIterInput = inputDeck.value<std::size_t>("integrator.max_iter");
    if (maxIterInput.has_value()) { intMaxIter_ = maxIterInput.value(); }
}

void io::SimControls::initPhysics(const utils::TrackedDeck& inputDeck)
{
    // Read IMF, CMF, and FeH
    imf_ = inputDeck.initPDF("stars.IMF", imfPrefix);
    cmf_ = inputDeck.initPDF("clusters.CMF");
    fehDist_ = inputDeck.initPDF("stars.FeH");

    // Read the tracks. Needs fehDist_ (just set above) to pick the
    // [Fe/H] range to load; done before readSpectra() (which used to
    // come first here) because readSpectra() itself now needs
    // tracks_'s own requested [Fe/H] range -- see its own comment.
    readTracks(inputDeck);

    // Reject an IMF extending above the tracks' maximum mass, and warn
    // if it extends below their minimum mass -- see
    // checkIMFWithinTracks()'s own comment
    checkIMFWithinTracks("SimControls", imf_, *tracks_);

    // Read the spectral synthesis model to use, if any -- spectra.model
    // is optional, since not every simulation needs spectra computed.
    // Needs tracks_ (just set above) to pick the [Fe/H] range a
    // library-based model is loaded over -- see readSpectra()'s own
    // comment for why that's the tracks' requested range, not their
    // padded grid.
    readSpectra(inputDeck);

    // Read the photometric filter collection to use, if any --
    // phot.filters is optional. Needs specsyn_ (just set above) to
    // check that a spectral synthesizer is actually available before
    // building a filter collection that will need one.
    readFilters(inputDeck);

    // Read the nebular emission controls and grid to use. Unlike
    // readSpectra/readFilters/readExtinct, nothing here is gated on
    // any input-deck key being present -- every nebular.* control
    // parameter independently falls back to its own default, and a
    // Nebular is constructed by default whenever a spectral
    // synthesizer (specsyn_, set above) is available for it to be
    // paired with (stars.tracks is already mandatory, via
    // readTracks() above). Done before readExtinct()
    // (which used to come first here) because readExtinct() itself
    // now needs nebular_ -- see Extinct's own constructor comment on
    // extinctLines_.
    readNebular(inputDeck);

    // Read the extinction curve to apply, if any -- extinct.AV is
    // optional. Needs specsyn_ (set above) to provide the wavelength
    // grid the extinction curve is interpolated onto, and nebular_
    // (just set above) to provide the line wavelengths extinctLines_
    // is interpolated onto.
    readExtinct(inputDeck);

    // Read the nucleosynthetic yield channels requested, if any -- see
    // readYields()'s own comment.
    readYields(inputDeck);

    // Read the stellar feedback controls, if any -- see
    // readFeedback()'s own comment.
    readFeedback(inputDeck);

    // In a galaxy simulation, read CLF, SFR, and the stochastic
    // cluster mass fraction
    if (simType_ == SimType::galaxy)
    {
        // CLF
        clf_ = inputDeck.initPDF("clusters.CLF");

        // SFR: exactly one of galaxy.sfr (a rate as a function of
        // time -- possibly constant, possibly not, see below) or
        // galaxy.sfr_dist (a distribution to draw a single, constant-
        // for-the-whole-simulation rate from once per Galaxy -- see
        // sfrDist()'s own comment) is required.
        const auto sfrNode = inputDeck.atPath("galaxy.sfr");
        const auto sfrDistNode = inputDeck.atPath("galaxy.sfr_dist");
        if (sfrNode && sfrDistNode)
        {
            throw std::runtime_error(
                "SimControls: galaxy.sfr and galaxy.sfr_dist cannot "
                "both be given");
        }
        if (sfrNode)
        {
            // galaxy.sfr requires special handling because here a
            // numerical value is not interpreted as a delta function
            // but as the normalization for a non-normalized PDF that
            // is constant in time -- see buildConstantSFR()'s own
            // comment
            const std::optional<double> sfr = sfrNode.value<double>();
            if (sfr.has_value())
            {
                sfr_ = buildConstantSFR(sfr.value());
            }
            else
            {
                // We have been given a file
                const std::optional<std::string> sfrFile = sfrNode.value<std::string>();
                if (sfrFile.has_value())
                {
                    sfr_ = pdfs::parsePDFDescriptor(sfrFile.value());
                }
                else
                {
                    throw std::runtime_error(
                        "SimControls: invalid entry for galaxy.sfr");
                }
            }
        }
        else if (sfrDistNode)
        {
            // galaxy.sfr_dist is a genuine distribution -- unlike
            // galaxy.sfr, a numerical value here is interpreted as an
            // ordinary delta function, via the same utils::
            // initPDFFromKey() every other PDF-valued key uses
            sfrDist_ = inputDeck.initPDF("galaxy.sfr_dist");
        }
        else
        {
            throw std::runtime_error(
                "SimControls: a galaxy-type simulation requires either "
                "galaxy.sfr or galaxy.sfr_dist to be given");
        }

        // Fraction of stellar mass formed in stochastically-treated
        // clusters, optional, defaults to fCluster_'s own in-class
        // default of 1.0
        const auto fClusterInput = inputDeck.value<double>("clusters.f_cluster");
        if (fClusterInput.has_value()) { fCluster_ = fClusterInput.value(); }
    }
    else
    {
        // These keys only mean anything for a galaxy-type simulation
        constexpr auto reason = "sim_type is \"cluster\"; this key is only used by galaxy simulations";
        inputDeck.markIgnored("clusters.CLF", reason);
        inputDeck.markIgnored("clusters.f_cluster", reason);
        inputDeck.markIgnored("galaxy", reason);
    }

    // If this simulation has a fixed [Fe/H], precompute the slice at
    // that value once here, up front, so that Cluster objects can
    // share it for the lifetime of the simulation instead of each
    // computing their own copy or racing to populate Tracks3D's
    // internal cache
    if (constFeH())
    {
        constFeHTracks_ = std::make_shared<tracks::Tracks2D>(tracks_->sliceConstFeH(fehDist_.getMin()));
    }

    // Read minimum stochastic mass
    auto minSM = inputDeck.value<double>("stars.min_stoch_mass");
    if (minSM.has_value())
    {
        minStochMass_ = minSM.value();
        updateFracStochMass();
    }
}

void io::SimControls::setOutputTimes(const utils::TrackedDeck& inputDeck)
{
    // This routine computes the output times. A user can specify the
    // times of outputs in one of two ways:
    // (1) The user can set output.output_times. This is tried first as
    //     a PDF, via utils::initPDFFromKey: a single number is
    //     interpreted as a delta function, giving one output per
    //     simulation at that time, and a string is interpreted as the
    //     name of a PDF file, giving one output per simulation drawn
    //     from that PDF. If that interpretation fails -- most commonly
    //     because the value is an array -- output.output_times is
    //     instead read as an explicit array of output times, via
    //     readOutputTimesArray, giving one output per trial at each of
    //     the specified times. This mirrors the try-then-fall-back
    //     convention used elsewhere for a key that can be given in more
    //     than one form (e.g. SimControls's own constructor, which
    //     tries its input deck argument as literal toml text before
    //     falling back to a file path).
    // (2) The user can specify all three of output.start_time,
    //     output.end_time, and output.ntime, with the first two interpreted
    //     as doubles (required to be >= 0) and the third as an unsigned int (which must be > 0). In
    //     this case the output times will be automatically generated as a
    //     uniformly-spaced array of ntime values from start_time to end_time.
    //     (For this option, start_time == end_time is allowed only if ntime = 1,
    //     and ntime == 1 is allowed only if start_time == end_time.)
    // (2a) For option 2, the user can also specify the optional boolean
    //      output.log_time; if this is specified, the array of values generated
    //      will be log-spaced rather than linearly spaced.
    // Thus in this routine we need to check which of these options the user has
    // provided, verify that only that option is been provided (e.g., the user
    // hasn't accidentially provided both output_times and start_time/end_time/ntime), and
    // fill the variables outTimes_ and outTimeDist_ based on them. For option 1's
    // PDF interpretation, outTimeDist_ will be set to a valid PDF and outTimes_
    // will be left empty, while for option 1's array interpretation or option 2,
    // outTimes_ will be a non-empty array and outTimeDist_ will be left as an
    // invalid, uninitialized PDF.

    // Determine which option(s) the user has specified
    const bool hasTimes = static_cast<bool>(inputDeck.atPath("output.output_times"));
    const bool hasStart = static_cast<bool>(inputDeck.atPath("output.start_time"));
    const bool hasEnd = static_cast<bool>(inputDeck.atPath("output.end_time"));
    const bool hasNTime = static_cast<bool>(inputDeck.atPath("output.ntime"));
    const bool hasRange = hasStart || hasEnd || hasNTime;

    const int nOptions = static_cast<int>(hasTimes) + static_cast<int>(hasRange);
    if (nOptions == 0)
    {
        throw std::runtime_error(
            "SimControls: must specify one of output.output_times, "
            "or output.start_time/output.end_time/output.ntime");
    }
    if (nOptions > 1)
    {
        throw std::runtime_error(
            "SimControls: only one of output.output_times, "
            "or output.start_time/output.end_time/output.ntime "
            "may be specified");
    }

    // Option 1: output.output_times, tried first as a PDF, falling
    // back to an explicit array of output times if that fails
    if (hasTimes)
    {
        try
        {
            outTimeDist_ = inputDeck.initPDF("output.output_times");
        }
        catch (const std::runtime_error&)
        {
            outTimes_ = readOutputTimesArray(inputDeck);
        }
        return;
    }

    // Option 2: a uniformly- or log-spaced grid of output times
    if (!(hasStart && hasEnd && hasNTime))
    {
        throw std::runtime_error(
            "SimControls: output.start_time, output.end_time, and "
            "output.ntime must all be specified together");
    }
    outTimes_ = generateOutputTimesRange(inputDeck);
}

// Read an optional output.<key> boolean, defaulting to true if absent
static auto readWriteFlag(const utils::TrackedDeck& inputDeck, const std::string& key) -> bool
{
    const auto value = inputDeck.value<bool>(key);
    return !value.has_value() || value.value();
}

void io::SimControls::readOutput(const utils::TrackedDeck& inputDeck)
{
    writeCluster_ = readWriteFlag(inputDeck, "output.write_cluster");
    writeClusterSpec_ = readWriteFlag(inputDeck, "output.write_cluster_spec");
    writeClusterPhot_ = readWriteFlag(inputDeck, "output.write_cluster_phot");
    writeClusterYields_ = readWriteFlag(inputDeck, "output.write_cluster_yields");
    writeClusterFeedback_ = readWriteFlag(inputDeck, "output.write_cluster_feedback");
    writeGalaxy_ = readWriteFlag(inputDeck, "output.write_galaxy");
    writeGalaxySpec_ = readWriteFlag(inputDeck, "output.write_galaxy_spec");
    writeGalaxyPhot_ = readWriteFlag(inputDeck, "output.write_galaxy_phot");
    writeGalaxyYields_ = readWriteFlag(inputDeck, "output.write_galaxy_yields");
    writeGalaxyFeedback_ = readWriteFlag(inputDeck, "output.write_galaxy_feedback");
}

// Set the [Fe/H] distribution, recomputing tracks2D() (constFeHTracks_)
// from the current tracks_ if the new fehDist_ is fixed -- mirrors the
// constructor's own post-readTracks() step, which does the same thing
// once, after both tracks_ and fehDist_ are set
void io::SimControls::setFeH(const std::string& feH)
{
    auto newFehDist = utils::initPDFFromString(feH);
    // Written so that a NaN range (a default-constructed Tracks3D) is
    // rejected, not silently accepted
    if (!(tracks_->fehMin() <= newFehDist.getMin() && newFehDist.getMax() <= tracks_->fehMax())) // NOLINT(readability-simplify-boolean-expr) -- the De Morgan form would accept a NaN range, since every comparison with NaN is false
    {
        throw std::runtime_error(
            "SimControls::setFeH: the requested [Fe/H] distribution, "
            "[" + std::to_string(newFehDist.getMin()) + ", " +
            std::to_string(newFehDist.getMax()) + "], is broader than "
            "the range the stellar tracks were loaded over, [" +
            std::to_string(tracks_->fehMin()) + ", " +
            std::to_string(tracks_->fehMax()) + "] (requested at "
            "construction via stars.FeH), so the [Fe/H] distribution "
            "cannot be broadened past that without risking an "
            "out-of-range interpolation. Construct a new SimControls "
            "with a wider stars.FeH range instead if you need one.");
    }

    // Nor beyond the [Fe/H] range the installed spectral synthesizer or
    // Yields was constructed for (see checkSpecsynReplacement()'s own
    // comment): e.g. after setTracks() has installed wider tracks, the
    // tracks check above would otherwise let fehDist_ widen past them
    const auto checkCovered = [&newFehDist](const std::string& what, const double lo, const double hi)
    {
        if (!(lo <= newFehDist.getMin() && newFehDist.getMax() <= hi)) // NOLINT(readability-simplify-boolean-expr) -- the De Morgan form would accept a NaN range, since every comparison with NaN is false
        {
            throw std::runtime_error(
                "SimControls::setFeH: the requested [Fe/H] distribution, [" +
                std::to_string(newFehDist.getMin()) + ", " + std::to_string(newFehDist.getMax()) +
                "], extends beyond the range the installed " + what + " was constructed "
                "for, [" + std::to_string(lo) + ", " + std::to_string(hi) + "]. Install a " +
                what + " constructed for a range covering it first.");
        }
    };
    if (specsyn_) { checkCovered("spectral synthesizer", specsyn_->requestedFehMin(), specsyn_->requestedFehMax()); }
    if (yields_) { checkCovered("Yields", yields_->requestedFehMin(), yields_->requestedFehMax()); }

    // Build the fixed-[Fe/H] slice, which can itself throw, before
    // changing any state; if the new distribution is not degenerate,
    // the slice is cleared instead -- otherwise constFeHTracks_ would
    // keep pointing at a stale slice from whichever single [Fe/H] value
    // used to apply, silently contradicting tracks2D()'s own "nullptr
    // if constFeH() is false" contract
    decltype(constFeHTracks_) newConstFeHTracks;
    if (newFehDist.getMin() == newFehDist.getMax())
    {
        newConstFeHTracks = std::make_shared<tracks::Tracks2D>(tracks_->sliceConstFeH(newFehDist.getMin()));
    }
    fehDist_ = std::move(newFehDist);
    constFeHTracks_ = std::move(newConstFeHTracks);
}

// Set the stellar tracks, recomputing tracks2D() (constFeHTracks_)
// from the new tracks_ if fehDist_ is already fixed -- see setFeH()'s
// own comment
// Throw unless tracks cover the current [Fe/H] distribution -- see
// this method's own header comment
void io::SimControls::checkTracksCoverFeH(const tracks::Tracks3D& tracks) const
{
    // The same requirement setFeH() enforces from the other side;
    // written so that a NaN range (a default-constructed Tracks3D) is
    // rejected rather than passing every comparison
    if (!(tracks.fehMin() <= fehDist_.getMin() && fehDist_.getMax() <= tracks.fehMax())) // NOLINT(readability-simplify-boolean-expr) -- the De Morgan form would accept a NaN range, since every comparison with NaN is false
    {
        throw std::invalid_argument(
            "SimControls::setTracks: the new stellar tracks were loaded over "
            "[Fe/H] in [" + std::to_string(tracks.fehMin()) + ", " +
            std::to_string(tracks.fehMax()) + "], which does not cover the "
            "current [Fe/H] distribution, [" + std::to_string(fehDist_.getMin()) +
            ", " + std::to_string(fehDist_.getMax()) + "]. Load the tracks over "
            "a range covering it, or narrow the distribution with setFeH() first.");
    }
}

// Throw if the current IMF extends above tracks' maximum mass -- see
// this method's own header comment
void io::SimControls::checkTracksCoverIMF(const tracks::Tracks3D& tracks) const
{
    // A default-constructed Tracks3D (NaN [Fe/H] range) has no mass
    // range to check at all; checkTracksCoverFeH() rejects it
    if (!imf_.valid() || std::isnan(tracks.fehMin())) { return; }
    if (imf_.getMax() > tracks.mMax())
    {
        throw std::invalid_argument(
            "SimControls::setTracks: the new stellar tracks' maximum mass, " +
            std::to_string(tracks.mMax()) + ", is below the current IMF's maximum "
            "mass, " + std::to_string(imf_.getMax()) + ". Load tracks extending to "
            "at least that mass, or lower the IMF's maximum mass with setIMF() first.");
    }
}

void io::SimControls::setTracks(std::unique_ptr<tracks::Tracks3D> tracks)
{
    if (!tracks)
    {
        throw std::invalid_argument("SimControls::setTracks: tracks must not be null");
    }
    checkTracksCoverFeH(*tracks);
    if (imf_.valid()) { checkIMFWithinTracks("SimControls::setTracks", imf_, *tracks); }
    // Build the fixed-[Fe/H] slice, which can itself throw, before
    // changing any state, so that a failure leaves this SimControls
    // unchanged rather than holding the new tracks with a stale slice
    decltype(constFeHTracks_) newConstFeHTracks;
    if (constFeH())
    {
        newConstFeHTracks = std::make_shared<tracks::Tracks2D>(tracks->sliceConstFeH(fehDist_.getMin()));
    }
    tracks_ = std::move(tracks);
    if (newConstFeHTracks) { constFeHTracks_ = std::move(newConstFeHTracks); }
}

// Throw if imf extends above tracks' maximum mass, and warn if it
// extends below their minimum mass -- see this method's own header
// comment
void io::SimControls::checkIMFWithinTracks(const std::string& who, const pdfs::PDF& imf,
    const tracks::Tracks3D& tracks)
{
    if (imf.getMax() > tracks.mMax())
    {
        throw std::invalid_argument(
            who + ": IMF maximum mass, " + std::to_string(imf.getMax()) +
            ", exceeds the maximum mass in the selected stellar tracks, " +
            std::to_string(tracks.mMax()) + "; stars above the tracks' maximum "
            "mass are not supported. Lower the IMF's maximum mass, or use tracks "
            "extending to at least that mass.");
    }
    if (tracks.mMin() > imf.getMin() && utils::isIORank())
    {
        std::cout << "slug: warning: minimum mass in selected tracks is "
            << tracks.mMin() << " but IMF minimum mass is " << imf.getMin()
            << "; stars with masses from " << imf.getMin() << " to "
            << tracks.mMin() << " will be treated as having zero luminosity\n";
    }
}

// Set the IMF, checking it against the current tracks before changing
// any state, so that a rejected IMF leaves this SimControls unchanged
void io::SimControls::setIMF(const std::string& imf)
{
    pdfs::PDF newIMF = utils::initPDFFromString(imf, imfPrefix);
    if (tracks_) { checkIMFWithinTracks("SimControls::setIMF", newIMF, *tracks_); }
    imf_ = std::move(newIMF);
    updateFracStochMass();
}

// Set the minimum isotope lifetime, rebuilding yields_ to match -- see
// this method's own header comment
void io::SimControls::setMinIsotopeLifetime(const double value)
{
    if (std::isnan(value) || value < 0.0)
    {
        throw std::invalid_argument(
            "SimControls::setMinIsotopeLifetime: value must be non-negative, got " + std::to_string(value));
    }
    const double oldValue = minIsotopeLifetime_;
    minIsotopeLifetime_ = value;
    if (!yields_) { return; }
    try
    {
        yields_->rebuildYieldGrid(yields_->requestedIsotopes());
    }
    catch (...)
    {
        minIsotopeLifetime_ = oldValue;
        yields_->rebuildYieldGrid(yields_->requestedIsotopes());
        throw;
    }
}

// Set the clustered-star A_V distribution, rebuilding extinct_'s own
// cached quantities if an extinction curve is already present -- see
// setAVDist()'s own header comment for why this call is harmless
// rather than strictly necessary
void io::SimControls::setAVDist(const std::string& avDist)
{
    avDist_ = utils::initPDFFromString(avDist);
    if (extinct_) { extinct_->rebuildCache(); }
}

// Set the field-star A_V distribution, rebuilding extinct_'s own
// cached quantities if an extinction curve is already present -- see
// setAVDistField()'s own header comment for why this one matters
void io::SimControls::setAVDistField(const std::string& avDistField)
{
    auto newAVDistField = utils::initPDFFromString(avDistField);
    if (!extinct_)
    {
        avDistField_ = std::move(newAVDistField);
        return;
    }

    // extinct_->rebuildCache() reads avDistField_ live (via
    // controls_.avDistField()), so the new value must already be in
    // place before calling it -- but if that call throws (e.g. a
    // degenerate avDistField -- see Extinct::computeExtinctionFacCts()),
    // avDistField_ must be restored to its previous value before
    // rethrowing: extinct_'s own cache stays at its previous, valid
    // state either way (rebuildCache() is itself failure-atomic -- see
    // its own comment), so leaving avDistField_ at the new, rejected
    // value would otherwise make SimControls::avDistField() disagree
    // with what extinct_'s cache actually encodes.
    auto oldAVDistField = std::move(avDistField_);
    avDistField_ = std::move(newAVDistField);
    try
    {
        extinct_->rebuildCache();
    }
    catch (...)
    {
        avDistField_ = std::move(oldAVDistField);
        throw;
    }
}

// Set the star formation rate -- mirrors the galaxy.sfr handling in
// the constructor above exactly, including not resolving a file name
// through utils::getFilePath (unlike setIMF()/setCMF()/setFeH()/
// setCLF(), which all go through utils::initPDFFromString()). Clears
// sfrDist_ back to invalid, mirroring setSFRDist()'s own identical
// clearing of sfr_ -- see either one's own header comment for why:
// only one of sfr_/sfrDist_ is ever meant to be valid at a time.
void io::SimControls::setSFR(const std::string& sfr)
{
    try
    {
        const double sfrVal = utils::stod(sfr);
        sfr_ = buildConstantSFR(sfrVal);
    }
    catch (const std::invalid_argument&)
    {
        // We have been given a file
        sfr_ = pdfs::parsePDFDescriptor(sfr);
    }
    sfrDist_ = pdfs::PDF();
}

// Track reader
void io::SimControls::readTracks(const utils::TrackedDeck& inputDeck)
{
    // Get required tracks key
    auto trackName = inputDeck.value<std::string>("stars.tracks", true);

    // Check for optional parameters
    auto registryName = inputDeck.value<std::string>("stars.track_registry");
    auto vvcrit = inputDeck.value<double>("stars.v_vcrit");
    auto afe = inputDeck.value<double>("stars.alphaFe");

    // Construct tracks from input data
    tracks_ = std::make_shared<tracks::Tracks3D>(
        trackName.value(), // NOLINT(bugprone-unchecked-optional-access) -- we verified this was valid a few lines ago
        fehDist_.getMin(),
        fehDist_.getMax(),
        vvcrit.value_or(tracks::defaultVVcrit),
        afe.value_or(tracks::defaultAFe),
        registryName.value_or(tracks::defaultRegistry));
}

// Spectral synthesizer reader
void io::SimControls::readSpectra(const utils::TrackedDeck& inputDeck)
{
    // spectra.model is optional -- if it is absent, this simulation
    // computes no spectra, and specsyn_ stays null. Nothing else in
    // [spectra] is read before this check, so that all of it (including
    // spectra.registry) is reported as ignored rather than as used.
    const auto modelNode = inputDeck.atPath("spectra.model");
    if (!modelNode)
    {
        constexpr auto reason = "spectra.model was not given, so no spectra are computed";
        inputDeck.markIgnored("spectra", reason);
        inputDeck.markIgnored("stars.CFe", reason); // only used by spectral synthesis
        return;
    }

    // Check for an optional alternative registry
    auto registryNameInput = inputDeck.value<std::string>("spectra.registry");
    const std::string registryName = registryNameInput.value_or(specsyn::defaultRegistry);

    // Load every library-based model over the [Fe/H] range the tracks
    // were requested for (tracks_->fehMin()/fehMax(), equal to
    // fehDist_'s own range at construction), not the tracks' own
    // padded grid (tracks_->feH(), which extends a few grid points
    // further on each side purely for Mesh3DInterpolator's own
    // benefit -- see Tracks3D::Tracks3D()'s own comment). Spectra are
    // only ever requested at [Fe/H] drawn from, or integrated over,
    // fehDist_, which setFeH() can never widen past the tracks'
    // requested range, so this covers every [Fe/H] spectral synthesis
    // can be asked for. Loading the padding as well would cost extra
    // library planes that are never used, could make a library that
    // covers fehDist_ but not the padding substitute a different
    // [alpha/Fe] grid to cover it (see SpecsynLibNoWind's own
    // constructor), and would make SpecsynLibChained warn about
    // clamping that can never happen.
    const double fehMin = tracks_->fehMin();
    const double fehMax = tracks_->fehMax();

    // Optional user-requested output wavelength grid: spectra.wl_min
    // and spectra.wl_max (in Angstrom), and spectra.nwl (the number
    // of output wavelengths). All three are individually optional,
    // but wl_min and wl_max only make sense together with an nwl to
    // say how finely to sample between them, so if either endpoint is
    // given, all three must be; nwl alone (a request to resample onto
    // the default wavelength range at a different resolution) is fine
    // on its own. Any not supplied falls back to specsyn::defaultWlMin/
    // defaultWlMax/defaultNWl -- a fixed grid from ~91 Angstrom (10
    // Rydberg, deep enough into the EUV to capture ionizing flux) to
    // 1e5 Angstrom (10 micron) at 2048 points -- rather than each
    // library's own native grid, so that every spectral synthesizer
    // built without explicit wavelength settings covers the same
    // physically-motivated range by default.
    const auto wlMinInput = inputDeck.value<double>("spectra.wl_min");
    const auto wlMaxInput = inputDeck.value<double>("spectra.wl_max");
    const auto nWlInput = inputDeck.value<unsigned long>("spectra.nwl");
    if ((wlMinInput.has_value() || wlMaxInput.has_value()) &&
        !(wlMinInput.has_value() && wlMaxInput.has_value() && nWlInput.has_value()))
    {
        throw std::runtime_error(
            "SimControls: spectra.wl_min and spectra.wl_max must be "
            "given together with each other and with spectra.nwl");
    }
    wlMin_ = wlMinInput.value_or(specsyn::defaultWlMin);
    wlMax_ = wlMaxInput.value_or(specsyn::defaultWlMax);
    nWl_ = nWlInput.value_or(specsyn::defaultNWl);

    // Optional redshift, read live by every Specsyn's/Extinct's own
    // wlObs() (see z()); 0 (no redshift) if not supplied
    const auto zInput = inputDeck.value<double>("spectra.z");
    z_ = zInput.value_or(0.0);

    // Optional stars.alphaFe and stars.CFe: if not supplied, fall back
    // to the library defaults. stars.alphaFe is the same key that
    // readTracks() also reads (tracks and spectra share the same AFe
    // value), while stars.CFe is spectra-only (tracks have no CFe axis).
    const auto afeInput = inputDeck.value<double>("stars.alphaFe");
    const double afe = afeInput.value_or(tracks::defaultAFe);
    const auto cfeInput = inputDeck.value<double>("stars.CFe");
    const double cfe = cfeInput.value_or(specsyn::defaultCFe);

    // A single string names one model directly, unless it's one of two
    // special values: "blackbody" (a lightweight synthesizer needing
    // no library at all) or "default" (expanding to
    // specsyn::defaultModelList). Anything else must be an array of
    // strings. "default" and an explicit array both end up chained
    // together via SpecsynLibChained, using the shared construction
    // call at the end of this function. Every spectral synthesizer
    // built below stores a live reference back to *this (for its own
    // integrator tolerances) -- see Specsyn's own controls_ member --
    // so this SimControls must outlive it, which holds as long as
    // readSpectra() is only ever called from this SimControls's own
    // constructor.
    std::vector<std::string> models;
    if (const auto model = modelNode.value<std::string>(); model.has_value())
    {
        if (model.value() == "blackbody")
        {
            specsyn_ = std::make_shared<specsyn::SpecsynBlackbody>(wlMin_, wlMax_, nWl_, *this);
            return;
        }

        if (model.value() == "default")
        {
            models = specsyn::defaultModelList;
        }
        else
        {
            // Look the model up in the registry, and use its WR_grid
            // entry (if any) to decide which SpecsynLib specialization
            // applies: Wolf-Rayet libraries -- parameterized by
            // transformed radius and stellar temperature rather than
            // logg and Teff -- need SpecsynLibWR, every other library
            // needs SpecsynLibNoWind
            auto [registry, registryPath] = specsyn::parseRegistry(registryName);
            const auto modelEntry = registry.at_path(model.value());
            if (!modelEntry)
            {
                throw std::runtime_error(
                    "SimControls: spectra.model '" + model.value() +
                    "' not found in spectra registry " + registryPath.string());
            }
            const bool wrGrid = modelEntry.at_path("WR_grid").value<bool>().value_or(false);

            if (wrGrid)
            {
                specsyn_ = std::make_shared<specsyn::SpecsynLibWR<specsyn::OOBPolicy::raise>>(
                    model.value(), fehMin, fehMax, registryName,
                    wlMin_, wlMax_, nWl_, *this);
            }
            else
            {
                specsyn_ = std::make_shared<specsyn::SpecsynLibNoWind<specsyn::OOBPolicy::raise>>(
                    model.value(), fehMin, fehMax,
                    afe, cfe,
                    std::numeric_limits<double>::quiet_NaN(), specsyn::defaultR,
                    registryName, wlMin_, wlMax_, nWl_, *this);
            }
            return;
        }
    }
    else
    {
        const toml::array* modelArr = modelNode.as_array();
        if (modelArr == nullptr)
        {
            throw std::runtime_error(
                "SimControls: spectra.model must be a string or an array of strings");
        }
        modelArr->for_each([&models](auto&& el) -> void {
            if constexpr (toml::is_string<decltype(el)>) { models.push_back(std::string(el)); }
        });
        if (models.empty())
        {
            throw std::runtime_error(
                "SimControls: spectra.model array must contain at least one string entry");
        }
    }

    specsyn_ = std::make_shared<specsyn::SpecsynLibChained>(
        models, fehMin, fehMax,
        afe, cfe, std::vector<double>{},
        specsyn::defaultR, registryName, wlMin_, wlMax_, nWl_, true, *this);
}

// Photometric filter collection reader
void io::SimControls::readFilters(const utils::TrackedDeck& inputDeck)
{
    // phot.system: optional; if present, must name one of the defined
    // photometric systems. Flambda (a raw flux, needing no Vega
    // spectrum or other extra data) is the default if the key is
    // absent entirely.
    phot::PhotSystem photSystem = phot::PhotSystem::Flambda;
    const auto systemNode = inputDeck.atPath("phot.system");
    if (systemNode)
    {
        const auto systemStr = systemNode.value<std::string>();
        if (!systemStr.has_value())
        {
            throw std::runtime_error("SimControls: phot.system must be a string");
        }
        if (systemStr.value() == "Flambda") { photSystem = phot::PhotSystem::Flambda; }
        else if (systemStr.value() == "Fnu") { photSystem = phot::PhotSystem::Fnu; }
        else if (systemStr.value() == "ST") { photSystem = phot::PhotSystem::ST; }
        else if (systemStr.value() == "AB") { photSystem = phot::PhotSystem::AB; }
        else if (systemStr.value() == "Vega") { photSystem = phot::PhotSystem::Vega; }
        else
        {
            throw std::runtime_error(
                "SimControls: phot.system '" + systemStr.value() +
                "' is not a recognized photometric system "
                "(expected Flambda, Fnu, ST, AB, or Vega)");
        }
    }

    // phot.registry: optional string override of the default filter registry
    const auto registryInput = inputDeck.value<std::string>("phot.registry");
    const std::string registryName = registryInput.value_or(phot::defaultRegistry);

    // phot.vega: optional string override of the default Vega
    // reference spectrum file. The global Vega spectrum (see
    // phot::vegaSpectrum()) is a lazily-constructed, program-wide
    // singleton -- only the very first call to vegaSpectrum() anywhere
    // determines which file actually gets loaded, so if the deck names
    // a non-default file, force that first call to happen here, now,
    // before any filter's own lazy Filter::fluxVega() can beat it to
    // the punch with the default file instead. The returned spectrum
    // itself is not needed here, only the side effect of loading it.
    const auto vegaInput = inputDeck.value<std::string>("phot.vega");
    if (vegaInput.has_value()) { phot::vegaSpectrum(vegaInput.value()); }

    // phot.filters: optional; a single string names one filter
    // directly, interpreted as an array of length 1; anything else
    // must be an array of strings
    std::vector<std::string> filterNames;
    const auto filtersNode = inputDeck.atPath("phot.filters");
    if (filtersNode)
    {
        if (const auto single = filtersNode.value<std::string>(); single.has_value())
        {
            filterNames.push_back(single.value());
        }
        else
        {
            const toml::array* filtersArr = filtersNode.as_array();
            if (filtersArr == nullptr)
            {
                throw std::runtime_error(
                    "SimControls: phot.filters must be a string or an array of strings");
            }
            filterNames = utils::stringArrayContents(filtersArr);
        }
    }

    // "Lbol" (bolometric luminosity) is photometry-like, but is
    // computed outside the filter machinery entirely (see
    // Cluster::computeLbol()); pull it out of the filter list here
    // and just record that it was requested
    const auto lbolIt = std::ranges::find(filterNames, "Lbol");
    if (lbolIt != filterNames.end())
    {
        filterNames.erase(lbolIt);
        computeLbol_ = true;
    }

    // Nothing further to do if no actual filters were requested
    if (filterNames.empty()) { return; }

    // Photometry requires a spectral synthesizer to generate the
    // spectra filters are convolved against
    if (specsyn_ == nullptr)
    {
        throw std::runtime_error(
            "SimControls: phot.filters was given but no spectral "
            "synthesizer was requested (spectra.model was not set "
            "in the input deck)");
    }

    filters_ = std::make_shared<phot::FilterCollection>(
        filterNames, photSystem, registryName);
}

// Build a valid PDF that always draws exactly x -- used by readExtinct()
// below to fill in whichever of avDist_/avDistField_ was not given an
// explicit distribution of its own (x = 0), so the two are always either
// both valid or both invalid, never just one, and avNebFac_'s default
// (x = 1)
static auto buildDelta(const double x) -> pdfs::PDF
{
    auto delta = std::make_unique<pdfs::PDFSegmentDelta>(x);
    return pdfs::PDF(std::move(delta));
}

// Throw unless avNebFac (a candidate for SimControls::avNebFac_) is
// non-negative everywhere -- see setAVNebFac()'s own header comment
static void checkAVNebFac(const pdfs::PDF& avNebFac, const std::string& source)
{
    if (avNebFac.getMin() < 0.0)
    {
        throw std::runtime_error("SimControls: " + source + " must not extend below 0 "
            "(it is the ratio of nebular to stellar V-band extinction), but its minimum is " +
            std::to_string(avNebFac.getMin()));
    }
}

// Set the nebular-to-stellar A_V ratio distribution, validating it
// before replacing the old one -- see setAVNebFac()'s own header comment
void io::SimControls::setAVNebFac(const std::string& avNebFac)
{
    auto newAVNebFac = utils::initPDFFromString(avNebFac);
    checkAVNebFac(newAVNebFac, "the nebular extinction factor");
    avNebFac_ = std::move(newAVNebFac);
}

// Extinction curve reader
void io::SimControls::readExtinct(const utils::TrackedDeck& inputDeck)
{
    // Nebular-to-stellar extinction ratio: equal unless
    // extinct.neb_factor (read below) says otherwise -- set before the
    // early return below, so avNebFac_ is valid even when no
    // extinction is applied at all
    avNebFac_ = buildDelta(1.0);

    // extinct.AV / extinct.AV_field: both optional; if neither is
    // given, this simulation applies no extinction at all, and
    // avDist_/avDistField_/extinct_ are left at their default/null
    // state. If either is given, extinct.model becomes mandatory
    // (below), and whichever of the two was not given is set to a
    // delta function PDF at 0 instead of being left invalid -- see
    // buildDelta()'s own comment for why.
    const auto avNode = inputDeck.atPath("extinct.AV");
    const auto avFieldNode = inputDeck.atPath("extinct.AV_field");
    if (!avNode && !avFieldNode)
    {
        inputDeck.markIgnored("extinct",
            "neither extinct.AV nor extinct.AV_field was given, so no extinction is applied");
        return;
    }

    avDist_ = avNode ? inputDeck.initPDF("extinct.AV") : buildDelta(0.0);
    avDistField_ = avFieldNode ? inputDeck.initPDF("extinct.AV_field") : buildDelta(0.0);

    // extinct.neb_factor: optional ratio of nebular to stellar V-band
    // extinction -- see avNebFac()'s own comment
    if (inputDeck.atPath("extinct.neb_factor"))
    {
        auto avNebFac = inputDeck.initPDF("extinct.neb_factor");
        checkAVNebFac(avNebFac, "extinct.neb_factor");
        avNebFac_ = std::move(avNebFac);
    }

    // extinct.model: required now that extinct.AV or extinct.AV_field
    // was given, names the extinction curve to use
    const auto model = inputDeck.value<std::string>("extinct.model", true);

    // extinct.registry: optional override of the default extinction
    // curve registry
    const auto registryInput = inputDeck.value<std::string>("extinct.registry");
    const std::string registryName = registryInput.value_or(extinct::defaultRegistry);

    // Extinction requires a spectral synthesizer to provide the
    // wavelength grid the curve is interpolated onto
    if (specsyn_ == nullptr)
    {
        throw std::runtime_error(
            "SimControls: extinct.AV or extinct.AV_field was given but "
            "no spectral synthesizer was requested (spectra.model was "
            "not set in the input deck)");
    }

    extinct_ = std::make_shared<extinct::Extinct>(
        model.value(), *this, registryName); // NOLINT(bugprone-unchecked-optional-access) -- required=true above guarantees model has a value or getTOMLKeyWithError already threw
}

// True if tableSymbol (elem::ionizationData's own two-character atomic
// symbol, first character always uppercase, second either lowercase or
// '\0' for a single-letter symbol -- see IonizationData.hpp's own
// table) matches input case-insensitively, e.g. tableSymbol {'N','a'}
// matches input "Na", "na", or "NA", but not "N" or "NAA"
static auto symbolMatches(const std::array<char, 2>& tableSymbol, const std::string& input) -> bool
{
    if (input.empty() || input.size() > 2) { return false; }
    if (std::toupper(static_cast<unsigned char>(input.front())) != tableSymbol[0]) { return false; }
    if (input.size() == 1) { return tableSymbol[1] == '\0'; }
    return tableSymbol[1] != '\0' &&
        std::tolower(static_cast<unsigned char>(input[1])) == tableSymbol[1];
}

// Parse one yields.isotopes entry (e.g. "H1", "Na22", "fe56") into a
// reference to its IsotopeData record in the global elem::isotopeTable()
// -- see readYields()'s own comment for the exact syntax accepted
static auto parseIsotopeEntry(const std::string& entry) -> std::reference_wrapper<const elem::IsotopeData>
{
    std::size_t split = 0;
    while (split < entry.size() && (std::isalpha(static_cast<unsigned char>(entry[split])) != 0)) { ++split; }
    const std::string symbolPart = entry.substr(0, split);
    const std::string massPart = entry.substr(split);

    const bool massIsNumeric = !massPart.empty() && std::ranges::all_of(
        massPart, [](unsigned char c) { return std::isdigit(c) != 0; });
    if (symbolPart.empty() || symbolPart.size() > 2 || !massIsNumeric)
    {
        throw std::runtime_error(
            "SimControls: yields.isotopes entry '" + entry + "' is not a valid "
            "isotope specifier (expected an element symbol followed by a mass "
            "number, e.g. 'H1', 'Na22', or 'fe56')");
    }

    unsigned int z = 0;
    for (std::size_t i = 0; i < static_cast<std::size_t>(elem::Symbols::nElem); ++i)
    {
        if (symbolMatches(elem::ionizationData.at(i).symbol(), symbolPart))
        {
            z = elem::ionizationData.at(i).Z();
            break;
        }
    }
    if (z == 0)
    {
        throw std::runtime_error(
            "SimControls: yields.isotopes entry '" + entry + "' names an "
            "unrecognized element symbol '" + symbolPart + "'");
    }
    try
    {
        const auto a = static_cast<unsigned int>(std::stoul(massPart));
        return elem::isotopeTable(z, a);
    }
    catch (const std::out_of_range&)
    {
        throw std::runtime_error(
            "SimControls: yields.isotopes entry '" + entry + "' does not "
            "correspond to a known isotope in the isotope table");
    }
}

// Nucleosynthetic yield channel reader
void io::SimControls::readYields(const utils::TrackedDeck& inputDeck)
{
    yieldChannels_.clear();

    // yields.channel1, yields.channel2, etc., stopping at the first N
    // for which yields.channelN is absent
    for (std::size_t i = 1; ; ++i)
    {
        const std::string tableKey = "yields.channel" + std::to_string(i);
        if (!inputDeck.atPath(tableKey)) { break; }

        // yields.channelN.channel: required, must match one of
        // yields::channelStr's own entries
        const auto channelInput = inputDeck.value<std::string>(tableKey + ".channel", true);
        const auto* const channelIt = std::ranges::find(
            yields::channelStr, channelInput.value()); // NOLINT(bugprone-unchecked-optional-access) -- required=true above guarantees channelInput has a value or getTOMLKeyWithError already threw
        if (channelIt == yields::channelStr.end())
        {
            throw std::runtime_error(
                "SimControls: " + tableKey + ".channel = '" + channelInput.value() + // NOLINT(bugprone-unchecked-optional-access) -- required=true above guarantees channelInput has a value or getTOMLKeyWithError already threw
                "' is not a recognized yield channel");
        }
        const auto channel = static_cast<yields::Channel>(
            std::distance(yields::channelStr.begin(), channelIt));

        // yields.channelN.model: required, but not itself validated
        // here -- see YieldChannel::YieldChannel()'s own comment for
        // where that happens
        const auto modelInput = inputDeck.value<std::string>(tableKey + ".model", true);

        // yields.channelN.m_min/m_max: both independently optional
        const auto mMinInput = inputDeck.value<double>(tableKey + ".m_min");
        const auto mMaxInput = inputDeck.value<double>(tableKey + ".m_max");

        yieldChannels_.push_back(yields::YieldChannelDescriptor{
            channel, modelInput.value(), mMinInput, mMaxInput}); // NOLINT(bugprone-unchecked-optional-access) -- required=true above guarantees modelInput has a value or getTOMLKeyWithError already threw
    }

    // yields.channel_decomposed: optional, read regardless of whether
    // yieldChannels_ ends up empty (harmless either way)
    yieldsChannelDecomposed_ = inputDeck.value<bool>("yields.channel_decomposed").value_or(true);

    // yields.no_decay: optional, read regardless of whether
    // yieldChannels_ ends up empty (harmless either way), like
    // yields.channel_decomposed above
    noDecay_ = inputDeck.value<bool>("yields.no_decay").value_or(false);

    // yields.min_isotope_lifetime: optional, likewise read regardless;
    // must be read before yields_ is built below, since
    // Yields::rebuildYieldGrid() uses it
    minIsotopeLifetime_ = inputDeck.value<double>("yields.min_isotope_lifetime")
        .value_or(yields::defaultMinIsotopeLifetime);
    if (std::isnan(minIsotopeLifetime_) || minIsotopeLifetime_ < 0.0)
    {
        throw std::runtime_error("SimControls: yields.min_isotope_lifetime must be non-negative, got " +
            std::to_string(minIsotopeLifetime_));
    }

    if (yieldChannels_.empty())
    {
        // Only the keys read after this point; a stray yields.channelN
        // table is deliberately not covered, so that one is reported
        constexpr auto reason = "no yields.channel1 was given, so no yields are computed";
        inputDeck.markIgnored("yields.registry", reason);
        inputDeck.markIgnored("yields.isotopes", reason);
        return;
    }

    // Sanity check: if yield channels were requested but the computed
    // yields would never be written anywhere, that's almost certainly
    // a mistake -- mirrors OutputManager's own check for phot.filters
    // vs. writeClusterPhot()/writeGalaxyPhot(). writeGalaxyYields_ is
    // meaningless for a cluster-type simulation (there is no Galaxy,
    // and so no galaxy_yields group/file, in that case), so it cannot
    // rescue the yields from going unwritten there the way it could in
    // a galaxy-type simulation -- writeClusterYields_ being false is
    // fatal on its own for a cluster-type simulation.
    if (!writeClusterYields_ &&
        (!writeGalaxyYields_ || simType_ == SimType::cluster))
    {
        throw std::runtime_error(
            "SimControls: yield channels were given, but "
            "output.write_cluster_yields and output.write_galaxy_yields "
            "are both false (or this is a cluster-type simulation and "
            "output.write_cluster_yields is false), so the computed "
            "yields would never be written");
    }

    // yields.registry: optional override of the default yield registry
    const auto registryInput = inputDeck.value<std::string>("yields.registry");
    const std::string registryName = registryInput.value_or(yields::defaultRegistry);

    yields_ = std::make_shared<yields::Yields>(*this, registryName);

    // yields.isotopes: optional, narrows the isotopes yields_ ends up
    // tabulating to this list plus every isotope on a decay chain
    // between it and what the loaded channels tabulate (e.g. Fe56 also
    // brings in Ni56 and Co56) -- see Yields::rebuildYieldGrid()'s own
    // comment for the exact rules, and parseIsotopeEntry()'s own
    // comment for the exact "<symbol><mass number>" syntax each entry must
    // follow (e.g. "H1", "Na22", "fe56"; case-insensitive on the
    // symbol). Must be an array of strings; absent entirely leaves
    // yields_->isotopes() at the union rebuildYieldGrid() already
    // built above, unrestricted.
    const auto isotopesNode = inputDeck.atPath("yields.isotopes");
    if (isotopesNode)
    {
        const toml::array* const isotopesArr = isotopesNode.as_array();
        if (isotopesArr == nullptr)
        {
            throw std::runtime_error("SimControls: yields.isotopes must be an array of strings");
        }
        const auto isotopeNames = utils::stringArrayContents(isotopesArr);
        elem::IsotopeList isotopes;
        isotopes.reserve(isotopeNames.size());
        for (const auto& name : isotopeNames)
        {
            isotopes.push_back(parseIsotopeEntry(name));
        }
        yields_->rebuildYieldGrid(isotopes);
    }
}

// Stellar feedback controls reader
void io::SimControls::readFeedback(const utils::TrackedDeck& inputDeck)
{
    // feedback.sn_mass_range: optional; if given, must be an array of
    // numbers, whose validity as mass limits setSNMassLimits() checks
    const auto node = inputDeck.atPath("feedback.sn_mass_range");
    if (node)
    {
        const toml::array* arr = node.as_array();
        if (arr == nullptr)
        {
            throw std::runtime_error(
                "SimControls: feedback.sn_mass_range must be an array of numbers");
        }
        std::vector<double> limits;
        limits.reserve(arr->size());
        for (const auto& elem : *arr)
        {
            const auto val = elem.value<double>();
            if (!val.has_value())
            {
                throw std::runtime_error(
                    "SimControls: feedback.sn_mass_range must be an array of numbers");
            }
            limits.push_back(val.value());
        }
        setSNMassLimits(std::move(limits));
    }

    // feedback.wr_winds: optional; if given, must name one of
    // feedback::wrWindModelStr's own entries, otherwise wrWindModel_
    // keeps its default
    const auto wrWindsInput = inputDeck.value<std::string>("feedback.wr_winds");
    if (wrWindsInput.has_value())
    {
        wrWindModel_ = feedback::wrWindModelFromString(wrWindsInput.value());
    }

    // feedback.ob_winds: likewise, but naming one of
    // feedback::obWindModelStr's own entries
    const auto obWindsInput = inputDeck.value<std::string>("feedback.ob_winds");
    if (obWindsInput.has_value())
    {
        obWindModel_ = feedback::obWindModelFromString(obWindsInput.value());
    }

    // feedback.agb_winds and feedback.other_winds: likewise
    const auto agbWindsInput = inputDeck.value<std::string>("feedback.agb_winds");
    if (agbWindsInput.has_value())
    {
        agbWindModel_ = feedback::agbWindModelFromString(agbWindsInput.value());
    }
    const auto otherWindsInput = inputDeck.value<std::string>("feedback.other_winds");
    if (otherWindsInput.has_value())
    {
        otherWindModel_ = feedback::otherWindModelFromString(otherWindsInput.value());
    }

    // The wind calculator itself: always built, whatever was given
    // above, since it reads each wind model live on every call rather
    // than being configured once here
    winds_ = std::make_shared<feedback::Winds>(*this);
}

void io::SimControls::setSNMassLimits(std::vector<double> limits)
{
    if (limits.size() % 2 != 0)
    {
        throw std::invalid_argument(
            "setSNMassLimits: mass limits must have an even number of "
            "elements, one (lower, upper) pair per mass interval");
    }
    if (std::ranges::any_of(limits, [](const double m) -> bool { return !std::isfinite(m) || m <= 0.0; }))
    {
        throw std::invalid_argument(
            "setSNMassLimits: mass limits must be finite and strictly positive");
    }
    // Written as !(a < b) rather than a >= b so that a NaN, for which
    // every comparison is false, also counts as a violation
    if (std::ranges::adjacent_find(limits,
            [](const double a, const double b) -> bool { return !(a < b); }) != limits.end())
    {
        throw std::invalid_argument(
            "setSNMassLimits: mass limits must be strictly increasing");
    }
    snMassLimits_ = std::move(limits);
}

auto io::SimControls::hasSN(const double mass) const -> bool
{
    // Explicit mass limits, if any, take precedence over yields
    if (!snMassLimits_.empty())
    {
        for (std::size_t i = 0; i + 1 < snMassLimits_.size(); i += 2)
        {
            if (mass >= snMassLimits_[i] && mass <= snMassLimits_[i + 1]) // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- i + 1 < size() by the loop condition
            {
                return true;
            }
        }
        return false;
    }

    // Otherwise defer to the core-collapse supernova yield channels,
    // if any
    if (yields_ != nullptr)
    {
        return yields_->hasYield(mass, yields::Channel::ccsn_);
    }
    return false;
}

auto io::SimControls::hasSN(const double mass, const double feH) const -> bool
{
    // Explicit mass limits, if any, take precedence over yields, and
    // with no yields there is nothing [Fe/H]-dependent to check -- in
    // either case hasSN(mass) gives the answer
    if (!snMassLimits_.empty() || yields_ == nullptr) { return hasSN(mass); }
    return yields_->hasYield(mass, feH, yields::Channel::ccsn_);
}

// Nebular emission controls and grid reader
void io::SimControls::readNebular(const utils::TrackedDeck& inputDeck)
{
    // nebular.compute_neb: whether nebular emission is computed at
    // all. Nebular emission is added to a synthesized spectrum, and the
    // Nebular grid is resampled onto the spectral synthesizer's own
    // wavelength grid, so it requires specsyn_ (like phot.filters and
    // extinct.AV -- see readFilters()/readExtinct()). The default
    // therefore depends on whether one is available: true if specsyn_
    // is set (nebular::defaultComputeNeb), false otherwise. Explicitly
    // asking for nebular emission with no spectral synthesizer is an
    // error, rather than being silently ignored.
    const auto computeNeb = inputDeck.value<bool>("nebular.compute_neb");
    if (specsyn_ == nullptr)
    {
        if (computeNeb.value_or(false))
        {
            throw std::runtime_error(
                "SimControls: nebular.compute_neb = true was given but no "
                "spectral synthesizer was requested (spectra.model was not "
                "set in the input deck)");
        }
        nebControls_.computeNeb_ = false;
        inputDeck.markIgnored("nebular",
            "no spectral synthesizer was requested (spectra.model was not given), so nebular emission is not computed");
        return;
    }
    nebControls_.computeNeb_ = computeNeb.value_or(nebular::defaultComputeNeb);

    // If nebular.compute_neb is false, every other nebular.* key is
    // skipped (their defaults are irrelevant, since nebular_ is left
    // null either way) -- mirrors readExtinct()'s own early return when
    // neither extinct.AV nor extinct.AV_field was given.
    if (!nebControls_.computeNeb_)
    {
        inputDeck.markIgnored("nebular", "nebular.compute_neb is false");
        return;
    }

    // Every remaining nebular.* control parameter is independently
    // optional -- unlike readSpectra()'s/readExtinct()'s own
    // all-or-nothing stanzas, there is no single key whose absence
    // should skip the rest of this stanza. Each one left absent
    // simply leaves nebControls_'s own field at whatever
    // NebularControls's own default member initializer already set
    // it to.
    const auto logU = inputDeck.value<double>("nebular.log_U");
    if (logU) { nebControls_.logU_ = logU.value(); }

    const auto covFac = inputDeck.value<double>("nebular.cov_fac");
    if (covFac) { nebControls_.covFac_ = covFac.value(); }

    const auto lineWidth = inputDeck.value<double>("nebular.line_width");
    if (lineWidth) { nebControls_.lineWidth_ = lineWidth.value(); }

    // nebular.table: optional override of the default nebular
    // emission table to load this track set's own grid from
    const auto tableInput = inputDeck.value<std::string>("nebular.table");
    const std::string tableName = tableInput.value_or(nebular::defaultTable);

    // stars.tracks/stars.v_vcrit: the same keys, read the same way, as
    // readTracks() above -- Tracks3D itself does not retain its own
    // track name after construction, so there is no way to recover it
    // from tracks_ here; stars.tracks is already mandatory (readTracks()
    // would have thrown before this method ever ran if it were absent),
    // so re-reading it here as required is always safe.
    const auto trackName = inputDeck.value<std::string>("stars.tracks", true);
    const auto vvcrit = inputDeck.value<double>("stars.v_vcrit");

    nebular_ = std::make_shared<nebular::Nebular>(
        tableName, trackName.value(), *this, vvcrit.value_or(tracks::defaultVVcrit)); // NOLINT(bugprone-unchecked-optional-access) -- required=true above guarantees trackName has a value or getTOMLKeyWithError already threw
}
