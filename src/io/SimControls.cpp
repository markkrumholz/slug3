/**
 * @file SimControls.cpp
 * @author Mark Krumholz
 * @date 2026-07-16
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 * @brief Implementation of SimControls
 */

#include "SimControls.hpp"
#include "../extinct/Extinct.hpp"
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
#include "../utils/ParseUtils.hpp"
#include "../utils/RngThread.hpp"
#include "../utils/TOMLUtils.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
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
static auto readOutputTimesArray(const toml::table& inputDeck) -> std::vector<double>
{
    const toml::array* arr = inputDeck.at_path("output.output_times").as_array();
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
static auto generateOutputTimesRange(const toml::table& inputDeck) -> std::vector<double>
{
    const auto startTimeInput = utils::getTOMLKeyWithError<double>(
        inputDeck, "output.start_time", true);
    if (!startTimeInput.has_value())
    {
        throw std::runtime_error("SimControls: output.start_time not found");
    }
    const double startTime = startTimeInput.value();

    const auto endTimeInput = utils::getTOMLKeyWithError<double>(
        inputDeck, "output.end_time", true);
    if (!endTimeInput.has_value())
    {
        throw std::runtime_error("SimControls: output.end_time not found");
    }
    const double endTime = endTimeInput.value();

    const auto nTimeInput = utils::getTOMLKeyWithError<unsigned long>(
        inputDeck, "output.ntime", true);
    if (!nTimeInput.has_value())
    {
        throw std::runtime_error("SimControls: output.ntime not found");
    }
    const unsigned long nTime = nTimeInput.value();

    const bool logTime = utils::getTOMLKeyWithError<bool>(
        inputDeck, "output.log_time").value_or(false);

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
    initControlFlow(inputDeck);
    initPhysics(inputDeck);
}

void io::SimControls::initControlFlow(const toml::table& inputDeck)
{
    // Determine simulation type
    const auto simTypeInput = utils::getTOMLKeyWithError<std::string>(
        inputDeck, "sim_type", true);
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
        utils::getTOMLKeyWithError<unsigned int>(inputDeck, "verbosity");
    if (verbosityInput.has_value()) { verbosity_ = verbosityInput.value(); }

    // Read output mode
    const auto outputMode = utils::getTOMLKeyWithError<std::string>(
        inputDeck, "outputs.output_mode");
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
        utils::getTOMLKeyWithError<std::string>(inputDeck, "output.model_name");
    if (modelNameInput.has_value()) { modelName_ = modelNameInput.value(); }

    // Read output directory
    const auto outDirInput =
        utils::getTOMLKeyWithError<std::string>(inputDeck, "outputs.out_dir");
    if (outDirInput.has_value()) { outDir_ = outDirInput.value(); }

    // Read number of trials
    const auto nTrialInput =
        utils::getTOMLKeyWithError<unsigned long>(inputDeck, "n_trial");
    if (nTrialInput.has_value())
    {
        nTrial_ = nTrialInput.value();
    }

    // Handle output time generation
    setOutputTimes(inputDeck);

    // If we have been given a specific RNG seed, use it
    const auto rngSeed =
        utils::getTOMLKeyWithError<unsigned long>(inputDeck, "rng_seed");
    if (rngSeed.has_value()) { utils::rng().seed(rngSeed.value()); }

    // Read optional integrator tolerance controls; defaults match
    // PDFIntegrator's own defaults so omitting them is a no-op
    const auto relTolInput = utils::getTOMLKeyWithError<double>(inputDeck, "integrator.rel_tol");
    if (relTolInput.has_value()) { intRelTol_ = relTolInput.value(); }
    const auto absTolInput = utils::getTOMLKeyWithError<double>(inputDeck, "integrator.abs_tol");
    if (absTolInput.has_value()) { intAbsTol_ = absTolInput.value(); }
    const auto maxIterInput = utils::getTOMLKeyWithError<std::size_t>(inputDeck, "integrator.max_iter");
    if (maxIterInput.has_value()) { intMaxIter_ = maxIterInput.value(); }
}

void io::SimControls::initPhysics(const toml::table& inputDeck)
{
    // Read IMF, CMF, and FeH
    imf_ = utils::initPDFFromKey(inputDeck, "stars.IMF", imfPrefix);
    cmf_ = utils::initPDFFromKey(inputDeck, "clusters.CMF");
    fehDist_ = utils::initPDFFromKey(inputDeck, "stars.FeH");

    // Read the tracks. Needs fehDist_ (just set above) to pick the
    // [Fe/H] range to load; done before readSpectra() (which used to
    // come first here) because readSpectra() itself now needs
    // tracks_.feH() -- see its own comment.
    readTracks(inputDeck);

    // Warn if the tracks don't extend down to the IMF's minimum mass:
    // stars in that gap have no track data, so they end up being
    // treated as having zero luminosity when spectra are computed
    // (see Cluster::computeSpec)
    if (tracks_.mMin() > imf_.getMin())
    {
        std::cout << "slug: warning: minimum mass in selected tracks is "
            << tracks_.mMin() << " but IMF minimum mass is " << imf_.getMin()
            << "; stars with masses from " << imf_.getMin() << " to "
            << tracks_.mMin() << " will be treated as having zero luminosity\n";
    }

    // Read the spectral synthesis model to use, if any -- spectra.model
    // is optional, since not every simulation needs spectra computed.
    // Needs tracks_ (just set above) to pick the [Fe/H] range a
    // library-based model is loaded over -- see readSpectra()'s own
    // comment for why that's tracks_.feH()'s own range, not fehDist_'s.
    readSpectra(inputDeck);

    // Read the photometric filter collection to use, if any --
    // phot.filters is optional. Needs specsyn_ (just set above) to
    // check that a spectral synthesizer is actually available before
    // building a filter collection that will need one.
    readFilters(inputDeck);

    // Read the extinction curve to apply, if any -- extinct.AV is
    // optional. Needs specsyn_ (just set above) to provide the
    // wavelength grid the extinction curve is interpolated onto.
    readExtinct(inputDeck);

    // In a galaxy simulation, read CLF, SFR, and the stochastic
    // cluster mass fraction
    if (simType_ == SimType::galaxy)
    {
        // CLF
        clf_ = utils::initPDFFromKey(inputDeck, "clusters.CLF");

        // SFR: exactly one of galaxy.sfr (a rate as a function of
        // time -- possibly constant, possibly not, see below) or
        // galaxy.sfr_dist (a distribution to draw a single, constant-
        // for-the-whole-simulation rate from once per Galaxy -- see
        // sfrDist()'s own comment) is required.
        const auto sfrNode = inputDeck.at_path("galaxy.sfr");
        const auto sfrDistNode = inputDeck.at_path("galaxy.sfr_dist");
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
            sfrDist_ = utils::initPDFFromKey(inputDeck, "galaxy.sfr_dist");
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
        const auto fClusterInput = utils::getTOMLKeyWithError<double>(
            inputDeck, "clusters.f_cluster");
        if (fClusterInput.has_value()) { fCluster_ = fClusterInput.value(); }
    }

    // If this simulation has a fixed [Fe/H], precompute the slice at
    // that value once here, up front, so that Cluster objects can
    // share it for the lifetime of the simulation instead of each
    // computing their own copy or racing to populate Tracks3D's
    // internal cache
    if (constFeH())
    {
        constFeHTracks_ = tracks_.sliceConstFeH(fehDist_.getMin());
    }

    // Read minimum stochastic mass
    auto minSM = utils::getTOMLKeyWithError<double>(inputDeck, "stars.min_stoch_mass");
    if (minSM.has_value())
    {
        minStochMass_ = minSM.value();
        fracStochMass_ = imf_.integral(minStochMass_, imf_.getMax());
    }
}

void io::SimControls::setOutputTimes(const toml::table& inputDeck)
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
    const bool hasTimes = static_cast<bool>(inputDeck.at_path("output.output_times"));
    const bool hasStart = static_cast<bool>(inputDeck.at_path("output.start_time"));
    const bool hasEnd = static_cast<bool>(inputDeck.at_path("output.end_time"));
    const bool hasNTime = static_cast<bool>(inputDeck.at_path("output.ntime"));
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
            outTimeDist_ = utils::initPDFFromKey(inputDeck, "output.output_times");
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

// Set the [Fe/H] distribution, recomputing tracks2D() (constFeHTracks_)
// from the current tracks_ if the new fehDist_ is fixed -- mirrors the
// constructor's own post-readTracks() step, which does the same thing
// once, after both tracks_ and fehDist_ are set
void io::SimControls::setFeH(const std::string& feH)
{
    fehDist_ = utils::initPDFFromString(feH);
    if (constFeH())
    {
        constFeHTracks_ = tracks_.sliceConstFeH(fehDist_.getMin());
    }
}

// Set the stellar tracks, recomputing tracks2D() (constFeHTracks_)
// from the new tracks_ if fehDist_ is already fixed -- see setFeH()'s
// own comment
void io::SimControls::setTracks(tracks::Tracks3D tracks)
{
    tracks_ = std::move(tracks);
    if (constFeH())
    {
        constFeHTracks_ = tracks_.sliceConstFeH(fehDist_.getMin());
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
void io::SimControls::readTracks(const toml::table& inputDeck)
{
    // Get required tracks key
    auto trackName = utils::getTOMLKeyWithError<std::string>(inputDeck, "stars.tracks", true);

    // Check for optional parameters
    auto registryName = utils::getTOMLKeyWithError<std::string>(inputDeck, "stars.track_registry");
    auto vvcrit = utils::getTOMLKeyWithError<double>(inputDeck, "stars.v_vcrit");
    auto afe = utils::getTOMLKeyWithError<double>(inputDeck, "stars.alphaFe");

    // Construct tracks from input data
    tracks_ = tracks::Tracks3D(
        trackName.value(), // NOLINT(bugprone-unchecked-optional-access) -- we verified this was valid a few lines ago
        fehDist_.getMin(),
        fehDist_.getMax(),
        vvcrit.value_or(tracks::defaultVVcrit),
        afe.value_or(tracks::defaultAFe),
        registryName.value_or(tracks::defaultRegistry));
}

// Spectral synthesizer reader
void io::SimControls::readSpectra(const toml::table& inputDeck)
{
    // Check for an optional alternative registry
    auto registryNameInput = utils::getTOMLKeyWithError<std::string>(
        inputDeck, "spectra.registry");
    const std::string registryName = registryNameInput.value_or(specsyn::defaultRegistry);

    // spectra.model is optional -- if it is absent, this simulation
    // computes no spectra, and specsyn_ stays null
    const auto modelNode = inputDeck.at_path("spectra.model");
    if (!modelNode) { return; }

    // Load every library-based model over tracks_.feH()'s own
    // [min, max] range, not fehDist_'s own -- tracks_.feH() is always
    // at least as wide as fehDist_ (see Tracks3D::Tracks3D()'s own
    // comment: it pads a few grid points beyond
    // [fehDist_.getMin(), fehDist_.getMax()] on each side, for
    // Mesh3DInterpolator's own benefit), and
    // Specsyn::specCtsHelper()'s [Fe/H] integration deliberately
    // evaluates spectra at every one of those padding grid points too
    // (not just the ones inside fehDist_'s own domain -- see its own
    // comment for why), so every atmosphere grid needs to cover them as
    // well, or spectral synthesis for a star at one of those padding
    // metallicities throws instead of returning a spectrum.
    const auto& fehGrid = tracks_.feH();
    const double fehMin = fehGrid.front();
    const double fehMax = fehGrid.back();

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
    const auto wlMinInput = utils::getTOMLKeyWithError<double>(inputDeck, "spectra.wl_min");
    const auto wlMaxInput = utils::getTOMLKeyWithError<double>(inputDeck, "spectra.wl_max");
    const auto nWlInput = utils::getTOMLKeyWithError<unsigned long>(inputDeck, "spectra.nwl");
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
    const auto zInput = utils::getTOMLKeyWithError<double>(inputDeck, "spectra.z");
    z_ = zInput.value_or(0.0);

    // Optional stars.alphaFe and stars.CFe: if not supplied, fall back
    // to the library defaults. stars.alphaFe is the same key that
    // readTracks() also reads (tracks and spectra share the same AFe
    // value), while stars.CFe is spectra-only (tracks have no CFe axis).
    const auto afeInput = utils::getTOMLKeyWithError<double>(inputDeck, "stars.alphaFe");
    const double afe = afeInput.value_or(tracks::defaultAFe);
    const auto cfeInput = utils::getTOMLKeyWithError<double>(inputDeck, "stars.CFe");
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
            specsyn_ = std::make_unique<specsyn::SpecsynBlackbody>(wlMin_, wlMax_, nWl_, *this);
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
                specsyn_ = std::make_unique<specsyn::SpecsynLibWR<specsyn::OOBPolicy::raise>>(
                    model.value(), fehMin, fehMax, registryName,
                    wlMin_, wlMax_, nWl_, *this);
            }
            else
            {
                specsyn_ = std::make_unique<specsyn::SpecsynLibNoWind<specsyn::OOBPolicy::raise>>(
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

    specsyn_ = std::make_unique<specsyn::SpecsynLibChained>(
        models, fehMin, fehMax,
        afe, cfe, std::vector<double>{},
        specsyn::defaultR, registryName, wlMin_, wlMax_, nWl_, true, *this);
}

// Photometric filter collection reader
void io::SimControls::readFilters(const toml::table& inputDeck)
{
    // phot.system: optional; if present, must name one of the defined
    // photometric systems. Flambda (a raw flux, needing no Vega
    // spectrum or other extra data) is the default if the key is
    // absent entirely.
    phot::PhotSystem photSystem = phot::PhotSystem::Flambda;
    const auto systemNode = inputDeck.at_path("phot.system");
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
    const auto registryInput = utils::getTOMLKeyWithError<std::string>(inputDeck, "phot.registry");
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
    const auto vegaInput = utils::getTOMLKeyWithError<std::string>(inputDeck, "phot.vega");
    if (vegaInput.has_value()) { phot::vegaSpectrum(vegaInput.value()); }

    // phot.filters: optional; a single string names one filter
    // directly, interpreted as an array of length 1; anything else
    // must be an array of strings
    std::vector<std::string> filterNames;
    const auto filtersNode = inputDeck.at_path("phot.filters");
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

    filters_ = std::make_unique<phot::FilterCollection>(
        filterNames, photSystem, registryName);
}

// Build a valid PDF that always draws exactly 0 -- used by readExtinct()
// below to fill in whichever of avDist_/avDistField_ was not given an
// explicit distribution of its own, so the two are always either both
// valid or both invalid, never just one
static auto buildDeltaAV0() -> pdfs::PDF
{
    auto delta = std::make_unique<pdfs::PDFSegmentDelta>(0.0);
    return pdfs::PDF(std::move(delta));
}

// Extinction curve reader
void io::SimControls::readExtinct(const toml::table& inputDeck)
{
    // extinct.AV / extinct.AV_field: both optional; if neither is
    // given, this simulation applies no extinction at all, and
    // avDist_/avDistField_/extinct_ are left at their default/null
    // state. If either is given, extinct.model becomes mandatory
    // (below), and whichever of the two was not given is set to a
    // delta function PDF at 0 instead of being left invalid -- see
    // buildDeltaAV0()'s own comment for why.
    const auto avNode = inputDeck.at_path("extinct.AV");
    const auto avFieldNode = inputDeck.at_path("extinct.AV_field");
    if (!avNode && !avFieldNode) { return; }

    avDist_ = avNode ? utils::initPDFFromKey(inputDeck, "extinct.AV") : buildDeltaAV0();
    avDistField_ = avFieldNode ? utils::initPDFFromKey(inputDeck, "extinct.AV_field") : buildDeltaAV0();

    // extinct.model: required now that extinct.AV or extinct.AV_field
    // was given, names the extinction curve to use
    const auto model = utils::getTOMLKeyWithError<std::string>(
        inputDeck, "extinct.model", true);

    // extinct.registry: optional override of the default extinction
    // curve registry
    const auto registryInput = utils::getTOMLKeyWithError<std::string>(
        inputDeck, "extinct.registry");
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

    extinct_ = std::make_unique<extinct::Extinct>(
        model.value(), specsyn_->wl(), *this, registryName); // NOLINT(bugprone-unchecked-optional-access) -- required=true above guarantees model has a value or getTOMLKeyWithError already threw
}
