/**
 * @file SimControls.cpp
 * @author Mark Krumholz
 * @date 2026-07-16
 * @brief Implmenentation of SimControls class
 */

#include "SimControls.hpp"
#include "../extern/tomlplusplus/toml.hpp"
#include "../utils/ParseUtils.hpp"
#include "../utils/ThreadVec.hpp"
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

core::SimControls::SimControls(const toml::table& inputDeck) :
    //outputMode_(OutputMode::h5),
    modelName_("slug_sim"),
    verbosity_(0),
    nTrial_(1),
    nTrialRemain_(1)
{
    // Read output mode
    //const auto outputMode = utils::getTOMLKeyWithError<std::string>(
    //    inputDeck, "outputs.output_mode");
    //if (outputMode.has_value())
    //{
    //    if (outputMode.value() == "h5" || outputMode.value() == "hdf5")
    //    { outputMode_ = OutputMode::h5; }
    //    else if (outputMode.value() == "ascii" || outputMode.value() == "txt")
    //    { outputMode_ = OutputMode::ascii; }
    //    else
    //    {
    //        throw std::runtime_error("SimControls: unknown output_mode "
    //             + outputMode.value());
    //    }
    //}

    // Read verbosity
    const auto verbosityInput =
        utils::getTOMLKeyWithError<unsigned int>(inputDeck, "verbosity");
    if (verbosityInput.has_value()) { verbosity_ = verbosityInput.value(); }

    // Read model name
    const auto modelNameInput =
        utils::getTOMLKeyWithError<std::string>(inputDeck, "model_name");
    if (modelNameInput.has_value()) { modelName_ = modelNameInput.value(); }

    // Read number of trials
    const auto nTrialInput = 
        utils::getTOMLKeyWithError<unsigned long>(inputDeck, "n_trial");
    if (nTrialInput.has_value())
    {
        nTrial_ = nTrialInput.value();
        nTrialRemain_ = nTrial_;    
    }

    // Handle output time generation
    setOutputTimes(inputDeck);

    // If we have been given a specific RNG seed, use it
    const auto rngSeed =
        utils::getTOMLKeyWithError<unsigned long>(inputDeck, "rng_seed");
    if (rngSeed.has_value()) { utils::rng().seed(rngSeed.value()); }
}

void core::SimControls::setOutputTimes(const toml::table& inputDeck)
{
    // This routine computes the output times. These are set in a somewhat
    // complex way: a user can specify specify the times of outputs in three distinct
    // ways:
    // (1) The user can set outputs.output_time_dist, which will be
    //     interpreted as a PDF. In this case there is one output per
    //     simulation, with the output time drawn from the PDF.
    // (2) The user can specify outputs.output_times, which will be
    //     interpreted as an explicit array of output times, wiht one
    //     output per trial at each of the specified times.
    // (3) The user can specify all three of outputs.start_time,
    //     outputs.end_time, and outputs.ntime, with the first two interpreted
    //     as doubles (required to be >= 0) and the third as an unsigned int (which must be > 0). In
    //     this case the outputs times will be automatically generated as a
    //     uniformly-spaced array of ntime values from start_time to end_time.
    //     (For this option, start_time == end_time is allowed only if ntime = 1,
    //     and ntime == 1 is allowed only if start_time == end_time.)
    // (3a) For option 3, the user can also specify the optional boolean
    //      output.log_time; if this is specified, the array of values generated
    //      will be log-spaced rather than linearly spaced.
    // Thus in this routine we need to check which of these options the user has
    // provided, verify that only that option is been provided (e.g., the user
    // hasn't accidentially provided both output_time_dist and output_times), and
    // fill the variables outTimes_ and outTimeDist_ based on them. For option 1,
    // outTimeDist_ will be set to a valid PDF and outTimes_ will be left empty,
    // while for options 2 or 3 outTimes_ will be a non-empty array and outTimeDist_
    // will be left as an invalid, uninitialized PDF.

    // Determine which option(s) the user has specified
    const bool hasDist = static_cast<bool>(inputDeck.at_path("outputs.output_time_dist"));
    const bool hasTimes = static_cast<bool>(inputDeck.at_path("outputs.output_times"));
    const bool hasStart = static_cast<bool>(inputDeck.at_path("outputs.start_time"));
    const bool hasEnd = static_cast<bool>(inputDeck.at_path("outputs.end_time"));
    const bool hasNTime = static_cast<bool>(inputDeck.at_path("outputs.ntime"));
    const bool hasRange = hasStart || hasEnd || hasNTime;

    const int nOptions = static_cast<int>(hasDist) +
        static_cast<int>(hasTimes) + static_cast<int>(hasRange);
    if (nOptions == 0)
    {
        throw std::runtime_error(
            "SimControls: must specify one of outputs.output_time_dist, "
            "outputs.output_times, or outputs.start_time/outputs.end_time/outputs.ntime");
    }
    if (nOptions > 1)
    {
        throw std::runtime_error(
            "SimControls: only one of outputs.output_time_dist, "
            "outputs.output_times, or outputs.start_time/outputs.end_time/outputs.ntime "
            "may be specified");
    }

    // Option 1: draw a single output time per trial from a distribution
    if (hasDist)
    {
        outTimeDist_ = utils::initPDFFromKey(inputDeck, "outputs.output_time_dist");
        return;
    }

    // Option 2: an explicit array of output times
    if (hasTimes)
    {
        const toml::array* arr = inputDeck.at_path("outputs.output_times").as_array();
        if (arr == nullptr)
        {
            throw std::runtime_error(
                "SimControls: outputs.output_times must be an array of numbers");
        }
        outTimes_.clear();
        outTimes_.reserve(arr->size());
        for (size_t i = 0; i < arr->size(); ++i)
        {
            const auto val = arr->at(i).value<double>();
            if (!val.has_value())
            {
                throw std::runtime_error(
                    "SimControls: outputs.output_times must be an array of numbers");
            }
            outTimes_.push_back(val.value());
        }
        return;
    }

    // Option 3: a uniformly- or log-spaced grid of output times
    if (!(hasStart && hasEnd && hasNTime))
    {
        throw std::runtime_error(
            "SimControls: outputs.start_time, outputs.end_time, and "
            "outputs.ntime must all be specified together");
    }
    const auto startTime = utils::getTOMLKeyWithError<double>(
        inputDeck, "outputs.start_time", true).value(); // NOLINT(bugprone-unchecked-optional-access) -- required=true guarantees this
    const auto endTime = utils::getTOMLKeyWithError<double>(
        inputDeck, "outputs.end_time", true).value(); // NOLINT(bugprone-unchecked-optional-access) -- required=true guarantees this
    const auto nTime = utils::getTOMLKeyWithError<unsigned long>(
        inputDeck, "outputs.ntime", true).value(); // NOLINT(bugprone-unchecked-optional-access) -- required=true guarantees this
    const auto logTime = utils::getTOMLKeyWithError<bool>(
        inputDeck, "outputs.log_time").value_or(false);

    if (startTime < 0.0 || endTime < 0.0)
    {
        throw std::runtime_error(
            "SimControls: outputs.start_time and outputs.end_time must be >= 0");
    }
    if (nTime == 0)
    {
        throw std::runtime_error("SimControls: outputs.ntime must be > 0");
    }
    if ((startTime == endTime) != (nTime == 1))
    {
        throw std::runtime_error(
            "SimControls: outputs.start_time == outputs.end_time is allowed "
            "only if outputs.ntime == 1, and outputs.ntime == 1 is allowed "
            "only if outputs.start_time == outputs.end_time");
    }
    if (logTime && startTime <= 0.0)
    {
        throw std::runtime_error(
            "SimControls: outputs.start_time must be > 0 when outputs.log_time is set");
    }

    outTimes_.assign(nTime, 0.0);
    if (nTime == 1)
    {
        outTimes_[0] = startTime;
    }
    else if (logTime)
    {
        const double logStart = std::log(startTime);
        const double logEnd = std::log(endTime);
        const double dLog = (logEnd - logStart) / static_cast<double>(nTime - 1);
        for (unsigned long i = 0; i < nTime; ++i)
        {
            outTimes_[i] = std::exp(logStart + static_cast<double>(i) * dLog);
        }
    }
    else
    {
        const double dt = (endTime - startTime) / static_cast<double>(nTime - 1);
        for (unsigned long i = 0; i < nTime; ++i)
        {
            outTimes_[i] = startTime + static_cast<double>(i) * dt;
        }
    }
}