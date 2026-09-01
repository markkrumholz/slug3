/**
 * @file SimGalaxy.cpp
 * @author Mark Krumholz
 * @brief Implementation of SimGalaxy
 * @date 2026-08-10
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#include "SimGalaxy.hpp"
#include "../io/OutputManager.hpp"
#include "../io/SimControls.hpp"
#include "Galaxy.hpp"
#include <atomic>
#include <exception> // NOLINT(misc-include-cleaner) -- correct header for std::exception_ptr/current_exception/rethrow_exception; clang-tidy-18's own header-mapping data doesn't yet attribute these symbols to it
#include <iostream>
#include <memory>
#include <utility>

core::SimGalaxy::SimGalaxy(const io::SimControls& simControls,
    std::unique_ptr<io::OutputManager> outputManager) :
    simControls_(simControls),
    outputManager_(std::move(outputManager))
{
}

void core::SimGalaxy::runTrial(const unsigned long trialNum)
{
    if (simControls_.verbosity() > 1)
    {
#ifdef _OPENMP
#pragma omp critical
#endif
        {
            std::cout << "slug: starting trial " << trialNum << " / "
                << simControls_.nTrial() << "\n";
        }
    }

    // Create galaxy for this trial
    Galaxy galaxy(simControls_);

    // Loop over output times, advancing the galaxy's internal state
    // and writing it out at each one -- writeGalaxy/writeGalaxySpec/
    // writeGalaxyPhot each also write out every currently-alive
    // cluster in the galaxy, so no separate writeCluster/
    // writeClusterSpec/writeClusterPhot calls are needed here (unlike
    // SimCluster::runTrial's single writeCluster call, there is no
    // single moment when every cluster this trial will ever form
    // already exists, since new clusters keep forming as the galaxy
    // advances)
    const auto outTimes = simControls_.outTimes();
    for (const auto outTime : outTimes)
    {
        galaxy.advance(outTime);
        outputManager_->writeGalaxy(trialNum, outTime, galaxy);
        outputManager_->writeGalaxySpec(trialNum, outTime, galaxy);
        outputManager_->writeGalaxyPhot(trialNum, outTime, galaxy);
    }

    trialsCompleted_.fetch_add(1, std::memory_order_relaxed);
}

void core::SimGalaxy::run()
{
    if (simControls_.verbosity() > 0)
    {
        std::cout << "slug: galaxy simulation starting with "
            << simControls_.nTrial() << " trials\n";
    }

#ifdef _OPENMP
    // See runTrial()'s own comment for why each trial is individually
    // wrapped in a try/catch here, rather than letting an exception
    // propagate out of the "#pragma omp parallel for" loop body
    // directly: every trial still runs (so one bad trial does not
    // lose every other thread's own in-flight work), and the first
    // exception caught is remembered and rethrown once the whole
    // parallel region has finished, so run()'s own caller sees the
    // same kind of failure it always has.
    std::exception_ptr firstError;
#pragma omp parallel for schedule(dynamic)
    for (unsigned long trialNum = 0; trialNum < simControls_.nTrial(); ++trialNum)
    {
        try
        {
            // Roll over to a new checkpoint every checkpointInterval()
            // trials, if checkpointing is enabled -- see
            // OutputManagerH5::checkpoint()'s own comment for what
            // this actually does and why
            if (simControls_.checkpointInterval() != 0 && trialNum != 0 &&
                trialNum % simControls_.checkpointInterval() == 0)
            {
                outputManager_->checkpoint();
            }
            runTrial(trialNum);
        }
        catch (const std::exception& error)
        {
#pragma omp critical
            {
                std::cerr << "slug: trial " << trialNum << " failed: " << error.what() << "\n";
                if (!firstError) { firstError = std::current_exception(); }
            }
        }
    }
    if (firstError) { std::rethrow_exception(firstError); }
#else
    for (unsigned long trialNum = 0; trialNum < simControls_.nTrial(); ++trialNum)
    {
        if (simControls_.checkpointInterval() != 0 && trialNum != 0 &&
            trialNum % simControls_.checkpointInterval() == 0)
        {
            outputManager_->checkpoint();
        }
        runTrial(trialNum);
    }
#endif

    if (simControls_.verbosity() > 0)
    {
        std::cout << "slug: simulation complete\n";
    }
}
