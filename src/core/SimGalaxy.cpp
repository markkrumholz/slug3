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
#include "../utils/MPIUtils.hpp"
#include "../utils/MiscUtils.hpp"
#include "../utils/SigtermGuard.hpp"
#include "Galaxy.hpp"
#include <algorithm>
#include <atomic>
#include <exception> // NOLINT(misc-include-cleaner) -- correct header for std::exception_ptr/current_exception/rethrow_exception; clang-tidy-18's own header-mapping data doesn't yet attribute these symbols to it
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

// See SimCluster.cpp's own identical helper for why this exists and
// why it reports cumulativeCompleted plainly, not as a fraction of
// nTrial.
static void printSigtermMessage(const unsigned long cumulativeCompleted, const unsigned long nTrial)
{
    std::cout << "slug: caught SIGTERM, stopping early with " <<
        cumulativeCompleted << " trials completed";
    if (utils::mpiSize() > 1)
    {
        std::cout << " on rank " << utils::mpiRank() <<
            " (this rank's own share of " << nTrial << " total trials across all ranks)";
    }
    std::cout << "\n";
}

core::SimGalaxy::SimGalaxy(const io::SimControls& simControls,
    std::unique_ptr<io::OutputManager> outputManager, const bool restart) :
    simControls_(simControls),
    outputManager_(std::move(outputManager)),
    restart_(restart),
    testFailTrial_(utils::getEnvULong("SLUG_TEST_FAIL_TRIAL")),
    testFailRank_(utils::getEnvULong("SLUG_TEST_FAIL_RANK"))
{
}

void core::SimGalaxy::runTrial(const unsigned long trialNum)
{
    // See SimCluster::runTrial()'s own identical comment
    if (utils::sigtermWasReceived()) { return; }

    // Test-only fault injection; see testFailTrial_'s own comment
    if (testFailTrial_ == trialNum)
    {
        throw std::runtime_error("injected test failure (SLUG_TEST_FAIL_TRIAL = " +
            std::to_string(trialNum) + ")");
    }

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
    // writeGalaxyPhot/writeGalaxyYields/writeGalaxyFeedback each also
    // write out every currently-alive cluster in the galaxy, so no
    // separate writeCluster/writeClusterSpec/writeClusterPhot/
    // writeClusterYields/writeClusterFeedback calls are needed here (unlike SimCluster::runTrial's single
    // writeCluster call, there is no single moment when every cluster
    // this trial will ever form already exists, since new clusters
    // keep forming as the galaxy advances)
    const auto outTimes = simControls_.outTimes();
    for (const auto outTime : outTimes)
    {
        galaxy.advance(outTime);
        outputManager_->writeGalaxy(trialNum, outTime, galaxy);
        outputManager_->writeGalaxySpec(trialNum, outTime, galaxy);
        outputManager_->writeGalaxyPhot(trialNum, outTime, galaxy);
        outputManager_->writeGalaxyYields(trialNum, outTime, galaxy);
        outputManager_->writeGalaxyFeedback(trialNum, outTime, galaxy);
    }

    trialsCompleted_.fetch_add(1, std::memory_order_relaxed);
}

// See SimCluster::runBatch()'s own header comment
auto core::SimGalaxy::runBatch(const unsigned long rankBatchStart, const unsigned long rankBatchEnd)
    -> std::exception_ptr
{
    // See SimCluster::runBatch()'s own identical comment
    std::exception_ptr firstError;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
    for (unsigned long trialNum = rankBatchStart; trialNum < rankBatchEnd; ++trialNum)
    {
        try
        {
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
#else
    for (unsigned long trialNum = rankBatchStart; trialNum < rankBatchEnd; ++trialNum)
    {
        try
        {
            runTrial(trialNum);
        }
        catch (const std::exception& error)
        {
            std::cerr << "slug: trial " << trialNum << " failed: " << error.what() << "\n";
            firstError = std::current_exception();
            break;
        }
    }
#endif
    return firstError;
}

auto core::SimGalaxy::run() -> int
{
    // See SimCluster::run()'s own identical comment
    const utils::SigtermGuard sigtermGuard("SimGalaxy::run: unable to install a SIGTERM handler");

    // See SimCluster::run()'s own identical comment for the full
    // rationale behind tracking numbering and counting separately
    const unsigned long priorTrialsCompleted = restart_ ? outputManager_->restartTrialsDone() : 0;
    const unsigned long numberingStart = restart_ ? outputManager_->restartMaxTrial() + 1 : 0;
    const unsigned long trialsRemaining = (priorTrialsCompleted < simControls_.nTrial()) ?
        (simControls_.nTrial() - priorTrialsCompleted) : 0;
    const unsigned long numberingEnd = numberingStart + trialsRemaining;

    // Gated to rank 0 only -- see SimCluster::run()'s own identical comment
    if (simControls_.verbosity() > 0 && utils::isIORank())
    {
        std::cout << "slug: galaxy simulation starting with "
            << simControls_.nTrial() << " trials";
        if (priorTrialsCompleted != 0)
        {
            std::cout << " (resuming: " << priorTrialsCompleted <<
                " already done, numbering new trials from " << numberingStart << ")";
        }
        std::cout << "\n";
    }

    // Trials run in batches of checkpointInterval() at a time (or, if
    // checkpointing is disabled, one batch covering every remaining
    // trial, i.e. today's original, unbatched behavior) -- each batch
    // its own "#pragma omp parallel for", so its own implicit barrier
    // guarantees every thread has finished every trial in the batch
    // (writeGalaxy/writeGalaxySpec/writeGalaxyPhot/writeGalaxyYields/
    // writeGalaxyFeedback for every output time, not just started it)
    // before this reaches the
    // checkpoint()
    // call below, and that this runs on a single thread, from outside
    // any active parallel region, exactly as OutputManagerH5::
    // checkpoint() itself requires (see its own comment). Without this
    // batching, checkpoint() was instead called per-thread, the
    // instant that thread's own assigned trial number happened to be
    // a multiple of checkpointInterval() -- under dynamic scheduling,
    // a different thread's own in-flight trial could straddle that
    // same moment, with no guarantee its own output was fully written
    // before the checkpoint closed, and no guarantee a lagging thread
    // would even reach an exact multiple soon (or at all) to roll over
    // on its own. Batching costs every thread idling until the
    // batch's own slowest trial finishes, once per checkpoint --
    // deliberately traded for that correctness guarantee.
    const unsigned long batchSize = (simControls_.checkpointInterval() != 0) ?
        simControls_.checkpointInterval() : trialsRemaining;

    for (unsigned long batchStart = numberingStart; batchStart < numberingEnd;
        batchStart += batchSize)
    {
        const unsigned long batchEnd =
            std::min(batchStart + batchSize, numberingEnd);

        // See SimCluster::run()'s own identical comment
        const auto [rankBatchStart, rankBatchEnd] = utils::mpiPartitionRange(batchStart, batchEnd);
        const std::exception_ptr firstError = runBatch(rankBatchStart, rankBatchEnd);

        // See SimCluster::run()'s own identical comments, here and on
        // the two stop conditions below
        const auto stop = utils::mpiAgreeBatchStop(utils::sigtermWasReceived(), firstError != nullptr);
        if (stop.error_)
        {
            outputManager_->notifyEarlyTermination(
                priorTrialsCompleted + trialsCompleted_.load(std::memory_order_relaxed));
            utils::rethrowAgreedError(firstError);
        }

        if (stop.sigterm_)
        {
            const auto cumulativeCompleted =
                priorTrialsCompleted + trialsCompleted_.load(std::memory_order_relaxed);
            outputManager_->notifyEarlyTermination(cumulativeCompleted);
            if (simControls_.verbosity() > 0)
            {
                printSigtermMessage(cumulativeCompleted, simControls_.nTrial());
            }
            return sigtermExitCode;
        }

        // Test-only fault injection; see SimCluster::run()'s own
        // identical comment
        if (batchStart == numberingStart &&
            testFailRank_ == static_cast<unsigned long>(utils::mpiRank()))
        {
            throw std::runtime_error("injected test failure (SLUG_TEST_FAIL_RANK = " +
                std::to_string(utils::mpiRank()) + ")");
        }

        // See SimCluster::run()'s own identical comment
        if (simControls_.checkpointInterval() != 0 && batchEnd < numberingEnd)
        {
            outputManager_->checkpoint(
                priorTrialsCompleted + trialsCompleted_.load(std::memory_order_relaxed));
        }
    }

    // See SimCluster::run()'s own identical comment
    outputManager_->notifyEarlyTermination(
        priorTrialsCompleted + trialsCompleted_.load(std::memory_order_relaxed));

    // I/O rank only, like the starting summary above: by here every
    // rank has passed notifyEarlyTermination()'s own synchronization,
    // so there is nothing rank-specific left to report
    if (simControls_.verbosity() > 0 && utils::isIORank())
    {
        std::cout << "slug: simulation complete\n";
    }
    return 0;
}
