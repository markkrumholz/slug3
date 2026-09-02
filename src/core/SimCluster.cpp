/**
 * @file SimCluster.cpp
 * @author Mark Krumholz
 * @brief Implementation of SimCluster
 * @date 2026-07-17
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#include "SimCluster.hpp"
#include "../io/OutputManager.hpp"
#include "../io/SimControls.hpp"
#include "../utils/UniqueIDManager.hpp"
#include "Cluster.hpp"
#include "SigtermGuard.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <exception> // NOLINT(misc-include-cleaner) -- correct header for std::exception_ptr/current_exception/rethrow_exception; clang-tidy-18's own header-mapping data doesn't yet attribute these symbols to it
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

// Round-trip-safe double -> string conversion: std::to_string(double)
// only ever prints a fixed 6 fractional digits, which is not enough
// to recover an arbitrary double's own exact bit pattern. Used below
// to serialize a failing cluster's own target_mass/form_time, so
// reconstructing it later (via Cluster's rng-state constructor
// overload) starts from the exact same values, not
// std::to_string()'s own rounded-off ones -- see runTrial()'s own
// comment for why that matters.
static auto toRoundTripString(const double value) -> std::string
{
    std::array<char, 32> buf{};
    const auto result = std::to_chars(buf.data(),
        buf.data() + buf.size(), value); // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic) -- buf is a fixed-size stack array, its own end pointer is always in-bounds
    return { buf.data(), result.ptr };
}

core::SimCluster::SimCluster(const io::SimControls& simControls,
    std::unique_ptr<io::OutputManager> outputManager, const bool restart) :
    simControls_(simControls),
    outputManager_(std::move(outputManager)),
    restart_(restart)
{
}

void core::SimCluster::runTrial(const unsigned long trialNum)
{
    // A SIGTERM caught during run()'s own currently-executing batch
    // (see its own comment) means no further trial should actually
    // start any work -- but a trial some other thread already dequeued
    // and started before the signal arrived is deliberately left to
    // run to completion elsewhere, not aborted, so no trial's own
    // output is ever left half-written. Checked first, before any
    // other work here (even the verbosity print below), so a trial
    // that never starts costs as close to nothing as possible. Safe
    // to bail out of individual trials independently like this,
    // without regard to what order they happen to finish in relative
    // to trial number, specifically because run() never treats
    // trialsCompleted_ as if it meant "trials [0, trialsCompleted_)
    // are done" -- see run()'s own comment for why.
    if (sigtermWasReceived()) { return; }

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

    // Create cluster for this trial
    Cluster cluster(utils::uniqueID().get(), simControls_.cmf().draw(), 0, simControls_);

    try
    {
        // Write time-invariant cluster properties to output
        outputManager_->writeCluster(trialNum, cluster);

        // Loop over output times
        const auto outTimes = simControls_.outTimes();
        for (const auto outTime : outTimes)
        {
            cluster.advance(outTime);
            outputManager_->writeClusterSpec(trialNum, outTime, cluster);
            outputManager_->writeClusterPhot(trialNum, outTime, cluster);
        }
    }
    catch (const std::exception& error)
    {
        // Append this cluster's own birth-time rng state (and the
        // uid/mass/time needed alongside it) to the error message, so
        // even a run with no output file to recover it from (e.g. a
        // non-HDF5 OutputManager, or a crash before writeCluster()'s
        // own flush -- see OutputManagerH5::writeCluster()'s comment)
        // still leaves enough in this trial's own error message to
        // deterministically reconstruct the exact failing cluster
        // afterward, via Cluster's rng-state constructor overload.
        throw std::runtime_error(
            std::string(error.what()) +
            " (cluster uid=" + std::to_string(cluster.uid()) +
            ", target_mass=" + toRoundTripString(cluster.targetMass()) +
            ", form_time=" + toRoundTripString(cluster.formTime()) +
            ", rng=\"" + cluster.rngState().data() + "\")");
    }

    trialsCompleted_.fetch_add(1, std::memory_order_relaxed);
}

auto core::SimCluster::run() -> int
{
    // See this method's own header comment: covers every exit path
    // below (normal completion, the early SIGTERM return, or firstError
    // being rethrown) uniformly, via its own destructor
    const SigtermGuard sigtermGuard("SimCluster::run: unable to install a SIGTERM handler");

    // Trial *numbering* and trial *counting* are deliberately tracked
    // separately here, and must not be conflated -- see
    // OutputManager::restartMaxTrial()'s own comment for the full
    // rationale, and runTrial()'s own comment for how that connects to
    // the per-trial SIGTERM check below. In short: under dynamic
    // OpenMP scheduling, a batch a SIGTERM interrupted partway through
    // can finish with a higher-numbered trial done while a lower-
    // numbered one is not (whichever thread happened to still be mid-
    // flight when the others stopped taking on new work), so "how many
    // trials are done" and "which trial numbers are safe to reuse" are
    // two different questions with two different answers:
    //   - priorTrialsCompleted (from OutputManager::restartTrialsDone())
    //     is an accurate *count* of trials the run being restarted had
    //     already completed, regardless of which specific numbers they
    //     were -- trialsRemaining, how many *more* this session needs
    //     to run to reach simControls_.nTrial() in total, follows
    //     directly from it and needs nothing else.
    //   - numberingStart (from OutputManager::restartMaxTrial() + 1)
    //     is the smallest trial number guaranteed never to have been
    //     written before, however ragged the run being restarted's own
    //     numbering ended up -- this, not priorTrialsCompleted, is what
    //     this session's own new trials must be numbered from to avoid
    //     colliding with (and so silently overwriting the identity of)
    //     one already on disk. It is fine for trial numbers to end up
    //     non-contiguous across a restart (e.g. 0..3, 5..9 -- number 4
    //     simply never used again) -- the total *count* across every
    //     checkpoint is what has to come out right, not the specific
    //     numbers.
    // trialsCompleted_ (this object's own atomic counter) tracks only
    // what *this* session itself completes, starting at 0 regardless
    // of restart_ -- cumulativeTrialsCompleted, computed fresh
    // wherever it's needed below, is what actually gets reported to
    // outputManager_, combining that with priorTrialsCompleted.
    const unsigned long priorTrialsCompleted = restart_ ? outputManager_->restartTrialsDone() : 0;
    const unsigned long numberingStart = restart_ ? outputManager_->restartMaxTrial() + 1 : 0;
    // Guards against underflow if priorTrialsCompleted somehow exceeds
    // simControls_.nTrial() (e.g. n_trial was lowered in the deck
    // between the original run and this restart) -- treated as
    // "nothing left to do" rather than wrapping around to an
    // enormous trial count.
    const unsigned long trialsRemaining = (priorTrialsCompleted < simControls_.nTrial()) ?
        (simControls_.nTrial() - priorTrialsCompleted) : 0;
    const unsigned long numberingEnd = numberingStart + trialsRemaining;

    if (simControls_.verbosity() > 0)
    {
        std::cout << "slug: cluster simulation starting with "
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
    // (writeCluster/writeClusterSpec/writeClusterPhot for every output
    // time, not just started it) before this reaches the checkpoint()
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

#ifdef _OPENMP
        // See runTrial()'s own comment for why each trial is
        // individually wrapped in a try/catch here, rather than
        // letting an exception propagate out of the "#pragma omp
        // parallel for" loop body directly: every trial in this batch
        // still runs (so one bad trial does not lose every other
        // thread's own in-flight work), and the first exception
        // caught is remembered and rethrown once the whole batch's
        // own parallel region has finished, so run()'s own caller
        // sees the same kind of failure it always has.
        std::exception_ptr firstError;
#pragma omp parallel for schedule(dynamic)
        for (unsigned long trialNum = batchStart; trialNum < batchEnd; ++trialNum)
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
        if (firstError)
        {
            // This run is ending abnormally, having completed fewer
            // than simControls_.nTrial() trials in total -- see
            // OutputManager::notifyEarlyTermination()'s own comment for
            // why the eventual destructor needs to be told that
            // explicitly, rather than assuming every trial finished the
            // way it otherwise would.
            outputManager_->notifyEarlyTermination(
                priorTrialsCompleted + trialsCompleted_.load(std::memory_order_relaxed));
            std::rethrow_exception(firstError);
        }
#else
        try
        {
            for (unsigned long trialNum = batchStart; trialNum < batchEnd; ++trialNum)
            {
                runTrial(trialNum);
            }
        }
        catch (...)
        {
            // See the identical #ifdef _OPENMP branch's own comment
            outputManager_->notifyEarlyTermination(
                priorTrialsCompleted + trialsCompleted_.load(std::memory_order_relaxed));
            throw;
        }
#endif

        // If SIGTERM was caught somewhere in this batch, this is the
        // correct, safe point to stop: every trial actually started
        // has now finished (runTrial()'s own check only ever stops a
        // trial from starting in the first place), so it is safe to
        // save the currently-open checkpoint (or, without
        // checkpointing, the run's own single output file) with the
        // true, accurate cumulative count of trials completed so far,
        // via notifyEarlyTermination() rather than checkpoint() --
        // there is nothing left to write into a new checkpoint, so
        // rolling over to one would just leave it empty and un-closed
        // for the destructor to eventually close with a wrong trial
        // count of its own (see notifyEarlyTermination()'s own
        // comment) -- and return sigtermExitCode instead of continuing
        // on to any later, not-yet-started batches.
        if (sigtermWasReceived())
        {
            const auto cumulativeCompleted =
                priorTrialsCompleted + trialsCompleted_.load(std::memory_order_relaxed);
            outputManager_->notifyEarlyTermination(cumulativeCompleted);
            if (simControls_.verbosity() > 0)
            {
                std::cout << "slug: caught SIGTERM, stopping early with "
                    << cumulativeCompleted << " / " << simControls_.nTrial() <<
                    " trials completed\n";
            }
            return sigtermExitCode;
        }

        // No SIGTERM was caught, so every trial in [batchStart,
        // batchEnd) completed -- this batch is gap-free, so
        // trialsCompleted_ now correctly counts exactly
        // (batchEnd - numberingStart) trials this session has done.
        // Safe to roll over to a new checkpoint now, unless this was
        // the final batch (in which case the current checkpoint is
        // simply left open for the destructor to close/consolidate,
        // exactly as when checkpointing is disabled entirely).
        if (simControls_.checkpointInterval() != 0 && batchEnd < numberingEnd)
        {
            outputManager_->checkpoint(
                priorTrialsCompleted + trialsCompleted_.load(std::memory_order_relaxed));
        }
    }

    if (simControls_.verbosity() > 0)
    {
        std::cout << "slug: simulation complete\n";
    }
    return 0;
}
