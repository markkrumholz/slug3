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
    std::unique_ptr<io::OutputManager> outputManager) :
    simControls_(simControls),
    outputManager_(std::move(outputManager))
{
}

void core::SimCluster::runTrial(const unsigned long trialNum)
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

    // Create cluster for this trial
    Cluster cluster(utils::getID(), simControls_.cmf().draw(), 0, simControls_);

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

void core::SimCluster::run()
{
    if (simControls_.verbosity() > 0)
    {
        std::cout << "slug: cluster simulation starting with "
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
