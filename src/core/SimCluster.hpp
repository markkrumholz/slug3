/**
 * @file SimCluster.hpp
 * @author Mark Krumholz
 * @brief A class to drive a cluster-type simulation end to end
 * @date 2026-07-17
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#ifndef SIMCLUSTER_HPP
#define SIMCLUSTER_HPP

#include "../io/OutputManager.hpp"
#include "../io/SimControls.hpp"
#include <atomic>
#include <memory>

namespace core
{

    /**
     * @class SimCluster
     * @brief Drives a cluster-type simulation end to end
     */
    class SimCluster
    {
    public:

        /**
         * @brief Initialize a cluster simulation
         * @param simControls Simulation controls (physics settings and
         *   control-flow settings together)
         * @param outputManager Output manager to which simulation
         *   results should be written; ownership is transferred to
         *   this SimCluster
         * @param restart Whether this run is resuming a previous,
         *   interrupted run from its most recent checkpoint; defaults
         *   to false. If true, run() asks outputManager for the
         *   number of trials the run being restarted had already
         *   completed (OutputManager::restartTrialsDone()) and starts
         *   from there instead of from trial 0 -- see run()'s own
         *   comment. outputManager must itself have already been
         *   constructed with its own equivalent restart flag set (see
         *   OutputManagerH5's own constructor); nothing here enforces
         *   that the two agree.
         * @details
         * simControls is stored by reference, so the object passed in
         * must outlive this SimCluster.
         */
        SimCluster(const io::SimControls& simControls,
            std::unique_ptr<io::OutputManager> outputManager, bool restart = false);

        // Disallow copying and moving: this object owns the output
        // manager exclusively, so duplicating or relocating it makes
        // no sense
        SimCluster(const SimCluster&) = delete;
        auto operator=(const SimCluster&) -> SimCluster& = delete;
        SimCluster(SimCluster&&) = delete;
        auto operator=(SimCluster&&) -> SimCluster& = delete;

        ~SimCluster() = default;

        /**
         * @brief Run the simulation
         */
        void run();

        /**
         * @brief Get the number of trials completed so far
         * @return Number of trials this simulation has finished
         *   running, out of simControls.nTrial() total
         * @details
         * Reads a relaxed atomic, so this is safe to call from another
         * thread while run() is still executing (e.g. to drive a
         * progress bar) without blocking on, or being blocked by, the
         * OpenMP worker threads run() itself uses.
         */
        [[nodiscard]] auto trialsCompleted() const -> unsigned long
        {
            return trialsCompleted_.load(std::memory_order_relaxed);
        }

    private:

        /**
         * @brief Run a single trial
         * @param trialNum Trial number to run
         * @details
         * Factored out of run() so its own OpenMP loop body can be a
         * single call wrapped in a try/catch: an uncaught exception
         * escaping an "#pragma omp parallel for" loop body cannot be
         * caught outside it (OpenMP requires catching within the same
         * thread, inside the structured block, or the program
         * terminates via std::terminate()), so run() catches around
         * this call instead, letting every other thread's own trials
         * finish normally rather than losing the whole run to one bad
         * trial.
         */
        void runTrial(unsigned long trialNum);

        const io::SimControls& simControls_; /**< Simulation controls (physics and control-flow settings) */
        std::unique_ptr<io::OutputManager> outputManager_; /**< Output manager */
        std::atomic<unsigned long> trialsCompleted_{0}; /**< Number of trials completed so far (see trialsCompleted()) */
        bool restart_ = false; /**< Whether this run is resuming a previous, interrupted run (see the constructor's own comment) */
    };

} // namespace core

#endif // SIMCLUSTER_HPP
