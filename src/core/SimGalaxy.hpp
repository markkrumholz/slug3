/**
 * @file SimGalaxy.hpp
 * @author Mark Krumholz
 * @brief A class to drive a galaxy-type simulation end to end
 * @date 2026-08-10
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#ifndef SIMGALAXY_HPP
#define SIMGALAXY_HPP

#include "../io/OutputManager.hpp"
#include "../io/SimControls.hpp"
#include <atomic>
#include <memory>

namespace core
{

    /**
     * @class SimGalaxy
     * @brief Drives a galaxy-type simulation end to end
     */
    class SimGalaxy
    {
    public:

        /**
         * @brief Initialize a galaxy simulation
         * @param simControls Simulation controls (physics settings and
         *   control-flow settings together)
         * @param outputManager Output manager to which simulation
         *   results should be written; ownership is transferred to
         *   this SimGalaxy
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
         * must outlive this SimGalaxy.
         */
        SimGalaxy(const io::SimControls& simControls,
            std::unique_ptr<io::OutputManager> outputManager, bool restart = false);

        // Disallow copying and moving: this object owns the output
        // manager exclusively, so duplicating or relocating it makes
        // no sense
        SimGalaxy(const SimGalaxy&) = delete;
        auto operator=(const SimGalaxy&) -> SimGalaxy& = delete;
        SimGalaxy(SimGalaxy&&) = delete;
        auto operator=(SimGalaxy&&) -> SimGalaxy& = delete;

        ~SimGalaxy() = default;

        /**
         * @brief Conventional shell exit code (128 + SIGTERM) run() returns when it stops early due to a caught SIGTERM
         * @details
         * Matches the exit code a shell would report for a process
         * actually killed by SIGTERM outright, so a caller (e.g. a
         * PBS/SLURM job script checking $? after the slug executable
         * exits) sees the same code whether the process caught the
         * signal gracefully (see run()'s own comment) or not.
         */
        static constexpr int sigtermExitCode = 143;

        /**
         * @brief Run the simulation
         * @return 0 if every trial completed normally; sigtermExitCode
         *   if a SIGTERM was caught and this stopped early instead
         *   (see this method's own comment)
         * @details
         * Installs a handler for SIGTERM before running (restored to
         * whatever was previously installed once this returns, by any
         * path -- normal completion, a caught SIGTERM, or an exception
         * propagating out): if SIGTERM is received while this is
         * running, every trial some thread has already started still
         * finishes normally (runTrial() only ever refuses to *start* a
         * new trial, never aborts one already in progress, so no
         * trial's own output is ever left half-written), but no
         * further trial starts once the currently in-flight ones have
         * all finished -- deliberately checked per trial, not just
         * once the whole current checkpoint-sized batch finishes,
         * since with a large checkpoint interval that batch could
         * itself take far longer than is acceptable to wait out (an
         * hour or more, for a large enough run) before responding.
         * Since dynamic scheduling gives no guarantee that trials
         * finish in numeric order, this can leave a higher-numbered
         * trial done while a lower-numbered one from the same batch
         * never started -- harmless for output correctness (see
         * OutputManager::restartMaxTrial()'s own comment for how a
         * later restart safely resumes numbering and counting despite
         * that), just not perfectly numerically contiguous. Once every
         * already-started trial in the interrupted batch finishes,
         * this returns sigtermExitCode instead of continuing on to any
         * later, not-yet-started batches -- intended to pair with the
         * standard PBS/SLURM practice of sending SIGTERM some minutes
         * before a job's own walltime limit, so a long run can save its
         * progress and exit cleanly instead of being killed outright
         * mid-trial. See OutputManager::notifyEarlyTermination() for
         * how the output itself ends up correctly reflecting only the
         * trials actually completed.
         */
        auto run() -> int;

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

#endif // SIMGALAXY_HPP
