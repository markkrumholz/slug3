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
         * @details
         * simControls is stored by reference, so the object passed in
         * must outlive this SimGalaxy.
         */
        SimGalaxy(const io::SimControls& simControls,
            std::unique_ptr<io::OutputManager> outputManager);

        // Disallow copying and moving: this object owns the output
        // manager exclusively, so duplicating or relocating it makes
        // no sense
        SimGalaxy(const SimGalaxy&) = delete;
        auto operator=(const SimGalaxy&) -> SimGalaxy& = delete;
        SimGalaxy(SimGalaxy&&) = delete;
        auto operator=(SimGalaxy&&) -> SimGalaxy& = delete;

        ~SimGalaxy() = default;

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
    };

} // namespace core

#endif // SIMGALAXY_HPP
