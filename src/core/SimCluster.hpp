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
         * @details
         * simControls is stored by reference, so the object passed in
         * must outlive this SimCluster.
         */
        SimCluster(const io::SimControls& simControls,
            std::unique_ptr<io::OutputManager> outputManager);

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

        const io::SimControls& simControls_; /**< Simulation controls (physics and control-flow settings) */
        std::unique_ptr<io::OutputManager> outputManager_; /**< Output manager */
        std::atomic<unsigned long> trialsCompleted_{0}; /**< Number of trials completed so far (see trialsCompleted()) */
    };

} // namespace core

#endif // SIMCLUSTER_HPP
