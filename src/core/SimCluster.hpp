/**
 * @file SimCluster.hpp
 * @author Mark Krumholz
 * @brief A class to drive a cluster-type simulation end to end
 * @date 2026-07-17
 */

#ifndef SIMCLUSTER_HPP
#define SIMCLUSTER_HPP

#include "../io/OutputManager.hpp"
#include "../io/SimControls.hpp"
#include "../io/SimPhysics.hpp"
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
         * @param simControls Simulation control flow settings
         * @param simPhysics Simulation physics settings
         * @param outputManager Output manager to which simulation
         *   results should be written; ownership is transferred to
         *   this SimCluster
         * @details
         * simControls and simPhysics are stored by reference, so the
         * objects passed in must outlive this SimCluster.
         */
        SimCluster(const io::SimControls& simControls,
            const io::SimPhysics& simPhysics,
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

    private:

        const io::SimControls& simControls_; /**< Simulation control flow settings */
        const io::SimPhysics& simPhysics_;   /**< Simulation physics settings */
        std::unique_ptr<io::OutputManager> outputManager_; /**< Output manager */
    };

} // namespace core

#endif // SIMCLUSTER_HPP
