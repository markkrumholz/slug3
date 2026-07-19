/**
 * @file OutputManager.hpp
 * @author Mark Krumholz
 * @brief A class to manage writing simulation outputs to disk
 * @date 2026-07-16
 */

#ifndef OUTPUTMANAGER_HPP
#define OUTPUTMANAGER_HPP

#include "SimControls.hpp"
#include "SimPhysics.hpp"
#include <string>
#include <toml.hpp>
#include <utility>

namespace core
{
    class Cluster;
} // namespace core

namespace io
{

    /**
     * @class OutputManager
     * @brief A base class to manage writing simulation outputs to disk
     * @details
     * The ascii and HDF5 output formats differ enough that almost every
     * member function needs its own implementation for each, so this
     * class defines only the interface and the behavior shared by both
     * formats; the format-specific work is implemented by the
     * OutputManagerAscii and OutputManagerH5 subclasses.
     */
    class OutputManager
    {
    public:

        /**
         * @brief Cache references to the simulation controls, physics, and input deck
         * @param simControls Simulation control flow settings
         * @param simPhysics Simulation physics settings
         * @param inputDeck The simulation's toml input deck
         * @details
         * simControls, simPhysics, and inputDeck are stored by
         * reference, so the objects passed in must outlive this
         * OutputManager. This base constructor does not open any
         * output files; that is left to the constructors of the
         * format-specific subclasses.
         */
        OutputManager(const SimControls& simControls, const SimPhysics& simPhysics,
            const toml::table& inputDeck);

        virtual ~OutputManager() = default;

        // Disallow copying and moving: subclasses represent exclusive
        // ownership of on-disk output, so duplicating them makes no sense
        OutputManager(const OutputManager&) = delete;
        auto operator=(const OutputManager&) -> OutputManager& = delete;
        OutputManager(OutputManager&&) = delete;
        auto operator=(OutputManager&&) -> OutputManager& = delete;

        /**
         * @brief Write a cluster's data as a row of the cluster output
         * @param trial Trial number to which this cluster belongs
         * @param cluster The cluster whose data should be written
         * @details
         * If cluster output was not enabled for this simulation, this
         * is a no-op.
         */
        virtual void writeCluster(unsigned long trial, const core::Cluster& cluster) = 0;

    protected:

        /**
         * @brief Return the current local date (YYYY-MM-DD) and time (HH:MM:SS)
         * @return A pair holding the date string followed by the time string
         */
        static auto currentDateAndTime() -> std::pair<std::string, std::string>;

        /**
         * @brief Return the calling thread's current rng state
         * @return The rng state, as a string suitable for writing to disk
         * so a run can later be reproduced
         */
        static auto currentRngStateString() -> std::string;

        const SimControls& simControls_; /**< Simulation control flow settings */
        const SimPhysics& simPhysics_;   /**< Simulation physics settings */
        const toml::table& inputDeck_;   /**< The simulation's toml input deck */
    };

} // namespace io

#endif // OUTPUTMANAGER_HPP
