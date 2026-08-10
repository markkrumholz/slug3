/**
 * @file OutputManager.hpp
 * @author Mark Krumholz
 * @brief A class to manage writing simulation outputs to disk
 * @date 2026-07-16
 */

#ifndef OUTPUTMANAGER_HPP
#define OUTPUTMANAGER_HPP

#include "SimControls.hpp"
#include <string>
#include <toml.hpp>
#include <utility>

namespace core
{
    class Cluster;
    class Galaxy;
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
         * @brief Cache references to the simulation controls and input deck
         * @param simControls Simulation controls (physics settings and
         *   control-flow settings together)
         * @param inputDeck The simulation's toml input deck
         * @details
         * simControls and inputDeck are stored by reference, so the
         * objects passed in must outlive this OutputManager. This base
         * constructor does not open any output files; that is left to
         * the constructors of the format-specific subclasses.
         */
        OutputManager(const SimControls& simControls, const toml::table& inputDeck);

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

        /**
         * @brief Write a cluster's spectrum as a row of the cluster-spectra datasets
         * @param trial Trial number to which this cluster belongs
         * @param time The output time at which the cluster's spectrum was computed, in yr
         * @param cluster The cluster whose spectrum should be written
         * @details
         * If spectral synthesis was not enabled for this simulation,
         * or the cluster has disrupted, this is a no-op.
         */
        virtual void writeClusterSpec(unsigned long trial, double time,
            const core::Cluster& cluster) = 0;

        /**
         * @brief Write a cluster's photometry as a row of the cluster-photometry datasets
         * @param trial Trial number to which this cluster belongs
         * @param time The output time at which the cluster's photometry was computed, in yr
         * @param cluster The cluster whose photometry should be written
         * @details
         * If no filter collection or bolometric luminosity was
         * requested for this simulation, or the cluster has
         * disrupted, this is a no-op.
         */
        virtual void writeClusterPhot(unsigned long trial, double time,
            const core::Cluster& cluster) = 0;

        /**
         * @brief Write a galaxy's data as a row of the galaxy output
         * @param trial Trial number to which this galaxy belongs
         * @param time The output time at which this row was recorded, in yr
         * @param galaxy The galaxy whose data should be written
         * @details
         * If galaxy output was not enabled for this simulation (only
         * possible for a galaxy-type simulation to begin with), this
         * is a no-op. Otherwise, after writing this row, also calls
         * writeCluster() on every currently-alive (non-disrupted)
         * cluster in galaxy, so that each individual cluster is
         * recorded in the clusters output too.
         */
        virtual void writeGalaxy(unsigned long trial, double time,
            const core::Galaxy& galaxy) = 0;

        /**
         * @brief Write a galaxy's spectrum as a row of the galaxy-spectra datasets
         * @param trial Trial number to which this galaxy belongs
         * @param time The output time at which the galaxy's spectrum was computed, in yr
         * @param galaxy The galaxy whose spectrum should be written
         * @details
         * If spectral synthesis was not enabled for this simulation,
         * this is a no-op. Otherwise, after writing this row, also
         * calls writeClusterSpec() on every currently-alive
         * (non-disrupted) cluster in galaxy.
         */
        virtual void writeGalaxySpec(unsigned long trial, double time,
            const core::Galaxy& galaxy) = 0;

        /**
         * @brief Write a galaxy's photometry as a row of the galaxy-photometry datasets
         * @param trial Trial number to which this galaxy belongs
         * @param time The output time at which the galaxy's photometry was computed, in yr
         * @param galaxy The galaxy whose photometry should be written
         * @details
         * If no filter collection or bolometric luminosity was
         * requested for this simulation, this is a no-op. Otherwise,
         * after writing this row, also calls writeClusterPhot() on
         * every currently-alive (non-disrupted) cluster in galaxy.
         */
        virtual void writeGalaxyPhot(unsigned long trial, double time,
            const core::Galaxy& galaxy) = 0;

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

        const SimControls& simControls_; /**< Simulation controls (physics and control-flow settings) */
        const toml::table& inputDeck_;   /**< The simulation's toml input deck */
    };

} // namespace io

#endif // OUTPUTMANAGER_HPP
