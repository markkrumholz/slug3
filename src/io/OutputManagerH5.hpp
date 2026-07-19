/**
 * @file OutputManagerH5.hpp
 * @author Mark Krumholz
 * @brief HDF5-output specialization of OutputManager
 * @date 2026-07-17
 */

#ifndef OUTPUTMANAGERH5_HPP
#define OUTPUTMANAGERH5_HPP

#include "OutputManager.hpp"
#include "SimControls.hpp"
#include "SimPhysics.hpp"
#include "hdf5.h" // NOLINT(misc-include-cleaner)
#include <toml.hpp>

namespace core
{
    class Cluster;
} // namespace core

namespace io
{

    /**
     * @class OutputManagerH5
     * @brief HDF5-output specialization of OutputManager
     */
    class OutputManagerH5 : public OutputManager
    {
    public:

        /**
         * @brief Open the output file and write its header
         * @param simControls Simulation control flow settings
         * @param simPhysics Simulation physics settings
         * @param inputDeck The simulation's toml input deck
         * @details
         * simControls, simPhysics, and inputDeck are stored by
         * reference, so the objects passed in must outlive this
         * OutputManagerH5.
         */
        OutputManagerH5(const SimControls& simControls, const SimPhysics& simPhysics,
            const toml::table& inputDeck);

        /**
         * @brief Close the output file
         */
        ~OutputManagerH5() override;

        OutputManagerH5(const OutputManagerH5&) = delete;
        auto operator=(const OutputManagerH5&) -> OutputManagerH5& = delete;
        OutputManagerH5(OutputManagerH5&&) = delete;
        auto operator=(OutputManagerH5&&) -> OutputManagerH5& = delete;

        /**
         * @brief Write a cluster's data as a row of the clusters datasets
         * @param trial Trial number to which this cluster belongs
         * @param cluster The cluster whose data should be written
         * @details
         * If cluster output was not enabled for this simulation, this
         * is a no-op.
         */
        void writeCluster(unsigned long trial, const core::Cluster& cluster) override;

        /**
         * @brief Write a cluster's spectrum as a row of the cluster-spectra datasets
         * @param trial Trial number to which this cluster belongs
         * @param time The output time at which the cluster's spectrum was computed, in yr
         * @param cluster The cluster whose spectrum should be written
         * @details
         * If spectral synthesis was not enabled for this simulation
         * (the cluster_spectra group does not exist), or the cluster
         * has disrupted, this is a no-op.
         */
        void writeClusterSpec(unsigned long trial, double time,
            const core::Cluster& cluster) override;

    private:

        /**
         * @brief Create the clusters group and its datasets, if cluster output is enabled
         */
        void openClustersGroup();

        /**
         * @brief Create the cluster_spectra group and its datasets, if a spectral synthesizer was requested
         */
        void openClusterSpectraGroup();

        hid_t file_ = -1; /**< Handle to the open HDF5 output file */ // NOLINT(misc-include-cleaner)
        hid_t clustersGroup_ = -1; /**< Handle to the open clusters group, if any */ // NOLINT(misc-include-cleaner)
        hid_t clusterSpectraGroup_ = -1; /**< Handle to the open cluster_spectra group, if any */ // NOLINT(misc-include-cleaner)
    };

} // namespace io

#endif // OUTPUTMANAGERH5_HPP
