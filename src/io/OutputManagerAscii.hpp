/**
 * @file OutputManagerAscii.hpp
 * @author Mark Krumholz
 * @brief Ascii-output specialization of OutputManager
 * @date 2026-07-17
 */

#ifndef OUTPUTMANAGERASCII_HPP
#define OUTPUTMANAGERASCII_HPP

#include "OutputManager.hpp"
#include "SimControls.hpp"
#include "SimPhysics.hpp"
#include <fstream>
#include <toml.hpp>

namespace core
{
    class Cluster;
} // namespace core

namespace io
{

    /**
     * @class OutputManagerAscii
     * @brief Ascii-output specialization of OutputManager
     */
    class OutputManagerAscii : public OutputManager
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
         * OutputManagerAscii.
         */
        OutputManagerAscii(const SimControls& simControls, const SimPhysics& simPhysics,
            const toml::table& inputDeck);

        /**
         * @brief Close the cluster output file, if it was opened
         */
        ~OutputManagerAscii() override;

        OutputManagerAscii(const OutputManagerAscii&) = delete;
        auto operator=(const OutputManagerAscii&) -> OutputManagerAscii& = delete;
        OutputManagerAscii(OutputManagerAscii&&) = delete;
        auto operator=(OutputManagerAscii&&) -> OutputManagerAscii& = delete;

        /**
         * @brief Write a cluster's data as a row of the cluster output file
         * @param trial Trial number to which this cluster belongs
         * @param cluster The cluster whose data should be written
         * @details
         * If cluster output was not enabled for this simulation, this
         * is a no-op.
         */
        void writeCluster(unsigned long trial, const core::Cluster& cluster) override;

        /**
         * @brief Write a cluster's spectrum
         * @param trial Trial number to which this cluster belongs
         * @param time The output time at which the cluster's spectrum was computed, in yr
         * @param cluster The cluster whose spectrum should be written
         * @details
         * Not yet implemented; ascii cluster-spectrum output will be
         * added in a future commit.
         */
        void writeClusterSpec(unsigned long trial, double time,
            const core::Cluster& cluster) override;

    private:

        std::ofstream clustersFile_; /**< Handle to the open cluster output file */
    };

} // namespace io

#endif // OUTPUTMANAGERASCII_HPP
