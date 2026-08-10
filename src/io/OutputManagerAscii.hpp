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
#include <fstream>
#include <toml.hpp>
#include <vector>

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
         * @param simControls Simulation controls (physics settings and
         *   control-flow settings together)
         * @param inputDeck The simulation's toml input deck
         * @details
         * simControls and inputDeck are stored by reference, so the
         * objects passed in must outlive this OutputManagerAscii.
         */
        OutputManagerAscii(const SimControls& simControls,
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
         * If spectral synthesis was not enabled for this simulation
         * (the cluster-spectra file was not opened), or the cluster
         * has disrupted, this is a no-op. Otherwise writes one line
         * per wavelength, each holding trial, time, uid, wavelength,
         * and specific luminosity, to the cluster-spectra file.
         */
        void writeClusterSpec(unsigned long trial, double time,
            const core::Cluster& cluster) override;

        /**
         * @brief Write a cluster's photometry
         * @param trial Trial number to which this cluster belongs
         * @param time The output time at which the cluster's photometry was computed, in yr
         * @param cluster The cluster whose photometry should be written
         * @details
         * If no filter collection or bolometric luminosity was
         * requested for this simulation (the cluster-photometry file
         * was not opened), or the cluster has disrupted, this is a
         * no-op. Otherwise writes one line, holding trial, time, uid,
         * one column per filter, and (if requested) a final "Lbol"
         * column, to the cluster-photometry file.
         */
        void writeClusterPhot(unsigned long trial, double time,
            const core::Cluster& cluster) override;

    private:

        /**
         * @brief Open the cluster output file and write its header, if cluster output is enabled
         */
        void openClustersFile();

        /**
         * @brief Open the cluster-spectra output file and write its header, if a spectral synthesizer was requested
         * @details
         * A no-op if output.write_cluster_spec is set to false in the
         * input deck (it defaults to true), even if a spectral
         * synthesizer was requested -- spectra can be wanted only as
         * an intermediate for computing photometry, in which case
         * writing them out as well just wastes disk space.
         */
        void openClusterSpectraFile();

        /**
         * @brief Open the cluster-photometry output file and write its header, if a filter collection or the bolometric luminosity was requested
         */
        void openClusterPhotFile();

        /**
         * @brief Open the galaxy output file and write its header, for a galaxy-type simulation
         * @details
         * A no-op unless SimControls::simType() is SimType::galaxy --
         * there is no Galaxy object, and so nothing to write here, for
         * a cluster-type simulation.
         */
        void openGalaxyFile();

        /**
         * @brief Open the galaxy-spectra output file and write its header, if a spectral synthesizer was requested
         * @details
         * A no-op unless SimControls::simType() is SimType::galaxy, or
         * if no spectral synthesizer was requested.
         */
        void openGalaxySpectraFile();

        /**
         * @brief Open the galaxy-photometry output file and write its header, if a filter collection or the bolometric luminosity was requested
         * @details
         * A no-op unless SimControls::simType() is SimType::galaxy, or
         * if neither a filter collection nor the bolometric luminosity
         * was requested.
         */
        void openGalaxyPhotFile();

        std::ofstream clustersFile_; /**< Handle to the open cluster output file */
        std::ofstream clusterSpectraFile_; /**< Handle to the open cluster-spectra output file, if any */
        std::ofstream clusterPhotFile_; /**< Handle to the open cluster-photometry output file, if any */
        std::ofstream galaxyFile_; /**< Handle to the open galaxy output file, if any (galaxy-type simulations only) */
        std::ofstream galaxySpectraFile_; /**< Handle to the open galaxy-spectra output file, if any */
        std::ofstream galaxyPhotFile_; /**< Handle to the open galaxy-photometry output file, if any */
        std::vector<double> wlObs_; /**< Observed-frame wavelength grid, if spectral synthesis is enabled -- shared by both the cluster- and galaxy-spectra files, since both are drawn from the same SimControls::specsyn() */
        std::vector<int> photColWidths_; /**< Column width used for each filter in the cluster- and galaxy-photometry files -- see computePhotColWidths() */
        std::vector<int> photExtinctColWidths_; /**< Column width used for each "<filter>_ex" column in the cluster- and galaxy-photometry files, if SimControls::extinct() is set -- see computePhotColWidths() */
    };

} // namespace io

#endif // OUTPUTMANAGERASCII_HPP
