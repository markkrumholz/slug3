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
#include "hdf5.h" // NOLINT(misc-include-cleaner)
#include <toml.hpp>

namespace core
{
    class Cluster;
    class Galaxy;
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
         * @param simControls Simulation controls (physics settings and
         *   control-flow settings together)
         * @param inputDeck The simulation's toml input deck
         * @details
         * simControls and inputDeck are stored by reference, so the
         * objects passed in must outlive this OutputManagerH5.
         */
        OutputManagerH5(const SimControls& simControls,
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
            core::Cluster& cluster) override;

        /**
         * @brief Write a cluster's photometry as a row of the cluster_phot datasets
         * @param trial Trial number to which this cluster belongs
         * @param time The output time at which the cluster's photometry was computed, in yr
         * @param cluster The cluster whose photometry should be written
         * @details
         * If no filter collection or bolometric luminosity was
         * requested for this simulation (the cluster_phot group does
         * not exist), or the cluster has disrupted, this is a no-op.
         */
        void writeClusterPhot(unsigned long trial, double time,
            core::Cluster& cluster) override;

        /**
         * @brief Write a galaxy's data as a row of the galaxy datasets
         * @param trial Trial number to which this galaxy belongs
         * @param time The output time at which this row was recorded, in yr
         * @param galaxy The galaxy whose data should be written
         * @details
         * If galaxy output was not enabled for this simulation (the
         * galaxy group does not exist), this is a no-op. Otherwise,
         * after writing this row, also calls writeCluster() on every
         * currently-alive (non-disrupted) cluster in galaxy.
         */
        void writeGalaxy(unsigned long trial, double time,
            core::Galaxy& galaxy) override;

        /**
         * @brief Write a galaxy's spectrum as a row of the galaxy_spectra datasets
         * @param trial Trial number to which this galaxy belongs
         * @param time The output time at which the galaxy's spectrum was computed, in yr
         * @param galaxy The galaxy whose spectrum should be written
         * @details
         * If spectral synthesis was not enabled for this simulation
         * (the galaxy_spectra group does not exist), this is a no-op.
         * Otherwise, after writing this row, also calls
         * writeClusterSpec() on every currently-alive (non-disrupted)
         * cluster in galaxy.
         */
        void writeGalaxySpec(unsigned long trial, double time,
            core::Galaxy& galaxy) override;

        /**
         * @brief Write a galaxy's photometry as a row of the galaxy_phot datasets
         * @param trial Trial number to which this galaxy belongs
         * @param time The output time at which the galaxy's photometry was computed, in yr
         * @param galaxy The galaxy whose photometry should be written
         * @details
         * If no filter collection or bolometric luminosity was
         * requested for this simulation (the galaxy_phot group does
         * not exist), this is a no-op. Otherwise, after writing this
         * row, also calls writeClusterPhot() on every currently-alive
         * (non-disrupted) cluster in galaxy.
         */
        void writeGalaxyPhot(unsigned long trial, double time,
            core::Galaxy& galaxy) override;

    private:

        /**
         * @brief Create the clusters group and its datasets, if cluster output is enabled
         * @details
         * A no-op if output.write_cluster (optional, defaults to true)
         * is set to false.
         */
        void openClustersGroup();

        /**
         * @brief Create the cluster_spectra group and its datasets, if a spectral synthesizer was requested
         * @details
         * A no-op if output.write_cluster_spec is set to false in the
         * input deck (it defaults to true), even if a spectral
         * synthesizer was requested -- spectra can be wanted only as
         * an intermediate for computing photometry, in which case
         * writing them out as well just wastes disk space.
         */
        void openClusterSpectraGroup();

        /**
         * @brief Create the cluster_phot group and its datasets, if a filter collection or the bolometric luminosity was requested
         * @details
         * A no-op if output.write_cluster_phot (optional, defaults to
         * true) is set to false.
         */
        void openClusterPhotGroup();

        /**
         * @brief Create the galaxy group and its datasets, for a galaxy-type simulation
         * @details
         * A no-op unless SimControls::simType() is SimType::galaxy --
         * there is no Galaxy object, and so nothing to write here, for
         * a cluster-type simulation -- or if output.write_galaxy
         * (optional, defaults to true) is set to false.
         */
        void openGalaxyGroup();

        /**
         * @brief Create the galaxy_spectra group and its datasets, if a spectral synthesizer was requested
         * @details
         * A no-op unless SimControls::simType() is SimType::galaxy, if
         * no spectral synthesizer was requested, or if
         * output.write_galaxy_spec (optional, defaults to true) is set
         * to false.
         */
        void openGalaxySpectraGroup();

        /**
         * @brief Create the galaxy_phot group and its datasets, if a filter collection or the bolometric luminosity was requested
         * @details
         * A no-op unless SimControls::simType() is SimType::galaxy, if
         * neither a filter collection nor the bolometric luminosity
         * was requested, or if output.write_galaxy_phot (optional,
         * defaults to true) is set to false.
         */
        void openGalaxyPhotGroup();

        hid_t file_ = -1; /**< Handle to the open HDF5 output file */ // NOLINT(misc-include-cleaner)
        hid_t clustersGroup_ = -1; /**< Handle to the open clusters group, if any */ // NOLINT(misc-include-cleaner)
        hid_t clusterSpectraGroup_ = -1; /**< Handle to the open cluster_spectra group, if any */ // NOLINT(misc-include-cleaner)
        hid_t clusterPhotGroup_ = -1; /**< Handle to the open cluster_phot group, if any */ // NOLINT(misc-include-cleaner)
        hid_t galaxyGroup_ = -1; /**< Handle to the open galaxy group, if any */ // NOLINT(misc-include-cleaner)
        hid_t galaxySpectraGroup_ = -1; /**< Handle to the open galaxy_spectra group, if any */ // NOLINT(misc-include-cleaner)
        hid_t galaxyPhotGroup_ = -1; /**< Handle to the open galaxy_phot group, if any */ // NOLINT(misc-include-cleaner)
    };

} // namespace io

#endif // OUTPUTMANAGERH5_HPP
