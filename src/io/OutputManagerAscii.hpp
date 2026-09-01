/**
 * @file OutputManagerAscii.hpp
 * @author Mark Krumholz
 * @brief Ascii-output specialization of OutputManager
 * @date 2026-07-17
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
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
    class Galaxy;
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
         * and specific luminosity (plus, if requested, extincted
         * and/or nebular-inclusive specific luminosity), to the
         * cluster-spectra file, then calls writeClusterNebLines() to do
         * the same for each nebular emission line's own luminosity, if
         * a nebular emission grid was requested.
         */
        void writeClusterSpec(unsigned long trial, double time,
            core::Cluster& cluster) override;

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
         * column, followed by extincted and/or nebular-inclusive
         * filter columns if requested, to the cluster-photometry file.
         */
        void writeClusterPhot(unsigned long trial, double time,
            core::Cluster& cluster) override;

        /**
         * @brief Write a galaxy's data as a row of the galaxy output file
         * @param trial Trial number to which this galaxy belongs
         * @param time The output time at which this row was recorded, in yr
         * @param galaxy The galaxy whose data should be written
         * @details
         * If galaxy output was not enabled for this simulation (the
         * galaxy file was not opened), this is a no-op. Otherwise,
         * after writing this row, also calls writeCluster() on every
         * currently-alive (non-disrupted) cluster in galaxy.
         */
        void writeGalaxy(unsigned long trial, double time,
            core::Galaxy& galaxy) override;

        /**
         * @brief Write a galaxy's spectrum
         * @param trial Trial number to which this galaxy belongs
         * @param time The output time at which the galaxy's spectrum was computed, in yr
         * @param galaxy The galaxy whose spectrum should be written
         * @details
         * If spectral synthesis was not enabled for this simulation
         * (the galaxy-spectra file was not opened), this is a no-op.
         * Otherwise writes one line per wavelength, each holding
         * trial, time, wavelength, and specific luminosity (plus, if
         * requested, extincted and/or nebular-inclusive specific
         * luminosity), to the galaxy-spectra file, then calls
         * writeGalaxyNebLines() to do the same for each nebular
         * emission line's own luminosity, if a nebular emission grid
         * was requested, then calls writeClusterSpec() on every
         * currently-alive (non-disrupted) cluster in galaxy.
         */
        void writeGalaxySpec(unsigned long trial, double time,
            core::Galaxy& galaxy) override;

        /**
         * @brief Write a galaxy's photometry
         * @param trial Trial number to which this galaxy belongs
         * @param time The output time at which the galaxy's photometry was computed, in yr
         * @param galaxy The galaxy whose photometry should be written
         * @details
         * If no filter collection or bolometric luminosity was
         * requested for this simulation (the galaxy-photometry file
         * was not opened), this is a no-op. Otherwise writes one
         * line, holding trial, time, one column per filter, and (if
         * requested) a final "Lbol" column, followed by extincted
         * and/or nebular-inclusive filter columns if requested, to the
         * galaxy-photometry file, then calls writeClusterPhot() on
         * every currently-alive (non-disrupted) cluster in galaxy.
         */
        void writeGalaxyPhot(unsigned long trial, double time,
            core::Galaxy& galaxy) override;

        /**
         * @brief Not supported for ascii output
         * @throws std::runtime_error always
         * @details
         * Ascii output has no way to reopen/append to a file it has
         * already finished writing the way OutputManagerH5 rolls over
         * to a new HDF5 file, so checkpointing is only ever supported
         * with HDF5 output -- see OutputManager::checkpoint()'s own
         * comment for why this still needs to exist (and fail loudly)
         * rather than simply not being overridden here at all.
         */
        void checkpoint() override;

    private:

        /**
         * @brief Open the cluster output file and write its header, if cluster output is enabled
         * @details
         * A no-op if output.write_cluster (optional, defaults to true)
         * is set to false.
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
         * @details
         * A no-op if output.write_cluster_phot (optional, defaults to
         * true) is set to false.
         */
        void openClusterPhotFile();

        /**
         * @brief Open the galaxy output file and write its header, for a galaxy-type simulation
         * @details
         * A no-op unless SimControls::simType() is SimType::galaxy --
         * there is no Galaxy object, and so nothing to write here, for
         * a cluster-type simulation -- or if output.write_galaxy
         * (optional, defaults to true) is set to false.
         */
        void openGalaxyFile();

        /**
         * @brief Open the galaxy-spectra output file and write its header, if a spectral synthesizer was requested
         * @details
         * A no-op unless SimControls::simType() is SimType::galaxy, if
         * no spectral synthesizer was requested, or if
         * output.write_galaxy_spec (optional, defaults to true) is set
         * to false.
         */
        void openGalaxySpectraFile();

        /**
         * @brief Open the galaxy-photometry output file and write its header, if a filter collection or the bolometric luminosity was requested
         * @details
         * A no-op unless SimControls::simType() is SimType::galaxy, if
         * neither a filter collection nor the bolometric luminosity
         * was requested, or if output.write_galaxy_phot (optional,
         * defaults to true) is set to false.
         */
        void openGalaxyPhotFile();

        /**
         * @brief Open the cluster-nebular-line-luminosity output file and write its header, if a nebular emission grid was requested
         * @details
         * A no-op if no spectral synthesizer was requested (mirroring
         * openClusterSpectraFile()'s own gate -- a nebular emission
         * grid can be built independently of a spectral synthesizer,
         * but Cluster never computes any line luminosities without
         * one), or if output.write_cluster_spec (optional, defaults to
         * true) is set to false, since this file is a sibling of the
         * cluster-spectra file, gated the same way.
         */
        void openClusterNebLinesFile();

        /**
         * @brief Open the galaxy-nebular-line-luminosity output file and write its header, if a nebular emission grid was requested
         * @details
         * A no-op unless SimControls::simType() is SimType::galaxy, if
         * no spectral synthesizer was requested, or if
         * output.write_galaxy_spec (optional, defaults to true) is set
         * to false -- see openClusterNebLinesFile()'s own comment.
         */
        void openGalaxyNebLinesFile();

        /**
         * @brief Write one row per nebular emission line for a cluster's spectrum
         * @param trial Trial number to which this cluster belongs
         * @param time The output time at which the cluster's spectrum was computed, in yr
         * @param cluster The cluster whose line luminosities should be written
         * @details
         * A no-op if the cluster-nebular-line-luminosity file was not
         * opened (see openClusterNebLinesFile()). Otherwise writes one
         * line per SimControls::nebular() line, each holding trial,
         * time, uid, line label, line wavelength, and line luminosity
         * (plus, if requested, the extincted line luminosity), to the
         * cluster-nebular-line-luminosity file. Called from
         * writeClusterSpec(), after it finishes writing the cluster's
         * own spectrum.
         */
        void writeClusterNebLines(unsigned long trial, double time, core::Cluster& cluster);

        /**
         * @brief Write one row per nebular emission line for a galaxy's spectrum
         * @param trial Trial number to which this galaxy belongs
         * @param time The output time at which the galaxy's spectrum was computed, in yr
         * @param galaxy The galaxy whose line luminosities should be written
         * @details
         * A no-op if the galaxy-nebular-line-luminosity file was not
         * opened (see openGalaxyNebLinesFile()). Otherwise writes one
         * line per SimControls::nebular() line, each holding trial,
         * time, line label, line wavelength, and line luminosity
         * (plus, if requested, the extincted line luminosity), to the
         * galaxy-nebular-line-luminosity file. Called from
         * writeGalaxySpec(), after it finishes writing the galaxy's
         * own spectrum.
         */
        void writeGalaxyNebLines(unsigned long trial, double time, core::Galaxy& galaxy);

        std::ofstream clustersFile_; /**< Handle to the open cluster output file */
        std::ofstream clusterSpectraFile_; /**< Handle to the open cluster-spectra output file, if any */
        std::ofstream clusterPhotFile_; /**< Handle to the open cluster-photometry output file, if any */
        std::ofstream clusterNebLinesFile_; /**< Handle to the open cluster-nebular-line-luminosity output file, if any */
        std::ofstream galaxyFile_; /**< Handle to the open galaxy output file, if any (galaxy-type simulations only) */
        std::ofstream galaxySpectraFile_; /**< Handle to the open galaxy-spectra output file, if any */
        std::ofstream galaxyPhotFile_; /**< Handle to the open galaxy-photometry output file, if any */
        std::ofstream galaxyNebLinesFile_; /**< Handle to the open galaxy-nebular-line-luminosity output file, if any */
        std::vector<double> wlObs_; /**< Observed-frame wavelength grid, if spectral synthesis is enabled -- shared by both the cluster- and galaxy-spectra files, since both are drawn from the same SimControls::specsyn() */
        std::vector<int> photColWidths_; /**< Column width used for each filter in the cluster- and galaxy-photometry files -- see computePhotColWidths() */
        std::vector<int> photExtinctColWidths_; /**< Column width used for each "<filter>_ex" column in the cluster- and galaxy-photometry files, if SimControls::extinct() is set -- see computePhotColWidths() */
        std::vector<int> photNebColWidths_; /**< Column width used for each "<filter>_neb" column in the cluster- and galaxy-photometry files, if SimControls::nebular() is set -- see computePhotColWidths() */
        std::vector<int> photNebExtinctColWidths_; /**< Column width used for each "<filter>_neb_ex" column in the cluster- and galaxy-photometry files, if both SimControls::nebular() and SimControls::extinct() are set -- see computePhotColWidths() */
        int lineLabelWidth_ = 0; /**< Column width used for the "line_label" column in the cluster- and galaxy-nebular-line-luminosity files, if SimControls::nebular() is set -- see computeLineLabelWidth() */
    };

} // namespace io

#endif // OUTPUTMANAGERASCII_HPP
