/**
 * @file OutputManager.hpp
 * @author Mark Krumholz
 * @brief A class to manage writing simulation outputs to disk
 * @date 2026-07-16
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
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
         * @brief Cache references to the simulation controls and input deck, and parse the output-control keys
         * @param simControls Simulation controls (physics settings and
         *   control-flow settings together)
         * @param inputDeck The simulation's toml input deck
         * @details
         * simControls and inputDeck are stored by reference, so the
         * objects passed in must outlive this OutputManager. This base
         * constructor does not open any output files; that is left to
         * the constructors of the format-specific subclasses.
         *
         * Also reads the six optional output.write_cluster/
         * write_cluster_spec/write_cluster_phot/write_galaxy/
         * write_galaxy_spec/write_galaxy_phot keys (each defaulting to
         * true) into writeCluster_/writeClusterSpec_/writeClusterPhot_/
         * writeGalaxy_/writeGalaxySpec_/writeGalaxyPhot_, for the
         * subclass constructors to gate their own group/file creation
         * on, then sanity-checks the resulting combination:
         * @throws std::runtime_error if every output relevant to this
         *   simulation's own SimType (all six for a galaxy-type
         *   simulation; just writeCluster_/writeClusterSpec_/
         *   writeClusterPhot_ for a cluster-type simulation, since
         *   write_galaxy* is meaningless when there is no Galaxy object
         *   at all) is false, since nothing would be written
         * @throws std::runtime_error if writeClusterPhot_ and
         *   writeGalaxyPhot_ are both false while SimControls::filters()
         *   is non-null, since that means photometry was requested but
         *   would never be written anywhere -- almost certainly a
         *   mistake, rather than a deliberate "compute filters() for
         *   some other purpose but discard the result" request
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
         * @param cluster The cluster whose spectrum should be written;
         *   not const, since this may need to lazily compute the
         *   spectrum if it is not already current (see
         *   core::Cluster::spec()'s own comment)
         * @details
         * If spectral synthesis was not enabled for this simulation,
         * or the cluster has disrupted, this is a no-op.
         */
        virtual void writeClusterSpec(unsigned long trial, double time,
            core::Cluster& cluster) = 0;

        /**
         * @brief Write a cluster's photometry as a row of the cluster-photometry datasets
         * @param trial Trial number to which this cluster belongs
         * @param time The output time at which the cluster's photometry was computed, in yr
         * @param cluster The cluster whose photometry should be
         *   written; not const -- see writeClusterSpec()'s own comment
         * @details
         * If no filter collection or bolometric luminosity was
         * requested for this simulation, or the cluster has
         * disrupted, this is a no-op.
         */
        virtual void writeClusterPhot(unsigned long trial, double time,
            core::Cluster& cluster) = 0;

        /**
         * @brief Write a galaxy's data as a row of the galaxy output
         * @param trial Trial number to which this galaxy belongs
         * @param time The output time at which this row was recorded, in yr
         * @param galaxy The galaxy whose data should be written; not
         *   const, since this needs a non-const reference to loop over
         *   galaxy's own clusters (see core::Galaxy::clusters()'s own
         *   comment)
         * @details
         * If galaxy output was not enabled for this simulation (only
         * possible for a galaxy-type simulation to begin with), this
         * is a no-op. Otherwise, after writing this row, also calls
         * writeCluster() on every currently-alive (non-disrupted)
         * cluster in galaxy, so that each individual cluster is
         * recorded in the clusters output too.
         */
        virtual void writeGalaxy(unsigned long trial, double time,
            core::Galaxy& galaxy) = 0;

        /**
         * @brief Write a galaxy's spectrum as a row of the galaxy-spectra datasets
         * @param trial Trial number to which this galaxy belongs
         * @param time The output time at which the galaxy's spectrum was computed, in yr
         * @param galaxy The galaxy whose spectrum should be written;
         *   not const -- see writeGalaxy()'s own comment
         * @details
         * If spectral synthesis was not enabled for this simulation,
         * this is a no-op. Otherwise, after writing this row, also
         * calls writeClusterSpec() on every currently-alive
         * (non-disrupted) cluster in galaxy.
         */
        virtual void writeGalaxySpec(unsigned long trial, double time,
            core::Galaxy& galaxy) = 0;

        /**
         * @brief Write a galaxy's photometry as a row of the galaxy-photometry datasets
         * @param trial Trial number to which this galaxy belongs
         * @param time The output time at which the galaxy's photometry was computed, in yr
         * @param galaxy The galaxy whose photometry should be written;
         *   not const -- see writeGalaxy()'s own comment
         * @details
         * If no filter collection or bolometric luminosity was
         * requested for this simulation, this is a no-op. Otherwise,
         * after writing this row, also calls writeClusterPhot() on
         * every currently-alive (non-disrupted) cluster in galaxy.
         */
        virtual void writeGalaxyPhot(unsigned long trial, double time,
            core::Galaxy& galaxy) = 0;

        /**
         * @brief Roll over to a new checkpoint
         * @param trialsCompleted Number of trials completed so far in
         *   the run, i.e. across every checkpoint including the one
         *   just closed, not just this checkpoint's own share of them
         * @details
         * Closes whatever output is currently open -- writing
         * trialsCompleted as a top-level attribute on it first, so a
         * later restart can tell how far the run had gotten as of
         * this checkpoint -- and opens a fresh one for the next
         * checkpoint, so the current checkpoint's output is fully
         * flushed to disk before more trials are written to it --
         * bounding how much work a crash or a walltime kill can lose,
         * at the cost of one more output file to eventually
         * consolidate. See OutputManagerH5::checkpoint() for the real
         * implementation, and its own class header comment for the
         * full checkpointing design.
         *
         * Only meaningful with HDF5 output. OutputManagerAscii's own
         * implementation just throws (see its own comment):
         * SimControls's own constructor already rejects the
         * combination of ascii output and a non-zero
         * checkpointInterval() up front when parsing a real input
         * deck, so this should never actually be reached from the
         * CLI -- but a caller building a SimControls directly (e.g.
         * from Python) could set up that same illegal combination via
         * setCheckpointInterval() without going through the
         * constructor's own check, so this still needs to fail safely
         * if it is ever actually called.
         */
        virtual void checkpoint(unsigned long trialsCompleted) = 0;

        /**
         * @brief Return the number of trials already completed by the run being restarted
         * @return The number of trials the run being restarted had
         *   already completed, as of the most recent checkpoint it
         *   left behind -- see OutputManagerH5::restartSetup()'s own
         *   comment for how that is determined
         * @details
         * Only meaningful when this OutputManager was itself
         * constructed to resume a previous, interrupted run; a caller
         * that did not request that (see OutputManagerH5's own
         * constructor) should never call this.
         *
         * Only meaningful with HDF5 output, for the same reason
         * checkpoint() is: restarting requires a checkpoint to restart
         * from, and only HDF5 output produces checkpoints at all.
         * OutputManagerAscii's own implementation just throws (see its
         * own comment): main.cpp already rejects the combination of
         * ascii output and a restart request before ever constructing
         * SimCluster/SimGalaxy (and, transitively, before either of
         * them could call this), so this should never actually be
         * reached from the CLI -- but, exactly as with checkpoint(),
         * still needs to fail safely if it is ever actually called by
         * a caller building these objects directly.
         */
        [[nodiscard]] virtual auto restartTrialsDone() const -> unsigned long = 0;

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

        // Output-control flags, parsed from the input deck's
        // output.write_cluster/write_cluster_spec/write_cluster_phot/
        // write_galaxy/write_galaxy_spec/write_galaxy_phot keys (each
        // optional, defaulting to true) by this base class's own
        // constructor -- see its own comment. Read by the H5/Ascii
        // subclass constructors to decide which groups/files to create
        // at all.
        bool writeCluster_ = true;      /**< Whether to write the clusters group/file */
        bool writeClusterSpec_ = true;  /**< Whether to write the cluster_spectra group/file */
        bool writeClusterPhot_ = true;  /**< Whether to write the cluster_phot group/file */
        bool writeGalaxy_ = true;       /**< Whether to write the galaxy group/file (galaxy-type simulations only) */
        bool writeGalaxySpec_ = true;   /**< Whether to write the galaxy_spectra group/file (galaxy-type simulations only) */
        bool writeGalaxyPhot_ = true;   /**< Whether to write the galaxy_phot group/file (galaxy-type simulations only) */
    };

} // namespace io

#endif // OUTPUTMANAGER_HPP
