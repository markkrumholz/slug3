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
         * @brief Cache a reference to the simulation controls, and sanity-check the output-control flags
         * @param simControls Simulation controls (physics settings and
         *   control-flow settings together)
         * @details
         * simControls is stored by reference, so the object passed in
         * must outlive this OutputManager. This base constructor does
         * not open any output files; that is left to the constructors
         * of the format-specific subclasses.
         *
         * Unlike simControls itself, this class no longer needs the
         * input deck at all: the six output.write_cluster/
         * write_cluster_spec/write_cluster_phot/write_galaxy/
         * write_galaxy_spec/write_galaxy_phot flags the subclass
         * constructors gate their own group/file creation on are
         * simControls's own writeCluster()/writeClusterSpec()/
         * writeClusterPhot()/writeGalaxy()/writeGalaxySpec()/
         * writeGalaxyPhot() (parsed by SimControls itself -- see its
         * own readOutput()), and the deck text a subclass constructor
         * records for provenance is simControls's own inputDeckStr() --
         * so this constructor only sanity-checks the write_* flags'
         * own resulting combination:
         * @throws std::runtime_error if every output relevant to this
         *   simulation's own SimType (all six for a galaxy-type
         *   simulation; just writeCluster()/writeClusterSpec()/
         *   writeClusterPhot() for a cluster-type simulation, since
         *   write_galaxy* is meaningless when there is no Galaxy object
         *   at all) is false, since nothing would be written
         * @throws std::runtime_error if writeClusterPhot() and
         *   writeGalaxyPhot() are both false while SimControls::filters()
         *   is non-null, since that means photometry was requested but
         *   would never be written anywhere -- almost certainly a
         *   mistake, rather than a deliberate "compute filters() for
         *   some other purpose but discard the result" request
         */
        explicit OutputManager(const SimControls& simControls);

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

        /**
         * @brief Return the largest trial number the run being restarted ever actually wrote
         * @return The highest trial number the run being restarted had
         *   actually written output for, as of the most recent
         *   checkpoint it left behind -- see
         *   OutputManagerH5::restartSetup()'s own comment for how that
         *   is determined
         * @details
         * Only meaningful under the same conditions as
         * restartTrialsDone() (only after a restart-constructed
         * OutputManager, only with HDF5 output -- OutputManagerAscii's
         * own implementation throws for the identical reason).
         *
         * Deliberately distinct from restartTrialsDone(), even though
         * both describe "how far did the run being restarted get":
         * under dynamic OpenMP scheduling, a batch that stopped
         * partway through (a caught SIGTERM, or a per-trial exception)
         * can finish with some higher-numbered trial done while a
         * lower-numbered one is not (whichever thread happened to
         * still be mid-flight when the others stopped taking on new
         * work) -- restartTrialsDone() (a plain count of how many
         * trials finished) stays accurate even then, but "trials
         * [0, restartTrialsDone()) are done" stops being a safe
         * assumption once that can happen, so a restart cannot safely
         * resume trial *numbering* from restartTrialsDone() alone. It
         * can always safely resume at restartMaxTrial() + 1, since
         * that is defined as the largest number ever actually written
         * -- guaranteed never to collide with one already on disk --
         * independently of restartTrialsDone() itself, which still
         * correctly says how many *more* trials are needed to reach
         * SimControls::nTrial(). SimCluster::run()'s/SimGalaxy::run()'s
         * own comment has the full detail of how the two combine.
         */
        [[nodiscard]] virtual auto restartMaxTrial() const -> unsigned long = 0;

        /**
         * @brief Record that this run is ending early, having only actually completed trialsCompleted trials
         * @param trialsCompleted Number of trials actually completed
         *   before this run stopped short of SimControls::nTrial() --
         *   either because a per-trial exception was thrown and
         *   rethrown out of SimCluster::run()/SimGalaxy::run(), or
         *   because a SIGTERM was caught and handled gracefully (see
         *   their own comments)
         * @details
         * Does not itself close or otherwise touch any output --
         * OutputManagerH5's own implementation just remembers
         * trialsCompleted, for its destructor to write as the
         * currently-open file's own final "trials_completed" (and,
         * where relevant, "restart_uid") attribute instead of
         * SimControls::nTrial(), which the destructor would otherwise
         * assume (see OutputManagerH5::closeOutputFile()'s own
         * comment) -- an assumption only ever true when run() actually
         * completed every trial, which stopping early specifically
         * means it did not. Deliberately does not roll over to a new
         * checkpoint the way checkpoint() does: there is nothing left
         * to write into one, so doing so would just leave an empty,
         * never-closed checkpoint behind for the destructor to
         * eventually close with a wrong trial count of its own.
         *
         * OutputManagerAscii's own implementation is a no-op, not a
         * throw (unlike checkpoint()/restartTrialsDone()): ascii
         * output has no trials_completed attribute (or any other
         * summary of how many trials it holds) to correct in the
         * first place, so there is nothing wrong to leave uncorrected
         * -- every row already written is already exactly as valid on
         * an early exit as a normal one.
         */
        virtual void notifyEarlyTermination(unsigned long trialsCompleted) = 0;

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
    };

} // namespace io

#endif // OUTPUTMANAGER_HPP
