/**
 * @file OutputManagerH5.hpp
 * @author Mark Krumholz
 * @brief HDF5-output specialization of OutputManager
 * @date 2026-07-17
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#ifndef OUTPUTMANAGERH5_HPP
#define OUTPUTMANAGERH5_HPP

#include "../utils/ThreadVec.hpp"
#include "OutputManager.hpp"
#include "SimControls.hpp"
#include "hdf5.h" // NOLINT(misc-include-cleaner)
#include <filesystem>
#include <optional>
#include <string>
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
     * @details
     * Unless the linked HDF5 build was itself configured with its
     * (opt-in) thread-safety support -- which is not something this
     * codebase can assume of every HDF5 install it might be built
     * against -- HDF5 is not safe to call from more than one thread at
     * a time AT ALL, even when two threads only ever touch entirely
     * separate files/objects and every individual call is otherwise
     * guarded by a mutex/critical section of its own: the library's
     * own internal global state (ID tables, allocators, etc.) is not
     * itself protected, so genuinely concurrent entry from two threads
     * can corrupt it regardless of what each call actually touches.
     * So, every method here that calls into HDF5 -- openOutputFile(),
     * closeOutputFile(), and every write* method -- shares a single,
     * global OpenMP critical section (see openOutputFile()'s own
     * comment), fully serializing entry into the library across every
     * thread, not just concurrent access to a shared file.
     *
     * On top of that global serialization, when built with OpenMP each
     * thread also gets its own private HDF5 file
     * (outDir/modelName/thread_NNNN.h5, one per OpenMP thread) rather
     * than sharing one -- this doesn't help with the concurrent-entry
     * problem above (that's why the critical section exists regardless),
     * but does still avoid a different class of problem: interleaved
     * mutations to one file's own internal metadata cache (B-tree
     * nodes, free-space managers, ...) from calls made by different
     * physical OS threads, which serializing calls in time does not,
     * by itself, guarantee is safe. By default
     * (SimControls::OutputMode::h5), those per-thread files are merged
     * back into a single outDir/modelName.h5 -- identical in format to
     * what this class always produced before, except that row order
     * within each dataset is no longer guaranteed, since threads may
     * process trials in any order -- once the run completes; see
     * consolidateFiles()'s own comment. SimControls::OutputMode::
     * h5divided instead leaves the per-thread files as-is, skipping
     * consolidation, for very large runs where merging into one file
     * would itself be costly. Without OpenMP, there is only ever one
     * thread, so this class behaves exactly as it always did: a single
     * outDir/modelName.h5, no per-thread directory, no consolidation
     * step (and the critical sections are all no-ops).
     */
    class OutputManagerH5 : public OutputManager
    {
    public:

        /**
         * @brief Open the output file(s) and write their header
         * @param simControls Simulation controls (physics settings and
         *   control-flow settings together)
         * @param inputDeck The simulation's toml input deck
         * @param restart Whether this run is resuming a previous,
         *   interrupted run from its most recent checkpoint (see
         *   restartSetup()'s own comment); defaults to false
         * @throws std::runtime_error if restart is true and
         *   simControls.outputMode() is OutputMode::ascii --
         *   restarting is only ever supported with HDF5 output. In
         *   practice this can only be reached by a caller building an
         *   OutputManagerH5 directly (e.g. from Python) with an ascii
         *   SimControls, since the CLI itself (main.cpp) checks for
         *   this same illegal combination before ever constructing an
         *   OutputManagerH5, and always builds an OutputManagerAscii
         *   instead when outputMode() is ascii -- kept here anyway as
         *   a second line of defense, mirroring the same
         *   defense-in-depth pattern SimControls::initControlFlow()'s
         *   own checkpointInterval()/ascii check and
         *   OutputManagerAscii::checkpoint()'s own throw already use.
         * @details
         * simControls and inputDeck are stored by reference, so the
         * objects passed in must outlive this OutputManagerH5. See
         * this class's own header comment for what "the output
         * file(s)" means when built with OpenMP.
         *
         * If restart is true, calls restartSetup() before
         * openNewOutputFiles(), so checkpointNumber_ (and, via it,
         * openNewOutputFiles()'s own choice of output path) already
         * reflects where the run being restarted left off by the time
         * the first file is actually opened.
         */
        OutputManagerH5(const SimControls& simControls,
            const toml::table& inputDeck, bool restart = false);

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

        /**
         * @brief Roll over to a new checkpoint
         * @param trialsCompleted Number of trials completed so far in
         *   the run, across every checkpoint including the one just
         *   closed -- see closeOutputFile()'s own comment for what
         *   this is used for
         * @details
         * Only meaningful if SimControls::checkpointInterval() is
         * non-zero. Closes every thread's own currently-open output
         * file (closeOutputFile()), increments checkpointNumber_, then
         * opens a fresh one for the new checkpoint for every thread
         * (openNewOutputFiles()) -- so the checkpoint just finished is
         * fully flushed to disk (closing an HDF5 file forces this)
         * before any more trials are written, bounding how much work a
         * crash or a walltime kill can lose to at most one
         * checkpoint's worth. consolidateFiles() is deliberately not
         * called here (only at the very end, from the destructor, once
         * per checkpoint -- see its own comment): consolidating after
         * every single checkpoint would slow down the run itself for
         * no benefit, since nothing needs the checkpoints merged until
         * the whole run is over.
         *
         * Requires the caller to already have ensured that no thread
         * is concurrently writing output when this runs -- specifically,
         * that every trial belonging to the checkpoint being closed has
         * already been written in full (not, say, one thread partway
         * through a trial's own later output times/photometry while
         * another has already moved on and rolled over) -- and that
         * this itself is called on exactly one thread, from outside any
         * active parallel region. SimCluster::run()'s/SimGalaxy::run()'s
         * own comment explains how they guarantee that: trials run in
         * batches of checkpointInterval() at a time, each batch its own
         * "#pragma omp parallel for", so the implicit barrier at the end
         * of that construct -- every thread has finished every trial in
         * the batch before any of them proceeds past it -- has already
         * elapsed by the time either of them calls this, and neither is
         * still inside a parallel region when it does. This method
         * relies on both of those guarantees rather than re-deriving
         * them itself (e.g. via its own locking): closeOutputFile()
         * below runs once per thread via its own fresh "#pragma omp
         * parallel" spawn, exactly mirroring the destructor's own
         * identical pattern, which is only correct because nothing else
         * is concurrently touching any thread's own handles at that
         * moment.
         */
        void checkpoint(unsigned long trialsCompleted) override;

        /**
         * @brief Return the number of trials already completed by the run being restarted
         * @return restartTrialsDone_, as set by restartSetup() when
         *   this OutputManagerH5 was constructed with restart = true
         *   (0 if it was not, or if restartSetup() found no existing
         *   checkpoint to resume from)
         * @details
         * Only meaningful when this OutputManagerH5 was itself
         * constructed with restart = true -- SimCluster::run()/
         * SimGalaxy::run() only ever call this when they were
         * themselves given restart = true, which the CLI (main.cpp)
         * only ever does alongside the same restart argument this
         * object's own constructor received, so the two stay in sync
         * in practice, but nothing here enforces that a caller
         * constructing these objects directly always pairs them
         * consistently.
         */
        [[nodiscard]] auto restartTrialsDone() const -> unsigned long override
        { return restartTrialsDone_; }

        /**
         * @brief Record that this run is ending early, having only actually completed trialsCompleted trials
         * @param trialsCompleted Number of trials actually completed
         * @details
         * Stores trialsCompleted into earlyTerminationTrialsCompleted_,
         * for the destructor to write as the currently-open file's own
         * final "trials_completed"/"restart_uid" attributes instead of
         * SimControls::nTrial()/utils::uniqueID().read() -- see the
         * destructor's own comment, and OutputManager::
         * notifyEarlyTermination()'s own comment for the full
         * rationale. Does not itself close anything, or roll over to a
         * new checkpoint: SimCluster::run()/SimGalaxy::run() call this
         * and then simply return, leaving the currently-open
         * checkpoint (or, without checkpointing, the run's own single
         * output file) for the ordinary destructor to close, same as
         * it always does once this object goes out of scope.
         */
        void notifyEarlyTermination(unsigned long trialsCompleted) override
        { earlyTerminationTrialsCompleted_ = trialsCompleted; }

    private:

        /**
         * @brief Find the most recent checkpoint on disk and resume from it
         * @details
         * Only called by the constructor, and only when it was passed
         * restart = true, before openNewOutputFiles() -- so
         * checkpointNumber_ already names the correct next checkpoint
         * by the time openNewOutputFiles() opens the first output
         * file of this (resumed) run.
         *
         * Scans simControls_.outDir() for entries whose name matches
         * this run's own checkpoint naming pattern (see
         * checkpointModelName()): either a file, modelName_chkNNNNN.h5
         * (the shape a checkpoint ends up in only without OpenMP, or
         * once fully consolidated -- see this class's own header
         * comment and consolidateFiles()'s own comment; consolidation
         * only ever happens once, from the destructor, once an entire
         * run has completed normally, so in practice a checkpoint
         * being restarted from will essentially always still be in
         * the unconsolidated, per-thread shape below), or a directory,
         * modelName_chkNNNNN, holding one thread_NNNN.h5 file per
         * OpenMP thread that was writing at the time the checkpoint
         * was closed. Whichever of the two exists for the largest
         * NNNNN found is the checkpoint being resumed from;
         * checkpointNumber_ is set to that NNNNN + 1 (the next
         * checkpoint openNewOutputFiles() should create), and
         * restartTrialsDone_ to the "trials_completed" attribute
         * closeOutputFile() wrote on that checkpoint's own file (or,
         * for the per-thread-directory shape, on every one of its own
         * thread_NNNN.h5 files -- read from each in turn and checked
         * to all agree, since closeOutputFile() always wrote them the
         * same value in the first place (see its own comment) and a
         * mismatch here can only mean the on-disk checkpoint itself is
         * corrupt or was tampered with; throws std::runtime_error if
         * so, or if a directory found this way turns out to hold no
         * thread_NNNN.h5 files at all). Also reads that same
         * checkpoint's own "restart_uid" attribute the same way, and
         * calls utils::uniqueID().set() with it, so this process's own
         * cluster/galaxy IDs resume exactly where the run being
         * restarted left off, rather than either colliding with IDs it
         * already used (a fresh process's own utils::uniqueID()
         * otherwise starts back at 0) or leaving a gap (see
         * closeOutputFile()'s own comment for the full detail).
         *
         * If no entry matching the checkpoint naming pattern exists at
         * all, checkpointNumber_ and restartTrialsDone_ are simply
         * left at their own default-constructed values of 0, and
         * utils::uniqueID() is left untouched -- so a restart of a run
         * that never actually got as far as its first checkpoint (or
         * was never checkpointed to begin with) just starts over from
         * trial 0, exactly like an ordinary, non-restart run would.
         *
         * If SimControls::verbosity() is non-zero, prints which
         * checkpoint (if any) this restarted from, how many trials it
         * reports already completed, and the next uid it resumed from,
         * to stdout.
         *
         * Caveat: the largest NNNNN found on disk is not necessarily
         * one that finished closing before the run being restarted
         * stopped -- checkpoint()/openNewOutputFiles() create a new
         * checkpoint's own file(s) at the *start* of the batch that
         * checkpoint will hold, not once it finishes, so a checkpoint
         * that was still being written when the process was killed
         * looks, on disk, exactly like one whose files simply don't
         * exist yet elsewhere -- present, but with no
         * "trials_completed" attribute written on it (see
         * closeOutputFile()'s own comment: that attribute is only
         * ever written right before a file is closed). This method
         * does not silently fall back to an earlier, complete
         * checkpoint in that case -- it throws, with a message naming
         * the incomplete checkpoint and suggesting it be deleted
         * before restarting again.
         */
        void restartSetup();

        /**
         * @brief Reset this run's output state and open a fresh output file (or set of files)
         * @details
         * Refactored out of the constructor so checkpoint() can call it
         * again each time a new checkpoint begins -- both callers run
         * this from outside any active parallel region (see
         * checkpoint()'s own comment for how it guarantees that), so
         * this can always spawn its own fresh "#pragma omp parallel"
         * team the same way the original, pre-checkpointing constructor
         * always did, with no need to detect or special-case being
         * called from an already-parallel context.
         *
         * Computes outDir_/modelName -- or, if
         * SimControls::checkpointInterval() is non-zero,
         * outDir_/modelName_chkNNNNN (see checkpointModelName()), where
         * NNNNN is checkpointNumber_ -- checks that neither it nor its
         * ".h5" sibling already exists, and (with OpenMP) creates the
         * directory each thread's own thread_NNNN.h5 will live in, then
         * opens each thread's own file there (openOutputFile()) -- or,
         * without OpenMP, opens the single resulting file directly.
         *
         * openOutputFile() itself resets the calling thread's own
         * handles to -1 before opening its file, once per thread, from
         * inside the "#pragma omp parallel" spawned below.
         */
        void openNewOutputFiles();

        /**
         * @brief Return modelName_chkNNNNN, NNNNN being checkpointNum zero-padded to 5 digits
         * @param checkpointNum The checkpoint number to format
         * @details
         * Shared by openNewOutputFiles() (with checkpointNumber_) and
         * the destructor's own per-checkpoint consolidateFiles() loop
         * (with every checkpoint number from 0 to checkpointNumber_ in
         * turn), so both compute the same name for a given checkpoint
         * number.
         */
        [[nodiscard]] auto checkpointModelName(unsigned long checkpointNum) const -> std::string;

        /**
         * @brief Open one HDF5 file at the given path and write its header
         * @param path Path of the file to open
         * @details
         * Resets the calling thread's own handles to -1, then creates
         * the file, writes its top-level slug-hash/date/time/rng_state
         * attributes and input_deck group, then calls
         * openClustersGroup()/openClusterSpectraGroup()/
         * openClusterPhotGroup()/openGalaxyGroup()/
         * openGalaxySpectraGroup()/openGalaxyPhotGroup() to create
         * whichever of those groups are enabled. Called once per
         * thread (each with its own path) from inside
         * openNewOutputFiles()'s own OpenMP parallel region when built
         * with OpenMP and called from the constructor, or directly --
         * once, by whichever single thread is calling -- otherwise;
         * see openNewOutputFiles()'s own comment.
         */
        void openOutputFile(const std::filesystem::path& path);

        /**
         * @brief Close whichever of this thread's own groups are open, then its file
         * @param trialsCompleted Number of trials completed so far in
         *   the run; written as this thread's own file's
         *   "trials_completed" top-level attribute before it is
         *   closed, so a later restart can tell how far the run had
         *   gotten as of the checkpoint this file belongs to. The
         *   destructor passes SimControls::nTrial() (every trial in
         *   the run must already be complete if this is being
         *   destroyed at all); checkpoint() passes on whatever count
         *   its own caller (SimCluster::run()/SimGalaxy::run()) gave
         *   it, i.e. how many trials had completed as of the batch
         *   boundary that triggered this checkpoint.
         * @details
         * Also writes utils::uniqueID().read() as this thread's own
         * file's "restart_uid" top-level attribute, right after
         * trials_completed -- the next ID utils::uniqueID().get()
         * would have handed out, had this checkpoint's own last trial
         * not yet finished, so a later restart can resume ID
         * generation from exactly that point (see restartSetup()'s own
         * comment) rather than either colliding with IDs already used
         * by the run being resumed (were it to start over from 0, in
         * a fresh process) or leaving gaps for IDs a since-abandoned,
         * never-closed later checkpoint's own in-flight trials had
         * already been handed (were it to just keep whatever a fresh
         * process's own counter already reached on its own, which is
         * always fewer than what the resumed run had actually reached
         * -- e.g. immediately after this file's own process starts,
         * before it has generated any IDs of its own at all).
         *
         * Called once per thread from inside the destructor's own
         * OpenMP parallel region when built with OpenMP, or once,
         * directly, otherwise; also called by checkpoint(), on just
         * the one thread rolling over to a new checkpoint.
         */
        void closeOutputFile(unsigned long trialsCompleted);

        /**
         * @brief Merge every thread_NNNN.h5 file in a directory into a single sibling HDF5 file
         * @param path Directory holding the thread_NNNN.h5 files to
         *   merge; it is an error if this is not a directory
         * @details
         * Only meaningful when built with OpenMP, and only called by
         * the destructor when SimControls::outputMode() is
         * OutputMode::h5 (not h5divided) -- see this class's own
         * header comment. If checkpointing is enabled (see
         * checkpointModelName()), the destructor calls this once per
         * checkpoint (path being that checkpoint's own
         * outDir_/modelName_chkNNNNN), not just once, since each
         * checkpoint's own thread_NNNN.h5 files are entirely separate
         * from every other checkpoint's -- deliberately not called
         * mid-run, from checkpoint() itself, so consolidating doesn't
         * slow down the run itself (see checkpoint()'s own comment).
         *
         * Finds every file in path matching thread_NNNN.h5, sorted by
         * name (equivalent to numeric order, since the thread number
         * is zero-padded to a fixed width). Copies the first
         * (thread_0000.h5) to a new file, path.h5 (a sibling of path,
         * in path's own parent directory -- e.g. outDir/modelName.h5,
         * given outDir/modelName), and opens that copy for updating.
         * Every remaining thread_NNNN.h5 file is then opened for
         * reading, and every extensible dataset in every one of its
         * groups (identified generically, by iterating the file's own
         * group/dataset structure, rather than by a hardcoded list of
         * names -- so this needs no update if a new group/dataset is
         * ever added elsewhere in this class) is appended onto the
         * correspondingly-named dataset in path.h5 -- every
         * thread_NNNN.h5 shares an identical group/dataset skeleton
         * with path.h5 (each was built by an identical
         * openOutputFile() call), so every append target is
         * guaranteed to already exist. Row order across threads is
         * not preserved -- rows from thread_0000.h5 come first, then
         * thread_0001.h5's own rows, etc. -- but nothing downstream
         * relies on the order rows were written in.
         *
         * Once every thread_NNNN.h5 file's contents have been merged
         * in and path.h5 has been closed, every thread_NNNN.h5 file
         * and path itself (now empty) are deleted, leaving only
         * path.h5 behind.
         */
        static void consolidateFiles(const std::filesystem::path& path);

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

        // Number of times checkpoint() has rolled over to a new
        // checkpoint; 0 until the first call. Only meaningful if
        // SimControls::checkpointInterval() is non-zero -- see
        // checkpointModelName()/openNewOutputFiles(). Not a ThreadVec
        // like the handles below, despite (like them) being read/
        // written from inside a "#pragma omp parallel" region: unlike
        // those, this is one single, shared value describing the
        // whole checkpoint set every thread's own file currently
        // belongs to, not a separate value per thread -- and it is
        // only ever mutated between parallel regions (see
        // checkpoint()'s own comment for why), never concurrently, so
        // it needs no locking of its own either.
        unsigned long checkpointNumber_ = 0;

        // Number of trials the run being restarted had already
        // completed, as of the checkpoint restartSetup() found and
        // resumed from -- 0 if this OutputManagerH5 was not
        // constructed with restart = true, or if restartSetup() found
        // no checkpoint to resume from. Set once, by restartSetup(),
        // before openNewOutputFiles() is ever called, and never
        // written again afterward, so (like checkpointNumber_ above)
        // it needs no locking of its own despite being read from
        // inside a "#pragma omp parallel" region by restartTrialsDone()'s
        // own callers.
        unsigned long restartTrialsDone_ = 0;

        // Set by notifyEarlyTermination(), from outside any active
        // parallel region, at most once, strictly before this object
        // is ever destroyed (see its own comment) -- read only by the
        // destructor itself, single-threaded, so (like
        // checkpointNumber_/restartTrialsDone_ above) needs no locking
        // of its own. std::nullopt (the default) means this run ended
        // normally, having completed every trial -- the destructor's
        // own long-standing assumption, unchanged from before this
        // existed.
        std::optional<unsigned long> earlyTerminationTrialsCompleted_;

        // Each thread's own copy of every HDF5 handle below is
        // private to it -- see this class's own header comment on why
        // -- reset to -1 (the "not open" sentinel) by openOutputFile()
        // itself, for the calling thread's own slot alone, just before
        // that thread's own file is opened; see openNewOutputFiles()'s
        // own comment for why this happens there rather than as a
        // separate, whole-ThreadVec reset up front the way earlier,
        // pre-checkpointing code did.
        utils::ThreadVec<hid_t> file_; /**< Handle to this thread's own open HDF5 output file */ // NOLINT(misc-include-cleaner)
        utils::ThreadVec<hid_t> clustersGroup_; /**< Handle to this thread's own open clusters group, if any */ // NOLINT(misc-include-cleaner)
        utils::ThreadVec<hid_t> clusterSpectraGroup_; /**< Handle to this thread's own open cluster_spectra group, if any */ // NOLINT(misc-include-cleaner)
        utils::ThreadVec<hid_t> clusterPhotGroup_; /**< Handle to this thread's own open cluster_phot group, if any */ // NOLINT(misc-include-cleaner)
        utils::ThreadVec<hid_t> galaxyGroup_; /**< Handle to this thread's own open galaxy group, if any */ // NOLINT(misc-include-cleaner)
        utils::ThreadVec<hid_t> galaxySpectraGroup_; /**< Handle to this thread's own open galaxy_spectra group, if any */ // NOLINT(misc-include-cleaner)
        utils::ThreadVec<hid_t> galaxyPhotGroup_; /**< Handle to this thread's own open galaxy_phot group, if any */ // NOLINT(misc-include-cleaner)
    };

} // namespace io

#endif // OUTPUTMANAGERH5_HPP
