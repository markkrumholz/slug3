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
         * @details
         * simControls and inputDeck are stored by reference, so the
         * objects passed in must outlive this OutputManagerH5. See
         * this class's own header comment for what "the output
         * file(s)" means when built with OpenMP.
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
         * @brief Open one HDF5 file at the given path and write its header
         * @param path Path of the file to open
         * @details
         * Creates the file, writes its top-level slug-hash/date/time/
         * rng_state attributes and input_deck group, then calls
         * openClustersGroup()/openClusterSpectraGroup()/
         * openClusterPhotGroup()/openGalaxyGroup()/
         * openGalaxySpectraGroup()/openGalaxyPhotGroup() to create
         * whichever of those groups are enabled. Called once per
         * thread (each with its own path) from inside the constructor's
         * own OpenMP parallel region when built with OpenMP, or once,
         * directly, otherwise -- see this class's own header comment.
         */
        void openOutputFile(const std::filesystem::path& path);

        /**
         * @brief Close whichever of this thread's own groups are open, then its file
         * @details
         * Called once per thread from inside the destructor's own
         * OpenMP parallel region when built with OpenMP, or once,
         * directly, otherwise.
         */
        void closeOutputFile();

        /**
         * @brief Merge every thread_NNNN.h5 file in a directory into a single sibling HDF5 file
         * @param path Directory holding the thread_NNNN.h5 files to
         *   merge; it is an error if this is not a directory
         * @details
         * Only meaningful when built with OpenMP, and only called by
         * the destructor when SimControls::outputMode() is
         * OutputMode::h5 (not h5divided) -- see this class's own
         * header comment.
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

        // Each thread's own copy of every HDF5 handle below is
        // private to it -- see this class's own header comment on why
        // -- initialized to -1 (the same "not open" sentinel the
        // plain hid_t members used before) for every thread by this
        // class's own constructor, before any thread's openOutputFile()
        // runs.
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
