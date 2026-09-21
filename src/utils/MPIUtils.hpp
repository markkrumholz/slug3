/**
 * @file MPIUtils.hpp
 * @author Mark Krumholz
 * @brief Thin wrappers around MPI, providing a single guarded surface for
 *   multi-process parallelism
 * @date 2026-09-11
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#ifndef MPIUTILS_HPP
#define MPIUTILS_HPP

#include <algorithm>
#include <iostream> // NOLINT(misc-include-cleaner) -- only used inside initMPI()'s own #ifdef SLUG_MPI block (std::cerr, for the MPI_THREAD_FUNNELED warning), so a build without SLUG_MPI never actually uses it, the same reason <mpi.h> below needs its own identical NOLINT
#include <utility>

#ifdef SLUG_MPI
#   include <mpi.h> // NOLINT(misc-include-cleaner)
#endif
#ifdef _OPENMP
#   include <omp.h> // NOLINT(misc-include-cleaner) -- only used inside initMPI()'s own "provided < MPI_THREAD_FUNNELED" branch, under #ifdef SLUG_MPI as well, so a build with _OPENMP but not SLUG_MPI never actually uses it
#endif

namespace utils
{

    /**
     * @brief The width of the unique-ID range this process's own rank owns
     * @details
     * When MPI is enabled, rank m's UniqueIDManager (see that file) is
     * seeded to start at m * mpiUidStride rather than 0, so that every
     * rank's own live-generated IDs are permanently non-overlapping, for
     * the life of the run, with no need to ever rewrite an ID already
     * written to disk. 2^40 (~1.1e12) is comfortably beyond the number of
     * clusters any single rank could plausibly generate in one run, while
     * still leaving room for up to 2^24 (~16.8 million) ranks before two
     * ranks' own ranges could ever meet, using the unsigned long (64-bit
     * on every platform this code targets) IDs are stored as.
     */
    inline constexpr unsigned long mpiUidStride = 1UL << 40U;

    namespace detail
    {
        /**
         * @brief Storage for this process's own MPI rank, filled in by initMPI()
         * @return A reference to a function-local static, 0 until initMPI()
         *   has run (and always 0 when this build was not compiled with
         *   SLUG_MPI)
         * @details
         * Cached, rather than queried from MPI on each call to mpiRank(),
         * because a process's rank never changes once MPI is up, and
         * because MPI_THREAD_FUNNELED (see initMPI()) only allows MPI
         * calls from the one thread that initialized MPI: mpiRank() and
         * isIORank() are called from code that can run inside an OpenMP
         * parallel region, or after MPI_Finalize(), where a fresh
         * MPI_Comm_rank() call would not be valid.
         */
        inline auto cachedMPIRank() -> int&
        {
            static int rank = 0;
            return rank;
        }

        /**
         * @brief Storage for the number of MPI ranks, filled in by initMPI()
         * @return A reference to a function-local static, 1 until initMPI()
         *   has run (and always 1 when this build was not compiled with
         *   SLUG_MPI) -- see cachedMPIRank() for why this is cached
         */
        inline auto cachedMPISize() -> int&
        {
            static int size = 1;
            return size;
        }
    } // namespace detail

    /**
     * @brief Initialize MPI, if this build was compiled with SLUG_MPI
     * @param argc Address of main()'s own argc
     * @param argv Address of main()'s own argv
     * @details
     * A no-op when SLUG_MPI is not defined. Requests
     * MPI_THREAD_FUNNELED, since MPI is only ever called from a single
     * thread, outside any active OpenMP parallel region (see
     * mpiBarrier()'s own callers). MPI_THREAD_FUNNELED itself only
     * constrains which thread may call MPI -- but the next level down,
     * MPI_THREAD_SINGLE, is a stronger promise from *this process* to
     * the MPI implementation that it will never be multithreaded at
     * all, not just that MPI calls are confined to one thread; an
     * implementation that could only actually provide MPI_THREAD_SINGLE
     * gives no guarantee that it behaves correctly once this build (if
     * also compiled with OpenMP) goes on to spawn worker threads
     * regardless, even ones that never call MPI themselves. Since this
     * build cannot un-compile its own OpenMP support at this point, the
     * next best thing is forcing OpenMP down to a single thread for the
     * rest of the run, so the "no threads beyond this one" promise
     * MPI_THREAD_SINGLE actually implies is still honored in practice
     * -- slower than intended, but correct, rather than silently
     * relying on a implementation-specific tolerance the standard
     * itself does not promise.
     *
     * Also caches this process's own rank and the total number of
     * ranks, which mpiRank(), mpiSize() and isIORank() then return
     * without making any further MPI call -- so they are safe to call
     * from any thread, and after finalizeMPI(), as well.
     */
    inline void initMPI([[maybe_unused]] int* argc, [[maybe_unused]] char*** argv)
    {
#ifdef SLUG_MPI
        int provided = 0;
        MPI_Init_thread(argc, argv, MPI_THREAD_FUNNELED, &provided);
        MPI_Comm_rank(MPI_COMM_WORLD, &detail::cachedMPIRank());
        MPI_Comm_size(MPI_COMM_WORLD, &detail::cachedMPISize());
        if (provided < MPI_THREAD_FUNNELED)
        {
            std::cerr << "slug: warning: MPI implementation provided a lesser "
                "thread support level than MPI_THREAD_FUNNELED; forcing "
                "single-threaded execution for this run, since the MPI "
                "standard does not guarantee correct behavior for a "
                "multithreaded process otherwise\n";
#ifdef _OPENMP
            omp_set_num_threads(1);
#endif
        }
#endif
    }

    /**
     * @brief Shut MPI down, if this build was compiled with SLUG_MPI
     * @details A no-op when SLUG_MPI is not defined.
     */
    inline void finalizeMPI()
    {
#ifdef SLUG_MPI
        MPI_Finalize();
#endif
    }

    /**
     * @class MPIGuard
     * @brief RAII wrapper calling initMPI() on construction, finalizeMPI() on destruction
     * @details
     * Construct exactly one of these in main(), as the very first thing
     * it does, and let it live for main()'s entire remaining scope. main()
     * has several early-return exit paths (bad usage, a failed parse, a
     * caught exception, normal completion); this guarantees finalizeMPI()
     * runs on every one of them without it needing to be called
     * explicitly at each, the same way a destructor guarantees cleanup
     * regardless of how a scope is exited.
     */
    class MPIGuard
    {
    public:
        /**
         * @brief Construct the guard, calling initMPI() with the given argc/argv
         * @param argc Address of main()'s own argc
         * @param argv Address of main()'s own argv
         */
        MPIGuard(int* argc, char*** argv) { initMPI(argc, argv); }

        /**
         * @brief Destroy the guard, calling finalizeMPI()
         */
        ~MPIGuard() { finalizeMPI(); }
        MPIGuard(const MPIGuard&) = delete;
        MPIGuard(MPIGuard&&) = delete;
        auto operator=(const MPIGuard&) -> MPIGuard& = delete;
        auto operator=(MPIGuard&&) -> MPIGuard& = delete;
    };

    /**
     * @brief Return this process's own MPI rank
     * @return This process's rank in MPI_COMM_WORLD, or 0 if this build
     *   was not compiled with SLUG_MPI, or initMPI() has not run
     * @details
     * Reads a value cached by initMPI() rather than calling MPI, so it
     * is safe to call from any thread, and after finalizeMPI().
     */
    [[nodiscard]] inline auto mpiRank() -> int { return detail::cachedMPIRank(); }

    /**
     * @brief Return the total number of MPI ranks in this run
     * @return The size of MPI_COMM_WORLD, or 1 if this build was not
     *   compiled with SLUG_MPI, or initMPI() has not run
     * @details Reads a value cached by initMPI() -- see mpiRank().
     */
    [[nodiscard]] inline auto mpiSize() -> int { return detail::cachedMPISize(); }

    /**
     * @brief Return whether this process should print run-wide informational output
     * @return True when this build was not compiled with SLUG_MPI, or
     *   is running as a single rank, or is rank 0 of several; false on
     *   every other rank
     * @details
     * Under MPI every rank runs the same setup code, so a message that
     * is identical on every rank (a warning about the input deck, a
     * run-wide summary) would otherwise be printed once per rank.
     * Guard such a print with `if (utils::isIORank())`. Do not guard
     * an error, or a message that reports something particular to this
     * rank (e.g. its own trial range): suppressing those on every rank
     * but 0 would hide real failures.
     */
    [[nodiscard]] inline auto isIORank() -> bool { return mpiRank() == 0; }

    /**
     * @brief Block until every MPI rank has reached this same call
     * @details
     * A no-op when SLUG_MPI is not defined. Must only be called from a
     * single thread, outside any active OpenMP parallel region -- see
     * OutputManagerH5::syncCheckpoints()'s own comment, its only caller.
     */
    inline void mpiBarrier()
    {
#ifdef SLUG_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif
    }

    /**
     * @brief Reconcile a per-rank boolean flag to true if any rank's own is true
     * @param local This rank's own value of the flag
     * @return local, unchanged, when SLUG_MPI is not defined or
     *   mpiSize() == 1; otherwise true if any rank passed true, false
     *   only if every rank passed false
     * @details
     * A signal like SIGTERM is caught independently, per-process (see
     * utils::sigtermWasReceived()) -- under MPI, only the rank(s) an
     * external SIGTERM was actually sent to (in principle, an entire
     * job's process group, but not guaranteed, and not even guaranteed
     * to be observed at the same moment by every rank that did receive
     * it) would ever see it, and every rank still needs to reach the
     * exact same decision at the exact same point for
     * OutputManagerH5::checkpoint()/its own destructor's barrier-
     * synchronized calls to stay in lockstep across ranks: a rank
     * stopping alone, while others continue on to their own next
     * mpiBarrier() call, would otherwise deadlock those other ranks
     * forever, since nothing they call is left to make that barrier's
     * matching call on the now-stopped rank's behalf. This is a single
     * cheap collective (one MPI_Allreduce), called at the same,
     * infrequent points sigtermWasReceived() itself already was
     * (once per batch, not per trial), so its own cost is negligible
     * next to those already-existing per-batch/per-checkpoint
     * collectives.
     */
    [[nodiscard]] inline auto mpiAllReceivedSigterm(const bool local) -> bool
    {
#ifdef SLUG_MPI
        int localFlag = local ? 1 : 0;
        int anyFlag = 0;
        MPI_Allreduce(&localFlag, &anyFlag, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        return anyFlag != 0;
#else
        return local;
#endif
    }

    /**
     * @brief Divide a contiguous trial-number range evenly across every MPI rank
     * @param rangeStart Start of the range to divide (inclusive)
     * @param rangeEnd End of the range to divide (exclusive)
     * @return The [start, end) sub-range of [rangeStart, rangeEnd) that
     *   this process's own rank should run -- every rank's own returned
     *   sub-range is disjoint from every other rank's, and their union
     *   is exactly [rangeStart, rangeEnd), so trial numbers themselves
     *   never collide across ranks (only unique IDs need mpiUidStride
     *   for that -- see its own comment)
     * @details
     * Splits [rangeStart, rangeEnd) into mpiSize() contiguous
     * sub-ranges, as evenly as possible: with total = rangeEnd -
     * rangeStart, base = total / mpiSize() trials go to every rank, and
     * the first (total % mpiSize()) ranks each get one additional
     * trial, so the sub-ranges' own sizes add up to exactly total
     * regardless of whether it divides evenly. When mpiSize() == 1
     * (including every build not compiled with SLUG_MPI), this always
     * returns {rangeStart, rangeEnd} unchanged.
     */
    [[nodiscard]] inline auto mpiPartitionRange(
        const unsigned long rangeStart, const unsigned long rangeEnd)
        -> std::pair<unsigned long, unsigned long>
    {
        const auto size = static_cast<unsigned long>(mpiSize());
        const auto rank = static_cast<unsigned long>(mpiRank());
        const unsigned long total = rangeEnd - rangeStart;
        const unsigned long base = total / size;
        const unsigned long remainder = total % size;
        const unsigned long myCount = base + ((rank < remainder) ? 1 : 0);
        const unsigned long precedingCount = rank * base + std::min(rank, remainder);
        const unsigned long start = rangeStart + precedingCount;
        return {start, start + myCount};
    }

} // namespace utils

#endif // MPIUTILS_HPP
