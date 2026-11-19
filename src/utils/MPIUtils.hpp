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
#include <array> // NOLINT(misc-include-cleaner) -- only used inside mpiAgreeBatchStop()'s own #ifdef SLUG_MPI block, so a build without SLUG_MPI never actually uses it
#include <exception> // NOLINT(misc-include-cleaner) -- correct header for std::exception_ptr/rethrow_exception/uncaught_exceptions; see the identical NOLINT in SimCluster.cpp
#include <iostream> // NOLINT(misc-include-cleaner) -- only used inside initMPI()'s own #ifdef SLUG_MPI block (std::cerr, for the MPI_THREAD_FUNNELED warning), so a build without SLUG_MPI never actually uses it, the same reason <mpi.h> below needs its own identical NOLINT
#include <stdexcept>
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
     * @struct BatchStop
     * @brief Whether any rank wants a run to stop at the end of the current batch, and why
     */
    struct BatchStop
    {
        bool sigterm_ = false; /**< True if any rank caught a SIGTERM during the batch */
        bool error_ = false;   /**< True if any rank had a trial throw during the batch */
    };

    /**
     * @brief Reconcile every rank's own end-of-batch stop conditions into one run-wide decision
     * @param localSigterm Whether this rank caught a SIGTERM during the batch
     * @param localError Whether a trial on this rank threw during the batch
     * @return {localSigterm, localError}, unchanged, when SLUG_MPI is
     *   not defined or mpiSize() == 1; otherwise each field is true if
     *   any rank passed true for it, and false only if every rank
     *   passed false
     * @details
     * Every rank must reach the same decision at the same point, so
     * that OutputManagerH5::checkpoint() and its destructor make their
     * barrier-synchronized calls in lockstep across ranks. A rank that
     * stops alone, while the others continue to their next collective
     * call, deadlocks those others forever, since nothing is left to
     * make the matching call on the stopped rank's behalf.
     *
     * Both conditions can arise on some ranks but not others. A signal
     * like SIGTERM is caught independently per process (see
     * utils::sigtermWasReceived()), and need not reach every rank, or
     * reach them at the same moment. A trial that throws does so only
     * on the rank running it. This is a single cheap collective (one
     * MPI_Allreduce), called once per batch rather than per trial, so
     * its cost is negligible next to the per-checkpoint collectives
     * that already exist.
     */
    [[nodiscard]] inline auto mpiAgreeBatchStop(const bool localSigterm, const bool localError) -> BatchStop
    {
#ifdef SLUG_MPI
        std::array<int, 2> localFlags = { localSigterm ? 1 : 0, localError ? 1 : 0 };
        std::array<int, 2> anyFlags = { 0, 0 };
        MPI_Allreduce(localFlags.data(), anyFlags.data(), 2, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        return { .sigterm_ = anyFlags[0] != 0, .error_ = anyFlags[1] != 0 };
#else
        return { .sigterm_ = localSigterm, .error_ = localError };
#endif
    }

    /**
     * @brief Return whether a flag is true on any MPI rank
     * @param local This rank's own value of the flag
     * @return local, unchanged, when SLUG_MPI is not defined or
     *   mpiSize() == 1; otherwise true if any rank passed true
     * @details
     * A collective call (one MPI_Allreduce): every rank must call it at
     * the same point. See mpiAgreeBatchStop() for the two-flag form
     * used at the end of each batch of trials.
     */
    [[nodiscard]] inline auto mpiAnyRank(const bool local) -> bool
    {
#ifdef SLUG_MPI
        const int localFlag = local ? 1 : 0;
        int anyFlag = 0;
        MPI_Allreduce(&localFlag, &anyFlag, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        return anyFlag != 0;
#else
        return local;
#endif
    }

    namespace detail
    {
        /**
         * @brief Storage for whether every rank has agreed to stop on an error
         * @return A reference to a function-local static, false until
         *   rethrowAgreedError() sets it
         */
        inline auto errorAgreedFlag() -> bool&
        {
            static bool agreed = false;
            return agreed;
        }
    } // namespace detail

    /**
     * @brief Return whether the exception currently ending the run is one every rank agreed on
     * @return True once rethrowAgreedError() has been called, false before
     * @details
     * An agreed error (see rethrowAgreedError()) is being thrown from the
     * same point on every rank, so every rank's shutdown -- in
     * particular OutputManagerH5's destructor, and its collective calls
     * -- proceeds in lockstep, exactly as for a normal completion. Any
     * other exception is local to the rank that threw it; see
     * unwindingFromLocalError() and mpiAbortIfMultiRank().
     */
    [[nodiscard]] inline auto errorIsAgreed() -> bool { return detail::errorAgreedFlag(); }

    /**
     * @brief Throw the error every rank agreed to stop on, from every rank
     * @param localError The first exception this rank hit, or a null
     *   exception_ptr if it hit none
     * @param elsewhere The message a rank with no error of its own
     *   throws; defaults to saying a trial failed on another rank
     * @throws The exception localError holds if it is non-null;
     *   otherwise a std::runtime_error with message elsewhere
     * @details
     * Call this, on every rank, once a collective agreement
     * (mpiAgreeBatchStop() after a batch of trials, or mpiAnyRank())
     * has reported that some rank had an error. It first marks the
     * error as agreed (see errorIsAgreed()), so the shutdown that
     * follows runs its collective calls normally. A rank with no error
     * of its own throws a generic one instead, since only the failing
     * rank has the original exception; that rank prints the failure
     * itself.
     */
    [[noreturn]] inline void rethrowAgreedError(const std::exception_ptr& localError,
        const char* elsewhere = "stopping because a trial failed on another MPI rank "
            "(see that rank's own error message)")
    {
        detail::errorAgreedFlag() = true;
        if (localError) { std::rethrow_exception(localError); }
        throw std::runtime_error(elsewhere);
    }

    /**
     * @brief Return whether this rank is unwinding from an exception the other ranks do not know about
     * @return True if this is a multi-rank MPI run, an exception is
     *   currently propagating (std::uncaught_exceptions() > 0), and
     *   that exception is not an agreed one (see errorIsAgreed());
     *   always false outside MPI, or with a single rank
     * @details
     * Intended for destructors that would otherwise make collective
     * MPI calls (e.g. OutputManagerH5's). When this returns true, the
     * other ranks are not unwinding along with this one: they are
     * carrying on with the run, or are already waiting in some other
     * collective. A collective call here would not match theirs, so
     * it would never complete, and the error would never reach main()
     * to be reported. Such a destructor should instead do only
     * rank-local cleanup, and leave main() to report the error and
     * call mpiAbortIfMultiRank().
     */
    [[nodiscard]] inline auto unwindingFromLocalError() -> bool
    {
        return mpiSize() > 1 && std::uncaught_exceptions() > 0 && !errorIsAgreed();
    }

    /**
     * @brief Abort every MPI rank, if this is a multi-rank MPI run
     * @param exitCode Error code to pass to MPI_Abort()
     * @details
     * A no-op when SLUG_MPI is not defined or mpiSize() == 1, leaving
     * the caller to exit normally. Otherwise calls MPI_Abort(), which
     * does not return: it terminates every rank in MPI_COMM_WORLD.
     * This is the backstop for an error the other ranks did not agree
     * on (see errorIsAgreed()): they may be blocked in a collective
     * call this rank will never make, and MPI_Finalize() would then
     * block forever as well.
     */
    inline void mpiAbortIfMultiRank([[maybe_unused]] const int exitCode)
    {
#ifdef SLUG_MPI
        if (mpiSize() > 1) { MPI_Abort(MPI_COMM_WORLD, exitCode); }
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
