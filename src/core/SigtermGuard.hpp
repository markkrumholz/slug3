/**
 * @file SigtermGuard.hpp
 * @author Mark Krumholz
 * @brief Shared SIGTERM-handling machinery for SimCluster::run()/SimGalaxy::run()
 * @date 2026-09-03
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#ifndef SIGTERMGUARD_HPP
#define SIGTERMGUARD_HPP

#include <csignal>
#include <stdexcept>
#include <string>

namespace core
{

    // Set (only) by signalHandler() below; read by sigtermWasReceived().
    // volatile sig_atomic_t is the one type the C++ standard guarantees
    // safe to write from inside a signal handler without risking
    // undefined behavior -- but that guarantee is about signal-handler
    // safety on whichever single thread the signal happened to be
    // delivered to (POSIX picks one, not necessarily this process's
    // main thread -- see SigtermGuard's own comment for why that
    // doesn't matter here), not about the value becoming visible to
    // every OpenMP worker thread's own read of it: that cross-thread
    // visibility is what sigtermWasReceived()'s own "#pragma omp
    // atomic read" actually provides. inline (a C++17 inline variable)
    // so SimCluster::run() and SimGalaxy::run() -- the only two
    // callers, never both running at once within a single process (see
    // SigtermGuard's own comment) -- share exactly one instance of
    // this flag, matching how a signal disposition installed via
    // std::signal() is itself process-wide, not per-translation-unit.
    inline volatile std::sig_atomic_t sigtermReceived = 0; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables) -- a signal handler can only communicate through global/static state; see this variable's own comment

    inline void signalHandler(int /*signum*/)
    {
        sigtermReceived = 1;
    }

    /**
     * @brief Check whether a SIGTERM has been caught since the currently-installed SigtermGuard was constructed
     * @return true if signalHandler() has run since the currently-
     *   installed SigtermGuard reset sigtermReceived to 0, false
     *   otherwise
     * @details
     * See sigtermReceived's own comment for why this is a "#pragma omp
     * atomic read" rather than a plain read of a volatile
     * sig_atomic_t, despite that alone already being signal-handler-safe
     */
    inline auto sigtermWasReceived() -> bool
    {
        std::sig_atomic_t value = 0;
#ifdef _OPENMP
#pragma omp atomic read
#endif
        value = sigtermReceived;
        return value != 0;
    }

    /**
     * @class SigtermGuard
     * @brief RAII guard installing (and later restoring) a SIGTERM handler
     * @details
     * Resets sigtermReceived and installs signalHandler for SIGTERM on
     * construction, restoring whatever disposition was previously in
     * place on destruction -- covering every one of run()'s own exit
     * paths (normal return, a caught SIGTERM, or an exception
     * propagating out) uniformly, since none of them can skip a local
     * variable's own destructor running. Without the restore, a single
     * process that calls SimCluster::run()/SimGalaxy::run() more than
     * once (e.g. a Python session driving several slug runs in a row
     * via run_sim()) would leave slug's own handler permanently
     * installed process-wide even after the run it was actually meant
     * for has finished, silently overriding whatever the embedding
     * application itself wanted SIGTERM to do afterward. Resetting
     * sigtermReceived here (not just once, at namespace-scope
     * initialization) matters for the same reason: without it, a
     * second run() call in that same process would immediately see the
     * *first* run's own already-set flag and exit early on its very
     * first checkpoint, having never actually received a SIGTERM of
     * its own at all.
     *
     * Shared by SimCluster::run() and SimGalaxy::run(), which are
     * otherwise independent, never-both-running-at-once drivers, each
     * with its own run()/runTrial() to guard -- installErrorMessage
     * lets each pass its own std::runtime_error message identifying
     * which one failed to install its handler.
     */
    class SigtermGuard
    {
    public:

        /**
         * @brief Install the SIGTERM handler
         * @param installErrorMessage Message std::runtime_error is
         *   constructed with if installation fails; conventionally
         *   names the caller (e.g. "SimCluster::run: unable to install
         *   a SIGTERM handler")
         */
        explicit SigtermGuard(const std::string& installErrorMessage)
        {
            // Reset the flag before installing the handler, not after
            // (and so deliberately not via a member initializer for
            // previousHandler_, despite what clang-tidy's
            // cppcoreguidelines-prefer-member-initializer suggests --
            // that would run the install first): a SIGTERM arriving
            // between the two would otherwise have this immediately
            // clear the very flag it just set, silently losing the
            // signal.
            sigtermReceived = 0;
            previousHandler_ = std::signal(SIGTERM, signalHandler); // NOLINT(cppcoreguidelines-prefer-member-initializer)
            if (previousHandler_ == SIG_ERR)
            {
                throw std::runtime_error(installErrorMessage);
            }
        }

        /**
         * @brief Restore whatever SIGTERM disposition was in place before construction
         */
        ~SigtermGuard()
        {
            // Best-effort restore -- this runs from a destructor, so
            // it must not throw, and there is nothing more useful to
            // do with a failure here anyway
            (void)std::signal(SIGTERM, previousHandler_); // NOLINT(cert-err33-c)
        }

        SigtermGuard(const SigtermGuard&) = delete;
        auto operator=(const SigtermGuard&) -> SigtermGuard& = delete;
        SigtermGuard(SigtermGuard&&) = delete;
        auto operator=(SigtermGuard&&) -> SigtermGuard& = delete;

    private:
        using Handler = void (*)(int);
        Handler previousHandler_ = nullptr;
    };

} // namespace core

#endif // SIGTERMGUARD_HPP
