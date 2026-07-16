/**
 * @file RngThread.hpp
 * @author Mark Krumholz
 * @brief Implements thread-safe random number generation
 * @date 2024-06-17
 */

#ifndef RNGUTILS_HPP
#define RNGUTILS_HPP

#include "ThreadVec.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#ifdef _OPENMP
#   include <omp.h>
#endif
#include <pcg_extras.hpp>
#include <pcg_random.hpp>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>

namespace utils {

    using RngType = pcg64;  /**< Alias for pcg generator */

    /**
     * @brief Fixed-width buffer type for a serialized rng state
     * @details
     * A pcg64 engine's stream insertion operator writes its full state
     * as three 128-bit values -- multiplier, increment, and state --
     * in decimal, separated by single spaces. Since 2^128 - 1 has 39
     * decimal digits, that text is at most 3*39 + 2 = 119 characters;
     * 128 bytes gives headroom for that plus a null terminator. Being
     * a fixed-width POD type, RngState (unlike a std::string) can be
     * written to disk or passed via MPI as a plain byte buffer.
     */
    inline constexpr size_t rngStateWidth = 128; /**< Width, in bytes, of RngState */
    using RngState = std::array<char, rngStateWidth>;

    /**
     * @class RngThread
     * @brief Thread-safe random number generator
     * @details
     * This class implements a thread-safe random number generator,
     * based on the pcg family of random number generators. Each thread
     * maintains a separate, private random stream, and calls to the
     * random number generator from a given thread are automatically
     * drawn from the correct stream.
     */
    class RngThread {

    public:

        /**
         * @brief Construct an RngThread
         */
        RngThread()
        {
            for (auto& r : rngEngines_)
            {
                pcg_extras::seed_seq_from<std::random_device> seedSource;
                r = std::make_unique<RngType>(seedSource);
            }
        }

        /**
         * @brief Construct an RngThread with a specified seed
         * @param seed The seed value to use
         * @details
         * The value seed will be used for the first thread, and
         * all subsequent threads will be seeded with values seed + 1,
         * seed + 2, etc.
         */
        RngThread(const RngType::state_type seed)
        {
            for (int i = 0; i < rngEngines_.size(); ++i)
            {
                rngEngines_[i] = std::make_unique<RngType>(seed + i); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
            }
        }

        ~RngThread() = default;

        // Disable copy and move constructors and assignment operators
        RngThread(const RngThread&) = delete;
        RngThread(RngThread&&) = delete;
        auto operator=(const RngThread&) -> RngThread& = delete;
        auto operator=(RngThread&&) -> RngThread& = delete;

        /**
         * @brief Re-seed the random number generator
         * @param seed The new seed
         * @details
         * The value seed will be used for the first thread, and
         * all subsequent threads will be seeded with values seed + 1,
         * seed + 2, etc. The result is equivalent to constructing
         * a new RngThread using the specified seed.
         */
        void seed(const RngType::state_type seed)
        {
            for (int i = 0; i < rngEngines_.size(); ++i)
            {
                rngEngines_[i]->seed(seed+i); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
            }
        }

        /**
         * @fn utils::RngThread::operator()() const
         * @brief Return the correct rng engine for the calling thread
         */
        auto operator()() const -> RngType& { return *rngEngines_(); }

        /**
         * @brief Return the serialized state of the rng engine
         * @return A fixed-width RngState buffer holding the serialized
         *   state of the rng engine private to the calling thread
         * @details
         * As with operator(), which engine's state is returned depends
         * on which thread calls this method. The returned buffer can
         * later be passed to setState() to restore the engine to
         * exactly this state. Bytes beyond the serialized text are
         * zeroed, so the buffer is null-terminated and safe to treat
         * as a C string.
         */
        [[nodiscard]] auto getState() const -> RngState
        {
            std::ostringstream stateStream;
            stateStream << *rngEngines_();
            const std::string state = stateStream.str();
            if (state.size() >= rngStateWidth)
            {
                throw std::runtime_error(
                    "RngThread: serialized state does not fit in the "
                    "fixed-width RngState buffer");
            }
            RngState buf{};
            std::ranges::copy(state, buf.begin());
            return buf;
        }

        /**
         * @brief Set the state of the rng engine
         * @param state A serialized state, as returned by getState()
         * @details
         * As with operator(), which engine's state is set depends on
         * which thread calls this method.
         */
        void setState(const RngState& state)
        {
            std::istringstream stateStream{std::string(state.data())};
            stateStream >> *rngEngines_();
        }

    private:

        ThreadVec<std::unique_ptr<pcg64> > rngEngines_;  /**< Pointers to rng engines */

    };

    /**
     * @brief Return a reference to the global, thread-safe random engine
     * @details
     * The instance is a function-local static, so it is constructed on
     * first use rather than before main() begins; this ensures that any
     * exception thrown during construction (which in practice cannot
     * happen, since the number of threads is always small) is caught
     * here rather than escaping before main() can run. There is no
     * reasonable way for the program to continue without a working
     * random number generator, so a construction failure is treated as
     * fatal: it is reported and the program exits immediately, rather
     * than letting the exception propagate to callers that have no
     * better way to handle it either. Being a local static in an inline
     * function also guarantees a single, program-wide instance, rather
     * than one per translation unit.
     */
    inline auto rng() -> RngThread&
    {
        try
        {
            static RngThread instance;
            return instance;
        }
        catch (const std::exception& error)
        {
            std::cerr << "FATAL: random number generator initialization "
                "failed: " << error.what() << "\n";
            std::exit(1); // NOLINT(concurrency-mt-unsafe) -- the program is terminating regardless; thread-safety of the shutdown path doesn't matter
        }
    }

} // namespace utils

#endif // RNGUTILS_HPP