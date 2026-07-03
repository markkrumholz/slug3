/**
 * @file RngThread.hpp
 * @author Mark Krumholz
 * @brief Implements thread-safe random number generation
 * @date 2024-06-17
 */

#ifndef RNGUTILS_HPP
#define RNGUTILS_HPP

#include <memory>
#include <random>
#ifdef _OPENMP
#   include <omp.h>
#endif
#include <pcg_random.hpp>
#include "ThreadVec.hpp"

namespace utils {

    using RngType = pcg64;  /**< Alias for pcg generator */

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
         * a new RngThread using the 
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

    private:

        ThreadVec<std::unique_ptr<pcg64> > rngEngines_;  /**< Pointers to rng engines */

    };

    // Create a static instance of an RngThread object,
    // which will be usable from throughout the code. This ensures
    // that all random numbers are handled in a thread-safe manner.
    static RngThread rng; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

} // namespace utils

#endif // RNGUTILS_HPP