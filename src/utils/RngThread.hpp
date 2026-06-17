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
            for (auto& r : rngEngines)
            {
                pcg_extras::seed_seq_from<std::random_device> seed_source;
                r = std::make_unique<RngType>(seed_source);
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
            for (size_t i = 0; auto& r : rngEngines)
            {
                r = std::make_unique<RngType>(seed + i);
            }
        }
        ~RngThread() = default;

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
            for (size_t i = 0; auto& r : rngEngines)
            {
                r->seed(seed);
            }
        }

        /**
         * @fn utils::RngThread::operator()() const
         * @brief Return the correct rng engine for the calling thread
         */
        auto operator()() const -> RngType& { return *rngEngines(); }

    private:

        ThreadVec<std::unique_ptr<pcg64> > rngEngines;  /**< Pointers to rng engines */

    };

    // Create a static instance of an RngThread object,
    // which will be usable from throughout the code
    static RngThread rng;

} // namespace utils

#endif // RNGUTILS_HPP