/**
 * @file ThreadVec.hpp
 * @author Mark Krumholz
 * @brief Utility class for handling multithreaded runs
 * @date 2024-06-17
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#ifndef THREADVEC_HPP
#define THREADVEC_HPP

#ifdef _OPENMP
#   include <algorithm>
#   include <omp.h>
#   include <thread>
#endif
#include <cassert>
#include <cstddef>
#include <vector>

namespace utils {

#ifdef _OPENMP
    // Multiplier applied to std::thread::hardware_concurrency() when
    // clamping ThreadVec's own construction-time size -- see
    // ThreadVec::ThreadVec()'s own comment for why this exists at
    // all. Generous on purpose: every T ThreadVec is instantiated
    // with in this codebase is a small handle (pointer-sized or
    // smaller), so erring on the large side here costs negligible
    // memory, while erring small risks the out-of-bounds access this
    // whole mechanism exists to prevent.
    inline constexpr int threadVecCapOversubscription = 8;
#endif

    /**
     * @class ThreadVec
     * @brief A class to automate private thread copies of objects
     * @tparam T Type of object the ThreadVec holds
     * @details
     * This class is a container to automate the process of each thread
     * having its own private copy of an object. It is a vector that on
     * creation automatically resizes itself large enough for the
     * program's own maximum possible OpenMP team size (see the
     * constructor's own comment), and when accessed by a thread
     * automatically returns the element of the vector that is private
     * to that thread.
     */
    template <class T> class ThreadVec {

    public:

        /**
         * @brief Construct an empty ThreadVec
         * @details
         * Sized via omp_get_thread_limit() rather than
         * omp_get_max_threads(): the latter only reflects the
         * *default* team size at the moment of construction, and two
         * things can raise the real team size above that later --
         * omp_set_num_threads() called after this ThreadVec already
         * exists, or a parallel region with an explicit
         * num_threads(n) clause requesting more threads than
         * omp_get_max_threads() currently reports. Every access below
         * is keyed by the real omp_get_thread_num() of whichever team
         * ends up running, so an undersized ThreadVec would then be
         * indexed out of bounds. omp_get_thread_limit() instead
         * reflects the program-wide ceiling on OpenMP threads
         * (bounded by OMP_THREAD_LIMIT if set), which is stable
         * regardless of any of that -- it also needs no parallel
         * region of its own, so it stays correct regardless of
         * whether construction happens inside or outside one (a
         * concern for omp_get_max_threads() too: constructing from
         * inside an already-active parallel region makes a fresh,
         * nested one collapse to a team of one by default, sizing
         * this ThreadVec for a single thread against the *outer*
         * team's own larger omp_get_thread_num() range).
         *
         * Most runtimes report a very large (effectively unbounded)
         * sentinel for omp_get_thread_limit() when OMP_THREAD_LIMIT
         * isn't set, so the result is clamped against a
         * hardware_concurrency-based cap (see
         * threadVecCapOversubscription) to avoid allocating an
         * enormous vector in the common case.
         */
        ThreadVec() {
#ifdef _OPENMP
            const int limit = omp_get_thread_limit();
            const int cap = static_cast<int>(std::max<unsigned>(1, std::thread::hardware_concurrency()))
                * threadVecCapOversubscription;
            obj_.resize(static_cast<std::size_t>(limit > 0 && limit < cap ? limit : cap));
#else
            obj_.resize(1);
#endif
        }
        ~ThreadVec() = default;

        // Disable copy and move constructors and assignment operators
        ThreadVec(const ThreadVec&) = delete;
        ThreadVec(ThreadVec&&) = delete;
        auto operator=(const ThreadVec&) -> ThreadVec& = delete;
        auto operator=(ThreadVec&&) -> ThreadVec& = delete;

        /**
         * @brief Return the number of distinct objects stored
         * @return The number of objects stored
         */
        [[nodiscard]] auto size() const { return obj_.size(); }
    
        /**
         * @fn utils::ThreadVec::operator()()
         * @brief Return the element private to this thread
         * @return Return the element of the ThreadVec private to this thread
         */
        auto operator()() -> auto& {
#ifdef _OPENMP
            const int ithread = omp_get_thread_num();
#else
            const int ithread = 0;
#endif
            assert(ithread >= 0 && static_cast<std::size_t>(ithread) < obj_.size());
            return obj_[ithread]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- hot path, must stay unchecked for performance; bounds asserted just above
        }

        /**
         * @fn utils::ThreadVec::operator()() const
         * @brief Return the element private to this thread
         * @return Return the element of the ThreadVec private to this thread
         */
        auto operator()() const -> const auto& {
#ifdef _OPENMP
            const int ithread = omp_get_thread_num();
#else
            const int ithread = 0;
#endif
            assert(ithread >= 0 && static_cast<std::size_t>(ithread) < obj_.size());
            return obj_[ithread]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- hot path, must stay unchecked for performance; bounds asserted just above
        }

        /**
         * @fn utils::ThreadVec::operator[](int)
         * @brief Return the object belonging to the specified thread
         * @param i The thread number whose object should be returned
         * @return The object belonging to thread i
         */
        auto operator[] (const int i) -> auto& {
            assert(i >= 0 && static_cast<std::size_t>(i) < obj_.size());
            return obj_[i]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- hot path, must stay unchecked for performance; bounds asserted just above
        }

        /**
         * @fn utils::ThreadVec::operator[](int) const
         * @brief Return the object belonging to the specified thread
         * @param i The thread number whose object should be returned
         * @return The object belonging to thread i
         */
        auto operator[] (const int i) const -> const auto& {
            assert(i >= 0 && static_cast<std::size_t>(i) < obj_.size());
            return obj_[i]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- hot path, must stay unchecked for performance; bounds asserted just above
        }

        /**
         * @brief Return the beginning of the thread vector
         */
        auto begin() { return obj_.begin(); }

        /**
         * @brief Return the end of the thread vector
         */
        auto end() { return obj_.end(); }
         
    private:

        std::vector<T> obj_;  /**< The stored objects */

    };


} // namespace utils

#endif // THREADVEC_HPP