/**
 * @file ThreadVec.hpp
 * @author Mark Krumholz
 * @brief Utility class for handling multithreaded runs
 * @date 2024-06-17
 */

#ifndef THREADVEC_HPP
#define THREADVEC_HPP

#ifdef _OPENMP
#   include <omp.h>
#endif
#include <vector>

namespace utils {

    /**
     * @class ThreadVec
     * @brief A class to automate private thread copies of objects
     * @tparam T Type of object the ThreadVec holds
     * @details
     * This class is a container to automate the process of each thread
     * having its own private copy of an object. It is a vector that on
     * creation automatically resizes itself to the number of threads in
     * the openMP threadpool, and when accessed by a thread
     * automatically returns the element of the vector that is private
     * to that thread.
     */
    template <class T> class ThreadVec {

    public:

        /**
         * @brief Construct an empty ThreadVec
         * @details
         * Sized via omp_get_max_threads() rather than by spawning a
         * parallel region and reading omp_get_num_threads() from
         * inside it: a ThreadVec can be constructed from inside an
         * already-active parallel region (e.g. one built fresh by
         * some per-thread work item), in which case that inner region
         * would be nested and, since nested parallelism is inactive
         * by default, would collapse to a team of one, sizing this
         * ThreadVec for a single thread. Every later access is keyed
         * by omp_get_thread_num() relative to the outer (real) team,
         * so that undersized ThreadVec would then be indexed out of
         * bounds. omp_get_max_threads() needs no parallel region of
         * its own and reflects the team size any (non-nested)
         * parallel region will actually use, so it stays correct
         * regardless of whether construction happens inside or
         * outside one.
         */
        ThreadVec() {
#ifdef _OPENMP
            obj_.resize(omp_get_max_threads());
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
            return obj_[ithread]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- hot path, must stay unchecked for performance
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
            return obj_[ithread]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- hot path, must stay unchecked for performance
        }

        /**
         * @fn utils::ThreadVec::operator[](int)
         * @brief Return the object belonging to the specified thread
         * @param i The thread number whose object should be returned
         * @return The object belonging to thread i
         */
        auto operator[] (const int i) -> auto& {
            return obj_[i]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- hot path, must stay unchecked for performance
        }

        /**
         * @fn utils::ThreadVec::operator[](int) const
         * @brief Return the object belonging to the specified thread
         * @param i The thread number whose object should be returned
         * @return The object belonging to thread i
         */
        auto operator[] (const int i) const -> const auto& {
            return obj_[i]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- hot path, must stay unchecked for performance
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