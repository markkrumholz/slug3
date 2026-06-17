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
         */
        ThreadVec() {
#ifdef _OPENMP
#pragma omp parallel
        {
            const int nthreads = omp_get_num_threads();
#pragma omp single
	        obj.resize(nthreads);
        }
#else
            obj.resize(1);
#endif
        }
        ~ThreadVec() = default;

        /**
         * @brief Return the number of distinct objects stored
         * @return The number of objects stored
         */
        size_t size() const { return obj.size(); }
    
        /**
         * @fn utils::ThreadVec::operator()()
         * @brief Return the element private to this thread
         * @return Return the element of the ThreadVec private to this thread
         */
        T& operator()() {
#ifdef _OPENMP
            const int ithread = omp_get_thread_num();
#else
            const int ithread = 0;
#endif
            return obj[ithread];
        }

        /**
         * @fn utils::ThreadVec::operator()() const
         * @brief Return the element private to this thread
         * @return Return the element of the ThreadVec private to this thread
         */
        const T& operator()() const {
#ifdef _OPENMP
            const int ithread = omp_get_thread_num();
#else
            const int ithread = 0;
#endif
            return obj[ithread];
        }

        /**
         * @fn utils::ThreadVec::operator[](int)
         * @brief Return the object belonging to the specified thread
         * @param i The thread number whose object should be returned
         * @return The object belonging to thread i
         */
        T& operator[] (const int i) {
            return obj[i];
        }

        /**
         * @fn utils::ThreadVec::operator[](int) const
         * @brief Return the object belonging to the specified thread
         * @param i The thread number whose object should be returned
         * @return The object belonging to thread i
         */
        const T& operator[] (const int i) const {
            return obj[i];
        }

        /**
         * @brief Return the beginning of the thread vector
         */
        auto begin() { return obj.begin(); }

        /**
         * @brief Return the end of the thread vector
         */
        auto end() { return obj.end(); }
         
    private:

        std::vector<T> obj;  /**< The stored objects */

    };


} // namespace utils

#endif // THREADVEC_HPP