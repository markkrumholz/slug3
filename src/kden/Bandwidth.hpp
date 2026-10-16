/**
 * @file Bandwidth.hpp
 * @author Mark Krumholz
 * @brief An inverse squared bandwidth matrix, defining a distance metric
 * @date 2026-10-15
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#ifndef BANDWIDTH_HPP
#define BANDWIDTH_HPP

#include <Eigen/Dense> // NOLINT(misc-include-cleaner) -- provides Eigen::MatrixXd/VectorXd/LLT/LDLT; clang-tidy's IWYU mapping doesn't know this header
#include <cstddef>
#include <functional>
#include <span>
#include <utility>
#include <vector>

namespace kden {

    /**
     * @class Bandwidth
     * @brief An inverse squared bandwidth matrix, and the operations that use it as a metric
     * @details
     * Holds an n x n inverse squared bandwidth matrix M (units of
     * inverse length squared), which defines the squared distance
     * between two points as d^T M d, d = x1 - x2. M acts as an
     * inverse covariance matrix: for a diagonal bandwidth h_i in each
     * dimension, its diagonal entries are 1 / h_i^2; in general it may
     * be any symmetric positive semi-definite matrix, and need not be
     * diagonal.
     *
     * Some dimensions may be unspecified: a coordinate that is NaN in
     * either point contributes no separation. With s the specified and
     * f the unspecified dimensions, the distance minimized over the
     * unspecified coordinates -- equivalently, the metric of the
     * Gaussian marginalized over them -- is given not by the s-s block
     * of M but by its Schur complement M_ss - M_sf M_ff^-1 M_fs (the
     * inverse of the s-s block of the covariance). reduce() returns a
     * Bandwidth holding that reduced matrix, and remembers which
     * dimensions it treats as unspecified.
     *
     * Every Bandwidth also precomputes what a kd tree search needs to
     * bound the distance from a search point to a box (see
     * BoxBounder), and whether its matrix is diagonal, in which case
     * distances and bounds cost O(n) rather than O(n^2). It is
     * immutable once built, so a single Bandwidth may be shared by any
     * number of threads.
     */
    class Bandwidth
    {
    public:

        /**
         * @brief Construct from an inverse squared bandwidth matrix
         * @param h2Inv The n x n matrix; must be symmetric and positive
         *   semi-definite. No dimension is unspecified.
         * @throws std::invalid_argument if h2Inv is not square, is empty,
         *   or is not symmetric
         * @throws std::domain_error if h2Inv is not positive semi-definite
         */
        explicit Bandwidth(Eigen::MatrixXd h2Inv); // NOLINT(misc-include-cleaner) -- Eigen::MatrixXd is provided by <Eigen/Dense> (included above); clang-tidy's IWYU mapping doesn't know that

        /**
         * @brief The Euclidean metric
         * @param nDim Number of dimensions
         * @return A Bandwidth holding the nDim x nDim identity matrix
         */
        [[nodiscard]] static auto identity(std::size_t nDim) -> Bandwidth;

        /** @brief Number of dimensions
         *  @return n */
        [[nodiscard]] auto nDim() const -> std::size_t { return unspecified_.size(); }
        /** @brief The inverse squared bandwidth matrix
         *  @return The n x n matrix; for a reduced Bandwidth, its rows and
         *    columns for unspecified dimensions are zero */
        [[nodiscard]] auto h2Inv() const -> const Eigen::MatrixXd& { return h2Inv_; }
        /** @brief Which dimensions are unspecified
         *  @return One flag per dimension, true if unspecified (all false
         *    unless this Bandwidth came from reduce()) */
        [[nodiscard]] auto unspecified() const -> const std::vector<bool>& { return unspecified_; }
        /** @brief Whether any dimension is unspecified
         *  @return True if this Bandwidth came from reduce() with at
         *    least one unspecified dimension */
        [[nodiscard]] auto isReduced() const -> bool { return spec_.size() != unspecified_.size(); }
        /** @brief Whether the matrix is diagonal (over the specified dimensions)
         *  @return True if distances and bounds use the O(n) diagonal path */
        [[nodiscard]] auto isDiagonal() const -> bool { return diagonal_; }
        /** @brief Number of specified dimensions
         *  @return The number of dimensions not flagged in unspecified() */
        [[nodiscard]] auto nSpecified() const -> std::size_t { return spec_.size(); }
        /** @brief Determinant of the matrix over the specified dimensions
         *  @return det M_s, with M_s the matrix's block over the specified
         *    dimensions (for a reduced Bandwidth, the Schur complement;
         *    see reduce()); 1 if no dimension is specified, and 0 (to
         *    rounding) if M_s is singular. The integral of
         *    exp(-d^T M_s d) over the specified dimensions is
         *    pi^(nSpecified / 2) / sqrt(det()). */
        [[nodiscard]] auto det() const -> double { return det_; }

        /**
         * @brief Reduce the metric for points with some coordinates unspecified
         * @param unspecified Which dimensions will be unspecified; must
         *   have n elements, and include every dimension this
         *   Bandwidth already treats as unspecified
         * @return A Bandwidth whose matrix, over the dimensions s that
         *   remain specified, is the Schur complement
         *   M_ss - M_sf M_ff^-1 M_fs of this one's (computed by
         *   Cholesky factorization of M_ff), and is zero elsewhere
         * @throws std::invalid_argument if unspecified has the wrong
         *   size or omits a dimension this Bandwidth leaves unspecified
         * @throws std::domain_error if the block of the matrix over the
         *   newly unspecified dimensions is not positive definite
         * @details
         * Costs O(n^3); for many searches with the same metric and the
         * same unspecified dimensions, reduce once and reuse the result.
         */
        [[nodiscard]] auto reduce(const std::vector<bool>& unspecified) const -> Bandwidth;

        /**
         * @brief Convolve with another bandwidth
         * @param other The other Bandwidth; must have the same number of
         *   dimensions, and the same unspecified dimensions (see
         *   unspecified())
         * @return The Bandwidth of the convolution of the two kernels,
         *   whose matrix, over the specified dimensions, is
         *   (M1^-1 + M2^-1)^-1, with M1 and M2 the two matrices there
         * @throws std::invalid_argument if other has a different number
         *   of dimensions or different unspecified dimensions
         * @throws std::domain_error if M1 + M2 is not positive definite
         *   (the two kernels are both infinitely broad along some
         *   direction)
         * @details
         * Convolving two Gaussians adds their covariances, and the
         * covariance is proportional to the inverse of the matrix, so
         * this is, e.g., the bandwidth of a kernel density estimate
         * smoothed by Gaussian measurement errors. It is computed as
         * M1 (M1 + M2)^-1 M2, by Cholesky factorization of M1 + M2,
         * which equals (M1^-1 + M2^-1)^-1 when both are invertible, but
         * needs neither to be: a singular matrix (a kernel infinitely
         * broad along some direction) gives a singular result. Costs
         * O(n^3).
         */
        [[nodiscard]] auto convolve(const Bandwidth& other) const -> Bandwidth;

        /**
         * @brief Which elements of a point are NaN
         * @param x The point
         * @return One flag per element, true if it is NaN
         */
        [[nodiscard]] static auto nanPattern(std::span<const double> x) -> std::vector<bool>;

        /**
         * @brief Check whether a point's unspecified coordinates match this Bandwidth's
         * @param x The point
         * @return True if x has n elements, NaN in exactly the dimensions
         *   this Bandwidth treats as unspecified
         */
        [[nodiscard]] auto matches(std::span<const double> x) const -> bool;

        /**
         * @brief Squared distance between two points
         * @param x1 Position of the first point; must have n elements,
         *   which may be NaN to leave a coordinate unspecified
         * @param x2 Position of the second point, likewise
         * @return d^T M d, with d_i = x1[i] - x2[i], or 0 if either is NaN
         * @details
         * For a metric that is not diagonal, NaN coordinates give the
         * distance minimized over the unspecified coordinates only if
         * this Bandwidth has been reduced for them (see reduce()).
         * Allocates a scratch vector; the overload taking one avoids
         * that, for use in loops.
         */
        [[nodiscard]] auto dist2(std::span<const double> x1,
            std::span<const double> x2) const -> double;

        /**
         * @brief Squared distance between two points, using caller-provided scratch space
         * @param x1 As for the other overload
         * @param x2 As for the other overload
         * @param scratch Scratch space; resized as needed, so may be
         *   reused across calls without further allocation
         * @return As for the other overload
         */
        [[nodiscard]] auto dist2(std::span<const double> x1,
            std::span<const double> x2, std::vector<double>& scratch) const -> double;

        /**
         * @class BoxBounder
         * @brief Rigorous lower and upper bounds on the distance from one point to axis-aligned boxes
         * @details
         * Built once per search point (see Bandwidth::boxBounder()).
         * With s the specified dimensions, the bound uses a factor W of
         * the matrix's s-s block, M_s = W^T W, from Eigen's pivoted
         * LDL^T factorization M_s = P^T L D L^T P, W = D^1/2 L^T P. For
         * any y, (x - y)^T M_s (x - y) = sum_k (w_k . (x - y))^2, where
         * w_k are the rows of W. Over a box with center c and
         * half-widths h, w_k . y covers exactly the interval
         * w_k . c +- |w_k| . h, so the minimum of each term over the box
         * is the squared gap between w_k . x and that interval (zero if
         * it lies inside); their sum is a rigorous lower bound on the
         * minimum of the sum, and exact for a diagonal metric (which
         * uses a direct O(n) path instead). Likewise the maximum of each
         * term is the squared distance from w_k . x to the far end of
         * the interval, (|w_k . x - w_k . c| + |w_k| . h)^2, and their
         * sum is a rigorous upper bound on the maximum of the sum (the
         * per-term maxima generally occur at different points of the
         * box), again exact for a diagonal metric, where it is the
         * distance to the box's farthest corner. A triangular, Cholesky-type
         * factor is used rather than the metric's principal axes
         * because it stays close to the coordinate axes as the metric
         * approaches diagonal, keeping the bound tight; the principal
         * axes can rotate arbitrarily when eigenvalues are nearly
         * equal. For non-diagonal metrics the lower bound is shrunk,
         * and the upper bound inflated, by a relative 1e-10, so that
         * rounding in the factorization can never carry either past a
         * true distance. Holds scratch space, so must not be shared between
         * threads; must not outlive the Bandwidth it came from.
         */
        class BoxBounder
        {
        public:

            /**
             * @brief Lower bound on the squared distance from the point to a box
             * @param lo Lower corner of the box (n elements)
             * @param hi Upper corner of the box (n elements)
             * @return A value no larger than the squared distance (as
             *   computed by dist2()) from the point to any point in the box
             */
            [[nodiscard]] auto lower(std::span<const double> lo,
                std::span<const double> hi) const -> double;

            /**
             * @brief Upper bound on the squared distance from the point to a box
             * @param lo Lower corner of the box (n elements)
             * @param hi Upper corner of the box (n elements)
             * @return A value no smaller than the squared distance (as
             *   computed by dist2()) from the point to any point in the box
             */
            [[nodiscard]] auto upper(std::span<const double> lo,
                std::span<const double> hi) const -> double;

            /**
             * @brief Lower and upper bounds together
             * @param lo Lower corner of the box (n elements)
             * @param hi Upper corner of the box (n elements)
             * @return The pair (lower(lo, hi), upper(lo, hi)), computed
             *   with a single pass over the box, so cheaper than calling
             *   both separately
             */
            [[nodiscard]] auto bounds(std::span<const double> lo,
                std::span<const double> hi) const -> std::pair<double, double>;

        private:

            friend class Bandwidth;

            /**
             * @brief Construct for a given point
             * @param bandwidth The Bandwidth; must outlive this
             * @param x The point; NaN in the bandwidth's unspecified dimensions
             */
            BoxBounder(const Bandwidth& bandwidth, std::span<const double> x);

            /**
             * @brief Shared implementation of lower(), upper(), and bounds()
             * @tparam WantLower Whether to compute the lower bound
             * @tparam WantUpper Whether to compute the upper bound
             * @param lo Lower corner of the box (n elements)
             * @param hi Upper corner of the box (n elements)
             * @return (lower bound, upper bound); an element that was not
             *   requested is zero
             */
            template <bool WantLower, bool WantUpper>
            auto compute(std::span<const double> lo,
                std::span<const double> hi) const -> std::pair<double, double>;

            std::reference_wrapper<const Bandwidth> bandwidth_;  /**< The metric */
            Eigen::VectorXd xs_;              /**< The point, specified dimensions only */ // NOLINT(misc-include-cleaner) -- Eigen::VectorXd is provided by <Eigen/Dense> (included above); clang-tidy's IWYU mapping doesn't know that
            Eigen::VectorXd wx_;              /**< W times xs_ (non-diagonal metrics only) */
            mutable Eigen::VectorXd center_;  /**< Scratch: a box's center */
            mutable Eigen::VectorXd half_;    /**< Scratch: a box's half-widths */
        };

        /**
         * @brief Make a box bounder for a search point
         * @param x The search point; must match this Bandwidth (see matches())
         * @return A BoxBounder for x, which must not outlive this Bandwidth
         */
        [[nodiscard]] auto boxBounder(std::span<const double> x) const -> BoxBounder;

    private:

        /**
         * @brief Construct with given unspecified dimensions (used by reduce())
         * @param h2Inv The matrix, zero in the unspecified rows and columns
         * @param unspecified Which dimensions are unspecified
         */
        Bandwidth(Eigen::MatrixXd h2Inv, std::vector<bool> unspecified);

        /**
         * @brief Validate the matrix and precompute the diagonal flag and box-bound factor
         * @throws std::domain_error if the specified block is not
         *   positive semi-definite
         */
        void prepare();

        Eigen::MatrixXd h2Inv_;           /**< The inverse squared bandwidth matrix */
        std::vector<bool> unspecified_;   /**< Which dimensions are unspecified */
        std::vector<std::size_t> spec_;   /**< The specified dimensions, in order */
        Eigen::MatrixXd ms_;              /**< h2Inv_'s block over the specified dimensions, contiguous */
        Eigen::VectorXd diag_;            /**< Diagonal of ms_ */
        bool diagonal_ = false;           /**< Whether ms_ is diagonal */
        double det_ = 1.0;                /**< Determinant of ms_ */
        Eigen::MatrixXd w_;               /**< Box-bound factor W (M_s = W^T W), over the specified dimensions; empty if diagonal */
        Eigen::MatrixXd absW_;            /**< Element-wise absolute value of W */
    };

} // namespace kden

#endif // BANDWIDTH_HPP
