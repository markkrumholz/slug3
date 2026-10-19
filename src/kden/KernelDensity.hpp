/**
 * @file KernelDensity.hpp
 * @author Mark Krumholz
 * @brief Kernel density estimation on a kd tree of weighted points
 * @date 2026-10-16
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#ifndef KERNELDENSITY_HPP
#define KERNELDENSITY_HPP

#include "Bandwidth.hpp"
#include "KDTree.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <exception>
#include <limits>
#include <memory>
#include <numbers>
#include <optional>
#include <span>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace kden {

    /**
     * @struct KernelDensityData
     * @brief Weighted points for a KernelDensity: a KDTreeData plus one weight per point
     * @tparam AuxType Type of the auxiliary data carried by each point,
     *   or void (the default) for none; see KDTreeData
     * @details
     * Adds a flat array of weights (w_, N doubles) to KDTreeData's
     * positions and auxiliary data, and extends swap() and validate()
     * to cover it, so that building a tree over the points reorders
     * the weights along with them. Like KDTreeData, a non-owning view
     * of the caller's storage (see OwnedKernelDensityData).
     */
    template <typename AuxType = void>
    struct KernelDensityData : public KDTreeData<AuxType>
    {
        std::span<double> w_;  /**< Weights, one per point (not owned) */

        /**
         * @brief Weight of a point
         * @param i Index of the point
         * @return Its weight
         */
        [[nodiscard]] auto weight(const std::size_t i) const -> double { return w_[i]; } // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)

        /**
         * @brief Check that the arrays are consistent with one another, and finite
         * @throws std::invalid_argument for any of the reasons
         *   KDTreeData::validate() does, or if w_ does not have one
         *   element per point, or any weight is negative or not finite
         * @details
         * Non-finite weights would make the nodes' weight sums
         * non-finite, and the kernel density estimates bound a node's
         * contribution by its total weight times the range of kernel
         * values over its bounding box, which holds only if no weight
         * is negative.
         */
        void validate() const
        {
            KDTreeData<AuxType>::validate();
            if (w_.size() != this->size())
            {
                throw std::invalid_argument("KernelDensityData: w_ must hold one element per point");
            }
            if (!std::ranges::all_of(w_, [](const double v) -> bool { return std::isfinite(v) && v >= 0.0; }))
            {
                throw std::invalid_argument("KernelDensityData: weights must be finite and non-negative");
            }
        }

        /**
         * @brief Exchange two points: their positions, auxiliary data, and weights
         * @param i Index of one point
         * @param j Index of the other
         */
        void swap(const std::size_t i, const std::size_t j)
        {
            KDTreeData<AuxType>::swap(i, j);
            std::swap(w_[i], w_[j]); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
        }
    };

    /**
     * @struct OwnedKernelDensityData
     * @brief Storage for a set of weighted points: owning vectors, viewed by a KernelDensityData
     * @tparam AuxType As for KDTreeData
     * @details
     * Adds owned weights to OwnedKDTreeData; view() returns a
     * KernelDensityData over the vectors, valid on the same terms.
     */
    template <typename AuxType = void>
    struct OwnedKernelDensityData : public OwnedKDTreeData<AuxType>
    {
        std::vector<double> w_;  /**< Weights, one per point */

        /**
         * @brief A view of these points
         * @return A KernelDensityData viewing x_, aux_, and w_
         */
        [[nodiscard]] auto view() -> KernelDensityData<AuxType>
        {
            KernelDensityData<AuxType> v;
            static_cast<KDTreeData<AuxType>&>(v) = OwnedKDTreeData<AuxType>::view();
            v.w_ = std::span<double>(w_);
            return v;
        }
    };

    /**
     * @class RefinementList
     * @brief The list of tree nodes a kernel density evaluation is refining, as a sum tree
     * @details
     * Holds a set of entries, each a tree node with a central estimate
     * of some quantity and an uncertainty on it, and provides, each in
     * O(1), the sums of all entries' estimates and uncertainties and
     * the entry with the largest uncertainty; inserting, replacing,
     * and erasing an entry each cost O(log n).
     *
     * Entries live in slots, the leaves of a complete binary tree
     * stored as an implicit array (internal node i has children 2i
     * and 2i + 1, and slot s is node capacity + s). Every internal
     * node holds the sums of its two children's estimates and
     * uncertainties, and which of the slots below it has the largest
     * uncertainty; changing a slot recomputes the nodes on its path to
     * the root. Each sum is thus always recomputed from its two
     * children's current values, never updated by adding and
     * subtracting changes, so its rounding error is bounded by a few
     * ulps per level of the current values, however many changes have
     * been made. A running total, by contrast, accumulates error from
     * every term ever added and removed: refinement removes parents
     * whose uncertainties can exceed the final one by many orders of
     * magnitude, so this error can grow to swamp the tolerance and
     * keep it from ever being met. Freed slots are reused, and the
     * capacity doubles when every slot is in use.
     */
    class RefinementList
    {
    public:

        /**
         * @brief Construct an empty list
         * @param capacity Initial number of slots; rounded up to a
         *   power of two (at least 1)
         */
        explicit RefinementList(const std::size_t capacity = 64)
        {
            reset(std::bit_ceil(std::max<std::size_t>(capacity, 1)));
        }

        /** @brief Sum of the entries' estimates
         *  @return The sum (zero if empty) */
        [[nodiscard]] auto estimate() const -> double { return est_[1]; } // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
        /** @brief Sum of the entries' uncertainties
         *  @return The sum (zero if empty) */
        [[nodiscard]] auto uncertainty() const -> double { return err_[1]; } // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
        /** @brief The slot with the largest uncertainty
         *  @return Its index; if the list is empty, or all uncertainties
         *    are zero, a slot whose uncertainty is zero */
        [[nodiscard]] auto worst() const -> std::size_t { return arg_[1] - capacity_; } // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
        /** @brief Uncertainty of the entry in a slot
         *  @param slot The slot
         *  @return Its uncertainty (zero if the slot is empty) */
        [[nodiscard]] auto uncertainty(const std::size_t slot) const -> double { return err_[capacity_ + slot]; } // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
        /** @brief Tree node of the entry in a slot
         *  @param slot The slot; must hold an entry
         *  @return The node's index */
        [[nodiscard]] auto node(const std::size_t slot) const -> std::size_t { return node_[slot]; } // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
        /** @brief Number of entries
         *  @return The number of slots in use */
        [[nodiscard]] auto size() const -> std::size_t { return used_ - free_.size(); }

        /**
         * @brief Add an entry, in a free slot
         * @param node Its tree node
         * @param est Its estimate
         * @param err Its uncertainty; must be non-negative
         */
        void insert(const std::size_t node, const double est, const double err)
        {
            std::size_t slot = 0;
            if (!free_.empty())
            {
                slot = free_.back();
                free_.pop_back();
            }
            else
            {
                if (used_ == capacity_) { grow(); }
                slot = used_++;
            }
            replace(slot, node, est, err);
        }

        /**
         * @brief Replace the entry in a slot
         * @param slot The slot; must hold an entry
         * @param node The new entry's tree node
         * @param est Its estimate
         * @param err Its uncertainty; must be non-negative
         */
        void replace(const std::size_t slot, const std::size_t node, const double est, const double err)
        {
            node_[slot] = node; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
            set(slot, est, err);
        }

        /**
         * @brief Remove the entry in a slot, freeing the slot for reuse
         * @param slot The slot; must hold an entry
         */
        void erase(const std::size_t slot)
        {
            set(slot, 0.0, 0.0);
            free_.push_back(slot);
        }

    private:

        /**
         * @brief Set a slot's values and recompute the nodes above it
         * @param slot The slot
         * @param est Its estimate
         * @param err Its uncertainty
         */
        void set(const std::size_t slot, const double est, const double err)
        {
            std::size_t i = capacity_ + slot;
            est_[i] = est; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
            err_[i] = err; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
            for (i >>= 1; i > 0; i >>= 1) { pull(i); }
        }

        /**
         * @brief Recompute an internal node from its two children
         * @param i The node
         */
        void pull(const std::size_t i)
        {
            const std::size_t l = 2 * i;
            const std::size_t r = l + 1;
            est_[i] = est_[l] + est_[r]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
            err_[i] = err_[l] + err_[r]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
            arg_[i] = err_[arg_[l]] >= err_[arg_[r]] ? arg_[l] : arg_[r]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
        }

        /**
         * @brief Set up an empty tree with a given number of slots
         * @param capacity The number of slots; a power of two
         */
        void reset(const std::size_t capacity)
        {
            capacity_ = capacity;
            est_.assign(2 * capacity, 0.0);
            err_.assign(2 * capacity, 0.0);
            arg_.assign(2 * capacity, 0);
            node_.assign(capacity, 0);
            for (std::size_t s = 0; s < capacity; ++s) { arg_[capacity + s] = capacity + s; } // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
            for (std::size_t i = capacity - 1; i > 0; --i) { pull(i); }
        }

        /**
         * @brief Double the number of slots, keeping every entry in its slot
         */
        void grow()
        {
            const std::vector<double> est(est_.begin() + static_cast<std::ptrdiff_t>(capacity_), est_.end());
            const std::vector<double> err(err_.begin() + static_cast<std::ptrdiff_t>(capacity_), err_.end());
            const std::vector<std::size_t> node = node_;
            const std::size_t old = capacity_;
            reset(2 * old);
            for (std::size_t s = 0; s < old; ++s)
            {
                est_[capacity_ + s] = est[s]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
                err_[capacity_ + s] = err[s]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
                node_[s] = node[s]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
            }
            for (std::size_t i = capacity_ - 1; i > 0; --i) { pull(i); }
        }

        std::size_t capacity_ = 0;          /**< Number of slots; a power of two */
        std::size_t used_ = 0;              /**< Number of slots ever used (in use or free) */
        std::vector<double> est_;           /**< Estimates: sums over internal nodes 1 to capacity_ - 1, slots' own values from capacity_ on */
        std::vector<double> err_;           /**< Uncertainties, likewise */
        std::vector<std::size_t> arg_;      /**< For each node, the position (capacity_ + slot) of the slot below it with the largest uncertainty */
        std::vector<std::size_t> node_;     /**< Tree node of each slot's entry */
        std::vector<std::size_t> free_;     /**< Freed slots, for reuse */
    };

    template <typename AuxType>
    class ConditionalDensity;

    /**
     * @class KernelDensity
     * @brief A kernel density estimate built on a kd tree of weighted points
     * @tparam AuxType Type of the auxiliary data each point carries, or
     *   void (the default) for none; see KDTreeData
     * @details
     * A KDTree whose points, a KernelDensityData, are the weighted
     * kernel centers of a kernel density estimate. Like KDTree, it is a
     * non-owning wrapper around its points, which it reorders in place
     * on construction. On top of the tree it keeps, for every node, the
     * sum of its points' weights, in a flat array indexed by node.
     *
     * The weights are normalized on construction, to sum to one. The
     * kernel is a Gaussian, K(d) = exp(-d^T M d), with M the inverse
     * squared bandwidth matrix (see Bandwidth), which is passed to each
     * evaluation, so different threads may evaluate the same estimate
     * with different bandwidths at once. The estimate at x is
     * p(x) = A sum_i w_i K(x - x_i), with A = sqrt(det M) / pi^(n / 2)
     * normalizing each kernel, and so the estimate, to unit integral.
     */
    template <typename AuxType = void>
    class KernelDensity : public KDTree<KernelDensityData<AuxType>>
    {
    public:

        /** The kd tree this is built on */
        using Tree = KDTree<KernelDensityData<AuxType>>;

        /**
         * What conditionalSample() returns: a conditional estimate's
         * points (K x nR, point-major), their weights (K), and their
         * auxiliary data (K elements, carried over from this estimate's
         * points; an empty placeholder if AuxType is void)
         */
        using Sample = std::tuple<std::vector<double>, std::vector<double>,
            typename KDTreeAuxStorage<AuxType>::Owned>;

        /**
         * @brief Construct a KernelDensity over a set of weighted points
         * @param data The kernel centers; as for the corresponding KDTree
         *   constructor. Every weight must also be finite and
         *   non-negative (see KernelDensityData::validate()), and they
         *   must not all be zero. Besides being reordered, the weights
         *   are rescaled in place to sum to one.
         * @param leafSize As for the corresponding KDTree constructor
         * @param splitBandwidth As for the corresponding KDTree constructor
         * @param noSplitDims As for the corresponding KDTree constructor;
         *   e.g., the dimensions that evaluations of a marginal density
         *   will leave unspecified
         * @throws std::invalid_argument as for the corresponding KDTree
         *   constructor, or if the weights sum to zero
         */
        KernelDensity(KernelDensityData<AuxType>& data,
            const std::size_t leafSize,
            const Bandwidth& splitBandwidth,
            const std::vector<std::size_t>& noSplitDims = {}) :
            Tree(normalizeWeights(data), leafSize, splitBandwidth, noSplitDims)
        {
            setNodeWeights();
        }

        /**
         * @brief Construct a KernelDensity, choosing split dimensions with the Euclidean metric
         * @param data As for the other constructor
         * @param leafSize As for the other constructor
         * @param noSplitDims As for the other constructor
         * @throws std::invalid_argument as for the other constructor
         */
        explicit KernelDensity(KernelDensityData<AuxType>& data, const std::size_t leafSize = 16,
            const std::vector<std::size_t>& noSplitDims = {}) :
            Tree(normalizeWeights(data), leafSize, noSplitDims)
        {
            setNodeWeights();
        }

        /**
         * @brief Sum of the weights of a node's points
         * @param idx Index of the node
         * @return The sum
         */
        [[nodiscard]] auto nodeW(const std::size_t idx) const -> double { return nodeW_[idx]; } // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)

        /**
         * @brief Bring the estimate up to date after its weights are changed in place
         * @throws std::invalid_argument if the weights are invalid (see
         *   KernelDensityData::validate()) or all zero
         * @details
         * The weights may be changed in place (but the positions may
         * not: the tree is built around them). Afterwards, call this to
         * renormalize the weights to sum to one, in place, and
         * recompute every node's weight sum. Costs O(N), in parallel if
         * built with OpenMP; unlike construction, it checks only the
         * weights, not the (unchanged) positions. Not thread-safe with
         * respect to evaluation: like any non-const member, it must not
         * run concurrently with pdf() or conditional() on the same
         * estimate (nor may the weights be changed while they run).
         */
        void reweight()
        {
            normalizeWeightArray(this->data_.get().w_);
            setNodeWeights();
        }

        /**
         * @brief Evaluate the kernel density estimate at a point, to within a tolerance
         * @param x Position at which to evaluate it; must have nDim
         *   elements. Elements may be NaN, to evaluate the marginal
         *   density over the other dimensions.
         * @param relTol Relative tolerance: stop once the uncertainty is
         *   less than relTol times the estimate
         * @param absTol Absolute tolerance: stop once the estimate plus
         *   its uncertainty is less than absTol
         * @param bandwidth The kernel's inverse squared bandwidth matrix,
         *   for nDim dimensions; if x has NaN elements, it is reduced for
         *   them as in KDTree::neighbors()
         * @return The estimate p(x), normalized to unit integral over
         *   all space (over the specified dimensions only, if x has NaN
         *   elements; marginalizing a Gaussian kernel over the others
         *   gives the Gaussian with the reduced bandwidth -- see
         *   Bandwidth::reduce())
         * @throws std::invalid_argument if x or bandwidth is not for
         *   nDim dimensions, or bandwidth has been reduced for a
         *   different pattern of NaNs than x's
         * @throws std::domain_error if the kernel cannot be normalized
         *   (bandwidth is singular over the specified dimensions)
         * @details
         * Follows kd_pdf in slug2's bayesphot. A list of tree nodes,
         * starting with just the root, is kept, each with a central
         * estimate of its contribution to p(x) and an uncertainty on it
         * (see nodePdf()); together with the exactly known
         * contributions of the leaves reached so far, these sum to
         * p(x). Until the summed uncertainty meets either tolerance,
         * the node with the largest uncertainty is replaced by its two
         * children; a child that is a leaf goes straight into the
         * leaves' sum (a sum of non-negative terms only, so with no
         * cancellation to accumulate rounding error) rather than into
         * the list. This ends, at the latest, once every leaf has been
         * reached. The list is a RefinementList, so each step costs
         * O(log n) in the length of the list, while its sums never
         * accumulate rounding error the way running totals would.
         * Thread-safe, as for KDTree::neighbors().
         */
        [[nodiscard]] auto pdf(const std::vector<double>& x,
            const double relTol,
            const double absTol,
            const Bandwidth& bandwidth) const -> double
        {
            if (x.size() != this->nDim_ || bandwidth.nDim() != this->nDim_)
            {
                throw std::invalid_argument("KernelDensity::pdf: x and bandwidth must be "
                    "for nDim dimensions");
            }
            if (bandwidth.matches(x)) { return evalPdf(x, relTol, absTol, bandwidth); }
            if (bandwidth.isReduced())
            {
                throw std::invalid_argument("KernelDensity::pdf: bandwidth was reduced for "
                    "a different pattern of NaNs than x's");
            }
            return evalPdf(x, relTol, absTol, bandwidth.reduce(Bandwidth::nanPattern(x)));
        }

        /**
         * @brief Evaluate the kernel density estimate at a point, with the Euclidean bandwidth
         * @param x As for the other overload
         * @param relTol As for the other overload
         * @param absTol As for the other overload
         * @return As for the other overload, with M the identity
         * @throws As for the other overload
         */
        [[nodiscard]] auto pdf(const std::vector<double>& x,
            const double relTol,
            const double absTol) const -> double
        {
            return pdf(x, relTol, absTol, this->euclidean_);
        }

        /**
         * @brief Condition the estimate on values of some dimensions, optionally marginalizing over others
         * @param x0 One element per dimension: a finite value fixes that
         *   dimension (the set F); NaN keeps it as a dimension of the
         *   result (the set R). Elements for dimensions in marginalDims
         *   are ignored.
         * @param marginalDims Dimensions to marginalize over (the set G);
         *   may be empty
         * @param massTol Largest fraction of the conditional estimate's
         *   total weight that may be dropped, in [0, 1); 0 keeps every
         *   point with non-zero weight
         * @param bandwidth The kernel's inverse squared bandwidth
         *   matrix, for nDim dimensions; must not be reduced
         * @param leafSize Leaf size of the result's tree
         * @return The conditional estimate p(y | x0_F) over the
         *   dimensions in R, as a ConditionalDensity
         * @throws std::invalid_argument if x0 or bandwidth is not for
         *   nDim dimensions, bandwidth is reduced, marginalDims has a
         *   dimension not below nDim or a repeated one, F or R is empty,
         *   or massTol is not in [0, 1)
         * @throws std::domain_error if the bandwidth's blocks needed
         *   are not positive definite, or every point's conditional
         *   weight underflows to zero (x0_F is too far from every point)
         * @details
         * Marginalizing a Gaussian over G replaces M by its Schur
         * complement M' = bandwidth.reduce(G). Then, writing a = x0_F -
         * x_F and b = y - x_R for a point x, completing the square in b
         * gives
         *
         *   d^T M' d = (b + K a)^T M'_RR (b + K a) + a^T S a,
         *
         * with K = M'_RR^-1 M'_RF and S = M'_FF - M'_FR M'_RR^-1 M'_RF,
         * the reduction of M over G and R together. So the estimate at
         * (x0_F, y) is itself a kernel density estimate in y, with kernel
         * metric M'_RR, points y'_i = x_i,R + K (x_i,F - x0_F), and
         * weights w'_i = w_i exp(-a_i^T S a_i).
         *
         * Most points' weights w'_i are negligible, so only those that
         * matter are kept. The tree is walked as in pdf(), with each
         * node's total conditional weight bounded by its weight times
         * exp(-a^T S a) at the bounds of a^T S a over its box (see
         * Bandwidth::BoxBounder; dimensions in G and R are ignored, so
         * a tree built with them in noSplitDims suits these queries).
         * The node with the largest upper bound is opened until the
         * upper bounds of the unopened interior nodes sum to at most
         * massTol / 2 of the total weight, and those nodes are dropped;
         * the points of the leaves reached are then sorted by weight,
         * and the lightest dropped, up to another massTol / 2. The
         * weight dropped is thus at most massTol of the total, so the
         * result's L1 distance from the exact conditional density is at
         * most 2 massTol. Its cost depends on the number of points kept,
         * not the size of the tree. conditionalSample() finds the same
         * points and weights without building the result. Thread-safe,
         * as for pdf().
         */
        [[nodiscard]] auto conditional(const std::vector<double>& x0,
            const std::vector<std::size_t>& marginalDims,
            const double massTol,
            const Bandwidth& bandwidth,
            const std::size_t leafSize = 16) const -> ConditionalDensity<AuxType>
        {
            auto pts = conditionalPoints(x0, marginalDims, massTol, bandwidth);
            return ConditionalDensity<AuxType>(std::move(pts.data_), Bandwidth(pts.metric_),
                std::move(pts.dims_), pts.marginal_, leafSize);
        }

        /**
         * @brief Find the sample points, weights, and auxiliary data of a conditional estimate, without building it
         * @param x0 As for conditional()
         * @param marginalDims As for conditional()
         * @param massTol As for conditional()
         * @param bandwidth As for conditional()
         * @return The points the conditional estimate p(y | x0_F) keeps,
         *   in the dimensions R (in increasing order), K x |R|,
         *   point-major; and their weights, K of them, w'_i = w_i
         *   exp(-a_i^T S a_i) (see conditional()), with w_i this
         *   estimate's normalized weights: not themselves normalized, so
         *   that their sum times sqrt(det S) / pi^(|F|/2) is the marginal
         *   density over F at x0_F (conditional()'s marginal()); and,
         *   unless AuxType is void, their auxiliary data, carried over
         *   from the points they came from (see Sample)
         * @throws As for conditional()
         * @details
         * The same points, weights, and auxiliary data as
         * conditional() keeps, in increasing order of weight, without
         * building the conditional estimate's tree: a
         * KernelDensity over them, with the kernel metric M'_RR -- the
         * block of bandwidth.reduce(G) over R, a function of the
         * bandwidth alone -- is the conditional estimate. Suited to
         * storing conditional estimates, to be rebuilt as needed.
         * Thread-safe, as for conditional().
         */
        [[nodiscard]] auto conditionalSample(const std::vector<double>& x0,
            const std::vector<std::size_t>& marginalDims,
            const double massTol,
            const Bandwidth& bandwidth) const -> Sample
        {
            auto pts = conditionalPoints(x0, marginalDims, massTol, bandwidth);
            return {std::move(pts.data_.x_), std::move(pts.data_.w_), std::move(pts.data_.aux_)};
        }

        /**
         * @brief Find the sample points and weights of a conditional estimate, with the Euclidean bandwidth
         * @param x0 As for the other overload
         * @param marginalDims As for the other overload
         * @param massTol As for the other overload
         * @return As for the other overload, with M the identity
         * @throws As for the other overload
         */
        [[nodiscard]] auto conditionalSample(const std::vector<double>& x0,
            const std::vector<std::size_t>& marginalDims,
            const double massTol) const -> Sample
        {
            return conditionalSample(x0, marginalDims, massTol, this->euclidean_);
        }

    protected:

        /**
         * @struct ConditionalPoints
         * @brief What conditional() and conditionalSample() find: a conditional estimate's points, before its tree is built
         */
        struct ConditionalPoints
        {
            OwnedKernelDensityData<AuxType> data_;  /**< The kept points, in the dimensions R; their unnormalized weights; and their auxiliary data */
            std::vector<std::size_t> dims_;         /**< The kept dimensions R, in increasing order */
            Eigen::MatrixXd metric_;                /**< The conditional kernel metric M'_RR */ // NOLINT(misc-include-cleaner) -- Eigen::MatrixXd is provided by <Eigen/Dense> (included via Bandwidth.hpp); clang-tidy's IWYU mapping doesn't know that
            double marginal_ = 0.0;                 /**< The marginal density over F at x0_F */
        };

        /**
         * @brief Find a conditional estimate's points, weights, metric, and marginal
         * @param x0 As for conditional()
         * @param marginalDims As for conditional()
         * @param massTol As for conditional()
         * @param bandwidth As for conditional()
         * @return The points and the rest (see ConditionalPoints)
         * @throws As for conditional()
         * @details
         * The work of conditional() and conditionalSample(); see
         * conditional() for the method.
         */
        [[nodiscard]] auto conditionalPoints(const std::vector<double>& x0,
            const std::vector<std::size_t>& marginalDims,
            const double massTol,
            const Bandwidth& bandwidth) const -> ConditionalPoints
        {
            // Sort the dimensions into fixed (F), kept (R), and
            // marginalized (G)
            const std::size_t nDim = this->nDim_;
            if (x0.size() != nDim || bandwidth.nDim() != nDim)
            {
                throw std::invalid_argument("KernelDensity::conditional: x0 and bandwidth must be "
                    "for nDim dimensions");
            }
            if (bandwidth.isReduced())
            {
                throw std::invalid_argument("KernelDensity::conditional: bandwidth must not be reduced");
            }
            if (!(massTol >= 0.0) || !(massTol < 1.0))   // written to reject NaN too
            {
                throw std::invalid_argument("KernelDensity::conditional: massTol must be in [0, 1)");
            }
            auto [dimsF, dimsR, inG, inGR] = classifyDims(x0, marginalDims);

            // The metrics: M' over F and R, S over F, and the shift K
            const Bandwidth mPrime = bandwidth.reduce(inG);
            const Bandwidth weightMetric = bandwidth.reduce(inGR);
            const auto [kernelMetric, shift] = conditionalMetric(mPrime.h2Inv(), dimsF, dimsR);

            // Walk the tree, collecting the points of the leaves reached
            std::vector<double> xF(nDim, std::numeric_limits<double>::quiet_NaN());
            for (const std::size_t d : dimsF) { xF[d] = x0[d]; }
            const auto [kept, keptMass] = selectConditional(xF, weightMetric, massTol);
            if (!(keptMass > 0.0))
            {
                throw std::domain_error("KernelDensity::conditional: every point's conditional "
                    "weight underflows to zero");
            }

            // The new points, in the kept dimensions
            const auto& pts = this->data_.get();
            const std::size_t nF = dimsF.size();
            const std::size_t nR = dimsR.size();
            OwnedKernelDensityData<AuxType> data;
            data.nDim_ = nR;
            data.x_.resize(kept.size() * nR);
            data.w_.resize(kept.size());
            if constexpr (KernelDensityData<AuxType>::hasAux) { data.aux_.reserve(kept.size()); }
            Eigen::VectorXd a(static_cast<Eigen::Index>(nF)); // NOLINT(misc-include-cleaner) -- Eigen::VectorXd is provided by <Eigen/Dense> (included via Bandwidth.hpp); clang-tidy's IWYU mapping doesn't know that
            for (std::size_t k = 0; k < kept.size(); ++k)
            {
                const auto [w, i] = kept[k]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
                const auto pos = pts.position(i);
                for (std::size_t f = 0; f < nF; ++f) { a(static_cast<Eigen::Index>(f)) = pos[dimsF[f]] - x0[dimsF[f]]; } // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
                const Eigen::VectorXd ka = shift * a;
                for (std::size_t r = 0; r < nR; ++r)
                {
                    data.x_[(k * nR) + r] = pos[dimsR[r]] + ka(static_cast<Eigen::Index>(r)); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
                }
                data.w_[k] = w; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
                if constexpr (KernelDensityData<AuxType>::hasAux) { data.aux_.push_back(pts.aux(i)); }
            }

            // Density of the marginal over F at x0_F: the original
            // weights sum to one, so the conditional weights sum to it,
            // up to the normalization of S's kernel
            const double marginal = keptMass * std::sqrt(weightMetric.det()) /
                std::pow(std::numbers::pi, 0.5 * static_cast<double>(nF));
            return {.data_ = std::move(data), .dims_ = std::move(dimsR), .metric_ = kernelMetric,
                .marginal_ = marginal};
        }

    public:

        /**
         * @brief Condition the estimate on values of some dimensions, with the Euclidean bandwidth
         * @param x0 As for the other overload
         * @param marginalDims As for the other overload
         * @param massTol As for the other overload
         * @param leafSize As for the other overload
         * @return As for the other overload, with M the identity
         * @throws As for the other overload
         */
        [[nodiscard]] auto conditional(const std::vector<double>& x0,
            const std::vector<std::size_t>& marginalDims,
            const double massTol,
            const std::size_t leafSize = 16) const -> ConditionalDensity<AuxType>
        {
            return conditional(x0, marginalDims, massTol, this->euclidean_, leafSize);
        }

        /**
         * @brief Evaluate the kernel density estimate at each of several points, in parallel
         * @param x The points, each as for the single-point pdf()
         * @param relTol As for the single-point pdf()
         * @param absTol As for the single-point pdf()
         * @param bandwidth The bandwidth, one per point
         * @return The estimate at each point, each exactly as the
         *   single-point pdf() returns it
         * @throws std::invalid_argument if x and bandwidth differ in
         *   length; otherwise, any exception the single-point pdf()
         *   throws for any one of the points (the first one caught, if
         *   several do)
         * @details
         * Runs the single-point pdf() for each point, in parallel
         * (dynamically scheduled, since queries can differ widely in
         * cost) if built with OpenMP.
         */
        [[nodiscard]] auto pdf(const std::vector<std::vector<double>>& x,
            const double relTol,
            const double absTol,
            const std::vector<Bandwidth>& bandwidth) const -> std::vector<double>
        {
            if (bandwidth.size() != x.size())
            {
                throw std::invalid_argument("KernelDensity::pdf: x and bandwidth must have the same "
                    "number of elements");
            }
            return parallelEval<double>(x.size(), [&](const std::size_t i) -> double
                { return pdf(x[i], relTol, absTol, bandwidth[i]); }); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
        }

        /**
         * @brief Evaluate the kernel density estimate at each of several points, in parallel, sharing one bandwidth
         * @param x The points, each as for the single-point pdf()
         * @param relTol As for the single-point pdf()
         * @param absTol As for the single-point pdf()
         * @param bandwidth The bandwidth, shared by every point
         * @return As for the version taking one bandwidth per point
         * @throws As for the version taking one bandwidth per point
         * @details
         * If bandwidth has not been reduced and every point has the
         * same pattern of NaNs, it is reduced once for all of them,
         * rather than once per point.
         */
        [[nodiscard]] auto pdf(const std::vector<std::vector<double>>& x,
            const double relTol,
            const double absTol,
            const Bandwidth& bandwidth) const -> std::vector<double>
        {
            const Bandwidth bw = this->sharedReduction(x, bandwidth);
            return parallelEval<double>(x.size(), [&](const std::size_t i) -> double
                { return pdf(x[i], relTol, absTol, bw); }); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
        }

        /**
         * @brief Evaluate the kernel density estimate at each of several points, in parallel, with the Euclidean bandwidth
         * @param x As for the other overloads
         * @param relTol As for the other overloads
         * @param absTol As for the other overloads
         * @return As for the other overloads, with M the identity
         * @throws As for the other overloads
         */
        [[nodiscard]] auto pdf(const std::vector<std::vector<double>>& x,
            const double relTol,
            const double absTol) const -> std::vector<double>
        {
            return pdf(x, relTol, absTol, this->euclidean_);
        }

        /**
         * @brief Evaluate, in parallel, the density of each of several points measured with errors
         * @param x The points, each as for the single-point pdf()
         * @param errDims The dimensions with measurement errors; each
         *   must be finite (specified) in every point
         * @param errCov The covariance matrix of each point's errors, in
         *   the dimensions errDims (in that order): one per point, each
         *   errDims.size() square, symmetric, and positive semi-definite
         * @param relTol As for the single-point pdf()
         * @param absTol As for the single-point pdf()
         * @param bandwidth The bandwidth, shared by every point
         * @return For each point x[i], the estimate at x[i] with the
         *   kernel convolved with its errors (see
         *   Bandwidth::convolveCov()), i.e. the density of measuring
         *   x[i] given errors with covariance errCov[i]
         * @throws std::invalid_argument if errCov does not have one
         *   matrix per point, or one has the wrong size, or errDims has
         *   a dimension not below nDim, a repeated one, or one that is
         *   NaN in a point; otherwise, as for the other overloads, and
         *   as for Bandwidth::convolveCov()
         * @details
         * Each point's bandwidth is convolved with its own errors in the
         * thread that evaluates it. If every point has the same pattern
         * of NaNs, bandwidth is reduced for it once, beforehand, and each
         * convolution done in the reduced space (reduction and
         * convolution with errors in specified dimensions commute).
         */
        [[nodiscard]] auto pdf(const std::vector<std::vector<double>>& x,
            const std::vector<std::size_t>& errDims,
            const std::vector<Eigen::MatrixXd>& errCov, // NOLINT(misc-include-cleaner) -- Eigen::MatrixXd is provided by <Eigen/Dense> (included via Bandwidth.hpp); clang-tidy's IWYU mapping doesn't know that
            const double relTol,
            const double absTol,
            const Bandwidth& bandwidth) const -> std::vector<double>
        {
            this->checkErrors(x, errDims, errCov, {}, "KernelDensity::pdf");
            const Bandwidth bw = this->sharedReduction(x, bandwidth);
            return parallelEval<double>(x.size(), [&](const std::size_t i) -> double
            {
                return pdf(x[i], relTol, absTol, bw.convolveCov(this->embedErrors(errDims, errCov[i]))); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
            });
        }

        /**
         * @brief Condition the estimate on each of several points, in parallel
         * @param x0 The points to condition on, each as for the
         *   single-point conditional()
         * @param marginalDims As for the single-point conditional(),
         *   shared by every point
         * @param massTol As for the single-point conditional()
         * @param bandwidth As for the single-point conditional(),
         *   shared by every point
         * @param leafSize As for the single-point conditional()
         * @return One conditional density per point, each exactly as the
         *   single-point conditional() returns it
         * @throws Any exception the single-point conditional() throws
         *   for any one of the points (the first one caught, if several
         *   do)
         * @details
         * Runs the single-point conditional() for each point, in
         * parallel (dynamically scheduled) if built with OpenMP. Each
         * result's own tree is then built on the thread that conditions
         * it: the parallel loops of that build are nested inside this
         * one, so (unless nested parallelism is enabled) run on that
         * thread alone.
         */
        [[nodiscard]] auto conditional(const std::vector<std::vector<double>>& x0,
            const std::vector<std::size_t>& marginalDims,
            const double massTol,
            const Bandwidth& bandwidth,
            const std::size_t leafSize = 16) const -> std::vector<ConditionalDensity<AuxType>>
        {
            return parallelEval<ConditionalDensity<AuxType>>(x0.size(),
                [&](const std::size_t i) -> ConditionalDensity<AuxType>
                { return conditional(x0[i], marginalDims, massTol, bandwidth, leafSize); }); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
        }

        /**
         * @brief Condition the estimate on each of several points, in parallel, with the Euclidean bandwidth
         * @param x0 As for the other overload
         * @param marginalDims As for the other overload
         * @param massTol As for the other overload
         * @param leafSize As for the other overload
         * @return As for the other overload, with M the identity
         * @throws As for the other overload
         */
        [[nodiscard]] auto conditional(const std::vector<std::vector<double>>& x0,
            const std::vector<std::size_t>& marginalDims,
            const double massTol,
            const std::size_t leafSize = 16) const -> std::vector<ConditionalDensity<AuxType>>
        {
            return conditional(x0, marginalDims, massTol, this->euclidean_, leafSize);
        }

        /**
         * @brief Condition the estimate, in parallel, on each of several points measured with errors
         * @param x0 The points to condition on, each as for the
         *   single-point conditional()
         * @param errDims The dimensions with measurement errors; each
         *   must be fixed (finite, and not in marginalDims) in every point
         * @param errCov The covariance matrix of each point's errors, in
         *   the dimensions errDims, as for the errors version of pdf()
         * @param marginalDims As for the single-point conditional(),
         *   shared by every point
         * @param massTol As for the single-point conditional()
         * @param bandwidth As for the single-point conditional(),
         *   shared by every point
         * @param leafSize As for the single-point conditional()
         * @return For each point x0[i], the conditional density given
         *   x0[i] with the kernel first convolved with its errors (see
         *   Bandwidth::convolveCov()): the posterior given a measurement
         *   x0[i] with errors of covariance errCov[i]
         * @throws std::invalid_argument as for the errors version of
         *   pdf(), with errDims required to be fixed rather than merely
         *   specified; otherwise, as for the other overloads, and as for
         *   Bandwidth::convolveCov()
         */
        [[nodiscard]] auto conditional(const std::vector<std::vector<double>>& x0,
            const std::vector<std::size_t>& errDims,
            const std::vector<Eigen::MatrixXd>& errCov,
            const std::vector<std::size_t>& marginalDims,
            const double massTol,
            const Bandwidth& bandwidth,
            const std::size_t leafSize = 16) const -> std::vector<ConditionalDensity<AuxType>>
        {
            this->checkErrors(x0, errDims, errCov, marginalDims, "KernelDensity::conditional");
            return parallelEval<ConditionalDensity<AuxType>>(x0.size(),
                [&](const std::size_t i) -> ConditionalDensity<AuxType>
            {
                return conditional(x0[i], marginalDims, massTol, // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
                    bandwidth.convolveCov(this->embedErrors(errDims, errCov[i])), leafSize); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
            });
        }

        /**
         * @brief Find the sample points, weights, and auxiliary data of conditional estimates for each of several points, in parallel
         * @param x0 The points to condition on, each as for the
         *   single-point conditionalSample()
         * @param marginalDims As for the single-point
         *   conditionalSample(), shared by every point
         * @param massTol As for the single-point conditionalSample()
         * @param bandwidth As for the single-point conditionalSample(),
         *   shared by every point
         * @return One result per point, each exactly as the single-point
         *   conditionalSample() returns it
         * @throws Any exception the single-point conditionalSample()
         *   throws for any one of the points (the first one caught, if
         *   several do)
         * @details
         * Runs the single-point conditionalSample() for each point, in
         * parallel (dynamically scheduled) if built with OpenMP.
         */
        [[nodiscard]] auto conditionalSample(const std::vector<std::vector<double>>& x0,
            const std::vector<std::size_t>& marginalDims,
            const double massTol,
            const Bandwidth& bandwidth) const
            -> std::vector<Sample>
        {
            return parallelEval<Sample>(x0.size(),
                [&](const std::size_t i) -> Sample
                { return conditionalSample(x0[i], marginalDims, massTol, bandwidth); }); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
        }

        /**
         * @brief Find the sample points, weights, and auxiliary data of conditional estimates for each of several points, in parallel, with the Euclidean bandwidth
         * @param x0 As for the other overload
         * @param marginalDims As for the other overload
         * @param massTol As for the other overload
         * @return As for the other overload, with M the identity
         * @throws As for the other overload
         */
        [[nodiscard]] auto conditionalSample(const std::vector<std::vector<double>>& x0,
            const std::vector<std::size_t>& marginalDims,
            const double massTol) const
            -> std::vector<Sample>
        {
            return conditionalSample(x0, marginalDims, massTol, this->euclidean_);
        }

        /**
         * @brief Find, in parallel, the sample points, weights, and auxiliary data of conditional estimates for each of several points measured with errors
         * @param x0 The points to condition on, each as for the
         *   single-point conditionalSample()
         * @param errDims As for the errors version of the batch
         *   conditional()
         * @param errCov As for the errors version of the batch
         *   conditional()
         * @param marginalDims As for the single-point
         *   conditionalSample(), shared by every point
         * @param massTol As for the single-point conditionalSample()
         * @param bandwidth As for the single-point conditionalSample(),
         *   shared by every point
         * @return For each point x0[i], the sample points, weights, and
         *   auxiliary data of the conditional estimate given x0[i] with the kernel first
         *   convolved with its errors, as the errors version of the
         *   batch conditional() would keep them; its kernel metric is
         *   that of the convolved bandwidth
         * @throws As for the errors version of the batch conditional()
         */
        [[nodiscard]] auto conditionalSample(const std::vector<std::vector<double>>& x0,
            const std::vector<std::size_t>& errDims,
            const std::vector<Eigen::MatrixXd>& errCov,
            const std::vector<std::size_t>& marginalDims,
            const double massTol,
            const Bandwidth& bandwidth) const
            -> std::vector<Sample>
        {
            this->checkErrors(x0, errDims, errCov, marginalDims, "KernelDensity::conditionalSample");
            return parallelEval<Sample>(x0.size(),
                [&](const std::size_t i) -> Sample
            {
                return conditionalSample(x0[i], marginalDims, massTol, // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
                    bandwidth.convolveCov(this->embedErrors(errDims, errCov[i]))); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
            });
        }

    protected:

        /**
         * @brief Evaluate a function for each of a set of indices, in parallel
         * @tparam T The function's result type; need only be movable
         * @tparam Eval Callable taking an index and returning a T
         * @param n Number of indices
         * @param eval The function
         * @return eval(i) for i = 0, ..., n - 1, in order
         * @throws The first exception any call throws (one must not
         *   escape an OpenMP region, so each is caught in the loop and
         *   the first rethrown afterwards)
         * @details
         * Dynamically scheduled OpenMP loop, if built with OpenMP.
         */
        template <typename T, typename Eval>
        [[nodiscard]] static auto parallelEval(const std::size_t n, const Eval& eval) -> std::vector<T>
        {
            std::vector<std::optional<T>> results(n);
            std::exception_ptr error = nullptr;
#ifdef _OPENMP
#           pragma omp parallel for schedule(dynamic)
#endif
            for (std::size_t i = 0; i < n; ++i)
            {
                try
                {
                    results[i].emplace(eval(i)); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
                }
                catch (...)
                {
#ifdef _OPENMP
#                   pragma omp critical(kernelDensityBatchError)
#endif
                    {
                        if (!error) { error = std::current_exception(); }
                    }
                }
            }
            if (error) { std::rethrow_exception(error); }
            std::vector<T> out;
            out.reserve(n);
            for (auto& r : results) { out.push_back(std::move(*r)); } // NOLINT(bugprone-unchecked-optional-access) -- every element was set, or an exception rethrown above
            return out;
        }

        /**
         * @brief Sort dimensions into fixed, kept, and marginalized, for conditional()
         * @param x0 As for conditional(); must have nDim elements
         * @param marginalDims As for conditional()
         * @return The fixed dimensions F and the kept dimensions R, in
         *   order, and flags marking the dimensions in G and in G or R
         * @throws std::invalid_argument if marginalDims has a dimension
         *   not below nDim or a repeated one, or F or R is empty
         */
        [[nodiscard]] auto classifyDims(const std::vector<double>& x0,
            const std::vector<std::size_t>& marginalDims) const
            -> std::tuple<std::vector<std::size_t>, std::vector<std::size_t>, std::vector<bool>, std::vector<bool>>
        {
            const std::size_t nDim = this->nDim_;
            std::vector<bool> inG(nDim, false);
            for (const std::size_t d : marginalDims)
            {
                if (d >= nDim || inG[d])
                {
                    throw std::invalid_argument("KernelDensity::conditional: marginalDims must hold "
                        "distinct dimensions below nDim");
                }
                inG[d] = true;
            }
            std::vector<std::size_t> dimsF;
            std::vector<std::size_t> dimsR;
            std::vector<bool> inGR = inG;
            for (std::size_t d = 0; d < nDim; ++d)
            {
                if (inG[d]) { continue; }
                if (std::isnan(x0[d])) // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
                {
                    dimsR.push_back(d);
                    inGR[d] = true;
                }
                else { dimsF.push_back(d); }
            }
            if (dimsF.empty() || dimsR.empty())
            {
                throw std::invalid_argument("KernelDensity::conditional: x0 must fix at least one "
                    "dimension and leave at least one unfixed");
            }
            return {std::move(dimsF), std::move(dimsR), std::move(inG), std::move(inGR)};
        }

        /**
         * @brief The kernel metric and point shift of a conditional estimate
         * @param m The (possibly reduced) inverse squared bandwidth matrix, nDim x nDim
         * @param dimsF The fixed dimensions
         * @param dimsR The kept dimensions
         * @return (M_RR, K = M_RR^-1 M_RF), |R| x |R| and |R| x |F|
         * @throws std::domain_error if M_RR is not positive definite
         */
        [[nodiscard]] static auto conditionalMetric(const Eigen::MatrixXd& m,
            const std::vector<std::size_t>& dimsF,
            const std::vector<std::size_t>& dimsR) -> std::pair<Eigen::MatrixXd, Eigen::MatrixXd> // NOLINT(misc-include-cleaner) -- Eigen::MatrixXd is provided by <Eigen/Dense> (included via Bandwidth.hpp); clang-tidy's IWYU mapping doesn't know that
        {
            const auto nF = static_cast<Eigen::Index>(dimsF.size());
            const auto nR = static_cast<Eigen::Index>(dimsR.size());
            Eigen::MatrixXd mRR(nR, nR);
            Eigen::MatrixXd mRF(nR, nF);
            for (Eigen::Index r = 0; r < nR; ++r)
            {
                const auto ri = static_cast<Eigen::Index>(dimsR[static_cast<std::size_t>(r)]); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
                for (Eigen::Index c = 0; c < nR; ++c) { mRR(r, c) = m(ri, static_cast<Eigen::Index>(dimsR[static_cast<std::size_t>(c)])); } // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
                for (Eigen::Index c = 0; c < nF; ++c) { mRF(r, c) = m(ri, static_cast<Eigen::Index>(dimsF[static_cast<std::size_t>(c)])); } // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
            }
            const Eigen::LLT<Eigen::MatrixXd> llt(mRR); // NOLINT(misc-include-cleaner) -- Eigen::LLT is provided by <Eigen/Dense> (included via Bandwidth.hpp); clang-tidy's IWYU mapping doesn't know that
            if (llt.info() != Eigen::Success) // NOLINT(misc-include-cleaner) -- Eigen::Success is provided by <Eigen/Dense> (included via Bandwidth.hpp); clang-tidy's IWYU mapping doesn't know that
            {
                throw std::domain_error("KernelDensity::conditional: the bandwidth's block over the "
                    "kept dimensions is not positive definite");
            }
            Eigen::MatrixXd k = llt.solve(mRF);
            return {std::move(mRR), std::move(k)};
        }

        /**
         * @brief Select the points carrying all but massTol of the conditional weight
         * @param xF The point; NaN except in the fixed dimensions
         * @param weightMetric The metric S of the conditional weights;
         *   must match xF
         * @param massTol As for conditional()
         * @return The kept points, as (conditional weight, index into
         *   data()) pairs, and the sum of their weights
         * @details
         * See conditional(). Here each RefinementList entry's
         * "estimate" is a lower bound on the node's conditional weight
         * and its "uncertainty" an upper bound, so the list opens the
         * node with the largest upper bound.
         */
        [[nodiscard]] auto selectConditional(const std::vector<double>& xF,
            const Bandwidth& weightMetric,
            const double massTol) const
            -> std::pair<std::vector<std::pair<double, std::size_t>>, double>
        {
            const auto& pts = this->data_.get();
            const auto bound = weightMetric.boxBounder(xF);
            std::vector<double> scratch;
            RefinementList list;
            std::vector<std::pair<double, std::size_t>> candidates;
            double leafMass = 0.0;
            auto add = [&](const std::size_t idx, const std::size_t slot) -> void
            {
                const auto& node = this->nodes_[idx];
                if (node.splitDim_ < 0)
                {
                    for (std::size_t i = node.start_; i < node.start_ + node.count_; ++i)
                    {
                        const double w = pts.weight(i) * std::exp(-weightMetric.dist2(xF, pts.position(i), scratch));
                        if (w > 0.0)
                        {
                            candidates.emplace_back(w, i);
                            leafMass += w;
                        }
                    }
                    if (slot != nullNode) { list.erase(slot); }
                    return;
                }
                const auto [r2min, r2max] = bound.bounds(this->nodeLo(idx), this->nodeHi(idx));
                const double w = nodeW_[idx]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
                const double lo = w * std::exp(-r2max);
                const double hi = w * std::exp(-r2min);
                if (slot != nullNode) { list.replace(slot, idx, lo, hi); }
                else { list.insert(idx, lo, hi); }
            };

            // Open nodes until the unopened ones can carry at most
            // massTol / 2 of the total
            add(rootNode, nullNode);
            while (list.uncertainty() > 0.5 * massTol * (leafMass + list.estimate()))
            {
                const std::size_t worst = list.worst();
                const auto& node = this->nodes_[list.node(worst)];
                add(node.lchild_, worst);
                add(node.rchild_, nullNode);
            }

            // Drop the lightest points, up to another massTol / 2
            const double budget = 0.5 * massTol * (leafMass + list.estimate());
            std::ranges::sort(candidates);
            double dropped = 0.0;
            std::size_t first = 0;
            while (first < candidates.size() && dropped + candidates[first].first <= budget) // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
            {
                dropped += candidates[first].first; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
                ++first;
            }
            candidates.erase(candidates.begin(), candidates.begin() + static_cast<std::ptrdiff_t>(first));
            double keptMass = 0.0;
            for (const auto& c : candidates) { keptMass += c.first; }
            return {std::move(candidates), keptMass};
        }

        /**
         * @brief The evaluation itself, for pdf()
         * @param x The point; must match bandwidth (see Bandwidth::matches())
         * @param relTol Relative tolerance
         * @param absTol Absolute tolerance
         * @param bandwidth The bandwidth
         * @return As for pdf()
         * @throws std::domain_error as for pdf()
         */
        [[nodiscard]] auto evalPdf(const std::vector<double>& x,
            const double relTol,
            const double absTol,
            const Bandwidth& bandwidth) const -> double
        {
            // Normalization: exp(-d^T M d) integrates to
            // pi^(n / 2) / sqrt(det M) over the n specified dimensions,
            // and the weights sum to one
            const double det = bandwidth.det();
            if (!(det > 0.0))
            {
                throw std::domain_error("KernelDensity::pdf: cannot normalize: bandwidth is singular");
            }
            const double norm = std::sqrt(det) /
                std::pow(std::numbers::pi, 0.5 * static_cast<double>(bandwidth.nSpecified()));

            // Evaluate a node, adding it to the list (in a given slot,
            // or a new one if slot is nullNode) or, if it is a leaf, to
            // the leaves' sum (freeing the slot)
            const auto bound = bandwidth.boxBounder(x);
            std::vector<double> scratch;
            RefinementList list;
            double leafSum = 0.0;
            auto add = [&](const std::size_t idx, const std::size_t slot) -> void
            {
                const auto [est, err] = nodePdf(idx, x, bandwidth, bound, scratch);
                if (this->nodes_[idx].splitDim_ < 0)
                {
                    leafSum += est;
                    if (slot != nullNode) { list.erase(slot); }
                }
                else if (slot != nullNode) { list.replace(slot, idx, est, err); }
                else { list.insert(idx, est, err); }
            };

            add(rootNode, nullNode);
            while (true)
            {
                // Converged, or exact (only leaves remain, or the
                // remaining nodes' kernel values have underflowed to zero)
                const double est = leafSum + list.estimate();
                const double err = list.uncertainty();
                const std::size_t worst = list.worst();
                if (err < relTol * est || (est + err) * norm < absTol || list.uncertainty(worst) == 0.0)
                {
                    return est * norm;
                }

                // Replace the worst node by its children
                const auto& node = this->nodes_[list.node(worst)];
                add(node.lchild_, worst);
                add(node.rchild_, nullNode);
            }
        }

        /**
         * @brief A node's contribution to the (unnormalized) density estimate at a point
         * @param idx Index of the node
         * @param x The point; must match bandwidth
         * @param bandwidth The bandwidth
         * @param bound A box bounder for x and bandwidth
         * @param scratch Scratch space for Bandwidth::dist2()
         * @return (central estimate, uncertainty) of the node's
         *   contribution sum_i w_i exp(-d_i^T M d_i), summed over its
         *   points. For a leaf, the sum itself, with zero uncertainty.
         *   For an interior node, with W its total weight and r2min and
         *   r2max lower and upper bounds on d^T M d over its bounding
         *   box (see Bandwidth::BoxBounder), the midpoint
         *   W (exp(-r2min) + exp(-r2max)) / 2 of the range the sum
         *   must lie in, and its half-width
         *   W (exp(-r2min) - exp(-r2max)) / 2.
         */
        [[nodiscard]] auto nodePdf(const std::size_t idx,
            const std::vector<double>& x,
            const Bandwidth& bandwidth,
            const Bandwidth::BoxBounder& bound,
            std::vector<double>& scratch) const -> std::pair<double, double>
        {
            const auto& node = this->nodes_[idx];
            if (node.splitDim_ < 0)
            {
                const auto& pts = this->data_.get();
                double sum = 0.0;
                for (std::size_t i = node.start_; i < node.start_ + node.count_; ++i)
                {
                    sum += pts.weight(i) * std::exp(-bandwidth.dist2(x, pts.position(i), scratch));
                }
                return {sum, 0.0};
            }
            const auto [r2min, r2max] = bound.bounds(this->nodeLo(idx), this->nodeHi(idx));
            const double kMax = std::exp(-r2min);
            const double kMin = std::exp(-r2max);
            const double w = nodeW_[idx]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
            return {0.5 * w * (kMax + kMin), 0.5 * w * (kMax - kMin)};
        }

        /**
         * @brief Check a set of weighted points and rescale their weights to sum to one
         * @param data The points
         * @return data, for use in the constructors' initializer lists
         * @throws std::invalid_argument if data is invalid (see
         *   KernelDensityData::validate(), which also rejects negative
         *   weights), or its weights sum to zero
         */
        static auto normalizeWeights(KernelDensityData<AuxType>& data) -> KernelDensityData<AuxType>&
        {
            data.validate();
            normalizeWeightArray(data.w_);
            return data;
        }

        /**
         * @brief Check a set of weights and rescale them to sum to one, in place
         * @param w The weights
         * @throws std::invalid_argument if any weight is negative or not
         *   finite, or all are zero
         * @details
         * The weights are scaled by the largest first, so that summing
         * finite weights cannot overflow. Each pass over the weights
         * runs in parallel if built with OpenMP, since reweight() calls
         * this whenever the weights are changed, e.g. on every step of
         * an optimization over them.
         */
        static void normalizeWeightArray(const std::span<double> w)
        {
            const std::size_t n = w.size();
            double wMax = 0.0;
            bool valid = true;
#ifdef _OPENMP
#           pragma omp parallel for reduction(max:wMax) reduction(&&:valid)
#endif
            for (std::size_t i = 0; i < n; ++i)
            {
                const double v = w[i]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
                valid = valid && std::isfinite(v) && v >= 0.0;
                wMax = std::max(wMax, v);
            }
            if (!valid)
            {
                throw std::invalid_argument("KernelDensity: weights must be finite and non-negative");
            }
            if (!(wMax > 0.0))
            {
                throw std::invalid_argument("KernelDensity: weights must not all be zero");
            }
            double wTot = 0.0;
#ifdef _OPENMP
#           pragma omp parallel for reduction(+:wTot)
#endif
            for (std::size_t i = 0; i < n; ++i)
            {
                w[i] /= wMax; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
                wTot += w[i]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
            }
#ifdef _OPENMP
#           pragma omp parallel for
#endif
            for (std::size_t i = 0; i < n; ++i)
            {
                w[i] /= wTot; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
            }
        }

        /**
         * @brief Compute every node's weight sum
         * @details
         * One pass up the tree, from the last level to the root (see
         * accumulateNode()). Nodes on the same level are independent,
         * so each level is processed in parallel if built with OpenMP.
         */
        void setNodeWeights()
        {
            const std::size_t nNodes = this->nodes_.size();
            nodeW_.assign(nNodes, 0.0);

            // Level k holds nodes 2^k - 1 through 2^(k+1) - 2
            for (std::size_t level = this->nLevels_; level-- > 0;)
            {
                const std::size_t first = (std::size_t{1} << level) - 1;
                const std::size_t last = std::min((std::size_t{2} << level) - 1, nNodes);
#ifdef _OPENMP
#               pragma omp parallel for
#endif
                for (std::size_t idx = first; idx < last; ++idx) { accumulateNode(idx); }
            }
        }

        /**
         * @brief Set one node's weight sum
         * @param idx Index of the node; if it is an interior node, its
         *   children's sums must already be set
         * @details
         * A leaf (or an unused slot below one, which has no points)
         * sums over its own points; an interior node adds up its two
         * children's sums. Writes only this node's own entry, so nodes
         * may be processed in parallel.
         */
        void accumulateNode(const std::size_t idx)
        {
            const auto& node = this->nodes_[idx];
            if (node.splitDim_ < 0)
            {
                const auto& pts = this->data_.get();
                double w = 0.0;
                for (std::size_t i = node.start_; i < node.start_ + node.count_; ++i) { w += pts.weight(i); }
                nodeW_[idx] = w; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
                return;
            }
            nodeW_[idx] = nodeW_[node.lchild_] + nodeW_[node.rchild_]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
        }

        std::vector<double> nodeW_;   /**< Sum of the weights of each node's points */
    };

    /**
     * @class ConditionalDensity
     * @brief A kernel density estimate conditioned on some dimensions' values, owning its points
     * @tparam AuxType As for KernelDensity
     * @details
     * Made by KernelDensity::conditional(). Holds the conditional
     * estimate's points (shifted, reweighted, and pruned copies of the
     * original ones, with their auxiliary data), a KernelDensity over
     * them, and the kernel metric to evaluate it with. The points and
     * the view of them the KernelDensity refers to are held through a
     * pointer, so their addresses survive moving the
     * ConditionalDensity.
     */
    template <typename AuxType = void>
    class ConditionalDensity
    {
    public:

        /**
         * @brief Evaluate the conditional density
         * @param y Position in the kept dimensions (see dims()); elements
         *   may be NaN, to marginalize over them too
         * @param relTol As for KernelDensity::pdf()
         * @param absTol As for KernelDensity::pdf()
         * @return p(y | x0_F), normalized to unit integral over the kept
         *   dimensions
         * @throws As for KernelDensity::pdf()
         */
        [[nodiscard]] auto pdf(const std::vector<double>& y, const double relTol, const double absTol) const -> double
        {
            return kde_.pdf(y, relTol, absTol, bandwidth_);
        }

        /**
         * @brief Evaluate the conditional density at each of several points, in parallel
         * @param y The points, each with one element per kept dimension
         *   (see dims())
         * @param relTol As for KernelDensity::pdf()
         * @param absTol As for KernelDensity::pdf()
         * @return The density at each point
         * @throws As for KernelDensity's batch pdf()
         */
        [[nodiscard]] auto pdf(const std::vector<std::vector<double>>& y, const double relTol,
            const double absTol) const -> std::vector<double>
        {
            return kde_.pdf(y, relTol, absTol, bandwidth_);
        }

        /** @brief The kept dimensions
         *  @return Their indices in the original estimate, in order;
         *    dimension i of this estimate is dims()[i] there */
        [[nodiscard]] auto dims() const -> const std::vector<std::size_t>& { return dims_; }
        /** @brief Density of the original estimate's marginal over the fixed dimensions, at x0_F
         *  @return p(x0_F), estimated from the points kept, so low by at
         *    most a fraction massTol; p(x0_F, y) = marginal() pdf(y) */
        [[nodiscard]] auto marginal() const -> double { return marginal_; }
        /** @brief The kernel metric of the conditional estimate
         *  @return M'_RR (see KernelDensity::conditional()) */
        [[nodiscard]] auto bandwidth() const -> const Bandwidth& { return bandwidth_; }
        /** @brief The conditional estimate itself
         *  @return Its KernelDensity, to be evaluated with bandwidth() */
        [[nodiscard]] auto kde() const -> const KernelDensity<AuxType>& { return kde_; }
        /** @brief The conditional estimate's points
         *  @return Its points, in tree order, with weights normalized to sum to one */
        [[nodiscard]] auto data() const -> const KernelDensityData<AuxType>& { return points_->view_; }

    private:

        friend class KernelDensity<AuxType>;

        /**
         * @struct Points
         * @brief The owned points and the view of them the KernelDensity is built over
         */
        struct Points
        {
            OwnedKernelDensityData<AuxType> owned_;  /**< The points */
            KernelDensityData<AuxType> view_;        /**< A view of owned_ */
        };

        /**
         * @brief Move a set of points onto the heap, with a view of them
         * @param data The points
         * @return The heap-allocated points and view
         */
        static auto makePoints(OwnedKernelDensityData<AuxType>&& data) -> std::unique_ptr<Points>
        {
            auto p = std::make_unique<Points>();
            p->owned_ = std::move(data);
            p->view_ = p->owned_.view();
            return p;
        }

        /**
         * @brief Construct from the conditional estimate's points
         * @param data The points (not yet normalized)
         * @param bandwidth The kernel metric
         * @param dims The kept dimensions
         * @param marginal p(x0_F)
         * @param leafSize Leaf size of the tree
         */
        ConditionalDensity(OwnedKernelDensityData<AuxType>&& data, Bandwidth bandwidth,
            std::vector<std::size_t> dims, const double marginal, const std::size_t leafSize) :
            points_(makePoints(std::move(data))),
            kde_(points_->view_, leafSize, bandwidth),
            bandwidth_(std::move(bandwidth)),
            dims_(std::move(dims)),
            marginal_(marginal)
        {}

        std::unique_ptr<Points> points_;                    /**< The points, owned */
        KernelDensity<AuxType> kde_;                        /**< The estimate over points_->view_ */
        Bandwidth bandwidth_;                               /**< Its kernel metric */
        std::vector<std::size_t> dims_;                     /**< The kept dimensions */
        double marginal_;                                   /**< p(x0_F) */
    };

} // namespace kden

#endif // KERNELDENSITY_HPP
