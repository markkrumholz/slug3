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
#include <cstdint>
#include <exception>
#include <functional>
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

        /**
         * @brief A one-dimensional percentile of the estimate
         * @param dim The dimension
         * @param pct The percentile, in (0, 100)
         * @param relTol Relative tolerance, in probability (see the
         *   return value); non-negative
         * @param absTol Absolute tolerance, in probability; non-negative
         * @param bandwidth The kernel's inverse squared bandwidth matrix,
         *   for nDim dimensions; must not be reduced
         * @return A value t of dimension dim at which the estimate's
         *   marginal cumulative distribution in dim, F(t), satisfies
         *   |F(t) - p| <= max(absTol, relTol min(p, 1 - p)), with p =
         *   pct / 100 -- a tolerance in probability, the same relative
         *   standard in both tails
         * @throws std::invalid_argument if dim is not below nDim, pct is
         *   not in (0, 100), a tolerance is negative or NaN, or
         *   bandwidth is not for nDim dimensions or is reduced
         * @throws std::domain_error if the estimate has no points, or the
         *   bandwidth's block over the other dimensions is not positive
         *   definite
         * @details
         * Marginalizing the kernel over every other dimension leaves a
         * one-dimensional Gaussian exp(-m u^2), with m the bandwidth
         * reduced to dim alone, so F(t) = sum_i w_i Phi(t - x_i), with
         * Phi(u) = erfc(-sqrt(m) u) / 2. Each term is monotone in x_i, so
         * a node of weight W whose points lie in [lo, hi] in dim adds
         * between W Phi(t - hi) and W Phi(t - lo) to F, for every t.
         *
         * A cut through the tree -- unopened nodes, plus points evaluated
         * exactly -- thus bounds F between two functions F_L <= F <= F_U,
         * and the percentile t* between the roots of F_U = p and F_L = p.
         * Starting from the root alone, each round finds that bracket and
         * a candidate t_c, the root of (F_L + F_U) / 2 = p, within it
         * (safeguarded Newton iterations, every root-find confined to the
         * current bracket, so that it only ever shrinks). It stops once
         * F_U(t_c) - p and p - F_L(t_c) are both within the tolerance,
         * which guarantees the result; otherwise it opens the nodes that
         * contribute most to F_U - F_L at t_c (every one within a factor
         * of two of the largest), a leaf's points becoming exact terms.
         * Terms lying wholly to the left or right of the bracket, whose
         * contribution is 1 or 0 there to within a small fraction of the
         * tolerance, are folded into constant bounds (rigorous, since the
         * bracket only shrinks), so the work stays with the points within
         * a few kernel widths of t*: it costs far less than evaluating
         * F over every point at each step of a root-find. That relies on
         * the tree splitting along dim: in a dimension in noSplitDims,
         * every node spans its whole range, so the walk must reach
         * nearly every point (still correct, but much slower).
         * Thread-safe, as for pdf().
         */
        [[nodiscard]] auto percentile(const std::size_t dim,
            const double pct,
            const double relTol,
            const double absTol,
            const Bandwidth& bandwidth) const -> double
        {
            const std::size_t nDim = this->nDim_;
            if (dim >= nDim)
            {
                throw std::invalid_argument("KernelDensity::percentile: dim must be below nDim");
            }
            if (!(pct > 0.0) || !(pct < 100.0))   // written to reject NaN too
            {
                throw std::invalid_argument("KernelDensity::percentile: pct must be in (0, 100)");
            }
            if (!(relTol >= 0.0) || !(absTol >= 0.0))
            {
                throw std::invalid_argument("KernelDensity::percentile: relTol and absTol must be non-negative");
            }
            if (bandwidth.nDim() != nDim)
            {
                throw std::invalid_argument("KernelDensity::percentile: bandwidth must be for nDim dimensions");
            }
            if (bandwidth.isReduced())
            {
                throw std::invalid_argument("KernelDensity::percentile: bandwidth must not be reduced");
            }
            if (this->data_.get().size() == 0)
            {
                throw std::domain_error("KernelDensity::percentile: the estimate has no points");
            }

            // The marginal kernel in dim: the bandwidth reduced to dim alone
            std::vector<bool> others(nDim, true);
            others[dim] = false; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
            const auto di = static_cast<Eigen::Index>(dim); // NOLINT(misc-include-cleaner) -- Eigen::Index is provided by <Eigen/Dense> (included via Bandwidth.hpp); clang-tidy's IWYU mapping doesn't know that
            const double m = bandwidth.reduce(others).h2Inv()(di, di);
            if (!(m > 0.0) || !std::isfinite(m))
            {
                throw std::domain_error("KernelDensity::percentile: the bandwidth's marginal in dim is not positive");
            }
            const double p = pct / 100.0;
            return evalPercentile(dim, p, std::max(absTol, relTol * std::min(p, 1.0 - p)), std::sqrt(m));
        }

        /**
         * @brief A one-dimensional percentile of the estimate, with the Euclidean bandwidth
         * @param dim As for the other overload
         * @param pct As for the other overload
         * @param relTol As for the other overload
         * @param absTol As for the other overload
         * @return As for the other overload, with M the identity
         * @throws As for the other overload
         */
        [[nodiscard]] auto percentile(const std::size_t dim,
            const double pct,
            const double relTol,
            const double absTol) const -> double
        {
            return percentile(dim, pct, relTol, absTol, this->euclidean_);
        }

        /**
         * @brief Several one-dimensional percentiles of the estimate, in parallel
         * @param dim As for the single-percentile version
         * @param pct The percentiles, each in (0, 100)
         * @param relTol As for the single-percentile version
         * @param absTol As for the single-percentile version
         * @param bandwidth As for the single-percentile version
         * @return One value per percentile, each as the single-percentile
         *   version returns it
         * @throws As for the single-percentile version (the first
         *   exception caught, if several are)
         * @details
         * Runs the single-percentile version for each, in parallel
         * (dynamically scheduled) if built with OpenMP.
         */
        [[nodiscard]] auto percentile(const std::size_t dim,
            const std::vector<double>& pct,
            const double relTol,
            const double absTol,
            const Bandwidth& bandwidth) const -> std::vector<double>
        {
            return parallelEval<double>(pct.size(), [&](const std::size_t i) -> double
                { return percentile(dim, pct[i], relTol, absTol, bandwidth); }); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
        }

        /**
         * @brief Marginalize the estimate over some dimensions
         * @param marginalDims Dimensions to marginalize over (the set G):
         *   distinct, below nDim, and leaving at least one; may be empty
         * @param bandwidth The kernel's inverse squared bandwidth
         *   matrix, for nDim dimensions; must not be reduced
         * @param leafSize Leaf size of the result's tree
         * @return The marginal estimate over the remaining dimensions R
         *   (in increasing order), as a ConditionalDensity -- with no
         *   dimensions fixed, its marginal() is 1
         * @throws std::invalid_argument if bandwidth is not for nDim
         *   dimensions or is reduced, or marginalDims has a dimension
         *   not below nDim or a repeated one, or leaves no dimension
         * @throws std::domain_error if the bandwidth's block over G is
         *   not positive definite
         * @details
         * Marginalizing a Gaussian kernel over G replaces its metric M
         * by the Schur complement M' = bandwidth.reduce(G), so the
         * marginal estimate is the kernel density estimate over the
         * points' coordinates in R, with the same weights, and kernel
         * metric M' over R: conditional() with no dimension fixed, so
         * that every point keeps its full weight, unshifted. Its pdf()
         * at y equals this estimate's pdf() at the point with y in R
         * and NaN in G. Copies every point's coordinates in R, weight,
         * and auxiliary data (in parallel, if built with OpenMP), then
         * builds the result's tree. Thread-safe, as for pdf().
         */
        [[nodiscard]] auto marginalize(const std::vector<std::size_t>& marginalDims,
            const Bandwidth& bandwidth,
            const std::size_t leafSize = 16) const -> ConditionalDensity<AuxType>
        {
            const std::size_t nDim = this->nDim_;
            if (bandwidth.nDim() != nDim)
            {
                throw std::invalid_argument("KernelDensity::marginalize: bandwidth must be for nDim dimensions");
            }
            if (bandwidth.isReduced())
            {
                throw std::invalid_argument("KernelDensity::marginalize: bandwidth must not be reduced");
            }
            std::vector<bool> inG(nDim, false);
            for (const std::size_t d : marginalDims)
            {
                if (d >= nDim || inG[d]) // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
                {
                    throw std::invalid_argument("KernelDensity::marginalize: marginalDims must hold "
                        "distinct dimensions below nDim");
                }
                inG[d] = true; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
            }
            std::vector<std::size_t> dimsR;
            for (std::size_t d = 0; d < nDim; ++d)
            {
                if (!inG[d]) { dimsR.push_back(d); } // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
            }
            if (dimsR.empty())
            {
                throw std::invalid_argument("KernelDensity::marginalize: marginalDims must leave at least one dimension");
            }

            // The kernel metric: the Schur complement over R
            const Eigen::MatrixXd mPrime = bandwidth.reduce(inG).h2Inv(); // NOLINT(misc-include-cleaner) -- Eigen::MatrixXd is provided by <Eigen/Dense> (included via Bandwidth.hpp); clang-tidy's IWYU mapping doesn't know that
            const auto nR = static_cast<Eigen::Index>(dimsR.size()); // NOLINT(misc-include-cleaner) -- Eigen::Index is provided by <Eigen/Dense> (included via Bandwidth.hpp); clang-tidy's IWYU mapping doesn't know that
            Eigen::MatrixXd metric(nR, nR);
            for (Eigen::Index r = 0; r < nR; ++r)
            {
                for (Eigen::Index c = 0; c < nR; ++c)
                {
                    metric(r, c) = mPrime(static_cast<Eigen::Index>(dimsR[static_cast<std::size_t>(r)]), // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
                        static_cast<Eigen::Index>(dimsR[static_cast<std::size_t>(c)])); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
                }
            }

            // Every point's coordinates in R, weight, and auxiliary data
            const auto& pts = this->data_.get();
            const std::size_t n = pts.size();
            const std::size_t nKept = dimsR.size();
            OwnedKernelDensityData<AuxType> data;
            data.nDim_ = nKept;
            data.x_.resize(n * nKept);
            data.w_.assign(pts.w_.begin(), pts.w_.end());
            if constexpr (KernelDensityData<AuxType>::hasAux) { data.aux_.assign(pts.aux_.begin(), pts.aux_.end()); }
#ifdef _OPENMP
#           pragma omp parallel for
#endif
            for (std::size_t i = 0; i < n; ++i)
            {
                for (std::size_t r = 0; r < nKept; ++r)
                {
                    data.x_[(i * nKept) + r] = pts.x_[(i * nDim) + dimsR[r]]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
                }
            }
            return ConditionalDensity<AuxType>(std::move(data), Bandwidth(metric), std::move(dimsR), 1.0, leafSize);
        }

        /**
         * @brief Marginalize the estimate over some dimensions, with the Euclidean bandwidth
         * @param marginalDims As for the other overload
         * @param leafSize As for the other overload
         * @return As for the other overload, with M the identity
         * @throws As for the other overload
         */
        [[nodiscard]] auto marginalize(const std::vector<std::size_t>& marginalDims,
            const std::size_t leafSize = 16) const -> ConditionalDensity<AuxType>
        {
            return marginalize(marginalDims, this->euclidean_, leafSize);
        }

    protected:

        /**
         * @class PercentileCut
         * @brief A cut through the tree, bounding a marginal cumulative distribution, for percentile()
         * @details
         * Holds the active terms -- unopened nodes, each with its weight
         * and extent in the dimension, and points evaluated exactly --
         * and constant bounds on the terms retired from them, with the
         * slack those add to the gap between the bounds. With s the
         * square root of the marginal kernel's metric, each term's
         * kernel cumulative distribution is Phi(u) = erfc(-s u) / 2.
         */
        class PercentileCut
        {
        public:

            /** Which bound on the cumulative distribution to evaluate */
            enum class Bound : std::uint8_t
            {
                lower,   /**< F_L */
                upper,   /**< F_U */
                mean     /**< (F_L + F_U) / 2 */
            };

            /**
             * @brief Start the cut at the root
             * @param kde The estimate
             * @param dim The dimension
             * @param s The square root of the marginal kernel's metric
             * @param tolP The tolerance in probability
             */
            PercentileCut(const KernelDensity& kde, const std::size_t dim, const double s, const double tolP) :
                kde_(std::cref(kde)), dim_(dim), s_(s), dnorm_(s / std::sqrt(std::numbers::pi)),
                slackBudget_(0.5 * tolP), retireTol_(1e-6 * tolP), tolP_(tolP)
            {
                addNode(rootNode, active_);
            }

            /**
             * @brief Evaluate a bound and its derivative
             * @param t Where to evaluate it
             * @param which The bound
             * @return The bound and its derivative at t
             */
            [[nodiscard]] auto eval(const double t, const Bound which) const -> std::pair<double, double>
            {
                double f = which == Bound::upper ? cU_ : cL_;
                if (which == Bound::mean) { f = 0.5 * (cL_ + cU_); }
                double df = 0.0;
                for (const auto& term : active_)
                {
                    if (which == Bound::mean && term.node_ != nullNode)
                    {
                        f += 0.5 * term.w_ * (cdf(t - term.lo_) + cdf(t - term.hi_));
                        df += 0.5 * term.w_ * (dens(t - term.lo_) + dens(t - term.hi_));
                        continue;
                    }
                    const double u = t - (which == Bound::upper ? term.lo_ : term.hi_);   // lo_ == hi_ for a point
                    f += term.w_ * cdf(u);
                    df += term.w_ * dens(u);
                }
                return {f, df};
            }

            /**
             * @brief Narrow a bracket on where a bound reaches a target
             * @param which The bound (non-decreasing)
             * @param target The target
             * @param a Left end of the bracket
             * @param b Right end of the bracket
             * @param ftol Stop once the bound at an end is within ftol of target
             * @return [l, r] within [a, b], with the bound at l at most,
             *   and at r at least, the target (l = r = a if the bound at a
             *   is at least the target, b if the bound at b is at most it),
             *   narrowed by safeguarded Newton iterations until the bound
             *   at an end is within ftol of the target, or [l, r] is at
             *   rounding
             */
            [[nodiscard]] auto solve(const Bound which, const double target, double a, double b,
                const double ftol) const -> std::pair<double, double>
            {
                if (eval(a, which).first >= target) { return {a, a}; }
                if (eval(b, which).first <= target) { return {b, b}; }
                double x = 0.5 * (a + b);
                for (int it = 0; it < 200; ++it)
                {
                    const auto [g, dg] = eval(x, which);
                    if (g < target) { a = x; }
                    else if (g > target) { b = x; }
                    else { return {x, x}; }
                    if (std::abs(g - target) <= ftol || atRounding(a, b)) { break; }
                    double next = dg > 0.0 ? x + ((target - g) / dg) : a;
                    if (!(next > a) || !(next < b)) { next = 0.5 * (a + b); }
                    x = next;
                }
                return {a, b};
            }

            /**
             * @brief Whether any active term is a node
             * @return True if some node is still unopened
             */
            [[nodiscard]] auto hasNodes() const -> bool
            {
                return std::ranges::any_of(active_, [](const Term& t) -> bool { return t.node_ != nullNode; });
            }

            /**
             * @brief Refine the cut for a bracket and candidate
             * @param tLo Left end of the bracket on the percentile
             * @param tHi Right end of the bracket
             * @param tC The candidate
             * @details
             * Retires the terms wholly left or right of [tLo, tHi] whose
             * slack fits the budget, then opens the nodes contributing
             * most to F_U - F_L at tC, largest first, until those left
             * unopened sum to at most a quarter of the tolerance (all of
             * them, for a zero tolerance), a leaf's points becoming exact
             * terms.
             */
            void refine(const double tLo, const double tHi, const double tC)
            {
                std::vector<Term> kept;
                kept.reserve(active_.size());
                std::vector<std::pair<double, std::size_t>> gaps;   // (gap at tC, index in kept)
                for (const auto& term : active_)
                {
                    if (retire(term, tLo, tHi)) { continue; }
                    if (term.node_ != nullNode)
                    {
                        gaps.emplace_back(term.w_ * (cdf(tC - term.lo_) - cdf(tC - term.hi_)), kept.size());
                    }
                    kept.push_back(term);
                }
                const std::vector<bool> open = toOpen(gaps, kept.size());
                active_.clear();
                for (std::size_t k = 0; k < kept.size(); ++k)
                {
                    const auto& term = kept[k]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
                    if (!open[k]) // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
                    {
                        active_.push_back(term);
                        continue;
                    }
                    const auto& node = kde_.get().nodes()[term.node_]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
                    addNode(node.lchild_, active_);
                    addNode(node.rchild_, active_);
                }
            }

        private:

            /** @brief One term of the cut */
            struct Term
            {
                double w_;           /**< Weight */
                double lo_;          /**< Least position in the dimension (that of the point, for a point) */
                double hi_;          /**< Greatest position in the dimension */
                std::size_t node_;   /**< The tree node, or nullNode for a single point */
            };

            /** @brief The kernel's cumulative distribution @param u Offset @return Phi(u) */
            [[nodiscard]] auto cdf(const double u) const -> double { return 0.5 * std::erfc(-s_ * u); }
            /** @brief Its complement @param u Offset @return 1 - Phi(u), accurately */
            [[nodiscard]] auto ccdf(const double u) const -> double { return 0.5 * std::erfc(s_ * u); }
            /** @brief Its density @param u Offset @return Phi'(u) */
            [[nodiscard]] auto dens(const double u) const -> double { return dnorm_ * std::exp(-s_ * s_ * u * u); }

            /**
             * @brief Whether a bracket is at rounding
             * @param a Left end
             * @param b Right end
             * @return True if [a, b] is within a few ulps
             */
            [[nodiscard]] static auto atRounding(const double a, const double b) -> bool
            {
                return b - a <= 4.0 * std::numeric_limits<double>::epsilon() * std::max(std::abs(a), std::abs(b));
            }

            /**
             * @brief Add a node to a list of terms: as one term, or as its points if a leaf
             * @param idx The node
             * @param terms The list
             */
            void addNode(const std::size_t idx, std::vector<Term>& terms) const
            {
                if (!(kde_.get().nodeW(idx) > 0.0)) { return; }
                const auto& node = kde_.get().nodes()[idx]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
                if (node.splitDim_ >= 0)
                {
                    terms.push_back({.w_ = kde_.get().nodeW(idx), .lo_ = kde_.get().nodeLo(idx)[dim_], // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
                        .hi_ = kde_.get().nodeHi(idx)[dim_], .node_ = idx}); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
                    return;
                }
                const auto& pts = kde_.get().data();
                for (std::size_t i = node.start_; i < node.start_ + node.count_; ++i)
                {
                    const double wi = pts.w_[i]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
                    const double xi = pts.position(i)[dim_]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
                    if (wi > 0.0) { terms.push_back({.w_ = wi, .lo_ = xi, .hi_ = xi, .node_ = nullNode}); }
                }
            }

            /**
             * @brief Retire a term wholly left or right of a bracket, if its slack fits
             * @param term The term
             * @param tLo Left end of the bracket
             * @param tHi Right end of the bracket
             * @return True if retired: folded into the constant bounds,
             *   which stay valid for every t in [tLo, tHi], since the
             *   bracket only shrinks
             */
            auto retire(const Term& term, const double tLo, const double tHi) -> bool
            {
                const double leftSlack = term.w_ * ccdf(tLo - term.hi_);
                if (leftSlack <= retireTol_ && slack_ + leftSlack <= slackBudget_)
                {
                    cL_ += term.w_ - leftSlack;
                    cU_ += term.w_;
                    slack_ += leftSlack;
                    return true;
                }
                const double rightSlack = term.w_ * cdf(tHi - term.lo_);
                if (rightSlack <= retireTol_ && slack_ + rightSlack <= slackBudget_)
                {
                    cU_ += rightSlack;
                    slack_ += rightSlack;
                    return true;
                }
                return false;
            }

            /**
             * @brief Which nodes to open: those with the largest gaps, until the rest fit a quarter of the tolerance
             * @param gaps (gap, index) of each active node
             * @param n Number of active terms
             * @return Whether to open each active term
             */
            [[nodiscard]] auto toOpen(std::vector<std::pair<double, std::size_t>>& gaps, const std::size_t n) const
                -> std::vector<bool>
            {
                std::vector<bool> open(n, false);
                std::ranges::sort(gaps, std::greater<>());
                double rest = 0.0;
                for (const auto& gap : gaps) { rest += gap.first; }
                for (const auto& [gap, k] : gaps)
                {
                    if (rest <= 0.25 * tolP_ && rest < gaps.front().first) { break; }
                    open[k] = true; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
                    rest -= gap;
                }
                return open;
            }

            std::reference_wrapper<const KernelDensity> kde_;   /**< The estimate */
            std::size_t dim_;             /**< The dimension */
            double s_;                    /**< Square root of the marginal kernel's metric */
            double dnorm_;                /**< s_ / sqrt(pi), the kernel density's peak */
            double slackBudget_;          /**< Most slack the retired terms may add to the gap */
            double retireTol_;            /**< Most slack one term may have to be retired */
            double tolP_;                 /**< The tolerance in probability */
            std::vector<Term> active_;    /**< The active terms */
            double cL_ = 0.0;             /**< Retired terms' lower bound */
            double cU_ = 0.0;             /**< Retired terms' upper bound */
            double slack_ = 0.0;          /**< Retired terms' total slack */
        };

        /**
         * @brief The work of percentile(), with its arguments checked
         * @param dim The dimension
         * @param p The fraction, in (0, 1)
         * @param tolP The tolerance in probability
         * @param s The square root of the marginal kernel's metric
         * @return The percentile (see percentile())
         */
        [[nodiscard]] auto evalPercentile(const std::size_t dim, const double p, const double tolP,
            const double s) const -> double
        {
            using Bound = typename PercentileCut::Bound;
            PercentileCut cut(*this, dim, s, tolP);

            // An initial bracket, wide enough that every term's kernel
            // cumulative distribution is 0 at its left end, 1 at its right
            const double reach = 40.0 / s;
            double tLo = this->nodeLo(rootNode)[dim] - reach; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
            double tHi = this->nodeHi(rootNode)[dim] + reach; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
            while (true)
            {
                // The bracket: t* is no less than where F_U reaches p, no
                // more than where F_L does; then the candidate, the end of
                // its own bracket where (F_L + F_U) / 2 is nearer p
                tLo = cut.solve(Bound::upper, p, tLo, tHi, tolP).first;
                tHi = cut.solve(Bound::lower, p, tLo, tHi, tolP).second;
                const auto [cl, cr] = cut.solve(Bound::mean, p, tLo, tHi, 0.1 * tolP);
                const double tC = std::abs(cut.eval(cl, Bound::mean).first - p) <=
                    std::abs(cut.eval(cr, Bound::mean).first - p) ? cl : cr;
                if (cut.eval(tC, Bound::upper).first - p <= tolP && p - cut.eval(tC, Bound::lower).first <= tolP)
                {
                    return tC;
                }
                if (!cut.hasNodes()) { return tC; }   // every term exact: tC is the root, to rounding
                cut.refine(tLo, tHi, tC);
            }
        }

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
     * Made by KernelDensity::conditional(), or by
     * KernelDensity::marginalize() (a conditional on no dimensions,
     * whose marginal() is 1). Holds the estimate's points (shifted,
     * reweighted, and pruned copies of the original ones, or for a
     * marginal, copies of them in the kept dimensions, with their
     * auxiliary data), a KernelDensity over them, and the kernel metric
     * to evaluate it with. The points and
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

        /**
         * @brief A one-dimensional percentile of the conditional density
         * @param dim Index of the dimension among the kept ones (see dims())
         * @param pct The percentile, in (0, 100)
         * @param relTol As for KernelDensity::percentile()
         * @param absTol As for KernelDensity::percentile()
         * @return As for KernelDensity::percentile(), evaluated with the
         *   conditional kernel metric
         * @throws As for KernelDensity::percentile()
         */
        [[nodiscard]] auto percentile(const std::size_t dim, const double pct, const double relTol,
            const double absTol) const -> double
        {
            return kde_.percentile(dim, pct, relTol, absTol, bandwidth_);
        }

        /**
         * @brief Several one-dimensional percentiles of the conditional density, in parallel
         * @param dim As for the single-percentile version
         * @param pct The percentiles, each in (0, 100)
         * @param relTol As for KernelDensity::percentile()
         * @param absTol As for KernelDensity::percentile()
         * @return One value per percentile
         * @throws As for KernelDensity's batch percentile()
         */
        [[nodiscard]] auto percentile(const std::size_t dim, const std::vector<double>& pct, const double relTol,
            const double absTol) const -> std::vector<double>
        {
            return kde_.percentile(dim, pct, relTol, absTol, bandwidth_);
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
