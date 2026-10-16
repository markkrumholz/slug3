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
#include <numbers>
#include <span>
#include <stdexcept>
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
     * the weights along with them.
     */
    template <typename AuxType = void>
    struct KernelDensityData : public KDTreeData<AuxType>
    {
        std::vector<double> w_;  /**< Weights, one per point */

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
     * sum of its points' weights and squared weights and their weighted
     * center of mass, in flat arrays indexed by node.
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
         * @brief Sum of the squared weights of a node's points
         * @param idx Index of the node
         * @return The sum
         */
        [[nodiscard]] auto nodeW2(const std::size_t idx) const -> double { return nodeW2_[idx]; } // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)

        /**
         * @brief Weighted center of mass of a node's points
         * @param idx Index of the node
         * @return Its nDim coordinates; zero if the node's total weight is zero
         * @details
         * Kept for the kernel density evaluation still to be built; if
         * the algorithm settled on does not need it, it (and its
         * nNodes x nDim storage) may be dropped.
         */
        [[nodiscard]] auto nodeCm(const std::size_t idx) const -> std::span<const double>
        {
            return std::span<const double>(nodeCm_).subspan(idx * this->nDim_, this->nDim_);
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

    protected:

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
            double wTot = 0.0;
            for (const double w : data.w_) { wTot += w; }
            if (!(wTot > 0.0))
            {
                throw std::invalid_argument("KernelDensity: weights must not all be zero");
            }
            for (double& w : data.w_) { w /= wTot; }
            return data;
        }

        /**
         * @brief Compute every node's weight sums and center of mass
         * @details
         * One pass up the tree, from the last level to the root (see
         * accumulateNode()). The first moments are accumulated in
         * nodeCm_ and divided by the weight sums only at the end, so
         * that zero-weight nodes need no special handling on the way
         * up. Nodes on the same level are independent, so each level is
         * processed in parallel if built with OpenMP.
         */
        void setNodeWeights()
        {
            const std::size_t nDim = this->nDim_;
            const std::size_t nNodes = this->nodes_.size();
            nodeW_.assign(nNodes, 0.0);
            nodeW2_.assign(nNodes, 0.0);
            nodeCm_.assign(nNodes * nDim, 0.0);

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

            // Turn first moments into centers of mass
#ifdef _OPENMP
#           pragma omp parallel for
#endif
            for (std::size_t idx = 0; idx < nNodes; ++idx)
            {
                const double w = nodeW_[idx]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
                const std::span<double> cm = std::span<double>(nodeCm_).subspan(idx * nDim, nDim);
                if (w != 0.0)
                {
                    for (auto& c : cm) { c /= w; }
                }
                else
                {
                    std::ranges::fill(cm, 0.0);
                }
            }
        }

        /**
         * @brief Set one node's weight sums and first moment
         * @param idx Index of the node; if it is an interior node, its
         *   children's values must already be set
         * @details
         * A leaf (or an unused slot below one, which has no points)
         * sums over its own points; an interior node adds up its two
         * children's sums. Leaves the unnormalized first moment, the
         * weighted sum of positions, in nodeCm_. Writes only this
         * node's own entries, so nodes may be processed in parallel.
         */
        void accumulateNode(const std::size_t idx)
        {
            const auto& node = this->nodes_[idx];
            const auto& pts = this->data_.get();
            const std::size_t nDim = this->nDim_;
            const std::span<double> cm = std::span<double>(nodeCm_).subspan(idx * nDim, nDim);
            if (node.splitDim_ < 0)
            {
                for (std::size_t i = node.start_; i < node.start_ + node.count_; ++i)
                {
                    const double w = pts.weight(i);
                    const auto pos = pts.position(i);
                    nodeW_[idx] += w; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
                    nodeW2_[idx] += w * w; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
                    for (std::size_t d = 0; d < nDim; ++d) { cm[d] += w * pos[d]; } // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
                }
                return;
            }
            const std::size_t l = node.lchild_;
            const std::size_t r = node.rchild_;
            nodeW_[idx] = nodeW_[l] + nodeW_[r]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
            nodeW2_[idx] = nodeW2_[l] + nodeW2_[r]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
            for (std::size_t d = 0; d < nDim; ++d)
            {
                cm[d] = nodeCm_[(l * nDim) + d] + nodeCm_[(r * nDim) + d]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
            }
        }

        std::vector<double> nodeW_;   /**< Sum of the weights of each node's points */
        std::vector<double> nodeW2_;  /**< Sum of the squared weights of each node's points */
        std::vector<double> nodeCm_;  /**< Weighted centers of mass of the nodes' points, nodes x nDim, node-major; may be dropped if the kernel density evaluation does not need it (see nodeCm()) */
    };

} // namespace kden

#endif // KERNELDENSITY_HPP
