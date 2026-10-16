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
#include <cmath>
#include <cstddef>
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
         *   element per point, or any weight is not finite
         * @details
         * Non-finite weights would make the nodes' weight sums
         * non-finite.
         */
        void validate() const
        {
            KDTreeData<AuxType>::validate();
            if (w_.size() != this->size())
            {
                throw std::invalid_argument("KernelDensityData: w_ must hold one element per point");
            }
            if (!std::ranges::all_of(w_, [](const double v) -> bool { return std::isfinite(v); }))
            {
                throw std::invalid_argument("KernelDensityData: weights must be finite");
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
         *   constructor. Every weight must also be finite (see
         *   KernelDensityData::validate()).
         * @param leafSize As for the corresponding KDTree constructor
         * @param splitBandwidth As for the corresponding KDTree constructor
         * @throws std::invalid_argument as for the corresponding KDTree constructor
         */
        KernelDensity(KernelDensityData<AuxType>& data,
            const std::size_t leafSize,
            const Bandwidth& splitBandwidth) :
            Tree(data, leafSize, splitBandwidth)
        {
            setNodeWeights();
        }

        /**
         * @brief Construct a KernelDensity, choosing split dimensions with the Euclidean metric
         * @param data As for the other constructor
         * @param leafSize As for the other constructor
         * @throws std::invalid_argument as for the other constructor
         */
        explicit KernelDensity(KernelDensityData<AuxType>& data, const std::size_t leafSize = 16) :
            Tree(data, leafSize)
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

    protected:

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
