/**
 * @file KDTree.hpp
 * @author Mark Krumholz
 * @brief A balanced kd tree of weighted points
 * @date 2026-10-14
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#ifndef KDTREE_HPP
#define KDTREE_HPP

#include <algorithm>
#include <cassert>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <functional>
#include <limits>
#include <mdspan> // NOLINT(misc-include-cleaner)
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

/**
 * @brief A namespace for kd trees and kernel density estimation
 */
namespace kden {

    /**
     * @brief Sentinel index marking a null (nonexistent) tree node
     * @details
     * All bits set, so it can never be a valid node index.
     */
    inline constexpr std::size_t nullNode = std::numeric_limits<std::size_t>::max();

    /**
     * @brief Index of the root node of a tree
     */
    inline constexpr std::size_t rootNode = 0;

    /**
     * @struct KDTreeNode
     * @brief A single node of a kd tree
     * @details
     * A node covers the contiguous block of count_ points that starts
     * at index start_ of the tree's point array, and the axis-aligned
     * box with corners lo_ and hi_. An interior node is split along
     * dimension splitDim_ into children lchild_ and rchild_; a leaf
     * has splitDim_ = -1 and both children equal to nullNode. The node
     * also caches the sum of its points' weights (w_), of their squared
     * weights (w2_), and their weighted center of mass (cm_). The
     * tree's dimensionality is a runtime quantity, so lo_, hi_, and
     * cm_ are vectors; they start empty and are sized (one element per
     * dimension) by the tree that owns the node.
     */
    struct KDTreeNode
    {
        std::size_t start_ = 0;                       /**< Index of the node's first point */
        std::size_t count_ = 0;                       /**< Number of points in the node */
        std::size_t lchild_ = nullNode;               /**< Index of the left child, or nullNode if this is a leaf */
        std::size_t rchild_ = nullNode;               /**< Index of the right child, or nullNode if this is a leaf */
        std::size_t parent_ = nullNode;               /**< Index of the parent node (nullNode for the root) */
        int splitDim_ = -1;                           /**< Dimension along which the node is split, or -1 if this is a leaf */
        std::vector<double> lo_;                      /**< Lower corner of the node's bounding box */
        std::vector<double> hi_;                      /**< Upper corner of the node's bounding box */
        double w_ = 0.0;                              /**< Sum of the weights of the node's points */
        double w2_ = 0.0;                             /**< Sum of the squared weights of the node's points */
        std::vector<double> cm_;                      /**< Weighted center of mass of the node's points */
    };

    /**
     * @brief Requirements on the type of a point stored in a KDTree
     * @tparam PtType The point type
     * @details
     * A point must be copyable and assignable (the tree reorders its
     * points in place), and must have public members x_, of type
     * std::vector<double>, holding its position (one element per
     * dimension of the tree), and w_, of type double, holding its
     * weight. It may carry any other information as well.
     */
    template <typename PtType>
    concept KDTreePoint = std::copyable<PtType> &&
        std::same_as<decltype(PtType::x_), std::vector<double>> &&
        std::same_as<decltype(PtType::w_), double>;

    /**
     * @class KDTree
     * @brief A balanced kd tree indexing a set of weighted points
     * @tparam PtType Type of a point; see KDTreePoint
     * @details
     * A non-owning wrapper around a vector of points that provides
     * tree-based access to them. The tree is balanced, and its nodes
     * are stored in a single contiguous array: node 0 is the root, and
     * the children of node i are nodes 2i + 1 and 2i + 2, so the tree
     * can be traversed with bit-shifting arithmetic. Building the tree
     * reorders the underlying points in place, so that every node's
     * points occupy a contiguous range of them.
     *
     * The number of dimensions, nDim, is a runtime parameter, and
     * every point's position x_ must have exactly nDim elements.
     * Distances may be measured using an inverse squared bandwidth
     * matrix h2Inv; if none is given, the metric is the usual
     * Euclidean one. The metric used to build the tree (which only
     * affects which dimension each node is split along) is not
     * stored: every query takes its own, so different threads can
     * query the same tree with different metrics at once.
     */
    template <typename PtType>
        requires KDTreePoint<PtType>
    class KDTree
    {
    public:

        // Tree traversal. Node 0 is the root, and the children of node
        // i are 2i + 1 (left, always odd) and 2i + 2 (right, always
        // even), so moving around the tree is pure index arithmetic.

        /**
         * @brief Index of a node's left child
         * @param idx Index of the node
         * @return 2 idx + 1
         */
        [[nodiscard]] static constexpr auto left(const std::size_t idx) -> std::size_t
        {
            return (idx << 1) + 1;
        }

        /**
         * @brief Index of a node's right child
         * @param idx Index of the node
         * @return 2 idx + 2
         */
        [[nodiscard]] static constexpr auto right(const std::size_t idx) -> std::size_t
        {
            return (idx << 1) + 2;
        }

        /**
         * @brief Index of a node's parent
         * @param idx Index of the node
         * @return (idx - 1) / 2, or nullNode if idx is the root
         */
        [[nodiscard]] static constexpr auto parent(const std::size_t idx) -> std::size_t
        {
            return idx == rootNode ? nullNode : (idx - 1) >> 1;
        }

        /**
         * @brief Index of a node's sibling
         * @param idx Index of the node
         * @return idx + 1 if idx is a left (odd) child, idx - 1 if it is
         *   a right (even) child, or nullNode if idx is the root, which
         *   has no sibling
         */
        [[nodiscard]] static constexpr auto sibling(const std::size_t idx) -> std::size_t
        {
            if (idx == rootNode) { return nullNode; }
            return (idx & 1) != 0 ? idx + 1 : idx - 1;
        }

        /**
         * @brief Index of the next node to visit after a node's whole subtree
         * @param idx Index of the node
         * @return The first node after idx and all its descendants in a
         *   depth-first, parent-before-children traversal, or nullNode
         *   if there is none (idx is on the tree's rightmost path)
         * @details
         * Climbs from idx while it is a right (even) child, then steps
         * to the right sibling of the left child reached. Calling this
         * in place of descending into idx's children skips its subtree,
         * which is how a traversal prunes nodes it does not need to
         * open.
         */
        [[nodiscard]] static constexpr auto next(std::size_t idx) -> std::size_t
        {
            while (idx != nullNode && (idx & 1) == 0) { idx = parent(idx); }
            return idx == nullNode ? nullNode : idx + 1;
        }

        /** Type of a non-owning view of an nDim x nDim inverse squared bandwidth matrix */
        using MatrixView = std::mdspan<const double, std::dextents<std::size_t, 2>>; // NOLINT(misc-include-cleaner) -- see the identical NOLINT on SpecsynLib.hpp's SpectraGrid alias

        /**
         * @brief Squared distance between two points in a given metric
         * @param x1 Position of the first point; elements may be NaN
         *   to leave that coordinate unspecified (see below)
         * @param x2 Position of the second point, likewise; must have
         *   the same number of elements as x1
         * @param h2Inv Inverse squared bandwidth matrix defining the
         *   metric, of size n x n where n is the number of elements of
         *   x1; if not given, the metric is Euclidean
         * @return With d_i = x1[i] - x2[i], or 0 if either is NaN:
         *   d^T h2Inv d if h2Inv is given, or the ordinary Euclidean
         *   squared distance d^T d if it is not
         * @details
         * h2Inv acts as the metric tensor (an inverse covariance
         * matrix): for a diagonal bandwidth h_i in each dimension, its
         * diagonal entries are 1 / h_i^2. The metric is an argument
         * rather than part of the tree because inference typically
         * uses a locally computed one (e.g. the tree's bandwidth
         * convolved with a particular observation's uncertainties),
         * and keeping it out of the tree's state lets different threads
         * use different metrics at the same time.
         *
         * A NaN coordinate in either point means "unspecified": that
         * dimension contributes zero separation, so the result is a
         * distance between a point and a line, a plane, etc., or
         * between two such objects. With s the specified dimensions,
         * the result is d_s^T M_ss d_s, using only the s-s block of
         * h2Inv. For a diagonal metric that is exactly the minimum
         * distance (equivalently, the metric of the Gaussian
         * marginalized over the unspecified dimensions). For a
         * non-diagonal metric it is not: both of those are given by the
         * Schur complement M_ss - M_sf M_ff^-1 M_fs, i.e. by the inverse
         * of the s-s block of the covariance, not by M_ss. Callers
         * that leave dimensions unspecified with a non-diagonal metric
         * should therefore pass that reduced matrix as the s-s block
         * of h2Inv; its other entries are then irrelevant.
         */
        [[nodiscard]] static constexpr auto dist2(const std::vector<double>& x1,
            const std::vector<double>& x2,
            const std::optional<MatrixView> h2Inv = std::nullopt) -> double
        {
            const std::size_t n = x1.size();
            assert(x2.size() == n);
            // Separation in dimension i, or 0 if either coordinate is unspecified
            auto sep = [&x1, &x2](const std::size_t i) -> double
            {
                const double a = x1[i]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
                const double b = x2[i]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
                return (std::isnan(a) || std::isnan(b)) ? 0.0 : a - b;
            };
            double r2 = 0.0;
            if (!h2Inv.has_value())
            {
                for (std::size_t i = 0; i < n; ++i)
                {
                    const double d = sep(i);
                    r2 += d * d;
                }
                return r2;
            }
            const MatrixView& m = *h2Inv;
            assert(m.extent(0) == n && m.extent(1) == n);
            for (std::size_t i = 0; i < n; ++i)
            {
                const double di = sep(i);
                if (di == 0.0) { continue; }
                for (std::size_t j = 0; j < n; ++j)
                {
                    r2 += di * sep(j) * m[i, j];
                }
            }
            return r2;
        }

        /**
         * @brief Construct a KDTree over a set of points
         * @param data The points to index; not copied, so must outlive
         *   the tree, and reordered in place as the tree is built
         * @param nDim Number of dimensions of the tree; must be at
         *   least 1, and equal to the size of every point's x_
         * @param leafSize Maximum number of points in a leaf; must be
         *   at least 1
         * @param h2Inv Inverse squared bandwidth matrix defining the
         *   metric used to choose split dimensions while building the
         *   tree (see chooseSplitDim()), of size nDim x nDim; if not
         *   given, the metric is Euclidean. Used only during
         *   construction and not stored; queries take their own metric.
         * @throws std::invalid_argument if nDim, leafSize, or the shape
         *   of h2Inv is invalid, or if any point's x_ does not have
         *   exactly nDim elements
         * @details
         * Sizes the tree, then initializes its root node to cover
         * every point: its bounding box, the sum of its points'
         * weights and squared weights, and their weighted center of
         * mass (left at zero if the total weight is zero). Finally
         * builds the rest of the tree by splitting every node, level
         * by level (see splitNode()), in parallel within each level if
         * built with OpenMP. This reorders data in place.
         */
        KDTree(std::vector<PtType>& data,
            const std::size_t nDim,
            const std::size_t leafSize = 16,
            const std::optional<MatrixView> h2Inv = std::nullopt) :
            nDim_(nDim),
            data_(std::ref(data)),
            leafSize_(leafSize)
        {
            if (nDim_ < 1)
            {
                throw std::invalid_argument("KDTree: nDim must be at least 1");
            }
            if (leafSize_ < 1)
            {
                throw std::invalid_argument("KDTree: leafSize must be at least 1");
            }
            if (h2Inv.has_value() &&
                (h2Inv->extent(0) != nDim_ || h2Inv->extent(1) != nDim_))
            {
                throw std::invalid_argument("KDTree: h2Inv must be of size nDim x nDim");
            }

            // Number of leaves, levels, and nodes: keep halving the
            // number of points until it falls below the leaf size,
            // doubling the number of leaves (and adding a level) each time
            std::size_t nNodes = 1;
            for (std::size_t n = data_.get().size(); n >= leafSize_; n >>= 1)
            {
                nLeaves_ <<= 1;
                nLevels_++;
                nNodes += nLeaves_;
            }
            nodes_.resize(nNodes);
            for (auto& node : nodes_)
            {
                node.lo_.resize(nDim_);
                node.hi_.resize(nDim_);
                node.cm_.resize(nDim_);
            }

            // Root node: covers every point. Check every point's
            // dimensionality once here, so nothing downstream has to.
            for (const auto& pt : data_.get())
            {
                if (pt.x_.size() != nDim_)
                {
                    throw std::invalid_argument(
                        "KDTree: every point's x_ must have exactly nDim elements");
                }
            }
            nodes_[rootNode].start_ = 0;
            nodes_[rootNode].count_ = data_.get().size();
            setNodeStats(rootNode);

            // Build the tree breadth first, one level at a time, so each
            // node is split after its parent. Nodes on the same level
            // are independent -- each touches only itself, its own two
            // children, and its own contiguous range of points -- so a
            // level can be split in parallel. Level k holds nodes
            // 2^k - 1 through 2^(k+1) - 2; nodes on the last level have
            // at most leafSize_ points, so splitNode() just marks them
            // as leaves.
            std::size_t first = rootNode;
            std::size_t width = 1;
            for (std::size_t level = 0; level < nLevels_; ++level)
            {
#ifdef _OPENMP
#               pragma omp parallel for
#endif
                for (std::size_t i = first; i < first + width; ++i)
                {
                    splitNode(i, h2Inv);
                }
                first += width;
                width <<= 1;
            }
        }

    private:

        /**
         * @brief Choose the dimension along which to split a node
         * @param idx Index of the node; its lo_ and hi_ must already be set
         * @param h2Inv Inverse squared bandwidth matrix defining the
         *   metric, or none for Euclidean
         * @return The dimension to split along, in [0, nDim)
         * @details
         * Splits along the dimension in which the node is widest,
         * measured in the tree's metric: the one maximizing
         * (hi_[i] - lo_[i])^2 h2Inv[i,i], the squared metric length of
         * the node's extent along axis i (for a diagonal bandwidth h_i,
         * its extent in units of the bandwidth, squared), or just
         * (hi_[i] - lo_[i])^2 if h2Inv is not given. Ties go to the
         * lowest dimension.
         */
        [[nodiscard]] auto chooseSplitDim(const std::size_t idx,
            const std::optional<MatrixView> h2Inv) const -> int
        {
            const auto& node = nodes_[idx];
            std::size_t best = 0;
            double bestSize = -1.0;
            for (std::size_t i = 0; i < nDim_; ++i)
            {
                const double width = node.hi_[i] - node.lo_[i]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
                double size = width * width;
                if (h2Inv.has_value()) { size *= (*h2Inv)[i, i]; }
                if (size > bestSize)
                {
                    best = i;
                    bestSize = size;
                }
            }
            return static_cast<int>(best);
        }

        /**
         * @brief Set a node's bounding box, weight sums, and center of mass from its points
         * @param idx Index of the node; its start_ and count_ must already be set
         * @details
         * lo_ and hi_ are the actual extent of the node's points (not
         * the planes that split them off from their siblings); cm_ is
         * their weighted center of mass, left at zero if their total
         * weight is zero. A node with no points gets lo_ = +inf and
         * hi_ = -inf in every dimension.
         */
        void setNodeStats(const std::size_t idx)
        {
            auto& node = nodes_[idx];
            const auto& pts = data_.get();
            node.lo_.assign(nDim_, std::numeric_limits<double>::infinity());
            node.hi_.assign(nDim_, -std::numeric_limits<double>::infinity());
            node.cm_.assign(nDim_, 0.0);
            node.w_ = 0.0;
            node.w2_ = 0.0;
            for (std::size_t i = node.start_; i < node.start_ + node.count_; ++i)
            {
                const auto& pt = pts[i]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
                for (std::size_t d = 0; d < nDim_; ++d)
                {
                    const double x = pt.x_[d]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
                    node.lo_[d] = std::min(node.lo_[d], x); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
                    node.hi_[d] = std::max(node.hi_[d], x); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
                    node.cm_[d] += pt.w_ * x; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
                }
                node.w_ += pt.w_;
                node.w2_ += pt.w_ * pt.w_;
            }
            if (node.w_ != 0.0)
            {
                for (auto& c : node.cm_) { c /= node.w_; }
            }
            else
            {
                node.cm_.assign(nDim_, 0.0);
            }
        }

        /**
         * @brief Partially sort a node's points about their median along its split dimension
         * @param idx Index of the node; its splitDim_ must already be set
         * @details
         * Reorders the node's points (an in-place selection, following
         * slug2's partition_node) so that, with m = start_ +
         * (count_ - 1) / 2, the point at index m is the one that would
         * be there if the node's points were fully sorted by their
         * coordinate along splitDim_, every point before it has a
         * coordinate no larger, and every point after it one no
         * smaller. Points are swapped whole, so any other data they
         * carry (e.g. their original index) moves with them.
         */
        void partitionNode(const std::size_t idx)
        {
            const auto& node = nodes_[idx];
            auto& pts = data_.get();
            const auto dim = static_cast<std::size_t>(node.splitDim_);
            auto key = [&pts, dim](const std::size_t i) -> double
            {
                return pts[i].x_[dim]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
            };

            std::size_t l = node.start_;
            std::size_t r = node.start_ + node.count_ - 1;
            const std::size_t m = node.start_ + ((node.count_ - 1) >> 1);

            // Iterate until the part of the list containing the middle
            // index has been narrowed down to that index alone
            while (l < r)
            {
                std::size_t lptr = l;
                std::size_t rptr = r;
                const double splitVal = key((lptr + rptr) / 2);

                // Put points below the partition value to the left, and
                // points above it to the right
                while (true)
                {
                    while (key(lptr) < splitVal && lptr < rptr) { lptr++; }
                    while (key(rptr) > splitVal && lptr <= rptr) { rptr--; }
                    if (lptr >= rptr) { break; }
                    std::swap(pts[lptr], pts[rptr]); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
                    lptr++;
                    rptr--;
                }

                // Make rptr the first point above the partition value
                if (key(rptr) > splitVal && lptr <= rptr) { rptr--; }
                rptr++;

                // Keep whichever side of the partition holds the middle index
                if (rptr > m) { r = rptr - 1; }
                else { l = rptr; }
            }
        }

        /**
         * @brief Split a node into two children, or mark it as a leaf
         * @param idx Index of the node; its start_, count_, and
         *   statistics (see setNodeStats()) must already be set
         * @param h2Inv Inverse squared bandwidth matrix defining the
         *   metric used to choose the split dimension, or none for
         *   Euclidean
         * @details
         * A node with at most leafSize_ points becomes a leaf
         * (splitDim_ = -1, no children). Otherwise it is split along
         * chooseSplitDim(idx, h2Inv): its points are partitioned about their
         * median along that dimension (see partitionNode()), the left
         * child gets the first (count_ + 1) / 2 of them and the right
         * child the rest, and each child's statistics are computed
         * from its own points.
         */
        void splitNode(const std::size_t idx, const std::optional<MatrixView> h2Inv)
        {
            auto& node = nodes_[idx];
            if (node.count_ <= leafSize_)
            {
                node.splitDim_ = -1;
                node.lchild_ = nullNode;
                node.rchild_ = nullNode;
                return;
            }

            // Partition the points along the split dimension
            node.splitDim_ = chooseSplitDim(idx, h2Inv);
            assert(node.splitDim_ >= 0 && static_cast<std::size_t>(node.splitDim_) < nDim_);
            partitionNode(idx);

            // Assign points to the children
            const std::size_t lIdx = left(idx);
            const std::size_t rIdx = right(idx);
            assert(rIdx < nodes_.size());
            node.lchild_ = lIdx;
            node.rchild_ = rIdx;
            auto& lNode = nodes_[lIdx];
            auto& rNode = nodes_[rIdx];
            lNode.parent_ = idx;
            rNode.parent_ = idx;
            lNode.start_ = node.start_;
            lNode.count_ = (node.count_ + 1) / 2;
            rNode.start_ = lNode.start_ + lNode.count_;
            rNode.count_ = node.count_ - lNode.count_;

            // Children's bounding boxes, weight sums, and centers of mass
            setNodeStats(lIdx);
            setNodeStats(rIdx);
        }

        std::size_t nDim_;                                      /**< Number of dimensions the tree is built over */
        std::reference_wrapper<std::vector<PtType>> data_;      /**< The (non-owned) points the tree indexes */
        std::vector<KDTreeNode> nodes_;                         /**< The tree's nodes, root first */
        std::size_t nLevels_ = 1;                               /**< Number of levels in the tree */
        std::size_t nLeaves_ = 1;                               /**< Number of leaves in the tree */
        std::size_t leafSize_;                                  /**< Maximum number of points in a leaf */
    };

} // namespace kden

#endif // KDTREE_HPP
