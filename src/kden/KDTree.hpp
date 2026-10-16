/**
 * @file KDTree.hpp
 * @author Mark Krumholz
 * @brief A balanced kd tree of points
 * @date 2026-10-14
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#ifndef KDTREE_HPP
#define KDTREE_HPP

#include "Bandwidth.hpp"
#include <Eigen/Dense> // NOLINT(misc-include-cleaner) -- provides Eigen::Index; clang-tidy's IWYU mapping doesn't know this header
#include <algorithm>
#include <cassert>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <exception>
#include <functional>
#include <limits>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>
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
     * at index start_ of the tree's point array. An interior node is
     * split along dimension splitDim_ into children lchild_ and
     * rchild_; a leaf has splitDim_ = -1 and both children equal to
     * nullNode. Its bounding box is not stored here but in flat
     * arrays owned by the tree (see KDTree::nodeLo() and nodeHi()), so
     * that a node is a small fixed-size record with no allocations of
     * its own; classes built on KDTree that keep further per-node data
     * (e.g. KernelDensity's weight sums) do likewise, in arrays
     * indexed by node.
     */
    struct KDTreeNode
    {
        std::size_t start_ = 0;                       /**< Index of the node's first point */
        std::size_t count_ = 0;                       /**< Number of points in the node */
        std::size_t lchild_ = nullNode;               /**< Index of the left child, or nullNode if this is a leaf */
        std::size_t rchild_ = nullNode;               /**< Index of the right child, or nullNode if this is a leaf */
        std::size_t parent_ = nullNode;               /**< Index of the parent node (nullNode for the root) */
        int splitDim_ = -1;                           /**< Dimension along which the node is split, or -1 if this is a leaf */
    };

    /**
     * @brief Storage type for a KDTreeData's auxiliary data
     * @tparam AuxType Type of one point's auxiliary data, or void for none
     * @details std::vector<AuxType>, or an empty placeholder if AuxType is void
     */
    template <typename AuxType>
    struct KDTreeAuxStorage
    {
        using Type = std::vector<AuxType>; /**< One element per point */
    };
    /** @brief No auxiliary data: an empty placeholder */
    template <>
    struct KDTreeAuxStorage<void>
    {
        using Type = std::monostate; /**< Empty */
    };

    /**
     * @struct KDTreeData
     * @brief The points a KDTree indexes, stored as a struct of arrays
     * @tparam AuxType Type of the auxiliary data carried by each point
     *   (e.g. an index into an original data set), or void (the
     *   default) for none
     * @details
     * Holds N points of nDim_ dimensions each, as one flat array of
     * positions (x_, N x nDim_ doubles, point-major: point i occupies
     * x_[i * nDim_] through x_[i * nDim_ + nDim_ - 1]) and, unless
     * AuxType is void, one of auxiliary data (aux_, N elements of
     * AuxType). Building a KDTree over it reorders both together, in
     * place. With AuxType void, aux_ is an empty placeholder that
     * occupies no storage.
     *
     * A type derived from this may add further per-point arrays (as
     * KernelDensityData adds weights) and have the tree reorder them
     * too, by hiding swap() with a version that swaps its own arrays
     * and then calls this one, and likewise extending validate(): the
     * tree is templated on the concrete data type (see KDTreePoints),
     * so calls its versions directly, with no virtual dispatch.
     */
    template <typename AuxType = void>
    struct KDTreeData
    {
        /** The type of the auxiliary data, or void */
        using Aux = AuxType;

        /** Whether points carry auxiliary data */
        static constexpr bool hasAux = !std::is_void_v<AuxType>;

        std::size_t nDim_ = 0;   /**< Number of dimensions of each position */
        std::vector<double> x_;  /**< Positions, N x nDim_, point-major */
        [[no_unique_address]] typename KDTreeAuxStorage<AuxType>::Type aux_; /**< Auxiliary data, one per point (if AuxType is not void) */

        /** @brief Number of points
         *  @return N, the number of positions (zero if nDim_ is zero) */
        [[nodiscard]] auto size() const -> std::size_t { return nDim_ == 0 ? 0 : x_.size() / nDim_; }

        /**
         * @brief Position of a point
         * @param i Index of the point
         * @return A view of its nDim_ coordinates
         */
        [[nodiscard]] auto position(const std::size_t i) const -> std::span<const double>
        {
            return std::span<const double>(x_).subspan(i * nDim_, nDim_);
        }

        /**
         * @brief Auxiliary data of a point
         * @param i Index of the point
         * @return Its auxiliary data
         */
        [[nodiscard]] auto aux(const std::size_t i) const -> const auto&
            requires hasAux
        {
            return aux_[i]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
        }

        /**
         * @brief Check that the arrays are consistent with one another, and finite
         * @throws std::invalid_argument if nDim_ is 0, x_'s size is not
         *   a multiple of nDim_, aux_ does not have one element per
         *   point, or any position coordinate is not finite
         * @details
         * Stored positions must be finite: NaN marks unspecified
         * dimensions only in search points, and a non-finite stored
         * coordinate would give a node non-finite bounds, which the
         * search's pruning bound cannot handle (and would break the
         * ordering the build relies on). Data with non-finite values
         * -- e.g. infinite magnitudes for clusters with no luminous
         * stars -- must be filtered before building a tree.
         */
        void validate() const
        {
            if (nDim_ < 1)
            {
                throw std::invalid_argument("KDTreeData: nDim_ must be at least 1");
            }
            if (x_.size() % nDim_ != 0)
            {
                throw std::invalid_argument("KDTreeData: x_ must hold nDim_ coordinates per point");
            }
            if (!std::ranges::all_of(x_, [](const double v) -> bool { return std::isfinite(v); }))
            {
                throw std::invalid_argument("KDTreeData: positions must be finite (NaN is "
                    "supported only in search points)");
            }
            if constexpr (hasAux)
            {
                if (aux_.size() != size())
                {
                    throw std::invalid_argument("KDTreeData: aux_ must hold one element per point");
                }
            }
        }

        /**
         * @brief Exchange two points: their positions and auxiliary data
         * @param i Index of one point
         * @param j Index of the other
         */
        void swap(const std::size_t i, const std::size_t j)
        {
            std::swap_ranges(x_.begin() + static_cast<std::ptrdiff_t>(i * nDim_),
                x_.begin() + static_cast<std::ptrdiff_t>((i + 1) * nDim_),
                x_.begin() + static_cast<std::ptrdiff_t>(j * nDim_));
            if constexpr (hasAux)
            {
                std::swap(aux_[i], aux_[j]); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
            }
        }
    };

    /**
     * @brief A set of points a KDTree can index: KDTreeData, or a type derived from it
     * @tparam Data The candidate type
     * @details
     * The tree reorders points through Data::swap() and checks them
     * with Data::validate(), so a derived type's own versions of those
     * (see KDTreeData) are the ones used.
     */
    template <typename Data>
    concept KDTreePoints = std::derived_from<Data, KDTreeData<typename Data::Aux>>;

    /**
     * @struct Neighbor
     * @brief One result of a nearest-neighbor search
     */
    struct Neighbor
    {
        std::size_t idx_ = 0;  /**< Index of the point in the tree's (reordered) KDTreeData */
        double dist2_ = 0.0;   /**< Squared distance from the search point */
    };

    /**
     * @class KDTree
     * @brief A balanced kd tree indexing a set of points
     * @tparam Data Type of the points: KDTreeData (the default, with no
     *   auxiliary data), or a type derived from it (see KDTreePoints)
     * @details
     * A non-owning wrapper around a set of points (a Data) that
     * provides tree-based access to them. The tree is balanced, and its nodes
     * are stored in a single contiguous array: node 0 is the root, and
     * the children of node i are nodes 2i + 1 and 2i + 2, so the tree
     * can be traversed with bit-shifting arithmetic. Building the tree
     * reorders the underlying points in place, so that every node's
     * points occupy a contiguous range of them.
     *
     * The number of dimensions, nDim, is a runtime parameter, taken
     * from the data. Distances are measured in a metric defined by an inverse
     * squared bandwidth matrix (see Bandwidth), or the Euclidean one
     * if none is given. The metric used to build the tree (which only
     * affects which dimension each node is split along) is not
     * stored: every search takes its own, so different threads can
     * search the same tree with different metrics at once.
     */
    template <KDTreePoints Data = KDTreeData<>>
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

        /**
         * @brief Construct a KDTree over a set of points
         * @param data The points to index; not copied, so must outlive
         *   the tree, and reordered in place as the tree is built. Its
         *   dimensionality sets the tree's. It must pass its own
         *   validate(): for KDTreeData, every position coordinate must
         *   be finite.
         * @param leafSize Maximum number of points in a leaf; must be
         *   at least 1
         * @param splitBandwidth Metric used to choose split dimensions
         *   while building the tree (see chooseSplitDim()), for nDim
         *   dimensions. Used only during construction and not stored;
         *   searches take their own metric.
         * @throws std::invalid_argument if data is inconsistent (see
         *   Data::validate()), leafSize is 0, or splitBandwidth is
         *   not for data's number of dimensions
         * @details
         * Sizes the tree, then initializes its root node to cover
         * every point, with their bounding box. Finally
         * builds the rest of the tree by splitting every node, level
         * by level (see splitNode()), in parallel within each level if
         * built with OpenMP. This reorders data in place.
         */
        KDTree(Data& data,
            const std::size_t leafSize,
            const Bandwidth& splitBandwidth) :
            nDim_(data.nDim_),
            data_(std::ref(data)),
            leafSize_(leafSize),
            euclidean_(Bandwidth::identity(std::max<std::size_t>(data.nDim_, 1)))
        {
            data.validate();
            if (leafSize_ < 1)
            {
                throw std::invalid_argument("KDTree: leafSize must be at least 1");
            }
            if (splitBandwidth.nDim() != nDim_)
            {
                throw std::invalid_argument("KDTree: splitBandwidth must be for nDim dimensions");
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
            nodeLo_.assign(nNodes * nDim_, 0.0);
            nodeHi_.assign(nNodes * nDim_, 0.0);

            // Root node: covers every point
            nodes_[rootNode].start_ = 0;
            nodes_[rootNode].count_ = data_.get().size();
            setNodeBounds(rootNode);

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
                    splitNode(i, splitBandwidth);
                }
                first += width;
                width <<= 1;
            }
        }

        /**
         * @brief Construct a KDTree, choosing split dimensions with the Euclidean metric
         * @param data As for the other constructor
         * @param leafSize As for the other constructor
         * @throws std::invalid_argument as for the other constructor
         */
        explicit KDTree(Data& data, const std::size_t leafSize = 16) :
            KDTree(data, leafSize, Bandwidth::identity(std::max<std::size_t>(data.nDim_, 1)))
        {}

        // Read-only access to the tree's structure
        /** @brief Number of dimensions of the tree
         *  @return The number of dimensions */
        [[nodiscard]] auto nDim() const -> std::size_t { return nDim_; }
        /** @brief Maximum number of points in a leaf
         *  @return The leaf size */
        [[nodiscard]] auto leafSize() const -> std::size_t { return leafSize_; }
        /** @brief Number of levels allocated for the tree
         *  @return The number of levels (an upper bound on the depth;
         *    see splitNode()) */
        [[nodiscard]] auto nLevels() const -> std::size_t { return nLevels_; }
        /** @brief Number of leaves allocated for the tree
         *  @return The number of leaf slots on the last level (an upper
         *    bound on the number of actual leaves) */
        [[nodiscard]] auto nLeaves() const -> std::size_t { return nLeaves_; }
        /** @brief The tree's nodes, in index order (root first)
         *  @return The nodes; slots below a leaf are unused, and hold
         *    default (empty) nodes */
        [[nodiscard]] auto nodes() const -> const std::vector<KDTreeNode>& { return nodes_; }
        /**
         * @brief Lower corner of a node's bounding box
         * @param idx Index of the node
         * @return Its nDim coordinates: the minimum of the node's points'
         *   positions in each dimension (the actual extent of its
         *   points, not the plane that split it from its sibling)
         */
        [[nodiscard]] auto nodeLo(const std::size_t idx) const -> std::span<const double>
        {
            return std::span<const double>(nodeLo_).subspan(idx * nDim_, nDim_);
        }
        /**
         * @brief Upper corner of a node's bounding box
         * @param idx Index of the node
         * @return Its nDim coordinates (see nodeLo())
         */
        [[nodiscard]] auto nodeHi(const std::size_t idx) const -> std::span<const double>
        {
            return std::span<const double>(nodeHi_).subspan(idx * nDim_, nDim_);
        }
        /** @brief The points the tree indexes, in tree order
         *  @return The (non-owned) points, reordered so that each node's
         *    points are contiguous; Neighbor::idx_ indexes into these */
        [[nodiscard]] auto data() const -> const Data& { return data_.get(); }

        /**
         * @brief Find the nearest neighbors of a point
         * @param x Position to search around; must have nDim elements.
         *   Elements may be NaN, to judge nearness only in the other
         *   dimensions.
         * @param nNeighbor Number of neighbors to find
         * @param bandwidth The metric, for nDim dimensions. If x has NaN
         *   elements and bandwidth has not been reduced for exactly that
         *   pattern (see Bandwidth::reduce()), it is reduced here, at
         *   O(nDim^3) cost; for many searches with the same metric and
         *   pattern, reduce once and pass the result.
         * @return The min(nNeighbor, number of points) points nearest
         *   to x, in order of increasing distance, each as the index of
         *   the point in data() and its squared distance from x (as
         *   computed by Bandwidth::dist2(), with the reduced bandwidth
         *   if x has NaN elements). Use data().position() and aux()
         *   (and any accessors Data adds) to get at the point itself.
         * @throws std::invalid_argument if x or bandwidth is not for
         *   nDim dimensions, or bandwidth has been reduced for a
         *   different pattern of NaNs than x's
         * @throws std::domain_error if reducing bandwidth fails (see
         *   Bandwidth::reduce())
         * @details
         * Exact k-nearest-neighbor search. The tree is walked depth
         * first, visiting the nearer child of each node first, so that
         * close neighbors are found early; a node is skipped, with its
         * whole subtree, if a rigorous lower bound on the distance from
         * x to any point in its bounding box (see
         * Bandwidth::BoxBounder) exceeds the distance to the current
         * k'th nearest neighbor. Thread-safe: neither the tree nor the
         * bandwidth is modified, so different threads may search at
         * once, sharing a bandwidth or each with its own.
         */
        [[nodiscard]] auto neighbors(const std::vector<double>& x,
            const std::size_t nNeighbor,
            const Bandwidth& bandwidth) const
            -> std::vector<Neighbor>
        {
            if (x.size() != nDim_ || bandwidth.nDim() != nDim_)
            {
                throw std::invalid_argument("KDTree::neighbors: x and bandwidth must be "
                    "for nDim dimensions");
            }
            if (bandwidth.matches(x)) { return search(x, nNeighbor, bandwidth); }
            if (bandwidth.isReduced())
            {
                throw std::invalid_argument("KDTree::neighbors: bandwidth was reduced for "
                    "a different pattern of NaNs than x's");
            }
            return search(x, nNeighbor, bandwidth.reduce(Bandwidth::nanPattern(x)));
        }

        /**
         * @brief Find the nearest neighbors of a point, in the Euclidean metric
         * @param x As for the other overload
         * @param nNeighbor As for the other overload
         * @return As for the other overload
         * @throws std::invalid_argument if x does not have nDim elements
         */
        [[nodiscard]] auto neighbors(const std::vector<double>& x,
            const std::size_t nNeighbor) const
            -> std::vector<Neighbor>
        {
            return neighbors(x, nNeighbor, euclidean_);
        }

        /**
         * @brief Find the nearest neighbors of each of several points, in parallel
         * @param x Positions to search around, one per search
         * @param nNeighbor Number of neighbors to find, one per search
         * @param bandwidth Metric to use, one per search
         * @return One result per search, each exactly as returned by the
         *   single-point version for the corresponding arguments
         * @throws std::invalid_argument if x, nNeighbor, and bandwidth do
         *   not all have the same number of elements; otherwise, any
         *   exception the single-point version throws for any one of
         *   the searches (the first one caught, if several do)
         * @details
         * Runs the single-point search for each element, in parallel
         * (dynamically scheduled, since searches can differ widely in
         * cost) if built with OpenMP.
         */
        [[nodiscard]] auto neighbors(const std::vector<std::vector<double>>& x,
            const std::vector<std::size_t>& nNeighbor,
            const std::vector<Bandwidth>& bandwidth) const
            -> std::vector<std::vector<Neighbor>>
        {
            if (bandwidth.size() != x.size())
            {
                throw std::invalid_argument("KDTree::neighbors: x, nNeighbor, and bandwidth "
                    "must all have the same number of elements");
            }
            return batchNeighbors(x, nNeighbor, [this, &bandwidth](const std::vector<double>& xi,
                const std::size_t ki, const std::size_t i) -> std::vector<Neighbor>
            {
                return neighbors(xi, ki, bandwidth[i]); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
            });
        }

        /**
         * @brief Find the nearest neighbors of each of several points, in parallel, sharing one metric
         * @param x Positions to search around, one per search
         * @param nNeighbor Number of neighbors to find, one per search
         * @param bandwidth The metric, shared by every search
         * @return One result per search, each exactly as returned by the
         *   single-point version
         * @throws As for the version taking one bandwidth per search
         * @details
         * If bandwidth has not been reduced and every search point has
         * the same pattern of NaNs, it is reduced once for all of them
         * rather than once per search.
         */
        [[nodiscard]] auto neighbors(const std::vector<std::vector<double>>& x,
            const std::vector<std::size_t>& nNeighbor,
            const Bandwidth& bandwidth) const
            -> std::vector<std::vector<Neighbor>>
        {
            // Reduce once up front if every point needs the same reduction
            if (!x.empty() && !bandwidth.isReduced() && x.front().size() == nDim_ &&
                !bandwidth.matches(x.front()))
            {
                const std::vector<bool> pattern = Bandwidth::nanPattern(x.front());
                const bool shared = std::ranges::all_of(x, [&pattern](const std::vector<double>& xi) -> bool
                    { return Bandwidth::nanPattern(xi) == pattern; });
                if (shared)
                {
                    const Bandwidth reduced = bandwidth.reduce(pattern);
                    return batchNeighbors(x, nNeighbor, [this, &reduced](const std::vector<double>& xi,
                        const std::size_t ki, const std::size_t) -> std::vector<Neighbor>
                    {
                        return neighbors(xi, ki, reduced);
                    });
                }
            }
            return batchNeighbors(x, nNeighbor, [this, &bandwidth](const std::vector<double>& xi,
                const std::size_t ki, const std::size_t) -> std::vector<Neighbor>
            {
                return neighbors(xi, ki, bandwidth);
            });
        }

        /**
         * @brief Find the nearest neighbors of each of several points, in parallel, in the Euclidean metric
         * @param x Positions to search around, one per search
         * @param nNeighbor Number of neighbors to find, one per search
         * @return One result per search
         * @throws As for the shared-bandwidth version
         */
        [[nodiscard]] auto neighbors(const std::vector<std::vector<double>>& x,
            const std::vector<std::size_t>& nNeighbor) const
            -> std::vector<std::vector<Neighbor>>
        {
            return neighbors(x, nNeighbor, euclidean_);
        }

    protected:

        /**
         * @brief The nearest-neighbor search itself
         * @param x The search point; must match bandwidth (see Bandwidth::matches())
         * @param nNeighbor Number of neighbors to find
         * @param bandwidth The metric
         * @return As for neighbors()
         * @details
         * Depth-first, nearer child first, with an explicit stack of
         * (node, lower bound) pairs: a node popped whose bound now
         * exceeds the current k'th distance is skipped. Candidates are
         * kept as (squared distance, point index) pairs, sorted by
         * distance.
         */
        [[nodiscard]] auto search(const std::vector<double>& x,
            const std::size_t nNeighbor,
            const Bandwidth& bandwidth) const
            -> std::vector<Neighbor>
        {
            const auto& pts = data_.get();
            const std::size_t k = std::min(nNeighbor, pts.size());
            if (k == 0) { return {}; }

            const auto bound = bandwidth.boxBounder(x);
            std::vector<double> scratch;
            std::vector<std::pair<double, std::size_t>> best;
            best.reserve(k + 1);
            auto kth = [&best, k]() -> double
            {
                return best.size() < k ? std::numeric_limits<double>::infinity() : best.back().first;
            };

            std::vector<std::pair<std::size_t, double>> stack;
            stack.reserve((2 * nLevels_) + 2);
            const auto& root = nodes_[rootNode];
            stack.emplace_back(rootNode, bound.lower(nodeLo(rootNode), nodeHi(rootNode)));
            while (!stack.empty())
            {
                const auto [idx, nodeBound] = stack.back();
                stack.pop_back();
                if (nodeBound > kth()) { continue; }
                const auto& node = nodes_[idx];

                // A leaf: check each of its points, inserting any closer
                // than the current k'th neighbor into the sorted list
                if (node.splitDim_ < 0)
                {
                    for (std::size_t i = node.start_; i < node.start_ + node.count_; ++i)
                    {
                        const double r2 = bandwidth.dist2(x, pts.position(i), scratch);
                        if (r2 > kth()) { continue; }
                        const auto pos = std::lower_bound(best.begin(), best.end(), r2,
                            [](const std::pair<double, std::size_t>& entry, const double value) -> bool { return entry.first < value; });
                        best.insert(pos, {r2, i});
                        if (best.size() > k) { best.pop_back(); }
                    }
                    continue;
                }

                // An interior node: push both children, nearer one last
                // so that it is searched first
                const double lBound = bound.lower(nodeLo(node.lchild_), nodeHi(node.lchild_));
                const double rBound = bound.lower(nodeLo(node.rchild_), nodeHi(node.rchild_));
                if (lBound <= rBound)
                {
                    stack.emplace_back(node.rchild_, rBound);
                    stack.emplace_back(node.lchild_, lBound);
                }
                else
                {
                    stack.emplace_back(node.lchild_, lBound);
                    stack.emplace_back(node.rchild_, rBound);
                }
            }

            std::vector<Neighbor> result;
            result.reserve(best.size());
            for (const auto& [r2, i] : best) { result.push_back({.idx_ = i, .dist2_ = r2}); }
            return result;
        }

        /**
         * @brief Run a batch of neighbor searches in parallel
         * @tparam Search Callable taking (x[i], nNeighbor[i], i) and
         *   returning one search's result
         * @param x Positions to search around, one per search
         * @param nNeighbor Number of neighbors to find, one per search
         * @param search The search to run for each element
         * @return One result per search
         * @throws std::invalid_argument if x and nNeighbor differ in
         *   length; otherwise, the first exception any search throws
         * @details
         * Dynamically scheduled OpenMP loop, if built with OpenMP. An
         * exception must not escape an OpenMP parallel region, so each
         * is caught in the loop and the first one rethrown afterwards.
         */
        template <typename Search>
        [[nodiscard]] auto batchNeighbors(const std::vector<std::vector<double>>& x,
            const std::vector<std::size_t>& nNeighbor, const Search& search) const
            -> std::vector<std::vector<Neighbor>>
        {
            if (nNeighbor.size() != x.size())
            {
                throw std::invalid_argument("KDTree::neighbors: x and nNeighbor "
                    "must have the same number of elements");
            }
            std::vector<std::vector<Neighbor>> result(x.size());
            std::exception_ptr error = nullptr;
#ifdef _OPENMP
#           pragma omp parallel for schedule(dynamic)
#endif
            for (std::size_t i = 0; i < x.size(); ++i)
            {
                try
                {
                    result[i] = search(x[i], nNeighbor[i], i); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
                }
                catch (...)
                {
#ifdef _OPENMP
#                   pragma omp critical(kdtreeNeighborsError)
#endif
                    {
                        if (!error) { error = std::current_exception(); }
                    }
                }
            }
            if (error) { std::rethrow_exception(error); }
            return result;
        }

        /**
         * @brief Choose the dimension along which to split a node
         * @param idx Index of the node; its bounding box must already be set
         * @param bandwidth The metric
         * @return The dimension to split along, in [0, nDim)
         * @details
         * Splits along the dimension in which the node is widest,
         * measured in the tree's metric: the one maximizing
         * (hi[i] - lo[i])^2 M_ii, the squared metric length of
         * the node's extent along axis i (for a diagonal bandwidth h_i,
         * its extent in units of the bandwidth, squared), or just
         * (hi[i] - lo[i])^2 for the Euclidean metric. Ties go to the
         * lowest dimension.
         */
        [[nodiscard]] auto chooseSplitDim(const std::size_t idx,
            const Bandwidth& bandwidth) const -> int
        {
            const auto lo = nodeLo(idx);
            const auto hi = nodeHi(idx);
            std::size_t best = 0;
            double bestSize = -1.0;
            for (std::size_t i = 0; i < nDim_; ++i)
            {
                const double width = hi[i] - lo[i]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
                double size = width * width;
                size *= bandwidth.h2Inv()(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(i)); // NOLINT(misc-include-cleaner) -- Eigen::Index is provided by <Eigen/Dense> (included above); clang-tidy's IWYU mapping doesn't know that
                if (size > bestSize)
                {
                    best = i;
                    bestSize = size;
                }
            }
            return static_cast<int>(best);
        }

        /**
         * @brief Set a node's bounding box from its points
         * @param idx Index of the node; its start_ and count_ must already be set
         * @details
         * The bounding box (nodeLo()/nodeHi()) is the actual extent of
         * the node's points, not the planes that split them off from
         * their siblings. A node with no points gets lo = +inf and
         * hi = -inf in every dimension. Writes only this node's own
         * slices of the flat arrays, so nodes may be processed in
         * parallel.
         */
        void setNodeBounds(const std::size_t idx)
        {
            const auto& node = nodes_[idx];
            const auto& pts = data_.get();
            const auto offset = static_cast<std::ptrdiff_t>(idx * nDim_);
            const auto len = static_cast<std::ptrdiff_t>(nDim_);
            const std::span<double> lo(nodeLo_.begin() + offset, nodeLo_.begin() + offset + len);
            const std::span<double> hi(nodeHi_.begin() + offset, nodeHi_.begin() + offset + len);
            std::ranges::fill(lo, std::numeric_limits<double>::infinity());
            std::ranges::fill(hi, -std::numeric_limits<double>::infinity());
            for (std::size_t i = node.start_; i < node.start_ + node.count_; ++i)
            {
                const auto pos = pts.position(i);
                for (std::size_t d = 0; d < nDim_; ++d)
                {
                    const double x = pos[d]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
                    lo[d] = std::min(lo[d], x); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
                    hi[d] = std::max(hi[d], x); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
                }
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
         * smaller. Points are swapped whole, with Data::swap(), so
         * auxiliary data and any per-point arrays a derived Data adds
         * move with their positions.
         */
        void partitionNode(const std::size_t idx)
        {
            const auto& node = nodes_[idx];
            auto& pts = data_.get();
            const auto dim = static_cast<std::size_t>(node.splitDim_);
            auto key = [&pts, dim, nDim = nDim_](const std::size_t i) -> double
            {
                return pts.x_[(i * nDim) + dim]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
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
                    pts.swap(lptr, rptr);
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
         *   bounding box (see setNodeBounds()) must already be set
         * @param bandwidth The metric used to choose the split dimension
         * @details
         * A node with at most leafSize_ points becomes a leaf
         * (splitDim_ = -1, no children). Otherwise it is split along
         * chooseSplitDim(idx, bandwidth): its points are partitioned about their
         * median along that dimension (see partitionNode()), the left
         * child gets the first (count_ + 1) / 2 of them and the right
         * child the rest, and each child's bounding box is computed
         * from its own points.
         */
        void splitNode(const std::size_t idx, const Bandwidth& bandwidth)
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
            node.splitDim_ = chooseSplitDim(idx, bandwidth);
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

            // Children's bounding boxes
            setNodeBounds(lIdx);
            setNodeBounds(rIdx);
        }

        std::size_t nDim_;                                      /**< Number of dimensions the tree is built over */
        std::reference_wrapper<Data> data_;                     /**< The (non-owned) points the tree indexes */
        std::vector<KDTreeNode> nodes_;                         /**< The tree's nodes, root first */
        std::vector<double> nodeLo_;                            /**< Lower corners of the nodes' bounding boxes, nodes x nDim, node-major */
        std::vector<double> nodeHi_;                            /**< Upper corners of the nodes' bounding boxes, likewise */
        std::size_t nLevels_ = 1;                               /**< Number of levels in the tree */
        std::size_t nLeaves_ = 1;                               /**< Number of leaves in the tree */
        std::size_t leafSize_;                                  /**< Maximum number of points in a leaf */
        Bandwidth euclidean_;                                   /**< The Euclidean metric, for searches given none */
    };

} // namespace kden

#endif // KDTREE_HPP
