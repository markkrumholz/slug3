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
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>
#ifdef _OPENMP
#   include <omp.h>
#endif

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
     * @brief Types for a KDTreeData's auxiliary data
     * @tparam AuxType Type of one point's auxiliary data, or void for none
     * @details View is a std::span<AuxType> over the caller's storage,
     *   and Owned a std::vector<AuxType> (see OwnedKDTreeData); both
     *   are an empty placeholder if AuxType is void
     */
    template <typename AuxType>
    struct KDTreeAuxStorage
    {
        using View = std::span<AuxType>;     /**< One element per point, not owned */
        using Owned = std::vector<AuxType>;  /**< One element per point, owned */
    };
    /** @brief No auxiliary data: an empty placeholder */
    template <>
    struct KDTreeAuxStorage<void>
    {
        using View = std::monostate;   /**< Empty */
        using Owned = std::monostate;  /**< Empty */
    };

    /**
     * @struct KDTreeData
     * @brief The points a KDTree indexes: a non-owning view of a struct of arrays
     * @tparam AuxType Type of the auxiliary data carried by each point
     *   (e.g. an index into an original data set), or void (the
     *   default) for none
     * @details
     * Views N points of nDim_ dimensions each, as one flat array of
     * positions (x_, N x nDim_ doubles, point-major: point i occupies
     * x_[i * nDim_] through x_[i * nDim_ + nDim_ - 1]) and, unless
     * AuxType is void, one of auxiliary data (aux_, N elements of
     * AuxType). Building a KDTree over it reorders both together, in
     * place, in the caller's own storage: the arrays are spans, so
     * they can view any contiguous storage (e.g. an OwnedKDTreeData's
     * vectors, or arrays owned by Python) without copying it. The
     * storage must outlive any tree built over it. With AuxType void,
     * aux_ is an empty placeholder that occupies no storage.
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
        std::span<double> x_;    /**< Positions, N x nDim_, point-major (not owned) */
        [[no_unique_address]] typename KDTreeAuxStorage<AuxType>::View aux_; /**< Auxiliary data, one per point (if AuxType is not void; not owned) */

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
     * @struct OwnedKDTreeData
     * @brief Storage for a set of KDTree points: owning vectors, viewed by a KDTreeData
     * @tparam AuxType As for KDTreeData
     * @details
     * For C++ callers that want the tree's points to own their
     * storage. view() returns a KDTreeData over the vectors; it stays
     * valid until a vector is resized or reallocated (moving this
     * object keeps it valid, since moving a vector keeps its buffer).
     */
    template <typename AuxType = void>
    struct OwnedKDTreeData
    {
        std::size_t nDim_ = 0;   /**< Number of dimensions of each position */
        std::vector<double> x_;  /**< Positions, N x nDim_, point-major */
        [[no_unique_address]] typename KDTreeAuxStorage<AuxType>::Owned aux_; /**< Auxiliary data, one per point (if AuxType is not void) */

        /**
         * @brief A view of these points
         * @return A KDTreeData viewing x_ and aux_
         */
        [[nodiscard]] auto view() -> KDTreeData<AuxType>
        {
            KDTreeData<AuxType> v;
            v.nDim_ = nDim_;
            v.x_ = std::span<double>(x_);
            if constexpr (KDTreeData<AuxType>::hasAux) { v.aux_ = std::span<AuxType>(aux_); }
            return v;
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
         * @param data The points to index; not copied, so both it and
         *   the storage it views must outlive the tree, and the storage
         *   is reordered in place as the tree is built. Its
         *   dimensionality sets the tree's. It must pass its own
         *   validate(): for KDTreeData, every position coordinate must
         *   be finite.
         * @param leafSize Maximum number of points in a leaf; must be
         *   at least 1
         * @param splitBandwidth Metric used to choose split dimensions
         *   while building the tree (see chooseSplitDim()), for nDim
         *   dimensions. Used only during construction and not stored;
         *   searches take their own metric.
         * @param noSplitDims Dimensions along which no node is split
         *   (see chooseSplitDim()); empty (the default) to allow every
         *   dimension. A tree for searches in which some dimensions
         *   will always be unspecified (NaN) can be built this way, so
         *   that its nodes are split only along the dimensions those
         *   searches use.
         * @throws std::invalid_argument if data is inconsistent (see
         *   Data::validate()), leafSize is 0, splitBandwidth is not
         *   for data's number of dimensions, or noSplitDims contains a
         *   dimension not below nDim or contains every dimension
         * @details
         * Sizes the tree, then initializes its root node to cover
         * every point, with their bounding box. Finally
         * builds the rest of the tree by splitting every node, level
         * by level (see splitNode()), in parallel within each level if
         * built with OpenMP -- across a level's nodes, or, for the top
         * few levels, within each large node (see partitionNode() and
         * setNodeBounds()). This reorders data in place; the order
         * produced depends only on the data, not on the number of
         * threads.
         */
        KDTree(Data& data,
            const std::size_t leafSize,
            const Bandwidth& splitBandwidth,
            const std::vector<std::size_t>& noSplitDims = {}) :
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
            if (std::ranges::any_of(noSplitDims, [this](const std::size_t d) -> bool { return d >= nDim_; }))
            {
                throw std::invalid_argument("KDTree: noSplitDims must contain only dimensions below nDim");
            }
            std::vector<bool> noSplit(nDim_, false);
            for (const std::size_t d : noSplitDims) { noSplit[d] = true; }
            if (std::ranges::all_of(noSplit, [](const bool b) -> bool { return b; }))
            {
                throw std::invalid_argument("KDTree: noSplitDims must leave at least one dimension to split along");
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
            // as leaves. A level with fewer nodes than threads (the top
            // few) is instead split one node at a time, so that each
            // split can use every thread (see partitionNode() and
            // setNodeBounds()); which way a level is split does not
            // change the tree built.
            std::size_t first = rootNode;
            std::size_t width = 1;
            for (std::size_t level = 0; level < nLevels_; ++level)
            {
#ifdef _OPENMP
                if (width >= static_cast<std::size_t>(omp_get_max_threads()))
                {
#                   pragma omp parallel for
                    for (std::size_t i = first; i < first + width; ++i)
                    {
                        splitNode(i, splitBandwidth, noSplitDims);
                    }
                }
                else
#endif
                {
                    for (std::size_t i = first; i < first + width; ++i)
                    {
                        splitNode(i, splitBandwidth, noSplitDims);
                    }
                }
                first += width;
                width <<= 1;
            }
        }

        /**
         * @brief Construct a KDTree, choosing split dimensions with the Euclidean metric
         * @param data As for the other constructor
         * @param leafSize As for the other constructor
         * @param noSplitDims As for the other constructor
         * @throws std::invalid_argument as for the other constructor
         */
        explicit KDTree(Data& data, const std::size_t leafSize = 16,
            const std::vector<std::size_t>& noSplitDims = {}) :
            KDTree(data, leafSize, Bandwidth::identity(std::max<std::size_t>(data.nDim_, 1)), noSplitDims)
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
            const Bandwidth bw = sharedReduction(x, bandwidth);
            return batchNeighbors(x, nNeighbor, [this, &bw](const std::vector<double>& xi,
                const std::size_t ki, const std::size_t) -> std::vector<Neighbor>
            {
                return neighbors(xi, ki, bw);
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

        /**
         * @brief Find, in parallel, the nearest neighbors of each of several points measured with errors
         * @param x Positions to search around, one per search
         * @param nNeighbor Number of neighbors to find, one per search
         * @param errDims The dimensions with measurement errors; each
         *   must be specified (not NaN) in every point
         * @param errCov The covariance matrix of each point's errors, in
         *   the dimensions errDims (in that order): one per point, each
         *   errDims.size() square, symmetric, and positive semi-definite
         * @param bandwidth The metric, shared by every search
         * @return One result per search: the neighbors of x[i] in the
         *   metric bandwidth convolved with its errors (see
         *   Bandwidth::convolveCov()), (M^-1 + 2 C_i)^-1, in which a
         *   difference the errors could explain counts for less
         * @throws std::invalid_argument if errCov does not have one
         *   matrix per point, or one has the wrong size, or errDims has
         *   a dimension not below nDim, a repeated one, or one that is
         *   NaN in a point; otherwise, as for the shared-bandwidth
         *   version, and as for Bandwidth::convolveCov()
         * @details
         * Each point's metric is convolved with its own errors in the
         * thread that searches around it. If every point has the same
         * pattern of NaNs, bandwidth is reduced for it once, beforehand,
         * and each convolution done in the reduced space.
         */
        [[nodiscard]] auto neighbors(const std::vector<std::vector<double>>& x,
            const std::vector<std::size_t>& nNeighbor,
            const std::vector<std::size_t>& errDims,
            const std::vector<Eigen::MatrixXd>& errCov, // NOLINT(misc-include-cleaner) -- Eigen::MatrixXd is provided by <Eigen/Dense> (included above); clang-tidy's IWYU mapping doesn't know that
            const Bandwidth& bandwidth) const
            -> std::vector<std::vector<Neighbor>>
        {
            checkErrors(x, errDims, errCov, {}, "KDTree::neighbors");
            const Bandwidth bw = sharedReduction(x, bandwidth);
            return batchNeighbors(x, nNeighbor, [this, &bw, &errDims, &errCov](const std::vector<double>& xi,
                const std::size_t ki, const std::size_t i) -> std::vector<Neighbor>
            {
                return neighbors(xi, ki, bw.convolveCov(embedErrors(errDims, errCov[i]))); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
            });
        }

    protected:

        /**
         * Nodes with at least this many points are partitioned
         * (selectBlocked()) and have their bounding boxes found
         * (setNodeBounds()) in blocks that can be processed in
         * parallel; smaller ones serially
         */
        static constexpr std::size_t blockedMinCount = std::size_t{1} << 16;

        /** Number of points per block, and per chunk of swaps, in the blocked algorithms */
        static constexpr std::size_t blockSize = std::size_t{1} << 14;

        /** Number of coordinates sampled to choose each round's pivots in selectBlocked() */
        static constexpr std::size_t pivotSampleSize = 1024;

        /**
         * @brief A shared bandwidth, reduced once for a batch of points if they all share a pattern of NaNs
         * @param x The points
         * @param bandwidth The bandwidth
         * @return bandwidth reduced for the points' common pattern of
         *   NaNs, if it is not reduced already, is for nDim dimensions,
         *   every point has nDim elements and the same pattern, and that
         *   pattern has a NaN; otherwise bandwidth itself (each query
         *   then reduces it as needed, or reports what is wrong)
         */
        [[nodiscard]] auto sharedReduction(const std::vector<std::vector<double>>& x,
            const Bandwidth& bandwidth) const -> Bandwidth
        {
            if (x.empty() || bandwidth.isReduced() || bandwidth.nDim() != this->nDim_ ||
                x.front().size() != this->nDim_ || bandwidth.matches(x.front()))
            {
                return bandwidth;
            }
            const std::vector<bool> pattern = Bandwidth::nanPattern(x.front());
            const bool shared = std::ranges::all_of(x, [this, &pattern](const std::vector<double>& xi) -> bool
                { return xi.size() == this->nDim_ && Bandwidth::nanPattern(xi) == pattern; });
            return shared ? bandwidth.reduce(pattern) : bandwidth;
        }

        /**
         * @brief Check a batch's measurement errors
         * @param x The points
         * @param errDims The dimensions with errors
         * @param errCov One covariance matrix per point
         * @param marginalDims Dimensions that must not be in errDims
         * @param caller The caller's name, for messages
         * @throws std::invalid_argument if errCov does not have one
         *   matrix per point, or one is not errDims.size() square;
         *   errDims has a dimension not below nDim, a repeated one, or
         *   one in marginalDims; or a dimension in errDims is NaN in a
         *   point (points that do not have nDim elements are left to
         *   the single-point call to report)
         */
        void checkErrors(const std::vector<std::vector<double>>& x,
            const std::vector<std::size_t>& errDims,
            const std::vector<Eigen::MatrixXd>& errCov,
            const std::vector<std::size_t>& marginalDims,
            const char* caller) const
        {
            const std::string name(caller);
            if (errCov.size() != x.size())
            {
                throw std::invalid_argument(name + ": errCov must have one matrix per point");
            }
            const auto k = static_cast<Eigen::Index>(errDims.size()); // NOLINT(misc-include-cleaner) -- Eigen::Index is provided by <Eigen/Dense> (included above); clang-tidy's IWYU mapping doesn't know that
            if (std::ranges::any_of(errCov, [k](const Eigen::MatrixXd& c) -> bool
                { return c.rows() != k || c.cols() != k; }))
            {
                throw std::invalid_argument(name + ": each errCov matrix must be errDims.size() square");
            }
            std::vector<bool> seen(this->nDim_, false);
            for (const std::size_t d : errDims)
            {
                if (d >= this->nDim_ || seen[d]) // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
                {
                    throw std::invalid_argument(name + ": errDims must hold distinct dimensions below nDim");
                }
                if (std::ranges::find(marginalDims, d) != marginalDims.end())
                {
                    throw std::invalid_argument(name + ": errDims must not include a marginalized dimension");
                }
                seen[d] = true; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
            }
            for (const auto& xi : x)
            {
                if (xi.size() != this->nDim_) { continue; }
                if (std::ranges::any_of(errDims, [&xi](const std::size_t d) -> bool
                    { return std::isnan(xi[d]); })) // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
                {
                    throw std::invalid_argument(name + ": every dimension in errDims must be "
                        "specified (not NaN) in every point");
                }
            }
        }

        /**
         * @brief A covariance matrix over some dimensions, embedded in all nDim
         * @param errDims The dimensions it covers
         * @param cov The matrix, errDims.size() square
         * @return The nDim x nDim matrix with cov in the rows and columns
         *   errDims, and zero elsewhere
         */
        [[nodiscard]] auto embedErrors(const std::vector<std::size_t>& errDims,
            const Eigen::MatrixXd& cov) const -> Eigen::MatrixXd
        {
            const auto n = static_cast<Eigen::Index>(this->nDim_);
            Eigen::MatrixXd full = Eigen::MatrixXd::Zero(n, n);
            for (std::size_t a = 0; a < errDims.size(); ++a)
            {
                for (std::size_t b = 0; b < errDims.size(); ++b)
                {
                    full(static_cast<Eigen::Index>(errDims[a]), static_cast<Eigen::Index>(errDims[b])) = // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
                        cov(static_cast<Eigen::Index>(a), static_cast<Eigen::Index>(b));
                }
            }
            return full;
        }

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
         * @param noSplitDims Dimensions not to split along; must leave
         *   at least one
         * @return The dimension to split along, in [0, nDim) and not in
         *   noSplitDims
         * @details
         * Splits along the dimension in which the node is widest,
         * measured in the tree's metric: the one maximizing
         * (hi[i] - lo[i])^2 M_ii, the squared metric length of
         * the node's extent along axis i (for a diagonal bandwidth h_i,
         * its extent in units of the bandwidth, squared), or just
         * (hi[i] - lo[i])^2 for the Euclidean metric, among the
         * dimensions not in noSplitDims. Ties go to the lowest
         * dimension.
         */
        [[nodiscard]] auto chooseSplitDim(const std::size_t idx,
            const Bandwidth& bandwidth,
            const std::vector<std::size_t>& noSplitDims) const -> int
        {
            const auto lo = nodeLo(idx);
            const auto hi = nodeHi(idx);
            std::size_t best = nDim_;
            double bestSize = -1.0;
            for (std::size_t i = 0; i < nDim_; ++i)
            {
                if (std::ranges::find(noSplitDims, i) != noSplitDims.end()) { continue; }
                const double width = hi[i] - lo[i]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
                double size = width * width;
                size *= bandwidth.h2Inv()(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(i)); // NOLINT(misc-include-cleaner) -- Eigen::Index is provided by <Eigen/Dense> (included above); clang-tidy's IWYU mapping doesn't know that
                if (size > bestSize)
                {
                    best = i;
                    bestSize = size;
                }
            }
            assert(best < nDim_);
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
         * parallel. A node with at least blockedMinCount points is
         * scanned in blocks of blockSize, in parallel if built with
         * OpenMP (and not already inside a parallel region), and the
         * blocks' extents then combined; the result is the same either
         * way.
         */
        void setNodeBounds(const std::size_t idx)
        {
            const auto& node = nodes_[idx];
            const auto offset = static_cast<std::ptrdiff_t>(idx * nDim_);
            const auto len = static_cast<std::ptrdiff_t>(nDim_);
            const std::span<double> lo(nodeLo_.begin() + offset, nodeLo_.begin() + offset + len);
            const std::span<double> hi(nodeHi_.begin() + offset, nodeHi_.begin() + offset + len);
            if (node.count_ < blockedMinCount)
            {
                rangeBounds(node.start_, node.start_ + node.count_, lo, hi);
                return;
            }

            // Each block's extent, then their combination
            const std::size_t nBlocks = (node.count_ + blockSize - 1) / blockSize;
            std::vector<double> blockLo(nBlocks * nDim_);
            std::vector<double> blockHi(nBlocks * nDim_);
#ifdef _OPENMP
#           pragma omp parallel for if(!omp_in_parallel())
#endif
            for (std::size_t b = 0; b < nBlocks; ++b)
            {
                const std::size_t start = node.start_ + (b * blockSize);
                const std::size_t end = std::min(start + blockSize, node.start_ + node.count_);
                rangeBounds(start, end, std::span<double>(blockLo).subspan(b * nDim_, nDim_),
                    std::span<double>(blockHi).subspan(b * nDim_, nDim_));
            }
            std::ranges::fill(lo, std::numeric_limits<double>::infinity());
            std::ranges::fill(hi, -std::numeric_limits<double>::infinity());
            for (std::size_t b = 0; b < nBlocks; ++b)
            {
                for (std::size_t d = 0; d < nDim_; ++d)
                {
                    lo[d] = std::min(lo[d], blockLo[(b * nDim_) + d]); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
                    hi[d] = std::max(hi[d], blockHi[(b * nDim_) + d]); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
                }
            }
        }

        /**
         * @brief Find the extent of a contiguous range of points
         * @param start Index of the first point
         * @param end One past the index of the last point
         * @param lo Set to the minimum of each coordinate (+inf if the range is empty)
         * @param hi Set to the maximum of each coordinate (-inf if the range is empty)
         */
        void rangeBounds(const std::size_t start, const std::size_t end,
            const std::span<double> lo, const std::span<double> hi) const
        {
            const auto& pts = data_.get();
            std::ranges::fill(lo, std::numeric_limits<double>::infinity());
            std::ranges::fill(hi, -std::numeric_limits<double>::infinity());
            for (std::size_t i = start; i < end; ++i)
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
         * @brief A point's coordinate along one dimension
         * @param i Index of the point
         * @param dim The dimension
         * @return The coordinate
         */
        [[nodiscard]] auto coord(const std::size_t i, const std::size_t dim) const -> double
        {
            return data_.get().x_[(i * nDim_) + dim]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
        }

        /**
         * @brief Partially sort a node's points about their median along its split dimension
         * @param idx Index of the node; its splitDim_ must already be set
         * @details
         * Reorders the node's points (an in-place selection) so that,
         * with m = start_ + (count_ - 1) / 2, the point at index m is
         * the one that would be there if the node's points were fully
         * sorted by their coordinate along splitDim_, every point
         * before it has a coordinate no larger, and every point after
         * it one no smaller. Points are swapped whole, with
         * Data::swap(), so auxiliary data and any per-point arrays a
         * derived Data adds move with their positions. A node with at
         * least blockedMinCount points is narrowed down with
         * selectBlocked(), which can run in parallel; the rest of the
         * work is done by selectSerial().
         */
        void partitionNode(const std::size_t idx)
        {
            const auto& node = nodes_[idx];
            const auto dim = static_cast<std::size_t>(node.splitDim_);
            const std::size_t m = node.start_ + ((node.count_ - 1) >> 1);
            if (node.count_ >= blockedMinCount)
            {
                selectBlocked(node.start_, node.start_ + node.count_, m, dim);
            }
            else
            {
                selectSerial(node.start_, node.start_ + node.count_ - 1, m, dim);
            }
        }

        /**
         * @brief Select, serially, the point of a given rank in a range
         * @param first Index of the first point of the range
         * @param last Index of the last point of the range (inclusive)
         * @param m Index, in [first, last], at which to put the point of
         *   rank m - first along dim
         * @param dim The dimension to order by
         * @details
         * An in-place quickselect (following slug2's partition_node):
         * afterwards, the points in [first, m) have coordinates no
         * larger than point m's, and those in (m, last] no smaller.
         */
        void selectSerial(const std::size_t first, const std::size_t last, const std::size_t m,
            const std::size_t dim)
        {
            auto& pts = data_.get();
            auto key = [this, dim](const std::size_t i) -> double { return coord(i, dim); };
            std::size_t l = first;
            std::size_t r = last;

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
         * @brief Select the point of a given rank in a large range, with passes that can run in parallel
         * @param first Index of the first point of the range
         * @param end One past the index of the last point of the range
         * @param m Index, in [first, end), at which to put the point of
         *   rank m - first along dim
         * @param dim The dimension to order by
         * @details
         * The same result as selectSerial() (though possibly a
         * different order of the points on either side of m). Each
         * round sorts an evenly spaced sample of the range's
         * coordinates and takes two of them, a little below and a
         * little above m's relative rank, as pivots; two
         * partitionBlocked() passes then narrow the range to the
         * points between the pivots, which contain m unless the
         * sample misjudged its rank (in which case the round still
         * narrows the range, to one side of a pivot). Rounds continue
         * until the range has fewer than blockedMinCount points, or
         * consists of equal coordinates, or a round fails to narrow
         * it; selectSerial() finishes the job. The order of points
         * this produces depends only on the data, not on the number
         * of threads.
         */
        void selectBlocked(const std::size_t first, const std::size_t end, const std::size_t m,
            const std::size_t dim)
        {
            std::size_t l = first;
            std::size_t r = end;
            std::vector<double> sample;
            while (r - l >= blockedMinCount)
            {
                // Pivots bracketing m's rank, from a sorted sample; the
                // sample's ranks scatter by about sqrt(nSample) / 2, so
                // a gap of twice sqrt(nSample) brackets m's rank with
                // high probability
                const std::size_t n = r - l;
                const std::size_t nSample = std::min(pivotSampleSize, n);
                sample.resize(nSample);
                for (std::size_t k = 0; k < nSample; ++k)
                {
                    sample[k] = coord(l + ((k * n) / nSample), dim); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
                }
                std::ranges::sort(sample);
                const auto gap = static_cast<std::size_t>(2.0 * std::sqrt(static_cast<double>(nSample)));
                const std::size_t rank = ((m - l) * nSample) / n;
                const double pivotLo = sample[rank >= gap ? rank - gap : 0]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
                const double pivotHi = sample[std::min(rank + gap, nSample - 1)]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)

                // Points below pivotLo first, then those up to pivotHi;
                // keep whichever part holds m
                const std::size_t oldL = l;
                const std::size_t oldR = r;
                const std::size_t below = partitionBlocked(l, r,
                    [this, dim, pivotLo](const std::size_t i) -> bool { return coord(i, dim) < pivotLo; });
                if (m < below)
                {
                    r = below;
                }
                else
                {
                    l = below;
                    const std::size_t upTo = partitionBlocked(l, r,
                        [this, dim, pivotHi](const std::size_t i) -> bool { return coord(i, dim) <= pivotHi; });
                    if (m >= upTo)
                    {
                        l = upTo;
                    }
                    else
                    {
                        r = upTo;
                        // Every point left has coordinate pivotLo, so
                        // any order is sorted
                        if (pivotLo == pivotHi) { return; }
                    }
                }
                if (l == oldL && r == oldR) { break; }
            }
            selectSerial(l, r - 1, m, dim);
        }

        /**
         * @brief Partition, serially, a range of points by a predicate
         * @tparam Pred Callable taking a point index and returning bool
         * @param first Index of the first point of the range
         * @param end One past the index of the last point of the range
         * @param pred The predicate
         * @return The index of the first point for which pred is
         *   false, after reordering the range so that pred holds for
         *   the points before it and not for those from it on
         */
        template <typename Pred>
        auto partitionSerial(const std::size_t first, const std::size_t end, const Pred& pred) -> std::size_t
        {
            auto& pts = data_.get();
            std::size_t i = first;
            std::size_t j = end;
            while (true)
            {
                while (i < j && pred(i)) { ++i; }
                while (i < j && !pred(j - 1)) { --j; }
                if (i >= j) { return i; }
                pts.swap(i, j - 1);
                ++i;
                --j;
            }
        }

        /**
         * @brief Partition a range of points by a predicate, in blocks that can be processed in parallel
         * @tparam Pred Callable taking a point index and returning bool;
         *   called concurrently, so must be safe to call from several
         *   threads at once
         * @param first Index of the first point of the range
         * @param end One past the index of the last point of the range
         * @param pred The predicate
         * @return As for partitionSerial()
         * @details
         * Each block of blockSize points is partitioned on its own
         * (partitionSerial()). With mid the returned index, the points
         * then out of place -- those failing pred before mid, and those
         * passing it from mid on -- are equal in number, and lie in at
         * most one interval per block on each side; pairing the k'th
         * of one side with the k'th of the other, the swaps that put
         * them in place touch distinct points, so are made in parallel,
         * in chunks of blockSize. Both stages run in parallel if built
         * with OpenMP (and not already inside a parallel region); the
         * blocks, and so the order of points produced, depend only on
         * the range's size, not on the number of threads.
         */
        template <typename Pred>
        auto partitionBlocked(const std::size_t first, const std::size_t end, const Pred& pred) -> std::size_t
        {
            const std::size_t nBlocks = (end - first + blockSize - 1) / blockSize;
            std::vector<std::size_t> split(nBlocks);
#ifdef _OPENMP
#           pragma omp parallel for if(!omp_in_parallel())
#endif
            for (std::size_t b = 0; b < nBlocks; ++b)
            {
                const std::size_t start = first + (b * blockSize);
                split[b] = partitionSerial(start, std::min(start + blockSize, end), pred); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
            }
            std::size_t mid = first;
            for (std::size_t b = 0; b < nBlocks; ++b)
            {
                mid += split[b] - (first + (b * blockSize)); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
            }

            // The out-of-place points, as runs [start, start + length)
            // in increasing order, with the number in earlier runs
            struct Run
            {
                std::size_t start_;   /**< First point of the run */
                std::size_t length_;  /**< Number of points in it */
                std::size_t before_;  /**< Number of out-of-place points in earlier runs */
            };
            std::vector<Run> low;   // failing pred, before mid
            std::vector<Run> high;  // passing pred, from mid on
            std::size_t nLow = 0;
            std::size_t nHigh = 0;
            for (std::size_t b = 0; b < nBlocks; ++b)
            {
                const std::size_t start = first + (b * blockSize);
                const std::size_t stop = std::min(start + blockSize, end);
                const std::size_t s = split[b]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
                if (s < mid && stop > s)
                {
                    const std::size_t length = std::min(stop, mid) - s;
                    low.push_back({.start_ = s, .length_ = length, .before_ = nLow});
                    nLow += length;
                }
                if (s > mid && s > start)
                {
                    const std::size_t runStart = std::max(start, mid);
                    high.push_back({.start_ = runStart, .length_ = s - runStart, .before_ = nHigh});
                    nHigh += s - runStart;
                }
            }
            assert(nLow == nHigh);

            // Swap the k'th out-of-place point on each side, for every k
            auto& pts = data_.get();
            const std::size_t nChunks = (nLow + blockSize - 1) / blockSize;
#ifdef _OPENMP
#           pragma omp parallel for if(!omp_in_parallel())
#endif
            for (std::size_t c = 0; c < nChunks; ++c)
            {
                const std::size_t k0 = c * blockSize;
                const std::size_t k1 = std::min(k0 + blockSize, nLow);
                auto runOf = [k0](const std::vector<Run>& runs) -> std::size_t
                {
                    const auto it = std::upper_bound(runs.begin(), runs.end(), k0,
                        [](const std::size_t k, const Run& run) -> bool { return k < run.before_; });
                    return static_cast<std::size_t>(it - runs.begin()) - 1;
                };
                std::size_t a = runOf(low);
                std::size_t h = runOf(high);
                std::size_t offA = k0 - low[a].before_; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
                std::size_t offH = k0 - high[h].before_; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
                for (std::size_t k = k0; k < k1; ++k)
                {
                    while (offA == low[a].length_) { ++a; offA = 0; } // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
                    while (offH == high[h].length_) { ++h; offH = 0; } // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
                    pts.swap(low[a].start_ + offA, high[h].start_ + offH); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
                    ++offA;
                    ++offH;
                }
            }
            return mid;
        }

        /**
         * @brief Split a node into two children, or mark it as a leaf
         * @param idx Index of the node; its start_, count_, and
         *   bounding box (see setNodeBounds()) must already be set
         * @param bandwidth The metric used to choose the split dimension
         * @param noSplitDims Dimensions not to split along
         * @details
         * A node with at most leafSize_ points becomes a leaf
         * (splitDim_ = -1, no children). Otherwise it is split along
         * chooseSplitDim(idx, bandwidth, noSplitDims): its points are partitioned about their
         * median along that dimension (see partitionNode()), the left
         * child gets the first (count_ + 1) / 2 of them and the right
         * child the rest, and each child's bounding box is computed
         * from its own points.
         */
        void splitNode(const std::size_t idx, const Bandwidth& bandwidth,
            const std::vector<std::size_t>& noSplitDims)
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
            node.splitDim_ = chooseSplitDim(idx, bandwidth, noSplitDims);
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
