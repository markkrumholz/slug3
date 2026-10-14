/**
 * @file KDTree.hpp
 * @author Mark Krumholz
 * @brief A balanced kd tree of weighted points
 * @date 2026-10-14
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#ifndef KDTREE_HPP
#define KDTREE_HPP

#include <cstddef>
#include <limits>
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

} // namespace kden

#endif // KDTREE_HPP
