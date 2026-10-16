/**
 * @file testKernelDensity.hpp
 * @author Mark Krumholz
 * @brief Unit tests for kden::KernelDensity and kden::KernelDensityData
 * @details
 * Checks that both constructors build the same tree a KDTree would
 * over the same positions, that the weights are reordered along with
 * the points, every node's weight sums and center of mass against
 * brute force, and the weights' argument checks.
 * @date 2026-10-16
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#ifndef TESTKERNELDENSITY_HPP
#define TESTKERNELDENSITY_HPP

#include "../../src/kden/Bandwidth.hpp"
#include "../../src/kden/KernelDensity.hpp"
#include "testKDTree.hpp"
#include <Eigen/Dense> // NOLINT(misc-include-cleaner)
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

/** Weighted test data: each point's auxiliary data is its original index */
using KDETestData = kden::KernelDensityData<std::size_t>;
/** A kernel density estimate over KDETestData */
using KDETest = kden::KernelDensity<std::size_t>;

/**
 * @brief A test weight, as a function of a point's original index
 * @param i The original index
 * @return A weight in [0.5, 2)
 */
inline auto kdeTestWeight(const std::size_t i) -> double
{
    return 0.5 + (1.5 * std::fmod(0.6180339887 * static_cast<double>(i), 1.0));
}

/**
 * @brief Make weighted test data: makeKDTestData's points, with kdeTestWeight() weights
 * @param nPt Number of points
 * @param nDim Number of dimensions
 * @param nCorr Number of correlated dimensions
 * @param seed Random seed
 * @return The weighted points
 */
inline auto makeKDETestData(const std::size_t nPt, const std::size_t nDim,
    const std::size_t nCorr, const unsigned seed) -> KDETestData
{
    KDETestData data;
    static_cast<KDTestData&>(data) = makeKDTestData(nPt, nDim, nCorr, seed);
    data.w_.resize(nPt);
    for (std::size_t i = 0; i < nPt; ++i) { data.w_[i] = kdeTestWeight(i); }
    return data;
}

/**
 * @brief Check every node's weight sums and center of mass against brute force
 * @param kde The kernel density estimate
 * @return Number of discrepancies
 */
inline auto checkNodeWeights(const KDETest& kde) -> int
{
    const auto& data = kde.data();
    const std::size_t nd = data.nDim_;
    int bad = 0;
    std::size_t idx = kden::rootNode;
    while (idx != kden::nullNode)
    {
        const auto& node = kde.nodes()[idx];
        double w = 0.0;
        double w2 = 0.0;
        std::vector<double> cm(nd, 0.0);
        for (std::size_t i = node.start_; i < node.start_ + node.count_; ++i)
        {
            const auto pos = data.position(i);
            w += data.weight(i);
            w2 += data.weight(i) * data.weight(i);
            for (std::size_t k = 0; k < nd; ++k) { cm[k] += data.weight(i) * pos[k]; }
        }
        const auto nodeCm = kde.nodeCm(idx);
        for (std::size_t k = 0; k < nd; ++k)
        {
            if (std::abs((cm[k] / w) - nodeCm[k]) > 1e-10 * (1.0 + std::abs(nodeCm[k]))) { ++bad; }
        }
        if (std::abs(w - kde.nodeW(idx)) > 1e-10 * w || std::abs(w2 - kde.nodeW2(idx)) > 1e-10 * w2) { ++bad; }
        idx = node.splitDim_ < 0 ? KDETest::next(idx) : KDETest::left(idx);
    }
    return bad;
}

/**
 * @brief Check KernelDensity construction: the tree, the reordered weights, and the node weights
 * @return Number of failures
 */
inline auto testKernelDensityConstruction() -> int
{
    int failures = 0;
    Eigen::MatrixXd m = Eigen::MatrixXd::Identity(4, 4);
    m(0, 1) = m(1, 0) = 0.5;
    const kden::Bandwidth splitBw(m);

    for (const std::size_t nPt : {1UL, 16UL, 17UL, 100UL, 5000UL})
    {
        for (const bool withMetric : {false, true})
        {
            const std::string what = "nPt = " + std::to_string(nPt) +
                (withMetric ? " with" : " without") + " a metric";
            KDETestData kdeData = makeKDETestData(nPt, 4, 2, 21);
            KDTestData treeData = makeKDTestData(nPt, 4, 2, 21);
            const KDETest kde = withMetric ? KDETest(kdeData, 8, splitBw) : KDETest(kdeData, 8);
            const KDTestTree tree = withMetric ? KDTestTree(treeData, 8, splitBw) : KDTestTree(treeData, 8);

            // The same tree as a KDTree builds over the same positions
            if (kdeData.aux_ != treeData.aux_ || kdeData.x_ != treeData.x_ ||
                kde.nLeaves() != tree.nLeaves() || checkKDTreeInvariants(kde, nPt) != 0)
            {
                std::cerr << "testKernelDensityConstruction: " << what
                    << ": tree differs from the equivalent KDTree\n";
                ++failures;
            }

            // Each weight moved with its point
            bool moved = true;
            for (std::size_t i = 0; i < nPt; ++i) { moved = moved && kdeData.w_[i] == kdeTestWeight(kdeData.aux_[i]); }
            if (!moved)
            {
                std::cerr << "testKernelDensityConstruction: " << what
                    << ": weights were not reordered with their points\n";
                ++failures;
            }

            const int bad = checkNodeWeights(kde);
            if (bad != 0)
            {
                std::cerr << "testKernelDensityConstruction: " << what << ": " << bad
                    << " node weight sums or centers of mass are wrong\n";
                ++failures;
            }
        }
    }

    // Invalid weights: wrong number, and non-finite
    int thrown = 0;
    const KDETestData data = makeKDETestData(10, 3, 2, 1);
    auto badW = data;
    badW.w_.pop_back();
    try { const KDETest t(badW); } catch (const std::invalid_argument&) { ++thrown; }
    auto nanW = data;
    nanW.w_[2] = std::numeric_limits<double>::quiet_NaN();
    try { const KDETest t(nanW); } catch (const std::invalid_argument&) { ++thrown; }
    auto infW = data;
    infW.w_[5] = -std::numeric_limits<double>::infinity();
    try { const KDETest t(infW); } catch (const std::invalid_argument&) { ++thrown; }
    if (thrown != 3)
    {
        std::cerr << "testKernelDensityConstruction: only " << thrown << " of 3 invalid constructions threw\n";
        ++failures;
    }
    return failures;
}

#endif // TESTKERNELDENSITY_HPP
