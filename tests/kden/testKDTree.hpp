/**
 * @file testKDTree.hpp
 * @author Mark Krumholz
 * @brief Unit tests for kden::KDTree
 * @details
 * Covers the traversal helpers, tree construction (every node's
 * statistics and the split invariants), and the nearest-neighbor
 * search, single-point and batched, against brute force for
 * Euclidean, diagonal, strongly correlated, and partially-unspecified
 * metrics. (Bandwidth itself is tested in testBandwidth.hpp.)
 * @date 2026-10-15
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#ifndef TESTKDTREE_HPP
#define TESTKDTREE_HPP

#include "../../src/kden/Bandwidth.hpp"
#include "../../src/kden/KDTree.hpp"
#include <Eigen/Dense> // NOLINT(misc-include-cleaner)
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

/** A test point: position, weight, and its index in the original data */
struct KDTestPoint
{
    std::vector<double> x_;    /**< Position */
    double w_ = 1.0;           /**< Weight */
    std::size_t origIdx_ = 0;  /**< Index in the original (unsorted) data */
};

using KDTestTree = kden::KDTree<KDTestPoint>;

/**
 * @brief Make a data set with strongly correlated dimensions
 * @param nPt Number of points
 * @param nDim Number of dimensions; the first nCorr share a common
 *   component, so are strongly correlated, and the rest are independent
 * @param nCorr Number of correlated dimensions
 * @param seed Random seed
 * @param coarse If true, round coordinates to integers, to create many
 *   duplicate coordinate values
 * @return The points, with origIdx_ set to their index
 */
inline auto makeKDTestData(const std::size_t nPt, const std::size_t nDim,
    const std::size_t nCorr, const unsigned seed, const bool coarse = false)
    -> std::vector<KDTestPoint>
{
    std::mt19937 gen(seed);
    std::normal_distribution<double> gauss(0.0, 1.0);
    std::uniform_real_distribution<double> unif(0.5, 2.0);
    std::vector<KDTestPoint> pts(nPt);
    for (std::size_t i = 0; i < nPt; ++i)
    {
        const double common = 3.0 * gauss(gen);
        pts[i].x_.resize(nDim);
        for (std::size_t d = 0; d < nDim; ++d)
        {
            double x = d < nCorr ? common + (0.2 * gauss(gen)) : gauss(gen);
            if (coarse) { x = std::round(x); }
            pts[i].x_[d] = x;
        }
        pts[i].w_ = unif(gen);
        pts[i].origIdx_ = i;
    }
    return pts;
}

/**
 * @brief A Silverman-like bandwidth: the inverse of the scaled data covariance
 * @param pts The data
 * @param nDim Number of dimensions
 * @return A Bandwidth holding (s^2 Cov)^-1, with s^2 = N^(-2 / (nDim + 4))
 */
inline auto silvermanBandwidth(const std::vector<KDTestPoint>& pts, const std::size_t nDim)
    -> kden::Bandwidth
{
    const auto n = static_cast<Eigen::Index>(nDim);
    Eigen::VectorXd mean = Eigen::VectorXd::Zero(n);
    for (const auto& p : pts) { mean += Eigen::Map<const Eigen::VectorXd>(p.x_.data(), n); }
    mean /= static_cast<double>(pts.size());
    Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(n, n);
    for (const auto& p : pts)
    {
        const Eigen::VectorXd dx = Eigen::Map<const Eigen::VectorXd>(p.x_.data(), n) - mean;
        cov += dx * dx.transpose();
    }
    cov /= static_cast<double>(pts.size());
    const double s2 = std::pow(static_cast<double>(pts.size()), -2.0 / static_cast<double>(nDim + 4));
    Eigen::MatrixXd m = (s2 * cov).inverse();
    return kden::Bandwidth(0.5 * (m + m.transpose()));
}

/**
 * @brief Check the traversal helpers on a 3-level tree
 * @return Number of failures
 */
inline auto testKDTreeTraversal() -> int
{
    // Tree indices: 0; 1 2; 3 4 5 6
    static_assert(KDTestTree::left(0) == 1 && KDTestTree::right(0) == 2);
    static_assert(KDTestTree::left(2) == 5 && KDTestTree::right(2) == 6);
    static_assert(KDTestTree::parent(0) == kden::nullNode && KDTestTree::parent(4) == 1 &&
                  KDTestTree::parent(6) == 2);
    static_assert(KDTestTree::sibling(0) == kden::nullNode && KDTestTree::sibling(3) == 4 &&
                  KDTestTree::sibling(4) == 3);
    static_assert(KDTestTree::next(3) == 4 && KDTestTree::next(4) == 2 && KDTestTree::next(1) == 2 &&
                  KDTestTree::next(6) == kden::nullNode && KDTestTree::next(0) == kden::nullNode);
    return 0;
}

/**
 * @brief Check every invariant of a built tree
 * @param tree The tree
 * @param nPt Number of points it was built from
 * @return Number of violations
 */
inline auto checkKDTreeInvariants(const KDTestTree& tree, const std::size_t nPt) -> int
{
    int bad = 0;
    const auto& pts = tree.data();
    const auto& nodes = tree.nodes();
    const std::size_t nd = tree.nDim();
    std::size_t covered = 0;
    std::size_t idx = kden::rootNode;
    while (idx != kden::nullNode)
    {
        const auto& node = nodes[idx];
        // Statistics, recomputed by brute force
        double w = 0.0;
        double w2 = 0.0;
        std::vector<double> lo(nd, std::numeric_limits<double>::infinity());
        std::vector<double> hi(nd, -std::numeric_limits<double>::infinity());
        std::vector<double> cm(nd, 0.0);
        for (std::size_t i = node.start_; i < node.start_ + node.count_; ++i)
        {
            w += pts[i].w_;
            w2 += pts[i].w_ * pts[i].w_;
            for (std::size_t k = 0; k < nd; ++k)
            {
                lo[k] = std::min(lo[k], pts[i].x_[k]);
                hi[k] = std::max(hi[k], pts[i].x_[k]);
                cm[k] += pts[i].w_ * pts[i].x_[k];
            }
        }
        for (std::size_t k = 0; k < nd; ++k)
        {
            if (lo[k] != node.lo_[k] || hi[k] != node.hi_[k] ||
                std::abs((cm[k] / w) - node.cm_[k]) > 1e-10 * (1.0 + std::abs(node.cm_[k]))) { ++bad; }
        }
        if (std::abs(w - node.w_) > 1e-10 * w || std::abs(w2 - node.w2_) > 1e-10 * w2) { ++bad; }

        if (node.splitDim_ < 0)
        {
            // Leaf: no children, at most leafSize points, contiguous with the previous leaf
            if (node.count_ > tree.leafSize() || node.start_ != covered ||
                node.lchild_ != kden::nullNode || node.rchild_ != kden::nullNode) { ++bad; }
            covered += node.count_;
            idx = KDTestTree::next(idx);
            continue;
        }

        // Interior node: children linked, split, and ordered along splitDim
        const auto& l = nodes[node.lchild_];
        const auto& r = nodes[node.rchild_];
        if (node.lchild_ != KDTestTree::left(idx) || node.rchild_ != KDTestTree::right(idx) ||
            l.parent_ != idx || r.parent_ != idx || l.start_ != node.start_ ||
            l.count_ != (node.count_ + 1) / 2 || r.start_ != l.start_ + l.count_ ||
            l.count_ + r.count_ != node.count_) { ++bad; }
        const auto d = static_cast<std::size_t>(node.splitDim_);
        double lmax = -std::numeric_limits<double>::infinity();
        double rmin = std::numeric_limits<double>::infinity();
        for (std::size_t i = l.start_; i < l.start_ + l.count_; ++i) { lmax = std::max(lmax, pts[i].x_[d]); }
        for (std::size_t i = r.start_; i < r.start_ + r.count_; ++i) { rmin = std::min(rmin, pts[i].x_[d]); }
        if (lmax > rmin) { ++bad; }
        idx = KDTestTree::left(idx);
    }
    if (covered != nPt) { ++bad; }

    // The points are still a permutation of the original data
    std::vector<int> seen(nPt, 0);
    for (const auto& p : pts) { seen[p.origIdx_]++; }
    if (std::ranges::any_of(seen, [](const int s) -> bool { return s != 1; })) { ++bad; }
    return bad;
}

/**
 * @brief Check tree construction, with and without a metric, and its argument checks
 * @return Number of failures
 */
inline auto testKDTreeConstruction() -> int
{
    int failures = 0;
    for (const std::size_t nPt : {1UL, 16UL, 17UL, 100UL, 5000UL})
    {
        for (const bool withMetric : {false, true})
        {
            auto pts = makeKDTestData(nPt, 4, 2, 7, true);
            // A fixed, strongly correlated metric (a data-derived one is
            // singular for the tiniest data sets)
            const Eigen::MatrixXd corr = (0.2 * Eigen::MatrixXd::Identity(4, 4)) +
                (0.8 * Eigen::MatrixXd::Ones(4, 4));
            const auto tree = withMetric ?
                std::make_unique<KDTestTree>(pts, 4, 16, kden::Bandwidth(corr.inverse())) :
                std::make_unique<KDTestTree>(pts, 4, 16);
            const int bad = checkKDTreeInvariants(*tree, nPt);
            if (bad != 0)
            {
                std::cerr << "testKDTreeConstruction: " << bad << " violations for nPt = "
                    << nPt << (withMetric ? " with" : " without") << " a metric\n";
                ++failures;
            }
        }
    }

    // Invalid arguments
    auto pts = makeKDTestData(10, 3, 2, 1);
    int thrown = 0;
    try { const KDTestTree t(pts, 0); } catch (const std::invalid_argument&) { ++thrown; }
    try { const KDTestTree t(pts, 2); } catch (const std::invalid_argument&) { ++thrown; }
    try { const KDTestTree t(pts, 3, 0); } catch (const std::invalid_argument&) { ++thrown; }
    try { const KDTestTree t(pts, 3, 16, kden::Bandwidth::identity(2)); } catch (const std::invalid_argument&) { ++thrown; }
    if (thrown != 4)
    {
        std::cerr << "testKDTreeConstruction: only " << thrown << " of 4 invalid constructions threw\n";
        ++failures;
    }
    return failures;
}

/**
 * @brief Compare one neighbor search with brute force
 * @param tree The tree
 * @param x The search point
 * @param k Number of neighbors requested
 * @param bandwidth The metric (reduced here for x's NaNs if needed, as the search does)
 * @param what Label to print on failure
 * @return 0 if the search returns exactly the brute-force neighbors, 1 otherwise
 */
inline auto checkNeighbors(const KDTestTree& tree, const std::vector<double>& x,
    const std::size_t k, const kden::Bandwidth& bandwidth, const std::string& what) -> int
{
    const kden::Bandwidth effective = bandwidth.matches(x) ?
        bandwidth : bandwidth.reduce(kden::Bandwidth::nanPattern(x));
    const auto& pts = tree.data();
    std::vector<double> byOrig(pts.size());
    for (const auto& p : pts) { byOrig[p.origIdx_] = effective.dist2(x, p.x_); }
    std::vector<double> sorted = byOrig;
    std::ranges::sort(sorted);

    const auto result = tree.neighbors(x, k, bandwidth);
    const std::size_t expect = std::min(k, pts.size());
    bool ok = result.size() == expect;
    for (std::size_t j = 0; ok && j < expect; ++j)
    {
        // Distances match brute force, in order, and each is the true
        // distance to the point returned (ties may return either point)
        ok = result[j].second == sorted[j] && byOrig[result[j].first.origIdx_] == result[j].second;
    }
    if (!ok)
    {
        std::cerr << "testKDTreeNeighbors: " << what << ": k = " << k
            << " search disagrees with brute force\n";
        return 1;
    }
    return 0;
}

/**
 * @brief Copy a point with one coordinate made NaN
 * @param x The point
 * @param dim The coordinate to make NaN
 * @return The copy
 */
inline auto x0WithNan(std::vector<double> x, const std::size_t dim) -> std::vector<double>
{
    x[dim] = std::numeric_limits<double>::quiet_NaN();
    return x;
}

/**
 * @brief Check the single-point neighbor search against brute force
 * @return Number of failures
 */
inline auto testKDTreeNeighbors() -> int
{
    int failures = 0;
    const std::size_t nDim = 6;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    auto pts = makeKDTestData(4000, nDim, 4, 3);
    const kden::Bandwidth full = silvermanBandwidth(pts, nDim);
    const kden::Bandwidth diag(Eigen::MatrixXd(full.h2Inv().diagonal().asDiagonal()));
    const kden::Bandwidth euclid = kden::Bandwidth::identity(nDim);
    const KDTestTree tree(pts, nDim, 8, full);

    // Partially specified searches leave dimensions 1 and 4 unspecified;
    // both the pre-reduced bandwidth and automatic reduction are tested
    std::vector<bool> unspec(nDim, false);
    unspec[1] = true;
    unspec[4] = true;
    const kden::Bandwidth reduced = full.reduce(unspec);

    std::mt19937 gen(9); // NOLINT(cert-msc32-c,cert-msc51-cpp) -- fixed seed for a reproducible test
    std::normal_distribution<double> gauss(0.0, 1.0);
    for (int q = 0; q < 25; ++q)
    {
        // Search points near (but not at) data points, and some far away
        std::vector<double> x = pts[static_cast<std::size_t>(q * 157) % pts.size()].x_;
        for (auto& v : x) { v += (q % 5 == 0 ? 5.0 : 0.05) * gauss(gen); }
        std::vector<double> xPart = x;
        xPart[1] = nan;
        xPart[4] = nan;
        for (const std::size_t k : {1UL, 7UL, 50UL})
        {
            failures += checkNeighbors(tree, x, k, euclid, "Euclidean");
            failures += checkNeighbors(tree, x, k, diag, "diagonal metric");
            failures += checkNeighbors(tree, x, k, full, "full correlated metric");
            failures += checkNeighbors(tree, xPart, k, reduced, "partially specified, pre-reduced");
            failures += checkNeighbors(tree, xPart, k, full, "partially specified, reduced automatically");
        }
    }

    // The Euclidean overload matches an explicit identity bandwidth
    const auto viaDefault = tree.neighbors(pts[3].x_, 5);
    const auto viaIdentity = tree.neighbors(pts[3].x_, 5, euclid);
    bool sameEuclid = viaDefault.size() == viaIdentity.size();
    for (std::size_t j = 0; sameEuclid && j < viaDefault.size(); ++j)
    {
        sameEuclid = viaDefault[j].second == viaIdentity[j].second;
    }
    if (!sameEuclid)
    {
        std::cerr << "testKDTreeNeighbors: default (Euclidean) search differs from identity bandwidth\n";
        ++failures;
    }

    // Edge cases: more neighbors than points, none, and duplicate points
    auto few = makeKDTestData(5, 3, 2, 4);
    const KDTestTree small(few, 3, 2);
    failures += checkNeighbors(small, {0.0, 0.0, 0.0}, 10, kden::Bandwidth::identity(3), "nNeighbor > number of points");
    if (!small.neighbors({0.0, 0.0, 0.0}, 0).empty())
    {
        std::cerr << "testKDTreeNeighbors: nNeighbor = 0 returned neighbors\n";
        ++failures;
    }
    auto dups = makeKDTestData(300, 3, 2, 6, true);
    const KDTestTree dupTree(dups, 3, 4);
    failures += checkNeighbors(dupTree, {0.0, 0.0, 1.0}, 40, kden::Bandwidth::identity(3), "duplicate points");

    // Invalid arguments: wrong-size point, wrong-size bandwidth, and a
    // bandwidth reduced for a different pattern of NaNs
    int thrown = 0;
    try { (void)small.neighbors({0.0, 0.0}, 3); } catch (const std::invalid_argument&) { ++thrown; }
    try { (void)small.neighbors({0.0, 0.0, 0.0}, 3, kden::Bandwidth::identity(2)); } catch (const std::invalid_argument&) { ++thrown; }
    try { (void)tree.neighbors(x0WithNan(pts[0].x_, 2), 3, reduced); } catch (const std::invalid_argument&) { ++thrown; }
    if (thrown != 3)
    {
        std::cerr << "testKDTreeNeighbors: only " << thrown << " of 3 invalid searches threw\n";
        ++failures;
    }
    return failures;
}

/**
 * @brief Check the batched neighbor searches against the single-point one
 * @return Number of failures
 */
inline auto testKDTreeNeighborsBatch() -> int
{
    int failures = 0;
    const std::size_t nDim = 5;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    auto pts = makeKDTestData(3000, nDim, 3, 12);
    const kden::Bandwidth full = silvermanBandwidth(pts, nDim);
    const KDTestTree tree(pts, nDim, 16);

    std::vector<std::vector<double>> xs;
    std::vector<std::size_t> ks;
    std::vector<kden::Bandwidth> bws;
    for (std::size_t q = 0; q < 64; ++q)
    {
        xs.push_back(pts[(q * 41) % pts.size()].x_);
        if (q % 3 == 0) { xs.back()[2] = nan; }
        ks.push_back(1 + (q % 30));
        bws.push_back(q % 2 == 0 ? full : kden::Bandwidth::identity(nDim));
    }

    // Compare a batch result with single-point searches
    auto sameAsSingle = [&tree](const std::vector<std::vector<std::pair<KDTestPoint, double>>>& batch,
        const std::vector<std::vector<double>>& x, const std::vector<std::size_t>& k,
        const auto& bandwidthFor) -> bool
    {
        if (batch.size() != x.size()) { return false; }
        for (std::size_t q = 0; q < x.size(); ++q)
        {
            const auto single = tree.neighbors(x[q], k[q], bandwidthFor(q));
            if (single.size() != batch[q].size()) { return false; }
            for (std::size_t j = 0; j < single.size(); ++j)
            {
                if (single[j].second != batch[q][j].second ||
                    single[j].first.origIdx_ != batch[q][j].first.origIdx_) { return false; }
            }
        }
        return true;
    };

    // One bandwidth per search
    if (!sameAsSingle(tree.neighbors(xs, ks, bws), xs, ks,
        [&bws](const std::size_t q) -> const kden::Bandwidth& { return bws[q]; }))
    {
        std::cerr << "testKDTreeNeighborsBatch: per-search-bandwidth batch differs from single searches\n";
        ++failures;
    }

    // One shared bandwidth, mixed NaN patterns (reduced per search)
    if (!sameAsSingle(tree.neighbors(xs, ks, full), xs, ks,
        [&full](const std::size_t) -> const kden::Bandwidth& { return full; }))
    {
        std::cerr << "testKDTreeNeighborsBatch: shared-bandwidth batch differs from single searches\n";
        ++failures;
    }

    // One shared bandwidth, a common NaN pattern (reduced once for the batch)
    std::vector<std::vector<double>> xsNan;
    for (const auto& x : xs) { xsNan.push_back(x); xsNan.back()[2] = nan; }
    if (!sameAsSingle(tree.neighbors(xsNan, ks, full), xsNan, ks,
        [&full](const std::size_t) -> const kden::Bandwidth& { return full; }))
    {
        std::cerr << "testKDTreeNeighborsBatch: common-pattern batch differs from single searches\n";
        ++failures;
    }

    // Mismatched argument lengths, and an error in one search, throw
    int thrown = 0;
    try { (void)tree.neighbors(xs, std::vector<std::size_t>(3, 1), bws); } catch (const std::invalid_argument&) { ++thrown; }
    try { (void)tree.neighbors(xs, ks, std::vector<kden::Bandwidth>(2, full)); } catch (const std::invalid_argument&) { ++thrown; }
    auto badXs = xs;
    badXs[17] = {0.0, 0.0};
    try { (void)tree.neighbors(badXs, ks, full); } catch (const std::invalid_argument&) { ++thrown; }
    if (thrown != 3)
    {
        std::cerr << "testKDTreeNeighborsBatch: only " << thrown << " of 3 invalid batches threw\n";
        ++failures;
    }
    return failures;
}

#endif // TESTKDTREE_HPP
