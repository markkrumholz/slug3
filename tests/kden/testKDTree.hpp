/**
 * @file testKDTree.hpp
 * @author Mark Krumholz
 * @brief Unit tests for kden::KDTree and kden::KDTreeData
 * @details
 * Covers the traversal helpers, tree construction (every node's
 * bounding box and the split invariants, with and without auxiliary
 * data), and the nearest-neighbor search, single-point and batched,
 * against brute force for Euclidean, diagonal, strongly correlated,
 * and partially-unspecified metrics. (Bandwidth itself is tested in
 * testBandwidth.hpp.) Test data carry their original index as
 * auxiliary data, so results can be matched to brute force point by
 * point even though building the tree reorders them.
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

/** Test data: each point's auxiliary data is its original index */
using KDTestData = kden::KDTreeData<std::size_t>;
/** A tree over KDTestData */
using KDTestTree = kden::KDTree<KDTestData>;

/**
 * @brief Make a data set with strongly correlated dimensions
 * @param nPt Number of points
 * @param nDim Number of dimensions; the first nCorr share a common
 *   component, so are strongly correlated, and the rest are independent
 * @param nCorr Number of correlated dimensions
 * @param seed Random seed
 * @param coarse If true, round coordinates to integers, to create many
 *   duplicate coordinate values
 * @return The points, with each point's original index as its
 *   auxiliary data
 */
inline auto makeKDTestData(const std::size_t nPt, const std::size_t nDim,
    const std::size_t nCorr, const unsigned seed, const bool coarse = false) -> KDTestData
{
    std::mt19937 gen(seed);
    std::normal_distribution<double> gauss(0.0, 1.0);
    KDTestData data;
    data.nDim_ = nDim;
    data.x_.resize(nPt * nDim);
    data.aux_.resize(nPt);
    for (std::size_t i = 0; i < nPt; ++i)
    {
        const double common = 3.0 * gauss(gen);
        for (std::size_t d = 0; d < nDim; ++d)
        {
            double x = d < nCorr ? common + (0.2 * gauss(gen)) : gauss(gen);
            if (coarse) { x = std::round(x); }
            data.x_[(i * nDim) + d] = x;
        }
        data.aux_[i] = i;
    }
    return data;
}

/**
 * @brief A point's position, as a vector
 * @param data The data
 * @param i Index of the point
 * @return A copy of its coordinates
 */
template <typename AuxType>
inline auto positionOf(const kden::KDTreeData<AuxType>& data, const std::size_t i) -> std::vector<double>
{
    const auto pos = data.position(i);
    return {pos.begin(), pos.end()};
}

/**
 * @brief A Silverman-like bandwidth: the inverse of the scaled data covariance
 * @param data The data
 * @return A Bandwidth holding (s^2 Cov)^-1, with s^2 = N^(-2 / (nDim + 4))
 */
inline auto silvermanBandwidth(const KDTestData& data) -> kden::Bandwidth
{
    const auto n = static_cast<Eigen::Index>(data.nDim_);
    const auto nPt = static_cast<Eigen::Index>(data.size());
    const Eigen::Map<const Eigen::MatrixXd> x(data.x_.data(), n, nPt);   // one column per point
    const Eigen::VectorXd mean = x.rowwise().mean();
    const Eigen::MatrixXd dx = x.colwise() - mean;
    const Eigen::MatrixXd cov = (dx * dx.transpose()) / static_cast<double>(nPt);
    const double s2 = std::pow(static_cast<double>(nPt), -2.0 / static_cast<double>(n + 4));
    const Eigen::MatrixXd m = (s2 * cov).inverse();
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
 * @brief Check one node's bounding box against brute force
 * @param tree The tree
 * @param idx Index of the node
 * @return Number of discrepancies
 */
template <typename Data>
inline auto checkNodeBounds(const kden::KDTree<Data>& tree, const std::size_t idx) -> int
{
    const auto& data = tree.data();
    const auto& node = tree.nodes()[idx];
    const auto nodeLo = tree.nodeLo(idx);
    const auto nodeHi = tree.nodeHi(idx);
    const std::size_t nd = data.nDim_;
    std::vector<double> lo(nd, std::numeric_limits<double>::infinity());
    std::vector<double> hi(nd, -std::numeric_limits<double>::infinity());
    for (std::size_t i = node.start_; i < node.start_ + node.count_; ++i)
    {
        const auto pos = data.position(i);
        for (std::size_t k = 0; k < nd; ++k)
        {
            lo[k] = std::min(lo[k], pos[k]);
            hi[k] = std::max(hi[k], pos[k]);
        }
    }
    int bad = 0;
    for (std::size_t k = 0; k < nd; ++k)
    {
        if (lo[k] != nodeLo[k] || hi[k] != nodeHi[k]) { ++bad; }
    }
    return bad;
}

/**
 * @brief Check every invariant of a built tree
 * @param tree The tree
 * @param nPt Number of points it was built from
 * @return Number of violations
 */
template <typename Data>
inline auto checkKDTreeInvariants(const kden::KDTree<Data>& tree, const std::size_t nPt) -> int
{
    using Tree = kden::KDTree<Data>;
    int bad = 0;
    const auto& data = tree.data();
    const auto& nodes = tree.nodes();
    std::size_t covered = 0;
    std::size_t idx = kden::rootNode;
    while (idx != kden::nullNode)
    {
        const auto& node = nodes[idx];
        bad += checkNodeBounds(tree, idx);
        if (node.splitDim_ < 0)
        {
            // Leaf: no children, at most leafSize points, contiguous with the previous leaf
            if (node.count_ > tree.leafSize() || node.start_ != covered ||
                node.lchild_ != kden::nullNode || node.rchild_ != kden::nullNode) { ++bad; }
            covered += node.count_;
            idx = Tree::next(idx);
            continue;
        }

        // Interior node: children linked, split, and ordered along splitDim
        const auto& l = nodes[node.lchild_];
        const auto& r = nodes[node.rchild_];
        if (node.lchild_ != Tree::left(idx) || node.rchild_ != Tree::right(idx) ||
            l.parent_ != idx || r.parent_ != idx || l.start_ != node.start_ ||
            l.count_ != (node.count_ + 1) / 2 || r.start_ != l.start_ + l.count_ ||
            l.count_ + r.count_ != node.count_) { ++bad; }
        const auto d = static_cast<std::size_t>(node.splitDim_);
        double lmax = -std::numeric_limits<double>::infinity();
        double rmin = std::numeric_limits<double>::infinity();
        for (std::size_t i = l.start_; i < l.start_ + l.count_; ++i) { lmax = std::max(lmax, data.position(i)[d]); }
        for (std::size_t i = r.start_; i < r.start_ + r.count_; ++i) { rmin = std::min(rmin, data.position(i)[d]); }
        if (lmax > rmin) { ++bad; }
        idx = Tree::left(idx);
    }
    if (covered != nPt) { ++bad; }

    // With auxiliary data (each point's original index), the points are
    // still a permutation of the original data
    if constexpr (Data::hasAux)
    {
        std::vector<int> seen(nPt, 0);
        for (std::size_t i = 0; i < data.size(); ++i) { seen[data.aux(i)]++; }
        if (std::ranges::any_of(seen, [](const int s) -> bool { return s != 1; })) { ++bad; }
    }
    return bad;
}

/**
 * @brief Check tree construction, with and without a metric and auxiliary data, and its argument checks
 * @return Number of failures
 */
inline auto testKDTreeConstruction() -> int
{
    int failures = 0;
    // A fixed, strongly correlated metric (a data-derived one is
    // singular for the tiniest data sets)
    const Eigen::MatrixXd corr = (0.2 * Eigen::MatrixXd::Identity(4, 4)) +
        (0.8 * Eigen::MatrixXd::Ones(4, 4));
    const kden::Bandwidth metric(corr.inverse());
    for (const std::size_t nPt : {1UL, 16UL, 17UL, 100UL, 5000UL})
    {
        for (const bool withMetric : {false, true})
        {
            auto data = makeKDTestData(nPt, 4, 2, 7, true);
            const auto tree = withMetric ? std::make_unique<KDTestTree>(data, 16, metric) :
                std::make_unique<KDTestTree>(data, 16);
            const int bad = checkKDTreeInvariants(*tree, nPt);
            if (bad != 0)
            {
                std::cerr << "testKDTreeConstruction: " << bad << " violations for nPt = "
                    << nPt << (withMetric ? " with" : " without") << " a metric\n";
                ++failures;
            }
        }
    }

    // Without auxiliary data: the same positions build the same tree
    auto withAux = makeKDTestData(5000, 4, 2, 7, true);
    kden::KDTreeData<> noAux{.nDim_ = withAux.nDim_, .x_ = withAux.x_, .aux_ = {}};
    const KDTestTree auxTree(withAux, 16);
    const kden::KDTree<> plainTree(noAux, 16);
    if (checkKDTreeInvariants(plainTree, 5000) != 0 || noAux.x_ != withAux.x_)
    {
        std::cerr << "testKDTreeConstruction: tree without auxiliary data differs from one with it\n";
        ++failures;
    }

    // Splitting restricted to some dimensions: a valid tree that never
    // splits along the others, with or without a metric
    for (const bool withMetric : {false, true})
    {
        auto restricted = makeKDTestData(5000, 4, 2, 8);
        const std::vector<std::size_t> noSplit{0, 2};
        const auto tree = withMetric ? std::make_unique<KDTestTree>(restricted, 16, metric, noSplit) :
            std::make_unique<KDTestTree>(restricted, 16, noSplit);
        bool splitsOk = true;
        bool splitsBoth = false;
        for (const auto& node : tree->nodes())
        {
            splitsOk = splitsOk && node.splitDim_ != 0 && node.splitDim_ != 2;
            splitsBoth = splitsBoth || node.splitDim_ == 3;
        }
        if (checkKDTreeInvariants(*tree, 5000) != 0 || !splitsOk || !splitsBoth)
        {
            std::cerr << "testKDTreeConstruction: tree with noSplitDims = {0, 2}"
                << (withMetric ? " and a metric" : "") << " is invalid or splits along an excluded dimension\n";
            ++failures;
        }
    }

    // Invalid data and arguments
    int thrown = 0;
    auto data = makeKDTestData(10, 3, 2, 1);
    try { const KDTestTree t(data, 0); } catch (const std::invalid_argument&) { ++thrown; }
    try { const KDTestTree t(data, 16, kden::Bandwidth::identity(2)); } catch (const std::invalid_argument&) { ++thrown; }
    auto badDim = data;
    badDim.nDim_ = 0;
    try { const KDTestTree t(badDim); } catch (const std::invalid_argument&) { ++thrown; }
    auto badX = data;
    badX.x_.pop_back();
    try { const KDTestTree t(badX); } catch (const std::invalid_argument&) { ++thrown; }
    auto badAux = data;
    badAux.aux_.pop_back();
    try { const KDTestTree t(badAux); } catch (const std::invalid_argument&) { ++thrown; }
    // Non-finite stored positions are rejected: NaN is
    // allowed only in search points, and infinities would give nodes
    // non-finite bounds
    auto nanX = data;
    nanX.x_[7] = std::numeric_limits<double>::quiet_NaN();
    try { const KDTestTree t(nanX); } catch (const std::invalid_argument&) { ++thrown; }
    auto infX = data;
    infX.x_[4] = std::numeric_limits<double>::infinity();
    try { const KDTestTree t(infX); } catch (const std::invalid_argument&) { ++thrown; }
    // noSplitDims out of range, or excluding every dimension
    try { const KDTestTree t(data, 16, std::vector<std::size_t>{3}); } catch (const std::invalid_argument&) { ++thrown; }
    try { const KDTestTree t(data, 16, std::vector<std::size_t>{2, 0, 1}); } catch (const std::invalid_argument&) { ++thrown; }
    if (thrown != 9)
    {
        std::cerr << "testKDTreeConstruction: only " << thrown << " of 9 invalid constructions threw\n";
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
    const auto& data = tree.data();
    std::vector<double> byOrig(data.size());
    for (std::size_t i = 0; i < data.size(); ++i) { byOrig[data.aux(i)] = effective.dist2(x, data.position(i)); }
    std::vector<double> sorted = byOrig;
    std::ranges::sort(sorted);

    const auto result = tree.neighbors(x, k, bandwidth);
    const std::size_t expect = std::min(k, data.size());
    bool ok = result.size() == expect;
    for (std::size_t j = 0; ok && j < expect; ++j)
    {
        // Distances match brute force, in order, and each is the true
        // distance to the point returned (ties may return either point)
        ok = result[j].dist2_ == sorted[j] && byOrig[data.aux(result[j].idx_)] == result[j].dist2_;
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
 * @brief Copy a vector with one element made NaN
 * @param x The vector
 * @param dim The element to make NaN
 * @return The copy
 */
inline auto withNanAt(std::vector<double> x, const std::size_t dim) -> std::vector<double>
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
    auto data = makeKDTestData(4000, nDim, 4, 3);
    const kden::Bandwidth full = silvermanBandwidth(data);
    const kden::Bandwidth diag(Eigen::MatrixXd(full.h2Inv().diagonal().asDiagonal()));
    const kden::Bandwidth euclid = kden::Bandwidth::identity(nDim);
    const KDTestTree tree(data, 8, full);

    // Partially specified searches leave dimensions 1 and 4 unspecified;
    // both a pre-reduced bandwidth and automatic reduction are tested
    std::vector<bool> unspec(nDim, false);
    unspec[1] = true;
    unspec[4] = true;
    const kden::Bandwidth reduced = full.reduce(unspec);

    std::mt19937 gen(9); // NOLINT(cert-msc32-c,cert-msc51-cpp) -- fixed seed for a reproducible test
    std::normal_distribution<double> gauss(0.0, 1.0);
    for (std::size_t q = 0; q < 25; ++q)
    {
        // Search points near (but not at) data points, and some far away
        std::vector<double> x = positionOf(tree.data(), (q * 157) % data.size());
        for (auto& v : x) { v += (q % 5 == 0 ? 5.0 : 0.05) * gauss(gen); }
        const std::vector<double> xPart = withNanAt(withNanAt(x, 1), 4);
        for (const std::size_t k : {1UL, 7UL, 50UL})
        {
            failures += checkNeighbors(tree, x, k, euclid, "Euclidean");
            failures += checkNeighbors(tree, x, k, diag, "diagonal metric");
            failures += checkNeighbors(tree, x, k, full, "full correlated metric");
            failures += checkNeighbors(tree, xPart, k, reduced, "partially specified, pre-reduced");
            failures += checkNeighbors(tree, xPart, k, full, "partially specified, reduced automatically");
        }
    }

    // A tree that never splits along the dimensions searches leave
    // unspecified still finds the exact neighbors, for both partially
    // and fully specified searches
    auto restricted = makeKDTestData(4000, nDim, 4, 3);
    const KDTestTree rTree(restricted, 8, full, std::vector<std::size_t>{1, 4});
    for (std::size_t q = 0; q < 10; ++q)
    {
        std::vector<double> x = positionOf(rTree.data(), (q * 157) % restricted.size());
        for (auto& v : x) { v += 0.05 * gauss(gen); }
        failures += checkNeighbors(rTree, withNanAt(withNanAt(x, 1), 4), 7, full, "restricted splits, partially specified");
        failures += checkNeighbors(rTree, x, 7, full, "restricted splits, fully specified");
    }

    // The Euclidean overload matches an explicit identity bandwidth
    const std::vector<double> x3 = positionOf(tree.data(), 3);
    const auto viaDefault = tree.neighbors(x3, 5);
    const auto viaIdentity = tree.neighbors(x3, 5, euclid);
    bool sameEuclid = viaDefault.size() == viaIdentity.size();
    for (std::size_t j = 0; sameEuclid && j < viaDefault.size(); ++j)
    {
        sameEuclid = viaDefault[j].dist2_ == viaIdentity[j].dist2_ && viaDefault[j].idx_ == viaIdentity[j].idx_;
    }
    if (!sameEuclid)
    {
        std::cerr << "testKDTreeNeighbors: default (Euclidean) search differs from identity bandwidth\n";
        ++failures;
    }

    // Edge cases: more neighbors than points, none, and duplicate points
    auto few = makeKDTestData(5, 3, 2, 4);
    const KDTestTree small(few, 2);
    const std::vector<double> origin{0.0, 0.0, 0.0};
    failures += checkNeighbors(small, origin, 10, kden::Bandwidth::identity(3), "nNeighbor > number of points");
    if (!small.neighbors(origin, 0).empty())
    {
        std::cerr << "testKDTreeNeighbors: nNeighbor = 0 returned neighbors\n";
        ++failures;
    }
    auto dups = makeKDTestData(300, 3, 2, 6, true);
    const KDTestTree dupTree(dups, 4);
    failures += checkNeighbors(dupTree, std::vector<double>{0.0, 0.0, 1.0}, 40, kden::Bandwidth::identity(3), "duplicate points");

    // Invalid arguments: wrong-size point, wrong-size bandwidth, and a
    // bandwidth reduced for a different pattern of NaNs
    int thrown = 0;
    try { (void)small.neighbors(std::vector<double>{0.0, 0.0}, 3); } catch (const std::invalid_argument&) { ++thrown; }
    try { (void)small.neighbors(origin, 3, kden::Bandwidth::identity(2)); } catch (const std::invalid_argument&) { ++thrown; }
    try { (void)tree.neighbors(withNanAt(x3, 2), 3, reduced); } catch (const std::invalid_argument&) { ++thrown; }
    if (thrown != 3)
    {
        std::cerr << "testKDTreeNeighbors: only " << thrown << " of 3 invalid searches threw\n";
        ++failures;
    }

    // A tree without auxiliary data gives the same distances
    kden::KDTreeData<> plain{.nDim_ = 3, .x_ = dups.x_, .aux_ = {}};
    const kden::KDTree<> plainTree(plain, 4);
    const auto a = plainTree.neighbors(std::vector<double>{0.0, 0.0, 1.0}, 40);
    const auto b = dupTree.neighbors(std::vector<double>{0.0, 0.0, 1.0}, 40);
    bool samePlain = a.size() == b.size();
    for (std::size_t j = 0; samePlain && j < a.size(); ++j) { samePlain = a[j].dist2_ == b[j].dist2_; }
    if (!samePlain)
    {
        std::cerr << "testKDTreeNeighbors: tree without auxiliary data gives different neighbors\n";
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
    auto data = makeKDTestData(3000, nDim, 3, 12);
    const kden::Bandwidth full = silvermanBandwidth(data);
    const KDTestTree tree(data, 16);

    std::vector<std::vector<double>> xs;
    std::vector<std::size_t> ks;
    std::vector<kden::Bandwidth> bws;
    for (std::size_t q = 0; q < 64; ++q)
    {
        xs.push_back(positionOf(tree.data(), (q * 41) % data.size()));
        if (q % 3 == 0) { xs.back() = withNanAt(xs.back(), 2); }
        ks.push_back(1 + (q % 30));
        bws.push_back(q % 2 == 0 ? full : kden::Bandwidth::identity(nDim));
    }

    // Compare a batch result with single-point searches
    auto sameAsSingle = [&tree](const std::vector<std::vector<kden::Neighbor>>& batch,
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
                if (single[j].dist2_ != batch[q][j].dist2_ || single[j].idx_ != batch[q][j].idx_) { return false; }
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
    for (const auto& x : xs) { xsNan.push_back(withNanAt(x, 2)); }
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
