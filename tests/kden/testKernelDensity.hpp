/**
 * @file testKernelDensity.hpp
 * @author Mark Krumholz
 * @brief Unit tests for kden::KernelDensity and kden::KernelDensityData
 * @details
 * Checks that both constructors build the same tree a KDTree would
 * over the same positions, that the weights are reordered along with
 * the points, every node's weight sums and center of mass against
 * brute force, and the weights' argument checks; RefinementList's
 * sums and largest entry against brute force; and the density
 * estimate pdf() against brute force, its normalization and
 * marginalization against numerical integration, and its argument
 * checks.
 * @date 2026-10-16
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#ifndef TESTKERNELDENSITY_HPP
#define TESTKERNELDENSITY_HPP

#include "../../src/kden/Bandwidth.hpp"
#include "../../src/kden/KernelDensity.hpp"
#include "testKDTree.hpp"
#include <Eigen/Dense> // NOLINT(misc-include-cleaner)
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <numbers>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
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

            // Each weight moved with its point, and was normalized
            double wTot = 0.0;
            for (std::size_t i = 0; i < nPt; ++i) { wTot += kdeTestWeight(i); }
            bool moved = true;
            double wSum = 0.0;
            for (std::size_t i = 0; i < nPt; ++i)
            {
                const double want = kdeTestWeight(kdeData.aux_[i]) / wTot;
                moved = moved && std::abs(kdeData.w_[i] - want) <= 1e-15 * want;
                wSum += kdeData.w_[i];
            }
            if (!moved || std::abs(wSum - 1.0) > 1e-12 || std::abs(kde.nodeW(kden::rootNode) - 1.0) > 1e-12)
            {
                std::cerr << "testKernelDensityConstruction: " << what
                    << ": weights were not reordered with their points and normalized\n";
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
    auto negW = data;
    negW.w_[7] = -0.1;
    try { const KDETest t(negW); } catch (const std::invalid_argument&) { ++thrown; }
    auto zeroW = data;
    std::ranges::fill(zeroW.w_, 0.0);
    try { const KDETest t(zeroW); } catch (const std::invalid_argument&) { ++thrown; }
    if (thrown != 5)
    {
        std::cerr << "testKernelDensityConstruction: only " << thrown << " of 5 invalid constructions threw\n";
        ++failures;
    }
    return failures;
}

/**
 * @brief Check RefinementList against brute force, through growth, slot reuse, and huge cancelling changes
 * @return Number of failures
 */
inline auto testRefinementList() -> int
{
    int failures = 0;
    std::mt19937 gen(51); // NOLINT(cert-msc32-c,cert-msc51-cpp) -- fixed seed for a reproducible test
    std::uniform_real_distribution<double> unif(0.0, 1.0);

    // Random inserts, replacements, and erasures, mirrored in a plain
    // map from slot to (node, estimate, uncertainty)
    kden::RefinementList list(2);
    std::vector<std::size_t> live;
    std::vector<std::tuple<std::size_t, double, double>> mirror;
    auto check = [&](const int step) -> bool
    {
        double est = 0.0;
        double err = 0.0;
        double maxErr = 0.0;
        for (const std::size_t s : live)
        {
            est += std::get<1>(mirror[s]);
            err += std::get<2>(mirror[s]);
            maxErr = std::max(maxErr, std::get<2>(mirror[s]));
        }
        const std::size_t w = list.worst();
        const bool ok = list.size() == live.size() &&
            std::abs(list.estimate() - est) <= 1e-12 * est && std::abs(list.uncertainty() - err) <= 1e-12 * err &&
            list.uncertainty(w) == maxErr &&
            (maxErr == 0.0 || (std::ranges::find(live, w) != live.end() && list.node(w) == std::get<0>(mirror[w])));
        if (!ok)
        {
            std::cerr << "testRefinementList: disagrees with brute force at step " << step << "\n";
        }
        return ok;
    };
    for (int step = 0; step < 3000; ++step)
    {
        const double r = unif(gen);
        const std::size_t node = static_cast<std::size_t>(step);
        const double est = unif(gen);
        const double err = unif(gen);
        if (live.empty() || r < 0.5)
        {
            // Insert: the list reuses a freed slot or takes a new one;
            // find which as the first slot not in use that holds the
            // new (unique) node
            list.insert(node, est, err);
            std::size_t slot = 0;
            while (std::ranges::find(live, slot) != live.end() || list.node(slot) != node) { ++slot; }
            if (slot >= mirror.size()) { mirror.resize(slot + 1, {kden::nullNode, 0.0, 0.0}); }
            mirror[slot] = {node, est, err};
            live.push_back(slot);
        }
        else
        {
            const std::size_t k = static_cast<std::size_t>(unif(gen) * static_cast<double>(live.size()));
            const std::size_t slot = live[k];
            if (r < 0.8)
            {
                list.replace(slot, node, est, err);
                mirror[slot] = {node, est, err};
            }
            else
            {
                list.erase(slot);
                mirror[slot] = {kden::nullNode, 0.0, 0.0};
                live.erase(live.begin() + static_cast<std::ptrdiff_t>(k));
            }
        }
        if (!check(step)) { ++failures; break; }
    }

    // No drift: an uncertainty 30 orders of magnitude above the rest,
    // replaced many times over, leaves the sums exactly those of the
    // current entries
    kden::RefinementList drift;
    drift.insert(0, 1.0, 1e30);
    for (std::size_t i = 1; i < 1000; ++i) { drift.insert(i, 1.0, 1e-3); }
    for (int rep = 0; rep < 1000; ++rep)
    {
        drift.replace(0, 0, 1.0, 1e30 * unif(gen));
        drift.replace(0, 0, 1.0, 1e-3);
    }
    double want = 0.0;
    for (int i = 0; i < 1000; ++i) { want += 1e-3; }
    if (std::abs(drift.uncertainty() - want) > 1e-12 || drift.estimate() != 1000.0)
    {
        std::cerr << "testRefinementList: sums drifted to " << drift.uncertainty() << " and "
            << drift.estimate() << " rather than " << want << " and 1000\n";
        ++failures;
    }
    return failures;
}

/**
 * @brief Brute-force kernel density estimate
 * @param data The weighted points
 * @param x Where to evaluate it; may have NaN elements
 * @param bandwidth The bandwidth; must match x (see Bandwidth::matches())
 * @return sqrt(det M) / (pi^(n / 2) sum_i w_i) sum_i w_i exp(-d_i^T M d_i),
 *   over the n specified dimensions
 */
inline auto brutePdf(const KDETestData& data, const std::vector<double>& x,
    const kden::Bandwidth& bandwidth) -> double
{
    double sum = 0.0;
    double wTot = 0.0;
    for (std::size_t i = 0; i < data.size(); ++i)
    {
        sum += data.weight(i) * std::exp(-bandwidth.dist2(x, data.position(i)));
        wTot += data.weight(i);
    }
    return std::sqrt(bandwidth.det()) * sum /
        (std::pow(std::numbers::pi, 0.5 * static_cast<double>(bandwidth.nSpecified())) * wTot);
}

/**
 * @brief Check pdf() against brute force, to within its tolerances and exactly
 * @return Number of failures
 */
inline auto testKernelDensityPdf() -> int
{
    int failures = 0;
    std::mt19937 gen(31); // NOLINT(cert-msc32-c,cert-msc51-cpp) -- fixed seed for a reproducible test
    std::normal_distribution<double> gauss(0.0, 1.0);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    KDETestData data = makeKDETestData(4000, 4, 2, 32);
    const KDETest kde(data, 8);

    // Bandwidths: Euclidean, diagonal, and full and correlated, all
    // broad enough that many points contribute
    Eigen::MatrixXd diag = Eigen::MatrixXd::Zero(4, 4);
    diag.diagonal() << 4.0, 1.0, 9.0, 2.0;
    const Eigen::MatrixXd corr = (0.2 * Eigen::MatrixXd::Identity(4, 4)) +
        (0.8 * Eigen::MatrixXd::Ones(4, 4));
    const std::vector<std::pair<std::string, kden::Bandwidth>> bandwidths{
        {"Euclidean", kden::Bandwidth::identity(4)},
        {"diagonal", kden::Bandwidth(diag)},
        {"full", kden::Bandwidth(4.0 * Eigen::MatrixXd(corr.inverse()))}};
    for (const auto& [name, bw] : bandwidths)
    {
        for (int trial = 0; trial < 20; ++trial)
        {
            std::vector<double> x(4);
            for (auto& xi : x) { xi = 2.0 * gauss(gen); }
            if (trial % 4 == 3) { x[1] = nan; }
            const kden::Bandwidth red = bw.matches(x) ? bw : bw.reduce(kden::Bandwidth::nanPattern(x));
            const double want = brutePdf(data, x, red);

            // Within the relative tolerance (which is relative to the
            // estimate, not the true value)
            const double relTol = 1e-3;
            const double got = kde.pdf(x, relTol, 0.0, bw);
            if (!(std::abs(got - want) <= relTol * got * (1.0 + 1e-9)))
            {
                std::cerr << "testKernelDensityPdf: " << name << ", trial " << trial << ": "
                    << got << " vs brute force " << want << " misses relTol = " << relTol << "\n";
                ++failures;
                break;
            }

            // Within the absolute tolerance
            const double absTol = 0.1 * want;
            const double gotAbs = kde.pdf(x, 0.0, absTol, bw);
            if (!(std::abs(gotAbs - want) <= absTol))
            {
                std::cerr << "testKernelDensityPdf: " << name << ", trial " << trial << ": "
                    << gotAbs << " vs brute force " << want << " misses absTol = " << absTol << "\n";
                ++failures;
                break;
            }

            // Zero tolerances: exact
            const double exact = kde.pdf(x, 0.0, 0.0, bw);
            if (std::abs(exact - want) > 1e-12 * want)
            {
                std::cerr << "testKernelDensityPdf: " << name << ", trial " << trial << ": "
                    << exact << " with zero tolerance differs from brute force " << want << "\n";
                ++failures;
                break;
            }
        }
    }

    // The default bandwidth is the Euclidean one
    const std::vector<double> x0{0.5, 0.5, -0.3, 0.1};
    if (kde.pdf(x0, 1e-4, 0.0) != kde.pdf(x0, 1e-4, 0.0, kden::Bandwidth::identity(4)))
    {
        std::cerr << "testKernelDensityPdf: default bandwidth is not the Euclidean one\n";
        ++failures;
    }

    // Far from every point, a loose absolute tolerance stops at the root
    const std::vector<double> far{100.0, 100.0, 100.0, 100.0};
    if (kde.pdf(far, 0.0, 1e-300) != 0.0)
    {
        std::cerr << "testKernelDensityPdf: estimate far from every point is not zero\n";
        ++failures;
    }

    // Errors: wrong-size point or bandwidth, a bandwidth reduced for a
    // different pattern, and a singular bandwidth
    int thrown = 0;
    try { (void)kde.pdf(std::vector<double>{0.0, 0.0}, 1e-3, 0.0); } catch (const std::invalid_argument&) { ++thrown; }
    try { (void)kde.pdf(x0, 1e-3, 0.0, kden::Bandwidth::identity(3)); } catch (const std::invalid_argument&) { ++thrown; }
    const kden::Bandwidth reduced = kden::Bandwidth::identity(4).reduce({true, false, false, false});
    try { (void)kde.pdf(std::vector<double>{0.0, nan, 0.0, 0.0}, 1e-3, 0.0, reduced); } catch (const std::invalid_argument&) { ++thrown; }
    Eigen::MatrixXd singular = Eigen::MatrixXd::Identity(4, 4);
    singular(3, 3) = 0.0;
    try { (void)kde.pdf(x0, 1e-3, 0.0, kden::Bandwidth(singular)); } catch (const std::domain_error&) { ++thrown; }
    if (thrown != 4)
    {
        std::cerr << "testKernelDensityPdf: only " << thrown << " of 4 invalid evaluations threw\n";
        ++failures;
    }
    return failures;
}

/**
 * @brief Check pdf()'s normalization and marginalization by numerical integration
 * @return Number of failures
 * @details
 * In two dimensions, with a full bandwidth: the estimate integrates to
 * one, and evaluating it with one coordinate NaN gives its integral
 * over that coordinate.
 */
inline auto testKernelDensityNormalization() -> int
{
    int failures = 0;
    KDETestData data = makeKDETestData(40, 2, 2, 41);
    const KDETest kde(data, 4);
    Eigen::MatrixXd m(2, 2);
    m << 3.0, 1.2, 1.2, 1.0;
    const kden::Bandwidth bw(m);
    const double nan = std::numeric_limits<double>::quiet_NaN();

    // Midpoint rule over a box wide enough to hold essentially all the
    // probability (the points have coordinates of a few, and the
    // kernels widths of order one)
    const double lim = 20.0;
    const int nGrid = 400;
    const double dx = 2.0 * lim / nGrid;
    double total = 0.0;
    const double xFix = 0.7;
    double marginal = 0.0;
    for (int j = 0; j < nGrid; ++j)
    {
        const double y = -lim + ((j + 0.5) * dx);
        marginal += kde.pdf(std::vector<double>{xFix, y}, 0.0, 0.0, bw) * dx;
        for (int i = 0; i < nGrid; ++i)
        {
            const double x = -lim + ((i + 0.5) * dx);
            total += kde.pdf(std::vector<double>{x, y}, 0.0, 0.0, bw) * dx * dx;
        }
    }
    if (std::abs(total - 1.0) > 1e-6)
    {
        std::cerr << "testKernelDensityNormalization: estimate integrates to " << total << ", not 1\n";
        ++failures;
    }
    const double margPdf = kde.pdf(std::vector<double>{xFix, nan}, 0.0, 0.0, bw);
    if (std::abs(marginal - margPdf) > 1e-6 * margPdf)
    {
        std::cerr << "testKernelDensityNormalization: marginal " << margPdf
            << " differs from the integral over the NaN dimension " << marginal << "\n";
        ++failures;
    }
    return failures;
}

#endif // TESTKERNELDENSITY_HPP
