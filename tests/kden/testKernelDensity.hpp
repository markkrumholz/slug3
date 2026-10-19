/**
 * @file testKernelDensity.hpp
 * @author Mark Krumholz
 * @brief Unit tests for kden::KernelDensity and kden::KernelDensityData
 * @details
 * Checks that both constructors build the same tree a KDTree would
 * over the same positions, that the weights are reordered along with
 * the points, every node's weight sum against
 * brute force, and the weights' argument checks; RefinementList's
 * sums and largest entry against brute force; and the density
 * estimate pdf() against brute force, its normalization and
 * marginalization against numerical integration, and its argument
 * checks; and conditional estimates (ConditionalDensity) against the
 * full estimate, exactly and within massTol's L1 bound, with and
 * without marginalized dimensions.
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
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <variant>
#include <utility>
#include <vector>

/** Weighted test data: each point's auxiliary data is its original index */
using KDETestData = kden::OwnedKernelDensityData<std::size_t>;
/** A view of KDETestData, which estimates are built over */
using KDETestView = kden::KernelDensityData<std::size_t>;
/** A kernel density estimate over KDETestView */
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
 * @brief Check every node's weight sum against brute force
 * @param kde The kernel density estimate
 * @return Number of discrepancies
 */
inline auto checkNodeWeights(const KDETest& kde) -> int
{
    const auto& data = kde.data();
    int bad = 0;
    std::size_t idx = kden::rootNode;
    while (idx != kden::nullNode)
    {
        const auto& node = kde.nodes()[idx];
        double w = 0.0;
        for (std::size_t i = node.start_; i < node.start_ + node.count_; ++i) { w += data.weight(i); }
        if (std::abs(w - kde.nodeW(idx)) > 1e-10 * w) { ++bad; }
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

    for (const std::size_t nPt : {1UL, 16UL, 17UL, 100UL, 5000UL, 300000UL})
    {
        for (const bool withMetric : {false, true})
        {
            const std::string what = "nPt = " + std::to_string(nPt) +
                (withMetric ? " with" : " without") + " a metric";
            KDETestData kdeData = makeKDETestData(nPt, 4, 2, 21);
            KDTestData treeData = makeKDTestData(nPt, 4, 2, 21);
            auto kdeView = kdeData.view();
            auto treeView = treeData.view();
            const KDETest kde = withMetric ? KDETest(kdeView, 8, splitBw) : KDETest(kdeView, 8);
            const KDTestTree tree = withMetric ? KDTestTree(treeView, 8, splitBw) : KDTestTree(treeView, 8);

            // The same tree as a KDTree builds over the same positions
            if (kdeData.aux_ != treeData.aux_ || kdeData.x_ != treeData.x_ ||
                kde.nLeaves() != tree.nLeaves() || checkKDTreeInvariants(kde, nPt) != 0)
            {
                std::cerr << "testKernelDensityConstruction: " << what
                    << ": tree differs from the equivalent KDTree\n";
                ++failures;
            }

            // Each weight moved with its point, and was normalized (to
            // within the rounding of summing nPt weights, which differs
            // between this sum and the constructor's; a weight on the
            // wrong point would be off by order unity)
            const double tol = 1e-15 * static_cast<double>(std::max<std::size_t>(nPt, 1000));
            double wTot = 0.0;
            for (std::size_t i = 0; i < nPt; ++i) { wTot += kdeTestWeight(i); }
            bool moved = true;
            double wSum = 0.0;
            for (std::size_t i = 0; i < nPt; ++i)
            {
                const double want = kdeTestWeight(kdeData.aux_[i]) / wTot;
                moved = moved && std::abs(kdeData.w_[i] - want) <= tol * want;
                wSum += kdeData.w_[i];
            }
            if (!moved || std::abs(wSum - 1.0) > tol || std::abs(kde.nodeW(kden::rootNode) - 1.0) > tol)
            {
                std::cerr << "testKernelDensityConstruction: " << what
                    << ": weights were not reordered with their points and normalized\n";
                ++failures;
            }

            const int bad = checkNodeWeights(kde);
            if (bad != 0)
            {
                std::cerr << "testKernelDensityConstruction: " << what << ": " << bad
                    << " node weight sums are wrong\n";
                ++failures;
            }
        }
    }

    // Reweighting in place: renormalized, with node sums recomputed
    {
        KDETestData owned = makeKDETestData(500, 3, 2, 3);
        auto view = owned.view();
        KDETest kde(view, 8);
        for (std::size_t i = 0; i < view.size(); ++i) { view.w_[i] = (i % 2 == 0) ? 3.0 : 0.0; }
        kde.reweight();
        double wSum = 0.0;
        for (std::size_t i = 0; i < view.size(); ++i) { wSum += view.w_[i]; }
        if (std::abs(wSum - 1.0) > 1e-12 || checkNodeWeights(kde) != 0)
        {
            std::cerr << "testKernelDensityConstruction: reweight() did not renormalize or update node sums\n";
            ++failures;
        }
        int threw = 0;
        std::ranges::fill(view.w_, 0.0);
        try { kde.reweight(); } catch (const std::invalid_argument&) { ++threw; }
        std::ranges::fill(view.w_, 1.0);
        view.w_[17] = -1.0;
        try { kde.reweight(); } catch (const std::invalid_argument&) { ++threw; }
        view.w_[17] = std::numeric_limits<double>::quiet_NaN();
        try { kde.reweight(); } catch (const std::invalid_argument&) { ++threw; }
        if (threw != 3)
        {
            std::cerr << "testKernelDensityConstruction: reweight() accepted invalid weights\n";
            ++failures;
        }
    }

    // Huge weights, whose sum overflows a double, still normalize
    KDETestData huge = makeKDETestData(100, 3, 2, 2);
    std::ranges::fill(huge.w_, 0.5 * std::numeric_limits<double>::max());
    auto hugeView = huge.view();
    const KDETest hugeKde(hugeView, 8);
    if (std::abs(hugeKde.nodeW(kden::rootNode) - 1.0) > 1e-12 || std::abs(huge.w_[0] - 0.01) > 1e-15)
    {
        std::cerr << "testKernelDensityConstruction: weights whose sum overflows did not normalize\n";
        ++failures;
    }

    // Invalid weights: wrong number, and non-finite
    int thrown = 0;
    const KDETestData data = makeKDETestData(10, 3, 2, 1);
    // Each case builds an estimate over a view of its own copy of data
    auto tryBuild = [&thrown](KDETestData d) -> void
    {
        auto v = d.view();
        try { const KDETest t(v); } catch (const std::invalid_argument&) { ++thrown; }
    };
    auto badW = data;
    badW.w_.pop_back();
    tryBuild(badW);
    auto nanW = data;
    nanW.w_[2] = std::numeric_limits<double>::quiet_NaN();
    tryBuild(nanW);
    auto infW = data;
    infW.w_[5] = -std::numeric_limits<double>::infinity();
    tryBuild(infW);
    auto negW = data;
    negW.w_[7] = -0.1;
    tryBuild(negW);
    auto zeroW = data;
    std::ranges::fill(zeroW.w_, 0.0);
    tryBuild(zeroW);
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
inline auto brutePdf(const KDETestView& data, const std::vector<double>& x,
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
    KDETestData owned = makeKDETestData(4000, 4, 2, 32);
    auto data = owned.view();
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

    // A tree that never splits along dimension 1 gives the same
    // answers, to within the tolerance, including for marginal
    // densities over dimension 1
    KDETestData rOwned = makeKDETestData(4000, 4, 2, 32);
    auto rData = rOwned.view();
    const KDETest rKde(rData, 8, std::vector<std::size_t>{1});
    for (int trial = 0; trial < 10; ++trial)
    {
        std::vector<double> x(4);
        for (auto& xi : x) { xi = 2.0 * gauss(gen); }
        if (trial % 2 == 0) { x[1] = nan; }
        const auto& bw = bandwidths[2].second;
        const kden::Bandwidth red = bw.matches(x) ? bw : bw.reduce(kden::Bandwidth::nanPattern(x));
        const double want = brutePdf(rData, x, red);
        const double got = rKde.pdf(x, 1e-3, 0.0, bw);
        const double exact = rKde.pdf(x, 0.0, 0.0, bw);
        if (!(std::abs(got - want) <= 1e-3 * got * (1.0 + 1e-9)) || std::abs(exact - want) > 1e-12 * want)
        {
            std::cerr << "testKernelDensityPdf: restricted splits, trial " << trial << ": " << got
                << " (exact " << exact << ") vs brute force " << want << "\n";
            ++failures;
            break;
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
    KDETestData owned = makeKDETestData(40, 2, 2, 41);
    auto data = owned.view();
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

/**
 * @brief Check conditional estimates against the full estimate they come from
 * @return Number of failures
 * @details
 * With massTol = 0, marginal() times the conditional pdf must equal
 * the full estimate at (x0_F, y) (marginalized over G, if any), and
 * marginal() the full estimate's marginal over F. With massTol > 0,
 * the conditional density must be within L1 distance 2 massTol of the
 * exact one, while keeping far fewer points.
 */
inline auto testConditionalDensity() -> int
{
    int failures = 0;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    std::mt19937 gen(61); // NOLINT(cert-msc32-c,cert-msc51-cpp) -- fixed seed for a reproducible test
    std::normal_distribution<double> gauss(0.0, 1.0);
    KDETestData owned = makeKDETestData(3000, 4, 2, 62);
    auto data = owned.view();
    const KDETest kde(data, 8);
    const Eigen::MatrixXd corr = (0.2 * Eigen::MatrixXd::Identity(4, 4)) +
        (0.8 * Eigen::MatrixXd::Ones(4, 4));
    const kden::Bandwidth bw(4.0 * Eigen::MatrixXd(corr.inverse()));

    // Exact (massTol = 0): fix 0 and 2, keep 1, and keep or marginalize 3
    for (const bool marginalize : {false, true})
    {
        const std::string what = marginalize ? "marginalizing dimension 3" : "no marginalization";
        const std::vector<std::size_t> marginalDims = marginalize ? std::vector<std::size_t>{3} : std::vector<std::size_t>{};
        for (int trial = 0; trial < 5; ++trial)
        {
            const double x0v = 1.5 * gauss(gen);
            const double x2v = 1.5 * gauss(gen);
            const std::vector<double> x0{x0v, nan, x2v, marginalize ? 0.3 : nan};
            const auto cd = kde.conditional(x0, marginalDims, 0.0, bw);
            const std::vector<std::size_t> wantDims = marginalize ? std::vector<std::size_t>{1} : std::vector<std::size_t>{1, 3};
            const double marg = kde.pdf(std::vector<double>{x0v, nan, x2v, nan}, 0.0, 0.0, bw);
            bool ok = cd.dims() == wantDims && std::abs(cd.marginal() - marg) <= 1e-10 * marg;
            for (int j = 0; ok && j < 5; ++j)
            {
                const double y1 = 2.0 * gauss(gen);
                const double y3 = 2.0 * gauss(gen);
                const std::vector<double> y = marginalize ? std::vector<double>{y1} : std::vector<double>{y1, y3};
                const double full = kde.pdf(std::vector<double>{x0v, y1, x2v, marginalize ? nan : y3}, 0.0, 0.0, bw);
                const double got = cd.marginal() * cd.pdf(y, 0.0, 0.0);
                ok = std::abs(got - full) <= 1e-10 * full + 1e-300;
                if (!ok)
                {
                    std::cerr << "testConditionalDensity: " << what << ", trial " << trial << ": "
                        << got << " vs full estimate " << full << "\n";
                }
            }
            if (!ok)
            {
                std::cerr << "testConditionalDensity: " << what << ", trial " << trial
                    << ": wrong dimensions or marginal " << cd.marginal() << " vs " << marg << "\n";
                ++failures;
                break;
            }
        }
    }

    // Pruned (massTol > 0), keeping one dimension: L1 distance from the
    // exact conditional density, by the midpoint rule
    for (const double massTol : {1e-1, 1e-2, 1e-3})
    {
        const std::vector<double> x0{0.4, nan, 0.9, -0.2};
        const auto exact = kde.conditional(x0, {}, 0.0, bw);
        const auto pruned = kde.conditional(x0, {}, massTol, bw);
        const double lim = 15.0;
        const int nGrid = 3000;
        const double dy = 2.0 * lim / nGrid;
        double l1 = 0.0;
        for (int j = 0; j < nGrid; ++j)
        {
            const std::vector<double> y{-lim + ((j + 0.5) * dy)};
            l1 += std::abs(pruned.pdf(y, 0.0, 0.0) - exact.pdf(y, 0.0, 0.0)) * dy;
        }
        if (l1 > 2.0 * massTol || pruned.data().size() >= exact.data().size() ||
            pruned.marginal() > exact.marginal() || pruned.marginal() < (1.0 - massTol) * exact.marginal())
        {
            std::cerr << "testConditionalDensity: massTol = " << massTol << ": L1 distance " << l1
                << ", " << pruned.data().size() << " of " << exact.data().size() << " points kept, marginal "
                << pruned.marginal() << " vs " << exact.marginal() << "\n";
            ++failures;
        }
    }

    // Auxiliary data comes along, and a moved-from result's estimate
    // still works from its new home
    auto cd = kde.conditional(std::vector<double>{0.0, nan, 0.0, nan}, {}, 1e-2, bw);
    const double before = cd.pdf(std::vector<double>{0.1, -0.2}, 0.0, 0.0);
    const auto moved = std::move(cd);
    bool auxOk = moved.pdf(std::vector<double>{0.1, -0.2}, 0.0, 0.0) == before;
    std::vector<int> seen(data.size(), 0);
    for (std::size_t i = 0; i < moved.data().size(); ++i) { auxOk = auxOk && ++seen[moved.data().aux(i)] == 1; }
    if (!auxOk)
    {
        std::cerr << "testConditionalDensity: moved result or auxiliary data is wrong\n";
        ++failures;
    }

    // Errors: nothing fixed, nothing kept, bad marginalDims, bad
    // massTol, a reduced bandwidth, and x0 far from every point
    int thrown = 0;
    try { (void)kde.conditional(std::vector<double>{nan, nan, nan, nan}, {}, 0.0, bw); } catch (const std::invalid_argument&) { ++thrown; }
    try { (void)kde.conditional(std::vector<double>{0.0, 0.0, 0.0, 0.0}, {}, 0.0, bw); } catch (const std::invalid_argument&) { ++thrown; }
    try { (void)kde.conditional(std::vector<double>{0.0, nan, 0.0, 0.0}, {4}, 0.0, bw); } catch (const std::invalid_argument&) { ++thrown; }
    try { (void)kde.conditional(std::vector<double>{0.0, nan, 0.0, 0.0}, {3, 3}, 0.0, bw); } catch (const std::invalid_argument&) { ++thrown; }
    try { (void)kde.conditional(std::vector<double>{0.0, nan, 0.0, 0.0}, {}, 1.0, bw); } catch (const std::invalid_argument&) { ++thrown; }
    try { (void)kde.conditional(std::vector<double>{0.0, nan, 0.0, 0.0}, {}, 0.0, bw.reduce({false, false, false, true})); } catch (const std::invalid_argument&) { ++thrown; }
    try { (void)kde.conditional(std::vector<double>{1e3, nan, 1e3, 1e3}, {}, 0.0, bw); } catch (const std::domain_error&) { ++thrown; }
    if (thrown != 7)
    {
        std::cerr << "testConditionalDensity: only " << thrown << " of 7 invalid conditionals threw\n";
        ++failures;
    }
    return failures;
}

/**
 * @brief Check the batch versions of pdf() and conditional(), with and without measurement errors
 * @return Number of failures
 * @details
 * Each batch result is compared with the single-point call it stands
 * for: exactly, where the batch does the same arithmetic, and to a
 * tight tolerance where it reduces the bandwidth before convolving with
 * the errors rather than after. Also checks that invalid errors are
 * rejected, including an exception thrown inside the parallel loop.
 */
inline auto testKernelDensityBatch() -> int
{
    int failures = 0;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    std::mt19937 gen(71); // NOLINT(cert-msc32-c,cert-msc51-cpp) -- fixed seed for a reproducible test
    std::normal_distribution<double> gauss(0.0, 1.0);
    KDETestData owned = makeKDETestData(3000, 4, 2, 72);
    auto data = owned.view();
    const KDETest kde(data, 8);
    const Eigen::MatrixXd corr = (0.2 * Eigen::MatrixXd::Identity(4, 4)) +
        (0.8 * Eigen::MatrixXd::Ones(4, 4));
    const kden::Bandwidth bw(4.0 * Eigen::MatrixXd(corr.inverse()));
    auto fail = [&failures](const std::string& what) -> void
    {
        std::cerr << "testKernelDensityBatch: " << what << "\n";
        ++failures;
    };

    // Points sharing a pattern of NaNs (dimensions 0 and 2 specified),
    // and points with mixed patterns; each with its own errors in 0 and 2
    constexpr std::size_t nPt = 40;
    std::vector<std::vector<double>> shared;
    std::vector<std::vector<double>> mixed;
    std::vector<Eigen::MatrixXd> errCov;
    std::vector<kden::Bandwidth> perPoint;
    const std::vector<std::size_t> errDims{2, 0};
    for (std::size_t i = 0; i < nPt; ++i)
    {
        const double a = 1.5 * gauss(gen);
        const double b = 1.5 * gauss(gen);
        shared.push_back({a, nan, b, nan});
        mixed.push_back(i % 2 == 0 ? std::vector<double>{a, nan, b, nan} :
            std::vector<double>{a, gauss(gen), b, nan});
        const double s0 = 0.05 + (0.3 * std::abs(gauss(gen)));
        const double s2 = 0.05 + (0.3 * std::abs(gauss(gen)));
        Eigen::MatrixXd c(2, 2);
        c << s2 * s2, 0.5 * s0 * s2, 0.5 * s0 * s2, s0 * s0;
        errCov.push_back(c);
        perPoint.push_back(kden::Bandwidth((1.0 + (0.1 * static_cast<double>(i))) * Eigen::MatrixXd(corr.inverse())));
    }
    auto embedded = [](const Eigen::MatrixXd& c) -> Eigen::MatrixXd
    {
        Eigen::MatrixXd full = Eigen::MatrixXd::Zero(4, 4);
        full(2, 2) = c(0, 0);
        full(2, 0) = c(0, 1);
        full(0, 2) = c(1, 0);
        full(0, 0) = c(1, 1);
        return full;
    };

    // pdf: shared and per-point bandwidths, the Euclidean metric
    for (const auto* pts : {&shared, &mixed})
    {
        const auto got = kde.pdf(*pts, 1e-3, 0.0, bw);
        const auto gotEuclid = kde.pdf(*pts, 1e-3, 0.0);
        const auto gotEach = kde.pdf(*pts, 1e-3, 0.0, perPoint);
        bool ok = got.size() == nPt && gotEuclid.size() == nPt && gotEach.size() == nPt;
        for (std::size_t i = 0; ok && i < nPt; ++i)
        {
            ok = got[i] == kde.pdf((*pts)[i], 1e-3, 0.0, bw) &&
                gotEuclid[i] == kde.pdf((*pts)[i], 1e-3, 0.0) &&
                gotEach[i] == kde.pdf((*pts)[i], 1e-3, 0.0, perPoint[i]);
        }
        if (!ok) { fail(std::string("batch pdf differs from single-point pdf for ") + (pts == &shared ? "shared" : "mixed") + " NaN patterns"); }
    }

    // pdf with errors: each point's kernel convolved with its own errors
    for (const auto* pts : {&shared, &mixed})
    {
        const auto got = kde.pdf(*pts, errDims, errCov, 0.0, 0.0, bw);
        bool ok = got.size() == nPt;
        for (std::size_t i = 0; ok && i < nPt; ++i)
        {
            const double want = kde.pdf((*pts)[i], 0.0, 0.0, bw.convolveCov(embedded(errCov[i])));
            ok = std::abs(got[i] - want) <= 1e-10 * want;
        }
        if (!ok) { fail(std::string("pdf with errors differs from pdf with the convolved bandwidth for ") + (pts == &shared ? "shared" : "mixed") + " NaN patterns"); }
    }

    // conditional, with and without errors, and the result's batch pdf
    {
        const auto cds = kde.conditional(shared, {3}, 0.0, bw);
        const auto cdsErr = kde.conditional(shared, errDims, errCov, {3}, 0.0, bw);
        const auto cdsEuclid = kde.conditional(shared, {3}, 0.0);
        bool ok = cds.size() == nPt && cdsErr.size() == nPt && cdsEuclid.size() == nPt;
        const std::vector<std::vector<double>> ys{{0.1}, {-0.7}, {1.2}};
        for (std::size_t i = 0; ok && i < nPt; ++i)
        {
            const auto one = kde.conditional(shared[i], {3}, 0.0, bw);
            const auto oneErr = kde.conditional(shared[i], {3}, 0.0, bw.convolveCov(embedded(errCov[i])));
            const auto oneEuclid = kde.conditional(shared[i], {3}, 0.0);
            ok = cds[i].marginal() == one.marginal() && cds[i].dims() == one.dims() &&
                cds[i].data().x_.size() == one.data().x_.size() &&
                std::ranges::equal(cds[i].data().x_, one.data().x_) &&
                cdsEuclid[i].marginal() == oneEuclid.marginal() &&
                cdsErr[i].marginal() == oneErr.marginal() &&
                std::ranges::equal(cdsErr[i].data().x_, oneErr.data().x_);
            // Each result's weights are normalized by a sum that runs
            // serially inside the batch's parallel loop, but may run in
            // parallel, in a different order, for a single call: equal
            // to rounding
            const auto py = cds[i].pdf(ys, 0.0, 0.0);
            for (std::size_t j = 0; ok && j < ys.size(); ++j)
            {
                const double want = one.pdf(ys[j], 0.0, 0.0);
                ok = std::abs(py[j] - want) <= 1e-12 * want;
            }
        }
        if (!ok) { fail("batch conditional differs from single-point conditional"); }
    }

    // Invalid errors
    int thrown = 0;
    auto expectThrow = [&thrown](auto&& f) -> void
    {
        try { f(); } catch (const std::invalid_argument&) { ++thrown; } catch (const std::domain_error&) { ++thrown; }
    };
    const std::vector<Eigen::MatrixXd> tooFew(errCov.begin(), errCov.end() - 1);
    std::vector<Eigen::MatrixXd> wrongSize = errCov;
    wrongSize[3] = Eigen::MatrixXd::Identity(3, 3);
    std::vector<Eigen::MatrixXd> notPsd = errCov;
    notPsd[nPt - 1] = -Eigen::MatrixXd::Identity(2, 2);
    expectThrow([&] { (void)kde.pdf(shared, errDims, tooFew, 0.0, 0.0, bw); });
    expectThrow([&] { (void)kde.pdf(shared, errDims, wrongSize, 0.0, 0.0, bw); });
    expectThrow([&] { (void)kde.pdf(shared, {0, 1}, errCov, 0.0, 0.0, bw); });   // 1 is NaN
    expectThrow([&] { (void)kde.pdf(shared, {0, 0}, errCov, 0.0, 0.0, bw); });   // repeated
    expectThrow([&] { (void)kde.pdf(shared, {0, 4}, errCov, 0.0, 0.0, bw); });   // out of range
    expectThrow([&] { (void)kde.pdf(shared, errDims, notPsd, 0.0, 0.0, bw); });  // thrown in the loop
    expectThrow([&] { (void)kde.pdf(shared, 0.0, 0.0, std::vector<kden::Bandwidth>(perPoint.begin(), perPoint.end() - 1)); });
    expectThrow([&] { (void)kde.conditional(shared, {0, 3}, std::vector<Eigen::MatrixXd>(nPt, Eigen::MatrixXd::Identity(2, 2)), {3}, 0.0, bw); }); // marginalized
    expectThrow([&] { (void)kde.conditional(shared, errDims, notPsd, {3}, 0.0, bw); });
    if (thrown != 9)
    {
        fail("only " + std::to_string(thrown) + " of 9 invalid batch calls threw");
    }
    return failures;
}

/**
 * @brief Check conditionalSample(): the points and weights conditional() keeps, single-point and batch
 * @return Number of failures
 * @details
 * The sample's points are those of the ConditionalDensity (which
 * reorders them building its tree), a KernelDensity over them with the
 * conditional metric reproduces it, and its weights' sum gives the
 * marginal density. The batch versions, with and without errors, give
 * exactly what the single-point version gives.
 */
inline auto testConditionalSample() -> int
{
    int failures = 0;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    KDETestData owned = makeKDETestData(3000, 4, 2, 81);
    auto data = owned.view();
    const KDETest kde(data, 8);
    const Eigen::MatrixXd corr = (0.2 * Eigen::MatrixXd::Identity(4, 4)) +
        (0.8 * Eigen::MatrixXd::Ones(4, 4));
    const kden::Bandwidth bw(4.0 * Eigen::MatrixXd(corr.inverse()));
    auto fail = [&failures](const std::string& what) -> void
    {
        std::cerr << "testConditionalSample: " << what << "\n";
        ++failures;
    };
    // Rows of a point-major array, sorted, to compare sets of points
    auto sortedRows = [](const std::span<const double> x, const std::size_t n) -> std::vector<std::vector<double>>
    {
        std::vector<std::vector<double>> rows;
        for (std::size_t i = 0; i < x.size() / n; ++i) { rows.emplace_back(x.begin() + static_cast<std::ptrdiff_t>(i * n), x.begin() + static_cast<std::ptrdiff_t>((i + 1) * n)); }
        std::ranges::sort(rows);
        return rows;
    };

    // Fix 0 and 2, keep 1, marginalize 3; exact and pruned
    const std::vector<double> x0{0.4, nan, -0.3, 0.0};
    const std::vector<std::size_t> marginalDims{3};
    for (const double massTol : {0.0, 1e-2})
    {
        const std::string what = "massTol = " + std::to_string(massTol);
        const auto [x, w, aux] = kde.conditionalSample(x0, marginalDims, massTol, bw);
        const auto cd = kde.conditional(x0, marginalDims, massTol, bw);
        if (w.empty() || x.size() != w.size() || w.size() != cd.data().size() || aux.size() != w.size() ||
            sortedRows(x, 1) != sortedRows(cd.data().x_, 1) || !std::ranges::is_sorted(w))
        {
            fail(what + ": the sample's points differ from the conditional estimate's");
            continue;
        }

        // Each point's auxiliary data is that of the same point in the
        // conditional estimate (and so of the original point it came from)
        std::vector<std::pair<std::size_t, double>> fromSample;
        std::vector<std::pair<std::size_t, double>> fromCd;
        for (std::size_t i = 0; i < w.size(); ++i)
        {
            fromSample.emplace_back(aux[i], x[i]);
            fromCd.emplace_back(cd.data().aux_[i], cd.data().x_[i]);
        }
        std::ranges::sort(fromSample);
        std::ranges::sort(fromCd);
        if (fromSample != fromCd)
        {
            fail(what + ": the sample's auxiliary data do not match the conditional estimate's");
        }

        // A KernelDensity over the sample, with the conditional metric,
        // is the conditional estimate
        kden::OwnedKernelDensityData<std::size_t> sample;
        sample.nDim_ = 1;
        sample.x_ = x;
        sample.w_ = w;
        sample.aux_.resize(w.size());
        auto view = sample.view();
        const KDETest rebuilt(view, 8, cd.bandwidth());
        bool ok = true;
        for (const double y : {-1.0, 0.0, 0.5, 2.0})
        {
            const double want = cd.pdf(std::vector<double>{y}, 0.0, 0.0);
            const double got = rebuilt.pdf(std::vector<double>{y}, 0.0, 0.0, cd.bandwidth());
            ok = ok && std::abs(got - want) <= 1e-12 * want;
        }
        if (!ok) { fail(what + ": an estimate rebuilt from the sample differs from the conditional estimate"); }

        // The weights' sum gives the marginal density over the fixed dimensions
        const kden::Bandwidth s = bw.reduce({false, true, false, true});
        double wSum = 0.0;
        for (const double wi : w) { wSum += wi; }
        const double marginal = wSum * std::sqrt(s.det()) / std::numbers::pi;
        if (std::abs(marginal - cd.marginal()) > 1e-12 * cd.marginal())
        {
            fail(what + ": the weights' sum does not give the marginal density");
        }
    }

    // Batches, with and without errors, and the Euclidean metric
    const std::vector<std::vector<double>> pts{{0.4, nan, -0.3, 0.0}, {-0.8, nan, 0.6, 0.0}, {1.1, nan, 1.4, 0.0}};
    const std::vector<std::size_t> errDims{0, 2};
    const std::vector<Eigen::MatrixXd> errCov{0.01 * Eigen::MatrixXd::Identity(2, 2),
        0.04 * Eigen::MatrixXd::Identity(2, 2), 0.09 * Eigen::MatrixXd::Identity(2, 2)};
    const auto batch = kde.conditionalSample(pts, marginalDims, 1e-2, bw);
    const auto batchErr = kde.conditionalSample(pts, errDims, errCov, marginalDims, 1e-2, bw);
    const auto batchEuclid = kde.conditionalSample(pts, marginalDims, 1e-2);
    bool ok = batch.size() == 3 && batchErr.size() == 3 && batchEuclid.size() == 3;
    for (std::size_t i = 0; ok && i < pts.size(); ++i)
    {
        Eigen::MatrixXd full = Eigen::MatrixXd::Zero(4, 4);
        full(0, 0) = errCov[i](0, 0);
        full(2, 2) = errCov[i](1, 1);
        ok = batch[i] == kde.conditionalSample(pts[i], marginalDims, 1e-2, bw) &&
            batchErr[i] == kde.conditionalSample(pts[i], marginalDims, 1e-2, bw.convolveCov(full)) &&
            batchEuclid[i] == kde.conditionalSample(pts[i], marginalDims, 1e-2);
    }
    if (!ok) { fail("batch conditionalSample differs from the single-point version"); }

    // Without auxiliary data, the third element is an empty placeholder
    kden::OwnedKernelDensityData<> plain;
    plain.nDim_ = owned.nDim_;
    plain.x_ = owned.x_;
    plain.w_ = owned.w_;
    auto plainView = plain.view();
    const kden::KernelDensity<> plainKde(plainView, 8);
    const auto [px, pw, paux] = plainKde.conditionalSample(x0, marginalDims, 0.0, bw);
    static_assert(std::is_same_v<std::remove_cvref_t<decltype(paux)>, std::monostate>);
    const auto [cx, cw, caux] = kde.conditionalSample(x0, marginalDims, 0.0, bw);
    if (sortedRows(px, 1) != sortedRows(cx, 1)) { fail("conditionalSample without auxiliary data keeps different points"); }
    return failures;
}

/**
 * @brief Check marginalize(): the marginal estimate over the remaining dimensions
 * @return Number of failures
 * @details
 * The marginal's pdf() equals the original's pdf() with NaN in the
 * marginalized dimensions; it keeps every point, weight, and
 * auxiliary datum; marginalizing over nothing reproduces the original;
 * and invalid dimensions are rejected.
 */
inline auto testKernelDensityMarginalize() -> int
{
    int failures = 0;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    KDETestData owned = makeKDETestData(3000, 4, 2, 91);
    auto data = owned.view();
    const KDETest kde(data, 8);
    const Eigen::MatrixXd corr = (0.2 * Eigen::MatrixXd::Identity(4, 4)) +
        (0.8 * Eigen::MatrixXd::Ones(4, 4));
    const kden::Bandwidth bw(4.0 * Eigen::MatrixXd(corr.inverse()));
    std::mt19937 gen(92); // NOLINT(cert-msc32-c,cert-msc51-cpp) -- fixed seed for a reproducible test
    std::normal_distribution<double> gauss(0.0, 1.5);

    for (const std::vector<std::size_t>& g : {std::vector<std::size_t>{3}, std::vector<std::size_t>{0, 2},
        std::vector<std::size_t>{}})
    {
        std::string what = "marginalDims = {";
        for (const std::size_t d : g) { what += std::to_string(d) + " "; }
        what += "}";
        const auto marg = kde.marginalize(g, bw);
        std::vector<std::size_t> wantDims;
        for (std::size_t d = 0; d < 4; ++d)
        {
            if (std::ranges::find(g, d) == g.end()) { wantDims.push_back(d); }
        }
        bool ok = marg.dims() == wantDims && marg.marginal() == 1.0 && marg.data().size() == data.size();

        // Every point, weight, and auxiliary datum is kept: match them by
        // aux (each point's original index, so unique). The weights are
        // renormalized building the result, so equal to rounding.
        std::vector<std::tuple<std::size_t, std::vector<double>, double>> fromMarg;
        std::vector<std::tuple<std::size_t, std::vector<double>, double>> fromOrig;
        for (std::size_t i = 0; ok && i < data.size(); ++i)
        {
            std::vector<double> xm(marg.data().position(i).begin(), marg.data().position(i).end());
            fromMarg.emplace_back(marg.data().aux(i), xm, marg.data().w_[i]);
            std::vector<double> xo;
            for (const std::size_t d : wantDims) { xo.push_back(data.position(i)[d]); }
            fromOrig.emplace_back(data.aux(i), xo, data.w_[i]);
        }
        std::ranges::sort(fromMarg);
        std::ranges::sort(fromOrig);
        for (std::size_t q = 0; ok && q < fromMarg.size(); ++q)
        {
            const auto& [am, xm, wm] = fromMarg[q];
            const auto& [ao, xo, wo] = fromOrig[q];
            ok = am == ao && xm == xo && std::abs(wm - wo) <= 1e-12 * wo;
        }

        // Its density is the original's with NaN in the marginalized dimensions
        for (int t = 0; ok && t < 5; ++t)
        {
            std::vector<double> y;
            std::vector<double> full(4, nan);
            for (const std::size_t d : wantDims)
            {
                y.push_back(gauss(gen));
                full[d] = y.back();
            }
            const double want = kde.pdf(full, 0.0, 0.0, bw);
            const double got = marg.pdf(y, 0.0, 0.0);
            ok = std::abs(got - want) <= 1e-10 * want;
        }
        if (!ok)
        {
            std::cerr << "testKernelDensityMarginalize: " << what << ": the marginal estimate is wrong\n";
            ++failures;
        }
    }

    // The Euclidean overload
    const auto margE = kde.marginalize({1});
    const std::vector<double> y{0.3, -0.2, 0.5};
    const double wantE = kde.pdf(std::vector<double>{0.3, nan, -0.2, 0.5}, 0.0, 0.0);
    if (std::abs(margE.pdf(y, 0.0, 0.0) - wantE) > 1e-10 * wantE)
    {
        std::cerr << "testKernelDensityMarginalize: the Euclidean marginal estimate is wrong\n";
        ++failures;
    }

    // Invalid dimensions or bandwidth
    int thrown = 0;
    auto expectThrow = [&thrown](auto&& f) -> void
    {
        try { f(); } catch (const std::invalid_argument&) { ++thrown; }
    };
    expectThrow([&] { (void)kde.marginalize({4}, bw); });
    expectThrow([&] { (void)kde.marginalize({1, 1}, bw); });
    expectThrow([&] { (void)kde.marginalize({0, 1, 2, 3}, bw); });
    expectThrow([&] { (void)kde.marginalize({1}, kden::Bandwidth::identity(3)); });
    expectThrow([&] { (void)kde.marginalize({1}, bw.reduce({true, false, false, false})); });
    if (thrown != 5)
    {
        std::cerr << "testKernelDensityMarginalize: only " << thrown << " of 5 invalid calls threw\n";
        ++failures;
    }
    return failures;
}

/**
 * @brief The marginal cumulative distribution of a kernel density estimate in one dimension, by brute force
 * @param data The estimate's points (weights normalized)
 * @param dim The dimension
 * @param m The marginal kernel's metric in dim
 * @param t Where to evaluate it
 * @return sum_i w_i erfc(-sqrt(m) (t - x_i)) / 2
 */
inline auto bruteCdf(const KDETestView& data, const std::size_t dim, const double m, const double t) -> double
{
    double f = 0.0;
    for (std::size_t i = 0; i < data.size(); ++i)
    {
        f += data.w_[i] * 0.5 * std::erfc(-std::sqrt(m) * (t - data.position(i)[dim]));
    }
    return f;
}

/**
 * @brief Check percentile(): the guaranteed tolerance in probability, against a brute-force cumulative distribution
 * @return Number of failures
 * @details
 * For several dimensions and percentiles (including both tails) and
 * tolerances (loose, tight, and zero), the brute-force cumulative
 * distribution at the result must be within the tolerance of the
 * requested fraction. Also checks a ConditionalDensity's percentiles,
 * the batch version, and argument checks.
 */
inline auto testKernelDensityPercentile() -> int
{
    int failures = 0;
    KDETestData owned = makeKDETestData(4000, 4, 2, 101);
    auto data = owned.view();
    const KDETest kde(data, 8);
    const Eigen::MatrixXd corr = (0.2 * Eigen::MatrixXd::Identity(4, 4)) +
        (0.8 * Eigen::MatrixXd::Ones(4, 4));
    const kden::Bandwidth bw(16.0 * Eigen::MatrixXd(corr.inverse()));
    const Eigen::MatrixXd cov = bw.h2Inv().inverse();
    const std::vector<double> pcts{0.1, 2.5, 16.0, 50.0, 84.0, 97.5, 99.9};

    for (const auto& [relTol, absTol] : {std::pair{1e-2, 1e-8}, std::pair{0.0, 1e-10}, std::pair{0.0, 0.0}})
    {
        for (std::size_t dim = 0; dim < 4; ++dim)
        {
            const double m = 1.0 / cov(static_cast<Eigen::Index>(dim), static_cast<Eigen::Index>(dim));
            for (const double pct : pcts)
            {
                const double p = pct / 100.0;
                const double tol = std::max({absTol, relTol * std::min(p, 1.0 - p), 1e-13});
                const double t = kde.percentile(dim, pct, relTol, absTol, bw);
                const double f = bruteCdf(data, dim, m, t);
                if (!(std::abs(f - p) <= tol))
                {
                    std::cerr << "testKernelDensityPercentile: dim " << dim << ", pct " << pct << ", relTol "
                        << relTol << ", absTol " << absTol << ": F(" << t << ") = " << f << ", off by "
                        << f - p << " > " << tol << "\n";
                    ++failures;
                }
            }
        }
    }

    // The batch version gives what single calls do
    const auto batch = kde.percentile(1, pcts, 1e-3, 1e-10, bw);
    bool same = batch.size() == pcts.size();
    for (std::size_t i = 0; same && i < pcts.size(); ++i) { same = batch[i] == kde.percentile(1, pcts[i], 1e-3, 1e-10, bw); }
    if (!same)
    {
        std::cerr << "testKernelDensityPercentile: batch percentiles differ from single ones\n";
        ++failures;
    }

    // A conditional density's percentiles, in its own metric
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const auto cd = kde.conditional(std::vector<double>{0.5, nan, -0.4, nan}, {}, 1e-3, bw);
    const Eigen::MatrixXd cdCov = cd.bandwidth().h2Inv().inverse();
    KDETestView cdView;
    cdView.nDim_ = cd.data().nDim_;
    cdView.x_ = std::span<double>(const_cast<double*>(cd.data().x_.data()), cd.data().x_.size()); // NOLINT(cppcoreguidelines-pro-type-const-cast) -- read only, for bruteCdf
    cdView.w_ = std::span<double>(const_cast<double*>(cd.data().w_.data()), cd.data().w_.size()); // NOLINT(cppcoreguidelines-pro-type-const-cast)
    for (std::size_t dim = 0; dim < 2; ++dim)
    {
        const double m = 1.0 / cdCov(static_cast<Eigen::Index>(dim), static_cast<Eigen::Index>(dim));
        for (const double pct : {5.0, 50.0, 95.0})
        {
            const double t = cd.percentile(dim, pct, 0.0, 1e-9);
            const double f = bruteCdf(cdView, dim, m, t);
            if (!(std::abs(f - (pct / 100.0)) <= 1e-9))
            {
                std::cerr << "testKernelDensityPercentile: conditional density, dim " << dim << ", pct " << pct
                    << ": F = " << f << "\n";
                ++failures;
            }
        }
    }

    // Argument checks
    int thrown = 0;
    auto expectThrow = [&thrown](auto&& f) -> void
    {
        try { f(); } catch (const std::invalid_argument&) { ++thrown; }
    };
    expectThrow([&] { (void)kde.percentile(4, 50.0, 1e-2, 1e-8, bw); });
    expectThrow([&] { (void)kde.percentile(0, 0.0, 1e-2, 1e-8, bw); });
    expectThrow([&] { (void)kde.percentile(0, 100.0, 1e-2, 1e-8, bw); });
    expectThrow([&] { (void)kde.percentile(0, nan, 1e-2, 1e-8, bw); });
    expectThrow([&] { (void)kde.percentile(0, 50.0, -1e-2, 1e-8, bw); });
    expectThrow([&] { (void)kde.percentile(0, 50.0, 1e-2, 1e-8, kden::Bandwidth::identity(3)); });
    expectThrow([&] { (void)kde.percentile(0, 50.0, 1e-2, 1e-8, bw.reduce({false, true, false, false})); });
    if (thrown != 7)
    {
        std::cerr << "testKernelDensityPercentile: only " << thrown << " of 7 invalid calls threw\n";
        ++failures;
    }
    return failures;
}

#endif // TESTKERNELDENSITY_HPP
