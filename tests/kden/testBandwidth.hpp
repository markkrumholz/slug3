/**
 * @file testBandwidth.hpp
 * @author Mark Krumholz
 * @brief Unit tests for kden::Bandwidth
 * @details
 * Covers distances (with and without unspecified coordinates, for
 * diagonal and full metrics), reduction (against an independent
 * inverse-covariance calculation, and composed reductions), and the box
 * bound (never above the true minimum distance to a box, and exact for
 * diagonal metrics), plus argument checking.
 * @date 2026-10-15
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#ifndef TESTBANDWIDTH_HPP
#define TESTBANDWIDTH_HPP

#include "../../src/kden/Bandwidth.hpp"
#include <Eigen/Dense> // NOLINT(misc-include-cleaner)
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

/**
 * @brief A random symmetric positive-definite matrix
 * @param n Size
 * @param gen Random generator
 * @return A M^T + 0.1 I, with A an n x n matrix of unit Gaussians
 */
inline auto randomSPD(const std::size_t n, std::mt19937& gen) -> Eigen::MatrixXd
{
    std::normal_distribution<double> gauss(0.0, 1.0);
    const auto ni = static_cast<Eigen::Index>(n);
    const Eigen::MatrixXd a = Eigen::MatrixXd::NullaryExpr(ni, ni, [&]() -> double { return gauss(gen); });
    return (a * a.transpose()) + (0.1 * Eigen::MatrixXd::Identity(ni, ni));
}

/**
 * @brief Check distances against hand-worked values
 * @return Number of failures
 */
inline auto testBandwidthDist2() -> int
{
    int failures = 0;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    auto check = [&failures](const std::string& what, const double got, const double want) -> void
    {
        if (std::abs(got - want) > 1e-12 * (1.0 + std::abs(want)))
        {
            std::cerr << "testBandwidthDist2: " << what << ": expected " << want << ", got " << got << "\n";
            ++failures;
        }
    };
    Eigen::MatrixXd full(2, 2);
    full << 2.0, 0.5, 0.5, 1.0;
    Eigen::MatrixXd diag(2, 2);
    diag << 2.0, 0.0, 0.0, 1.0;
    const kden::Bandwidth fullBw(full);
    const kden::Bandwidth diagBw(diag);
    const kden::Bandwidth euclid = kden::Bandwidth::identity(2);
    const std::vector<double> a{1.0, 2.0};
    const std::vector<double> b{4.0, 6.0};  // d = (-3, -4)

    if (!euclid.isDiagonal() || !diagBw.isDiagonal() || fullBw.isDiagonal())
    {
        std::cerr << "testBandwidthDist2: diagonal detection is wrong\n";
        ++failures;
    }
    check("Euclidean", euclid.dist2(a, b), 25.0);
    check("diagonal", diagBw.dist2(a, b), 34.0);
    check("full", fullBw.dist2(a, b), 46.0);
    check("self", fullBw.dist2(a, a), 0.0);
    check("Euclidean, x1[1] unspecified", euclid.dist2(std::vector<double>{1.0, nan}, b), 9.0);
    check("diagonal, x2[0] unspecified", diagBw.dist2(a, std::vector<double>{nan, 6.0}), 16.0);
    check("disjoint unspecified", fullBw.dist2(std::vector<double>{1.0, nan}, std::vector<double>{nan, 6.0}), 0.0);
    const kden::Bandwidth reduced = fullBw.reduce({false, true});
    check("full, reduced: 9 * (2 - 0.5^2)", reduced.dist2(std::vector<double>{1.0, nan}, b), 15.75);
    if (!reduced.isReduced() || !reduced.isDiagonal() || !reduced.matches(std::vector<double>{1.0, nan}) || reduced.matches(a))
    {
        std::cerr << "testBandwidthDist2: reduced bandwidth reports the wrong state\n";
        ++failures;
    }

    // The scratch-space overload agrees with the allocating one
    std::vector<double> scratch;
    check("scratch overload", fullBw.dist2(a, b, scratch), fullBw.dist2(a, b));
    return failures;
}

/**
 * @brief Check reduce() against the inverse of the covariance's specified block
 * @return Number of failures
 */
inline auto testBandwidthReduce() -> int
{
    int failures = 0;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    std::mt19937 gen(5); // NOLINT(cert-msc32-c,cert-msc51-cpp) -- fixed seed for a reproducible test
    std::normal_distribution<double> gauss(0.0, 1.0);
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    for (int trial = 0; trial < 500; ++trial)
    {
        const std::size_t n = 2 + static_cast<std::size_t>(trial % 6);
        const Eigen::MatrixXd m = randomSPD(n, gen);
        const kden::Bandwidth bw(m);
        std::vector<double> x1(n);
        std::vector<double> x2(n);
        std::vector<bool> unspec(n, false);
        std::vector<Eigen::Index> s;
        for (std::size_t i = 0; i < n; ++i)
        {
            x1[i] = gauss(gen);
            x2[i] = gauss(gen);
            if (unif(gen) < 0.35) { unspec[i] = true; x1[i] = nan; }
            else { s.push_back(static_cast<Eigen::Index>(i)); }
        }
        const kden::Bandwidth red = bw.reduce(unspec);
        const double got = red.dist2(x1, x2);
        double want = 0.0;
        if (!s.empty())
        {
            const Eigen::MatrixXd sigma = m.inverse();
            const auto ns = static_cast<Eigen::Index>(s.size());
            Eigen::MatrixXd sigmaSS(ns, ns);
            Eigen::VectorXd ds(ns);
            for (Eigen::Index i = 0; i < ns; ++i)
            {
                const auto si = static_cast<std::size_t>(s[static_cast<std::size_t>(i)]);
                ds(i) = x1[si] - x2[si];
                for (Eigen::Index j = 0; j < ns; ++j) { sigmaSS(i, j) = sigma(s[static_cast<std::size_t>(i)], s[static_cast<std::size_t>(j)]); }
            }
            want = ds.dot(sigmaSS.inverse() * ds);
        }
        if (std::abs(got - want) > 1e-9 * (1.0 + std::abs(want)))
        {
            std::cerr << "testBandwidthReduce: disagrees with the inverse covariance block: "
                << got << " vs " << want << "\n";
            ++failures;
            break;
        }
    }

    // Reducing in two steps agrees with reducing in one
    const Eigen::MatrixXd m = randomSPD(5, gen);
    const kden::Bandwidth bw(m);
    const kden::Bandwidth once = bw.reduce({true, false, true, false, false});
    const kden::Bandwidth twice = bw.reduce({true, false, false, false, false}).reduce({true, false, true, false, false});
    if (!once.h2Inv().isApprox(twice.h2Inv(), 1e-10))
    {
        std::cerr << "testBandwidthReduce: two-step reduction differs from one-step\n";
        ++failures;
    }

    // Errors: wrong size, un-reducing a dimension, a non-positive-definite block
    int thrown = 0;
    try { (void)bw.reduce({true, false}); } catch (const std::invalid_argument&) { ++thrown; }
    try { (void)once.reduce({false, false, true, false, false}); } catch (const std::invalid_argument&) { ++thrown; }
    Eigen::MatrixXd notPD(2, 2);
    notPD << 1.0, 2.0, 2.0, 1.0;
    try { (void)kden::Bandwidth(notPD); } catch (const std::domain_error&) { ++thrown; }
    Eigen::MatrixXd asym(2, 2);
    asym << 1.0, 0.5, 0.0, 1.0;
    try { (void)kden::Bandwidth(asym); } catch (const std::invalid_argument&) { ++thrown; }
    if (thrown != 4)
    {
        std::cerr << "testBandwidthReduce: only " << thrown << " of 4 invalid operations threw\n";
        ++failures;
    }
    return failures;
}

/**
 * @brief Check the box bound: never above the true distance, exact for diagonal metrics
 * @return Number of failures
 */
inline auto testBandwidthBoxBound() -> int
{
    int failures = 0;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    std::mt19937 gen(8); // NOLINT(cert-msc32-c,cert-msc51-cpp) -- fixed seed for a reproducible test
    std::normal_distribution<double> gauss(0.0, 1.0);
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    for (int trial = 0; trial < 400; ++trial)
    {
        const std::size_t n = 2 + static_cast<std::size_t>(trial % 7);
        const bool diagonal = trial % 3 == 0;
        Eigen::MatrixXd m = randomSPD(n, gen);
        if (diagonal) { m = Eigen::MatrixXd(m.diagonal().asDiagonal()); }
        kden::Bandwidth bw(m);
        std::vector<double> x(n);
        std::vector<double> lo(n);
        std::vector<double> hi(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            x[i] = 3.0 * gauss(gen);
            const double c = gauss(gen);
            const double h = unif(gen);
            lo[i] = c - h;
            hi[i] = c + h;
        }
        if (trial % 4 == 1) { x[0] = nan; bw = bw.reduce(kden::Bandwidth::nanPattern(x)); }
        const auto bound = bw.boxBounder(x);
        const double b = bound(lo, hi);

        // The bound must not exceed the distance to any point in the box
        double minSampled = std::numeric_limits<double>::infinity();
        std::vector<double> y(n);
        for (int sample = 0; sample < 2000; ++sample)
        {
            for (std::size_t i = 0; i < n; ++i)
            {
                // Mix interior points with points on the box's faces and corners
                const double r = unif(gen);
                y[i] = r < 0.25 ? lo[i] : (r < 0.5 ? hi[i] : lo[i] + ((hi[i] - lo[i]) * unif(gen)));
            }
            minSampled = std::min(minSampled, bw.dist2(x, y));
        }
        if (b > minSampled * (1.0 + 1e-12))
        {
            std::cerr << "testBandwidthBoxBound: bound " << b << " exceeds a sampled distance "
                << minSampled << " (n = " << n << ")\n";
            ++failures;
            break;
        }

        // For a diagonal metric, the bound is the exact distance to the box
        if (diagonal && trial % 4 != 1)
        {
            std::vector<double> nearest(n);
            for (std::size_t i = 0; i < n; ++i) { nearest[i] = std::clamp(x[i], lo[i], hi[i]); }
            const double exact = bw.dist2(x, nearest);
            if (std::abs(b - exact) > 1e-12 * (1.0 + exact))
            {
                std::cerr << "testBandwidthBoxBound: diagonal bound " << b << " is not the exact "
                    "box distance " << exact << "\n";
                ++failures;
                break;
            }
        }
    }
    return failures;
}

#endif // TESTBANDWIDTH_HPP
