/**
 * @file testBandwidth.hpp
 * @author Mark Krumholz
 * @brief Unit tests for kden::Bandwidth
 * @details
 * Covers distances (with and without unspecified coordinates, for
 * diagonal and full metrics), reduction (against an independent
 * inverse-covariance calculation, and composed reductions), and the box
 * bounds (the lower bound never above the true minimum distance to a box
 * and the upper never below the true maximum, both exact for diagonal
 * metrics), convolution (against the formula, reduction, and numerical
 * convolution) by another bandwidth or by a covariance matrix, plus
 * argument checking.
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
#include <cstdint>
#include <iostream>
#include <limits>
#include <numbers>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
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
        // (Exactly symmetric when computed; with nothing to reduce, reduce()
        // returns bw itself, whose randomSPD matrix need not be)
        if (red.isReduced() && red.h2Inv() != red.h2Inv().transpose())
        {
            std::cerr << "testBandwidthReduce: reduced matrix is not exactly symmetric\n";
            ++failures;
            break;
        }
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
 * @brief Check the box bounds: they bracket the true distances, and are exact for diagonal metrics
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
        const double b = bound.lower(lo, hi);
        const double u = bound.upper(lo, hi);

        // bounds() agrees with lower() and upper()
        const auto [bLo, bHi] = bound.bounds(lo, hi);
        if (bLo != b || bHi != u)
        {
            std::cerr << "testBandwidthBoxBound: bounds() gives (" << bLo << ", " << bHi
                << "), lower()/upper() give (" << b << ", " << u << ")\n";
            ++failures;
            break;
        }

        // The squared distance is a convex quadratic, so its maximum
        // over the box is at one of the 2^n corners
        double maxCorner = 0.0;
        std::vector<double> y(n);
        for (std::uint32_t corner = 0; corner < (1U << n); ++corner)
        {
            for (std::size_t i = 0; i < n; ++i) { y[i] = ((corner >> i) & 1U) != 0U ? hi[i] : lo[i]; }
            maxCorner = std::max(maxCorner, bw.dist2(x, y));
        }
        if (u < maxCorner * (1.0 - 1e-12))
        {
            std::cerr << "testBandwidthBoxBound: upper bound " << u << " is below the maximum "
                "distance to a corner " << maxCorner << " (n = " << n << ")\n";
            ++failures;
            break;
        }

        // The lower bound must not exceed the distance to any point in
        // the box, nor the upper bound fall below it
        double minSampled = std::numeric_limits<double>::infinity();
        double maxSampled = 0.0;
        for (int sample = 0; sample < 2000; ++sample)
        {
            for (std::size_t i = 0; i < n; ++i)
            {
                // Mix interior points with points on the box's faces and corners
                const double r = unif(gen);
                y[i] = r < 0.25 ? lo[i] : (r < 0.5 ? hi[i] : lo[i] + ((hi[i] - lo[i]) * unif(gen)));
            }
            const double d2 = bw.dist2(x, y);
            minSampled = std::min(minSampled, d2);
            maxSampled = std::max(maxSampled, d2);
        }
        if (b > minSampled * (1.0 + 1e-12))
        {
            std::cerr << "testBandwidthBoxBound: bound " << b << " exceeds a sampled distance "
                << minSampled << " (n = " << n << ")\n";
            ++failures;
            break;
        }
        if (u < maxSampled * (1.0 - 1e-12))
        {
            std::cerr << "testBandwidthBoxBound: upper bound " << u << " is below a sampled distance "
                << maxSampled << " (n = " << n << ")\n";
            ++failures;
            break;
        }

        // For a diagonal metric, the bounds are the exact distances to
        // the nearest point of the box and to its farthest corner
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
            if (std::abs(u - maxCorner) > 1e-12 * (1.0 + maxCorner))
            {
                std::cerr << "testBandwidthBoxBound: diagonal upper bound " << u << " is not the "
                    "exact farthest-corner distance " << maxCorner << "\n";
                ++failures;
                break;
            }
        }
    }
    return failures;
}

/**
 * @brief Check convolve() against the formula, reduction, and numerical convolution
 * @return Number of failures
 */
inline auto testBandwidthConvolve() -> int
{
    int failures = 0;
    std::mt19937 gen(12); // NOLINT(cert-msc32-c,cert-msc51-cpp) -- fixed seed for a reproducible test
    auto fail = [&failures](const std::string& what) -> void
    {
        std::cerr << "testBandwidthConvolve: " << what << "\n";
        ++failures;
    };

    // (M1^-1 + M2^-1)^-1, symmetric in its arguments, and commuting
    // with reduction (marginalizing commutes with adding covariances)
    for (int trial = 0; trial < 200; ++trial)
    {
        const std::size_t n = 2 + static_cast<std::size_t>(trial % 6);
        const Eigen::MatrixXd m1 = randomSPD(n, gen);
        const Eigen::MatrixXd m2 = randomSPD(n, gen);
        const kden::Bandwidth b1(m1);
        const kden::Bandwidth b2(m2);
        const kden::Bandwidth c = b1.convolve(b2);
        const Eigen::MatrixXd want = (m1.inverse() + m2.inverse()).inverse();
        if (!c.h2Inv().isApprox(want, 1e-9) || !b2.convolve(b1).h2Inv().isApprox(want, 1e-9))
        {
            fail("disagrees with (M1^-1 + M2^-1)^-1 for n = " + std::to_string(n));
            break;
        }
        if (c.h2Inv() != c.h2Inv().transpose())
        {
            fail("result is not exactly symmetric for n = " + std::to_string(n));
            break;
        }
        std::vector<bool> unspec(n, false);
        unspec[0] = true;
        const Eigen::MatrixXd viaReduced = b1.reduce(unspec).convolve(b2.reduce(unspec)).h2Inv();
        const Eigen::MatrixXd reducedAfter = c.reduce(unspec).h2Inv();
        if (!viaReduced.isApprox(reducedAfter, 1e-9))
        {
            fail("convolving reduced bandwidths differs from reducing the convolution");
            break;
        }
    }

    // Diagonal: h^2 add, so 1 / M adds; and a singular matrix (an
    // infinitely broad kernel) gives a singular result
    Eigen::MatrixXd d1 = Eigen::MatrixXd::Zero(2, 2);
    d1.diagonal() << 4.0, 1.0;
    Eigen::MatrixXd d2 = Eigen::MatrixXd::Zero(2, 2);
    d2.diagonal() << 4.0, 3.0;
    const kden::Bandwidth dc = kden::Bandwidth(d1).convolve(kden::Bandwidth(d2));
    if (!dc.isDiagonal() || std::abs(dc.h2Inv()(0, 0) - 2.0) > 1e-14 || std::abs(dc.h2Inv()(1, 1) - 0.75) > 1e-14)
    {
        fail("diagonal convolution is wrong");
    }
    Eigen::MatrixXd sing = Eigen::MatrixXd::Zero(2, 2);
    sing(0, 0) = 1.0;
    const kden::Bandwidth sc = kden::Bandwidth(sing).convolve(kden::Bandwidth::identity(2));
    if (std::abs(sc.h2Inv()(0, 0) - 0.5) > 1e-14 || sc.h2Inv().cwiseAbs().sum() - 0.5 > 1e-14)
    {
        fail("convolution with a singular bandwidth is wrong");
    }

    // The normalized kernels sqrt(det M) / pi exp(-d^T M d) convolve
    // to the normalized kernel of the result, in two dimensions
    Eigen::MatrixXd f1(2, 2);
    f1 << 2.0, 0.8, 0.8, 1.0;
    Eigen::MatrixXd f2(2, 2);
    f2 << 0.7, -0.3, -0.3, 1.5;
    const kden::Bandwidth k1(f1);
    const kden::Bandwidth k2(f2);
    const kden::Bandwidth k12 = k1.convolve(k2);
    auto kernel = [](const kden::Bandwidth& b, const double x, const double y) -> double
    {
        return std::sqrt(b.det()) / std::numbers::pi * std::exp(-b.dist2(std::vector<double>{x, y}, std::vector<double>{0.0, 0.0}));
    };
    const double lim = 8.0;
    const int nGrid = 400;
    const double dx = 2.0 * lim / nGrid;
    for (const auto& [x, y] : std::vector<std::pair<double, double>>{{0.0, 0.0}, {0.7, -0.4}, {-1.2, 1.5}})
    {
        double conv = 0.0;
        for (int i = 0; i < nGrid; ++i)
        {
            const double tx = -lim + ((i + 0.5) * dx);
            for (int j = 0; j < nGrid; ++j)
            {
                const double ty = -lim + ((j + 0.5) * dx);
                conv += kernel(k1, x - tx, y - ty) * kernel(k2, tx, ty) * dx * dx;
            }
        }
        const double want = kernel(k12, x, y);
        if (std::abs(conv - want) > 1e-8 * want)
        {
            fail("numerical convolution " + std::to_string(conv) + " differs from the convolved kernel " +
                std::to_string(want));
            break;
        }
    }

    // convolveCov(C): (M^-1 + 2 C)^-1; the same as convolving with
    // (2 C)^-1 when C is invertible; exact in the dimensions C leaves
    // alone when C is singular; and commuting with reduction
    for (int trial = 0; trial < 100; ++trial)
    {
        const std::size_t n = 2 + static_cast<std::size_t>(trial % 5);
        const auto ni = static_cast<Eigen::Index>(n);
        const Eigen::MatrixXd m = randomSPD(n, gen);
        const kden::Bandwidth b(m);
        Eigen::MatrixXd cov = randomSPD(n, gen);
        const Eigen::MatrixXd want = (m.inverse() + (2.0 * cov)).inverse();
        const kden::Bandwidth c = b.convolveCov(cov);
        const Eigen::MatrixXd viaConvolve = b.convolve(kden::Bandwidth(Eigen::MatrixXd(0.5 * cov.inverse()))).h2Inv();
        if (!c.h2Inv().isApprox(want, 1e-9) || !c.h2Inv().isApprox(viaConvolve, 1e-9) ||
            c.h2Inv() != c.h2Inv().transpose())
        {
            fail("convolveCov disagrees with (M^-1 + 2 C)^-1 for n = " + std::to_string(n));
            break;
        }

        // Errors in only the first dimension: a singular C
        Eigen::MatrixXd single = Eigen::MatrixXd::Zero(ni, ni);
        single(0, 0) = 0.3;
        const Eigen::MatrixXd wantSingle = (m.inverse() + (2.0 * single)).inverse();
        if (!b.convolveCov(single).h2Inv().isApprox(wantSingle, 1e-9) ||
            !b.convolveCov(Eigen::MatrixXd::Zero(ni, ni)).h2Inv().isApprox(m, 1e-12))
        {
            fail("convolveCov is wrong for a singular covariance, n = " + std::to_string(n));
            break;
        }

        std::vector<bool> unspec(n, false);
        unspec[n - 1] = true;
        if (!b.reduce(unspec).convolveCov(cov).h2Inv().isApprox(c.reduce(unspec).h2Inv(), 1e-9))
        {
            fail("convolveCov does not commute with reduce, n = " + std::to_string(n));
            break;
        }
    }
    {
        int covThrown = 0;
        Eigen::MatrixXd asymCov(2, 2);
        asymCov << 1.0, 0.5, 0.0, 1.0;
        Eigen::MatrixXd negCov = -Eigen::MatrixXd::Identity(2, 2);
        Eigen::MatrixXd nanCov = Eigen::MatrixXd::Identity(2, 2);
        nanCov(0, 1) = nanCov(1, 0) = std::numeric_limits<double>::quiet_NaN();
        const kden::Bandwidth id2 = kden::Bandwidth::identity(2);
        try { (void)id2.convolveCov(Eigen::MatrixXd::Identity(3, 3)); } catch (const std::invalid_argument&) { ++covThrown; }
        try { (void)id2.convolveCov(asymCov); } catch (const std::invalid_argument&) { ++covThrown; }
        try { (void)id2.convolveCov(nanCov); } catch (const std::invalid_argument&) { ++covThrown; }
        try { (void)id2.convolveCov(negCov); } catch (const std::domain_error&) { ++covThrown; }
        if (covThrown != 4)
        {
            fail("only " + std::to_string(covThrown) + " of 4 invalid convolveCov calls threw");
        }
    }

    // Errors: different sizes, different unspecified dimensions, and
    // two kernels both infinitely broad along the same direction
    int thrown = 0;
    try { (void)k1.convolve(kden::Bandwidth::identity(3)); } catch (const std::invalid_argument&) { ++thrown; }
    try { (void)k1.convolve(k2.reduce({true, false})); } catch (const std::invalid_argument&) { ++thrown; }
    try { (void)kden::Bandwidth(sing).convolve(kden::Bandwidth(sing)); } catch (const std::domain_error&) { ++thrown; }
    if (thrown != 3)
    {
        fail("only " + std::to_string(thrown) + " of 3 invalid convolutions threw");
    }
    return failures;
}

#endif // TESTBANDWIDTH_HPP
