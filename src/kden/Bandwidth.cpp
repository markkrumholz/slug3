/**
 * @file Bandwidth.cpp
 * @author Mark Krumholz
 * @brief Implementation of kden::Bandwidth
 * @date 2026-10-15
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#include "Bandwidth.hpp"
#include <Eigen/Dense> // NOLINT(misc-include-cleaner)
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <functional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

// Public constructor: a full (unreduced) metric
kden::Bandwidth::Bandwidth(Eigen::MatrixXd h2Inv) : // NOLINT(misc-include-cleaner) -- Eigen::MatrixXd is provided by <Eigen/Dense> (included above); clang-tidy's IWYU mapping doesn't know that
    h2Inv_(std::move(h2Inv))
{
    if (h2Inv_.rows() == 0 || h2Inv_.rows() != h2Inv_.cols())
    {
        throw std::invalid_argument("Bandwidth: h2Inv must be a non-empty square matrix");
    }
    const double scale = h2Inv_.cwiseAbs().maxCoeff();
    if ((h2Inv_ - h2Inv_.transpose()).cwiseAbs().maxCoeff() > 1e-12 * scale)
    {
        throw std::invalid_argument("Bandwidth: h2Inv must be symmetric");
    }
    unspecified_.assign(static_cast<std::size_t>(h2Inv_.rows()), false);
    prepare();
}

// Private constructor: a reduced metric (see reduce())
kden::Bandwidth::Bandwidth(Eigen::MatrixXd h2Inv, std::vector<bool> unspecified) :
    h2Inv_(std::move(h2Inv)),
    unspecified_(std::move(unspecified))
{
    prepare();
}

auto kden::Bandwidth::identity(const std::size_t nDim) -> Bandwidth
{
    const auto n = static_cast<Eigen::Index>(nDim); // NOLINT(misc-include-cleaner) -- Eigen::Index is provided by <Eigen/Dense> (included above); clang-tidy's IWYU mapping doesn't know that
    return Bandwidth(Eigen::MatrixXd::Identity(n, n));
}

// Find the specified dimensions, extract the matrix's block over them,
// decide whether it is diagonal, and, if not, factor it for box bounds
void kden::Bandwidth::prepare()
{
    spec_.clear();
    for (std::size_t i = 0; i < unspecified_.size(); ++i)
    {
        if (!unspecified_[i]) { spec_.push_back(i); }
    }
    const auto ns = static_cast<Eigen::Index>(spec_.size());
    ms_.resize(ns, ns);
    for (Eigen::Index a = 0; a < ns; ++a)
    {
        for (Eigen::Index b = 0; b < ns; ++b)
        {
            ms_(a, b) = h2Inv_(static_cast<Eigen::Index>(spec_[static_cast<std::size_t>(a)]), // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
                               static_cast<Eigen::Index>(spec_[static_cast<std::size_t>(b)])); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
        }
    }
    diag_ = ms_.diagonal();
    diagonal_ = ns == 0 || (ms_ - Eigen::MatrixXd(diag_.asDiagonal())).cwiseAbs().maxCoeff() == 0.0;

    if (diagonal_)
    {
        if (ns > 0 && diag_.minCoeff() < 0.0)
        {
            throw std::domain_error("Bandwidth: h2Inv is not positive semi-definite");
        }
        det_ = diag_.prod();
        w_.resize(0, 0);
        absW_.resize(0, 0);
        return;
    }

    // Box-bound factor: M_s = P^T L D L^T P, W = D^1/2 L^T P
    const Eigen::LDLT<Eigen::MatrixXd> ldlt(ms_); // NOLINT(misc-include-cleaner) -- Eigen::LDLT is provided by <Eigen/Dense> (included above); clang-tidy's IWYU mapping doesn't know that
    const Eigen::VectorXd& d = ldlt.vectorD();
    if (ldlt.info() != Eigen::Success || d.minCoeff() < -1e-12 * std::max(d.maxCoeff(), 0.0)) // NOLINT(misc-include-cleaner) -- Eigen::Success is provided by <Eigen/Dense> (included above); clang-tidy's IWYU mapping doesn't know that
    {
        throw std::domain_error("Bandwidth: h2Inv is not positive semi-definite");
    }
    det_ = d.cwiseMax(0.0).prod();   // L is unit triangular and P a permutation
    Eigen::MatrixXd perm = Eigen::MatrixXd::Identity(ns, ns);
    perm = ldlt.transpositionsP() * perm;
    w_ = d.cwiseMax(0.0).cwiseSqrt().asDiagonal() * Eigen::MatrixXd(ldlt.matrixU()) * perm;
    absW_ = w_.cwiseAbs();
}

auto kden::Bandwidth::convolve(const Bandwidth& other) const -> Bandwidth
{
    if (other.nDim() != nDim() || other.unspecified_ != unspecified_)
    {
        throw std::invalid_argument("Bandwidth::convolve: the bandwidths must have the same "
            "number of dimensions and the same unspecified dimensions");
    }

    // M1 (M1 + M2)^-1 M2 over the specified dimensions
    const Eigen::LLT<Eigen::MatrixXd> llt(ms_ + other.ms_); // NOLINT(misc-include-cleaner) -- Eigen::LLT is provided by <Eigen/Dense> (included above); clang-tidy's IWYU mapping doesn't know that
    if (llt.info() != Eigen::Success)
    {
        throw std::domain_error("Bandwidth::convolve: the sum of the two matrices is not "
            "positive definite");
    }
    // Symmetrized into a separate matrix: assigning back to the one
    // being transposed would alias, since Eigen evaluates lazily
    const Eigen::MatrixXd prod = ms_ * llt.solve(other.ms_);
    const Eigen::MatrixXd conv = 0.5 * (prod + prod.transpose());
    return {embed(conv), unspecified_};
}

auto kden::Bandwidth::convolveCov(const Eigen::MatrixXd& cov) const -> Bandwidth
{
    const auto ni = static_cast<Eigen::Index>(nDim());
    if (cov.rows() != ni || cov.cols() != ni || !cov.allFinite())
    {
        throw std::invalid_argument("Bandwidth::convolveCov: cov must be a finite nDim x nDim matrix");
    }
    const double scale = std::max(cov.cwiseAbs().maxCoeff(), 1.0e-300);
    if ((cov - cov.transpose()).cwiseAbs().maxCoeff() > 1e-12 * scale)
    {
        throw std::invalid_argument("Bandwidth::convolveCov: cov must be symmetric");
    }

    // C's block over the specified dimensions, which must be positive
    // semi-definite
    const auto ns = static_cast<Eigen::Index>(spec_.size());
    Eigen::MatrixXd cs(ns, ns);
    for (Eigen::Index a = 0; a < ns; ++a)
    {
        for (Eigen::Index b = 0; b < ns; ++b)
        {
            cs(a, b) = cov(static_cast<Eigen::Index>(spec_[static_cast<std::size_t>(a)]), // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
                           static_cast<Eigen::Index>(spec_[static_cast<std::size_t>(b)])); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
        }
    }
    if (ns > 0)
    {
        const Eigen::LDLT<Eigen::MatrixXd> ldlt(cs);
        const Eigen::VectorXd& d = ldlt.vectorD();
        if (ldlt.info() != Eigen::Success || d.minCoeff() < -1e-12 * std::max(d.maxCoeff(), 0.0))
        {
            throw std::domain_error("Bandwidth::convolveCov: cov is not positive semi-definite");
        }
    }

    // (M^-1 + 2 C)^-1 = (I + 2 M C)^-1 M; I + 2 M C is invertible for
    // positive semi-definite M and C, since M C's eigenvalues are real
    // and non-negative. Symmetrized into a separate matrix, as in
    // convolve().
    const Eigen::MatrixXd lhs = Eigen::MatrixXd::Identity(ns, ns) + (2.0 * ms_ * cs);
    const Eigen::MatrixXd prod = lhs.partialPivLu().solve(ms_);
    const Eigen::MatrixXd conv = 0.5 * (prod + prod.transpose());
    return {embed(conv), unspecified_};
}

auto kden::Bandwidth::embed(const Eigen::MatrixXd& block) const -> Eigen::MatrixXd
{
    const auto ni = static_cast<Eigen::Index>(nDim());
    Eigen::MatrixXd full = Eigen::MatrixXd::Zero(ni, ni);
    for (std::size_t r = 0; r < spec_.size(); ++r)
    {
        for (std::size_t c = 0; c < spec_.size(); ++c)
        {
            full(static_cast<Eigen::Index>(spec_[r]), static_cast<Eigen::Index>(spec_[c])) = // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
                block(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(c));
        }
    }
    return full;
}

auto kden::Bandwidth::reduce(const std::vector<bool>& unspecified) const -> Bandwidth
{
    const std::size_t n = nDim();
    if (unspecified.size() != n)
    {
        throw std::invalid_argument("Bandwidth::reduce: unspecified must have nDim elements");
    }
    std::vector<std::size_t> keep;   // dimensions that stay specified
    std::vector<std::size_t> drop;   // dimensions newly unspecified
    for (std::size_t i = 0; i < n; ++i)
    {
        if (unspecified_[i] && !unspecified[i])
        {
            throw std::invalid_argument("Bandwidth::reduce: unspecified must include every "
                "dimension this Bandwidth already leaves unspecified");
        }
        if (!unspecified[i]) { keep.push_back(i); }
        else if (!unspecified_[i]) { drop.push_back(i); }
    }
    if (drop.empty()) { return *this; }

    // Schur complement of the newly unspecified block, over what remains
    auto block = [this](const std::vector<std::size_t>& rows, const std::vector<std::size_t>& cols)
    {
        Eigen::MatrixXd b(static_cast<Eigen::Index>(rows.size()), static_cast<Eigen::Index>(cols.size()));
        for (std::size_t r = 0; r < rows.size(); ++r)
        {
            for (std::size_t c = 0; c < cols.size(); ++c)
            {
                b(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(c)) =
                    h2Inv_(static_cast<Eigen::Index>(rows[r]), static_cast<Eigen::Index>(cols[c])); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
            }
        }
        return b;
    };
    const Eigen::LLT<Eigen::MatrixXd> llt(block(drop, drop)); // NOLINT(misc-include-cleaner) -- Eigen::LLT is provided by <Eigen/Dense> (included above); clang-tidy's IWYU mapping doesn't know that
    if (llt.info() != Eigen::Success)
    {
        throw std::domain_error("Bandwidth::reduce: the block of h2Inv over the newly "
            "unspecified dimensions is not positive definite");
    }
    const Eigen::MatrixXd raw = block(keep, keep) - (block(keep, drop) * llt.solve(block(drop, keep)));
    const Eigen::MatrixXd schur = 0.5 * (raw + raw.transpose());   // not in place: see convolve()

    const auto ni = static_cast<Eigen::Index>(n);
    Eigen::MatrixXd reduced = Eigen::MatrixXd::Zero(ni, ni);
    for (std::size_t r = 0; r < keep.size(); ++r)
    {
        for (std::size_t c = 0; c < keep.size(); ++c)
        {
            reduced(static_cast<Eigen::Index>(keep[r]), static_cast<Eigen::Index>(keep[c])) = // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
                schur(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(c));
        }
    }
    return {std::move(reduced), unspecified};
}

auto kden::Bandwidth::nanPattern(std::span<const double> x) -> std::vector<bool>
{
    std::vector<bool> pattern(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) { pattern[i] = std::isnan(x[i]); } // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
    return pattern;
}

auto kden::Bandwidth::matches(std::span<const double> x) const -> bool
{
    if (x.size() != unspecified_.size()) { return false; }
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        if (std::isnan(x[i]) != unspecified_[i]) { return false; } // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
    }
    return true;
}

auto kden::Bandwidth::dist2(std::span<const double> x1, std::span<const double> x2) const -> double
{
    std::vector<double> scratch;
    return dist2(x1, x2, scratch);
}

auto kden::Bandwidth::dist2(std::span<const double> x1, std::span<const double> x2,
    std::vector<double>& scratch) const -> double
{
    assert(x1.size() == nDim() && x2.size() == nDim());
    const std::size_t ns = spec_.size();

    // Separations over the specified dimensions, zero where either
    // coordinate is NaN; the unspecified dimensions' rows and columns
    // of a reduced matrix are zero, so they never contribute
    scratch.resize(ns);
    for (std::size_t a = 0; a < ns; ++a)
    {
        const std::size_t i = spec_[a]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
        const double p = x1[i]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
        const double q = x2[i]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
        scratch[a] = (std::isnan(p) || std::isnan(q)) ? 0.0 : p - q; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
    }

    double r2 = 0.0;
    if (diagonal_)
    {
        for (std::size_t a = 0; a < ns; ++a)
        {
            const double d = scratch[a]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
            r2 += diag_(static_cast<Eigen::Index>(a)) * d * d;
        }
        return r2;
    }

    // d^T M_s d, using symmetry: sum_a d_a (M_aa d_a + 2 sum_{b<a} M_ab d_b)
    for (std::size_t a = 0; a < ns; ++a)
    {
        const double da = scratch[a]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
        if (da == 0.0) { continue; }
        const auto ai = static_cast<Eigen::Index>(a);
        double off = 0.0;
        for (std::size_t b = 0; b < a; ++b)
        {
            off += ms_(ai, static_cast<Eigen::Index>(b)) * scratch[b]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
        }
        r2 += da * ((ms_(ai, ai) * da) + (2.0 * off));
    }
    return r2;
}

auto kden::Bandwidth::boxBounder(std::span<const double> x) const -> BoxBounder
{
    return {*this, x};
}

kden::Bandwidth::BoxBounder::BoxBounder(const Bandwidth& bandwidth, std::span<const double> x) :
    bandwidth_(std::cref(bandwidth))
{
    const auto& spec = bandwidth.spec_;
    const auto ns = static_cast<Eigen::Index>(spec.size());
    xs_.resize(ns);
    for (Eigen::Index a = 0; a < ns; ++a)
    {
        xs_(a) = x[spec[static_cast<std::size_t>(a)]]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
    }
    if (!bandwidth.diagonal_)
    {
        wx_ = bandwidth.w_ * xs_;
        center_.resize(ns);
        half_.resize(ns);
    }
}

template <bool WantLower, bool WantUpper>
auto kden::Bandwidth::BoxBounder::compute(std::span<const double> lo,
    std::span<const double> hi) const -> std::pair<double, double>
{
    const Bandwidth& bw = bandwidth_.get();
    const auto& spec = bw.spec_;
    const auto ns = static_cast<Eigen::Index>(spec.size());
    double lowerBound = 0.0; // NOLINT(misc-const-correctness) -- unmodified only in instantiations where if constexpr drops the lower bound
    double upperBound = 0.0; // NOLINT(misc-const-correctness) -- likewise for the upper bound

    // Diagonal metric: the exact distances to the nearest point of the
    // box and to its farthest corner, dimension by dimension
    if (bw.diagonal_)
    {
        for (Eigen::Index a = 0; a < ns; ++a)
        {
            const std::size_t i = spec[static_cast<std::size_t>(a)]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
            const double dlo = xs_(a) - lo[i]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
            const double dhi = hi[i] - xs_(a); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
            if constexpr (WantLower)
            {
                const double gap = std::max({-dlo, -dhi, 0.0});
                lowerBound += bw.diag_(a) * gap * gap;
            }
            if constexpr (WantUpper)
            {
                const double far = std::max(std::abs(dlo), std::abs(dhi));
                upperBound += bw.diag_(a) * far * far;
            }
        }
        return {lowerBound, upperBound};
    }

    // General metric: separating-axis bounds along the rows of W; along
    // row k the box projects onto the interval w_k . c +- |w_k| . h
    for (Eigen::Index a = 0; a < ns; ++a)
    {
        const std::size_t i = spec[static_cast<std::size_t>(a)]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
        center_(a) = 0.5 * (lo[i] + hi[i]); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
        half_(a) = 0.5 * (hi[i] - lo[i]); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
    }
    for (Eigen::Index k = 0; k < ns; ++k)
    {
        const double offset = std::abs(wx_(k) - bw.w_.row(k).dot(center_));
        const double halfWidth = bw.absW_.row(k).dot(half_);
        if constexpr (WantLower)
        {
            const double gap = std::max(offset - halfWidth, 0.0);
            lowerBound += gap * gap;
        }
        if constexpr (WantUpper)
        {
            const double far = offset + halfWidth;
            upperBound += far * far;
        }
    }
    return {lowerBound * (1.0 - 1e-10), upperBound * (1.0 + 1e-10)};
}

auto kden::Bandwidth::BoxBounder::lower(std::span<const double> lo,
    std::span<const double> hi) const -> double
{
    return compute<true, false>(lo, hi).first;
}

auto kden::Bandwidth::BoxBounder::upper(std::span<const double> lo,
    std::span<const double> hi) const -> double
{
    return compute<false, true>(lo, hi).second;
}

auto kden::Bandwidth::BoxBounder::bounds(std::span<const double> lo,
    std::span<const double> hi) const -> std::pair<double, double>
{
    return compute<true, true>(lo, hi);
}
