/**
 * @file RandomDist.hpp
 * @author Mark Krumholz
 * @brief Portable random number distributions
 * @date 2026-10-13
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 * @details
 * The C++ standard specifies random number engines exactly, but leaves
 * the algorithms that the standard distributions (std::normal_distribution
 * and friends) use to turn engine output into deviates up to the
 * implementation. As a result the same engine state yields different
 * deviates under different standard libraries (e.g., libstdc++ versus
 * libc++), so neither runs nor clusters reconstructed from a stored rng
 * state (see slug_reader.get_cluster) are reproducible across compilers.
 *
 * The classes and functions here replace the standard distributions with
 * slug's own implementations. They deliberately reproduce, operation for
 * operation, the algorithms used by GCC 14's libstdc++, so that every
 * build of slug now generates the same deviates that a libstdc++ build
 * always has, and rng states recorded by earlier libstdc++ runs remain
 * valid.
 *
 * Bitwise agreement also requires that the compiler not contract
 * expressions such as a * b + c into fused multiply-adds, which some
 * compilers do by default on targets that have them (e.g., arm64), but
 * libstdc++ builds for baseline x86_64 do not. Contraction is therefore
 * switched off for this file's code only: by pragmas scoped to each
 * function body for Clang, and by an optimize pragma pushed and popped
 * around this file's definitions for GCC. The only remaining source of
 * disagreement is last-bit differences between platforms'
 * implementations of log, exp, and lgamma, which can change the
 * sequence of draws only in the very rare event that such a difference
 * flips the outcome of a comparison.
 *
 * Note that some of these distributions carry state between calls (e.g.,
 * the Marsaglia polar method generates normal deviates in pairs), so
 * reproducing a sequence of draws also requires constructing and
 * discarding distribution objects at the same points as the run being
 * reproduced.
 */

#ifndef RANDOMDIST_HPP
#define RANDOMDIST_HPP

#include "RngThread.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

#if defined(__GNUC__) && !defined(__clang__)
#   pragma GCC push_options
#   pragma GCC optimize("fp-contract=off")
#endif

namespace utils {

    /**
     * @brief Draw a uniform deviate in [0, 1)
     * @param rng The rng engine to draw from
     * @return A uniform deviate in the range [0, 1)
     * @details
     * This consumes exactly one engine output x, and returns x / 2^64,
     * rounded to the nearest double. In the rare (probability 2^-54)
     * case that the rounding produces exactly 1, it instead returns the
     * largest double smaller than 1. This reproduces libstdc++'s
     * std::generate_canonical as of GCC 14 (pre-P0952), and therefore
     * also std::uniform_real_distribution<double>(0, 1). Newer versions
     * of libstdc++ that implement P0952 instead draw again in that
     * case, but are otherwise identical.
     */
    inline auto canonical(RngType& rng) -> double
    {
        static_assert(RngType::min() == 0 &&
                      RngType::max() == std::numeric_limits<std::uint64_t>::max(),
                      "canonical() assumes an engine with a full 64-bit range");
        constexpr double twoTo64 = 0x1p64;
        const double u = static_cast<double>(rng()) / twoTo64;
        if (u >= 1.0) [[unlikely]] { return std::nextafter(1.0, 0.0); }
        return u;
    }

    /**
     * @class NormalDist
     * @brief Normal distribution
     * @details
     * Draws normal deviates using the Marsaglia polar method, which
     * generates deviates in pairs: the first call draws a pair and
     * returns one member of it, and the second call returns the other
     * member without consuming any engine output. This reproduces
     * libstdc++'s std::normal_distribution<double>.
     *
     * Because of this saved deviate, an object's draws are determined
     * by the engine state only together with whether a deviate is
     * saved. To keep runs reproducible from a recorded rng state (see
     * RngThread::getState), an object of this class -- or of any class
     * that holds one, i.e., LognormalDist and PoissonDist -- must never
     * outlive a point at which the rng state is saved or restored. In
     * practice this means constructing it locally within the function
     * that draws from it, as pdfs::PDFSegmentNormal::draw does, rather
     * than storing it as a long-lived member.
     */
    class NormalDist {

    public:

        /**
         * @brief Construct a NormalDist
         * @param mean The mean of the distribution
         * @param stddev The standard deviation of the distribution
         */
        explicit NormalDist(const double mean = 0.0, const double stddev = 1.0)
            : mean_(mean), stddev_(stddev) {}

        /**
         * @brief Draw a deviate
         * @param rng The rng engine to draw from
         * @return A normal deviate
         */
        auto operator()(RngType& rng) -> double
        {
#ifdef __clang__
#   pragma clang fp contract(off)
#endif
            double ret = NAN;
            if (savedAvailable_)
            {
                savedAvailable_ = false;
                ret = saved_;
            }
            else
            {
                double x = NAN;
                double y = NAN;
                double r2 = NAN;
                while (true)
                {
                    x = (2.0 * canonical(rng)) - 1.0;
                    y = (2.0 * canonical(rng)) - 1.0;
                    r2 = (x * x) + (y * y);
                    if (r2 <= 1.0 && r2 != 0.0) { break; }
                }
                const double mult = std::sqrt(-2.0 * std::log(r2) / r2);
                saved_ = x * mult;
                savedAvailable_ = true;
                ret = y * mult;
            }
            return (ret * stddev_) + mean_;
        }

    private:

        double mean_;                  /**< Mean */
        double stddev_;                /**< Standard deviation */
        double saved_ = 0.0;           /**< Second member of the last pair drawn */
        bool savedAvailable_ = false;  /**< Whether saved_ is yet to be returned */

    };

    /**
     * @class LognormalDist
     * @brief Lognormal distribution
     * @details
     * Returns exp(s * n + m), where n is drawn from an internal
     * NormalDist with zero mean and unit dispersion; since that
     * NormalDist generates deviates in pairs, so does this class. This
     * reproduces libstdc++'s std::lognormal_distribution<double>.
     */
    class LognormalDist {

    public:

        /**
         * @brief Construct a LognormalDist
         * @param m The mean of the logarithm of the deviate
         * @param s The standard deviation of the logarithm of the deviate
         */
        explicit LognormalDist(const double m = 0.0, const double s = 1.0)
            : m_(m), s_(s) {}

        /**
         * @brief Draw a deviate
         * @param rng The rng engine to draw from
         * @return A lognormal deviate
         */
        auto operator()(RngType& rng) -> double
        {
#ifdef __clang__
#   pragma clang fp contract(off)
#endif
            return std::exp((s_ * nd_(rng)) + m_);
        }

    private:

        double m_;       /**< Mean of the logarithm */
        double s_;       /**< Standard deviation of the logarithm */
        NormalDist nd_;  /**< Unit normal distribution */

    };

    /**
     * @class DiscreteDist
     * @brief Distribution over a finite set of integers with specified weights
     * @details
     * Returns an index i in [0, n) with probability proportional to
     * weight i. This reproduces libstdc++'s std::discrete_distribution:
     * the weights are divided by their sum, the cumulative distribution
     * is formed by summing them in order (with its final element then
     * set to exactly 1), and the deviate is the index of the first
     * element of the cumulative distribution that is not less than a
     * uniform deviate drawn by canonical(). As in libstdc++, if there
     * are fewer than two weights the result is always 0, and no engine
     * output is consumed.
     */
    class DiscreteDist {

    public:

        /**
         * @brief Construct a DiscreteDist
         * @param weights The weights; these must be non-negative, and
         *   if there are two or more of them their sum must be positive
         *   and finite
         * @throws std::invalid_argument if there are two or more weights
         *   and their sum is not positive and finite
         */
        explicit DiscreteDist(const std::span<const double> weights)
        {
            if (weights.size() < 2) { return; }
            const double sum = std::accumulate(weights.begin(), weights.end(), 0.0);
            if (!std::isfinite(sum) || sum <= 0.0)
            {
                throw std::invalid_argument(
                    "DiscreteDist: sum of weights must be positive and finite");
            }
            cdf_.resize(weights.size());
            double cum = 0.0;
            for (std::size_t i = 0; i < weights.size(); ++i)
            {
                cum += weights[i] / sum; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
                cdf_[i] = cum;           // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
            }
            cdf_.back() = 1.0;
        }

        /**
         * @brief Draw a deviate
         * @param rng The rng engine to draw from
         * @return An index in the range [0, n), where n is the number of weights
         */
        auto operator()(RngType& rng) const -> std::size_t
        {
            if (cdf_.empty()) { return 0; }
            const double p = canonical(rng);
            return static_cast<std::size_t>(
                std::lower_bound(cdf_.begin(), cdf_.end(), p) - cdf_.begin());
        }

    private:

        std::vector<double> cdf_;  /**< Cumulative distribution */

    };

    /**
     * @class PoissonDist
     * @brief Poisson distribution
     * @details
     * For means below 12, deviates are drawn by multiplying uniform
     * deviates together until their product falls to exp(-mean) or
     * below. For larger means, they are drawn by the rejection method
     * of Devroye (1986, Non-Uniform Random Variate Generation,
     * Springer-Verlag, Ch. X, Sects. 3.3 & 3.4, including its errata),
     * which also draws from an internal unit NormalDist. Both
     * reproduce libstdc++'s std::poisson_distribution<unsigned int>,
     * including its choice of constants, the ordering of its branches,
     * and its handling of the case x = 1 (libstdc++ bug 83237).
     */
    class PoissonDist {

    public:

        /**
         * @brief Construct a PoissonDist
         * @param mean The mean of the distribution; must be non-negative
         *   and finite
         * @throws std::invalid_argument if mean is negative or not finite
         */
        explicit PoissonDist(const double mean) : mean_(mean)
        {
#ifdef __clang__
#   pragma clang fp contract(off)
#endif
            if (!std::isfinite(mean) || mean < 0.0)
            {
                throw std::invalid_argument(
                    "PoissonDist: mean must be non-negative and finite");
            }
            if (mean_ >= largeMean)
            {
                const double m = std::floor(mean_);
                lmThr_ = std::log(mean_);
                lfm_ = std::lgamma(m + 1.0); // NOLINT(concurrency-mt-unsafe) -- only races on the global signgam, which is never read; libstdc++'s std::poisson_distribution calls lgamma identically
                sm_ = std::sqrt(m);
                const double dx = std::sqrt(2.0 * m * std::log(32.0 * m / piOver4));
                d_ = std::round(std::max(6.0, std::min(m, dx)));
                const double cx = (2.0 * m) + d_;
                scx_ = std::sqrt(cx / 2.0);
                oneOverCx_ = 1.0 / cx;
                c2b_ = std::sqrt(piOver4 * cx) * std::exp(oneOverCx_);
                cb_ = 2.0 * cx * std::exp(-d_ * oneOverCx_ * (1.0 + (d_ / 2.0))) / d_;
            }
            else
            {
                lmThr_ = std::exp(-mean_);
            }
        }

        /**
         * @brief Draw a deviate
         * @param rng The rng engine to draw from
         * @return A Poisson deviate
         */
        auto operator()(RngType& rng) -> unsigned int
        {
#ifdef __clang__
#   pragma clang fp contract(off)
#endif
            if (mean_ < largeMean)
            {
                unsigned int x = 0;
                double prod = 1.0;
                while (true)
                {
                    prod *= canonical(rng);
                    x += 1;
                    if (prod <= lmThr_) { break; }
                }
                return x - 1;
            }

            // Large mean: rejection method
            constexpr double naf = (1.0 - std::numeric_limits<double>::epsilon()) / 2.0;
            constexpr double thr =
                static_cast<double>(std::numeric_limits<unsigned int>::max()) + naf;
            constexpr double sqrtPiOver2 = 1.2533141373155002512;
            constexpr double oneOver78 = 1.0 / 78.0;
            constexpr double expOneOver78 = 1.0129030479320018583;
            const double m = std::floor(mean_);
            const double c1 = sm_ * sqrtPiOver2;
            const double c2 = c2b_ + c1;
            const double c3 = c2 + 1.0;
            const double c4 = c3 + 1.0;
            const double c5 = c4 + expOneOver78;
            const double c = cb_ + c5;
            const double twoCx = 2.0 * ((2.0 * m) + d_);

            while (true)
            {
                const double u = c * canonical(rng);
                const double e = -std::log(1.0 - canonical(rng));
                double x = 0.0;
                double w = 0.0;
                if (u <= c1)
                {
                    const double n = nd_(rng);
                    const double y = (-std::abs(n) * sm_) - 1.0;
                    x = std::floor(y);
                    w = -n * n / 2.0;
                    if (x < -m) { continue; }
                }
                else if (u <= c2)
                {
                    const double n = nd_(rng);
                    const double y = 1.0 + (std::abs(n) * scx_);
                    x = std::ceil(y);
                    w = y * (2.0 - y) * oneOverCx_;
                    if (x > d_) { continue; }
                }
                else if (u <= c3)
                {
                    x = -1.0;
                }
                else if (u <= c4)
                {
                    x = 0.0;
                }
                else if (u <= c5)
                {
                    x = 1.0;
                    w = oneOver78;
                }
                else
                {
                    const double v = -std::log(1.0 - canonical(rng));
                    const double y = d_ + (v * twoCx / d_);
                    x = std::ceil(y);
                    w = -d_ * oneOverCx_ * (1.0 + (y / 2.0));
                }
                const bool reject =
                    (w - e - (x * lmThr_) > lfm_ - std::lgamma(x + m + 1.0)) || // NOLINT(concurrency-mt-unsafe) -- see constructor
                    (x + m >= thr);
                if (!reject)
                {
                    return static_cast<unsigned int>(x + m + naf);
                }
            }
        }

    private:

        static constexpr double largeMean = 12.0;  /**< Mean at which the algorithm switches */
        static constexpr double piOver4 = 0.78539816339744830962;  /**< pi / 4 */

        double mean_;             /**< Mean */
        double lmThr_ = 0.0;      /**< exp(-mean) for small means, log(mean) for large */
        double lfm_ = 0.0;        /**< log(floor(mean)!) */
        double sm_ = 0.0;         /**< sqrt(floor(mean)) */
        double d_ = 0.0;          /**< Devroye's parameter delta */
        double scx_ = 0.0;        /**< sqrt(c_x / 2) */
        double oneOverCx_ = 0.0;  /**< 1 / c_x */
        double c2b_ = 0.0;        /**< sqrt(pi c_x / 4) exp(1 / c_x) */
        double cb_ = 0.0;         /**< Area of the exponential tail of the hat function */
        NormalDist nd_;           /**< Unit normal distribution */

    };

} // namespace utils

#if defined(__GNUC__) && !defined(__clang__)
#   pragma GCC pop_options
#endif

#endif // RANDOMDIST_HPP
