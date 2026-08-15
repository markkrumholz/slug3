/**
 * @file GKIntegrator.hpp
 * @author Mark Krumholz
 * @brief Adaptive Gauss-Kronrod integration of vector-valued integrands
 * @date 2026-08-15
 */

#ifndef GKINTEGRATOR_HPP
#define GKINTEGRATOR_HPP

#include "GKIntegratorData.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace utils
{

    /**
     * @brief The outcome of applying a single Gauss-Kronrod rule to one interval
     * @details
     * What GKIntegrator::quadSingle() returns, and what
     * GKIntegrator::integrate() bisects and re-integrates as it
     * adaptively refines the overall integration domain -- see both
     * of their own comments.
     */
    struct GKIntegrationInterval
    {
        double a_;                  /**< Lower limit of this interval */
        double b_;                  /**< Upper limit of this interval */
        std::vector<double> quad_;  /**< The Kronrod rule's own quadrature estimate over [a_, b_], one value per quantity f_ returns */
        std::vector<double> err_;   /**< This interval's own error estimate, one value per quantity f_ returns -- see quadSingle()'s own comment for how */
    };

    /**
     * @class GKIntegrator
     * @brief Adaptively integrate a vector-valued function of one variable
     * @tparam F Type of the callable integrand: either a standalone
     *   callable, invocable as f(x, ...), where x is the (scalar)
     *   integration variable and ... is any number of additional
     *   arguments forwarded from integrate()/quadSingle(); or a
     *   pointer to a member function, in which case the first of
     *   those additional arguments must instead be an instance of the
     *   class f is a member of (x is still passed as that member
     *   function's own first ordinary argument, immediately after the
     *   instance) -- see invokeF()'s own comment for how the two
     *   cases are told apart and reordered as needed. Either way, f
     *   must return a contiguous container of nInt doubles (e.g.
     *   std::vector<double> or std::array<double, nInt>) -- the value
     *   of the integrand at x.
     * @tparam Order Which Gauss-Kronrod pair to integrate each
     *   subinterval with -- see GKOrder's own comment; selected at
     *   compile time (rather than a runtime option) so the
     *   corresponding GKRule<Order> specialization's own fixed-size
     *   xgk/wg/wgk arrays are available with no indirection or
     *   runtime dispatch.
     * @details
     * An adaptation of the GSL's gsl_integration_qag() to vector-valued
     * integrands, using C++ templating rather than gsl_function's
     * pointer-to-void mechanism to pass any extra arguments f itself
     * requires beyond the integration variable.
     */
    template <class F, GKOrder Order>
    class GKIntegrator
    {
    public:

        /**
         * @brief Construct a GKIntegrator
         * @param f The integrand; see this class's own @tparam F
         * @param nInt Number of quantities f returns per point
         * @param maxIter Maximum number of integrate()'s own bisection
         *   iterations (0 = unlimited) -- see its own comment; not a
         *   count of raw evaluations of f_ itself
         * @param absTol Required absolute error
         * @param relTol Required relative error
         */
        GKIntegrator(
            F f,
            std::size_t nInt,
            std::size_t maxIter = 0,
            double absTol = 0.0,
            double relTol = 1e-6
        ) :
            f_(std::move(f)),
            nInt_(nInt),
            maxIter_(maxIter),
            absTol_(absTol),
            relTol_(relTol)
        { }

        /**
         * @brief Apply a single, non-adaptive Gauss-Kronrod rule to one interval
         * @param a Lower limit of the interval
         * @param b Upper limit of the interval
         * @param args Any additional arguments f_ requires, forwarded
         *   to it unchanged after x at every point evaluated
         * @return A GKIntegrationInterval with a_/b_ set to a/b, quad_
         *   set to the ngk-point Kronrod rule's own estimate of the
         *   integral (one value per quantity f_ returns), and err_ set
         *   to an error estimate for each of those same values -- the
         *   absolute difference between the Kronrod estimate and the
         *   embedded (lower-order) Gauss rule's own estimate of the
         *   same integral
         * @details
         * A direct translation of slug2's own
         * slug_imf_integrator<T>::integrate_gk (src/utils/
         * slug_imf_integrator.cpp), generalized from a single
         * scalar-or-vector quantity T to an arbitrary, runtime-sized
         * nInt_ of them, and with the IMF weighting integrate_gk
         * itself folds in (imf_val) dropped entirely -- f_ is
         * evaluated on its own here, unweighted; PDFIntegrator is
         * where that weighting belongs instead, layered on top of this
         * class, not inside it.
         *
         * Builds the symmetric grid of gknum = 2 * ngk - 1 abscissae
         * spanning [a, b] from GKRule<Order>::xgk (see its own
         * comment for the half-plus-center storage convention this
         * mirrors), evaluates f_ at every one of them, then forms two
         * weighted sums over those same nInt_-length values: the full
         * Kronrod sum (all gknum points, weighted by
         * GKRule<Order>::wgk) and the embedded Gauss sum (only the
         * subset of points the lower-order Gauss rule itself uses,
         * weighted by GKRule<Order>::wg) -- including the shared
         * center point (x = (a + b) / 2) in the Gauss sum too,
         * whenever GKRule<Order>::ngk is even (see GKRule's own
         * comment for why that parity is exactly the right test).
         * Both sums are scaled by the interval's own half-length to
         * convert from the standard [-1, 1] quadrature domain to
         * [a, b], and their absolute difference is returned as this
         * call's own error estimate -- a cheap, standard proxy for the
         * Kronrod estimate's own true error, since the embedded Gauss
         * rule is enough lower-order that the two estimates should
         * differ substantially only where the Kronrod one is itself
         * unreliable.
         */
        template <class... Args>
        [[nodiscard]] auto quadSingle(const double a, const double b, Args&&... args) const
        -> GKIntegrationInterval
        {
            using Rule = GKRule<Order>;
            constexpr std::size_t ngk = Rule::ngk;
            constexpr std::size_t gknum = (2 * ngk) - 1;

            const double center = 0.5 * (a + b);
            const double halfLength = 0.5 * (b - a);

            std::array<double, gknum> xk{};
            for (std::size_t i = 0; i < gknum / 2; ++i)
            {
                xk[i] = center - (halfLength * Rule::xgk[i]); // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- i < gknum / 2 == ngk - 1 < ngk == Rule::xgk.size() by construction
                xk[gknum - i - 1] = center + (halfLength * Rule::xgk[i]); // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- gknum - i - 1 >= gknum - gknum/2 > 0 and < gknum, and i < ngk as above
            }
            xk[gknum / 2] = center; // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- gknum / 2 < gknum by construction

            // Evaluate f_ at every point first, rather than
            // interleaving evaluation with the summation loops below,
            // to mirror integrate_gk's own structure -- each point's
            // own nInt_ values are copied out of whatever contiguous
            // container f_ returns (see this class's own @tparam F)
            // into a plain std::vector<double>, since nInt_ itself is
            // only known at runtime. args is forwarded exactly once,
            // into this tuple, rather than at each of the gknum calls
            // below: repeatedly forwarding the same forwarding-reference
            // parameter would risk moving from an rvalue-bound argument
            // on its first use, leaving it in a moved-from state for
            // every later point -- std::apply instead hands every call
            // the tuple's own elements, exactly as deduced (not const,
            // so that invokeF()/invokeFMember() can still bind them to
            // a non-const reference parameter when F's own signature --
            // e.g. a member-function-pointer instantiation bound to a
            // specific, concrete Args... -- demands one).
            std::tuple<Args...> argsTuple(std::forward<Args>(args)...);
            std::vector<std::vector<double>> fVal(gknum);
            for (std::size_t i = 0; i < gknum; ++i)
            {
                const auto val = std::apply(
                    [this, &xk, i](auto&... unpackedArgs) { return invokeF(xk[i], unpackedArgs...); }, // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- i < gknum == xk.size() by construction
                    argsTuple);
                fVal[i].assign(std::begin(val), std::end(val));
            }

            std::vector<double> result(nInt_, 0.0);
            std::vector<double> gaussQuad(nInt_, 0.0);

            // Central point: always part of the Kronrod sum (weight
            // wgk[ngk - 1], the last stored weight -- see GKRule's own
            // comment for why); part of the Gauss sum too only when
            // ngk is even (see this function's own comment for why).
            const std::size_t centerIdx = gknum / 2;
            for (std::size_t k = 0; k < nInt_; ++k)
            {
                result[k] = fVal[centerIdx][k] * Rule::wgk[ngk - 1]; // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- ngk - 1 < ngk == Rule::wgk.size() by construction
            }
            if constexpr (ngk % 2 == 0)
            {
                for (std::size_t k = 0; k < nInt_; ++k)
                {
                    gaussQuad[k] = fVal[centerIdx][k] * Rule::wg[(ngk / 2) - 1]; // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- ngk / 2 - 1 < ng == Rule::wg.size() whenever ngk is even, by the Gauss/Kronrod pairing GKRule's own comment describes
                }
            }

            // Points shared between the Gauss and Kronrod sums
            for (std::size_t i = 0; i < (ngk - 1) / 2; ++i)
            {
                const std::size_t p1 = (2 * i) + 1;
                const std::size_t p2 = gknum - (2 * i) - 2;
                for (std::size_t k = 0; k < nInt_; ++k)
                {
                    const double sum = fVal[p1][k] + fVal[p2][k]; // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- p1, p2 < gknum == fVal.size() by construction
                    gaussQuad[k] += Rule::wg[i] * sum; // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- i < (ngk - 1) / 2 <= ng == Rule::wg.size(), by the Gauss/Kronrod pairing GKRule's own comment describes
                    result[k] += Rule::wgk[p1] * sum; // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- p1 < ngk == Rule::wgk.size() by construction
                }
            }

            // Points that appear only in the Kronrod sum
            for (std::size_t i = 0; i < ngk / 2; ++i)
            {
                const std::size_t p1 = 2 * i;
                const std::size_t p2 = gknum - (2 * i) - 1;
                for (std::size_t k = 0; k < nInt_; ++k)
                {
                    result[k] += Rule::wgk[p1] * (fVal[p1][k] + fVal[p2][k]); // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- p1 < ngk == Rule::wgk.size(), and p1, p2 < gknum == fVal.size(), by construction
                }
            }

            std::vector<double> err(nInt_);
            for (std::size_t k = 0; k < nInt_; ++k)
            {
                result[k] *= halfLength;
                gaussQuad[k] *= halfLength;
                err[k] = std::abs(result[k] - gaussQuad[k]);
            }
            return { a, b, std::move(result), std::move(err) };
        }

        /**
         * @brief Adaptively integrate f_ over [a, b]
         * @param a Lower limit of integration
         * @param b Upper limit of integration
         * @param args Any additional arguments f_ requires; see
         *   quadSingle()'s own args parameter
         * @return The integral estimate, one value per quantity f_
         *   returns (length nInt_)
         * @details
         * Starts from a single quadSingle() call over the whole
         * [a, b] interval; if that single interval's own largest
         * elementwise error (max_k |err_[k]|) is already below absTol_,
         * or its own largest elementwise relative error
         * (max_k |err_[k]| / |quad_[k]|) is already below relTol_,
         * returns immediately -- no bisection needed.
         *
         * Otherwise, repeatedly bisects whichever interval currently
         * contributes the largest relative error to the running total
         * (quadSum, tracked incrementally: each iteration undoes one
         * interval's own old contribution to quadSum/errSum, replaces
         * that interval in place with a fresh quadSingle() over its
         * own left half, appends a new interval for its own right
         * half, and adds both halves' own contributions back in),
         * stopping -- and returning quadSum -- as soon as either the
         * largest element of errSum drops below absTol_, or the
         * largest elementwise relative error (|errSum[k]| / |quadSum[k]|)
         * drops below relTol_, or the number of bisection iterations
         * exceeds maxIter_ (0 = unlimited) -- not a count of raw
         * evaluations of f_ itself, which each iteration costs two
         * further quadSingle() calls' worth of (2 * (2 * ngk - 1)
         * further evaluations).
         *
         * Which interval to bisect next is decided by each interval's
         * own relative error against the *global* running total
         * (|intervals[i].err_[k]| / |quadSum[k]|, maximized over k),
         * not against that interval's own local quad_ -- so bisection
         * effort concentrates on whichever interval's own uncertainty
         * matters most to the overall answer, mirroring slug2's own
         * identical convention in slug_imf_integrator<T>::
         * integrate_range (see quadSingle()'s own comment for that
         * function's sibling, integrate_gk, which this class's
         * quadSingle() itself translates).
         */
        template <class... Args>
        [[nodiscard]] auto integrate(const double a, const double b, Args&&... args) const -> std::vector<double>
        {
            // args is forwarded exactly once, into this tuple, for the
            // same reason quadSingle() itself does this -- see its own
            // comment (including for why this is not const).
            std::tuple<Args...> argsTuple(std::forward<Args>(args)...);
            const auto callQuadSingle = [this, &argsTuple](const double lo, const double hi) -> GKIntegrationInterval
            {
                return std::apply(
                    [this, lo, hi](auto&... unpackedArgs) { return quadSingle(lo, hi, unpackedArgs...); },
                    argsTuple);
            };

            std::vector<GKIntegrationInterval> intervals;
            intervals.push_back(callQuadSingle(a, b));

            const double absErr = *std::ranges::max_element(intervals[0].err_);
            if (absErr < absTol_) { return intervals[0].quad_; }
            {
                double maxRelErr = 0.0;
                for (std::size_t k = 0; k < nInt_; ++k)
                {
                    maxRelErr = std::max(maxRelErr,
                        std::abs(intervals[0].err_[k] / intervals[0].quad_[k])); // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- k < nInt_ == intervals[0].err_.size() == intervals[0].quad_.size() by construction
                }
                if (maxRelErr < relTol_) { return intervals[0].quad_; }
            }

            std::vector<double> quadSum = intervals[0].quad_;
            std::vector<double> errSum = intervals[0].err_;
            std::size_t itCounter = 1;
            std::size_t intervalPtr = 0;

            while (true)
            {
                const double xMid = 0.5 * (intervals[intervalPtr].a_ + intervals[intervalPtr].b_);
                const double bOld = intervals[intervalPtr].b_; // intervals[intervalPtr] itself is about to be overwritten below, taking its own b_ with it -- the right half bisected off it still needs the original upper limit, not the new one
                for (std::size_t k = 0; k < nInt_; ++k)
                {
                    quadSum[k] -= intervals[intervalPtr].quad_[k]; // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- k < nInt_ == quadSum.size() == intervals[intervalPtr].quad_.size() by construction
                    errSum[k] -= intervals[intervalPtr].err_[k]; // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- see above, for errSum/err_
                }

                intervals[intervalPtr] = callQuadSingle(intervals[intervalPtr].a_, xMid);
                intervals.push_back(callQuadSingle(xMid, bOld));

                const auto& left = intervals[intervalPtr];
                const auto& right = intervals.back();
                for (std::size_t k = 0; k < nInt_; ++k)
                {
                    quadSum[k] += left.quad_[k] + right.quad_[k]; // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- k < nInt_ == quadSum.size() == left.quad_.size() == right.quad_.size() by construction
                    errSum[k] += left.err_[k] + right.err_[k]; // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- see above, for errSum/err_
                }

                ++itCounter;
                if (maxIter_ != 0 && itCounter > maxIter_) { return quadSum; }

                double maxAbsErr = 0.0;
                double maxRelErr = 0.0;
                for (std::size_t k = 0; k < nInt_; ++k)
                {
                    maxAbsErr = std::max(maxAbsErr, std::abs(errSum[k])); // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- k < nInt_ == errSum.size() by construction
                    maxRelErr = std::max(maxRelErr, std::abs(errSum[k] / quadSum[k])); // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- see above, for quadSum
                }
                if (maxAbsErr < absTol_ || maxRelErr < relTol_) { return quadSum; }

                double bestRelErr = -1.0;
                std::size_t best = 0;
                for (std::size_t i = 0; i < intervals.size(); ++i)
                {
                    double relErr = 0.0;
                    for (std::size_t k = 0; k < nInt_; ++k)
                    {
                        relErr = std::max(relErr,
                            std::abs(intervals[i].err_[k] / quadSum[k])); // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- k < nInt_ == intervals[i].err_.size() == quadSum.size() by construction
                    }
                    if (relErr > bestRelErr) { bestRelErr = relErr; best = i; }
                }
                intervalPtr = best;
            }
        }

    private:

        /**
         * @brief Evaluate f_ at x, transparently supporting F being a pointer to member function
         * @param x The integration variable
         * @param args Any additional arguments f_ requires -- if F is
         *   a pointer to member function, the first of these must be
         *   an instance of the class it is a member of (see this
         *   class's own @tparam F)
         * @return f_'s own return value at (x, args...)
         * @details
         * Dispatches on std::is_member_function_pointer_v<F> at
         * compile time: for a standalone callable, just forwards
         * straight to std::invoke(f_, x, args...); for a pointer to
         * member function, hands off to invokeFMember() instead,
         * which reorders arguments into the shape std::invoke's own
         * INVOKE protocol requires -- see its own comment.
         */
        template <class... CallArgs>
        [[nodiscard]] auto invokeF(const double x, CallArgs&&... args) const
        {
            if constexpr (std::is_member_function_pointer_v<F>)
            {
                return invokeFMember(x, std::forward<CallArgs>(args)...);
            }
            else
            {
                return std::invoke(f_, x, std::forward<CallArgs>(args)...);
            }
        }

        /**
         * @brief Reorder (x, instance, rest...) into the (instance, x, rest...) order std::invoke's own INVOKE protocol requires for a pointer to member function
         * @param x The integration variable
         * @param obj Instance of the class f_ is a member of
         * @param rest Any further arguments f_ requires, beyond x and obj
         * @return f_'s own return value at (x, rest...), called on obj
         */
        template <class Obj, class... Rest>
        [[nodiscard]] auto invokeFMember(const double x, Obj&& obj, Rest&&... rest) const
        {
            return std::invoke(f_, std::forward<Obj>(obj), x, std::forward<Rest>(rest)...);
        }

        F f_;                  /**< The integrand */
        std::size_t nInt_;     /**< Number of quantities f returns per point */
        std::size_t maxIter_;  /**< Maximum number of integrate()'s own bisection iterations */
        double absTol_;        /**< Required absolute error */
        double relTol_;        /**< Required relative error */
    };

} // namespace utils

#endif // GKINTEGRATOR_HPP
