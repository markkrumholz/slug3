/**
 * @file PDFIntegrator.hpp
 * @author Mark Krumholz
 * @brief Evaluate integrals of the form int_a^b p(x) f(x) dx, using GKIntegrator
 * @date 2026-08-15
 */

#ifndef PDFINTEGRATOR_HPP
#define PDFINTEGRATOR_HPP

#include "../pdfs/PDF.hpp"
#include "GKIntegrator.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace utils
{

    /**
     * @class PDFIntegrator
     * @brief Evaluate integrals of the form int_a^b p(x) f(x) dx, using GKIntegrator
     * @tparam F Type of the callable integrand: either a standalone
     *   callable, invocable as f(x, ...), where x is the (scalar)
     *   integration variable and ... is any number of additional
     *   arguments forwarded from integrate(); or a pointer to a
     *   member function, in which case the first of those additional
     *   arguments must instead be an instance of the class f is a
     *   member of (x is still passed as that member function's own
     *   first ordinary argument, immediately after the instance).
     *   Either way, f must return a contiguous container of nInt_
     *   doubles (e.g. std::vector<double> or std::array<double,
     *   nInt_>) -- the value of the integrand at x.
     * @tparam Order Which Gauss-Kronrod pair GKIntegrator integrates
     *   each subinterval with -- see GKOrder's own comment. Defaults
     *   to GKOrder::GK15.
     * @details
     * Evaluates int_a^b p(x) f(x) dx by adaptively bisecting [a, b]
     * with GKIntegrator, weighting f by p at every point it visits --
     * see integrand()'s own comment for exactly how the two are
     * combined. Benchmarked (as PDFIntegratorGK, its own name before
     * replacing an earlier, cubature-package-based implementation of
     * this same class) head to head against that implementation on a
     * real, full-scale run: consistently faster, in one representative
     * case by roughly an order of magnitude -- see this class's own
     * git history for the comparison.
     *
     * integrate() is not safe to call concurrently on a single, shared
     * PDFIntegrator instance: integrand() (see its own comment) reads
     * the current call's own real-space integration bounds back from
     * two mutable members integrate() sets just before it, rather than
     * from a per-call context object. Every call site in this codebase
     * constructs a fresh PDFIntegrator immediately before using it, so
     * this has not been a practical limitation; revisit if a future
     * caller ever needs to share one PDFIntegrator across concurrent
     * integrate() calls.
     */
    template <class F, GKOrder Order = GKOrder::GK15>
    class PDFIntegrator
    {
    public:

        /**
         * @brief Construct a PDFIntegrator
         * @param p The PDF p(x) weighting the integrand; must outlive
         *   this PDFIntegrator, since it is stored by reference
         * @param f The integrand f(x); see this class's own @tparam F
         * @param nInt The number of quantities f returns per point
         * @param logTransform Whether to integrate over ln(x) instead
         *   of x directly -- a mathematically exact change of
         *   variables (int p(x) f(x) dx = int p(e^u) f(e^u) e^u du,
         *   u = ln(x)) that leaves the value of the integral, f's own
         *   required signature, and the coordinate f/p are evaluated
         *   at (always the real, untransformed x, regardless of this
         *   parameter) all unchanged -- it only changes which
         *   variable GKIntegrator itself subdivides. Defaults to
         *   false (no transform). integrate() throws
         *   std::runtime_error if this is true but its own integration
         *   bounds (after clamping to p's own support) are not both
         *   strictly positive, since ln() of a non-positive bound is
         *   undefined.
         * @param maxIter See GKIntegrator's own maxIter parameter
         * @param absTol See GKIntegrator's own absTol parameter
         * @param relTol See GKIntegrator's own relTol parameter
         */
        PDFIntegrator(
            const pdfs::PDF& p,
            F f,
            std::size_t nInt,
            bool logTransform = false,
            std::size_t maxIter = 0,
            double absTol = 0.0,
            double relTol = 1e-6
        ) :
            p_(p),
            f_(std::move(f)),
            nInt_(nInt),
            logTransform_(logTransform),
            maxIter_(maxIter),
            absTol_(absTol),
            relTol_(relTol)
        { }

        // Disallow copying and moving: p_ holds a reference, so a copy
        // would alias the same PDF as the original rather than being
        // an independent instance, and there is no real need to copy
        // or move a PDFIntegrator in practice, since it is meant to be
        // constructed and used immediately
        PDFIntegrator(const PDFIntegrator&) = delete;
        PDFIntegrator(PDFIntegrator&&) = delete;
        auto operator=(const PDFIntegrator&) -> PDFIntegrator& = delete;
        auto operator=(PDFIntegrator&&) -> PDFIntegrator& = delete;
        ~PDFIntegrator() = default;

        /**
         * @brief Evaluate p(x) f(x) at a single point, for GKIntegrator's own benefit
         * @param x The integration variable -- if logTransform_ is
         *   true, this is ln of the real coordinate (the domain
         *   integrate() itself hands GKIntegrator), not the real
         *   coordinate directly
         * @param args Any additional arguments f_ requires; if F is a
         *   pointer to member function, the first of these must be an
         *   instance of the class it is a member of (see this class's
         *   own @tparam F)
         * @return p(xReal) * f(xReal, args...), elementwise, further
         *   scaled by xReal (the change-of-variables Jacobian
         *   dx = x d(ln x)) if logTransform_ is true, where xReal is x
         *   itself (logTransform_ false) or exp(x) (logTransform_
         *   true), clamped into [aReal_, bReal_]
         * @details
         * First undoes the log transform, if any (exp(x) -> xReal),
         * then clamps xReal into [aReal_, bReal_] -- the current
         * integrate() call's own real-space integration bounds (see
         * their own comment for why these, not p_'s own full
         * getMin()/getMax(), are the correct clamp target: an
         * isochrone-segment-scale sub-range of p_'s own domain is the
         * common case in this codebase, and clamping to p_'s own full
         * domain instead could let a point land outside such a
         * segment's own valid range). This clamp matters because exp()
         * and ln() are not exact inverses under floating point, so
         * exp(x) can land fractionally outside [aReal_, bReal_] even
         * when x itself was exactly ln(aReal_)/ln(bReal_)'s own
         * (correctly clamped) value.
         *
         * Evaluates f_ at xReal (via invokeF(), which transparently
         * supports F being a pointer to member function -- see its
         * own comment), multiplies elementwise by p_(xReal), and, if
         * logTransform_, by xReal again (the Jacobian).
         *
         * Args&... here, not the more usual Args&&... forwarding
         * reference: integrate() only ever names this function
         * through &PDFIntegrator::template integrand<Args...>, with
         * Args already fixed to its own deduced types -- at that
         * point Args&& is no longer a forwarding reference at all,
         * just an ordinary (possibly rvalue) reference type, and for
         * any Args element integrate() deduced as a plain value (e.g.
         * a pointer instance passed as a prvalue, as PDFIntegrator's
         * own integrate() itself does when it is F for an *outer*
         * GKIntegrator/PDFIntegrator -- see specsyn::Specsyn::
         * specCtsHelper() for a real example, nesting one
         * PDFIntegrator inside another this way), that becomes a
         * genuine rvalue-reference parameter. But GKIntegrator's own
         * internal machinery (see its quadSingle()/integrate() own
         * comments on argsTuple) always hands every call an lvalue,
         * to stay safe across the many repeated calls one adaptive
         * integral makes -- so a plain lvalue reference is the only
         * parameter type guaranteed to bind correctly here, regardless
         * of what value category the original argument at
         * integrate()'s own call site had.
         */
        template <class... Args>
        [[nodiscard]] auto integrand(const double x, Args&... args) const -> std::vector<double>
        {
            const double xReal = logTransform_ ? std::exp(x) : x;
            const double xClamped = std::clamp(xReal, aReal_, bReal_);

            const auto val = invokeF(xClamped, args...);

            double weight = p_(xClamped);
            if (logTransform_) { weight *= xClamped; }

            std::vector<double> result(std::begin(val), std::end(val));
            for (double& v : result) { v *= weight; }
            return result;
        }

        /**
         * @brief Evaluate the integral of p(x) f(x) from a to b
         * @param a Lower limit of integration
         * @param b Upper limit of integration
         * @param args Any additional arguments f_ requires, exactly
         *   as they would be passed to integrand()
         * @return The integral, one value per quantity f_ returns
         *   (length nInt_)
         * @throws std::runtime_error if logTransform_ is true, but a/b
         *   (after clamping to p_'s own support) are not both strictly
         *   positive
         * @details
         * Clamps [a, b] to p_'s own support, records the result in
         * aReal_/bReal_ for integrand()'s own benefit (see its own
         * comment for why), log-transforms a/b themselves if
         * logTransform_ is set, then constructs a GKIntegrator whose
         * own integrand is this class's integrand() (a pointer to
         * member function, so this is passed as the first extra
         * argument to GKIntegrator::integrate() below -- see
         * GKIntegrator's own @tparam F comment), passing maxIter_/
         * absTol_/relTol_ through unchanged, and returns its result.
         */
        template <class... Args>
        [[nodiscard]] auto integrate(double a, double b, Args&&... args) const -> std::vector<double>
        {
            a = std::max(a, p_.getMin());
            b = std::min(b, p_.getMax());
            aReal_ = a;
            bReal_ = b;

            if (logTransform_)
            {
                if (a <= 0.0 || b <= 0.0)
                {
                    throw std::runtime_error(
                        "PDFIntegrator::integrate: logTransform is set, but the "
                        "integration bounds (after clamping to the PDF's own support) "
                        "are not both strictly positive: a = " + std::to_string(a) +
                        ", b = " + std::to_string(b));
                }
                a = std::log(a);
                b = std::log(b);
            }

            using IntegrandFn = decltype(&PDFIntegrator::template integrand<Args...>);
            const GKIntegrator<IntegrandFn, Order> integ(
                &PDFIntegrator::template integrand<Args...>, nInt_, maxIter_, absTol_, relTol_);
            return integ.integrate(a, b, this, std::forward<Args>(args)...);
        }

    private:

        /**
         * @brief Evaluate f_ at x, transparently supporting F being a pointer to member function
         * @param x The integration variable, already untransformed
         *   and clamped -- see integrand()'s own comment
         * @param args Any additional arguments f_ requires -- if F is
         *   a pointer to member function, the first of these must be
         *   an instance of the class it is a member of (see this
         *   class's own @tparam F)
         * @return f_'s own return value at (x, args...)
         * @details
         * Mirrors GKIntegrator::invokeF()'s own identical dispatch
         * (see its own comment): for a standalone callable, forwards
         * straight to std::invoke(f_, x, args...); for a pointer to
         * member function, hands off to invokeFMember() instead,
         * which reorders arguments into the shape std::invoke's own
         * INVOKE protocol requires.
         */
        template <class... Args>
        [[nodiscard]] auto invokeF(const double x, Args&&... args) const
        {
            if constexpr (std::is_member_function_pointer_v<F>)
            {
                return invokeFMember(x, std::forward<Args>(args)...);
            }
            else
            {
                return std::invoke(f_, x, std::forward<Args>(args)...);
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

        const pdfs::PDF& p_; /**< The PDF weighting the integrand */
        F f_;                 /**< The integrand */
        std::size_t nInt_;    /**< Number of quantities f returns per point */
        bool logTransform_;   /**< Whether to integrate over ln(x) instead of x */
        std::size_t maxIter_; /**< See GKIntegrator's own maxIter_ */
        double absTol_;       /**< Required absolute error */
        double relTol_;       /**< Required relative error */

        // The current integrate() call's own real-space integration
        // bounds (after clamping to p_'s own support) -- set at the
        // top of integrate(), read by integrand() (see its own
        // comment for why) -- mutable so integrate() itself can stay
        // const. See this class's own @details for the concurrency
        // trade-off this makes.
        mutable double aReal_ = 0.0;
        mutable double bReal_ = 0.0;
    };

} // namespace utils

#endif // PDFINTEGRATOR_HPP
