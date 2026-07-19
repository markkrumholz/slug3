/**
 * @file PDFIntegrator.hpp
 * @author Mark Krumholz
 * @brief Evaluate integrals of the form int_a^b p(x) f(x) dx
 * @date 2026-07-19
 */

#ifndef PDFINTEGRATOR_HPP
#define PDFINTEGRATOR_HPP

#include "../pdfs/PDF.hpp"
#include <algorithm>
#include <cstddef>
#include <cubature.h> // NOLINT(misc-include-cleaner)
#include <functional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace utils
{

    /**
     * @class PDFIntegrator
     * @brief Evaluate integrals of the form int_a^b p(x) f(x) dx
     * @tparam F Type of the callable integrand. Must accept a double
     *   (the integration variable) as its first argument, plus any
     *   number of additional arguments (the first of which must be
     *   an instance of the owning class, if F is a pointer to member
     *   function), and return a fixed- or dynamically-sized
     *   contiguous container of doubles (e.g. std::vector<double> or
     *   std::array<double, N>).
     * @details
     * This class evaluates integrals of the form int_a^b p(x) f(x)
     * dx, where p(x) is a pdfs::PDF and f(x) is an arbitrary,
     * user-specified, vector-valued function -- either a standalone
     * callable or a class member function. Integration is performed
     * with the cubature package's p-adaptive routine, pcubature.
     * Although this class is general purpose, in practice it will
     * most often be used to integrate quantities against a stellar
     * IMF.
     */
    template <class F>
    class PDFIntegrator
    {
    public:

        /**
         * @brief Construct a PDFIntegrator
         * @param p The PDF p(x) that weights the integrand; p must
         *   outlive this PDFIntegrator, since it is stored by
         *   reference
         * @param f The integrand f(x); may be a standalone callable
         *   or a pointer to a member function, in which case the
         *   first of the additional arguments passed to operator()
         *   or integrate() must be an instance of the class f is a
         *   member of
         * @param nInt The number of quantities f returns (i.e. the
         *   size of the container returned by f)
         * @param maxEval Maximum number of integrand evaluations
         *   pcubature is allowed to make; 0 means no limit
         * @param reqAbsError Required absolute error
         * @param reqRelError Required relative error
         * @param norm Method used to combine the per-component error
         *   estimates into the single value checked against
         *   reqAbsError/reqRelError; see cubature.h's error_norm
         */
        PDFIntegrator(
            const pdfs::PDF& p,
            F f,
            unsigned nInt,
            std::size_t maxEval = 0,
            double reqAbsError = 0.0,
            double reqRelError = 1e-6,
            error_norm norm = ERROR_INDIVIDUAL // NOLINT(misc-include-cleaner)
        ) :
        p_(p),
        f_(std::move(f)),
        nInt_(nInt),
        maxEval_(maxEval),
        reqAbsError_(reqAbsError),
        reqRelError_(reqRelError),
        norm_(norm)
        { }

        // Disallow copying and moving: p_ is a reference, so a copy
        // would alias the same PDF as the original rather than being
        // an independent instance, which is surprising; there is no
        // real need to copy or move a PDFIntegrator in practice, since
        // it is meant to be constructed and used immediately
        PDFIntegrator(const PDFIntegrator&) = delete;
        PDFIntegrator(PDFIntegrator&&) = delete;
        auto operator=(const PDFIntegrator&) -> PDFIntegrator& = delete;
        auto operator=(PDFIntegrator&&) -> PDFIntegrator& = delete;
        ~PDFIntegrator() = default;

        /**
         * @brief Evaluate f at a point
         * @param x The point at which to evaluate f
         * @param args Any additional arguments f requires; if f is a
         *   pointer to member function, the first of these must be
         *   an instance of the class it is a member of
         * @return The value of f(x, args...)
         */
        template <class... Args>
        [[nodiscard]] auto operator()(double x, Args&&... args) const
        {
            // std::invoke's INVOKE protocol calls a pointer to member
            // function as invoke(f, instance, args...) -- the
            // instance must come immediately after f, not after x --
            // whereas the API here takes x first and the instance
            // second (as the first of args), so that case needs its
            // arguments reordered before calling std::invoke.
            if constexpr (std::is_member_function_pointer_v<F>)
            {
                return invokeMember(x, std::forward<Args>(args)...);
            }
            else
            {
                return std::invoke(f_, x, std::forward<Args>(args)...);
            }
        }

        /**
         * @brief Evaluate the integral of p(x) f(x) from a to b
         * @param a Lower limit of integration
         * @param b Upper limit of integration
         * @param args Any additional arguments f requires, exactly as
         *   they would be passed to operator()
         * @return The integral, in a container of the same type
         *   returned by f
         */
        template <class... Args>
        [[nodiscard]] auto integrate(double a, double b, Args&&... args) const
        {
            using Result = decltype((*this)(a, args...));

            // Restrict [a, b] to p's own support: a caller may
            // legitimately ask for an integral over a range wider
            // than where p is defined (e.g. integrating over
            // [0, 2] when p is only defined on [1, 2]), and p itself
            // is not guaranteed to be evaluable outside its own
            // domain.
            a = std::max(a, p_.getMin());
            b = std::min(b, p_.getMax());

            // Bundle a pointer back to this PDFIntegrator, the
            // (already-clamped) integration bounds, and the extra
            // arguments into a context local to this call, so
            // integrand() (which pcubature calls back with only a
            // single void* to work with) can reach all of them.
            // integrand() re-clamps each point pcubature hands it
            // into [a, b], since pcubature computes its evaluation
            // points via a floating-point affine map from its
            // reference interval onto [a, b] that can land a point a
            // few ULPs outside [a, b] at the very edge, which would
            // otherwise reach f() with an out-of-domain argument.
            // Being local to this call, this makes integrate() safe
            // to call concurrently on a single, shared PDFIntegrator
            // instance -- e.g. from different threads -- since no
            // state is shared between calls. Args are stored exactly
            // as deduced (so an lvalue argument is stored as a
            // reference, not copied) rather than decayed: integrate()
            // blocks on pcubature, so ctx does not need to outlive
            // the caller's own arguments, and some arguments (e.g. an
            // Isochrone, a vector of unique_ptr's) are not copyable
            // at all.
            Context<Args...> ctx{ this, a, b, std::tuple<Args...>(std::forward<Args>(args)...) };

            Result result{};
            if constexpr (requires { result.resize(nInt_); })
            {
                result.resize(nInt_);
            }
            std::vector<double> errBuf(nInt_); // discarded; pcubature requires it regardless

            pcubature(nInt_, &integrand<Args...>, &ctx, 1, &a, &b,
                maxEval_, reqAbsError_, reqRelError_, norm_,
                std::data(result), errBuf.data());

            return result;
        }

    private:

        // Helper for operator(): splits the (x, instance, rest...)
        // argument order this class's API uses into the
        // (instance, x, rest...) order std::invoke's INVOKE protocol
        // requires to dispatch a pointer to member function
        template <class Obj, class... Rest>
        [[nodiscard]] auto invokeMember(double x, Obj&& obj, Rest&&... rest) const
        {
            return std::invoke(f_, std::forward<Obj>(obj), x, std::forward<Rest>(rest)...);
        }

        // Bundles a pointer back to this PDFIntegrator, the
        // (already-clamped) integration bounds, and the extra
        // arguments passed to integrate(), so integrand() can recover
        // all of them from the single void* fdata pcubature hands it
        template <class... Args>
        struct Context
        {
            const PDFIntegrator* self_;
            double a_;
            double b_;
            std::tuple<Args...> args_;
        };

        // The pcubature-compatible trampoline: unpacks the Context
        // pointed to by fdata, evaluates p(x) * f(x, args...) at the
        // single point *x (clamped into [ctx->a_, ctx->b_], since
        // pcubature's own evaluation points can land a few ULPs
        // outside that range at the very edge due to floating-point
        // roundoff), and writes the result into fval
        template <class... Args>
        static auto integrand(unsigned /*ndim*/, const double* x, void* fdata,
            unsigned /*fdim*/, double* fval) -> int
        {
            const auto* ctx = static_cast<Context<Args...>*>(fdata);
            const double xClamped = std::clamp(*x, ctx->a_, ctx->b_);
            const double weight = ctx->self_->p_(xClamped);
            const auto val = std::apply(
                [&](auto&&... args) -> auto { return (*ctx->self_)(xClamped, std::forward<decltype(args)>(args)...); },
                ctx->args_);
            std::ranges::transform(val, fval,
                [weight](const double v) -> double { return weight * v; });
            return 0;
        }

        const pdfs::PDF& p_;       /**< The PDF weighting the integrand */
        F f_;                      /**< The integrand */
        unsigned nInt_;            /**< Number of quantities f returns */
        std::size_t maxEval_;      /**< Maximum number of integrand evaluations */
        double reqAbsError_;       /**< Required absolute error */
        double reqRelError_;       /**< Required relative error */
        error_norm norm_;          /**< Error norm used to combine per-component errors */ // NOLINT(misc-include-cleaner)
    };

} // namespace utils

#endif // PDFINTEGRATOR_HPP
