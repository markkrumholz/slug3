/**
 * @file PDFIntegratorND.hpp
 * @author Mark Krumholz
 * @brief Evaluate integrals of the form int_a1^b1 ... int_aN^bN p_1(x_1) ... p_N(x_N) f(x) d^N x
 * @date 2026-08-10
 */

#ifndef PDFINTEGRATORND_HPP
#define PDFINTEGRATORND_HPP

#include "../pdfs/PDF.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cubature.h> // NOLINT(misc-include-cleaner)
#include <functional>
#include <span>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace utils
{

    /**
     * @class PDFIntegratorND
     * @brief Evaluate integrals of the form int p_1(x_1) ... p_N(x_N) f(x) d^N x
     * @tparam F Type of the callable integrand. Must accept a
     *   std::span<double> of length N (the integration variables, one
     *   per dimension) as its first argument, plus any number of
     *   additional arguments (the first of which must be an instance
     *   of the owning class, if F is a pointer to member function),
     *   and return a fixed- or dynamically-sized contiguous container
     *   of doubles (e.g. std::vector<double> or std::array<double, M>).
     * @tparam N Number of integration dimensions
     * @details
     * Generalizes PDFIntegrator to N dimensions: this class evaluates
     * integrals of the form
     * int p_1(x_1) p_2(x_2) ... p_N(x_N) f(x) d^N x, where each
     * p_i(x_i) is a pdfs::PDF weighting a single component of the
     * N-dimensional integration variable x, f(x) is an arbitrary,
     * user-specified, vector-valued function -- either a standalone
     * callable or a class member function -- and the integration
     * range is the tensor product of each dimension's own [a_i, b_i]
     * interval. Integration is performed with the cubature package's
     * h-adaptive routine, hcubature, which (unlike PDFIntegrator's own
     * p-adaptive pcubature) subdivides the integration domain rather
     * than raising the degree of a single tensor-product quadrature
     * rule, and so scales far better to N > 1 dimensions. In practice
     * this class is expected to be used to integrate quantities
     * against the joint distribution of a continuously-sampled stellar
     * population's mass, age, and metallicity.
     */
    template <class F, std::size_t N>
    class PDFIntegratorND
    {
    public:

        /**
         * @brief Construct a PDFIntegratorND
         * @param p The N PDFs p_1(x_1), ..., p_N(x_N) weighting each
         *   dimension of the integrand; every element of p must
         *   outlive this PDFIntegratorND, since each is stored by
         *   reference
         * @param f The integrand f(x); may be a standalone callable
         *   or a pointer to a member function, in which case the
         *   first of the additional arguments passed to operator()
         *   or integrate() must be an instance of the class f is a
         *   member of
         * @param nInt The number of quantities f returns (i.e. the
         *   size of the container returned by f)
         * @param maxEval Maximum number of integrand evaluations
         *   hcubature is allowed to make; 0 means no limit
         * @param reqAbsError Required absolute error
         * @param reqRelError Required relative error
         * @param norm Method used to combine the per-component error
         *   estimates into the single value checked against
         *   reqAbsError/reqRelError; see cubature.h's error_norm
         */
        PDFIntegratorND(
            std::array<std::reference_wrapper<const pdfs::PDF>, N> p,
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

        // Disallow copying and moving -- see PDFIntegrator's own
        // identical comment: p_ holds references, so a copy would
        // alias the same PDFs as the original rather than being an
        // independent instance, and there is no real need to copy or
        // move a PDFIntegratorND in practice, since it is meant to be
        // constructed and used immediately
        PDFIntegratorND(const PDFIntegratorND&) = delete;
        PDFIntegratorND(PDFIntegratorND&&) = delete;
        auto operator=(const PDFIntegratorND&) -> PDFIntegratorND& = delete;
        auto operator=(PDFIntegratorND&&) -> PDFIntegratorND& = delete;
        ~PDFIntegratorND() = default;

        /**
         * @brief Evaluate f at a point
         * @param x The point (one value per dimension) at which to
         *   evaluate f
         * @param args Any additional arguments f requires; if f is a
         *   pointer to member function, the first of these must be
         *   an instance of the class it is a member of
         * @return The value of f(x, args...)
         */
        template <class... Args>
        [[nodiscard]] auto operator()(std::span<double> x, Args&&... args) const
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
         * @brief Evaluate the integral of p_1(x_1) ... p_N(x_N) f(x) over the box [a, b]
         * @param a Lower limit of integration in each dimension
         * @param b Upper limit of integration in each dimension
         * @param args Any additional arguments f requires, exactly as
         *   they would be passed to operator()
         * @return The integral, in a container of the same type
         *   returned by f
         */
        template <class... Args>
        [[nodiscard]] auto integrate(std::array<double, N> a,
            std::array<double, N> b, Args&&... args) const
        {
            using Result = decltype((*this)(std::span<double>(a), args...));

            // Restrict each dimension's [a_i, b_i] to p_i's own
            // support -- see PDFIntegrator::integrate()'s own
            // identical comment on why. Unchecked indexing throughout
            // this performance-critical class: d < N by every loop's
            // own condition, and a/b/p_ are all N-element containers
            // by construction.
            for (std::size_t d = 0; d < N; ++d)
            {
                a[d] = std::max(a[d], p_[d].get().getMin()); // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index)
                b[d] = std::min(b[d], p_[d].get().getMax()); // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index)
            }

            // Bundle a pointer back to this PDFIntegratorND, the
            // (already-clamped) integration bounds, and the extra
            // arguments into a context local to this call -- see
            // PDFIntegrator::integrate()'s own identical comment on
            // why this makes integrate() safe to call concurrently on
            // a single, shared PDFIntegratorND instance, and on why
            // Args are stored exactly as deduced rather than decayed.
            Context<Args...> ctx{ this, a, b, std::tuple<Args...>(std::forward<Args>(args)...) };

            Result result{};
            if constexpr (requires { result.resize(nInt_); })
            {
                result.resize(nInt_);
            }
            std::vector<double> errBuf(nInt_); // discarded; hcubature requires it regardless

            hcubature(nInt_, &integrand<Args...>, &ctx, N, a.data(), b.data(),
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
        [[nodiscard]] auto invokeMember(std::span<double> x, Obj&& obj, Rest&&... rest) const
        {
            return std::invoke(f_, std::forward<Obj>(obj), x, std::forward<Rest>(rest)...);
        }

        // Bundles a pointer back to this PDFIntegratorND, the
        // (already-clamped) integration bounds, and the extra
        // arguments passed to integrate(), so integrand() can recover
        // all of them from the single void* fdata hcubature hands it
        template <class... Args>
        struct Context
        {
            const PDFIntegratorND* self_;
            std::array<double, N> a_;
            std::array<double, N> b_;
            std::tuple<Args...> args_;
        };

        // The hcubature-compatible trampoline: unpacks the Context
        // pointed to by fdata, evaluates
        // p_1(x_1) * ... * p_N(x_N) * f(x, args...) at the single
        // point x (an N-element array in hcubature's own convention,
        // clamped element-wise into [ctx->a_, ctx->b_] -- see
        // PDFIntegrator::integrand()'s own identical comment on why),
        // and writes the result into fval. The clamped coordinates are
        // copied into a local buffer (rather than clamping x itself,
        // which hcubature owns) so that buffer can be handed to f as
        // a std::span<double>.
        template <class... Args>
        static auto integrand(unsigned /*ndim*/, const double* x, void* fdata,
            unsigned /*fdim*/, double* fval) -> int
        {
            const auto* ctx = static_cast<Context<Args...>*>(fdata);

            std::array<double, N> xClamped{};
            for (std::size_t d = 0; d < N; ++d)
            {
                // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index,cppcoreguidelines-pro-bounds-pointer-arithmetic) -- x is hcubature's own N-element array, guaranteed length N by construction; xClamped/ctx->a_/ctx->b_ are all N-element containers
                xClamped[d] = std::clamp(x[d], ctx->a_[d], ctx->b_[d]);
            }

            double weight = 1.0;
            for (std::size_t d = 0; d < N; ++d)
            {
                weight *= ctx->self_->p_[d].get()(xClamped[d]); // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index)
            }

            const std::span<double> xSpan(xClamped);
            const auto val = std::apply(
                [&](auto&&... args) -> auto { return (*ctx->self_)(xSpan, std::forward<decltype(args)>(args)...); },
                ctx->args_);
            std::ranges::transform(val, fval,
                [weight](const double v) -> double { return weight * v; });
            return 0;
        }

        std::array<std::reference_wrapper<const pdfs::PDF>, N> p_; /**< The PDFs weighting each dimension of the integrand */
        F f_;                      /**< The integrand */
        unsigned nInt_;            /**< Number of quantities f returns */
        std::size_t maxEval_;      /**< Maximum number of integrand evaluations */
        double reqAbsError_;       /**< Required absolute error */
        double reqRelError_;       /**< Required relative error */
        error_norm norm_;          /**< Error norm used to combine per-component errors */ // NOLINT(misc-include-cleaner)
    };

} // namespace utils

#endif // PDFINTEGRATORND_HPP
