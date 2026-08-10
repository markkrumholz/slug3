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
#include <mdspan> // NOLINT(misc-include-cleaner)
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
     * @tparam F Type of the callable integrand.
     *   If Vectorized is false, F must accept a std::span<double> of
     *   length N (a single point's integration variables, one per
     *   dimension) as its first argument, plus any number of
     *   additional arguments (the first of which must be an instance
     *   of the owning class, if F is a pointer to member function),
     *   and return a fixed- or dynamically-sized contiguous container
     *   of doubles of length nInt (e.g. std::vector<double> or
     *   std::array<double, nInt>) -- the value of the integrand at
     *   that one point.
     *   If Vectorized is true, F must instead accept a
     *   std::mdspan<double, std::extents<std::size_t,
     *   std::dynamic_extent, N>> of shape (npts, N) -- npts points,
     *   read as points[i, d] for the d-th coordinate of the i-th point
     *   -- as its first argument (again followed by any number of
     *   additional arguments), and return a dynamically-sized
     *   contiguous container of npts * nInt doubles, laid out as
     *   result[i * nInt + k] for the k-th quantity at the i-th point
     *   (the same layout cubature.h's own integrand_v uses for fval),
     *   evaluating the integrand at every one of the npts points in a
     *   single call.
     * @tparam N Number of integration dimensions
     * @tparam Vectorized Whether to use cubature's vectorized
     *   interface (hcubature_v), which hands F every point it needs
     *   evaluated at a given refinement step in one call, rather than
     *   the plain interface (hcubature), which calls F once per point.
     *   Defaults to false. Batching evaluations this way is
     *   substantially more efficient for an integrand whose per-point
     *   cost is dominated by work that can be shared across points
     *   (e.g. constructing one isochrone per distinct age and
     *   evaluating every point at that age against it, rather than
     *   rebuilding the isochrone from scratch for each point
     *   individually) -- but requires F itself to be written to take
     *   advantage of that batching, which is why this is a template
     *   parameter (selecting F's own required signature) rather than a
     *   runtime option.
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
     * h-adaptive routine, hcubature (or its vectorized counterpart,
     * hcubature_v, if Vectorized is true), which (unlike
     * PDFIntegrator's own p-adaptive pcubature) subdivides the
     * integration domain rather than raising the degree of a single
     * tensor-product quadrature rule, and so scales far better to
     * N > 1 dimensions. In practice this class is expected to be used
     * to integrate quantities against the joint distribution of a
     * continuously-sampled stellar population's mass, age, and
     * metallicity.
     */
    template <class F, std::size_t N, bool Vectorized = false>
    class PDFIntegratorND
    {
    public:

        // The N-dimensional extents type of a single hcubature_v
        // callback's own batch of points: npts (dynamic, known only
        // at call time) rows of N (fixed, known at compile time)
        // coordinates each.
        using PointsExtents = std::extents<std::size_t, std::dynamic_extent, N>; // NOLINT(misc-include-cleaner)

        // An (npts, N) view over one hcubature_v callback's own batch
        // of points, read as points[i, d] for the d-th coordinate of
        // the i-th point -- the type F itself must accept when
        // Vectorized is true; see this class's own @tparam F.
        using PointsView = std::mdspan<double, PointsExtents>; // NOLINT(misc-include-cleaner)

        // The type operator()/invokeMember/f_ itself is actually
        // called with: a single point (as a span) if Vectorized is
        // false, or a whole batch of points (as an mdspan) if
        // Vectorized is true. Declared up front (rather than beside
        // the other private implementation details below) since it
        // names the parameter type of operator(), declared next.
        using Arg = std::conditional_t<Vectorized, PointsView, std::span<double>>;

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
         *   member of. See this class's own @tparam F for the
         *   signature f must have, which depends on Vectorized.
         * @param nInt The number of quantities f returns per point
         *   (i.e. the size of the container returned by f, if
         *   Vectorized is false; or 1/npts of the size of the
         *   container returned by f, if Vectorized is true)
         * @param maxEval Maximum number of integrand evaluations
         *   hcubature/hcubature_v is allowed to make; 0 means no limit
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
         * @brief Evaluate f at a point (Vectorized == false) or a batch of points (Vectorized == true)
         * @param x The point(s) at which to evaluate f -- a
         *   std::span<double> of length N if Vectorized is false, or a
         *   std::mdspan of shape (npts, N) if Vectorized is true; see
         *   this class's own @tparam F for the exact contract
         * @param args Any additional arguments f requires; if f is a
         *   pointer to member function, the first of these must be
         *   an instance of the class it is a member of
         * @return The value of f(x, args...)
         */
        template <class... Args>
        [[nodiscard]] auto operator()(Arg x, Args&&... args) const
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
            using Result = decltype((*this)(std::declval<Arg>(), args...));

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
            std::vector<double> errBuf(nInt_); // discarded; hcubature/hcubature_v requires it regardless

            if constexpr (Vectorized)
            {
                hcubature_v(nInt_, &integrandV<Args...>, &ctx, N, a.data(), b.data(),
                    maxEval_, reqAbsError_, reqRelError_, norm_,
                    std::data(result), errBuf.data());
            }
            else
            {
                hcubature(nInt_, &integrand<Args...>, &ctx, N, a.data(), b.data(),
                    maxEval_, reqAbsError_, reqRelError_, norm_,
                    std::data(result), errBuf.data());
            }

            return result;
        }

    private:

        // Helper for operator(): splits the (x, instance, rest...)
        // argument order this class's API uses into the
        // (instance, x, rest...) order std::invoke's INVOKE protocol
        // requires to dispatch a pointer to member function
        template <class Obj, class... Rest>
        [[nodiscard]] auto invokeMember(Arg x, Obj&& obj, Rest&&... rest) const
        {
            return std::invoke(f_, std::forward<Obj>(obj), x, std::forward<Rest>(rest)...);
        }

        // Bundles a pointer back to this PDFIntegratorND, the
        // (already-clamped) integration bounds, and the extra
        // arguments passed to integrate(), so integrand()/integrandV()
        // can recover all of them from the single void* fdata
        // hcubature/hcubature_v hands them
        template <class... Args>
        struct Context
        {
            const PDFIntegratorND* self_;
            std::array<double, N> a_;
            std::array<double, N> b_;
            std::tuple<Args...> args_;
        };

        // The hcubature-compatible trampoline (Vectorized == false):
        // unpacks the Context pointed to by fdata, evaluates
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

        // The hcubature_v-compatible trampoline (Vectorized == true):
        // like integrand() above, but handling an entire batch of npt
        // points -- laid out by hcubature_v itself as
        // x[i * ndim + d] for the d-th coordinate of the i-th point,
        // and expected back in fval as fval[i * fdim + k] for the
        // k-th quantity at the i-th point -- in a single call, so that
        // f can share work (e.g. one isochrone per distinct age)
        // across every point in the batch. Mirrors integrand()'s own
        // clamp-into-a-local-buffer approach, but for the whole batch
        // at once, so that buffer can be handed to f as a single
        // PointsView.
        template <class... Args>
        static auto integrandV(unsigned /*ndim*/, std::size_t npt, const double* x,
            void* fdata, unsigned fdim, double* fval) -> int
        {
            const auto* ctx = static_cast<Context<Args...>*>(fdata);

            std::vector<double> xClamped(npt * N);
            for (std::size_t i = 0; i < npt; ++i)
            {
                for (std::size_t d = 0; d < N; ++d)
                {
                    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index,cppcoreguidelines-pro-bounds-pointer-arithmetic) -- x is hcubature_v's own (npt, ndim) array, guaranteed length npt * N by construction; xClamped is sized npt * N just above
                    xClamped[(i * N) + d] = std::clamp(x[(i * N) + d], ctx->a_[d], ctx->b_[d]);
                }
            }

            std::vector<double> weight(npt, 1.0);
            for (std::size_t i = 0; i < npt; ++i)
            {
                for (std::size_t d = 0; d < N; ++d)
                {
                    weight[i] *= ctx->self_->p_[d].get()(xClamped[(i * N) + d]); // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index)
                }
            }

            const PointsView points(xClamped.data(), npt);
            const auto val = std::apply(
                [&](auto&&... args) -> auto { return (*ctx->self_)(points, std::forward<decltype(args)>(args)...); },
                ctx->args_);

            // val holds npt * fdim values, laid out identically to
            // fval (val[i * fdim + k] is the k-th quantity at the
            // i-th point, per this class's own @tparam F contract) --
            // weight each point's own block of fdim values by that
            // point's own weight and copy into fval
            for (std::size_t i = 0; i < npt; ++i)
            {
                for (unsigned k = 0; k < fdim; ++k)
                {
                    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index,cppcoreguidelines-pro-bounds-pointer-arithmetic) -- fval is hcubature_v's own (npt, fdim) array, guaranteed length npt * fdim by construction; val is guaranteed the same length by this class's own @tparam F contract
                    fval[(i * fdim) + k] = weight[i] * val[(i * fdim) + k];
                }
            }
            return 0;
        }

        std::array<std::reference_wrapper<const pdfs::PDF>, N> p_; /**< The PDFs weighting each dimension of the integrand */
        F f_;                      /**< The integrand */
        unsigned nInt_;            /**< Number of quantities f returns per point */
        std::size_t maxEval_;      /**< Maximum number of integrand evaluations */
        double reqAbsError_;       /**< Required absolute error */
        double reqRelError_;       /**< Required relative error */
        error_norm norm_;          /**< Error norm used to combine per-component errors */ // NOLINT(misc-include-cleaner)
    };

} // namespace utils

#endif // PDFINTEGRATORND_HPP
