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
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cubature.h> // NOLINT(misc-include-cleaner)
#include <functional>
#include <mdspan> // NOLINT(misc-include-cleaner)
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace utils
{

    // A dependent "false" usable inside a discarded if-constexpr
    // branch's static_assert, so the assert only fires if that branch
    // is actually reached for some real instantiation -- a bare
    // static_assert(false, ...) in such a branch is ill-formed
    // regardless of whether the branch is ever taken on some
    // toolchains, since it doesn't depend on any template parameter.
    template <class...>
    inline constexpr bool alwaysFalse = false;

    /**
     * @brief Selects which cubature package routine PDFIntegratorND uses
     * @details
     * hAdaptive subdivides the integration domain into progressively
     * smaller regions, concentrating effort where the integrand
     * varies most -- well suited to integrands with sharp, localized
     * features, but each subdivision's own quadrature points are
     * recomputed from scratch, so identical coordinate values recur
     * across separate evaluations only incidentally (e.g. when a
     * region happens to be split along a different dimension than a
     * sibling). pAdaptive instead keeps the domain fixed and raises
     * the degree of a single nested Clenshaw-Curtis product rule one
     * dimension at a time, explicitly caching every point ever
     * evaluated so no coordinate value is ever requested twice over
     * the life of one integral -- better suited to smooth integrands,
     * and far friendlier to caching per-coordinate work (e.g. one
     * isochrone per distinct age, reused across every mass point
     * evaluated at that age) that is shared across many points.
     */
    enum class CubatureMethod : std::uint8_t
    {
        hAdaptive, /**< cubature's h-adaptive routine (hcubature/hcubature_v) */
        pAdaptive  /**< cubature's p-adaptive routine (pcubature/pcubature_v) */
    };

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
     * @tparam Method Which cubature package routine to integrate with;
     *   see CubatureMethod's own comment for the tradeoffs. Defaults
     *   to CubatureMethod::hAdaptive, which is the better default for
     *   N > 1 (pAdaptive's single global tensor-product grid scales
     *   far worse with dimension than hAdaptive's domain subdivision)
     *   and for integrands with sharp, localized features anywhere in
     *   the domain. CubatureMethod::pAdaptive is worth selecting
     *   instead when the integrand is smooth and per-point cost is
     *   dominated by work shareable across many points at a fixed
     *   coordinate in one dimension (e.g. one isochrone per distinct
     *   age): pAdaptive's own nested-quadrature point cache guarantees
     *   no coordinate value is ever re-requested over the life of one
     *   integral, so pairing it with Vectorized == true and an
     *   F-side cache keyed on that dimension's coordinate gives every
     *   cached value a guaranteed hit on every later reuse, unlike
     *   hAdaptive's domain subdivision, where a coordinate recurring
     *   across separate evaluations is possible but incidental, not
     *   guaranteed.
     * @details
     * Generalizes PDFIntegrator to N dimensions -- indeed, PDFIntegrator
     * is a thin alias for PDFIntegratorND<F, 1> (see PDFIntegrator.hpp),
     * so this class is the sole implementation for every N, including
     * N = 1. This class evaluates integrals of the form
     * int p_1(x_1) p_2(x_2) ... p_N(x_N) f(x) d^N x, where each
     * p_i(x_i) is a pdfs::PDF weighting a single component of the
     * N-dimensional integration variable x, f(x) is an arbitrary,
     * user-specified, vector-valued function -- either a standalone
     * callable or a class member function -- and the integration
     * range is the tensor product of each dimension's own [a_i, b_i]
     * interval. Integration is performed with the cubature package's
     * h-adaptive routine, hcubature/hcubature_v, or its p-adaptive
     * routine, pcubature/pcubature_v, according to Method (see its own
     * @tparam comment above for the tradeoff between the two). In
     * practice this class is expected to be used to integrate
     * quantities against a stellar IMF (N = 1, via the PDFIntegrator
     * alias) or against the joint distribution of a
     * continuously-sampled stellar population's mass, age, and
     * metallicity (N = 3). Any dimension may optionally be integrated
     * in log space instead of linear space (see the logTransform
     * constructor parameter below) -- a mathematically exact change
     * of variables that often converges far faster for a dimension
     * (e.g. stellar mass) spanning several orders of magnitude with
     * sharp behavior concentrated near one edge of its domain, since
     * the transform spreads that behavior out relative to the
     * quadrature rule's own sample spacing.
     *
     * For backward compatibility with PDFIntegrator's own pre-existing
     * API, N == 1 accepts two additional, non-obvious forms beyond
     * this class's own general @tparam F contract, auto-detected via
     * std::is_invocable_v against F's own (compile-time) type -- never
     * ambiguous in practice, since every real F in this codebase has a
     * single, concrete call signature, though a hypothetical F valid
     * under more than one of these forms (e.g. a generic lambda) would
     * silently prefer whichever is checked first, in the order listed:
     * - N == 1, Vectorized == false: F may take a plain double (the
     *   single dimension's coordinate directly, unwrapped from its
     *   1-element span) instead of a std::span<double>.
     * - N == 1, Vectorized == true: F may take a plain
     *   std::span<double> of length npts (the batch's values, without
     *   the degenerate size-1 second dimension) instead of a
     *   std::mdspan of shape (npts, 1).
     */
    template <class F, std::size_t N, bool Vectorized = false, CubatureMethod Method = CubatureMethod::hAdaptive>
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
         * @param logTransform For each dimension d, whether to
         *   integrate over ln(x_d) instead of x_d directly -- a
         *   mathematically exact change of variables (int p(x) f(x)
         *   dx = int p(e^u) f(e^u) e^u du, u = ln(x)) that leaves the
         *   value of the integral, f's own required signature, and
         *   the coordinates f/p are evaluated at (always the real,
         *   untransformed x_d, regardless of this parameter) all
         *   unchanged -- it only changes which variable the
         *   underlying cubature routine subdivides/raises degree in.
         *   Defaults to every dimension false (no transform, matching
         *   this class's own pre-existing behavior). Every dimension
         *   with logTransform[d] == true must have strictly positive
         *   integration bounds (after clamping to p[d]'s own support
         *   in integrate()); integrate() throws std::runtime_error
         *   otherwise, since ln() of a non-positive bound is undefined.
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
            std::array<bool, N> logTransform = {},
            std::size_t maxEval = 0,
            double reqAbsError = 0.0,
            double reqRelError = 1e-6,
            error_norm norm = ERROR_INDIVIDUAL // NOLINT(misc-include-cleaner)
        ) :
        p_(p),
        f_(std::move(f)),
        nInt_(nInt),
        logTransform_(logTransform),
        maxEval_(maxEval),
        reqAbsError_(reqAbsError),
        reqRelError_(reqRelError),
        norm_(norm)
        { }

        /**
         * @brief Construct a PDFIntegratorND from a single PDF (N == 1 only)
         * @param p The one-dimensional PDF p(x) weighting the
         *   integrand; p must outlive this PDFIntegratorND, since it
         *   is stored by reference
         * @param f The integrand f(x); see the array-taking
         *   constructor's own comment
         * @param nInt The number of quantities f returns per point
         * @param logTransform Whether to integrate over ln(x) instead
         *   of x directly; see the array-taking constructor's own
         *   comment for the full contract
         * @param maxEval Maximum number of integrand evaluations
         *   hcubature/hcubature_v is allowed to make; 0 means no limit
         * @param reqAbsError Required absolute error
         * @param reqRelError Required relative error
         * @param norm Method used to combine the per-component error
         *   estimates into the single value checked against
         *   reqAbsError/reqRelError; see cubature.h's error_norm
         * @details
         * Convenience overload for the common N == 1 case, preserving
         * PDFIntegrator's own pre-existing single-PDF constructor
         * signature (mirrors Interpolator1D's own analogous
         * single-value convenience constructor). Only participates in
         * overload resolution when N == 1 -- for N != 1, a single PDF
         * doesn't unambiguously specify all N dimensions, so callers
         * must use the array-taking constructor above instead.
         */
        PDFIntegratorND(
            const pdfs::PDF& p,
            F f,
            unsigned nInt,
            std::array<bool, N> logTransform = {},
            std::size_t maxEval = 0,
            double reqAbsError = 0.0,
            double reqRelError = 1e-6,
            error_norm norm = ERROR_INDIVIDUAL // NOLINT(misc-include-cleaner)
        ) requires (N == 1)
        : PDFIntegratorND(std::array<std::reference_wrapper<const pdfs::PDF>, N>{ std::cref(p) },
            std::move(f), nInt, logTransform, maxEval, reqAbsError, reqRelError, norm)
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
         *   this class's own @tparam F for the exact contract,
         *   including the two additional forms accepted when N == 1
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
            else if constexpr (std::is_invocable_v<F, Arg, Args...>)
            {
                return std::invoke(f_, x, std::forward<Args>(args)...);
            }
            else if constexpr (N == 1 && !Vectorized && std::is_invocable_v<F, double, Args...>)
            {
                // PDFIntegrator's own pre-existing API: F takes the
                // single dimension's coordinate directly -- see this
                // class's own @tparam F.
                return std::invoke(f_, x[0], std::forward<Args>(args)...); // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- x has length N == 1 in this branch
            }
            else if constexpr (N == 1 && Vectorized && std::is_invocable_v<F, std::span<double>, Args...>)
            {
                // Flat-span convenience for the vectorized N == 1
                // case -- see this class's own @tparam F. Safe to
                // reinterpret x's own contiguous buffer this way:
                // PointsView uses the default (row-major) layout, so
                // for a single column, consecutive points are already
                // contiguous with no padding.
                return std::invoke(f_, std::span<double>(x.data_handle(), x.extent(0)),
                    std::forward<Args>(args)...);
            }
            else
            {
                static_assert(alwaysFalse<F>,
                    "F is not invocable with any signature PDFIntegratorND supports "
                    "for this N/Vectorized combination -- see this class's own @tparam F");
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
         * @throws std::runtime_error if logTransform[d] is true for
         *   some dimension d whose integration bounds, after clamping
         *   to p[d]'s own support, are not both strictly positive
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

                if (logTransform_[d]) // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index)
                {
                    if (a[d] <= 0.0 || b[d] <= 0.0) // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index)
                    {
                        throw std::runtime_error(
                            "PDFIntegratorND::integrate: dimension " + std::to_string(d) +
                            " has logTransform set, but its integration bounds "
                            "(after clamping to the PDF's own support) are not "
                            "both strictly positive: a = " + std::to_string(a[d]) + // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index)
                            ", b = " + std::to_string(b[d])); // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index)
                    }
                    a[d] = std::log(a[d]); // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index)
                    b[d] = std::log(b[d]); // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index)
                }
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
            std::vector<double> errBuf(nInt_); // discarded; every cubature routine used here requires it regardless

            // hcubature/hcubature_v and pcubature/pcubature_v share an
            // identical calling convention (cubature.h), so Method
            // only selects which pair of routines gets called here --
            // integrand()/integrandV() themselves are agnostic to it.
            if constexpr (Vectorized)
            {
                if constexpr (Method == CubatureMethod::hAdaptive)
                {
                    hcubature_v(nInt_, &integrandV<Args...>, &ctx, N, a.data(), b.data(),
                        maxEval_, reqAbsError_, reqRelError_, norm_,
                        std::data(result), errBuf.data());
                }
                else
                {
                    pcubature_v(nInt_, &integrandV<Args...>, &ctx, N, a.data(), b.data(),
                        maxEval_, reqAbsError_, reqRelError_, norm_,
                        std::data(result), errBuf.data());
                }
            }
            else
            {
                if constexpr (Method == CubatureMethod::hAdaptive)
                {
                    hcubature(nInt_, &integrand<Args...>, &ctx, N, a.data(), b.data(),
                        maxEval_, reqAbsError_, reqRelError_, norm_,
                        std::data(result), errBuf.data());
                }
                else
                {
                    pcubature(nInt_, &integrand<Args...>, &ctx, N, a.data(), b.data(),
                        maxEval_, reqAbsError_, reqRelError_, norm_,
                        std::data(result), errBuf.data());
                }
            }

            return result;
        }

        /**
         * @brief Evaluate the integral of p(x) f(x) from a to b (N == 1 only)
         * @param a Lower limit of integration
         * @param b Upper limit of integration
         * @param args Any additional arguments f requires, exactly as
         *   they would be passed to operator()
         * @return The integral, in a container of the same type
         *   returned by f
         * @details
         * Convenience overload for the common N == 1 case, preserving
         * PDFIntegrator's own pre-existing bare-double integrate()
         * signature (mirrors the single-PDF constructor's own
         * identical rationale). Only participates in overload
         * resolution when N == 1; for N != 1, a's/b's own dimension
         * count would be ambiguous, so callers must use the
         * array-taking overload above instead.
         */
        template <class... Args>
        [[nodiscard]] auto integrate(const double a, const double b, Args&&... args) const
            requires (N == 1)
        {
            return integrate(std::array<double, N>{ a }, std::array<double, N>{ b },
                std::forward<Args>(args)...);
        }

    private:

        // Helper for operator(): splits the (x, instance, rest...)
        // argument order this class's API uses into the
        // (instance, x, rest...) order std::invoke's INVOKE protocol
        // requires to dispatch a pointer to member function. Mirrors
        // operator()'s own N == 1 fallback dispatch (see its comments
        // for the rationale behind each branch), just with the member
        // instance threaded through every std::invoke call.
        template <class Obj, class... Rest>
        [[nodiscard]] auto invokeMember(Arg x, Obj&& obj, Rest&&... rest) const
        {
            if constexpr (std::is_invocable_v<F, Obj, Arg, Rest...>)
            {
                return std::invoke(f_, std::forward<Obj>(obj), x, std::forward<Rest>(rest)...);
            }
            else if constexpr (N == 1 && !Vectorized && std::is_invocable_v<F, Obj, double, Rest...>)
            {
                return std::invoke(f_, std::forward<Obj>(obj),
                    x[0], std::forward<Rest>(rest)...); // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- x has length N == 1 in this branch
            }
            else if constexpr (N == 1 && Vectorized && std::is_invocable_v<F, Obj, std::span<double>, Rest...>)
            {
                return std::invoke(f_, std::forward<Obj>(obj),
                    std::span<double>(x.data_handle(), x.extent(0)), std::forward<Rest>(rest)...);
            }
            else
            {
                static_assert(alwaysFalse<F>,
                    "F is not invocable with any signature PDFIntegratorND supports "
                    "for this N/Vectorized combination -- see this class's own @tparam F");
            }
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
        // clamped element-wise into [ctx->a_, ctx->b_], which are
        // themselves already in log space for any dimension d with
        // self_->logTransform_[d] == true -- see
        // PDFIntegrator::integrand()'s own identical comment on why
        // clamping happens at all), and writes the result into fval.
        // xInvTransform undoes that log transform dimension-by-
        // dimension (exp() where logTransform_[d] is true, identity
        // otherwise) to recover the real coordinate p and f are always
        // evaluated at, picking up the change-of-variables Jacobian
        // (dx = x d(ln x)) as an extra factor of xInvTransform[d] in
        // weight for exactly those dimensions. The clamped coordinates
        // are copied into a local buffer (rather than clamping x
        // itself, which hcubature owns) so that buffer can be handed
        // to f as a std::span<double>.
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

            std::array<double, N> xInvTransform{};
            for (std::size_t d = 0; d < N; ++d)
            {
                xInvTransform[d] = ctx->self_->logTransform_[d] // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index)
                    ? std::exp(xClamped[d]) : xClamped[d]; // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index)
            }

            double weight = 1.0;
            for (std::size_t d = 0; d < N; ++d)
            {
                weight *= ctx->self_->p_[d].get()(xInvTransform[d]); // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index)
                if (ctx->self_->logTransform_[d]) { weight *= xInvTransform[d]; } // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index)
            }

            const std::span<double> xSpan(xInvTransform);
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
        // clamp-into-a-local-buffer-then-undo-the-log-transform
        // approach (see its own comment for the Jacobian rationale),
        // but for the whole batch at once, so that buffer can be
        // handed to f as a single PointsView.
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

            std::vector<double> xInvTransform(npt * N);
            for (std::size_t i = 0; i < npt; ++i)
            {
                for (std::size_t d = 0; d < N; ++d)
                {
                    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index,cppcoreguidelines-pro-bounds-pointer-arithmetic) -- xClamped/xInvTransform are both sized npt * N just above
                    xInvTransform[(i * N) + d] = ctx->self_->logTransform_[d] // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index)
                        ? std::exp(xClamped[(i * N) + d]) : xClamped[(i * N) + d];
                }
            }

            std::vector<double> weight(npt, 1.0);
            for (std::size_t i = 0; i < npt; ++i)
            {
                for (std::size_t d = 0; d < N; ++d)
                {
                    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index,cppcoreguidelines-pro-bounds-pointer-arithmetic) -- xInvTransform is sized npt * N just above
                    weight[i] *= ctx->self_->p_[d].get()(xInvTransform[(i * N) + d]); // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index)
                    if (ctx->self_->logTransform_[d]) // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index)
                    {
                        weight[i] *= xInvTransform[(i * N) + d]; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
                    }
                }
            }

            const PointsView points(xInvTransform.data(), npt);
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
        std::array<bool, N> logTransform_; /**< Whether each dimension is integrated over ln(x) instead of x */
        std::size_t maxEval_;      /**< Maximum number of integrand evaluations */
        double reqAbsError_;       /**< Required absolute error */
        double reqRelError_;       /**< Required relative error */
        error_norm norm_;          /**< Error norm used to combine per-component errors */ // NOLINT(misc-include-cleaner)
    };

} // namespace utils

#endif // PDFINTEGRATORND_HPP
