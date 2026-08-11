/**
 * @file PDFIntegrator.hpp
 * @author Mark Krumholz
 * @brief Evaluate integrals of the form int_a^b p(x) f(x) dx
 * @date 2026-07-19
 */

#ifndef PDFINTEGRATOR_HPP
#define PDFINTEGRATOR_HPP

#include "../pdfs/PDF.hpp"
#include "PDFIntegratorND.hpp"
#include <array>
#include <cstddef>
#include <functional>

namespace utils
{

    /**
     * @class PDFIntegrator
     * @brief Evaluate integrals of the form int_a^b p(x) f(x) dx
     * @tparam F Type of the callable integrand: either the plain,
     *   scalar-double form this class originally shipped with (must
     *   accept a double, the integration variable, as its first
     *   argument), or PDFIntegratorND's own more general
     *   std::span<double>-of-length-1 form -- both work, auto-detected
     *   from F's own type; see PDFIntegratorND's own @tparam F for the
     *   full rules (including the plus-any-number-of-additional-
     *   arguments and return-type contract, which are unchanged here).
     * @details
     * A thin subclass of PDFIntegratorND<F, 1> (see PDFIntegratorND.hpp
     * for the full implementation this class now shares with the
     * N > 1 case) that adds nothing beyond inheriting its constructors
     * and a set of explicit class template argument deduction (CTAD)
     * guides mirroring them, so PDFIntegrator's own template argument F
     * can still be deduced from a constructor call without spelling it
     * explicitly, exactly preserving PDFIntegrator's own pre-existing
     * one-dimensional API and ergonomics -- including its single-PDF
     * constructor overload and scalar-double integrand signature -- so
     * every existing call site continues to work unchanged.
     *
     * This is a subclass, not a plain `using PDFIntegrator =
     * PDFIntegratorND<F, 1>;` alias template, specifically so that
     * CTAD keeps working portably: CTAD through an alias template is a
     * C++20 feature (P1814) that not every toolchain this project
     * targets implements yet (as of 2026-08, clang-tidy-18 -- the
     * version CI's own clang-tidy job uses -- rejects it outright,
     * even though it happily accepts every other C++23 feature this
     * codebase otherwise relies on), whereas explicit deduction guides
     * on a genuine class template are ordinary C++17 and work
     * everywhere. Although this class is general purpose, in practice
     * it will most often be used to integrate quantities against a
     * stellar IMF, whose steep dynamic range often benefits from the
     * inherited logTransform constructor parameter -- see
     * PDFIntegratorND's own comment on it.
     */
    template <class F>
    class PDFIntegrator : public PDFIntegratorND<F, 1>
    {
    public:
        using PDFIntegratorND<F, 1>::PDFIntegratorND;
    };

    // Explicit deduction guides, one per PDFIntegratorND<F, 1>
    // constructor -- see this class's own comment for why these exist
    // instead of relying on CTAD through an alias template. Default
    // arguments are repeated here (rather than left to the inherited
    // constructor alone) since a deduction guide's own parameter list,
    // not the constructor it mirrors, is what overload resolution
    // consults to decide whether an omitted trailing argument is
    // permitted while still deducing F.
    template <class F>
    PDFIntegrator(
        std::array<std::reference_wrapper<const pdfs::PDF>, 1> p,
        F f,
        unsigned nInt,
        std::array<bool, 1> logTransform = {},
        std::size_t maxEval = 0,
        double reqAbsError = 0.0,
        double reqRelError = 1e-6,
        error_norm norm = ERROR_INDIVIDUAL // NOLINT(misc-include-cleaner)
    ) -> PDFIntegrator<F>;

    template <class F>
    PDFIntegrator(
        const pdfs::PDF& p,
        F f,
        unsigned nInt,
        std::array<bool, 1> logTransform = {},
        std::size_t maxEval = 0,
        double reqAbsError = 0.0,
        double reqRelError = 1e-6,
        error_norm norm = ERROR_INDIVIDUAL // NOLINT(misc-include-cleaner)
    ) -> PDFIntegrator<F>;

} // namespace utils

#endif // PDFINTEGRATOR_HPP
