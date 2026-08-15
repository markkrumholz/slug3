/**
 * @file PDFReflect.hpp
 * @author Mark Krumholz
 * @brief A non-owning, reflected view of another PDF.
 * @date 2026-08-14
 */

#ifndef PDFREFLECT_HPP
#define PDFREFLECT_HPP

#include "PDF.hpp"
#include <functional>
#include <limits>
#include <vector>

namespace pdfs {

    /**
     * @class PDFReflect
     * @brief A non-owning view of a PDF, reflected about a pivot point
     * @details
     * PDFReflect wraps a non-owning reference to another PDF and
     * presents it, unmodified in shape, but reflected: a point x is
     * mapped to reflect(x) = 2 * pivot - x before being handed to (or
     * after being returned from) the wrapped PDF, where pivot defaults
     * to the midpoint of the wrapped PDF's own support,
     * (getMin() + getMax()) / 2, if not given explicitly. This is
     * useful when an integrator or sampler needs to treat "distance
     * from some reference point" as its natural coordinate (e.g.
     * stellar age -- distance from the current simulation time --
     * when the underlying PDF is indexed by formation time instead)
     * without having to construct a whole new PDF: a PDFReflect can be
     * passed anywhere a `const PDF&` is expected (e.g. PDFIntegrator's
     * own p parameter), and every method below performs
     * the appropriate coordinate flip transparently. An explicit pivot
     * need not be the midpoint of the wrapped PDF's own support (e.g.
     * reflecting a star-formation-rate PDF about half of the current
     * simulation time, rather than half of its own, typically much
     * larger, declared support, to obtain an age coordinate) -- see
     * the two-argument constructor below.
     *
     * Since PDFReflect derives from PDF but never populates seg_ or
     * wgt_ with its own segments (getWeights() returns wgt_, which is
     * set to a copy of the wrapped PDF's own weights purely so the
     * inherited, unmodified integral() -- see its own comment below --
     * still returns the correct total), valid() (which checks
     * seg_.empty()) always reports false for a PDFReflect; that is not
     * meaningful for this view and should not be relied on. Likewise,
     * getMin()/getMax() always report the wrapped PDF's own support,
     * unchanged, regardless of pivot -- not that support's own image
     * under reflect() -- so that they remain useful as a safety clamp
     * on integration bounds (see PDFIntegrator::integrate()'s own
     * clamp) rather than an increasingly extreme range as pivot moves
     * away from the wrapped PDF's own midpoint.
     */
    class PDFReflect : public PDF {
    public:

        // Un-hide PDF::integral()'s no-argument overload: declaring
        // integral(a, b) below would otherwise hide every base-class
        // overload of the name "integral" in this class's scope, per
        // ordinary C++ name-hiding rules, even though the no-argument
        // form needs no change (see integral(a,b)'s own comment).
        using PDF::integral;

        /**
         * @brief Construct a reflected view of an existing PDF, about the midpoint of its own support
         * @param pdf The PDF to view, reflected about
         *   (pdf.getMin() + pdf.getMax()) / 2; stored by reference, so
         *   it must outlive this PDFReflect
         * @details
         * Delegates to the explicit-pivot constructor below with
         * pivot = (pdf.getMin() + pdf.getMax()) / 2, which reduces
         * reflect(x) = 2 * pivot - x to pdf.getMax() - (x - pdf.getMin()) --
         * i.e. this constructor's own behavior is unchanged from
         * reflecting about the midpoint of the wrapped PDF's own
         * support.
         */
        explicit PDFReflect(const PDF& pdf) :
            PDFReflect(pdf, 0.5 * (pdf.getMin() + pdf.getMax()))
        { }

        /**
         * @brief Construct a reflected view of an existing PDF, about an explicit pivot
         * @param pdf The PDF to view, reflected about pivot; stored by
         *   reference, so it must outlive this PDFReflect
         * @param pivot The point to reflect about: x maps to
         *   2 * pivot - x. Need not be the midpoint of pdf's own
         *   support, or even lie within it (e.g. reflecting about half
         *   of a simulation's current time, to obtain an age
         *   coordinate from a star-formation-rate PDF whose own
         *   support is deliberately much larger than any realistic
         *   simulation time -- see io::SimControls's own
         *   buildConstantSFR()).
         * @details
         * Copies pdf's own sMin_/sMax_/weights, unmodified by pivot --
         * see this class's own comment for why -- so that getMin()/
         * getMax() (unmodified, inherited) and the no-argument
         * integral() (also unmodified, inherited -- see its own
         * comment) report the correct values without needing to be
         * overridden.
         */
        PDFReflect(const PDF& pdf, const double pivot) :
            pdf_(std::cref(pdf)),
            pivot_(pivot)
        {
            sMin_ = pdf.getMin();
            sMax_ = pdf.getMax();
            wgt_ = pdf.getWeights();
        }

        /**
         * @brief Evaluate the reflected PDF at a given point
         * @param x The point at which to evaluate the PDF
         * @return pdf(reflect(x))
         */
        auto operator()(double x) const -> double override
        {
            return pdf_.get()(reflect(x));
        }

        /**
         * @brief Calculate the expectation value of the reflected PDF over its entire range
         * @return reflect(pdf.expectationValue())
         */
        [[nodiscard]] auto expectationValue() const -> double override
        {
            return reflect(pdf_.get().expectationValue());
        }

        /**
         * @brief Calculate the expectation value of the reflected PDF over a specified range
         * @param a The lower limit of the range, in the reflected (view) coordinate
         * @param b The upper limit of the range, in the reflected (view) coordinate
         * @return reflect(pdf.expectationValue(reflect(b), reflect(a)))
         */
        [[nodiscard]] auto expectationValue(const double a, const double b) const -> double override
        {
            const double aReflect = reflect(b);
            const double bReflect = reflect(a);
            return reflect(pdf_.get().expectationValue(aReflect, bReflect));
        }

        /**
         * @brief Calculate the integral of the reflected PDF over a specified range
         * @param a The lower limit of the range, in the reflected (view) coordinate
         * @param b The upper limit of the range, in the reflected (view) coordinate
         * @return pdf.integral(reflect(b), reflect(a))
         * @details
         * The no-argument integral() needs no override -- see the
         * constructor's own comment.
         */
        [[nodiscard]] auto integral(const double a, const double b) const -> double override
        {
            const double aReflect = reflect(b);
            const double bReflect = reflect(a);
            return pdf_.get().integral(aReflect, bReflect);
        }

        /**
         * @brief Sample a random value from the reflected PDF within a specified range
         * @param a The lower limit of the sampling range, in the reflected (view) coordinate
         * @param b The upper limit of the sampling range, in the reflected (view) coordinate
         * @return reflect(pdf.draw(reflect(b), reflect(a)))
         */
        [[nodiscard]]
        auto draw(const double a = std::numeric_limits<double>::lowest(),
            const double b = std::numeric_limits<double>::max()) const -> double override
        {
            const double aReflect = reflect(b);
            const double bReflect = reflect(a);
            return reflect(pdf_.get().draw(aReflect, bReflect));
        }

        /**
         * @brief Sample nDraw random values from the reflected PDF within a specified range
         * @param nDraw Number of samples to draw
         * @param a The lower limit of the sampling range, in the reflected (view) coordinate
         * @param b The upper limit of the sampling range, in the reflected (view) coordinate
         * @return pdf.draw(nDraw, reflect(b), reflect(a)), with reflect() applied to every element
         */
        [[nodiscard]] auto draw(unsigned int nDraw,
            double a = std::numeric_limits<double>::lowest(),
            double b = std::numeric_limits<double>::max())
            const -> std::vector<double> override
        {
            const double aReflect = reflect(b);
            const double bReflect = reflect(a);
            auto result = pdf_.get().draw(nDraw, aReflect, bReflect);
            for (auto& r : result) { r = reflect(r); }
            return result;
        }

        /**
         * @brief Draw a sample from the reflected PDF targeting a fixed total value
         * @param target Target total to draw
         * @param a The lower limit of the sampling range, in the reflected (view) coordinate
         * @param b The upper limit of the sampling range, in the reflected (view) coordinate
         * @return pdf.drawTarget(target, reflect(b), reflect(a)), with reflect() applied to every element
         */
        [[nodiscard]] auto drawTarget(double target,
            double a = std::numeric_limits<double>::lowest(),
            double b = std::numeric_limits<double>::max()) const
            -> std::vector<double> override
        {
            const double aReflect = reflect(b);
            const double bReflect = reflect(a);
            auto result = pdf_.get().drawTarget(target, aReflect, bReflect);
            for (auto& r : result) { r = reflect(r); }
            return result;
        }

    private:

        /**
         * @brief Reflect a point about pivot_
         * @param x Point to reflect
         * @return 2 * pivot_ - x
         */
        [[nodiscard]] auto reflect(const double x) const -> double
        {
            return (2.0 * pivot_) - x;
        }

        std::reference_wrapper<const PDF> pdf_; /**< The PDF this object is a reflected view of */
        double pivot_; /**< The point reflect() reflects about */
    };

} // namespace pdfs

#endif // PDFREFLECT_HPP
