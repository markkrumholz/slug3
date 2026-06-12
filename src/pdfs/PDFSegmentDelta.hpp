/**
 * @file PDFSegmentDelta.hpp
 * @author Mark Krumholz
 * @brief Class to represent a delta-function segment of a PDF.
 * @details
 * This class represents a delta-function segment of a PDF, defined by a single value. It implements the interface defined by the PDFSegment class, providing
 * methods to evaluate the PDF at a given point and to sample a random value from the PDF segment according to the delta-function distribution.
 * @date 2024-06-12
 */

#ifndef PDFSEGMENTDELTA_HPP
#define PDFSEGMENTDELTA_HPP

#include <stdexcept>
#include "PDFSegment.hpp"

namespace pdfs {

    /**
     * @class PDFSegmentDelta
     * @brief Class representing a delta-function segment of a PDF.
     * @details
     * This class implements the PDFSegment interface for a delta-function distribution, defined by a single value. It provides methods to evaluate the PDF at a given point and to sample a random value from the PDF segment according to the delta-function distribution.
     */
    class PDFSegmentDelta : public PDFSegment {
    public:

        // Constructor and destructor
        /**
         * @brief Constructor for PDFSegmentDelta.
         * @param sValue The value at which the delta function is centered.
         * @param rng Reference to the random number generator to be used for sampling.
         */
        PDFSegmentDelta(double sValue, rngType &rng) :
            PDFSegment(sValue, sValue, rng) {}
        ~PDFSegmentDelta() override = default;

        // Evaluation functions
        auto operator()(double x) const -> double override {
            (void)x;
            throw std::runtime_error("PDFSegmentDelta: a delta function cannot be evaluated at a point.");
        }
        auto expectationValue(const double a, const double b) const -> double override {
            if (a >= b) {
                return 0.0; // Invalid range for expectation value calculation
            }
            return sMin_; // The expectation value of a delta function is just the value at which it is centered
        }
        auto expectationValue() const -> double override {
            return sMin_; // The expectation value of a delta function is just the value at which it is centered
        }
        auto integral(const double a, const double b) const -> double override {
            if (a <= sMin_ && b >= sMin_) {
                return 1.0; // The integral of a delta function over a range that includes its center is 1
            } else {
                return 0.0; // The integral of a delta function over a range that does not include its center is 0
            }
        }

        // Drawing functions
        auto draw(const double a, const double b) const -> double override {
            return sMin_; // Sampling from a delta function always returns the same value
        }
        auto draw() const -> double override {
            return sMin_;  // Sampling from a delta function always returns the same value
        }
    };

} // namespace pdfs

#endif // PDFSEGMENTDELTA_HPP
