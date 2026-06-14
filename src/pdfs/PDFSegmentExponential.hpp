/**
 * @file PDFSegmentExponential.hpp
 * @author Mark Krumholz
 * @brief Class to represent an exponential segment of a PDF.
 * @details
 * This class represents an exponential segment of a PDF, defined by a lower
 * and upper limit, and an exponential scale length. It implements the interface defined
 * by the PDFSegment class, providing methods to evaluate the PDF at a given
 * point and to sample a random value from the PDF segment according to the exponential distribution.
 * @date 2024-06-12
 */

#ifndef PDFSEGMENTEXPONENTIAL_HPP
#define PDFSEGMENTEXPONENTIAL_HPP

#include <cmath>
#include <random>
#include "PDFSegment.hpp"

namespace pdfs {

    /**
     * @class PDFSegmentExponential
     * @brief Class representing an exponential segment of a PDF.
     * @details
     * This class implements the PDFSegment interface for an exponential distribution,
     * defined by an exponential scale length. It provides methods to evaluate the PDF at a
     * given point and to sample a random value from the PDF segment according to
     * the exponential distribution.
     */
    class PDFSegmentExponential : public PDFSegment {
    public:

        // Constructor and destructor
        /**
         * @brief Constructor for PDFSegmentExponential.
         * @param sMin The lower limit of the segment.
         * @param sMax The upper limit of the segment.
         * @param scale The exponential scale length of the distribution.
         * @param rng Reference to the random number generator to be used for sampling.
         */
        PDFSegmentExponential(double sMin, double sMax, double scale, rngType &rng) :
            PDFSegment(sMin, sMax, rng), scale_(scale) {
                // Calculate normalization constant for the PDF segment
                norm_ = 1.0 / (scale_ * (std::exp(-sMin_ / scale_) - std::exp(-sMax_ / scale_)));
            }
        ~PDFSegmentExponential() override = default;

        // Evaluation functions
        auto operator()(double x) const -> double override {
            if (x < sMin_ || x > sMax_) {
                return 0.0; // PDF is zero outside the segment
            }
            return norm_ * std::exp(-x / scale_); // PDF value at x
        }
        auto expectationValue(const double a, const double b) const -> double override {
            if (a >= b || a > sMax_ || b < sMin_) {
                return 0.0; // Invalid range for expectation value calculation
            } else if (a == sMax_) {
                return sMax_; // Handle edge cases
            } else if (b == sMin_) {
                return sMin_; // Handle edge cases
            }
            const double a_clamped = std::max(a, sMin_);
            const double b_clamped = std::min(b, sMax_);
            return scale_ +
                (a_clamped * std::exp(-a_clamped / scale_) -
                b_clamped * std::exp(-b_clamped / scale_)) /
                (std::exp(-a_clamped / scale_) - std::exp(-b_clamped / scale_));
        }
        auto expectationValue() const -> double override {
            return scale_ +
                (sMin_ * std::exp(-sMin_ / scale_) -
                sMax_ * std::exp(-sMax_ / scale_)) /
                (std::exp(-sMin_ / scale_) - std::exp(-sMax_ / scale_));
        }
        auto integral(const double a, const double b) const -> double override {
            if (a >= b || a > sMax_ || b < sMin_) {
                return 0.0; // Invalid range for integral calculation
            }
            const double a_clamped = std::max(a, sMin_);
            const double b_clamped = std::min(b, sMax_);
            return norm_ * scale_ * (std::exp(-a_clamped / scale_) - std::exp(-b_clamped / scale_));
        }

        // Drawing functions
        auto draw(const double a, const double b) const -> double override {
            const double a_clamped = std::max(a, sMin_);
            const double b_clamped = std::min(b, sMax_);
            const double u = std::uniform_real_distribution<double>(0.0, 1.0)(rng_);
            const double exp_a = std::exp(-a_clamped / scale_);
            const double exp_b = std::exp(-b_clamped / scale_);
            return -scale_ * std::log(exp_a - u * (exp_a - exp_b)); // Inverse transform sampling for exponential distribution
        }
        auto draw() const -> double override {
            return draw(sMin_, sMax_);
        }

    private:
        double scale_; /**< Exponential scale length */
        double norm_;  /**< Normalization constant for the PDF segment */
    };
}

#endif // PDFSEGMENTEXPONENTIAL_HPP
