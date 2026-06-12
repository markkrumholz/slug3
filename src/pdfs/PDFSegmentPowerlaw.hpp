/**
 * @file PDFSegmentPowerlaw.hpp
 * @author Mark Krumholz
 * @brief Class to represent a power-law segment of a PDF.
 * @details
 * This class represents a power-law segment of a PDF, defined by a lower
 * and upper limit, and a power-law index. It implements the interface defined
 * by the PDFSegment class, providing methods to evaluate the PDF at a given
 * point and to sample a random value from the PDF segment according to the
 * power-law distribution.
 * @date 2024-06-12
 */

#ifndef PDFSEGMENTPOWERLAW_HPP
#define PDFSEGMENTPOWERLAW_HPP

#include <cmath>
#include <random>
#include "PDFSegment.hpp"

namespace pdfs {

    /**
     * @class PDFSegmentPowerlaw
     * @brief Class representing a power-law segment of a PDF.
     * @details
     * This class implements the PDFSegment interface for a power-law distribution,
     * defined by a power-law index. It provides methods to evaluate the PDF at a
     * given point and to sample a random value from the PDF segment according to
     * the power-law distribution.
     */
    class PDFSegmentPowerlaw : public PDFSegment {
    public:

        // Constructor and destructor
        /**
         * @brief Constructor for PDFSegmentPowerlaw.
         * @param sMin The lower limit of the segment.
         * @param sMax The upper limit of the segment.
         * @param alpha The power-law index of the distribution.
         * @param rng Reference to the random number generator to be used for sampling.
         */
        PDFSegmentPowerlaw(double sMin, double sMax, double alpha, rngType &rng) :
            PDFSegment(sMin, sMax, rng), alpha_(alpha) {
                // Calculate normalization constant for the PDF segment
                if (alpha_ != -1) {
                    norm_ = (alpha_ + 1) / (std::pow(sMax_, alpha_ + 1) - std::pow(sMin_, alpha_ + 1));
                } else {
                    norm_ = 1.0 / std::log(sMax_ / sMin_);
                }
            }
        ~PDFSegmentPowerlaw() override = default;

        // Evaluation functions
        auto operator()(double x) const -> double override {
            if (x < sMin_ || x > sMax_) {
                return 0.0; // PDF is zero outside the segment
            }
            return norm_ * std::pow(x, alpha_); // PDF value at x
        }

        auto expectationValue(const double a, const double b) const -> double override {
            if (a >= b) {
                return 0.0; // Invalid range for expectation value calculation
            }
            const double a_clamped = std::max(a, sMin_);
            const double b_clamped = std::min(b, sMax_);
            if (alpha_ != -1 && alpha_ != -2) {
                return (alpha_ + 1) / (alpha_ + 2) *
                    (std::pow(b_clamped, alpha_ + 2) - std::pow(a_clamped, alpha_ + 2)) /
                    (std::pow(b_clamped, alpha_ + 1) - std::pow(a_clamped, alpha_ + 1));
            } else if (alpha_ == -1) {
                return (b_clamped - a_clamped) / (std::log(b_clamped / a_clamped));
            } else {
                return a_clamped * b_clamped * std::log(b_clamped / a_clamped) / (b_clamped - a_clamped);
            }
        }

        auto expectationValue() const -> double override {
            return expectationValue(sMin_, sMax_);
        }

        auto integral(const double a, const double b) const -> double override {
            if (a >= b) {
                return 0.0; // Invalid range for integral calculation
            }
            const double a_clamped = std::max(a, sMin_);
            const double b_clamped = std::min(b, sMax_);
            if (alpha_ != -1) {
                return norm_ / (alpha_ + 1) *
                    (std::pow(b_clamped, alpha_ + 1) - std::pow(a_clamped, alpha_ + 1));
            } else {
                return norm_ * std::log(b_clamped / a_clamped);
            }
        }

        // Drawing functions
        auto draw(const double a, const double b) const -> double override {
            const double a_clamped = std::max(a, sMin_);
            const double b_clamped = std::min(b, sMax_);
            if (a_clamped >= b_clamped) {
                return 0.0; // Invalid range for drawing
            }
            std::uniform_real_distribution<double> dist(0.0, 1.0);
            const double u = dist(rng_); // Uniform random number in [0, 1)
            if (alpha_ != -1) {
                return std::pow(
                    u * std::pow(b_clamped, alpha_ + 1) +
                    (1 - u) * std::pow(a_clamped, alpha_ + 1),
                    1.0 / (alpha_ + 1)
                );
            } else {
                return a_clamped * std::pow(b_clamped / a_clamped, u);
            }
        }
        auto draw() const -> double override {
            return draw(sMin_, sMax_);
        }

    private:
        double alpha_;  /**< Power-law index */
        double norm_;   /**< Normalization constant for the PDF segment */
    };

} // namespace pdfs

#endif // PDFSEGMENTPOWERLAW_HPP
