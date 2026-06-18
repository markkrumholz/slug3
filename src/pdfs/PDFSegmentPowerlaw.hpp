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

#include "../utils/RngThread.hpp"
#include "PDFCommons.hpp"
#include "PDFSegment.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <random>

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
         */
        PDFSegmentPowerlaw(double sMin, double sMax, double alpha) :
            PDFSegment(sMin, sMax), alpha_(alpha) {
                // Calculate normalization constant for the PDF segment
                if (alpha_ != -1) {
                    norm_ = (alpha_ + 1) / (std::pow(sMax_, alpha_ + 1) - std::pow(sMin_, alpha_ + 1));
                } else {
                    norm_ = 1.0 / std::log(sMax_ / sMin_);
                }
            }
        /**
         * @brief Construct PDFSegmentPowerlaw from a PDF file contents.
         * @param file File stream from which to construct
         * @param fmt Format of the file being read
         * @param sMin The lower limit of the segment
         * @param sMax The upper limit of the segment
         * @param wgt The weight of the segment
         * @details
         * How the arguments are interpreted depends on fmt; if fmt
         * is basic, then sMin and sMax are inputs, and wgt
         * is ignored, while if fmt is advanced, then 
         * sMin and sMax are ignored and wgt is an output.
        */        
        PDFSegmentPowerlaw(std::ifstream& file, 
            FileFormats fmt,
            double &sMin,
            double &sMax,
            double &wgt);
        PDFSegmentPowerlaw(const PDFSegmentPowerlaw&) = default;
        auto operator=(const PDFSegmentPowerlaw&) -> PDFSegmentPowerlaw& = default;
        PDFSegmentPowerlaw(PDFSegmentPowerlaw&&) = default;
        auto operator=(PDFSegmentPowerlaw&&) -> PDFSegmentPowerlaw& = default;
        ~PDFSegmentPowerlaw() override = default;

        // Evaluation functions
        [[nodiscard]] auto operator()(double x) const -> double override {
            if (x < sMin_ || x > sMax_) {
                return 0.0; // PDF is zero outside the segment
            }
            return norm_ * std::pow(x, alpha_); // PDF value at x
        }

        [[nodiscard]] auto expectationValue(const double a, const double b) const -> double override {
            if (a >= b || a > sMax_ || b < sMin_) {
                return 0.0; // Invalid range for expectation value calculation
            }
            if (a == sMax_) {
                return sMax_; // Handle edge cases
            }
            if (b == sMin_) {
                return sMin_; // Handle edge cases
            }
            const double aClamped = std::max(a, sMin_);
            const double bClamped = std::min(b, sMax_);
            if (alpha_ != -1 && alpha_ != -2) {
                return (alpha_ + 1) / (alpha_ + 2) *
                    (std::pow(bClamped, alpha_ + 2) - std::pow(aClamped, alpha_ + 2)) /
                    (std::pow(bClamped, alpha_ + 1) - std::pow(aClamped, alpha_ + 1));
            }
            if (alpha_ == -1) {
                return (bClamped - aClamped) / (std::log(bClamped / aClamped));
            }
            return aClamped * bClamped * std::log(bClamped / aClamped) / (bClamped - aClamped);
        }

        [[nodiscard]] auto expectationValue() const -> double override {
            return expectationValue(sMin_, sMax_);
        }

        [[nodiscard]] auto integral(const double a, const double b) const -> double override {
            if (a >= b || a >= sMax_ || b <= sMin_) {
                return 0.0; // Invalid range for integral calculation
            }
            const double aClamped = std::max(a, sMin_);
            const double bClamped = std::min(b, sMax_);
            if (alpha_ != -1) {
                // Special case
                return norm_ / (alpha_ + 1) *
                    (std::pow(bClamped, alpha_ + 1) - std::pow(aClamped, alpha_ + 1));
            }
            // General case
            return norm_ * std::log(bClamped / aClamped);
        }

        // Drawing functions
        [[nodiscard]] auto draw(const double a, const double b) const -> double override {
            const double aClamped = std::max(a, sMin_);
            const double bClamped = std::min(b, sMax_);
            if (aClamped >= bClamped) {
                return 0.0; // Invalid range for drawing
            }
            std::uniform_real_distribution<double> dist(0.0, 1.0);
            const double u = dist(utils::rng()); // Uniform random number in [0, 1)
            if (alpha_ != -1)
            {
                // Special case
                return std::pow(
                    (u * std::pow(bClamped, alpha_ + 1)) +
                    ((1 - u) * std::pow(aClamped, alpha_ + 1)),
                    1.0 / (alpha_ + 1)
                );
            }
            // General case
            return aClamped * std::pow(bClamped / aClamped, u);
        }
        [[nodiscard]] auto draw() const -> double override {
            return draw(sMin_, sMax_);
        }

    private:
        double alpha_;  /**< Power-law index */
        double norm_;   /**< Normalization constant for the PDF segment */
    };

} // namespace pdfs

#endif // PDFSEGMENTPOWERLAW_HPP
