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

#include "../utils/RngThread.hpp"
#include "PDFCommons.hpp"
#include "PDFSegment.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <random>
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
         * @brief Explicit constructor for PDFSegmentExponential.
         * @param sMin The lower limit of the segment.
         * @param sMax The upper limit of the segment.
         * @param scale The exponential scale length of the distribution.
         */
        PDFSegmentExponential(double sMin, double sMax, double scale) :
            PDFSegment(sMin, sMax), scale_(scale)
        {
            // Calculate normalization constant for the PDF segment
            const double exMin = std::exp(-sMin_ / scale_);
            const double exMax = std::exp(-sMax_ / scale_);
            norm_ = 1.0 / (scale_ * (exMin - exMax));
        }
        /**
         * @brief Construct PDFSegmentExponential from a PDF file contents.
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
        PDFSegmentExponential(std::ifstream& file, 
            FileFormats fmt,
            double &sMin,
            double &sMax,
            double &wgt);
        PDFSegmentExponential(const PDFSegmentExponential&) = default;
        auto operator=(const PDFSegmentExponential&) -> PDFSegmentExponential& = default;
        PDFSegmentExponential(PDFSegmentExponential&&) = default;
        auto operator=(PDFSegmentExponential&&) -> PDFSegmentExponential& = default;
        ~PDFSegmentExponential() override = default;

        // Evaluation functions
        [[nodiscard]] auto operator()(double x) const -> double override {
            if (x < sMin_ || x > sMax_) {
                return 0.0; // PDF is zero outside the segment
            }
            return norm_ * std::exp(-x / scale_); // PDF value at x
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
            const double num1 = aClamped * std::exp(-aClamped / scale_);
            const double num2 = bClamped * std::exp(-bClamped / scale_);
            const double denom1 = std::exp(-aClamped / scale_);
            const double denom2 = std::exp(-bClamped / scale_);
            return scale_ + ((num1 - num2) / (denom1 - denom2));
        }
        [[nodiscard]] auto expectationValue() const -> double override {
            const double num1 = sMin_ * std::exp(-sMin_ / scale_);
            const double num2 = sMax_ * std::exp(-sMax_ / scale_);
            const double denom1 = std::exp(-sMin_ / scale_);
            const double denom2 = std::exp(-sMax_ / scale_);
            return scale_ + ((num1 - num2) / (denom1 - denom2));
        }
        [[nodiscard]] auto integral(const double a, const double b) const -> double override {
            if (a >= b || a > sMax_ || b < sMin_) {
                return 0.0; // Invalid range for integral calculation
            }
            const double aClamped = std::max(a, sMin_);
            const double bClamped = std::min(b, sMax_);
            return norm_ * scale_ * (std::exp(-aClamped / scale_) - std::exp(-bClamped / scale_));
        }

        // Drawing functions
        [[nodiscard]] auto draw(const double a, const double b) const -> double override {
            const double aClamped = std::max(a, sMin_);
            const double bClamped = std::min(b, sMax_);
            const double u = std::uniform_real_distribution<double>(0.0, 1.0)(utils::rng()());
            const double expA = std::exp(-aClamped / scale_);
            const double expB= std::exp(-bClamped / scale_);
            return -scale_ * std::log(expA - (u * (expA - expB))); // Inverse transform sampling for exponential distribution
        }
        [[nodiscard]] auto draw() const -> double override {
            return draw(sMin_, sMax_);
        }

    private:
        double scale_; /**< Exponential scale length */
        double norm_;  /**< Normalization constant for the PDF segment */
    };

} // namespace pdfs

#endif // PDFSEGMENTEXPONENTIAL_HPP
