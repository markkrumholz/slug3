/**
 * @file PDFSegmentNormal.hpp
 * @author Mark Krumholz
 * @brief Class to represent a normal segment of a PDF.
 * @details
 * This class represents a normal segment of a PDF, defined by a lower
 * and upper limit, a mean, and a standard deviation. It implements the interface defined
 * by the PDFSegment class, providing methods to evaluate the PDF at a given
 * point and to sample a random value from the PDF segment according to the normal distribution.
 * @date 2024-06-12
 */

#ifndef PDFSEGMENTNORMAL_HPP
#define PDFSEGMENTNORMAL_HPP

#include "../utils/RngThread.hpp"
#include "PDFCommons.hpp"
#include "PDFSegment.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <numbers>
#include <random>

namespace pdfs {

    /**
     * @class PDFSegmentNormal
     * @brief Class representing a normal segment of a PDF.
     * @details
     * This class implements the PDFSegment interface for a normal distribution,
     * defined by a mean and a standard deviation. It provides methods to evaluate the PDF at a
     * given point and to sample a random value from the PDF segment according to
     * the normal distribution.
     */
    class PDFSegmentNormal : public PDFSegment {
    public:

        // Constructor and destructor
        /**
         * @brief Constructor for PDFSegmentNormal.
         * @param sMin The lower limit of the segment.
         * @param sMax The upper limit of the segment.
         * @param mean The mean of the normal distribution.
         * @param stddev The standard deviation of the normal distribution.
         */
        PDFSegmentNormal(double sMin, double sMax, double mean, double stddev) :
            PDFSegment(sMin, sMax), 
            mean_(mean), 
            stddev_(stddev),
            norm_(std::numbers::sqrt2 * std::numbers::inv_sqrtpi /
                    stddev_ /
                    (std::erf((sMax_ - mean_) / (stddev_ * std::numbers::sqrt2)) -
                     std::erf((sMin_ - mean_) / (stddev_ * std::numbers::sqrt2)))
                )
            { }
        /**
         * @brief Construct PDFSegmentNormal from a PDF file contents.
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
        PDFSegmentNormal(std::ifstream& file, 
            FileFormats fmt,
            double &sMin,
            double &sMax,
            double &wgt);
        PDFSegmentNormal(const PDFSegmentNormal&) = default;
        auto operator=(const PDFSegmentNormal&) -> PDFSegmentNormal& = default;
        PDFSegmentNormal(PDFSegmentNormal&&) = default;
        auto operator=(PDFSegmentNormal&&) -> PDFSegmentNormal& = default;
        ~PDFSegmentNormal() override = default;

        // Evaluation functions
        [[nodiscard]] auto operator()(double x) const -> double override {
            if (x < sMin_ || x > sMax_) {
                return 0.0; // PDF is zero outside the segment
            }
            return norm_ * std::exp(-0.5 * std::pow((x - mean_) / stddev_, 2)); // PDF value at x
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
            const double dxLoNorm = (aClamped - mean_) / (std::numbers::sqrt2 * stddev_);
            const double dxHiNorm = (bClamped - mean_) / (std::numbers::sqrt2 * stddev_);
            const double denom = std::erf(dxHiNorm) - std::erf(dxLoNorm);
            if (denom == 0.0) {
                return 0.0; // Avoid division by zero if the PDF is negligible in the range
            }
            const double numLo = std::exp(-std::pow(dxLoNorm, 2));
            const double numHi = std::exp(-std::pow(dxHiNorm, 2));
            return mean_ + 
                (stddev_ * std::numbers::sqrt2 * std::numbers::inv_sqrtpi *
                    (numLo - numHi) / denom);
        }
        [[nodiscard]] auto expectationValue() const -> double override {
            return expectationValue(sMin_, sMax_); // Expectation value over the full range of the segment
        }
        [[nodiscard]] auto integral(const double a, const double b) const -> double override {
            if (a >= b || a >= sMax_ || b <= sMin_) {
                return 0.0; // Invalid range for integral calculation
            }
            const double aClamped = std::max(a, sMin_);
            const double bClamped = std::min(b, sMax_);
            const double dxLoNorm = (aClamped - mean_) / (std::numbers::sqrt2 * stddev_);
            const double dxHiNorm = (bClamped - mean_) / (std::numbers::sqrt2 * stddev_);
            return norm_ /
                (std::numbers::sqrt2 * std::numbers::inv_sqrtpi) *
                stddev_ *
                (std::erf(dxHiNorm) - std::erf(dxLoNorm));
        }

        // Drawing functions
        [[nodiscard]] auto draw(const double a, const double b) const -> double override {
            const double aClamped = std::max(a, sMin_);
            const double bClamped = std::min(b, sMax_);
            if (aClamped >= bClamped) {
                return 0.0; // Invalid range for drawing
            }
            std::normal_distribution<double> dist(mean_, stddev_);
            double sample = NAN;
            while (true) {
                sample = dist(utils::rng());
                if (sample >= aClamped && sample <= bClamped) { break; } // Rejection sampling
            }
            return sample;
        }
        [[nodiscard]] auto draw() const -> double override {
            return draw(sMin_, sMax_);
        }
    private:
        double mean_;   /**< Mean of the normal distribution */
        double stddev_; /**< Standard deviation of the normal distribution */
        double norm_;   /**< Normalization constant for the PDF segment */
    };

} // namespace pdfs

#endif // PDFSEGMENTNORMAL_HPP
