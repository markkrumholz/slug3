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

#include <cmath>
#include <random>
#include "PDFCommons.hpp"
#include "PDFSegment.hpp"

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
         * @param rng Reference to the random number generator to be used for sampling.
         */
        PDFSegmentNormal(double sMin, double sMax, double mean, double stddev, RngType &rng) :
            PDFSegment(sMin, sMax, rng), mean_(mean), stddev_(stddev) {
                norm_ = std::sqrt(2.0 / M_PI) / stddev_ /
                    (std::erf((sMax_ - mean_) / (stddev_ * std::sqrt(2))) -
                     std::erf((sMin_ - mean_) / (stddev_ * std::sqrt(2))));
            }
        /**
         * @brief Construct PDFSegmentNormal from a PDF file contents.
         * @param file File stream from which to construct
         * @param rng Reference to the random number generator to be used for sampling.
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
            RngType& rng,
            FileFormats fmt,
            double &sMin,
            double &sMax,
            double &wgt);
        ~PDFSegmentNormal() override = default;

        // Evaluation functions
        auto operator()(double x) const -> double override {
            if (x < sMin_ || x > sMax_) {
                return 0.0; // PDF is zero outside the segment
            }
            return norm_ * std::exp(-0.5 * std::pow((x - mean_) / stddev_, 2)); // PDF value at x
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
            const double dxLoNorm = (a_clamped - mean_) / (std::sqrt(2.0) *stddev_);
            const double dxHiNorm = (b_clamped - mean_) / (std::sqrt(2.0) *stddev_);
            const double denom = std::erf(dxHiNorm) - std::erf(dxLoNorm);
            if (denom == 0.0) {
                return 0.0; // Avoid division by zero if the PDF is negligible in the range
            }
            return mean_ + stddev_ * std::sqrt(2 / M_PI) *
                    (std::exp(-std::pow(dxLoNorm, 2)) -
                     std::exp(-std::pow(dxHiNorm, 2))) / denom;
        }
        auto expectationValue() const -> double override {
            return expectationValue(sMin_, sMax_); // Expectation value over the full range of the segment
        }
        auto integral(const double a, const double b) const -> double override {
            if (a >= b || a >= sMax_ || b <= sMin_) {
                return 0.0; // Invalid range for integral calculation
            }
            const double a_clamped = std::max(a, sMin_);
            const double b_clamped = std::min(b, sMax_);
            const double dxLoNorm = (a_clamped - mean_) / (std::sqrt(2.0) * stddev_);
            const double dxHiNorm = (b_clamped - mean_) / (std::sqrt(2.0) * stddev_);
            return norm_ * std::sqrt(M_PI / 2) * stddev_ *
                (std::erf(dxHiNorm) - std::erf(dxLoNorm));
        }

        // Drawing functions
        auto draw(const double a, const double b) const -> double override {
            const double a_clamped = std::max(a, sMin_);
            const double b_clamped = std::min(b, sMax_);
            if (a_clamped >= b_clamped) {
                return 0.0; // Invalid range for drawing
            }
            std::normal_distribution<double> dist(mean_, stddev_);
            double sample;
            do {
                sample = dist(rng_);
            } while (sample < a_clamped || sample > b_clamped); // Rejection sampling to ensure the sample is within the specified range
            return sample;
        }
        auto draw() const -> double override {
            return draw(sMin_, sMax_);
        }
    private:
        double mean_;   /**< Mean of the normal distribution */
        double stddev_; /**< Standard deviation of the normal distribution */
        double norm_;   /**< Normalization constant for the PDF segment */
    };

}

#endif // PDFSEGMENTNORMAL_HPP
