/**
 * @file PDFSegmentLognormal.hpp
 * @author Mark Krumholz
 * @brief Class to represent a lognormal segment of a PDF.
 * @details
 * This class represents a lognormal segment of a PDF, defined by a lower
 * and upper limit, a mean, and a standard deviation of the underlying normal
 * distribution. It implements the interface defined by the PDFSegment class,
 * providing methods to evaluate the PDF at a given point and to sample a
 * random value from the PDF segment according to the lognormal distribution.
 * @date 2024-06-12
 */

#ifndef PDFSEGMENTLOGNORMAL_HPP
#define PDFSEGMENTLOGNORMAL_HPP

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
     * @class PDFSegmentLognormal
     * @brief Class representing a lognormal segment of a PDF.
     * @details
     * This class implements the PDFSegment interface for a lognormal distribution,
     * defined by the mean and standard deviation of the underlying normal
     * distribution. It provides methods to evaluate the PDF at a given point and
     * to sample a random value from the PDF segment according to the lognormal
     * distribution.
     */
    class PDFSegmentLognormal : public PDFSegment {
    public:

        // Constructor and destructor
        /**
         * @brief Constructor for PDFSegmentLognormal.
         * @param sMin The lower limit of the segment.
         * @param sMax The upper limit of the segment.
         * @param mean The mean of the x (NOT the mean of log(x)).
         * @param stddev The standard deviation of the normal distribution (i.e., stddev of log(x)).
         * @details
         * This class represents a distribution that is normal in log(x), but note
         * that, for convenience, the value given by mean is the mean of *x*, not
         * the mean of *log(x)*. On the other hand, stddev is the standard deviation
         * of *log(x)*.
         */
        PDFSegmentLognormal(double sMin, double sMax, double mean, double stddev) :
            PDFSegment(sMin, sMax), 
            mean_(mean), 
            stddev_(stddev),
            logMean_(std::log(mean_)),
            root2dev_(std::numbers::sqrt2 * stddev_)
            {
                norm_ = std::numbers::sqrt2 * std::numbers::inv_sqrtpi / 
                    stddev_ / (
                    std::erf( -std::log(sMin_/mean_) / root2dev_ ) -
                    std::erf( -std::log(sMax_/mean_) / root2dev_ )
                    );
            }
        /**
         * @brief Construct PDFSegmentLognormal from a PDF file contents.
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
        PDFSegmentLognormal(std::ifstream& file, 
            FileFormats fmt,
            double &sMin,
            double &sMax,
            double &wgt);
        PDFSegmentLognormal(const PDFSegmentLognormal&) = default;
        auto operator=(const PDFSegmentLognormal&) -> PDFSegmentLognormal& = default;
        PDFSegmentLognormal(PDFSegmentLognormal&&) = default;
        auto operator=(PDFSegmentLognormal&&) -> PDFSegmentLognormal& = default;
       ~PDFSegmentLognormal() override = default;

        // Evaluation functions
        [[nodiscard]] auto operator()(double x) const -> double override {
            if (x < sMin_ || x > sMax_)
            {
                return 0.0; // PDF is zero outside the segment
            }
            return norm_ * 
                std::exp( -0.5 * std::pow( std::log(x / mean_) / stddev_, 2) ) / x;
        }
        [[nodiscard]] auto expectationValue(const double a, const double b) const -> double override {
            if (a >= b || a > sMax_ || b < sMin_)
            {
                return 0.0; // Invalid range for expectation value calculation
            }
            if (a == sMax_)
            {
                return sMax_; // Handle edge cases
            }
            if (b == sMin_)
            {
                return sMin_; // Handle edge cases
            }
            const double aClamped = std::max(a, sMin_);
            const double bClamped = std::min(b, sMax_);
            const double logA = std::log(aClamped);
            const double logB = std::log(bClamped);
            return mean_ * std::exp(std::pow(stddev_, 2) / 2) *
                (
                    std::erf( ((logMean_ - logA + std::pow(stddev_, 2) ) / root2dev_ ) ) -
                    std::erf( ((logMean_ - logB + std::pow(stddev_, 2) ) / root2dev_ ) )
                ) / (
                    std::erf( (logB - logMean_) / root2dev_ ) -
                    std::erf( (logA - logMean_) / root2dev_ )
                );
        }
        [[nodiscard]] auto expectationValue() const -> double override {
            return expectationValue(sMin_, sMax_);
        }
        [[nodiscard]] auto integral(const double a, const double b) const -> double override {
            if (a >= b || a >= sMax_ || b <= sMin_) {
                return 0.0; // Invalid range for expectation value calculation
            }
            const double aClamped = std::max(a, sMin_);
            const double bClamped = std::min(b, sMax_);
            const double logA = std::log(aClamped);
            const double logB = std::log(bClamped);
            return norm_ /
                (std::numbers::inv_sqrtpi * std::numbers::sqrt2) *
                stddev_ * (
                std::erf( (logMean_ - logA) / root2dev_ ) -
                std::erf( (logMean_ - logB) / root2dev_ )
                );
        }

        // Drawing functions
        [[nodiscard]] auto draw(const double a, const double b) const -> double override {
            const double aClamped = std::max(a, sMin_);
            const double bClamped = std::min(b, sMax_);
            if (aClamped >= bClamped) {
                return 0.0; // Invalid range for drawing
            }
            std::lognormal_distribution<double> dist(logMean_, stddev_);
            double sample = NAN;
            while (true) {
                sample = dist(utils::rng()());
                if (sample >= aClamped && sample <= bClamped) {break; } // Rejection sampling
            } 
            return sample;
        }
        [[nodiscard]] auto draw() const -> double override {
            return draw(sMin_, sMax_);
        }

    private:
        double mean_;    /**< Mean of the underlying normal distribution */
        double stddev_;  /**< Standard deviation of the underlying normal distribution */
        double norm_;    /**< Normalization constant for the PDF segment */
        double logMean_; /**< Log of the mean; cached for convenience */
        double root2dev_; /**< sqrt(2) * stddev_; cached for convenience */
    };

} // namespace pdfs

#endif // PDFSEGMENTLOGNORMAL_HPP
