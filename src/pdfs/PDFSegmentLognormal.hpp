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

#include <algorithm>
#include <cmath>
#include <random>
#include "PDFCommons.hpp"
#include "PDFSegment.hpp"
#include "../utils/RngThread.hpp"

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
            PDFSegment(sMin, sMax), mean_(mean), stddev_(stddev) {
                log_mean_ = std::log(mean_);
                root2dev_ = std::sqrt(2.0) * stddev_;
                norm_ = std::sqrt(2.0 / M_PI) / stddev_ / (
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
       ~PDFSegmentLognormal() override = default;

        // Evaluation functions
        auto operator()(double x) const -> double override {
            if (x < sMin_ || x > sMax_) {
                return 0.0; // PDF is zero outside the segment
            }
            return norm_ * 
                std::exp( -0.5 * std::pow( std::log(x / mean_) / stddev_, 2) ) / x;
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
            const double log_a = std::log(a_clamped);
            const double log_b = std::log(b_clamped);
            return mean_ * std::exp(std::pow(stddev_, 2) / 2) *
                (
                    std::erf( ((log_mean_ - log_a + std::pow(stddev_, 2) ) / root2dev_ ) ) -
                    std::erf( ((log_mean_ - log_b + std::pow(stddev_, 2) ) / root2dev_ ) )
                ) / (
                    std::erf( (log_b - log_mean_) / root2dev_ ) -
                    std::erf( (log_a - log_mean_) / root2dev_ )
                );
        }
        auto expectationValue() const -> double override {
            return expectationValue(sMin_, sMax_);
        }
        auto integral(const double a, const double b) const -> double override {
            if (a >= b || a >= sMax_ || b <= sMin_) {
                return 0.0; // Invalid range for expectation value calculation
            }
            const double a_clamped = std::max(a, sMin_);
            const double b_clamped = std::min(b, sMax_);
            const double log_a = std::log(a_clamped);
            const double log_b = std::log(b_clamped);
            return norm_ * std::sqrt( M_PI / 2.0 ) * stddev_ * (
                std::erf( (log_mean_ - log_a) / root2dev_ ) -
                std::erf( (log_mean_ - log_b) / root2dev_ )
            );
        }

        // Drawing functions
        auto draw(const double a, const double b) const -> double override {
            const double a_clamped = std::max(a, sMin_);
            const double b_clamped = std::min(b, sMax_);
            if (a_clamped >= b_clamped) {
                return 0.0; // Invalid range for drawing
            }
            std::lognormal_distribution<double> dist(log_mean_, stddev_);
            double sample;
            do {
                sample = dist(utils::rng());
            } while (sample < a_clamped || sample > b_clamped); // Rejection sampling
            return sample;
        }
        auto draw() const -> double override {
            return draw(sMin_, sMax_);
        }

    private:
        double mean_;   /**< Mean of the underlying normal distribution */
        double stddev_; /**< Standard deviation of the underlying normal distribution */
        double norm_;   /**< Normalization constant for the PDF segment */
        double log_mean_; /**< Log of the mean; cached for convenience */
        double root2dev_; /**< sqrt(2) * stddev_; cached for convenience */
    };

}

#endif // PDFSEGMENTLOGNORMAL_HPP
