/**
 * @file PDFSegmentSchechter.hpp
 * @author Mark Krumholz
 * @brief Class to represent a Schechter-function segment of a PDF.
 * @details
 * This class represents a Schechter-function segment of a PDF, defined by a lower
 * and upper limit, a characteristic scale, and a power-law index. It implements the interface defined
 * by the PDFSegment class, providing methods to evaluate the PDF at a given
 * point and to sample a random value from the PDF segment according to the Schechter-function distribution.
 * @date 2024-06-12
 */

#ifndef PDFSEGMENTSCHECHTER_HPP
#define PDFSEGMENTSCHECHTER_HPP

#include <cmath>
#include <random>
#include <gsl/gsl_sf_gamma.h>
#include "PDFCommons.hpp"
#include "PDFSegment.hpp"
#include "../utils/RngThread.hpp"

namespace pdfs {

    /**
     * @class PDFSegmentSchechter
     * @brief Class representing a Schechter-function segment of a PDF.
     * @details
     * This class implements the PDFSegment interface for a Schechter-function distribution,
     * defined by a characteristic scale and a power-law index. It provides methods to evaluate the PDF at a
     * given point and to sample a random value from the PDF segment according to the Schechter-function distribution.
     */
    class PDFSegmentSchechter : public PDFSegment {
    public:

        /**
         * @brief Constructor for PDFSegmentSchechter.
         * @param sMin The lower limit of the segment.
         * @param sMax The upper limit of the segment.
         * @param sStar The characteristic scale of the Schechter function.
         * @param alpha The power-law index of the Schechter function.
         */
        PDFSegmentSchechter(double sMin, double sMax, double sStar, double alpha) :
            PDFSegment(sMin, sMax), sStar_(sStar), alpha_(alpha) {
                // Calculate normalization constant for the PDF segment
                norm_ = 1.0 / (
                    std::pow(sStar_, alpha_ + 1) * (
                        gsl_sf_gamma_inc(alpha_ + 1, sMin_ / sStar_) -
                        gsl_sf_gamma_inc(alpha_ + 1, sMax_ / sStar_)
                    )
                );
            }
        /**
         * @brief Construct PDFSegmentSchechter from a PDF file contents.
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
        PDFSegmentSchechter(std::ifstream& file, 
            FileFormats fmt,
            double &sMin,
            double &sMax,
            double &wgt);
        PDFSegmentSchechter(const PDFSegmentSchechter&) = default;
        auto operator=(const PDFSegmentSchechter&) -> PDFSegmentSchechter& = default;
        PDFSegmentSchechter(PDFSegmentSchechter&&) = default;
        auto operator=(PDFSegmentSchechter&&) -> PDFSegmentSchechter& = default;
        ~PDFSegmentSchechter() override = default;

        // Evaluation functions
        auto operator()(double x) const -> double override {
            if (x < sMin_ || x > sMax_) {
                return 0.0;
            }
            return norm_ * std::pow(x, alpha_) * std::exp(-x / sStar_);
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
            return sStar_ * (
                gsl_sf_gamma_inc(alpha_ + 2, a_clamped / sStar_) -
                gsl_sf_gamma_inc(alpha_ + 2, b_clamped / sStar_)
            ) / (
                gsl_sf_gamma_inc(alpha_ + 1, a_clamped / sStar_) -
                gsl_sf_gamma_inc(alpha_ + 1, b_clamped / sStar_)
            );
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
            return norm_ * std::pow(sStar_, alpha_ + 1) * (
                gsl_sf_gamma_inc(alpha_ + 1, a_clamped / sStar_) -
                gsl_sf_gamma_inc(alpha_ + 1, b_clamped / sStar_)
            );
        }

        // Drawing function
        auto draw(const double a, const double b) const -> double override {
            const double a_clamped = std::max(a, sMin_);
            const double b_clamped = std::min(b, sMax_);
            if (a_clamped >= b_clamped) {
                return 0.0; // Invalid range for drawing
            }
            std::uniform_real_distribution<double> dist(0.0, 1.0);
            const double u = dist(utils::rng()); // Uniform random number in [0, 1)
            // Find the value of the deviate y by numerically solving
            // u = \int_a^y x^alpha exp(-x / sStar) dx /
            //     \int_a^b x^alpha exp(-x / sStar) dx
            // using a simple bisection search.
            const double gamma_a = gsl_sf_gamma_inc(alpha_ + 1, a_clamped / sStar_);
            const double gamma_b = gsl_sf_gamma_inc(alpha_ + 1, b_clamped / sStar_);
            const double denom = gamma_a - gamma_b;
            double y = a_clamped;
            double step = (b_clamped - a_clamped) / 2.0;
            while (true) {
                y += step;
                const double gamma_y = gsl_sf_gamma_inc(alpha_ + 1, y / sStar_);
                const double resid = u - (gamma_a - gamma_y) / denom;
                if (fabs(resid) < 1e-8) {
                    break;
                } else if (resid < 0) {
                    y -= step;
                    step /= 2.0;
                    if (step < 1e-8) {
                        break;
                    }
                }
            }
            return y;
        }
        auto draw() const -> double override {
            return draw(sMin_, sMax_);
        }

    private:
        double sStar_;  /**< Characteristic scale of the Schechter function */
        double alpha_;  /**< Power-law index of the Schechter function */
        double norm_;   /**< Normalization constant for the PDF segment */
    };

} // namespace pdfs

#endif // PDFSEGMENTSCHECHTER_HPP
