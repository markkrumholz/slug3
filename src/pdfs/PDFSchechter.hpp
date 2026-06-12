/**
 * @file PDFSchechter.hpp
 * @author Mark Krumholz
 * @brief Class to represent a Schechter-function segment of a PDF.
 * @details
 * This class represents a Schechter-function segment of a PDF, defined by a lower
 * and upper limit, a characteristic scale, and a power-law index. It implements the interface defined
 * by the PDFSegment class, providing methods to evaluate the PDF at a given
 * point and to sample a random value from the PDF segment according to the Schechter-function distribution.
 * @date 2024-06-12
 */

#ifndef PDFSCHECHTER_HPP
#define PDFSCHECHTER_HPP

#include <cmath>
#include <random>
#include <gsl/gsl_sf_gamma.h>
#include "PDFSegment.hpp"

namespace pdfs {

    /**
     * @class PDFSchechter
     * @brief Class representing a Schechter-function segment of a PDF.
     * @details
     * This class implements the PDFSegment interface for a Schechter-function distribution,
     * defined by a characteristic scale and a power-law index. It provides methods to evaluate the PDF at a
     * given point and to sample a random value from the PDF segment according to the Schechter-function distribution.
     */
    class PDFSchechter : public PDFSegment {
    public:

        /**
         * @brief Constructor for PDFSchechter.
         * @param sMin The lower limit of the segment.
         * @param sMax The upper limit of the segment.
         * @param sStar The characteristic scale of the Schechter function.
         * @param alpha The power-law index of the Schechter function.
         * @param rng Reference to the random number generator to be used for sampling.
         */
        PDFSchechter(double sMin, double sMax, double sStar, double alpha, rngType &rng) :
            PDFSegment(sMin, sMax, rng), sStar_(sStar), alpha_(alpha) {
                // Calculate normalization constant for the PDF segment
                norm_ = 1.0 / (sStar_ * (
                    gsl_sf_gamma_inc(alpha_ + 1, sMin_ / sStar_) -
                    gsl_sf_gamma_inc(alpha_ + 1, sMax_ / sStar_)
                ));
            }
        ~PDFSchechter() override = default;

        // Evaluation functions
        auto operator()(double x) const -> double override {
            if (x < sMin_ || x > sMax_) {
                return 0.0;
            }
            return norm_ * std::pow(x / sStar_, alpha_) * std::exp(-x / sStar_);
        }
        auto expectationValue(const double a, const double b) const -> double override {
            if (a >= b) {
                return 0.0; // Invalid range for expectation value calculation
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
        auto integral(const double a, const double b) const -> double override {
            if (a >= b) {
                return 0.0; // Invalid range for integral calculation
            }
            const double a_clamped = std::max(a, sMin_);
            const double b_clamped = std::min(b, sMax_);
            return norm_ * sStar_ * (
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
            const double u = dist(rng_); // Uniform random number in [0, 1)
            const double gamma_a = gsl_sf_gamma_inc(alpha_ + 1, a_clamped / sStar_);
            const double gamma_b = gsl_sf_gamma_inc(alpha_ + 1, b_clamped / sStar_);
            const double target = u * (gamma_b - gamma_a) + gamma_a;
            // Inverse transform sampling using the incomplete gamma function
            double x = a_clamped;
            double step = (b_clamped - a_clamped) / 2.0;
            while (step > 1e-6) {
                double gamma_x = gsl_sf_gamma_inc(alpha_ + 1, x / sStar_);
                if (gamma_x < target) {
                    x += step;
                } else {
                    x -= step;
                }
                step /= 2.0;
            }
            return x;
        }

    private:
        double sStar_;  /**< Characteristic scale of the Schechter function */
        double alpha_;  /**< Power-law index of the Schechter function */
        double norm_;   /**< Normalization constant for the PDF segment */
    };

}

#endif // PDFSCHECHTER_HPP