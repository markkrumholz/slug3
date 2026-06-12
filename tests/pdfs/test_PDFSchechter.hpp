/**
 * @file test_PDFSchechter.hpp
 * @author Mark Krumholz
 * @brief Unit tests for the PDFSchechter class.
 * @details
 * This file contains unit tests for the PDFSchechter class defined 
 * in the src/pdfs directory.
 * @date 2024-06-12
 */

#ifndef TEST_PDFSCHECHTER_HPP
#define TEST_PDFSCHECHTER_HPP

#include <cmath>
#include <cstdio>
#include <gsl/gsl_sf_gamma.h>
#include "../src/pdfs/PDFSchechter.hpp"
#include "../tests/testUtils.hpp"

/**
 * @brief Unit test for the PDFSchechter class.
 * @return 0 if the test passes, 1 if it fails.
 * @details
 * This function tests the PDFSchechter class by evaluating the PDF at
 * specific points, calculating expectation values and integrals over
 * specified ranges, and sampling random values from the distribution.
 * It checks that the PDF evaluates to the expected values at the mean
 * and at points one standard deviation away from the mean, as well as
 * outside the range of the distribution.
 */
auto test_PDFSchechter() -> int
{
    rngType rng(42); // Create a random number generator with a fixed seed for reproducibility
    double sMin = 1.0; // The lower limit of the segment
    double sMax = 10.0; // The upper limit of the segment
    double sStar = 5.0; // The characteristic scale of the Schechter function

    // Loop over three different alpha values to make sure we cover the different cases
    // for the expectation value and integral calculations
    std::vector<double> alphas = {-2.0, -1.0, 0.0}; // The power-law indices to test
    for (double alpha : alphas) {

        pdfs::PDFSchechter ps(sMin, sMax, sStar, alpha, rng); // Create a PDFSchechter with sMin=1, sMax=10, sStar=5, alpha=alpha

        // Test PDF evaluation at specific points
        double x1 = 1.0; // At the lower limit
        double x2 = 5.0; // Within the range
        double x3 = 10.0; // At the upper limit
        double x4 = 0.5; // Below the lower limit
        double x5 = 15.0; // Above the upper limit
        double norm = 1.0 / (std::pow(sStar, alpha + 1) * (
            gsl_sf_gamma_inc(alpha + 1, sMin / sStar) -
            gsl_sf_gamma_inc(alpha + 1, sMax / sStar)
        ));
        if (!testUtils::approxEqual(ps(x1), norm  * std::exp(-x1 / sStar))) {
            std::cerr << "test_PDFSchechter: PDF evaluation with alpha=" << alpha << " at x=1.0 failed: expected " << norm << ", got " << ps(x1) << std::endl;
            return 1;
        }
        if (!testUtils::approxEqual(ps(x2), norm * std::pow(x2, alpha) * std::exp(-x2 / sStar))) {
            std::cerr << "test_PDFSchechter: PDF evaluation with alpha=" << alpha << " at x=5.0 failed: expected " << norm * std::pow(x2, alpha) << ", got " << ps(x2) << std::endl;
            return 1;
        }
        if (!testUtils::approxEqual(ps(x3), norm * std::pow(x3, alpha) * std::exp(-x3 / sStar))) {
            std::cerr << "test_PDFSchechter: PDF evaluation with alpha=" << alpha << " at x=10.0 failed: expected " << norm * std::pow(x3, alpha) << ", got " << ps(x3) << std::endl;
            return 1;
        }
        if (!testUtils::approxEqual(ps(x4), 0.0)) {
            std::cerr << "test_PDFSchecter: PDF evaluation with alpha=" << alpha << " at x=0.5 failed: expected 0.0, got " << ps(x4) << std::endl;
            return 1;
        }
        if (!testUtils::approxEqual(ps(x5), 0.0)) {
            std::cerr << "test_PDFSchecter: PDF evaluation with alpha=" << alpha << " at x=15.0 failed: expected 0.0, got " << ps(x5) << std::endl;
            return 1;
        }

        // Test expectation value calculation over full range
        double expected_ev_full =
            sStar * (
                gsl_sf_gamma_inc(alpha + 2, sMin / sStar) -
                gsl_sf_gamma_inc(alpha + 2, sMax / sStar)
            ) / (
                gsl_sf_gamma_inc(alpha + 1, sMin / sStar) -
                gsl_sf_gamma_inc(alpha + 1, sMax / sStar)
            );
        if (!testUtils::approxEqual(ps.expectationValue(), expected_ev_full)) {
            std::cerr << "test_PDFSchechter: Expectation value calculation with alpha=" << alpha << " failed: expected " << expected_ev_full << ", got " << ps.expectationValue() << std::endl;
            return 1;
        }

        // Test expectation value calculation over a specified range
        double a = 2.0;
        double b = 8.0;
        double expected_ev =
            sStar * (
                gsl_sf_gamma_inc(alpha + 2, a / sStar) -
                gsl_sf_gamma_inc(alpha + 2, b / sStar)
            ) / (
                gsl_sf_gamma_inc(alpha + 1, a / sStar) -
                gsl_sf_gamma_inc(alpha + 1, b / sStar)
            );
        if (!testUtils::approxEqual(ps.expectationValue(a, b), expected_ev)) {
            std::cerr << "test_PDFSchechter: Expectation value calculation with alpha=" << alpha << " failed: expected " << expected_ev << ", got " << ps.expectationValue(a, b) << std::endl;
            return 1;
        }

        // Test integral calculation over a specified range
        double expected_integral =
            norm * std::pow(sStar, alpha + 1) * (
                gsl_sf_gamma_inc(alpha + 1, a / sStar) -
                gsl_sf_gamma_inc(alpha + 1, b / sStar)
            );
        if (!testUtils::approxEqual(ps.integral(a, b), expected_integral)) {
            std::cerr << "test_PDFSchechter: Integral calculation with alpha=" << alpha << " failed: expected " << expected_integral << ", got " << ps.integral(a, b) << std::endl;
            return 1;
        }

        // Test random sampling
        const int num_samples = 10000;
        double sum_samples = 0.0;
        for (int i = 0; i < num_samples; ++i) {
            double sample = ps.draw(a, b);
            if (sample < a || sample > b) {        
                std::cerr << "test_PDFSchechter: Random sampling with alpha=" << alpha << " failed: sample " << sample << " is out of bounds [" << a << ", " << b << "]" << std::endl;
                return 1;
            }
            sum_samples += sample;
        }
        double sample_mean = sum_samples / num_samples;
        if (!testUtils::approxEqual(sample_mean, expected_ev, 1e-2)) {
            std::cerr << "test_PDFSchecter: Random sampling with alpha=" << alpha << " mean failed: expected " << expected_ev << ", got " << sample_mean << std::endl;
            return 1;
        }
    }

    return 0; // If we get here, the test passed
}


#endif // TEST_PDFSCHECHTER_HPP