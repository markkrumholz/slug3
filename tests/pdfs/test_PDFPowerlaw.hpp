/** 
 * @file test_PDFPowerlaw.hpp
 * @author Mark Krumholz
 * @brief Unit tests for the PDFPowerlaw class.
 * @details
 * This file contains unit tests for the PDFPowerlaw class,
 * which represents a power-law segment of a PDF. The tests
 * cover the evaluation of the PDF at specific points, the
 * calculation of expectation values and integrals over specified
 * ranges, and the sampling of random values from the distribution.
 * @date 2024-06-12
 */

#ifndef TEST_PDFPOWERLAW_HPP
#define TEST_PDFPOWERLAW_HPP

#include <cmath>
#include <cstdio>
#include "../src/pdfs/PDFPowerlaw.hpp"
#include "../tests/testUtils.hpp"

/**
 * @brief Unit test for the PDFPowerlaw class.
 * @return 0 if the test passes, 1 if it fails.
 * @details
 * This function tests the PDFPowerlaw class by evaluating the PDF at
 * specific points, calculating expectation values and integrals over
 * specified ranges, and sampling random values from the distribution.
 * It checks that the PDF evaluates to the expected values at the lower
 * and upper limits, as well as within the range.
 */
auto test_PDFPowerlaw() -> int
{
    rngType rng(42); // Create a random number generator with a fixed seed for reproducibility

    // Loop over three different alpha values to make sure we cover the different cases
    // for the expectation value and integral calculations
    std::vector<double> alphas = {-2.0, -1.0, 0.0};
    for (double alpha : alphas) {

        pdfs::PDFPowerlaw pl(1.0, 10.0, alpha, rng); // Create a PDFPowerlaw with sMin=1, sMax=10, alpha=alpha

        // Test PDF evaluation at specific points
        double x1 = 1.0; // At the lower limit
        double x2 = 5.0; // Within the range
        double x3 = 10.0; // At the upper limit
        double x4 = 0.5; // Below the lower limit
        double x5 = 15.0; // Above the upper limit
        double norm;
        if (alpha != -1) {
            norm = (alpha + 1) / (std::pow(10.0, alpha + 1) - std::pow(1.0, alpha + 1));
        } else {
            norm = 1.0 / std::log(10.0);
        }
        if (!testUtils::approxEqual(pl(x1), norm)) {
            std::cerr << "test_PDFPowerlaw: PDF evaluation with alpha=" << alpha << " at x=1.0 failed: expected " << norm << ", got " << pl(x1) << std::endl;
            return 1;
        }
        if (!testUtils::approxEqual(pl(x2), norm * std::pow(x2, alpha))) {
            std::cerr << "test_PDFPowerlaw: PDF evaluation with alpha=" << alpha << " at x=5.0 failed: expected " << norm * std::pow(x2, alpha) << ", got " << pl(x2) << std::endl;
            return 1;
        }
        if (!testUtils::approxEqual(pl(x3), norm * std::pow(x3, alpha))) {
            std::cerr << "test_PDFPowerlaw: PDF evaluation with alpha=" << alpha << " at x=10.0 failed: expected " << norm * std::pow(x3, alpha) << ", got " << pl(x3) << std::endl;
            return 1;
        }
        if (!testUtils::approxEqual(pl(x4), 0.0)) {
            std::cerr << "test_PDFPowerlaw: PDF evaluation with alpha=" << alpha << " at x=0.5 failed: expected 0.0, got " << pl(x4) << std::endl;
            return 1;
        }
        if (!testUtils::approxEqual(pl(x5), 0.0)) {
            std::cerr << "test_PDFPowerlaw: PDF evaluation with alpha=" << alpha << " at x=15.0 failed: expected 0.0, got " << pl(x5) << std::endl;
            return 1;
        }

        // Test expectation value calculation over full range
        double expected_ev_full;
        if (alpha == -2.0) {
            expected_ev_full = 10.0/9.0 * std::log(10.0); // Expected expectation value for alpha=-2 over [1, 10]
        } else if (alpha == -1.0) {
            expected_ev_full = 9.0 / std::log(10.0); // Expected expectation value for alpha=-1 over [1, 10]
        } else {
            expected_ev_full = (alpha + 1) / (alpha + 2) * (std::pow(10.0, alpha + 2) - std::pow(1.0, alpha + 2)) / (std::pow(10.0, alpha + 1) - std::pow(1.0, alpha + 1)); // Expected expectation value for alpha=0 over [1, 10]
        }
        if (!testUtils::approxEqual(pl.expectationValue(), expected_ev_full)) {
            std::cerr << "test_PDFPowerlaw: Expectation value calculation with alpha=" << alpha << " failed: expected " << expected_ev_full << ", got " << pl.expectationValue() << std::endl;
            return 1;
        }

        // Test expectation value calculation over a specified range
        double a = 2.0;
        double b = 8.0;
        double expected_ev;
        if (alpha == -2.0) {
            expected_ev = (std::log(b/a) / (1.0/a - 1.0/b)); // Expected expectation value for alpha=-2 over [2, 8]
        } else if (alpha == -1.0) {
            expected_ev = (b - a) / std::log(b/a); // Expected expectation value for alpha=-1 over [2, 8]
        } else {
            expected_ev = (alpha + 1) / (alpha + 2) * (std::pow(b, alpha + 2) - std::pow(a, alpha + 2)) / (std::pow(b, alpha + 1) - std::pow(a, alpha + 1)); // Expected expectation value for alpha=0 over [2, 8]
        }
        if (!testUtils::approxEqual(pl.expectationValue(a, b), expected_ev)) {
            std::cerr << "test_PDFPowerlaw: Expectation value calculation with alpha=" << alpha << " failed: expected " << expected_ev << ", got " << pl.expectationValue(a, b) << std::endl;
            return 1;
        }

        // Test integral calculation over a specified range
        double expected_integral;
        if (alpha == -2.0) {
            expected_integral = norm * (1.0/a - 1.0/b); // Expected integral for alpha=-2 over [2, 8]
        }  else if (alpha == -1.0) {
            expected_integral = norm * std::log(b/a); // Expected integral for alpha=-1 over [2, 8]
        } else {
            expected_integral = norm / (alpha + 1) * (std::pow(b, alpha + 1) - std::pow(a, alpha + 1)); // Expected integral for alpha=0 over [2, 8]
        }
        if (!testUtils::approxEqual(pl.integral(a, b), expected_integral)) {
            std::cerr << "test_PDFPowerlaw: Integral calculation with alpha=" << alpha << " failed: expected " << expected_integral << ", got " << pl.integral(a, b) << std::endl;
            return 1;
        }

        // Test random sampling
        const int num_samples = 100000;
        double sum_samples = 0.0;
        for (int i = 0; i < num_samples; ++i) {
            double sample = pl.draw(a, b);
            if (sample < a || sample > b) {        
                std::cerr << "test_PDFPowerlaw: Random sampling with alpha=" << alpha << " failed: sample " << sample << " is out of bounds [" << a << ", " << b << "]" << std::endl;
                return 1;
            }
            sum_samples += sample;
        }
        double sample_mean = sum_samples / num_samples;
        if (!testUtils::approxEqual(sample_mean, expected_ev, 1e-2)) {
            std::cerr << "test_PDFPowerlaw: Random sampling with alpha=" << alpha << " mean failed: expected " << expected_ev << ", got " << sample_mean << std::endl;
            return 1;
        }
    }

    // If we reach this point, all tests passed
    return 0;
}

#endif // TEST_PDFPOWERLAW_HPP