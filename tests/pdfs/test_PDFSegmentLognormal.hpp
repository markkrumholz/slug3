/**
 * @file test_PDFSegmentLognormal.hpp
 * @author Mark Krumholz
 * @brief Unit tests for the PDFSegmentLognormal class.
 * @details
 * This file contains unit tests for the PDFSegmentLognormal class defined
 * in the src/pdfs directory.
 * @date 2024-06-12
 */

#ifndef TEST_PDFSEGMENTLOGNORMAL_HPP
#define TEST_PDFSEGMENTLOGNORMAL_HPP

#include <cmath>
#include <cstdio>
#include "../src/pdfs/PDFSegmentLognormal.hpp"
#include "../tests/testUtils.hpp"

/**
 * @brief Unit test for the PDFSegmentLognormal class.
 * @return 0 if the test passes, 1 if it fails.
 * @details
 * This function tests the PDFSegmentLognormal class by evaluating the PDF at
 * specific points, calculating expectation values and integrals over
 * specified ranges, and sampling random values from the distribution.
 * It checks that the PDF evaluates to the expected values at the mean
 * and at points one standard deviation away from the mean, as well as
 * outside the range of the distribution.
 */
auto test_PDFSegmentLognormal() -> int
{
    pdfs::rngType rng(42); // Create a random number generator with a fixed seed for reproducibility
    double mean = 5.0; // The mean of x (NOT the mean of log(x))
    double stdDev = 1.0; // The standard deviation of log(x)
    double sMin = 1.0; // The lower limit of the segment
    double sMax = 6.0; // The upper limit of the segment
    pdfs::PDFSegmentLognormal pln(sMin, sMax, mean, stdDev, rng); // Create a PDFSegmentLognormal with mean=5, stdDev=1, sMin=1, sMax=6

    // Pre-compute cached quantities matching those in the constructor
    double log_mean = std::log(mean);
    double root2dev = std::sqrt(2.0) * stdDev;
    double norm = std::sqrt(2.0 / M_PI) / stdDev / (
        std::erf(-std::log(sMin / mean) / root2dev) -
        std::erf(-std::log(sMax / mean) / root2dev)
    ); // Normalization constant for the PDFSegmentLognormal

    // Test PDF evaluation at specific points
    double x1 = 1.0; // At the lower limit
    double x2 = 5.0; // Within the range, at the mean
    double x3 = 6.0; // At the upper limit
    double x4 = 0.5; // Below the lower limit
    double x5 = 15.0; // Above the upper limit
    if (!testUtils::approxEqual(pln(x1), norm / x1 * std::exp(-0.5 * std::pow(std::log(x1 / mean) / stdDev, 2)))) {
        std::cerr << "test_PDFSegmentLognormal: PDF evaluation at x=1.0 failed: expected "
                  << norm * std::exp(-0.5 * std::pow(std::log(x1 / mean) / stdDev, 2))
                  << ", got " << pln(x1) << std::endl;
        return 1;
    }
    if (!testUtils::approxEqual(pln(x2), norm / x2 * std::exp(-0.5 * std::pow(std::log(x2 / mean) / stdDev, 2)))) {
        std::cerr << "test_PDFSegmentLognormal: PDF evaluation at x=5.0 failed: expected "
                  << norm * std::exp(-0.5 * std::pow(std::log(x2 / mean) / stdDev, 2))
                  << ", got " << pln(x2) << std::endl;
        return 1;
    }
    if (!testUtils::approxEqual(pln(x3), norm / x3 * std::exp(-0.5 * std::pow(std::log(x3 / mean) / stdDev, 2)))) {
        std::cerr << "test_PDFSegmentLognormal: PDF evaluation at x=6.0 failed: expected "
                  << norm * std::exp(-0.5 * std::pow(std::log(x3 / mean) / stdDev, 2))
                  << ", got " << pln(x3) << std::endl;
        return 1;
    }
    if (!testUtils::approxEqual(pln(x4), 0.0)) {
        std::cerr << "test_PDFSegmentLognormal: PDF evaluation at x=0.5 failed: expected 0.0, got " << pln(x4) << std::endl;
        return 1;
    }
    if (!testUtils::approxEqual(pln(x5), 0.0)) {
        std::cerr << "test_PDFSegmentLognormal: PDF evaluation at x=15.0 failed: expected 0.0, got " << pln(x5) << std::endl;
        return 1;
    }

    // Test expectation value calculation over full range
    double expected_expectation_value =
        mean * std::exp(std::pow(stdDev, 2) / 2) *
        (
            std::erf( (log_mean - std::log(sMin) + std::pow(stdDev,2)) / root2dev ) -
            std::erf( (log_mean - std::log(sMax) + std::pow(stdDev,2)) / root2dev)
        ) / (
            std::erf( (std::log(sMax) - log_mean) / root2dev ) -
            std::erf( (std::log(sMin) - log_mean) / root2dev )
        );
    if (!testUtils::approxEqual(pln.expectationValue(), expected_expectation_value)) {
        std::cerr << "test_PDFSegmentLognormal: Expectation value calculation over full range failed: expected "
                  << expected_expectation_value << ", got " << pln.expectationValue() << std::endl;
        return 1;
    }

    // Test expectation value calculation over a specified range
    double a = 2.0; // Lower limit of the range for expectation value calculation
    double b = 5.0; // Upper limit of the range for expectation value calculation
    double log_a = std::log(a);
    double log_b = std::log(b);
    double expected_expectation_value_range =
        mean * std::exp(std::pow(stdDev, 2) / 2) *
        (
            std::erf( (log_mean - log_a + std::pow(stdDev,2)) / root2dev ) -
            std::erf( (log_mean - log_b + std::pow(stdDev,2)) / root2dev)
        ) / (
            std::erf( (log_b - log_mean) / root2dev ) -
            std::erf( (log_a - log_mean) / root2dev )
        );
    if (!testUtils::approxEqual(pln.expectationValue(a, b), expected_expectation_value_range)) {
        std::cerr << "test_PDFSegmentLognormal: Expectation value calculation over range [2.0, 5.0] failed: expected "
                  << expected_expectation_value_range << ", got " << pln.expectationValue(a, b) << std::endl;
        return 1;
    }

    // Test integral calculation over full range
    if (!testUtils::approxEqual(pln.integral(sMin, sMax), 1.0)) {
        std::cerr << "test_PDFSegmentLognormal: Integral calculation over full range failed: expected 1.0, got " << pln.integral(sMin, sMax) << std::endl;
        return 1;
    }
    
    // Test integral calculation over a specified range
    double expected_integral = 
        norm * std::sqrt( M_PI / 2.0 ) * stdDev * (
                std::erf( (log_mean - log_a) / root2dev ) -
                std::erf( (log_mean - log_b) / root2dev )
        );
    if (!testUtils::approxEqual(pln.integral(a, b), expected_integral)) {
        std::cerr << "test_PDFSegmentLognormal: Integral calculation over range [2.0, 5.0] failed: expected "
                  << expected_integral << ", got " << pln.integral(a, b) << std::endl;
        return 1;
    }

    // Test sampling from the distribution
    const int num_samples = 100000; // Number of samples to draw
    double sample_sum = 0.0; // Sum of the drawn samples for calculating the sample mean
    for (int i = 0; i < num_samples; ++i) {
        double sample = pln.draw(a, b); // Draw a sample from the PDFSegmentLognormal over the range [2, 5]
        if (sample < a || sample > b) {
            std::cerr << "test_PDFSegmentLognormal: Sample drawn from PDFSegmentLognormal is out of range: got "
                      << sample << ", expected between " << a << " and " << b << std::endl;
            return 1;
        }
        sample_sum += sample; // Add the sample to the sum for calculating the sample mean
    }
    double sample_mean = sample_sum / num_samples; // Calculate the sample mean
    if (!testUtils::approxEqual(sample_mean, expected_expectation_value_range, 0.01)) {
        std::cerr << "test_PDFSegmentLognormal: Sample mean from drawn samples does not match expected expectation value: expected "
                  << expected_expectation_value_range << ", got " << sample_mean << std::endl;
        return 1;
    }

    return 0; // If we get here, the test passed

}
#endif // TEST_PDFSEGMENTLOGNORMAL_HPP
