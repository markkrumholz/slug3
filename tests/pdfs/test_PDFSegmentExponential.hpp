/**
 * @file test_PDFSegmentExponential.hpp
 * @author Mark Krumholz
 * @brief Unit tests for the PDFSegmentExponential class.
 * @details
 * This file contains unit tests for the PDFSegmentExponential class,
 * which represents an exponential segment of a PDF. The tests
 * cover the evaluation of the PDF at specific points, the
 * calculation of expectation values and integrals over specified
 * ranges, and the sampling of random values from the distribution.
 * @date 2024-06-12
 */

#ifndef TEST_PDFSEGMENTEXPONENTIAL_HPP
#define TEST_PDFSEGMENTEXPONENTIAL_HPP

#include <cmath>
#include <cstdio>
#include "../src/pdfs/PDFSegmentExponential.hpp"
#include "../tests/testUtils.hpp"

/**
 * @brief Unit test for the PDFSegmentExponential class.
 * @return 0 if the test passes, 1 if it fails.
 * @details
 * This function tests the PDFSegmentExponential class by evaluating the PDF at
 * specific points, calculating expectation values and integrals over
 * specified ranges, and sampling random values from the distribution.
 * It checks that the PDF evaluates to the expected values at the lower
 * and upper limits, as well as within the range.
 */
auto test_PDFSegmentExponential() -> int
{
    pdfs::RngType rng(42); // Create a random number generator with a fixed seed for reproducibility
    double sMin = 1.0; // The lower limit of the segment
    double sMax = 10.0; // The upper limit of the segment
    double scale = 2.0; // The exponential scale length
    pdfs::PDFSegmentExponential pe(sMin, sMax, scale, rng); // Create a PDFSegmentExponential with sMin=1, sMax=10, scale=2

    // Test PDF evaluation at specific points
    double x1 = 1.0; // At the lower limit
    double x2 = 5.0; // Within the range
    double x3 = 10.0; // At the upper limit
    double x4 = 0.5; // Below the lower limit
    double x5 = 15.0; // Above the upper limit
    double norm = 1.0 / (scale * (std::exp(-sMin / scale) - std::exp(-sMax / scale))); // Normalization constant for the PDF
    if (!testUtils::approxEqual(pe(x1), norm * std::exp(-x1 / scale))) {
        std::cerr << "test_PDFSegmentExponential: PDF evaluation at x=1.0 failed: expected " << norm * std::exp(-x1 / scale) << ", got " << pe(x1) << std::endl;
        return 1;
    }
    if (!testUtils::approxEqual(pe(x2), norm * std::exp(-x2 / scale))) {
        std::cerr << "test_PDFSegmentExponential: PDF evaluation at x=5.0 failed: expected " << norm * std::exp(-x2 / scale) << ", got " << pe(x2) << std::endl;
        return 1;
    }
    if (!testUtils::approxEqual(pe(x3), norm * std::exp(-x3 / scale))) {
        std::cerr << "test_PDFSegmentExponential: PDF evaluation at x=10.0 failed: expected " << norm * std::exp(-x3 / scale) << ", got " << pe(x3) << std::endl;
        return 1;
    }
    if (!testUtils::approxEqual(pe(x4), 0.0)) {
        std::cerr << "test_PDFSegmentExponential: PDF evaluation at x=0.5 failed: expected 0.0, got " << pe(x4) << std::endl;
        return 1;
    }
    if (!testUtils::approxEqual(pe(x5), 0.0)) {
        std::cerr << "test_PDFSegmentExponential: PDF evaluation at x=15.0 failed: expected 0.0, got " << pe(x5) << std::endl;
        return 1;
    }

    // Test expectation value calculation over full range
    double expected_expectation_value = scale +
        (sMin * std::exp(-sMin / scale) - sMax * std::exp(-sMax / scale)) /
        (std::exp(-sMin / scale) - std::exp(-sMax / scale));
    if (!testUtils::approxEqual(pe.expectationValue(), expected_expectation_value)) {
        std::cerr << "test_PDFSegmentExponential: Expectation value calculation over full range failed: expected " << expected_expectation_value << ", got " << pe.expectationValue() << std::endl;
        return 1;
    }

    // Test expectation value calculation over a specified range
    double a = 2.0; // Lower limit of the range for expectation value calculation
    double b = 8.0; // Upper limit of the range for expectation value calculation
    double expected_expectation_value_range = scale +
        (std::max(a, sMin) * std::exp(-std::max(a, sMin) / scale) - std::min(b, sMax) * std::exp(-std::min(b, sMax) / scale)) /
        (std::exp(-std::max(a, sMin) / scale) - std::exp(-std::min(b, sMax) / scale));
    if (!testUtils::approxEqual(pe.expectationValue(a, b), expected_expectation_value_range)) {
        std::cerr << "test_PDFSegmentExponential: Expectation value calculation over range [2, 8] failed: expected " << expected_expectation_value_range << ", got " << pe.expectationValue(a, b) << std::endl;
        return 1;
    }

    // Test integral calculation over full range
    if (!testUtils::approxEqual(pe.integral(sMin, sMax), 1.0)) {
        std::cerr << "test_PDFSegmentExponential: Integral calculation over full range failed: expected 1.0, got " << pe.integral(sMin, sMax) << std::endl;
        return 1;
    }

    // Test integral calculation over a specified range
    double expected_integral = norm * scale * (std::exp(-std::max(a, sMin) / scale) - std::exp(-std::min(b, sMax) / scale));
    if (!testUtils::approxEqual(pe.integral(a, b), expected_integral)) {
        std::cerr << "test_PDFSegmentExponential: Integral calculation over range [2, 8] failed: expected " << expected_integral << ", got " << pe.integral(a, b) << std::endl;
        return 1;
    }

    // Test sampling from the distribution
    const int num_samples = 100000; // Number of samples to draw
    double sample_sum = 0.0; // Sum of the drawn samples for calculating the sample mean
    for (int i = 0; i < num_samples; ++i) {
        double sample = pe.draw(a, b); // Draw a sample from the PDFSegmentExponential over the range [2, 8]
        if (sample < a || sample > b) {
            std::cerr << "test_PDFSegmentExponential: Sample drawn from PDFSegmentExponential is out of range: got " << sample << ", expected between " << a << " and " << b << std::endl;
            return 1;
        }
        sample_sum += sample; // Add the sample to the sum for calculating the sample mean
    }
    double sample_mean = sample_sum / num_samples; // Calculate the sample mean
    if (!testUtils::approxEqual(sample_mean, expected_expectation_value_range, 0.01)) {
        std::cerr << "test_PDFSegmentExponential: Sample mean from drawn samples does not match expected expectation value: expected " << expected_expectation_value_range << ", got " << sample_mean << std::endl;
        return 1;
    }

    return 0; // If we get here, the test passed
}

#endif // TEST_PDFSEGMENTEXPONENTIAL_HPP
