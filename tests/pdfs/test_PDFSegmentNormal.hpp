/**
 * @file test_PDFSegmentNormal.hpp
 * @author Mark Krumholz
 * @brief Unit tests for the PDFSegmentNormal class.
 * @details
 * This file contains unit tests for the PDFSegmentNormal class defined
 * in the src/pdfs directory.
 * @date 2024-06-12
 */

#ifndef TEST_PDFSEGMENTNORMAL_HPP
#define TEST_PDFSEGMENTNORMAL_HPP

#include <cmath>
#include <cstdio>
#include "../src/pdfs/PDFSegmentNormal.hpp"
#include "../tests/testUtils.hpp"

/**
 * @brief Unit test for the PDFSegmentNormal class.
 * @return 0 if the test passes, 1 if it fails.
 * @details
 * This function tests the PDFSegmentNormal class by evaluating the PDF at
 * specific points, calculating expectation values and integrals over
 * specified ranges, and sampling random values from the distribution.
 * It checks that the PDF evaluates to the expected values at the mean
 * and at points one standard deviation away from the mean, as well as
 * outside the range of the distribution.
 */
auto test_PDFSegmentNormal() -> int
{
    rngType rng(42); // Create a random number generator with a fixed seed for reproducibility
    double mean = 5.0; // The mean of the normal distribution
    double stdDev = 1.0; // The standard deviation of the normal distribution
    double sMin = 1.0; // The lower limit of the segment
    double sMax = 6.0; // The upper limit of the segment
    pdfs::PDFSegmentNormal pn(sMin, sMax, mean, stdDev, rng); // Create a PDFSegmentNormal with mean=5, stdDev=1, sMin=1, sMax=6

    // Test PDF evaluation at specific points
    double x1 = 1.0; // At the lower limit
    double x2 = 5.0; // Within the range
    double x3 = 6.0; // At the upper limit
    double x4 = 0.5; // Below the lower limit
    double x5 = 15.0; // Above the upper limit
    double norm = std::sqrt(2.0 / M_PI) / stdDev /
        (std::erf((sMax - mean) / (stdDev * std::sqrt(2))) -
         std::erf((sMin - mean) / (stdDev * std::sqrt(2)))); // Normalization constant for the PDFSegmentNormal
    if (!testUtils::approxEqual(pn(x1), norm * std::exp(-0.5 * std::pow((x1 - mean) / stdDev, 2)))) {
        std::cerr << "test_PDFSegmentNormal: PDF evaluation at x=1.0 failed: expected " << norm *
            std::exp(-0.5 * std::pow((x1 - mean) / stdDev, 2)) << ", got " << pn(x1) << std::endl;
        return 1;
    }
    if (!testUtils::approxEqual(pn(x2), norm * std::exp(-0.5 * std::pow((x2 - mean) / stdDev, 2)))) {
        std::cerr << "test_PDFSegmentNormal: PDF evaluation at x=5.0 failed: expected " << norm *
            std::exp(-0.5 * std::pow((x2 - mean) / stdDev, 2)) << ", got " << pn(x2) << std::endl;
        return 1;
    }
    if (!testUtils::approxEqual(pn(x3), norm * std::exp(-0.5 * std::pow((x3 - mean) / stdDev, 2)))) {
        std::cerr << "test_PDFSegmentNormal: PDF evaluation at x=10.0 failed: expected " << norm *
            std::exp(-0.5 * std::pow((x3 - mean) / stdDev, 2)) << ", got " << pn(x3) << std::endl;
        return 1;
    }
    if (!testUtils::approxEqual(pn(x4), 0.0)) {
        std::cerr << "test_PDFSegmentNormal: PDF evaluation at x=0.5 failed: expected 0.0, got " << pn(x4) << std::endl;
        return 1;
    }
    if (!testUtils::approxEqual(pn(x5), 0.0)) {
        std::cerr << "test_PDFSegmentNormal: PDF evaluation at x=15.0 failed: expected 0.0, got " << pn(x5) << std::endl;
        return 1;
    }

    // Test expectation value calculation over full range
    double dxLoNorm = (sMin - mean) / (std::sqrt(2.0) * stdDev);
    double dxHiNorm = (sMax - mean) / (std::sqrt(2.0) * stdDev);
    double expected_expectation_value = mean +
        std::sqrt(2.0 / M_PI) * stdDev *
        (std::exp(-std::pow(dxLoNorm, 2)) -
         std::exp(-std::pow(dxHiNorm, 2))) /
        (std::erf(dxHiNorm) - std::erf(dxLoNorm));
    if (!testUtils::approxEqual(pn.expectationValue(), expected_expectation_value)) {
        std::cerr << "test_PDFSegmentNormal: Expectation value calculation over full range failed: expected " << expected_expectation_value << ", got " << pn.expectationValue() << std::endl;
        return 1;
    }

    // Test expectation value calculation over a specified range
    double a = 2.0; // Lower limit of the range for expectation value calculation
    double b = 5.0; // Upper limit of the range for expectation value calculation
    dxLoNorm = (a - mean) / (std::sqrt(2.0) * stdDev);
    dxHiNorm = (b - mean) / (std::sqrt(2.0) * stdDev);
    double expected_expectation_value_range = mean +
        std::sqrt(2.0 / M_PI) * stdDev *
        (std::exp(-std::pow(dxLoNorm, 2)) -
         std::exp(-std::pow(dxHiNorm, 2))) /
        (std::erf(dxHiNorm) - std::erf(dxLoNorm));
    if (!testUtils::approxEqual(pn.expectationValue(a, b), expected_expectation_value_range)) {
        std::cerr << "test_PDFSegmentNormal: Expectation value calculation over range [2.0, 5.0] failed: expected " << expected_expectation_value_range << ", got " << pn.expectationValue(a, b) << std::endl;
        return 1;
    }

    // Test integral calculation over full range
    if (!testUtils::approxEqual(pn.integral(sMin, sMax), 1.0)) {
        std::cerr << "test_PDFSegmentNormal: Integral calculation over full range failed: expected 1.0, got " << pn.integral(sMin, sMax) << std::endl;
        return 1;
    }
    
    // Test integral calculation over a specified range
    double expected_integral = norm * stdDev * std::sqrt(M_PI / 2) *
        (std::erf(dxHiNorm) - std::erf(dxLoNorm));
    if (!testUtils::approxEqual(pn.integral(a, b), expected_integral)) {
        std::cerr << "test_PDFSegmentNormal: Integral calculation over range [2.0, 5.0] failed: expected " << expected_integral << ", got " << pn.integral(a, b) << std::endl;
        return 1;
    }

    // Test sampling from the distribution
    const int num_samples = 100000; // Number of samples to draw
    double sample_sum = 0.0; // Sum of the drawn samples for calculating the
    for (int i = 0; i < num_samples; ++i) {
        double sample = pn.draw(a, b); // Draw a sample from the PDFSegmentNormal over the range [2, 5]
        if (sample < a || sample > b) {
            std::cerr << "test_PDFSegmentNormal: Sample drawn from PDFSegmentNormal is out of range: got " << sample << ", expected between " << a << " and " << b << std::endl;
            return 1;
        }
        sample_sum += sample; // Add the sample to the sum for calculating the sample mean
    }
    double sample_mean = sample_sum / num_samples; // Calculate the sample mean
    if (!testUtils::approxEqual(sample_mean, expected_expectation_value_range, 0.01)) {
        std::cerr << "test_PDFSegmentNormal: Sample mean from drawn samples does not match expected expectation value: expected " << expected_expectation_value_range << ", got " << sample_mean << std::endl;
        return 1;
    }

    return 0; // If we get here, the test passed

}
#endif // TEST_PDFSEGMENTNORMAL_HPP
