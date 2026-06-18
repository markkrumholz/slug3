/**
 * @file testPDFSegmentNormal.hpp
 * @author Mark Krumholz
 * @brief Unit tests for the PDFSegmentNormal class.
 * @details
 * This file contains unit tests for the PDFSegmentNormal class defined
 * in the src/pdfs directory.
 * @date 2024-06-12
 */

#ifndef TESTPDFSEGMENTNORMAL_HPP
#define TESTPDFSEGMENTNORMAL_HPP

#include "../src/pdfs/PDFSegmentNormal.hpp"
#include "../src/utils/RngThread.hpp"
#include "../tests/testUtils.hpp"
#include <cmath>
#include <iostream>
#include <numbers>

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
inline auto testPDFSegmentNormal() -> int
{
    // Set the rng seed to a fixed value for reproducibility
    utils::rng.seed(42);

    const double mean = 5.0; // The mean of the normal distribution
    const double stdDev = 1.0; // The standard deviation of the normal distribution
    const double sMin = 1.0; // The lower limit of the segment
    const double sMax = 6.0; // The upper limit of the segment
    const pdfs::PDFSegmentNormal pn(sMin, sMax, mean, stdDev); // Create a PDFSegmentNormal with mean=5, stdDev=1, sMin=1, sMax=6

    // Test PDF evaluation at specific points
    const double x1 = 1.0; // At the lower limit
    const double x2 = 5.0; // Within the range
    const double x3 = 6.0; // At the upper limit
    const double x4 = 0.5; // Below the lower limit
    const double x5 = 15.0; // Above the upper limit
    const double norm = std::numbers::sqrt2 * std::numbers::inv_sqrtpi / stdDev /
        (std::erf((sMax - mean) / (stdDev * std::numbers::sqrt2)) -
         std::erf((sMin - mean) / (stdDev * std::numbers::sqrt2))); // Normalization constant for the PDFSegmentNormal
    if (!testUtils::approxEqual(pn(x1), norm * std::exp(-0.5 * std::pow((x1 - mean) / stdDev, 2)))) {
        std::cerr << "testPDFSegmentNormal: PDF evaluation at x=1.0 failed: expected " << norm *
            std::exp(-0.5 * std::pow((x1 - mean) / stdDev, 2)) << ", got " << pn(x1) << "\n";
        return 1;
    }
    if (!testUtils::approxEqual(pn(x2), norm * std::exp(-0.5 * std::pow((x2 - mean) / stdDev, 2)))) {
        std::cerr << "testPDFSegmentNormal: PDF evaluation at x=5.0 failed: expected " << norm *
            std::exp(-0.5 * std::pow((x2 - mean) / stdDev, 2)) << ", got " << pn(x2) << "\n";
        return 1;
    }
    if (!testUtils::approxEqual(pn(x3), norm * std::exp(-0.5 * std::pow((x3 - mean) / stdDev, 2)))) {
        std::cerr << "testPDFSegmentNormal: PDF evaluation at x=10.0 failed: expected " << norm *
            std::exp(-0.5 * std::pow((x3 - mean) / stdDev, 2)) << ", got " << pn(x3) << "\n";
        return 1;
    }
    if (!testUtils::approxEqual(pn(x4), 0.0)) {
        std::cerr << "testPDFSegmentNormal: PDF evaluation at x=0.5 failed: expected 0.0, got " << pn(x4) << "\n";
        return 1;
    }
    if (!testUtils::approxEqual(pn(x5), 0.0)) {
        std::cerr << "testPDFSegmentNormal: PDF evaluation at x=15.0 failed: expected 0.0, got " << pn(x5) << "\n";
        return 1;
    }

    // Test expectation value calculation over full range
    double dxLoNorm = (sMin - mean) / (std::numbers::sqrt2 * stdDev);
    double dxHiNorm = (sMax - mean) / (std::numbers::sqrt2 * stdDev);
    double numLo = std::exp(-std::pow(dxLoNorm, 2));
    double numHi = std::exp(-std::pow(dxHiNorm, 2));
    double denomLo = std::erf(dxLoNorm);
    double denomHi = std::erf(dxHiNorm);
     const double expectedExpectationValue = mean +
        (std::numbers::sqrt2 * std::numbers::inv_sqrtpi * stdDev *
        (numLo - numHi) / (denomHi - denomLo));
    if (!testUtils::approxEqual(pn.expectationValue(), expectedExpectationValue)) {
        std::cerr << "testPDFSegmentNormal: Expectation value calculation over full range failed: expected " << expectedExpectationValue << ", got " << pn.expectationValue() << "\n";
        return 1;
    }

    // Test expectation value calculation over a specified range
    const double a = 2.0; // Lower limit of the range for expectation value calculation
    const double b = 5.0; // Upper limit of the range for expectation value calculation
    dxLoNorm = (a - mean) / (std::numbers::sqrt2 * stdDev);
    dxHiNorm = (b - mean) / (std::numbers::sqrt2 * stdDev);
    numLo = std::exp(-std::pow(dxLoNorm, 2));
    numHi = std::exp(-std::pow(dxHiNorm, 2));
    denomLo = std::erf(dxLoNorm);
    denomHi = std::erf(dxHiNorm);
    const double expectedExpectationValueRange = mean +
        (std::numbers::sqrt2 * std::numbers::inv_sqrtpi * stdDev *
        ((numLo - numHi) / (denomHi - denomLo)));
    if (!testUtils::approxEqual(pn.expectationValue(a, b), expectedExpectationValueRange)) {
        std::cerr << "testPDFSegmentNormal: Expectation value calculation over range [2.0, 5.0] failed: expected " << expectedExpectationValueRange << ", got " << pn.expectationValue(a, b) << "\n";
        return 1;
    }

    // Test integral calculation over full range
    if (!testUtils::approxEqual(pn.integral(sMin, sMax), 1.0)) {
        std::cerr << "testPDFSegmentNormal: Integral calculation over full range failed: expected 1.0, got " << pn.integral(sMin, sMax) << "\n";
        return 1;
    }
    
    // Test integral calculation over a specified range
    const double expectedIntegral = norm * stdDev /
        (std::numbers::sqrt2 * std::numbers::inv_sqrtpi) *
        (std::erf(dxHiNorm) - std::erf(dxLoNorm));
    if (!testUtils::approxEqual(pn.integral(a, b), expectedIntegral)) {
        std::cerr << "testPDFSegmentNormal: Integral calculation over range [2.0, 5.0] failed: expected " << expectedIntegral << ", got " << pn.integral(a, b) << "\n";
        return 1;
    }

    // Test sampling from the distribution
    const int numSamples = 100000; // Number of samples to draw
    double sampleSum = 0.0; // Sum of the drawn samples for calculating the
    for (int i = 0; i < numSamples; ++i) {
        const double sample = pn.draw(a, b); // Draw a sample from the PDFSegmentNormal over the range [2, 5]
        if (sample < a || sample > b) {
            std::cerr << "testPDFSegmentNormal: Sample drawn from PDFSegmentNormal is out of range: got " << sample << ", expected between " << a << " and " << b << "\n";
            return 1;
        }
        sampleSum += sample; // Add the sample to the sum for calculating the sample mean
    }
    const double sampleMean = sampleSum / numSamples; // Calculate the sample mean
    if (!testUtils::approxEqual(sampleMean, expectedExpectationValueRange, 0.01)) {
        std::cerr << "testPDFSegmentNormal: Sample mean from drawn samples does not match expected expectation value: expected " << expectedExpectationValueRange << ", got " << sampleMean << "\n";
        return 1;
    }

    return 0; // If we get here, the test passed

}
#endif // TESTPDFSEGMENTNORMAL_HPP
