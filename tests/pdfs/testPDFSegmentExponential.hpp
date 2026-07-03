/**
 * @file testPDFSegmentExponential.hpp
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

#ifndef TESTPDFSEGMENTEXPONENTIAL_HPP
#define TESTPDFSEGMENTEXPONENTIAL_HPP

#include "../src/pdfs/PDFSegmentExponential.hpp"
#include "../src/utils/MiscUtils.hpp"
#include "../src/utils/RngThread.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

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
inline auto testPDFSegmentExponential() -> int
{
    // Set the rng seed to a fixed value for reproducibility
    utils::rng().seed(42);

    // Create a segment
    const double sMin = 1.0; // The lower limit of the segment
    const double sMax = 10.0; // The upper limit of the segment
    const double scale = 2.0; // The exponential scale length
    const pdfs::PDFSegmentExponential pe(sMin, sMax, scale); // Create a PDFSegmentExponential with sMin=1, sMax=10, scale=2

    // Test PDF evaluation at specific points
    const double x1 = 1.0; // At the lower limit
    const double x2 = 5.0; // Within the range
    const double x3 = 10.0; // At the upper limit
    const double x4 = 0.5; // Below the lower limit
    const double x5 = 15.0; // Above the upper limit
    const double norm = 1.0 / (scale * (std::exp(-sMin / scale) - std::exp(-sMax / scale))); // Normalization constant for the PDF
    if (!utils::approxEqual(pe(x1), norm * std::exp(-x1 / scale))) {
        std::cerr << "testPDFSegmentExponential: PDF evaluation at x=1.0 failed: expected " << norm * std::exp(-x1 / scale) << ", got " << pe(x1) << "\n";
        return 1;
    }
    if (!utils::approxEqual(pe(x2), norm * std::exp(-x2 / scale))) {
        std::cerr << "testPDFSegmentExponential: PDF evaluation at x=5.0 failed: expected " << norm * std::exp(-x2 / scale) << ", got " << pe(x2) << "\n";
        return 1;
    }
    if (!utils::approxEqual(pe(x3), norm * std::exp(-x3 / scale))) {
        std::cerr << "testPDFSegmentExponential: PDF evaluation at x=10.0 failed: expected " << norm * std::exp(-x3 / scale) << ", got " << pe(x3) << "\n";
        return 1;
    }
    if (!utils::approxEqual(pe(x4), 0.0)) {
        std::cerr << "testPDFSegmentExponential: PDF evaluation at x=0.5 failed: expected 0.0, got " << pe(x4) << "\n";
        return 1;
    }
    if (!utils::approxEqual(pe(x5), 0.0)) {
        std::cerr << "testPDFSegmentExponential: PDF evaluation at x=15.0 failed: expected 0.0, got " << pe(x5) << "\n";
        return 1;
    }

    // Test expectation value calculation over full range
    const double numMin = sMin * std::exp(-sMin / scale);
    const double numMax = sMax * std::exp(-sMax / scale);
    const double denomMin = std::exp(-sMin / scale);
    const double denomMax = std::exp(-sMax / scale);
    const double expectedExpectationValue = scale +
        ((numMin - numMax) / (denomMin - denomMax));
    if (!utils::approxEqual(pe.expectationValue(), expectedExpectationValue)) {
        std::cerr << "testPDFSegmentExponential: Expectation value calculation over full range failed: expected " << expectedExpectationValue << ", got " << pe.expectationValue() << "\n";
        return 1;
    }

    // Test expectation value calculation over a specified range
    const double a = 2.0; // Lower limit of the range for expectation value calculation
    const double b = 8.0; // Upper limit of the range for expectation value calculation
    const double numMin1 = std::max(a, sMin) * std::exp(-std::max(a, sMin) / scale);
    const double numMax1 = std::min(b, sMax) * std::exp(-std::min(b, sMax) / scale);
    const double denomMin1 = std::exp(-std::max(a, sMin) / scale);
    const double denomMax1 = std::exp(-std::min(b, sMax) / scale);
    const double expectedExpectationValueRange = scale +
        ((numMin1 - numMax1) / (denomMin1 - denomMax1));
    if (!utils::approxEqual(pe.expectationValue(a, b), expectedExpectationValueRange)) {
        std::cerr << "testPDFSegmentExponential: Expectation value calculation over range [2, 8] failed: expected " << expectedExpectationValueRange << ", got " << pe.expectationValue(a, b) << "\n";
        return 1;
    }

    // Test integral calculation over full range
    if (!utils::approxEqual(pe.integral(sMin, sMax), 1.0)) {
        std::cerr << "testPDFSegmentExponential: Integral calculation over full range failed: expected 1.0, got " << pe.integral(sMin, sMax) << "\n";
        return 1;
    }

    // Test integral calculation over a specified range
    const double expectedIntegral = norm * scale * (std::exp(-std::max(a, sMin) / scale) - std::exp(-std::min(b, sMax) / scale));
    if (!utils::approxEqual(pe.integral(a, b), expectedIntegral)) {
        std::cerr << "testPDFSegmentExponential: Integral calculation over range [2, 8] failed: expected " << expectedIntegral << ", got " << pe.integral(a, b) << "\n";
        return 1;
    }

    // Test sampling from the distribution
    const int numSamples = 100000; // Number of samples to draw
    double sampleSum = 0.0; // Sum of the drawn samples for calculating the sample mean
    for (int i = 0; i < numSamples; ++i) {
        const double sample = pe.draw(a, b); // Draw a sample from the PDFSegmentExponential over the range [2, 8]
        if (sample < a || sample > b) {
            std::cerr << "testPDFSegmentExponential: Sample drawn from PDFSegmentExponential is out of range: got " << sample << ", expected between " << a << " and " << b << "\n";
            return 1;
        }
        sampleSum += sample; // Add the sample to the sum for calculating the sample mean
    }
    const double sampleMean = sampleSum / numSamples; // Calculate the sample mean
    if (!utils::approxEqual(sampleMean, expectedExpectationValueRange, 0.01)) {
        std::cerr << "testPDFSegmentExponential: Sample mean from drawn samples does not match expected expectation value: expected " << expectedExpectationValueRange << ", got " << sampleMean << "\n";
        return 1;
    }

    return 0; // If we get here, the test passed
}

#endif // TESTPDFSEGMENTEXPONENTIAL_HPP
