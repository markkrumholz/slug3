/**
 * @file testPDFSegmentLognormal.hpp
 * @author Mark Krumholz
 * @brief Unit tests for the PDFSegmentLognormal class.
 * @details
 * This file contains unit tests for the PDFSegmentLognormal class defined
 * in the src/pdfs directory.
 * @date 2024-06-12
 */

#ifndef TESTPDFSEGMENTLOGNORMAL_HPP
#define TESTPDFSEGMENTLOGNORMAL_HPP

#include "../src/pdfs/PDFSegmentLognormal.hpp"
#include "../src/utils/MiscUtils.hpp"
#include "../src/utils/RngThread.hpp"
#include <cmath>
#include <iostream>
#include <numbers>

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
inline auto testPDFSegmentLognormal() -> int
{
    // Set the rng seed to a fixed value for reproducibility
    utils::rng.seed(42);

    const double mean = 5.0; // The mean of x (NOT the mean of log(x))
    const double stdDev = 1.0; // The standard deviation of log(x)
    const double sMin = 1.0; // The lower limit of the segment
    const double sMax = 6.0; // The upper limit of the segment
    const pdfs::PDFSegmentLognormal pln(sMin, sMax, mean, stdDev); // Create a PDFSegmentLognormal with mean=5, stdDev=1, sMin=1, sMax=6

    // Pre-compute cached quantities matching those in the constructor
    const double logMean = std::log(mean);
    const double root2dev = std::numbers::sqrt2 * stdDev;
    const double norm = std::numbers::sqrt2 * std::numbers::inv_sqrtpi 
        / stdDev / (
        std::erf(-std::log(sMin / mean) / root2dev) -
        std::erf(-std::log(sMax / mean) / root2dev)
    ); // Normalization constant for the PDFSegmentLognormal

    // Test PDF evaluation at specific points
    const double x1 = 1.0; // At the lower limit
    const double x2 = 5.0; // Within the range, at the mean
    const double x3 = 6.0; // At the upper limit
    const double x4 = 0.5; // Below the lower limit
    const double x5 = 15.0; // Above the upper limit
    if (!utils::approxEqual(pln(x1), norm / x1 * std::exp(-0.5 * std::pow(std::log(x1 / mean) / stdDev, 2)))) {
        std::cerr << "testPDFSegmentLognormal: PDF evaluation at x=1.0 failed: expected "
                  << norm * std::exp(-0.5 * std::pow(std::log(x1 / mean) / stdDev, 2))
                  << ", got " << pln(x1) << "\n";
        return 1;
    }
    if (!utils::approxEqual(pln(x2), norm / x2 * std::exp(-0.5 * std::pow(std::log(x2 / mean) / stdDev, 2)))) {
        std::cerr << "testPDFSegmentLognormal: PDF evaluation at x=5.0 failed: expected "
                  << norm * std::exp(-0.5 * std::pow(std::log(x2 / mean) / stdDev, 2))
                  << ", got " << pln(x2) << "\n";
        return 1;
    }
    if (!utils::approxEqual(pln(x3), norm / x3 * std::exp(-0.5 * std::pow(std::log(x3 / mean) / stdDev, 2)))) {
        std::cerr << "testPDFSegmentLognormal: PDF evaluation at x=6.0 failed: expected "
                  << norm * std::exp(-0.5 * std::pow(std::log(x3 / mean) / stdDev, 2))
                  << ", got " << pln(x3) << "\n";
        return 1;
    }
    if (!utils::approxEqual(pln(x4), 0.0)) {
        std::cerr << "testPDFSegmentLognormal: PDF evaluation at x=0.5 failed: expected 0.0, got " << pln(x4) << "\n";
        return 1;
    }
    if (!utils::approxEqual(pln(x5), 0.0)) {
        std::cerr << "testPDFSegmentLognormal: PDF evaluation at x=15.0 failed: expected 0.0, got " << pln(x5) << "\n";
        return 1;
    }

    // Test expectation value calculation over full range
    const double expectedExpectationValue =
        mean * std::exp(std::pow(stdDev, 2) / 2) *
        (
            std::erf( (logMean - std::log(sMin) + std::pow(stdDev,2)) / root2dev ) -
            std::erf( (logMean - std::log(sMax) + std::pow(stdDev,2)) / root2dev)
        ) / (
            std::erf( (std::log(sMax) - logMean) / root2dev ) -
            std::erf( (std::log(sMin) - logMean) / root2dev )
        );
    if (!utils::approxEqual(pln.expectationValue(), expectedExpectationValue)) {
        std::cerr << "testPDFSegmentLognormal: Expectation value calculation over full range failed: expected "
                  << expectedExpectationValue << ", got " << pln.expectationValue() << "\n";
        return 1;
    }

    // Test expectation value calculation over a specified range
    const double a = 2.0; // Lower limit of the range for expectation value calculation
    const double b = 5.0; // Upper limit of the range for expectation value calculation
    const double logA = std::log(a);  // NOLINT modernize-use-std-numbers
    const double logB = std::log(b);
    const double expectedExpectationValueRange =
        mean * std::exp(std::pow(stdDev, 2) / 2) *
        (
            std::erf( (logMean - logA + std::pow(stdDev,2)) / root2dev ) -
            std::erf( (logMean - logB + std::pow(stdDev,2)) / root2dev)
        ) / (
            std::erf( (logB - logMean) / root2dev ) -
            std::erf( (logA - logMean) / root2dev )
        );
    if (!utils::approxEqual(pln.expectationValue(a, b), expectedExpectationValueRange)) {
        std::cerr << "testPDFSegmentLognormal: Expectation value calculation over range [2.0, 5.0] failed: expected "
                  << expectedExpectationValueRange << ", got " << pln.expectationValue(a, b) << "\n";
        return 1;
    }

    // Test integral calculation over full range
    if (!utils::approxEqual(pln.integral(sMin, sMax), 1.0)) {
        std::cerr << "testPDFSegmentLognormal: Integral calculation over full range failed: expected 1.0, got " << pln.integral(sMin, sMax) << "\n";
        return 1;
    }
    
    // Test integral calculation over a specified range
    const double expectedIntegral = 
        norm / (std::numbers::sqrt2 * std::numbers::inv_sqrtpi) *
            stdDev * (
                std::erf( (logMean - logA) / root2dev ) -
                std::erf( (logMean - logB) / root2dev )
        );
    if (!utils::approxEqual(pln.integral(a, b), expectedIntegral)) {
        std::cerr << "testPDFSegmentLognormal: Integral calculation over range [2.0, 5.0] failed: expected "
                  << expectedIntegral << ", got " << pln.integral(a, b) << "\n";
        return 1;
    }

    // Test sampling from the distribution
    const int numSamples = 100000; // Number of samples to draw
    double sampleSum = 0.0; // Sum of the drawn samples for calculating the sample mean
    for (int i = 0; i < numSamples; ++i) {
        const double sample = pln.draw(a, b); // Draw a sample from the PDFSegmentLognormal over the range [2, 5]
        if (sample < a || sample > b) {
            std::cerr << "testPDFSegmentLognormal: Sample drawn from PDFSegmentLognormal is out of range: got "
                      << sample << ", expected between " << a << " and " << b << "\n";
            return 1;
        }
        sampleSum += sample; // Add the sample to the sum for calculating the sample mean
    }
    const double sampleMean = sampleSum / numSamples; // Calculate the sample mean
    if (!utils::approxEqual(sampleMean, expectedExpectationValueRange, 0.01)) {
        std::cerr << "testPDFSegmentLognormal: Sample mean from drawn samples does not match expected expectation value: expected "
                  << expectedExpectationValueRange << ", got " << sampleMean << "\n";
        return 1;
    }

    return 0; // If we get here, the test passed

}
#endif // TESTPDFSEGMENTLOGNORMAL_HPP
