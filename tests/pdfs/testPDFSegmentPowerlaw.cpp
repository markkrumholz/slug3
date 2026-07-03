/**
 * @file testPDFSegmentPowerlaw.cpp
 * @author Mark Krumholz
 * @brief Unit tests for the PDFSegmentPowerlaw class.
 * @date 2024-06-12
 */

#include "../src/pdfs/PDFSegmentPowerlaw.hpp"
#include "../src/utils/MiscUtils.hpp"
#include "../src/utils/RngThread.hpp"
#include "testPDFSegmentPowerlaw.hpp"
#include <cmath>
#include <iostream>
#include <numbers>
#include <vector>

static auto
testPDFSegmentPowerlawAlpha(const double alpha) -> int     // NOLINT misc-use-anonymous-namespace
{
    const double sMin = 1.0; // The lower limit of the segment
    const double sMax = 10.0; // The upper limit of the segment
    const pdfs::PDFSegmentPowerlaw pl(sMin, sMax, alpha); // Create a PDFSegmentPowerlaw with sMin=1, sMax=10, alpha=alpha

    // Test PDF evaluation at specific points
    const double x1 = 1.0; // At the lower limit
    const double x2 = 5.0; // Within the range
    const double x3 = 10.0; // At the upper limit
    const double x4 = 0.5; // Below the lower limit
    const double x5 = 15.0; // Above the upper limit
    double norm = NAN;
    if (alpha != -1) {
        norm = (alpha + 1) / (std::pow(10.0, alpha + 1) - std::pow(1.0, alpha + 1));
    } else {
        norm = 1.0 / std::numbers::ln10;
    }
    if (!utils::approxEqual(pl(x1), norm)) {
        std::cerr << "testPDFSegmentPowerlaw: PDF evaluation with alpha=" << alpha << " at x=1.0 failed: expected " << norm << ", got " << pl(x1) << "\n";
        return 1;
    }
    if (!utils::approxEqual(pl(x2), norm * std::pow(x2, alpha))) {
        std::cerr << "testPDFSegmentPowerlaw: PDF evaluation with alpha=" << alpha << " at x=5.0 failed: expected " << norm * std::pow(x2, alpha) << ", got " << pl(x2) << "\n";
        return 1;
    }
    if (!utils::approxEqual(pl(x3), norm * std::pow(x3, alpha))) {
        std::cerr << "testPDFSegmentPowerlaw: PDF evaluation with alpha=" << alpha << " at x=10.0 failed: expected " << norm * std::pow(x3, alpha) << ", got " << pl(x3) << "\n";
        return 1;
    }
    if (!utils::approxEqual(pl(x4), 0.0)) {
        std::cerr << "testPDFSegmentPowerlaw: PDF evaluation with alpha=" << alpha << " at x=0.5 failed: expected 0.0, got " << pl(x4) << "\n";
        return 1;
    }
    if (!utils::approxEqual(pl(x5), 0.0)) {
        std::cerr << "testPDFSegmentPowerlaw: PDF evaluation with alpha=" << alpha << " at x=15.0 failed: expected 0.0, got " << pl(x5) << "\n";
        return 1;
    }

    // Test expectation value calculation over full range
    double expectedEVFull = NAN;
    if (alpha == -2.0) {
        expectedEVFull = 10.0/9.0 * std::numbers::ln10; // Expected expectation value for alpha=-2 over [1, 10]
    } else if (alpha == -1.0) {
        expectedEVFull = 9.0 / std::numbers::ln10; // Expected expectation value for alpha=-1 over [1, 10]
    } else {
        expectedEVFull = (alpha + 1) / (alpha + 2) * (std::pow(10.0, alpha + 2) - std::pow(1.0, alpha + 2)) / (std::pow(10.0, alpha + 1) - std::pow(1.0, alpha + 1)); // Expected expectation value for alpha=0 over [1, 10]
    }
    if (!utils::approxEqual(pl.expectationValue(), expectedEVFull)) {
        std::cerr << "testPDFSegmentPowerlaw: Expectation value calculation with alpha=" << alpha << " failed: expected " << expectedEVFull << ", got " << pl.expectationValue() << "\n";
        return 1;
    }

    // Test expectation value calculation over a specified range
    const double a = 2.0;
    const double b = 8.0;
    double expectedEV = NAN;
    if (alpha == -2.0) {
        expectedEV = (std::log(b/a) / ((1.0/a) - (1.0/b))); // Expected expectation value for alpha=-2 over [2, 8]
    } else if (alpha == -1.0) {
        expectedEV = (b - a) / std::log(b/a); // Expected expectation value for alpha=-1 over [2, 8]
    } else {
        expectedEV = (alpha + 1) / (alpha + 2) * (std::pow(b, alpha + 2) - std::pow(a, alpha + 2)) / (std::pow(b, alpha + 1) - std::pow(a, alpha + 1)); // Expected expectation value for alpha=0 over [2, 8]
    }
    if (!utils::approxEqual(pl.expectationValue(a, b), expectedEV)) {
        std::cerr << "testPDFSegmentPowerlaw: Expectation value calculation with alpha=" << alpha << " failed: expected " << expectedEV << ", got " << pl.expectationValue(a, b) << "\n";
        return 1;
    }

    // Test integral calculation over full range
    if (!utils::approxEqual(pl.integral(sMin, sMax), 1.0)) {
        std::cerr << "testPDFSegmentPowerlaw: Integral calculation with alpha=" << alpha << " failed: expected 1.0, got " << pl.integral(a, b) << "\n";
        return 1;
    }

    // Test integral calculation over a specified range
    double expectedIntegral = NAN;
    if (alpha == -2.0) {
        expectedIntegral = norm * ((1.0/a) - (1.0/b)); // Expected integral for alpha=-2 over [2, 8]
    }  else if (alpha == -1.0) {
        expectedIntegral = norm * std::log(b/a); // Expected integral for alpha=-1 over [2, 8]
    } else {
        expectedIntegral = norm / (alpha + 1) * (std::pow(b, alpha + 1) - std::pow(a, alpha + 1)); // Expected integral for alpha=0 over [2, 8]
    }
    if (!utils::approxEqual(pl.integral(a, b), expectedIntegral)) {
        std::cerr << "testPDFSegmentPowerlaw: Integral calculation with alpha=" << alpha << " failed: expected " << expectedIntegral << ", got " << pl.integral(a, b) << "\n";
        return 1;
    }

    // Test random sampling
    const int numSamples = 100000;
    double sumSamples = 0.0;
    for (int i = 0; i < numSamples; ++i) {
        const double sample = pl.draw(a, b);
        if (sample < a || sample > b) {
            std::cerr << "testPDFSegmentPowerlaw: Random sampling with alpha=" << alpha << " failed: sample " << sample << " is out of bounds [" << a << ", " << b << "]" << "\n";
            return 1;
        }
        sumSamples += sample;
    }
    const double sampleMean = sumSamples / numSamples;
    if (!utils::approxEqual(sampleMean, expectedEV, 2e-2)) {
        std::cerr << "testPDFSegmentPowerlaw: Random sampling with alpha=" << alpha << " mean failed: expected " << expectedEV << ", got " << sampleMean << "\n";
        return 1;
    }

    return 0; // Passed
}

auto testPDFSegmentPowerlaw() -> int
{
    // Set the rng seed to a fixed value for reproducibility
    utils::rng().seed(42);

    // Loop over three different alpha values to make sure we cover the different cases
    // for the expectation value and integral calculations
    const std::vector<double> alphas = {-2.0, -1.0, 0.0};
    for (const double alpha : alphas) {
        if (testPDFSegmentPowerlawAlpha(alpha) == 1) { return 1; }
    }

    // If we reach this point, all tests passed
    return 0;
}
