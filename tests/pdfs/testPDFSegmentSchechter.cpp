/**
 * @file testPDFSegmentSchechter.cpp
 * @author Mark Krumholz
 * @brief Unit tests for the PDFSegmentSchechter class.
 * @date 2024-06-12
 */

#include "../src/pdfs/PDFSegmentSchechter.hpp"
#include "../src/utils/RngThread.hpp"
#include "../tests/testUtils.hpp"
#include "testPDFSegmentSchechter.hpp"
#include <cmath>
#include <gsl/gsl_sf_gamma.h>
#include <iostream>
#include <vector>

static auto
testPDFSegmentSchechterAlpha(const double alpha) -> int     // NOLINT misc-use-anonymous-namespace
{
    const double sMin = 1.0; // The lower limit of the segment
    const double sMax = 10.0; // The upper limit of the segment
    const double sStar = 5.0; // The characteristic scale of the Schechter function
    const pdfs::PDFSegmentSchechter ps(sMin, sMax, sStar, alpha); // Create a PDFSegmentSchechter with sMin=1, sMax=10, sStar=5, alpha=alpha

    // Test PDF evaluation at specific points
    const double x1 = 1.0; // At the lower limit
    const double x2 = 5.0; // Within the range
    const double x3 = 10.0; // At the upper limit
    const double x4 = 0.5; // Below the lower limit
    const double x5 = 15.0; // Above the upper limit
    const double norm = 1.0 / (std::pow(sStar, alpha + 1) * (
        gsl_sf_gamma_inc(alpha + 1, sMin / sStar) -
        gsl_sf_gamma_inc(alpha + 1, sMax / sStar)
    ));
    if (!testUtils::approxEqual(ps(x1), norm  * std::exp(-x1 / sStar))) {
        std::cerr << "testPDFSegmentSchechter: PDF evaluation with alpha=" << alpha << " at x=1.0 failed: expected " << norm << ", got " << ps(x1) << "\n";
        return 1;
    }
    if (!testUtils::approxEqual(ps(x2), norm * std::pow(x2, alpha) * std::exp(-x2 / sStar))) {
        std::cerr << "testPDFSegmentSchechter: PDF evaluation with alpha=" << alpha << " at x=5.0 failed: expected " << norm * std::pow(x2, alpha) << ", got " << ps(x2) << "\n";
        return 1;
    }
    if (!testUtils::approxEqual(ps(x3), norm * std::pow(x3, alpha) * std::exp(-x3 / sStar))) {
        std::cerr << "testPDFSegmentSchechter: PDF evaluation with alpha=" << alpha << " at x=10.0 failed: expected " << norm * std::pow(x3, alpha) << ", got " << ps(x3) << "\n";
        return 1;
    }
    if (!testUtils::approxEqual(ps(x4), 0.0)) {
        std::cerr << "testPDFSegmentSchechter: PDF evaluation with alpha=" << alpha << " at x=0.5 failed: expected 0.0, got " << ps(x4) << "\n";
        return 1;
    }
    if (!testUtils::approxEqual(ps(x5), 0.0)) {
        std::cerr << "testPDFSegmentSchechter: PDF evaluation with alpha=" << alpha << " at x=15.0 failed: expected 0.0, got " << ps(x5) << "\n";
        return 1;
    }

    // Test expectation value calculation over full range
    const double expectedEVFull =
        sStar * (
            gsl_sf_gamma_inc(alpha + 2, sMin / sStar) -
            gsl_sf_gamma_inc(alpha + 2, sMax / sStar)
        ) / (
            gsl_sf_gamma_inc(alpha + 1, sMin / sStar) -
            gsl_sf_gamma_inc(alpha + 1, sMax / sStar)
        );
    if (!testUtils::approxEqual(ps.expectationValue(), expectedEVFull)) {
        std::cerr << "testPDFSegmentSchechter: Expectation value calculation with alpha=" << alpha << " failed: expected " << expectedEVFull << ", got " << ps.expectationValue() << "\n";
        return 1;
    }

    // Test expectation value calculation over a specified range
    const double a = 2.0;
    const double b = 8.0;
    const double expectedEV =
        sStar * (
            gsl_sf_gamma_inc(alpha + 2, a / sStar) -
            gsl_sf_gamma_inc(alpha + 2, b / sStar)
        ) / (
            gsl_sf_gamma_inc(alpha + 1, a / sStar) -
            gsl_sf_gamma_inc(alpha + 1, b / sStar)
        );
    if (!testUtils::approxEqual(ps.expectationValue(a, b), expectedEV)) {
        std::cerr << "testPDFSegmentSchechter: Expectation value calculation with alpha=" << alpha << " failed: expected " << expectedEV << ", got " << ps.expectationValue(a, b) << "\n";
        return 1;
    }

    // Test integral calculation over full range
    if (!testUtils::approxEqual(ps.integral(sMin, sMax), 1.0)) {
        std::cerr << "testPDFSegmentSchechter: Integral calculation with alpha=" << alpha << " failed: expected 1.0, got " << ps.integral(sMin, sMax) << "\n";
        return 1;
    }

    // Test integral calculation over a specified range
    const double expectedIntegral =
        norm * std::pow(sStar, alpha + 1) * (
            gsl_sf_gamma_inc(alpha + 1, a / sStar) -
            gsl_sf_gamma_inc(alpha + 1, b / sStar)
        );
    if (!testUtils::approxEqual(ps.integral(a, b), expectedIntegral)) {
        std::cerr << "testPDFSegmentSchechter: Integral calculation with alpha=" << alpha << " failed: expected " << expectedIntegral << ", got " << ps.integral(a, b) << "\n";
        return 1;
    }

    // Test random sampling
    const int numSamples = 10000;
    double sumSamples = 0.0;
    for (int i = 0; i < numSamples; ++i) {
        const double sample = ps.draw(a, b);
        if (sample < a || sample > b) {
            std::cerr << "testPDFSegmentSchechter: Random sampling with alpha=" << alpha << " failed: sample " << sample << " is out of bounds [" << a << ", " << b << "]" << "\n";
            return 1;
        }
        sumSamples += sample;
    }
    const double sampleMean = sumSamples / numSamples;
    if (!testUtils::approxEqual(sampleMean, expectedEV, 5e-2)) {
        std::cerr << "testPDFSegmentSchechter: Random sampling with alpha=" << alpha << " mean failed: expected " << expectedEV << ", got " << sampleMean << "\n";
        return 1;
    }

    return 0; // Passed
}

auto testPDFSegmentSchechter() -> int
{
    // Set the rng seed to a fixed value for reproducibility
    utils::rng.seed(42);


    // Loop over three different alpha values to make sure we cover the different cases
    // for the expectation value and integral calculations
    const std::vector<double> alphas = {-2.0, -1.0, 0.0}; // The power-law indices to test
    for (const double alpha : alphas) {
        if (testPDFSegmentSchechterAlpha(alpha) == 1) { return 1; }
    }

    return 0; // If we get here, the test passed
}

