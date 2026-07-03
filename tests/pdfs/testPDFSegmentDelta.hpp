/**
 * @file testPDFSegmentDelta.hpp
 * @author Mark Krumholz
 * @brief Unit tests for the PDFSegmentDelta class.
 * @details
 * This file contains unit tests for the PDFSegmentDelta class,
 * which represents a delta-function segment of a PDF. The tests
 * cover the evaluation of the PDF at specific points, the
 * calculation of expectation values and integrals over specified
 * ranges, and the sampling of random values from the distribution.
 * @date 2024-06-12
 */

#ifndef TESTPDFSEGMENTDELTA_HPP
#define TESTPDFSEGMENTDELTA_HPP

#include "../src/pdfs/PDFSegmentDelta.hpp"
#include "../src/utils/RngThread.hpp"
#include <iostream>


/**
 * @brief Unit test for the PDFSegmentDelta class.
 * @return 0 if the test passes, 1 if it fails.
 * @details
 * This function tests the PDFSegmentDelta class by calculating
 * expectation values and integrals over specified ranges
 * and sampling random values from the distribution.
 */
inline auto testPDFSegmentDelta() -> int
{
    // Set the rng seed to a fixed value for reproducibility
    utils::rng().seed(42);

    const double x = 5.0; // The value at which the delta function is centered
    const pdfs::PDFSegmentDelta pd(x); // Create a PDFSegmentDelta with sValue=5.0

    // Test expectation value calculation
    if (pd.expectationValue() != x) {
        std::cerr << "test_PDFSegmentDelta: Expectation value calculation failed: expected " << x << ", got " << pd.expectationValue() << "\n";
        return 1;
    }
    if (pd.expectationValue(0.0, 10.0) != x) {
        std::cerr << "test_PDFSegmentDelta: Expectation value calculation with range failed: expected " << x << ", got " << pd.expectationValue(0.0, 10.0) << "\n";
        return 1;
    }

    // Test integral calculation
    if (pd.integral(0.0, 10.0) != 1.0) {
        std::cerr << "test_PDFSegmentDelta: Integral calculation over range that includes center failed: expected 1.0, got " << pd.integral(0.0, 10.0) << "\n";
        return 1;
    }

    // Test integral calculation over a range that does not include the center
    if (pd.integral(0.0, 4.0) != 0.0) {
        std::cerr << "test_PDFSegmentDelta: Integral calculation over range that does not include center failed: expected 0.0, got " << pd.integral(0.0, 4.0) << "\n";
        return 1;
    }

    // Test sampling from the distribution
    for (int i = 0; i < 10; ++i) {
        if (pd.draw(0.0, 10.0) != x) {
            std::cerr << "test_PDFSegmentDelta: Sampling from distribution failed: expected " << x << ", got " << pd.draw(0.0, 10.0) << "\n";
            return 1;
        }
    }

    return 0; // If we get here, the test passed
}

#endif // TESTPDFSEGMENTDELTA_HPP
