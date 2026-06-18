/**
 * @file testPDFSegmentPowerlaw.hpp
 * @author Mark Krumholz
 * @brief Unit tests for the PDFSegmentPowerlaw class.
 * @details
 * This file contains unit tests for the PDFSegmentPowerlaw class,
 * which represents a power-law segment of a PDF. The tests
 * cover the evaluation of the PDF at specific points, the
 * calculation of expectation values and integrals over specified
 * ranges, and the sampling of random values from the distribution.
 * @date 2024-06-12
 */

#ifndef TESTPDFSEGMENTPOWERLAW_HPP
#define TESTPDFSEGMENTPOWERLAW_HPP

/**
 * @brief Unit test for the PDFSegmentPowerlaw class.
 * @return 0 if the test passes, 1 if it fails.
 * @details
 * This function tests the PDFSegmentPowerlaw class by evaluating the PDF at
 * specific points, calculating expectation values and integrals over
 * specified ranges, and sampling random values from the distribution.
 * It checks that the PDF evaluates to the expected values at the lower
 * and upper limits, as well as within the range.
 */
auto testPDFSegmentPowerlaw() -> int;

#endif // TESTPDFSEGMENTPOWERLAW_HPP
