/**
 * @file testPDFSegmentSchechter.hpp
 * @author Mark Krumholz
 * @brief Unit tests for the PDFSegmentSchechter class.
 * @details
 * This file contains unit tests for the PDFSegmentSchechter class defined
 * in the src/pdfs directory.
 * @date 2024-06-12
 */

#ifndef TESTPDFSEGMENTSCHECHTER_HPP
#define TESTPDFSEGMENTSCHECHTER_HPP

/**
 * @brief Unit test for the PDFSegmentSchechter class.
 * @return 0 if the test passes, 1 if it fails.
 * @details
 * This function tests the PDFSegmentSchechter class by evaluating the PDF at
 * specific points, calculating expectation values and integrals over
 * specified ranges, and sampling random values from the distribution.
 * It checks that the PDF evaluates to the expected values at the mean
 * and at points one standard deviation away from the mean, as well as
 * outside the range of the distribution.
 */
auto testPDFSegmentSchechter() -> int;


#endif // TESTPDFSEGMENTSCHECHTER_HPP
