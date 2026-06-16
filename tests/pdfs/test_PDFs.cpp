/**
 * @file test_PDFs.cpp
 * @author Mark Krumholz
 * @brief Unit tests for the PDF classes.
 * @details
 * This file contains unit tests for the PDF classes defined in the src/pdfs directory, including PDFSegmentDelta, PDFSegmentExponential, PDFSegmentNormal, PDFSegmentLognormal, PDFSegmentPowerlaw, and PDFSegmentSchechter. The tests cover the evaluation of the PDF at specific points, the calculation of expectation values and integrals over specified ranges, and the sampling of random values from the distributions.
 * @date 2024-06-12
 */

#include "test_PDF.hpp"
#include "test_PDFSegmentDelta.hpp"
#include "test_PDFSegmentExponential.hpp"
#include "test_PDFSegmentNormal.hpp"
#include "test_PDFSegmentLognormal.hpp"
#include "test_PDFSegmentPowerlaw.hpp"
#include "test_PDFSegmentSchechter.hpp"

int main() {
    int result = 0;
    result += test_PDFSegmentDelta();
    result += test_PDFSegmentExponential();
    result += test_PDFSegmentNormal();
    result += test_PDFSegmentLognormal();
    result += test_PDFSegmentPowerlaw();
    result += test_PDFSegmentSchechter();
    result += test_PDF();
    return result;
}
