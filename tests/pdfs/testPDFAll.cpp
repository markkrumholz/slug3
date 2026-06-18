/**
 * @file testPDFAll.cpp
 * @author Mark Krumholz
 * @brief Unit tests for the PDF classes.
 * @details
 * This file contains unit tests for the PDF classes defined in the src/pdfs directory, including PDFSegmentDelta, PDFSegmentExponential, PDFSegmentNormal, PDFSegmentLognormal, PDFSegmentPowerlaw, and PDFSegmentSchechter. The tests cover the evaluation of the PDF at specific points, the calculation of expectation values and integrals over specified ranges, and the sampling of random values from the distributions.
 * @date 2024-06-12
 */

#include "testPDF.hpp"
#include "testPDFSegmentDelta.hpp"
#include "testPDFSegmentExponential.hpp"
#include "testPDFSegmentLognormal.hpp"
#include "testPDFSegmentNormal.hpp"
#include "testPDFSegmentPowerlaw.hpp"
#include "testPDFSegmentSchechter.hpp"

auto main() -> int {
    int result = 0;
    result += testPDFSegmentDelta();
    result += testPDFSegmentExponential();
    result += testPDFSegmentNormal();
    result += testPDFSegmentLognormal();
    result += testPDFSegmentPowerlaw();
    result += testPDFSegmentSchechter();
    result += testPDF();
    return result;
}
