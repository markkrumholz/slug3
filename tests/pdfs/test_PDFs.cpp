/**
 * @file test_PDFs.cpp
 * @author Mark Krumholz
 * @brief Unit tests for the PDF classes.
 * @details
 * This file contains unit tests for the PDF classes defined in the src/pdfs directory, including PDFDelta, PDFExponential, PDFNormal, PDFPowerlaw, and PDFSchechter. The tests cover the evaluation of the PDF at specific points, the calculation of expectation values and integrals over specified ranges, and the sampling of random values from the distributions.
 * @date 2024-06-12
 */

#include "test_PDFDelta.hpp"
#include "test_PDFPowerlaw.hpp"

int main() {
    int result = 0;
    result += test_PDFDelta();
    result += test_PDFPowerlaw();
    return result;
}