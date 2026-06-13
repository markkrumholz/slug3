/**
 * @file test_PDF.hpp
 * @author Mark Krumholz
 * @brief Unit tests for the PDF class.
 * @details
 * This file contains unit tests for the PDF class,
 * which represents a PDF made up of multiple segments.
 * @date 2024-06-12
 */

#ifndef TEST_PDF_HPP
#define TEST_PDF_HPP

#include <cmath>
#include <cstdio>
#include <valarray>
#include "../src/pdfs/PDF.hpp"
#include "../src/pdfs/PDFSegmentLognormal.hpp"
#include "../src/pdfs/PDFSegmentPowerlaw.hpp"
#include "../tests/testUtils.hpp"

 /** 
  * @brief Unit test for the PDF class.
  * @return 0 if the test passes, 1 if it fails.
  * @details
  * This function tests the PDF class.
  */
auto test_PDF() -> int
{
    rngType rng(42); // Create a random number generator with a fixed seed for reproducibility

    // Create a properly normalized PDF consisting of a
    // lognormal plus a powerlaw with weights of 1.6 and 0.4
    double lnMin = 0.1;
    double lnMax = 1.0;
    double lnMean = 0.5;
    double lnDisp = 0.2;
    double plMin = lnMax;
    double plMax = 100.0;
    double plAlpha = -2.3;
    pdfs::PDFSegmentLognormal pln(lnMin, lnMax, lnMean, lnDisp, rng);
    pdfs::PDFSegmentPowerlaw ppl(plMin, plMax, plAlpha, rng);
    std::array<pdfs::PDFSegment*, 2> seg = { &pln, &ppl };
    std::array<double, 2> wgt = { 1.6, 0.4 };
    std::array<double, 2> wNorm = { 0.8, 0.2 };
    bool normalized = true;
    pdfs::PDF<2> pdf(seg, wgt, pdfs::samplingMethods::stopNearest, 
        normalized);

    // Compute normalizations each segment should have
    std::array<double, 2> norm = {
        wNorm[0] * std::sqrt(2.0 / M_PI) / lnDisp / (
            std::erf(-std::log(lnMin / lnMean) / (std::sqrt(2.0) * lnDisp)) -
            std::erf(-std::log(lnMax / lnMean) / (std::sqrt(2.0) * lnDisp))
        ),
        wNorm[1] * (plAlpha + 1) / (
            std::pow(plMax, plAlpha + 1) - 
            std::pow(plMin, plAlpha + 1)
        )
    };

    // Test PDF evaluation at specific points
    double x1 = 0.1; // At the lower limit
    double x2 = 0.5; // In the first segment
    double x3 = 50.0; // In the second segment
    double x4 = 100.0; // At the upper limit
    double x5 = 0.05; // Below the lower limit
    double x6 = 150.0; // Above the upper limit
    double expect1 = norm[0] / x1 * 
        std::exp(-0.5 * std::pow(std::log(x1 / lnMean) / lnDisp, 2));
    double expect2 = norm[0] / x2 * 
        std::exp(-0.5 * std::pow(std::log(x2 / lnMean) / lnDisp, 2));
    double expect3 = norm[1] * std::pow(x3, plAlpha);
    double expect4 = norm[1] * std::pow(x4, plAlpha);
    if (!testUtils::approxEqual(pdf(x1), expect1)) {
        std::cerr << "test_PDF: PDF evaluation at x=" << x1 << " failed: expected " << expect1 << ", got " << pdf(x1) << std::endl;
            return 1;
    }
    if (!testUtils::approxEqual(pdf(x2), expect2)) {
        std::cerr << "test_PDF: PDF evaluation at x=" << x2 << " failed: expected " << expect2 << ", got " << pdf(x2) << std::endl;
            return 1;
    }
    if (!testUtils::approxEqual(pdf(x3), expect3)) {
        std::cerr << "test_PDF: PDF evaluation at x=" << x3 << " failed: expected " << expect3 << ", got " << pdf(x3) << std::endl;
            return 1;
    }
    if (!testUtils::approxEqual(pdf(x4), expect4)) {
        std::cerr << "test_PDF: PDF evaluation at x=" << x4 << " failed: expected " << expect4 << ", got " << pdf(x4) << std::endl;
            return 1;
    }
    if (pdf(x5) != 0.0) {
        std::cerr << "test_PDF: PDF evaluation at x=" << x5 << " failed: expected " << 0.0 << ", got " << pdf(x5) << std::endl;
            return 1;
    }
    if (pdf(x6) != 0.0) {
        std::cerr << "test_PDF: PDF evaluation at x=" << x6 << " failed: expected " << 0.0 << ", got " << pdf(x5) << std::endl;
            return 1;
    }

    return 0; // If we have gotten here, tests have passed
}

#endif // TEST_PDF_HPP