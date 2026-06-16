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
#include <filesystem>
#include <vector>
#include "../src/pdfs/PDF.hpp"
#include "../src/pdfs/PDFFileParser.hpp"
#include "../src/pdfs/PDFSegmentDelta.hpp"
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

    // Create properly normalized PDF and non-normalized PDFs
    // consisting of a delta function plus a lognormal plus a
    // powerlaw with weights of 0.2, 1.4, and 0.4
    double deltaMean = 0.01;
    double lnMin = 0.1;
    double lnMax = 1.0;
    double lnMean = 0.5;
    double lnDisp = 0.2;
    double plMin = lnMax;
    double plMax = 100.0;
    double plAlpha = -2.3;
    pdfs::PDFSegmentDelta pd(deltaMean, rng);
    pdfs::PDFSegmentLognormal pln(lnMin, lnMax, lnMean, lnDisp, rng);
    pdfs::PDFSegmentPowerlaw ppl(plMin, plMax, plAlpha, rng);
    std::vector<pdfs::PDFSegment*> seg = { &pd, &pln, &ppl };
    std::vector<double> wgt = { 0.2, 1.4, 0.4 };
    std::vector<double> wNorm = { 0.1, 0.7, 0.2 };
    pdfs::PDF pdf(seg, wgt, rng, pdfs::samplingMethods::stopNearest, true);
    pdfs::PDF pdfNN(seg, wgt, rng, pdfs::samplingMethods::stopNearest, false);

    // Compute normalizations each segment should have after normalization
    std::array<double, 3> norm = {
        0.0, // Delta function has no meaningful normalization
        wNorm[1] * std::sqrt(2.0 / M_PI) / lnDisp / (
            std::erf(-std::log(lnMin / lnMean) / (std::sqrt(2.0) * lnDisp)) -
            std::erf(-std::log(lnMax / lnMean) / (std::sqrt(2.0) * lnDisp))
        ),
        wNorm[2] * (plAlpha + 1) / (
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
    double expect1 = norm[1] / x1 * 
        std::exp(-0.5 * std::pow(std::log(x1 / lnMean) / lnDisp, 2));
    double expect2 = norm[1] / x2 * 
        std::exp(-0.5 * std::pow(std::log(x2 / lnMean) / lnDisp, 2));
    double expect3 = norm[2] * std::pow(x3, plAlpha);
    double expect4 = norm[2] * std::pow(x4, plAlpha);
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

    // Check that non-normalized PDF is larger by the normalization factor
    if (!testUtils::approxEqual(pdfNN(x4), 2*pdf(x4))) {
        std::cerr << "test_PDF: non-normalized PDF evaluation at x=" << x4 << " failed: expected " << 2*pdf(x4) << ", got " << pdfNN(x4) << std::endl;
        return 1;
    }

    // Check the expectation value over the full range
    double expectation_value_expected = 
        wNorm[0] * pd.expectationValue() +
        wNorm[1] * pln.expectationValue() +
        wNorm[2] * ppl.expectationValue();
    if (!testUtils::approxEqual(pdf.expectationValue(), expectation_value_expected)) {
        std::cerr << "test_PDF: Expectation value calculation over full range failed; expected "
            << expectation_value_expected << ", got "
            << pdf.expectationValue() << std::endl;
            return 1;
    }

    // Check expectation value over part of range
    double a = 0.1;
    double b = 1.0;
    expectation_value_expected = (
            wNorm[0] * pd.integral(a,b) * pd.expectationValue(a,b) +
            wNorm[1] * pln.integral(a,b) * pln.expectationValue(a,b) +
            wNorm[2] * ppl.integral(a,b) * ppl.expectationValue(a,b)
        ) / (
            wNorm[0] * pd.integral(a,b) +
            wNorm[1] * pln.integral(a,b) +
            wNorm[2] * ppl.integral(a,b)
        );
    if (!testUtils::approxEqual(pdf.expectationValue(a,b), expectation_value_expected)) {
        std::cerr << "test_PDF: Expectation value calculation over range ["
            << a << ", " << b << "] failed; expected "
            << expectation_value_expected << ", got "
            << pdf.expectationValue(a,b) << std::endl;
            return 1;
    }

    // Check that normalized and non-normalized versions of PDF have equal
    // expectation value
    if (!testUtils::approxEqual(pdf.expectationValue(a,b), pdfNN.expectationValue(a,b))) {
        std::cerr << "test_PDF: Expectation values of normalized and non-normalized PDFs"
            " differ; normalized = " << pdf.expectationValue(a,b) <<
            " non-normalized = " << pdfNN.expectationValue(a,b) << std::endl;
        return 1;
    }
    
    // Check integral over full range
    if (!testUtils::approxEqual(pdf.integral(), 1.0)) {
        std::cerr << "test_PDF: Expectation value calculation over full range failed; expected "
            << "1.0, got "
            << pdf.integral() << std::endl;
            return 1;
    }

    // Check integral over partial range
    double integral_expected =
        wNorm[1] * pln.integral(a,lnMax) +
        wNorm[2] * ppl.integral(lnMax,b);
    if (!testUtils::approxEqual(pdf.integral(a,b), integral_expected)) {
        std::cerr << "test_PDF: Integral calculation over range ["
            << a << ", " << b << "] failed; expected "
            << integral_expected << ", got "
            << pdf.integral(a,b) << std::endl;
            return 1;
    }

    // Check integral of non-normalized PDF is related to integral of normalized PDF as expected
    if (!testUtils::approxEqual(pdfNN.integral(a,b), 2*pdf.integral(a,b))) {
        std::cerr << "test_PDF: Integral of non-normalized PDF does not match "
            "expected multiple of normalized PDF; non-normalized = " <<
            pdfNN.integral(a,b) << ", expected " <<
            2*pdf.integral(a,b) << std::endl;
        return 1;
    }

    // Test sampling from the distribution
    const int num_samples = 10000; // Number of samples to draw
    double sample_sum = 0.0; // Sum of the drawn samples for calculating the sample mean
    for (int i = 0; i < num_samples; ++i) {
        double sample = pdf.draw(a, b); // Draw a sample from the PDFSegmentExponential over the range [2, 8]
        if (sample < a || sample > b) {
            std::cerr << "test_PDF: Sample drawn from PDFSegmentExponential is out of range: got " << sample << ", expected between " << a << " and " << b << std::endl;
            return 1;
        }
        sample_sum += sample; // Add the sample to the sum for calculating the sample mean
    }
    double sample_mean = sample_sum / num_samples; // Calculate the sample mean
    if (!testUtils::approxEqual(sample_mean, expectation_value_expected, 0.01)) {
        std::cerr << "test_PDF: Sample mean from drawn samples does not match expected expectation value: expected " << expectation_value_expected << ", got " << sample_mean << std::endl;
        return 1;
    }

    // Test Poisson sampling; should produce mean matching expectation value
    double target = 1.0e4;
    pdf.setSampling(pdfs::samplingMethods::poisson);
    auto sample = pdf.drawTarget(target, a, b);
    sample_sum = 0.0;
    for (auto s : sample) sample_sum += s;
    sample_mean = sample_sum / sample.size();
    if (!testUtils::approxEqual(sample_mean, expectation_value_expected, 0.01)) {
        std::cerr << "test_PDF: Sample mean from Poisson sampling does not match expected expectation value: expected " << expectation_value_expected << ", got " << sample_mean << std::endl;
        return 1;        
    }

    // Test construction of a PDF from a valid file in basic mode
    std::filesystem::path assetDir = "assets";
    std::string fileName = "chabrier_imf.txt";
    try
    {
        // Read PDF
        auto pdfBasic = pdfs::parsePDFDescriptor(
            (assetDir / fileName).string(), rng);

        // Construct a PDF by hand that should match the one we
        // just read; parameters given here match those in chabrier_imf.txt
        pdfs::PDFSegmentLognormal plnCompare(0.08, 1, 0.2, 0.55*std::log(10), rng);
        pdfs::PDFSegmentPowerlaw pplCompare(1, 120, -2.35, rng);
        std::vector<pdfs::PDFSegment *> segCompare = { &plnCompare, &pplCompare };
        std::vector<double> wgtCompare = { 1.0, plnCompare(1.0) / pplCompare(1.0) };
        pdfs::PDF pdfCompare(segCompare, wgtCompare, rng);

        // Verify that the two PDFs agree on various quantities
        if (pdfBasic.expectationValue() != pdfCompare.expectationValue())
        {
            std::cerr << "test_PDF: PDF constructed from basic-mode file "
                << fileName << " does not match hand-constructed "
                << "pdf; expectation value for file-based PDF = " 
                << pdfBasic.expectationValue() << ", for hand-constructed = "
                << pdfCompare.expectationValue() << std::endl;
            return 1;
        }
        if (pdfBasic.integral(0.5,20) != pdfCompare.integral(0.5,20))
        {
            std::cerr << "test_PDF: PDF constructed from basic-mode file "
                << fileName << " does not match hand-constructed "
                << "pdf; integral from [0.5,20] for file-based PDF = " 
                << pdfBasic.integral(0.5,20) << ", for hand-constructed = "
                << pdfCompare.expectationValue(0.5,20) << std::endl;
            return 1;
        }        

    }
    catch (const std::exception& error)
    {
        std::cerr << "test_PDF: Failed to parse valid basic-mode PDF:"
        " file 'chabrier_imf.txt'" << std::endl;
        return 1;
    }

    return 0; // If we have gotten here, tests have passed
}

#endif // TEST_PDF_HPP