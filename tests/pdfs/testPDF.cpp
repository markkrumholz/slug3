/**
 * @file testPDF.cpp
 * @author Mark Krumholz
 * @brief Unit tests for the PDF class.
 * @date 2024-06-12
 */

#include "../src/pdfs/PDF.hpp"
#include "../src/pdfs/PDFCommons.hpp"
#include "../src/pdfs/PDFFileParser.hpp"
#include "../src/pdfs/PDFSegment.hpp"
#include "../src/pdfs/PDFSegmentDelta.hpp"
#include "../src/pdfs/PDFSegmentLognormal.hpp"
#include "../src/pdfs/PDFSegmentPowerlaw.hpp"
#include "../src/utils/MiscUtils.hpp"
#include "../src/utils/RngThread.hpp"
#include "testPDF.hpp"
#include <array>
#include <cmath>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <numbers>
#include <string>
#include <utility>
#include <vector>

static auto testPDFParsing() -> int    // NOLINT misc-use-anonymous-namespace
{
    // Test construction of a PDF from a valid file in basic mode
    const std::filesystem::path assetDir = "assets";
    std::string fileName = "chabrier_imf.txt";
    try
    {
        // Read PDF
        auto pdfBasic = pdfs::parsePDFDescriptor(
            (assetDir / fileName).string());

        // Construct a PDF by hand that should match the one we
        // just read; parameters given here match those in chabrier_imf.txt
        auto plnCmp = std::make_unique<pdfs::PDFSegmentLognormal>(0.08, 1, 0.2, 0.55*std::numbers::ln10);
        auto pplCmp = std::make_unique<pdfs::PDFSegmentPowerlaw>(1, 120, -2.35);
        const std::vector<double> wgtCompare = { 1.0, (*plnCmp)(1.0) / (*pplCmp)(1.0) };
        std::vector<std::unique_ptr<pdfs::PDFSegment>> segCompare;
        segCompare.push_back(std::move(plnCmp));
        segCompare.push_back(std::move(pplCmp));
        const pdfs::PDF pdfCompare(std::move(segCompare), wgtCompare);

        // Verify that the two PDFs agree on various quantities
        if (pdfBasic.expectationValue() != pdfCompare.expectationValue())
        {
            std::cerr << "testPDF: PDF constructed from basic-mode file "
                << fileName << " does not match hand-constructed "
                << "pdf; expectation value for file-based PDF = " 
                << pdfBasic.expectationValue() << ", for hand-constructed = "
                << pdfCompare.expectationValue() << "\n";
            return 1;
        }
        if (pdfBasic.integral(0.5,20) != pdfCompare.integral(0.5,20))
        {
            std::cerr << "testPDF: PDF constructed from basic-mode file "
                << fileName << " does not match hand-constructed "
                << "pdf; integral from [0.5,20] for file-based PDF = " 
                << pdfBasic.integral(0.5,20) << ", for hand-constructed = "
                << pdfCompare.expectationValue(0.5,20) << "\n";
            return 1;
        }        

    }
    catch (const std::exception& error)
    {
        std::cerr << "testPDF: Failed to parse valid basic-mode PDF:"
        " file 'chabrier_imf.txt'" << "\n";
        return 1;
    }

    // Test construction of a PDF using a method setting
    fileName = "wk06.txt";
    try
    {
        // Read PDF
        auto pdfWK = pdfs::parsePDFDescriptor(
            (assetDir / fileName).string());

        // Verify that sampling mode is set to sorted
        if (pdfWK.getSampling() != pdfs::SamplingMethods::sorted)
        {
            std::cerr << "testPDF: failed to correctly set "
                "sampling method from basic-mode PDF descriptor "
                "file 'wk06.txt'" << "\n";
            return 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testPDF: Failed to parse valid basic-mode PDF:"
        " file 'wk06.txt'" << "\n";
        return 1;
    }
    // Test construction of a PDF from an advanced format file
    fileName = "sfh_burst.txt";
    try
    {
        // Read PDF
        auto pdfAdv = pdfs::parsePDFDescriptor(
            (assetDir / fileName).string());

        // Verify that integral is correct
        if (pdfAdv.integral() != 1e3)
        {
            std::cerr << "testPDF: advanced format PDF construction "
                "failed; expected integral = 1e3, found "
                << pdfAdv.integral() << "\n";
            return 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "testPDF: Failed to parse valid advanced-mode PDF:"
        " file 'sfh_burst.txt'" << "\n";
        return 1;
    }

    return 0; // Passed
}

static auto testPDFSampling(    // NOLINT misc-use-anonymous-namespace
    const double a, 
    const double b, 
    const double expectationValueExpected,
    pdfs::PDF& pdf) -> int
{
    const int numSamples = 10000; // Number of samples to draw
    double sampleSum = 0.0; // Sum of the drawn samples for calculating the sample mean
    for (int i = 0; i < numSamples; ++i) {
        const double sample = pdf.draw(a, b); // Draw a sample from the PDFSegmentExponential over the range [2, 8]
        if (sample < a || sample > b) {
            std::cerr << "testPDF: Sample drawn from PDFSegmentExponential is out of range: got " << sample << ", expected between " << a << " and " << b << "\n";
            return 1;
        }
        sampleSum += sample; // Add the sample to the sum for calculating the sample mean
    }
    double sampleMean = sampleSum / static_cast<double>(numSamples); // Calculate the sample mean
    if (!utils::approxEqual(sampleMean, expectationValueExpected, 0.01)) {
        std::cerr << "testPDF: Sample mean from drawn samples does not match expected expectation value: expected " << expectationValueExpected << ", got " << sampleMean << "\n";
        return 1;
    }

    // Test Poisson sampling; should produce mean matching expectation value
    double target = 1.0e4;
    pdf.setSampling(pdfs::SamplingMethods::poisson);
    auto sample = pdf.drawTarget(target, a, b);
    sampleSum = 0.0;
    for (auto s : sample) { sampleSum += s; }
    sampleMean = sampleSum / static_cast<double>(sample.size());
    if (!utils::approxEqual(sampleMean, expectationValueExpected, 0.01)) {
        std::cerr << "testPDF: Sample mean from Poisson sampling does not match expected expectation value: expected " << expectationValueExpected << ", got " << sampleMean << "\n";
        return 1;        
    }

    // Test stop-before, stop-after, and stop-50 sampling; stop-before
    // should always produce sums < target, stop-after > target, and
    // stop-50 should be smaller and larger than target equally often
    target = 100.0;
    const int nTest = 1000;
    int countLess = 0;
    for (int i = 0; i < nTest; i++)
    {
        // Stop before
        pdf.setSampling(pdfs::SamplingMethods::stopBefore);
        auto sampleBefore = pdf.drawTarget(target);
        sampleSum = 0.0;
        for (auto s : sampleBefore) { sampleSum += s; }
        if (sampleSum > target)
        {
            std::cerr << "testPDF: stopBefore sampling test failed: "
                << "target = " << target << ", got sum = " << sampleSum
                << "\n";
            return 1;
        }

        // Stop after
        pdf.setSampling(pdfs::SamplingMethods::stopAfter);
        auto sampleAfter = pdf.drawTarget(target);
        sampleSum = 0.0;
        for (auto s : sampleAfter) { sampleSum += s; }
        if (sampleSum < target)
        {
            std::cerr << "testPDF: stopAfter sampling test failed: "
                << "target = " << target << ", got sum = " << sampleSum
                << "\n";
            return 1;
        }

        // Stop 50
        pdf.setSampling(pdfs::SamplingMethods::stop50);
        auto sample50 = pdf.drawTarget(target);
        sampleSum = 0.0;
        for (auto s : sample50) { sampleSum += s; }
        if (sampleSum < target) { countLess++; }
    }
    const double fracLess = static_cast<double>(countLess) / nTest;
    const double tol = 4 / std::sqrt(nTest);  // 4 sigma tolerance
    if (!utils::approxEqual(fracLess, 0.5, tol))
    {
        std::cerr << "testPDF: stop50 sampling test failed: "
            << "expected 50% below target, got "
            << 100 * fracLess << "\n";
        return 1;
    }

    return 0; // Passed
}


auto testPDF() -> int
{
   // Set the rng seed to a fixed value for reproducibility
    utils::rng().seed(42);

    // Create properly normalized PDF and non-normalized PDFs
    // consisting of a delta function plus a lognormal plus a
    // powerlaw with weights of 0.2, 1.4, and 0.4
    const double deltaMean = 0.01;
    const double lnMin = 0.1;
    const double lnMax = 1.0;
    const double lnMean = 0.5;
    const double lnDisp = 0.2;
    const double plMin = lnMax;
    const double plMax = 100.0;
    const double plAlpha = -2.3;
    auto pd1 = std::make_unique<pdfs::PDFSegmentDelta>(deltaMean);
    auto pln1 = std::make_unique<pdfs::PDFSegmentLognormal>(lnMin, lnMax, lnMean, lnDisp);
    auto ppl1 = std::make_unique<pdfs::PDFSegmentPowerlaw>(plMin, plMax, plAlpha);
    auto& pd = *pd1;
    auto& pln = *pln1;
    auto& ppl = *ppl1;
    std::vector<std::unique_ptr<pdfs::PDFSegment>> seg1;
    seg1.push_back(std::move(pd1));
    seg1.push_back(std::move(pln1));
    seg1.push_back(std::move(ppl1));
    std::vector<std::unique_ptr<pdfs::PDFSegment>> seg2;
    seg2.push_back(std::make_unique<pdfs::PDFSegmentDelta>(deltaMean));
    seg2.push_back(std::make_unique<pdfs::PDFSegmentLognormal>(lnMin, lnMax, lnMean, lnDisp));
    seg2.push_back(std::make_unique<pdfs::PDFSegmentPowerlaw>(plMin, plMax, plAlpha));
    const std::vector<double> wgt = { 0.2, 1.4, 0.4 };
    const std::vector<double> wNorm = { 0.1, 0.7, 0.2 };
    pdfs::PDF pdf(std::move(seg1), wgt, pdfs::SamplingMethods::stopNearest, true);
    const pdfs::PDF pdfNN(std::move(seg2), wgt, pdfs::SamplingMethods::stopNearest, false);

    // Compute normalizations each segment should have after normalization
    std::array<double, 3> norm = {
        0.0, // Delta function has no meaningful normalization
        wNorm.at(1) * std::numbers::sqrt2 * std::numbers::inv_sqrtpi / lnDisp / (
            std::erf(-std::log(lnMin / lnMean) / (std::numbers::sqrt2 * lnDisp)) -
            std::erf(-std::log(lnMax / lnMean) / (std::numbers::sqrt2 * lnDisp))
        ),
        wNorm.at(2) * (plAlpha + 1) / (
            std::pow(plMax, plAlpha + 1) - 
            std::pow(plMin, plAlpha + 1)
        )
    };

    // Test PDF evaluation at specific points
    const double x1 = 0.1; // At the lower limit
    const double x2 = 0.5; // In the first segment
    const double x3 = 50.0; // In the second segment
    const double x4 = 100.0; // At the upper limit
    const double x5 = 0.05; // Below the lower limit
    const double x6 = 150.0; // Above the upper limit
    const double expect1 = norm.at(1) / x1 * 
        std::exp(-0.5 * std::pow(std::log(x1 / lnMean) / lnDisp, 2));
    const double expect2 = norm.at(1) / x2 * 
        std::exp(-0.5 * std::pow(std::log(x2 / lnMean) / lnDisp, 2));
    const double expect3 = norm.at(2) * std::pow(x3, plAlpha);
    const double expect4 = norm.at(2) * std::pow(x4, plAlpha);
    if (!utils::approxEqual(pdf(x1), expect1)) {
        std::cerr << "testPDF: PDF evaluation at x=" << x1 << " failed: expected " << expect1 << ", got " << pdf(x1) << "\n";
            return 1;
    }
    if (!utils::approxEqual(pdf(x2), expect2)) {
        std::cerr << "testPDF: PDF evaluation at x=" << x2 << " failed: expected " << expect2 << ", got " << pdf(x2) << "\n";
            return 1;
    }
    if (!utils::approxEqual(pdf(x3), expect3)) {
        std::cerr << "testPDF: PDF evaluation at x=" << x3 << " failed: expected " << expect3 << ", got " << pdf(x3) << "\n";
            return 1;
    }
    if (!utils::approxEqual(pdf(x4), expect4)) {
        std::cerr << "testPDF: PDF evaluation at x=" << x4 << " failed: expected " << expect4 << ", got " << pdf(x4) << "\n";
            return 1;
    }
    if (pdf(x5) != 0.0) {
        std::cerr << "testPDF: PDF evaluation at x=" << x5 << " failed: expected " << 0.0 << ", got " << pdf(x5) << "\n";
            return 1;
    }
    if (pdf(x6) != 0.0) {
        std::cerr << "testPDF: PDF evaluation at x=" << x6 << " failed: expected " << 0.0 << ", got " << pdf(x5) << "\n";
            return 1;
    }

    // Check that non-normalized PDF is larger by the normalization factor
    if (!utils::approxEqual(pdfNN(x4), 2*pdf(x4))) {
        std::cerr << "testPDF: non-normalized PDF evaluation at x=" << x4 << " failed: expected " << 2*pdf(x4) << ", got " << pdfNN(x4) << "\n";
        return 1;
    }

    // Check the expectation value over the full range
    double expectationValueExpected = 
        (wNorm.at(0) * pd.expectationValue()) +
        (wNorm.at(1) * pln.expectationValue()) +
        (wNorm.at(2) * ppl.expectationValue());
    if (!utils::approxEqual(pdf.expectationValue(), expectationValueExpected)) {
        std::cerr << "testPDF: Expectation value calculation over full range failed; expected "
            << expectationValueExpected << ", got "
            << pdf.expectationValue() << "\n";
            return 1;
    }

    // Check expectation value over part of range
    const double a = 0.1;
    const double b = 1.0;
    expectationValueExpected = (
            (wNorm.at(0) * pd.integral(a,b) * pd.expectationValue(a,b)) +
            (wNorm.at(1) * pln.integral(a,b) * pln.expectationValue(a,b)) +
            (wNorm.at(2) * ppl.integral(a,b) * ppl.expectationValue(a,b))
        ) / (
            (wNorm.at(0) * pd.integral(a,b)) +
            (wNorm.at(1) * pln.integral(a,b)) +
            (wNorm.at(2) * ppl.integral(a,b))
        );
    if (!utils::approxEqual(pdf.expectationValue(a,b), expectationValueExpected)) {
        std::cerr << "testPDF: Expectation value calculation over range ["
            << a << ", " << b << "] failed; expected "
            << expectationValueExpected << ", got "
            << pdf.expectationValue(a,b) << "\n";
            return 1;
    }

    // Check that normalized and non-normalized versions of PDF have equal
    // expectation value
    if (!utils::approxEqual(pdf.expectationValue(a,b), pdfNN.expectationValue(a,b))) {
        std::cerr << "testPDF: Expectation values of normalized and non-normalized PDFs"
            " differ; normalized = " << pdf.expectationValue(a,b) <<
            " non-normalized = " << pdfNN.expectationValue(a,b) << "\n";
        return 1;
    }
    
    // Check integral over full range
    if (!utils::approxEqual(pdf.integral(), 1.0)) {
        std::cerr << "testPDF: Expectation value calculation over full range failed; expected "
            << "1.0, got "
            << pdf.integral() << "\n";
            return 1;
    }

    // Check integral over partial range
    const double integralExpected =
        (wNorm.at(1) * pln.integral(a,lnMax)) +
        (wNorm.at(2) * ppl.integral(lnMax,b));
    if (!utils::approxEqual(pdf.integral(a,b), integralExpected)) {
        std::cerr << "testPDF: Integral calculation over range ["
            << a << ", " << b << "] failed; expected "
            << integralExpected << ", got "
            << pdf.integral(a,b) << "\n";
            return 1;
    }

    // Check integral of non-normalized PDF is related to integral of normalized PDF as expected
    if (!utils::approxEqual(pdfNN.integral(a,b), 2*pdf.integral(a,b))) {
        std::cerr << "testPDF: Integral of non-normalized PDF does not match "
            "expected multiple of normalized PDF; non-normalized = " <<
            pdfNN.integral(a,b) << ", expected " <<
            2*pdf.integral(a,b) << "\n";
        return 1;
    }

    // Test sampling from the distribution
    if (testPDFSampling(a, b, expectationValueExpected, pdf) == 1) { return 1; }

    // Test parsing
    if (testPDFParsing() == 1) { return 1; }

    return 0; // If we have gotten here, tests have passed
}