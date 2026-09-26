/**
 * @file testPDFIntegrator.cpp
 * @author Mark Krumholz
 * @brief Unit tests for the PDFIntegrator class.
 * @date 2026-07-19
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#include "../../src/pdfs/PDF.hpp"
#include "../../src/pdfs/PDFFileParser.hpp"
#include "../../src/pdfs/PDFReflect.hpp"
#include "../../src/pdfs/PDFSegment.hpp"
#include "../../src/pdfs/PDFSegmentDelta.hpp"
#include "../../src/pdfs/PDFSegmentPowerlaw.hpp"
#include "../../src/utils/PDFIntegrator.hpp"
#include "testPDFIntegrator.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

// Plain-function integrand: returns the constant value 2 for every x,
// wrapped in a 1-element array since PDFIntegrator requires a
// container-valued return
static auto constTwo(double /*x*/) -> std::array<double, 1>
{
    return { 2.0 };
}

namespace
{
    // Trivial class exercising the member-function-integrand case:
    // constructed from a vector of numbers m_, its method f(x)
    // returns m_[i] * x for every element of m_
    class Multiplier
    {
    public:
        explicit Multiplier(std::vector<double> m) : m_(std::move(m)) { }

        [[nodiscard]] auto f(const double x) const -> std::vector<double>
        {
            std::vector<double> result(m_.size());
            std::ranges::transform(m_, result.begin(),
                [x](const double mi) -> double { return mi * x; });
            return result;
        }

    private:
        std::vector<double> m_;
    };
} // namespace

// Verify the plain-function case: f(x) = {2.0} for every x, so
// int_a^b p(x) f(x) dx should equal 2 * PDF::integral(a, b)
static auto testPlainFunction(const pdfs::PDF& imf) -> int
{
    const utils::PDFIntegrator integrator(imf, constTwo, 1U);

    const double a = imf.getMin();
    const double b = imf.getMax();
    const auto result = integrator.integrate(a, b);

    const double expected = 2.0 * imf.integral(a, b);
    constexpr double tol = 1e-6;
    if (std::abs(result.at(0) - expected) > tol * std::abs(expected))
    {
        std::cerr << "testPDFIntegrator: plain-function case: expected "
            << expected << ", got " << result.at(0) << "\n";
        return 1;
    }
    return 0;
}

// Verify the member-function case: f(x) = m_ * x, so
// int_a^b p(x) f(x) dx should equal
// m_ * PDF::expectationValue(a, b) * PDF::integral(a, b), since
// expectationValue(a,b) is defined as
// [int_a^b x p(x) dx] / [int_a^b p(x) dx]
static auto testMemberFunction(const pdfs::PDF& imf) -> int
{
    const std::vector<double> m = { 1.0, -2.5, 3.0 };
    const Multiplier mult(m);

    const utils::PDFIntegrator integrator(
        imf, &Multiplier::f, static_cast<unsigned>(m.size()));

    const double a = imf.getMin();
    const double b = imf.getMax();
    const auto result = integrator.integrate(a, b, mult);

    const double firstMoment = imf.expectationValue(a, b) * imf.integral(a, b);
    constexpr double tol = 1e-6;
    for (std::size_t i = 0; i < m.size(); ++i)
    {
        const double expected = m.at(i) * firstMoment;
        if (std::abs(result.at(i) - expected) > tol * std::abs(expected))
        {
            std::cerr << "testPDFIntegrator: member-function case: at i = " << i
                << " expected " << expected << ", got " << result.at(i) << "\n";
            return 1;
        }
    }
    return 0;
}

// Verify that delta-function segments of a mixed PDF contribute their
// weight times f at their own location. Uses a PDF with a flat segment
// on [lo, hi] (weight 0.6) and a delta at xDelta (weight 0.4), and
// f(x) = x, so the integral over [a, b] must equal the PDF's own first
// moment there, expectationValue(a, b) * integral(a, b), which counts
// the delta; over the full range that is 0.6 * (lo + hi) / 2 +
// 0.4 * xDelta. Checks the full range, a subrange excluding the delta,
// a subrange ending exactly on it (included, as in
// PDFSegmentDelta::integral()), a zero-width interval at the delta
// (0, as in PDF::integral(a, b)), a PDFReflect view (delta reflected),
// and the log-transformed path.
static auto testDeltaSegments() -> int
{
    constexpr double tol = 1e-6;
    int result = 0;
    const Multiplier mult({ 1.0 });
    auto makePDF = [](const double lo, const double hi, const double xDelta) -> pdfs::PDF
    {
        std::vector<std::unique_ptr<pdfs::PDFSegment>> segs;
        segs.push_back(std::make_unique<pdfs::PDFSegmentPowerlaw>(lo, hi, 0.0));
        segs.push_back(std::make_unique<pdfs::PDFSegmentDelta>(xDelta));
        return { std::move(segs), std::vector<double>{ 0.6, 0.4 } };
    };
    auto check = [&](const char* label, const pdfs::PDF& pdf, const double a, const double b,
        const bool logTransform, const double expected) -> void
    {
        const utils::PDFIntegrator integrator(pdf, &Multiplier::f, 1U, logTransform);
        const double got = integrator.integrate(a, b, mult).at(0);
        const double moment = pdf.expectationValue(a, b) * pdf.integral(a, b);
        if (std::abs(got - expected) > tol * std::abs(expected) ||
            std::abs(got - moment) > tol * std::abs(moment))
        {
            std::cerr << "testPDFIntegrator: delta segments, " << label << ": expected "
                << expected << " (first moment " << moment << "), got " << got << "\n";
            result = 1;
        }
    };

    const pdfs::PDF mixed = makePDF(0.0, 1.0, 0.8);
    check("full range", mixed, 0.0, 1.0, false, (0.6 * 0.5) + (0.4 * 0.8));
    check("subrange excluding the delta", mixed, 0.0, 0.7, false, 0.6 * 0.7 * 0.7 / 2.0);
    check("subrange ending on the delta", mixed, 0.0, 0.8, false, (0.6 * 0.8 * 0.8 / 2.0) + (0.4 * 0.8));

    // A zero-width interval at the delta's own location integrates to
    // 0, matching PDF::integral(a, b)
    {
        const utils::PDFIntegrator integrator(mixed, &Multiplier::f, 1U);
        const double got = integrator.integrate(0.8, 0.8, mult).at(0);
        if (got != 0.0 || mixed.integral(0.8, 0.8) != 0.0)
        {
            std::cerr << "testPDFIntegrator: delta segments, zero-width interval: expected 0, got "
                << got << "\n";
            result = 1;
        }
    }
    const pdfs::PDFReflect reflected(mixed);
    check("reflected", reflected, 0.0, 1.0, false, (0.6 * 0.5) + (0.4 * 0.2));
    const pdfs::PDF mixedLog = makePDF(1.0, 2.0, 1.8);
    check("log transform", mixedLog, 1.0, 2.0, true, (0.6 * 1.5) + (0.4 * 1.8));
    return result;
}

auto testPDFIntegrator() -> int
{
    const pdfs::PDF imf = pdfs::parsePDFDescriptor("data/imfs/chabrier.toml");

    int result = 0;
    result += testPlainFunction(imf);
    result += testMemberFunction(imf);
    result += testDeltaSegments();
    return result;
}
