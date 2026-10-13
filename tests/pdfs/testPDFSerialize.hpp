/**
 * @file testPDFSerialize.hpp
 * @author Mark Krumholz
 * @brief Unit tests for PDF::toToml() and parsePDFTomlString()
 * @details
 * Checks that serializing a PDF with toToml() and reading it back with
 * parsePDFTomlString() reproduces it exactly -- identical weights,
 * limits, sampling method, normalization flag, evaluations, and draws
 * -- for every segment type, for PDFs read from basic- and
 * advanced-format descriptors, and for PDFs built directly in code.
 * @date 2026-10-13
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#ifndef TESTPDFSERIALIZE_HPP
#define TESTPDFSERIALIZE_HPP

#include "../../src/pdfs/PDF.hpp"
#include "../../src/pdfs/PDFFileParser.hpp"
#include "../../src/pdfs/PDFReflect.hpp"
#include "../../src/pdfs/PDFSegmentDelta.hpp"
#include "../../src/pdfs/PDFSegmentPowerlaw.hpp"
#include "../../src/utils/RngThread.hpp"
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

/**
 * @brief Check that a PDF survives a toToml()/parsePDFTomlString() round trip exactly
 * @param label Label to print on failure
 * @param pdf The PDF to check
 * @param points Points at which to compare evaluations, integrals, and expectation values
 * @return 0 if the round trip is exact, 1 otherwise
 */
inline auto checkPDFRoundTrip(const std::string& label, const pdfs::PDF& pdf,
    const std::vector<double>& points) -> int
{
    const std::string descriptor = pdf.toToml();
    const pdfs::PDF copy = pdfs::parsePDFTomlString(descriptor);
    auto fail = [&label, &descriptor](const std::string& what) -> int
    {
        std::cerr << "testPDFSerialize (" << label << "): " << what
            << " differs after round trip; descriptor was:\n" << descriptor << "\n";
        return 1;
    };

    // Header-level state
    const auto& w = pdf.getWeights();
    const auto& wCopy = copy.getWeights();
    if (w.size() != wCopy.size()) { return fail("number of segments"); }
    for (std::size_t i = 0; i < w.size(); ++i)
    {
        if (w[i] != wCopy[i]) { return fail("weight " + std::to_string(i)); } // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
    }
    if (pdf.getMin() != copy.getMin() || pdf.getMax() != copy.getMax()) { return fail("limits"); }
    if (pdf.normalized() != copy.normalized()) { return fail("normalized()"); }
    if (pdf.getSampling() != copy.getSampling()) { return fail("sampling method"); }

    // Serializing the copy gives the same text
    if (copy.toToml() != descriptor) { return fail("re-serialized text"); }

    // Evaluations, integrals, and expectation values, compared exactly
    for (const double x : points)
    {
        if (pdf(x) != copy(x)) { return fail("value at " + std::to_string(x)); }
        if (pdf.integral(pdf.getMin(), x) != copy.integral(copy.getMin(), x))
        {
            return fail("integral up to " + std::to_string(x));
        }
    }
    if (pdf.expectationValue() != copy.expectationValue()) { return fail("expectation value"); }

    // Draws from identical rng states
    constexpr std::size_t nDraw = 200;
    constexpr std::uint64_t seed = 12345;
    std::vector<double> drawsA(nDraw);
    std::vector<double> drawsB(nDraw);
    utils::rng().seed(seed);
    for (auto& d : drawsA) { d = pdf.draw(); }
    utils::rng().seed(seed);
    for (auto& d : drawsB) { d = copy.draw(); }
    if (drawsA != drawsB) { return fail("draw sequence"); }
    return 0;
}

/**
 * @brief Round-trip PDFs of every segment type and origin through toToml()
 * @return Number of failures
 */
inline auto testPDFSerializeRoundTrip() -> int
{
    int failures = 0;

    // Basic format, multi-segment: a Chabrier-like lognormal +
    // powerlaw IMF, whose weights are computed for continuity and
    // then normalized. disp = 0.11 dex is chosen deliberately: it is
    // one of the values for which ln(10) * disp / ln(10) != disp in
    // floating point, so this checks that the lognormal segment writes
    // back the dispersion it was given rather than recomputing it.
    failures += checkPDFRoundTrip("basic lognormal+powerlaw",
        pdfs::parsePDFTomlString(R"(
breakpoints = [0.08, 1.0, 120.0]
method = "stop_before"
[segment1]
type = "lognormal"
mean = 0.2
disp = 0.11
[segment2]
type = "powerlaw"
slope = -2.35
)"), {0.1, 0.5, 1.0, 3.0, 50.0});

    // Basic format, one segment of each remaining continuous type
    failures += checkPDFRoundTrip("basic normal",
        pdfs::parsePDFTomlString(R"(
breakpoints = [1.0, 100.0]
[segment1]
type = "normal"
mean = 2.1
disp = 0.5
)"), {1.5, 2.0, 2.7, 10.0});
    failures += checkPDFRoundTrip("basic exponential+schechter",
        pdfs::parsePDFTomlString(R"(
breakpoints = [0.0, 1.0, 10.0]
method = "poisson"
[segment1]
type = "exponential"
scale = 0.7
[segment2]
type = "schechter"
slope = -1.5
x_star = 3.3
)"), {0.2, 0.9, 1.5, 7.0});

    // Advanced format, including a delta segment and explicit
    // (non-normalized) weights
    failures += checkPDFRoundTrip("advanced delta+powerlaw",
        pdfs::parsePDFTomlString(R"(
format = "advanced"
method = "sorted_sampling"
[segment1]
type = "delta"
min = 0.5
max = 0.5
weight = 0.3
[segment2]
type = "powerlaw"
slope = -1.0
min = 1.0
max = 10.0
weight = 2.7
)"), {0.5, 2.0, 9.0});

    // Built directly in code: a delta (as SimControls builds for a
    // constant [Fe/H] or A_V), and a flat powerlaw with a large,
    // non-unit weight and hence not normalized (as
    // SimControls::buildConstantSFR() builds for a constant SFR)
    failures += checkPDFRoundTrip("code-built delta",
        pdfs::PDF(std::make_unique<pdfs::PDFSegmentDelta>(-0.37)), {-0.37});
    failures += checkPDFRoundTrip("code-built constant SFR",
        pdfs::PDF(std::make_unique<pdfs::PDFSegmentPowerlaw>(0.0, 1e15, 0.0), 2.5e15),
        {1e6, 1e9, 1e12});

    return failures;
}

/**
 * @brief Check the error cases of toToml() and the "normalized" key
 * @return Number of failures
 */
inline auto testPDFSerializeErrors() -> int
{
    int failures = 0;
    auto expectThrow = [&failures](const std::string& what, auto&& fn) -> void
    {
        try
        {
            fn();
            std::cerr << "testPDFSerialize: expected an exception for " << what << "\n";
            ++failures;
        }
        catch (const std::exception&) {} // NOLINT(bugprone-empty-catch) -- expected
    };

    // An invalid (empty) PDF, and a PDFReflect view, cannot be serialized
    expectThrow("toToml() on an empty PDF", [] { (void)pdfs::PDF().toToml(); });
    const pdfs::PDF base(std::make_unique<pdfs::PDFSegmentPowerlaw>(1.0, 2.0, 0.0));
    expectThrow("toToml() on a PDFReflect", [&base] { (void)pdfs::PDFReflect(base).toToml(); });
    pdfs::PDF noMethod(std::make_unique<pdfs::PDFSegmentPowerlaw>(1.0, 2.0, 0.0));
    noMethod.setSampling(pdfs::SamplingMethods::none);
    expectThrow("toToml() with sampling method none", [&noMethod] { (void)noMethod.toToml(); });

    // "normalized" is only allowed in advanced format, and must be a boolean
    expectThrow("normalized in basic format", [] {
        (void)pdfs::parsePDFTomlString(R"(
breakpoints = [1.0, 2.0]
normalized = true
[segment1]
type = "powerlaw"
slope = 0.0
)");
    });
    expectThrow("non-boolean normalized", [] {
        (void)pdfs::parsePDFTomlString(R"(
format = "advanced"
normalized = 1
[segment1]
type = "powerlaw"
slope = 0.0
min = 1.0
max = 2.0
weight = 1.0
)");
    });

    // Invalid toml text
    expectThrow("invalid toml", [] { (void)pdfs::parsePDFTomlString("this is [ not toml"); });

    // normalized = true marks the PDF normalized without rescaling its
    // weights (which here deliberately do not sum to 1)
    const pdfs::PDF marked = pdfs::parsePDFTomlString(R"(
format = "advanced"
normalized = true
[segment1]
type = "powerlaw"
slope = 0.0
min = 1.0
max = 2.0
weight = 0.25
)");
    if (!marked.normalized() || marked.getWeights()[0] != 0.25)
    {
        std::cerr << "testPDFSerialize: normalized = true should set normalized() "
            "without rescaling the weights\n";
        ++failures;
    }
    return failures;
}

#endif // TESTPDFSERIALIZE_HPP
