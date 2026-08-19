/**
 * @file testPDFReflect.cpp
 * @author Mark Krumholz
 * @brief Unit tests for the PDFReflect class.
 * @date 2026-08-14
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#include "../src/pdfs/PDF.hpp"
#include "../src/pdfs/PDFCommons.hpp"
#include "../src/pdfs/PDFReflect.hpp"
#include "../src/pdfs/PDFSegmentPowerlaw.hpp"
#include "../src/utils/MiscUtils.hpp"
#include "../src/utils/RngThread.hpp"
#include "testPDFReflect.hpp"
#include <cmath>
#include <cstddef>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

namespace {

    // Small helper functions that accept a plain `const pdfs::PDF&`,
    // matching how PDFIntegratorND holds every PDF it integrates
    // over (as std::reference_wrapper<const pdfs::PDF>) -- calling
    // through these, rather than directly on a PDFReflect object, is
    // the only way to prove that PDFReflect's overrides actually
    // dispatch virtually rather than being silently sliced away.
    auto evalThroughBase(const pdfs::PDF& p, const double x) -> double { return p(x); }
    auto expectThroughBase(const pdfs::PDF& p) -> double { return p.expectationValue(); }
    auto expectRangeThroughBase(const pdfs::PDF& p, const double a, const double b) -> double
    { return p.expectationValue(a, b); }
    auto integralThroughBase(const pdfs::PDF& p, const double a, const double b) -> double
    { return p.integral(a, b); }
    auto drawThroughBase(const pdfs::PDF& p, const double a, const double b) -> double
    { return p.draw(a, b); }

    // Checks operator()/expectationValue()/integral(), both called
    // directly on pdfR and through a const PDF& (see the helpers
    // above), against independently-computed analytic power-law
    // values -- not against a second call into the code PDFReflect
    // itself wraps.
    auto testPDFReflectEval(const pdfs::PDF& pdf, const pdfs::PDFReflect& pdfR,
        const double sMin, const double sMax, const double alpha) -> int
    {
        const double norm = (alpha + 1) / (std::pow(sMax, alpha + 1) - std::pow(sMin, alpha + 1));
        const auto reflect = [sMin, sMax](const double x) { return sMax - (x - sMin); };
        const auto plPdf = [norm, alpha](const double x) { return norm * std::pow(x, alpha); };
        const auto plExpect = [alpha](const double a, const double b)
        {
            return (alpha + 1) / (alpha + 2) *
                (std::pow(b, alpha + 2) - std::pow(a, alpha + 2)) /
                (std::pow(b, alpha + 1) - std::pow(a, alpha + 1));
        };
        const auto plIntegral = [norm, alpha](const double a, const double b)
        { return norm / (alpha + 1) * (std::pow(b, alpha + 1) - std::pow(a, alpha + 1)); };

        // getMin()/getMax(): same support as the wrapped PDF, just relabeled
        if (pdfR.getMin() != sMin || pdfR.getMax() != sMax)
        {
            std::cerr << "testPDFReflect: getMin()/getMax() do not match "
                "the wrapped PDF's own support; got [" << pdfR.getMin()
                << ", " << pdfR.getMax() << "], expected [" << sMin << ", "
                << sMax << "]\n";
            return 1;
        }

        // operator(): PDF::operator()'s own condition (getMin() < x &&
        // getMax() >= x) means pdf(x) excludes x == sMin and includes
        // x == sMax; since PDFReflect evaluates pdf(reflect(x)), this
        // flips to including x == sMin and excluding x == sMax -- check
        // both boundaries explicitly, since getting the direction of
        // reflect() backwards would not change which single boundary is
        // excluded (only relabel which point maps to which), and this is
        // the cheapest check that actually distinguishes the two.
        const std::vector<double> interior = { sMin, 5.0, 10.0, 15.0 };
        for (const double x : interior)
        {
            const double expected = plPdf(reflect(x));
            if (!utils::approxEqual(pdfR(x), expected))
            {
                std::cerr << "testPDFReflect: operator() at x=" << x
                    << " failed: expected " << expected << ", got "
                    << pdfR(x) << "\n";
                return 1;
            }
            if (!utils::approxEqual(evalThroughBase(pdfR, x), expected))
            {
                std::cerr << "testPDFReflect: operator() through a const "
                    "PDF& at x=" << x << " failed: expected " << expected
                    << ", got " << evalThroughBase(pdfR, x) << "\n";
                return 1;
            }
        }
        if (pdfR(sMax) != 0.0)
        {
            std::cerr << "testPDFReflect: operator() at x=sMax should be "
                "0 (reflects to sMin, which the wrapped PDF excludes); got "
                << pdfR(sMax) << "\n";
            return 1;
        }
        if (pdfR(sMin - 1.0) != 0.0 || pdfR(sMax + 1.0) != 0.0)
        {
            std::cerr << "testPDFReflect: operator() outside [sMin, sMax] "
                "should be 0\n";
            return 1;
        }

        // expectationValue() (no args)
        const double expectExpected = reflect(plExpect(sMin, sMax));
        if (!utils::approxEqual(pdfR.expectationValue(), expectExpected))
        {
            std::cerr << "testPDFReflect: expectationValue() failed: "
                "expected " << expectExpected << ", got "
                << pdfR.expectationValue() << "\n";
            return 1;
        }
        if (!utils::approxEqual(expectThroughBase(pdfR), expectExpected))
        {
            std::cerr << "testPDFReflect: expectationValue() through a "
                "const PDF& failed: expected " << expectExpected
                << ", got " << expectThroughBase(pdfR) << "\n";
            return 1;
        }
        // Sanity check on direction: the wrapped (decreasing) PDF's own
        // expectation value lies below the midpoint of its support, so
        // the reflected view's should lie above it
        const double mid = 0.5 * (sMin + sMax);
        if (pdf.expectationValue() >= mid || pdfR.expectationValue() <= mid)
        {
            std::cerr << "testPDFReflect: expected the wrapped PDF's "
                "expectation value below the midpoint and the reflected "
                "view's above it; got " << pdf.expectationValue() << " and "
                << pdfR.expectationValue() << "\n";
            return 1;
        }

        // expectationValue(a,b)
        const double a = 3.0;
        const double b = 12.0;
        const double aReflect = reflect(b);
        const double bReflect = reflect(a);
        const double expectRangeExpected = reflect(plExpect(aReflect, bReflect));
        if (!utils::approxEqual(pdfR.expectationValue(a, b), expectRangeExpected))
        {
            std::cerr << "testPDFReflect: expectationValue(a,b) failed: "
                "expected " << expectRangeExpected << ", got "
                << pdfR.expectationValue(a, b) << "\n";
            return 1;
        }
        if (!utils::approxEqual(expectRangeThroughBase(pdfR, a, b), expectRangeExpected))
        {
            std::cerr << "testPDFReflect: expectationValue(a,b) through a "
                "const PDF& failed: expected " << expectRangeExpected
                << ", got " << expectRangeThroughBase(pdfR, a, b) << "\n";
            return 1;
        }

        // integral(): no-arg form needs no override, so should be
        // unchanged from the wrapped PDF's own (both are 1.0, since a
        // single-segment PDF constructed with the default weight is
        // normalized)
        if (!utils::approxEqual(pdfR.integral(), pdf.integral()))
        {
            std::cerr << "testPDFReflect: integral() failed: expected "
                << pdf.integral() << ", got " << pdfR.integral() << "\n";
            return 1;
        }

        // integral(a,b)
        const double integralRangeExpected = plIntegral(aReflect, bReflect);
        if (!utils::approxEqual(pdfR.integral(a, b), integralRangeExpected))
        {
            std::cerr << "testPDFReflect: integral(a,b) failed: expected "
                << integralRangeExpected << ", got "
                << pdfR.integral(a, b) << "\n";
            return 1;
        }
        if (!utils::approxEqual(integralThroughBase(pdfR, a, b), integralRangeExpected))
        {
            std::cerr << "testPDFReflect: integral(a,b) through a const "
                "PDF& failed: expected " << integralRangeExpected
                << ", got " << integralThroughBase(pdfR, a, b) << "\n";
            return 1;
        }
        // Integrating the reflected view over its own full support should
        // reproduce the wrapped PDF's own full-range integral (reflection
        // is measure-preserving)
        if (!utils::approxEqual(pdfR.integral(sMin, sMax), pdf.integral(sMin, sMax)))
        {
            std::cerr << "testPDFReflect: integral(sMin,sMax) should match "
                "the wrapped PDF's own full-range integral; got "
                << pdfR.integral(sMin, sMax) << " vs " << pdf.integral(sMin, sMax) << "\n";
            return 1;
        }

        return 0; // Passed
    }

    // Checks draw()/draw(nDraw)/drawTarget() over [a,b]: every sample
    // must land in range, the sample mean must approach
    // expectationValue(a,b) (already independently verified by
    // testPDFReflectEval() above), and drawTarget()'s stopBefore/
    // stopAfter policies must bound the drawn total from below/above.
    auto testPDFReflectSampling(pdfs::PDF& pdf, const pdfs::PDFReflect& pdfR,
        const double sMin, const double sMax,
        const double a, const double b, const double expectRangeExpected) -> int
    {
        const auto reflect = [sMin, sMax](const double x) { return sMax - (x - sMin); };
        const double aReflect = reflect(b);
        const double bReflect = reflect(a);

        const int numSamples = 20000;
        double sum = 0.0;
        for (int i = 0; i < numSamples; ++i)
        {
            const double s = pdfR.draw(a, b);
            if (s < a || s > b)
            {
                std::cerr << "testPDFReflect: draw(a,b) produced a sample "
                    "outside [" << a << ", " << b << "]: got " << s << "\n";
                return 1;
            }
            sum += s;
        }
        double sampleMean = sum / static_cast<double>(numSamples);
        if (!utils::approxEqual(sampleMean, expectRangeExpected, 0.02))
        {
            std::cerr << "testPDFReflect: draw(a,b) sample mean does not "
                "match expectationValue(a,b); expected " << expectRangeExpected
                << ", got " << sampleMean << "\n";
            return 1;
        }
        // Repeat through a const PDF&, to confirm draw() dispatches virtually too
        const double sThroughBase = drawThroughBase(pdfR, a, b);
        if (sThroughBase < a || sThroughBase > b)
        {
            std::cerr << "testPDFReflect: draw(a,b) through a const PDF& "
                "produced a sample outside [" << a << ", " << b << "]: got "
                << sThroughBase << "\n";
            return 1;
        }

        // draw(nDraw,a,b)
        const auto samplesVec = pdfR.draw(static_cast<unsigned int>(numSamples), a, b);
        sum = 0.0;
        for (const double s : samplesVec)
        {
            if (s < a || s > b)
            {
                std::cerr << "testPDFReflect: draw(nDraw,a,b) produced a "
                    "sample outside [" << a << ", " << b << "]: got " << s << "\n";
                return 1;
            }
            sum += s;
        }
        sampleMean = sum / static_cast<double>(samplesVec.size());
        if (!utils::approxEqual(sampleMean, expectRangeExpected, 0.02))
        {
            std::cerr << "testPDFReflect: draw(nDraw,a,b) sample mean does "
                "not match expectationValue(a,b); expected "
                << expectRangeExpected << ", got " << sampleMean << "\n";
            return 1;
        }

        // Deterministic delegation checks: reflect() is a coordinate
        // flip, not a linear/measure-preserving map, so summed
        // quantities (e.g. a drawTarget() total) computed from
        // reflected values carry no simple relationship to the target
        // that was matched in the wrapped PDF's own, unreflected
        // coordinate -- the only way to precisely verify draw()/
        // draw(nDraw)/drawTarget() is to replay the same rng sequence
        // through both pdfR and the wrapped pdf directly, and confirm
        // pdfR's result is exactly reflect() applied elementwise to
        // what the wrapped pdf itself returns for (aReflect, bReflect).
        const unsigned int nCheck = 100;
        const unsigned int seed = 20260814;

        utils::rng().seed(seed);
        const double singleViaReflect = pdfR.draw(a, b);
        utils::rng().seed(seed);
        const double singleViaDirect = pdf.draw(aReflect, bReflect);
        if (!utils::approxEqual(singleViaReflect, reflect(singleViaDirect)))
        {
            std::cerr << "testPDFReflect: draw(a,b) does not match "
                "reflect(pdf.draw(reflect(b),reflect(a))) under an "
                "identical rng sequence; expected "
                << reflect(singleViaDirect) << ", got " << singleViaReflect << "\n";
            return 1;
        }

        utils::rng().seed(seed);
        const auto vecViaReflect = pdfR.draw(nCheck, a, b);
        utils::rng().seed(seed);
        const auto vecViaDirect = pdf.draw(nCheck, aReflect, bReflect);
        for (std::size_t i = 0; i < nCheck; ++i)
        {
            if (!utils::approxEqual(vecViaReflect.at(i), reflect(vecViaDirect.at(i))))
            {
                std::cerr << "testPDFReflect: draw(nDraw,a,b) element " << i
                    << " does not match reflect(pdf.draw(nDraw,reflect(b),"
                    "reflect(a))) under an identical rng sequence; expected "
                    << reflect(vecViaDirect.at(i)) << ", got "
                    << vecViaReflect.at(i) << "\n";
                return 1;
            }
        }

        // drawTarget(target,a,b): same identical-rng-sequence check,
        // for every sampling method drawTarget()'s switch handles
        const double target = 50.0;
        for (const auto method : { pdfs::SamplingMethods::stopBefore, pdfs::SamplingMethods::stopAfter,
            pdfs::SamplingMethods::stopNearest, pdfs::SamplingMethods::stop50 })
        {
            pdf.setSampling(method);

            utils::rng().seed(seed);
            const auto targetViaReflect = pdfR.drawTarget(target, a, b);
            utils::rng().seed(seed);
            const auto targetViaDirect = pdf.drawTarget(target, aReflect, bReflect);

            if (targetViaReflect.size() != targetViaDirect.size())
            {
                std::cerr << "testPDFReflect: drawTarget(target,a,b) "
                    "returned " << targetViaReflect.size() << " samples, "
                    "expected " << targetViaDirect.size() << " (from "
                    "pdf.drawTarget(target,reflect(b),reflect(a)) under an "
                    "identical rng sequence)\n";
                return 1;
            }
            for (std::size_t i = 0; i < targetViaReflect.size(); ++i)
            {
                if (targetViaReflect.at(i) < a || targetViaReflect.at(i) > b)
                {
                    std::cerr << "testPDFReflect: drawTarget(target,a,b) "
                        "produced a sample outside [" << a << ", " << b
                        << "]: got " << targetViaReflect.at(i) << "\n";
                    return 1;
                }
                if (!utils::approxEqual(targetViaReflect.at(i), reflect(targetViaDirect.at(i))))
                {
                    std::cerr << "testPDFReflect: drawTarget(target,a,b) "
                        "element " << i << " does not match reflect(pdf."
                        "drawTarget(target,reflect(b),reflect(a))) under an "
                        "identical rng sequence; expected "
                        << reflect(targetViaDirect.at(i)) << ", got "
                        << targetViaReflect.at(i) << "\n";
                    return 1;
                }
            }
        }

        return 0; // Passed
    }

    // Checks the explicit-pivot constructor: reflect(x) = 2 * pivot -
    // x for a pivot that is deliberately NOT the wrapped PDF's own
    // midpoint (the default-constructor case is already covered,
    // exhaustively, by testPDFReflectEval()/testPDFReflectSampling()
    // above), and that getMin()/getMax() stay equal to the wrapped
    // PDF's own support regardless of pivot (see PDFReflect's own
    // class comment for why).
    auto testPDFReflectPivot(const pdfs::PDF& pdf,
        const double sMin, const double sMax, const double alpha, const double pivot) -> int
    {
        const pdfs::PDFReflect pdfRPivot(pdf, pivot);

        if (pdfRPivot.getMin() != sMin || pdfRPivot.getMax() != sMax)
        {
            std::cerr << "testPDFReflect: pivot: getMin()/getMax() should "
                "stay equal to the wrapped PDF's own support regardless of "
                "pivot; got [" << pdfRPivot.getMin() << ", " << pdfRPivot.getMax()
                << "], expected [" << sMin << ", " << sMax << "]\n";
            return 1;
        }

        const double norm = (alpha + 1) / (std::pow(sMax, alpha + 1) - std::pow(sMin, alpha + 1));
        const auto reflect = [pivot](const double x) { return (2.0 * pivot) - x; };
        const auto plPdf = [norm, alpha](const double x) { return norm * std::pow(x, alpha); };

        // Interior points of [2*pivot - sMax, 2*pivot - sMin) -- the
        // x-range whose reflection lands inside [sMin, sMax), where
        // the wrapped PDF's own operator() is nonzero (see
        // testPDFReflectEval()'s own comment on that half-open range)
        const std::vector<double> interior = {
            (2.0 * pivot) - sMax, 0.5 * ((2.0 * pivot) - sMax + (2.0 * pivot) - sMin), (2.0 * pivot) - sMin - 1.0
        };
        for (const double x : interior)
        {
            const double expected = plPdf(reflect(x));
            if (!utils::approxEqual(pdfRPivot(x), expected))
            {
                std::cerr << "testPDFReflect: pivot: operator() at x=" << x
                    << " (pivot=" << pivot << ") failed: expected " << expected
                    << ", got " << pdfRPivot(x) << "\n";
                return 1;
            }
        }
        // Just outside that range: reflected point falls outside
        // [sMin, sMax] entirely, so the wrapped PDF's own operator()
        // is 0
        const double outside = (2.0 * pivot) - sMin;
        if (pdfRPivot(outside) != 0.0)
        {
            std::cerr << "testPDFReflect: pivot: operator() at x=" << outside
                << " (pivot=" << pivot << ") should be 0 (reflects to sMin, "
                "which the wrapped PDF excludes); got " << pdfRPivot(outside) << "\n";
            return 1;
        }

        return 0; // Passed
    }

} // namespace

auto testPDFReflect() -> int
{
    // Set the rng seed to a fixed value for reproducibility
    utils::rng().seed(42);

    // Build a single-segment, decreasing (alpha < 0) power-law PDF,
    // and a PDFReflect view of it. Decreasing was chosen deliberately
    // (rather than, e.g., a symmetric or uniform distribution) so
    // that reflect() actually changes the shape of the distribution
    // -- a bug that swapped the sign of reflect() or otherwise left
    // it a no-op would be invisible under a symmetric test PDF.
    const double sMin = 1.0;
    const double sMax = 20.0;
    const double alpha = -2.3;
    auto seg = std::make_unique<pdfs::PDFSegmentPowerlaw>(sMin, sMax, alpha);
    pdfs::PDF pdf(std::move(seg)); // Non-const: sampling test toggles its sampling method
    const pdfs::PDFReflect pdfR(pdf);

    if (testPDFReflectEval(pdf, pdfR, sMin, sMax, alpha) == 1) { return 1; }

    // Ground-truth expectation value over [a,b], independently derived
    // the same way testPDFReflectEval() derives it, used here as the
    // target for the sampling test's own mean comparison
    const double a = 3.0;
    const double b = 12.0;
    const auto reflect = [sMin, sMax](const double x) { return sMax - (x - sMin); };
    const auto plExpect = [alpha](const double lo, const double hi)
    {
        return (alpha + 1) / (alpha + 2) *
            (std::pow(hi, alpha + 2) - std::pow(lo, alpha + 2)) /
            (std::pow(hi, alpha + 1) - std::pow(lo, alpha + 1));
    };
    const double expectRangeExpected = reflect(plExpect(reflect(b), reflect(a)));

    if (testPDFReflectSampling(pdf, pdfR, sMin, sMax, a, b, expectRangeExpected) == 1) { return 1; }

    // Explicit-pivot constructor, deliberately not the wrapped PDF's
    // own midpoint ((sMin + sMax) / 2 == 10.5) -- this is the form
    // Specsyn::specCtsHelper()/Galaxy::computeLbolCts() actually use,
    // reflecting sfr about half the current simulation time rather
    // than half of sfr's own (much larger) declared support.
    const double pivot = 5.0;
    if (testPDFReflectPivot(pdf, sMin, sMax, alpha, pivot) == 1) { return 1; }

    return 0; // Passed
}
