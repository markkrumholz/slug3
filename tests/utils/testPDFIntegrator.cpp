/**
 * @file testPDFIntegrator.cpp
 * @author Mark Krumholz
 * @brief Unit tests for the PDFIntegrator class.
 * @date 2026-07-19
 */

#include "../../src/pdfs/PDF.hpp"
#include "../../src/pdfs/PDFFileParser.hpp"
#include "../../src/utils/PDFIntegrator.hpp"
#include "testPDFIntegrator.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
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

auto testPDFIntegrator() -> int
{
    const pdfs::PDF imf = pdfs::parsePDFDescriptor("data/imfs/chabrier.toml");

    int result = 0;
    result += testPlainFunction(imf);
    result += testMemberFunction(imf);
    return result;
}
