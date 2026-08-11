/**
 * @file testPDFIntegratorND.cpp
 * @author Mark Krumholz
 * @brief Unit tests for the PDFIntegratorND class.
 * @date 2026-08-10
 */

#include "../../src/pdfs/PDF.hpp"
#include "../../src/pdfs/PDFFileParser.hpp"
#include "../../src/utils/PDFIntegratorND.hpp"
#include "testPDFIntegratorND.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iostream>
#include <mdspan> // NOLINT(misc-include-cleaner)
#include <span>
#include <utility>
#include <vector>

// Plain-function integrand for the 1-dimensional case: returns the
// constant value 2 for every x, wrapped in a 1-element array since
// PDFIntegratorND requires a container-valued return -- mirrors
// testPDFIntegrator.cpp's own constTwo(), just taking a span instead
// of a bare double
static auto constTwo1D(std::span<double> /*x*/) -> std::array<double, 1>
{
    return { 2.0 };
}

// Plain-function integrand for the 2-dimensional case: returns the
// constant value 3 for every x
static auto constThree2D(std::span<double> /*x*/) -> std::array<double, 1>
{
    return { 3.0 };
}

// The (npts, 2) view type a Vectorized = true, N = 2 integrand must
// accept -- named independently of PDFIntegratorND's own nested
// PointsView alias, since that alias is itself a member of the class
// template being instantiated (which needs F's type up front), so it
// can't be used to help spell F's own signature.
using PointsView2D = std::mdspan<double, std::extents<std::size_t, std::dynamic_extent, 2>>; // NOLINT(misc-include-cleaner)

// Plain-function integrand for the 2-dimensional, vectorized case:
// returns the constant value 3 for every point in the batch, laid out
// as result[i] for the i-th of points.extent(0) points (nInt = 1, so
// there is only one quantity per point)
static auto constThree2DVec(const PointsView2D points) -> std::vector<double>
{
    // Deliberately NOT a braced-init-list: std::vector<double>{n, v}
    // would call the initializer_list constructor (a 2-element vector
    // {n, v}), not the (count, value) fill constructor this actually needs
    return std::vector<double>(points.extent(0), 3.0); // NOLINT(modernize-return-braced-init-list)
}

namespace
{
    // Trivial class exercising the member-function-integrand case in
    // 2 dimensions: constructed from a vector of numbers m_, its
    // method f(x) returns m_[i] * x[0] * x[1] for every element of m_
    class Multiplier2D
    {
    public:
        explicit Multiplier2D(std::vector<double> m) : m_(std::move(m)) { }

        [[nodiscard]] auto f(const std::span<double> x) const -> std::vector<double>
        {
            std::vector<double> result(m_.size());
            std::ranges::transform(m_, result.begin(),
                [x](const double mi) -> double { return mi * x[0] * x[1]; });
            return result;
        }

    private:
        std::vector<double> m_;
    };

    // Vectorized counterpart to Multiplier2D: f(points) returns
    // m_[k] * points[i, 0] * points[i, 1] for every point i and every
    // element k of m_, laid out as result[i * m_.size() + k] (the
    // same convention PDFIntegratorND's own @tparam F documents for
    // fval)
    class Multiplier2DVec
    {
    public:
        explicit Multiplier2DVec(std::vector<double> m) : m_(std::move(m)) { }

        [[nodiscard]] auto f(const PointsView2D points) const -> std::vector<double>
        {
            const auto npts = points.extent(0);
            std::vector<double> result(npts * m_.size());
            for (std::size_t i = 0; i < npts; ++i)
            {
                const double x0 = points[i, 0];
                const double x1 = points[i, 1];
                for (std::size_t k = 0; k < m_.size(); ++k)
                {
                    result[(i * m_.size()) + k] = m_[k] * x0 * x1;
                }
            }
            return result;
        }

    private:
        std::vector<double> m_;
    };
} // namespace

// Verify the N = 1 case reduces to the same behavior as PDFIntegrator
// itself: f(x) = {2.0} for every x, so int_a^b p(x) f(x) dx should
// equal 2 * PDF::integral(a, b)
static auto test1D(const pdfs::PDF& imf) -> int
{
    const utils::PDFIntegratorND<decltype(&constTwo1D), 1> integrator(
        { std::cref(imf) }, constTwo1D, 1U);

    const double a = imf.getMin();
    const double b = imf.getMax();
    const auto result = integrator.integrate({ a }, { b });

    const double expected = 2.0 * imf.integral(a, b);
    constexpr double tol = 1e-6;
    if (std::abs(result.at(0) - expected) > tol * std::abs(expected))
    {
        std::cerr << "testPDFIntegratorND: 1D case: expected "
            << expected << ", got " << result.at(0) << "\n";
        return 1;
    }
    return 0;
}

// Verify the plain-function, 2-dimensional case: f(x) = {3.0} for
// every x, so int p1(x1) p2(x2) f(x) d^2x should equal
// 3 * p1.integral(a1, b1) * p2.integral(a2, b2), since p1 and p2 are
// independent (the integral over the tensor-product box factorizes)
static auto testPlainFunction2D(const pdfs::PDF& p1, const pdfs::PDF& p2) -> int
{
    const utils::PDFIntegratorND<decltype(&constThree2D), 2> integrator(
        { std::cref(p1), std::cref(p2) }, constThree2D, 1U);

    const double a1 = p1.getMin();
    const double b1 = p1.getMax();
    const double a2 = p2.getMin();
    const double b2 = p2.getMax();
    const auto result = integrator.integrate({ a1, a2 }, { b1, b2 });

    const double expected = 3.0 * p1.integral(a1, b1) * p2.integral(a2, b2);
    constexpr double tol = 1e-4;
    if (std::abs(result.at(0) - expected) > tol * std::abs(expected))
    {
        std::cerr << "testPDFIntegratorND: plain-function 2D case: expected "
            << expected << ", got " << result.at(0) << "\n";
        return 1;
    }
    return 0;
}

// Verify the member-function, 2-dimensional case: f(x) = m_ * x1 * x2,
// so int p1(x1) p2(x2) f(x) d^2x should equal
// m_ * [p1.expectationValue(a1,b1) * p1.integral(a1,b1)] *
// [p2.expectationValue(a2,b2) * p2.integral(a2,b2)], by the same
// factorization argument as testPlainFunction2D
static auto testMemberFunction2D(const pdfs::PDF& p1, const pdfs::PDF& p2) -> int
{
    const std::vector<double> m = { 1.0, -2.5, 3.0 };
    const Multiplier2D mult(m);

    const utils::PDFIntegratorND<decltype(&Multiplier2D::f), 2> integrator(
        { std::cref(p1), std::cref(p2) }, &Multiplier2D::f, static_cast<unsigned>(m.size()));

    const double a1 = p1.getMin();
    const double b1 = p1.getMax();
    const double a2 = p2.getMin();
    const double b2 = p2.getMax();
    const auto result = integrator.integrate({ a1, a2 }, { b1, b2 }, mult);

    const double firstMoment1 = p1.expectationValue(a1, b1) * p1.integral(a1, b1);
    const double firstMoment2 = p2.expectationValue(a2, b2) * p2.integral(a2, b2);
    constexpr double tol = 1e-4;
    for (std::size_t i = 0; i < m.size(); ++i)
    {
        const double expected = m.at(i) * firstMoment1 * firstMoment2;
        if (std::abs(result.at(i) - expected) > tol * std::abs(expected))
        {
            std::cerr << "testPDFIntegratorND: member-function 2D case: at i = " << i
                << " expected " << expected << ", got " << result.at(i) << "\n";
            return 1;
        }
    }
    return 0;
}

// Verify the plain-function, 2-dimensional, vectorized case gives the
// same result as testPlainFunction2D's own non-vectorized version --
// f(points) = {3.0} at every point in the batch
static auto testPlainFunctionVec2D(const pdfs::PDF& p1, const pdfs::PDF& p2) -> int
{
    const utils::PDFIntegratorND<decltype(&constThree2DVec), 2, true> integrator(
        { std::cref(p1), std::cref(p2) }, constThree2DVec, 1U);

    const double a1 = p1.getMin();
    const double b1 = p1.getMax();
    const double a2 = p2.getMin();
    const double b2 = p2.getMax();
    const auto result = integrator.integrate({ a1, a2 }, { b1, b2 });

    const double expected = 3.0 * p1.integral(a1, b1) * p2.integral(a2, b2);
    constexpr double tol = 1e-4;
    if (std::abs(result.at(0) - expected) > tol * std::abs(expected))
    {
        std::cerr << "testPDFIntegratorND: plain-function vectorized 2D case: expected "
            << expected << ", got " << result.at(0) << "\n";
        return 1;
    }
    return 0;
}

// Verify the member-function, 2-dimensional, vectorized case gives the
// same result as testMemberFunction2D's own non-vectorized version
static auto testMemberFunctionVec2D(const pdfs::PDF& p1, const pdfs::PDF& p2) -> int
{
    const std::vector<double> m = { 1.0, -2.5, 3.0 };
    const Multiplier2DVec mult(m);

    const utils::PDFIntegratorND<decltype(&Multiplier2DVec::f), 2, true> integrator(
        { std::cref(p1), std::cref(p2) }, &Multiplier2DVec::f, static_cast<unsigned>(m.size()));

    const double a1 = p1.getMin();
    const double b1 = p1.getMax();
    const double a2 = p2.getMin();
    const double b2 = p2.getMax();
    const auto result = integrator.integrate({ a1, a2 }, { b1, b2 }, mult);

    const double firstMoment1 = p1.expectationValue(a1, b1) * p1.integral(a1, b1);
    const double firstMoment2 = p2.expectationValue(a2, b2) * p2.integral(a2, b2);
    constexpr double tol = 1e-4;
    for (std::size_t i = 0; i < m.size(); ++i)
    {
        const double expected = m.at(i) * firstMoment1 * firstMoment2;
        if (std::abs(result.at(i) - expected) > tol * std::abs(expected))
        {
            std::cerr << "testPDFIntegratorND: member-function vectorized 2D case: at i = " << i
                << " expected " << expected << ", got " << result.at(i) << "\n";
            return 1;
        }
    }
    return 0;
}

auto testPDFIntegratorND() -> int
{
    const pdfs::PDF imf = pdfs::parsePDFDescriptor("data/imfs/chabrier.toml");
    const pdfs::PDF salpeter = pdfs::parsePDFDescriptor("data/imfs/salpeter.toml");

    int result = 0;
    result += test1D(imf);
    result += testPlainFunction2D(imf, salpeter);
    result += testMemberFunction2D(imf, salpeter);
    result += testPlainFunctionVec2D(imf, salpeter);
    result += testMemberFunctionVec2D(imf, salpeter);
    return result;
}
