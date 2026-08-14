/**
 * @file testPDFReflect.hpp
 * @author Mark Krumholz
 * @brief Unit tests for the PDFReflect class.
 * @date 2026-08-14
 */

#ifndef TESTPDFREFLECT_HPP
#define TESTPDFREFLECT_HPP

/**
 * @brief Unit test for the PDFReflect class.
 * @return 0 if the test passes, 1 if it fails.
 * @details
 * Wraps a single-segment power-law PDF in a PDFReflect and checks
 * operator()/expectationValue()/integral()/draw()/drawTarget() against
 * independently-computed analytic values, both calling the PDFReflect
 * object directly and through a `const PDF&` reference -- the latter
 * is essential, since PDFIntegratorND (the motivating use case for
 * this class) only ever holds PDFs by `const PDF&`, so every method
 * PDFReflect overrides must actually dispatch virtually.
 */
auto testPDFReflect() -> int;

#endif // TESTPDFREFLECT_HPP
