/**
 * @file testElemData.hpp
 * @author Mark Krumholz
 * @brief Unit tests for elem::ElemData
 * @date 2026-08-31
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#ifndef TESTELEMDATA_HPP
#define TESTELEMDATA_HPP

#include "../../src/elem/ElemData.hpp"
#include <iostream>

namespace
{
    // A static_assert (rather than a runtime check) that ElemData's
    // constructor, accessors, and comparison operators are actually
    // usable in a constant expression -- proof that they're genuinely
    // constexpr, not just callable at runtime despite being marked so
    constexpr elem::ElemData testFe({'F', 'e'}, 26U);
    static_assert(testFe.Z() == 26U);
    static_assert(testFe.symbol()[0] == 'F' && testFe.symbol()[1] == 'e');
    static_assert(elem::ElemData({'H', '\0'}, 1U) < elem::ElemData({'H', 'e'}, 2U));
    static_assert(elem::ElemData({'H', '\0'}, 1U) == elem::ElemData({'H', '\0'}, 1U));
} // namespace

/**
 * @brief Unit test for ElemData construction and its symbol()/Z() accessors
 * @return 0 if the test passes, 1 if it fails.
 */
inline auto testElemDataConstruction() -> int
{
    const elem::ElemData fe({'F', 'e'}, 26U);
    if (fe.Z() != 26U)
    {
        std::cerr << "testElemDataConstruction: Z() returned " << fe.Z()
            << ", expected 26\n";
        return 1;
    }
    if (fe.symbol()[0] != 'F' || fe.symbol()[1] != 'e')
    {
        std::cerr << "testElemDataConstruction: symbol() returned {'"
            << fe.symbol()[0] << "', '" << fe.symbol()[1]
            << "'}, expected {'F', 'e'}\n";
        return 1;
    }

    // Single-character symbols pad the second slot with '\0'
    const elem::ElemData h({'H', '\0'}, 1U);
    if (h.symbol()[1] != '\0')
    {
        std::cerr << "testElemDataConstruction: single-character symbol's "
            "second slot should be '\\0', got '" << h.symbol()[1] << "'\n";
        return 1;
    }

    return 0;
}

/**
 * @brief Unit test for ElemData's comparison operators
 * @return 0 if the test passes, 1 if it fails.
 * @details
 * ElemData compares purely on Z, regardless of symbol, so two records
 * with different symbols but the same Z should compare equal.
 */
inline auto testElemDataComparisons() -> int
{
    const elem::ElemData h({'H', '\0'}, 1U);
    const elem::ElemData he({'H', 'e'}, 2U);
    const elem::ElemData hDup({'X', 'X'}, 1U); // same Z as h, different symbol

    if (!(h == hDup) || (h != hDup))
    {
        std::cerr << "testElemDataComparisons: records with equal Z but "
            "different symbols should compare equal\n";
        return 1;
    }
    if (!(h != he) || (h == he))
    {
        std::cerr << "testElemDataComparisons: records with different Z "
            "should not compare equal\n";
        return 1;
    }
    if (!(h < he))
    {
        std::cerr << "testElemDataComparisons: h (Z=1) should be < he (Z=2)\n";
        return 1;
    }
    if (!(he > h))
    {
        std::cerr << "testElemDataComparisons: he (Z=2) should be > h (Z=1)\n";
        return 1;
    }
    if (!(h <= hDup) || !(h >= hDup))
    {
        std::cerr << "testElemDataComparisons: records with equal Z should "
            "satisfy both <= and >=\n";
        return 1;
    }
    if (!(h <= he) || (he <= h))
    {
        std::cerr << "testElemDataComparisons: h <= he should hold and "
            "he <= h should not\n";
        return 1;
    }
    if (!(he >= h) || (h >= he))
    {
        std::cerr << "testElemDataComparisons: he >= h should hold and "
            "h >= he should not\n";
        return 1;
    }

    return 0;
}

/**
 * @brief Unit tests for elem::ElemData
 * @return 0 if the test passes, 1 if it fails.
 */
inline auto testElemData() -> int
{
    int result = 0;
    result += testElemDataConstruction();
    result += testElemDataComparisons();
    return result;
}

#endif // TESTELEMDATA_HPP
