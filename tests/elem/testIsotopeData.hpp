/**
 * @file testIsotopeData.hpp
 * @author Mark Krumholz
 * @brief Unit tests for elem::IsotopeData
 * @date 2026-08-31
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#ifndef TESTISOTOPEDATA_HPP
#define TESTISOTOPEDATA_HPP

#include "../../src/elem/IsotopeData.hpp"
#include <iostream>
#include <stdexcept>

/**
 * @brief Unit test for IsotopeData construction with default lifetime/daughters
 * @return 0 if the test passes, 1 if it fails.
 * @details
 * Omitting lifetime and daughters should default to a stable isotope
 * (lifetime() == 0, stable() == true) with an empty daughters() list.
 */
inline auto testIsotopeDataConstructionStable() -> int
{
    const elem::IsotopeData h1({'H', '\0'}, 1U, 1U);

    if (h1.Z() != 1U || h1.A() != 1U)
    {
        std::cerr << "testIsotopeDataConstructionStable: Z()/A() are ("
            << h1.Z() << ", " << h1.A() << "), expected (1, 1)\n";
        return 1;
    }
    if (h1.lifetime() != 0.0)
    {
        std::cerr << "testIsotopeDataConstructionStable: lifetime() = "
            << h1.lifetime() << ", expected 0 (default)\n";
        return 1;
    }
    if (!h1.stable())
    {
        std::cerr << "testIsotopeDataConstructionStable: stable() should be "
            "true when lifetime_ == 0\n";
        return 1;
    }
    if (!h1.daughters().empty())
    {
        std::cerr << "testIsotopeDataConstructionStable: daughters() should "
            "be empty by default\n";
        return 1;
    }

    return 0;
}

/**
 * @brief Unit test for IsotopeData construction with an explicit lifetime and daughters
 * @return 0 if the test passes, 1 if it fails.
 * @details
 * Modeled loosely on carbon-14's beta decay to nitrogen-14 (the
 * lifetime is illustrative, not necessarily the real physical value)
 * -- checks that lifetime(), stable(), and daughters() all report
 * back what was passed to the constructor.
 */
inline auto testIsotopeDataConstructionUnstable() -> int
{
    constexpr double lifetime = 8267.0; // years, illustrative
    const elem::IsotopeData c14({'C', '\0'}, 6U, 14U, lifetime,
        {{7U, 14U, 1.0}});

    if (c14.lifetime() != lifetime)
    {
        std::cerr << "testIsotopeDataConstructionUnstable: lifetime() = "
            << c14.lifetime() << ", expected " << lifetime << "\n";
        return 1;
    }
    if (c14.stable())
    {
        std::cerr << "testIsotopeDataConstructionUnstable: stable() should "
            "be false when lifetime_ > 0\n";
        return 1;
    }
    if (c14.daughters().size() != 1)
    {
        std::cerr << "testIsotopeDataConstructionUnstable: daughters().size() "
            "= " << c14.daughters().size() << ", expected 1\n";
        return 1;
    }
    const auto& d = c14.daughters()[0];
    if (d.Z_ != 7U || d.A_ != 14U || d.branchingRatio_ != 1.0)
    {
        std::cerr << "testIsotopeDataConstructionUnstable: daughter is ("
            << d.Z_ << ", " << d.A_ << ", " << d.branchingRatio_
            << "), expected (7, 14, 1.0)\n";
        return 1;
    }

    return 0;
}

/**
 * @brief Unit test for IsotopeData's constructor validation
 * @return 0 if the test passes, 1 if it fails.
 * @details
 * A negative lifetime, or a positive lifetime with no daughters,
 * should both throw std::runtime_error. A stable isotope (lifetime ==
 * 0) is allowed to carry a nonempty daughters() list -- the
 * constructor only requires daughters when lifetime is strictly
 * positive -- so that case is checked as well, and should not throw.
 */
inline auto testIsotopeDataValidation() -> int
{
    try
    {
        const elem::IsotopeData bad({'X', 'X'}, 99U, 1U, -1.0);
        std::cerr << "testIsotopeDataValidation: negative lifetime should "
            "have thrown\n";
        return 1;
    }
    catch (const std::runtime_error&) { /* expected */ }

    try
    {
        const elem::IsotopeData bad({'X', 'X'}, 99U, 1U, 1.0);
        std::cerr << "testIsotopeDataValidation: positive lifetime with no "
            "daughters should have thrown\n";
        return 1;
    }
    catch (const std::runtime_error&) { /* expected */ }

    try
    {
        const elem::IsotopeData ok({'X', 'X'}, 99U, 1U, 0.0, {{98U, 1U, 1.0}});
        if (!ok.stable())
        {
            std::cerr << "testIsotopeDataValidation: lifetime == 0 with "
                "nonempty daughters should still report stable() == true\n";
            return 1;
        }
    }
    catch (const std::runtime_error& e)
    {
        std::cerr << "testIsotopeDataValidation: lifetime == 0 with nonempty "
            "daughters should not throw, but got: " << e.what() << "\n";
        return 1;
    }

    return 0;
}

/**
 * @brief Unit test for IsotopeData's (Z, A) comparison operators
 * @return 0 if the test passes, 1 if it fails.
 * @details
 * Unlike ElemData/IonizationData, IsotopeData compares on (Z, A): Z
 * takes priority, with A only as a tiebreaker when Z matches. Checks
 * that equality/inequality require both Z and A to match, that
 * ordering uses A to break ties between isotopes of the same element,
 * and that a difference in Z outranks any difference in A (even when
 * A would suggest the opposite order).
 */
inline auto testIsotopeDataComparisons() -> int
{
    const elem::IsotopeData c12({'C', '\0'}, 6U, 12U);
    const elem::IsotopeData c14({'C', '\0'}, 6U, 14U, 8267.0, {{7U, 14U, 1.0}});
    const elem::IsotopeData c12Dup({'C', '\0'}, 6U, 12U);
    const elem::IsotopeData n13({'N', '\0'}, 7U, 13U); // Z=7 > 6, but A=13 < 14

    if (!(c12 == c12Dup) || (c12 != c12Dup))
    {
        std::cerr << "testIsotopeDataComparisons: isotopes with matching Z "
            "and A should compare equal\n";
        return 1;
    }
    if ((c12 == c14) || !(c12 != c14))
    {
        std::cerr << "testIsotopeDataComparisons: isotopes with the same Z "
            "but different A should not compare equal\n";
        return 1;
    }
    if (!(c12 < c14) || !(c14 > c12))
    {
        std::cerr << "testIsotopeDataComparisons: same Z, so A should break "
            "the tie: c12 (A=12) should be < c14 (A=14)\n";
        return 1;
    }
    if (!(c14 < n13) || !(n13 > c14))
    {
        std::cerr << "testIsotopeDataComparisons: Z should outrank A: c14 "
            "(Z=6, A=14) should be < n13 (Z=7, A=13) despite A going the "
            "other way\n";
        return 1;
    }
    if (!(c12 <= c12Dup) || !(c12 >= c12Dup))
    {
        std::cerr << "testIsotopeDataComparisons: equal isotopes should "
            "satisfy both <= and >=\n";
        return 1;
    }

    return 0;
}

/**
 * @brief Unit tests for elem::IsotopeData
 * @return 0 if the test passes, 1 if it fails.
 */
inline auto testIsotopeData() -> int
{
    int result = 0;
    result += testIsotopeDataConstructionStable();
    result += testIsotopeDataConstructionUnstable();
    result += testIsotopeDataValidation();
    result += testIsotopeDataComparisons();
    return result;
}

#endif // TESTISOTOPEDATA_HPP
