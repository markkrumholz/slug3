/**
 * @file testIonizationData.hpp
 * @author Mark Krumholz
 * @brief Unit tests for elem::IonizationData and the elem::ionizationData table
 * @date 2026-08-31
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#ifndef TESTIONIZATIONDATA_HPP
#define TESTIONIZATIONDATA_HPP

#include "../../src/elem/ElemCommons.hpp"
#include "../../src/elem/IonizationData.hpp"
#include <cmath>
#include <cstddef>
#include <iostream>

namespace
{
    // Proof that IonizationData's constructor and accessors -- symbol()/
    // Z() inherited from ElemData, plus ionPot() -- are genuinely usable
    // in a constant expression, matching the class's own doc comment
    constexpr elem::IonizationData testH({'H', '\0'}, 1U,
        elem::detail::makeIonPot({13.59844}));
    static_assert(testH.Z() == 1U);
    static_assert(testH.ionPot()[0] == 13.59844);
} // namespace

/**
 * @brief Unit test for IonizationData construction and its accessors
 * @return 0 if the test passes, 1 if it fails.
 * @details
 * Checks that symbol()/Z() (inherited from ElemData) and ionPot() all
 * report back exactly what was passed to the constructor, and that
 * ionPot() entries beyond the number of values supplied are padded
 * with quiet_NaN(), per detail::makeIonPot's own contract.
 */
inline auto testIonizationDataConstruction() -> int
{
    const elem::IonizationData c({'C', '\0'}, 6U,
        elem::detail::makeIonPot({11.26030, 24.38332, 47.8878}));

    if (c.Z() != 6U)
    {
        std::cerr << "testIonizationDataConstruction: Z() returned " << c.Z()
            << ", expected 6\n";
        return 1;
    }
    if (c.symbol()[0] != 'C' || c.symbol()[1] != '\0')
    {
        std::cerr << "testIonizationDataConstruction: symbol() returned {'"
            << c.symbol()[0] << "', '" << c.symbol()[1]
            << "'}, expected {'C', '\\0'}\n";
        return 1;
    }
    if (c.ionPot()[0] != 11.26030 || c.ionPot()[1] != 24.38332 || c.ionPot()[2] != 47.8878)
    {
        std::cerr << "testIonizationDataConstruction: ionPot()[0..2] did not "
            "match the values passed to the constructor\n";
        return 1;
    }
    for (std::size_t i = 3; i < elem::maxIP; ++i)
    {
        if (!std::isnan(c.ionPot()[i]))
        {
            std::cerr << "testIonizationDataConstruction: ionPot()[" << i
                << "] should be padded with NaN beyond the 3 supplied "
                "values, got " << c.ionPot()[i] << "\n";
            return 1;
        }
    }

    return 0;
}

/**
 * @brief Unit test for IonizationData's inherited comparison operators
 * @return 0 if the test passes, 1 if it fails.
 * @details
 * IonizationData inherits ElemData's comparison operators unchanged,
 * so two records should compare purely on Z, ignoring their (possibly
 * quite different) ionPot() data.
 */
inline auto testIonizationDataComparisons() -> int
{
    const elem::IonizationData h({'H', '\0'}, 1U,
        elem::detail::makeIonPot({13.59844}));
    const elem::IonizationData he({'H', 'e'}, 2U,
        elem::detail::makeIonPot({24.58738, 54.41776}));
    const elem::IonizationData hDup({'H', '\0'}, 1U, elem::detail::makeEmptyIonPot());

    if (!(h == hDup))
    {
        std::cerr << "testIonizationDataComparisons: records with equal Z "
            "should compare equal even with different ionPot() data\n";
        return 1;
    }
    if (!(h < he) || !(he > h))
    {
        std::cerr << "testIonizationDataComparisons: h (Z=1) should be < he "
            "(Z=2) and he should be > h\n";
        return 1;
    }

    return 0;
}

/**
 * @brief Sanity check on the compile-time elem::ionizationData table
 * @return 0 if the test passes, 1 if it fails.
 * @details
 * Checks the table's size against Symbols::nElem, that it is sorted
 * by (and indexed by) atomic number Z starting from 1, and spot-checks
 * hydrogen's symbol and ionization potential against its CRC reference
 * value.
 */
inline auto testIonizationDataTable() -> int
{
    if (elem::ionizationData.size() != static_cast<std::size_t>(elem::Symbols::nElem))
    {
        std::cerr << "testIonizationDataTable: ionizationData.size() = "
            << elem::ionizationData.size() << ", expected "
            << static_cast<std::size_t>(elem::Symbols::nElem) << "\n";
        return 1;
    }

    for (std::size_t i = 0; i < elem::ionizationData.size(); ++i)
    {
        const auto expectedZ = static_cast<unsigned int>(i + 1);
        if (elem::ionizationData[i].Z() != expectedZ)
        {
            std::cerr << "testIonizationDataTable: ionizationData[" << i
                << "].Z() = " << elem::ionizationData[i].Z() << ", expected "
                << expectedZ << " (table should be indexed by Z-1)\n";
            return 1;
        }
    }

    // Spot-check: hydrogen's single ionization potential
    if (elem::ionizationData[0].symbol()[0] != 'H' || elem::ionizationData[0].symbol()[1] != '\0')
    {
        std::cerr << "testIonizationDataTable: ionizationData[0] is not hydrogen\n";
        return 1;
    }
    if (elem::ionizationData[0].ionPot()[0] != 13.59844)
    {
        std::cerr << "testIonizationDataTable: hydrogen's first ionization "
            "potential is " << elem::ionizationData[0].ionPot()[0]
            << ", expected 13.59844 eV\n";
        return 1;
    }

    return 0;
}

/**
 * @brief Unit tests for elem::IonizationData and the elem::ionizationData table
 * @return 0 if the test passes, 1 if it fails.
 */
inline auto testIonizationData() -> int
{
    int result = 0;
    result += testIonizationDataConstruction();
    result += testIonizationDataComparisons();
    result += testIonizationDataTable();
    return result;
}

#endif // TESTIONIZATIONDATA_HPP
