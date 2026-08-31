/**
 * @file testIsotopeTable.hpp
 * @author Mark Krumholz
 * @brief Unit tests for elem::IsotopeTable
 * @date 2026-08-31
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#ifndef TESTISOTOPETABLE_HPP
#define TESTISOTOPETABLE_HPP

#include "../../src/elem/IsotopeTable.hpp"
#include "../../src/utils/MiscUtils.hpp"
#include <iostream>
#include <stdexcept>
#include <utility>

/**
 * @brief Unit test for IsotopeTable's default-path construction and overall shape
 * @return 0 if the test passes, 1 if it fails.
 * @details
 * Checks that the default-constructed table (reading data/elem/
 * isotopes.h5) has exactly the 3183 entries in slug2's own
 * lib/yields/isotope_data.txt (see data/tools/elem/
 * build_isotope_table.py, which built isotopes.h5 from that file),
 * and that table() and size() agree.
 */
inline auto testIsotopeTableConstruction() -> int
{
    const elem::IsotopeTable isotopes;

    constexpr std::size_t expectedSize = 3183;
    if (isotopes.size() != expectedSize)
    {
        std::cerr << "testIsotopeTableConstruction: size() = " << isotopes.size()
            << ", expected " << expectedSize << "\n";
        return 1;
    }
    if (isotopes.table().size() != isotopes.size())
    {
        std::cerr << "testIsotopeTableConstruction: table().size() = "
            << isotopes.table().size() << " does not match size() = "
            << isotopes.size() << "\n";
        return 1;
    }

    return 0;
}

/**
 * @brief Unit test for a stable isotope's entry in the table
 * @return 0 if the test passes, 1 if it fails.
 * @details
 * Hydrogen-1 (Z=1, A=1) is stable in slug2's own data (lifetime -1,
 * translated to elem::IsotopeData's 0-means-stable convention).
 */
inline auto testIsotopeTableStableEntry() -> int
{
    const elem::IsotopeTable isotopes;

    const auto it = isotopes.table().find({1U, 1U});
    if (it == isotopes.table().end())
    {
        std::cerr << "testIsotopeTableStableEntry: H-1 (Z=1, A=1) not found "
            "in the table\n";
        return 1;
    }
    const auto& h1 = it->second;
    if (h1.Z() != 1U || h1.A() != 1U)
    {
        std::cerr << "testIsotopeTableStableEntry: entry at key (1, 1) has "
            "Z()/A() = (" << h1.Z() << ", " << h1.A() << ")\n";
        return 1;
    }
    if (!h1.stable())
    {
        std::cerr << "testIsotopeTableStableEntry: H-1 should be stable\n";
        return 1;
    }
    if (h1.symbol()[0] != 'H' || h1.symbol()[1] != '\0')
    {
        std::cerr << "testIsotopeTableStableEntry: H-1's symbol should be "
            "{'H', '\\0'}, got {'" << h1.symbol()[0] << "', '"
            << h1.symbol()[1] << "'}\n";
        return 1;
    }

    return 0;
}

/**
 * @brief Unit test for an unstable isotope with a tabulated decay channel
 * @return 0 if the test passes, 1 if it fails.
 * @details
 * Tritium (Z=1, A=3) beta-decays to helium-3 (Z=2, A=3) with branching
 * ratio 1, and a mean lifetime of 5.605e8 s in slug2's own data --
 * this checks IsotopeTable reproduces both the lifetime and the
 * daughter list exactly.
 */
inline auto testIsotopeTableUnstableEntry() -> int
{
    const elem::IsotopeTable isotopes;

    const auto it = isotopes.table().find({1U, 3U});
    if (it == isotopes.table().end())
    {
        std::cerr << "testIsotopeTableUnstableEntry: H-3 (Z=1, A=3) not "
            "found in the table\n";
        return 1;
    }
    const auto& h3 = it->second;
    if (h3.stable())
    {
        std::cerr << "testIsotopeTableUnstableEntry: H-3 (tritium) should "
            "not be stable\n";
        return 1;
    }
    if (!utils::approxEqual(h3.lifetime(), 5.605e8, 1e3))
    {
        std::cerr << "testIsotopeTableUnstableEntry: H-3 lifetime() = "
            << h3.lifetime() << ", expected 5.605e8 s\n";
        return 1;
    }
    if (h3.daughters().size() != 1)
    {
        std::cerr << "testIsotopeTableUnstableEntry: H-3 daughters().size() "
            "= " << h3.daughters().size() << ", expected 1\n";
        return 1;
    }
    const auto& daughter = h3.daughters()[0];
    if (daughter.Z_ != 2U || daughter.A_ != 3U || daughter.branchingRatio_ != 1.0)
    {
        std::cerr << "testIsotopeTableUnstableEntry: H-3's daughter is ("
            << daughter.Z_ << ", " << daughter.A_ << ", "
            << daughter.branchingRatio_ << "), expected (2, 3, 1.0) "
            "(He-3, branching ratio 1)\n";
        return 1;
    }

    return 0;
}

/**
 * @brief Unit test for an unstable isotope with no tabulated decay channel
 * @return 0 if the test passes, 1 if it fails.
 * @details
 * Hydrogen-6 (Z=1, A=6) is unstable (lifetime 4.18e-22 s) in slug2's
 * own data, but that data lists no daughter for it -- this is exactly
 * the case the relaxed IsotopeData constructor validation exists for
 * (see IsotopeData.hpp's own constructor comment and
 * testIsotopeData.hpp's testIsotopeDataValidation), so this is a
 * regression test that the real data file actually round-trips
 * through IsotopeTable without IsotopeTable's own construction (or
 * IsotopeData's constructor) throwing.
 */
inline auto testIsotopeTableUnstableNoDaughterEntry() -> int
{
    const elem::IsotopeTable isotopes;

    const auto it = isotopes.table().find({1U, 6U});
    if (it == isotopes.table().end())
    {
        std::cerr << "testIsotopeTableUnstableNoDaughterEntry: H-6 (Z=1, "
            "A=6) not found in the table\n";
        return 1;
    }
    const auto& h6 = it->second;
    if (h6.stable())
    {
        std::cerr << "testIsotopeTableUnstableNoDaughterEntry: H-6 should "
            "not be stable\n";
        return 1;
    }
    if (!h6.daughters().empty())
    {
        std::cerr << "testIsotopeTableUnstableNoDaughterEntry: H-6 should "
            "have no tabulated daughters, got "
            << h6.daughters().size() << "\n";
        return 1;
    }

    return 0;
}

/**
 * @brief Unit test for elem::isotopeTable()'s global-singleton behavior
 * @return 0 if the test passes, 1 if it fails.
 * @details
 * isotopeTable() is a function-local static (mirroring utils::rng()
 * in RngThread.hpp), so every call anywhere in the program must
 * return a reference to the exact same instance -- checked here both
 * by address and by content (looking up the same H-1 entry through
 * both references).
 */
inline auto testIsotopeTableGlobalSingleton() -> int
{
    auto& first = elem::isotopeTable();
    auto& second = elem::isotopeTable();

    if (&first != &second)
    {
        std::cerr << "testIsotopeTableGlobalSingleton: isotopeTable() "
            "returned two different instances\n";
        return 1;
    }
    if (first.size() != second.size())
    {
        std::cerr << "testIsotopeTableGlobalSingleton: test bug: two "
            "references to the same instance disagree on size()\n";
        return 1;
    }

    const auto it = first.table().find({1U, 1U});
    if (it == first.table().end() || !it->second.stable())
    {
        std::cerr << "testIsotopeTableGlobalSingleton: global instance is "
            "missing the expected stable H-1 entry\n";
        return 1;
    }

    return 0;
}

/**
 * @brief Unit test for the isotopeTable(z, a) lookup shorthand
 * @return 0 if the test passes, 1 if it fails.
 * @details
 * Checks isotopeTable(z, a) against isotopeTable().table().at({z, a})
 * for a present (Z, A) pair (tritium, reusing
 * testIsotopeTableUnstableEntry's own H-3 reference values), and that
 * it throws std::out_of_range for a (Z, A) pair with no entry in the
 * table (Z=0 is not a valid atomic number, so it can never collide
 * with a real isotope).
 */
inline auto testIsotopeTableLookup() -> int
{
    const auto& h3 = elem::isotopeTable(1U, 3U);
    const auto& h3Ref = elem::isotopeTable().table().at({1U, 3U});
    if (&h3 != &h3Ref)
    {
        std::cerr << "testIsotopeTableLookup: isotopeTable(1, 3) did not "
            "return the same record as isotopeTable().table().at({1, 3})\n";
        return 1;
    }
    if (h3.Z() != 1U || h3.A() != 3U)
    {
        std::cerr << "testIsotopeTableLookup: isotopeTable(1, 3) has Z()/A() "
            "= (" << h3.Z() << ", " << h3.A() << "), expected (1, 3)\n";
        return 1;
    }

    try
    {
        (void)elem::isotopeTable(0U, 0U);
        std::cerr << "testIsotopeTableLookup: isotopeTable(0, 0) should "
            "have thrown (no such isotope)\n";
        return 1;
    }
    catch (const std::out_of_range&) { /* expected */ }

    return 0;
}

/**
 * @brief Unit tests for elem::IsotopeTable
 * @return 0 if the test passes, 1 if it fails.
 */
inline auto testIsotopeTable() -> int
{
    int result = 0;
    result += testIsotopeTableConstruction();
    result += testIsotopeTableStableEntry();
    result += testIsotopeTableUnstableEntry();
    result += testIsotopeTableUnstableNoDaughterEntry();
    result += testIsotopeTableGlobalSingleton();
    result += testIsotopeTableLookup();
    return result;
}

#endif // TESTISOTOPETABLE_HPP
