/**
 * @file testDecayChain.hpp
 * @author Mark Krumholz
 * @brief Unit tests for elem::DecayChain
 * @date 2026-09-19
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#ifndef TESTDECAYCHAIN_HPP
#define TESTDECAYCHAIN_HPP

#include "../../src/elem/DecayChain.hpp"
#include "../../src/elem/IsotopeTable.hpp"
#include "../../src/utils/MiscUtils.hpp"
#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <utility>

/**
 * @brief Unit test for DecayChain::products() on the real Ni56 chain
 * @return 0 if the test passes, 1 if it fails.
 * @details
 * Ni56 -> Co56 -> Fe56 (stable) is a simple, non-branching,
 * branching-ratio-1 chain in the real isotope table (see
 * data/tools/elem/build_isotope_table.py's own source data) -- the
 * exact example from DecayChain's own class comment. products() must
 * list all three, in that order.
 */
inline auto testDecayChainProductsNi56() -> int
{
    const auto& ni56 = elem::isotopeTable(28U, 56U);
    const elem::DecayChain chain(ni56);

    const auto& products = chain.products();
    if (products.size() != 3)
    {
        std::cerr << "testDecayChainProductsNi56: products().size() = "
            << products.size() << ", expected 3\n";
        return 1;
    }
    const std::array<std::pair<unsigned int, unsigned int>, 3> expected{{
        {28U, 56U}, {27U, 56U}, {26U, 56U}
    }};
    for (std::size_t i = 0; i < expected.size(); ++i)
    {
        const auto& iso = products[i].get(); // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- i < expected.size() == products.size() by construction
        if (iso.Z() != expected[i].first || iso.A() != expected[i].second) // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- see above
        {
            std::cerr << "testDecayChainProductsNi56: products()[" << i
                << "] = (" << iso.Z() << ", " << iso.A() << "), expected ("
                << expected[i].first << ", " << expected[i].second << ")\n"; // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- see above
            return 1;
        }
    }

    return 0;
}

/**
 * @brief Unit test for DecayChain::yield(0) on the real Ni56 chain
 * @return 0 if the test passes, 1 if it fails.
 * @details
 * At t=0, all of the starting isotope's own atoms are still Ni56, and
 * none have yet become Co56 or Fe56.
 */
inline auto testDecayChainYieldAtZero() -> int
{
    const auto& ni56 = elem::isotopeTable(28U, 56U);
    const elem::DecayChain chain(ni56);

    const auto y = chain.yield(0.0);
    if (y.size() != 3)
    {
        std::cerr << "testDecayChainYieldAtZero: yield(0).size() = "
            << y.size() << ", expected 3\n";
        return 1;
    }
    if (!utils::approxEqual(y[0], 1.0, 1e-12) ||
        !utils::approxEqual(y[1], 0.0, 1e-12) ||
        !utils::approxEqual(y[2], 0.0, 1e-12))
    {
        std::cerr << "testDecayChainYieldAtZero: yield(0) = ["
            << y[0] << ", " << y[1] << ", " << y[2]
            << "], expected [1, 0, 0]\n";
        return 1;
    }

    return 0;
}

/**
 * @brief Unit test for DecayChain::yield(t) against the textbook Bateman formula
 * @return 0 if the test passes, 1 if it fails.
 * @details
 * For the 3-isotope chain N1 -> N2 -> N3 (N3 stable), the closed-form
 * Bateman solution is well known and simple enough to write out
 * directly here, independent of DecayChain's own (more general)
 * implementation:
 *   N1(t) = exp(-l1 t)
 *   N2(t) = l1 / (l2 - l1) * (exp(-l1 t) - exp(-l2 t))
 *   N3(t) = 1 - N1(t) - N2(t)
 * Checked at several multiples of Ni56's own lifetime, using the real
 * Ni56/Co56 lifetimes from the isotope table.
 */
inline auto testDecayChainYieldMatchesBateman() -> int
{
    const auto& ni56 = elem::isotopeTable(28U, 56U);
    const auto& co56 = elem::isotopeTable(27U, 56U);
    const elem::DecayChain chain(ni56);

    const double l1 = 1.0 / ni56.lifetime();
    const double l2 = 1.0 / co56.lifetime();

    for (const double tMult : {0.1, 0.5, 1.0, 2.0, 5.0, 10.0})
    {
        const double t = tMult * ni56.lifetime();
        const auto y = chain.yield(t);

        const double n1 = std::exp(-l1 * t);
        const double n2 = l1 / (l2 - l1) * (std::exp(-l1 * t) - std::exp(-l2 * t));
        const double n3 = 1.0 - n1 - n2;

        if (!utils::approxEqual(y[0], n1, 1e-9) ||
            !utils::approxEqual(y[1], n2, 1e-9) ||
            !utils::approxEqual(y[2], n3, 1e-9))
        {
            std::cerr << "testDecayChainYieldMatchesBateman: at t = " << tMult
                << " * lifetime, yield(t) = [" << y[0] << ", " << y[1] << ", "
                << y[2] << "], expected [" << n1 << ", " << n2 << ", " << n3
                << "]\n";
            return 1;
        }
    }

    return 0;
}

/**
 * @brief Unit test that DecayChain::yield(t) conserves total atom count
 * @return 0 if the test passes, 1 if it fails.
 * @details
 * Every step in the Ni56 -> Co56 -> Fe56 chain has branching ratio 1
 * (no atoms are lost to an untracked decay channel), so the sum of
 * yield(t) over every product must stay exactly 1 for any t -- checked
 * at several widely separated times, including well past both
 * lifetimes.
 */
inline auto testDecayChainMassBalance() -> int
{
    const auto& ni56 = elem::isotopeTable(28U, 56U);
    const elem::DecayChain chain(ni56);

    for (const double t : {0.0, 1e5, 1e6, 1e7, 1e8, 1e9})
    {
        const auto y = chain.yield(t);
        const double total = std::accumulate(y.begin(), y.end(), 0.0);
        if (!utils::approxEqual(total, 1.0, 1e-9))
        {
            std::cerr << "testDecayChainMassBalance: at t = " << t
                << ", sum(yield(t)) = " << total << ", expected 1\n";
            return 1;
        }
    }

    return 0;
}

/**
 * @brief Unit test for DecayChain on an already-stable starting isotope
 * @return 0 if the test passes, 1 if it fails.
 * @details
 * Fe56 (the stable end of the Ni56 chain) has no daughters, so its own
 * chain is just itself, and its abundance should stay exactly 1 for
 * any t (a stable isotope doesn't decay).
 */
inline auto testDecayChainStableIsotope() -> int
{
    const auto& fe56 = elem::isotopeTable(26U, 56U);
    const elem::DecayChain chain(fe56);

    const auto& products = chain.products();
    if (products.size() != 1 || products[0].get().Z() != 26U || products[0].get().A() != 56U)
    {
        std::cerr << "testDecayChainStableIsotope: products() should contain "
            "only Fe56 itself\n";
        return 1;
    }

    for (const double t : {0.0, 1.0, 1e10})
    {
        const auto y = chain.yield(t);
        if (y.size() != 1 || !utils::approxEqual(y[0], 1.0, 1e-12))
        {
            std::cerr << "testDecayChainStableIsotope: yield(" << t
                << ") = " << (y.empty() ? -1.0 : y[0]) << ", expected [1]\n";
            return 1;
        }
    }

    return 0;
}

/**
 * @brief Unit test that DecayChain rejects a daughter absent from the isotope table
 * @return 0 if the test passes, 1 if it fails.
 * @details
 * Builds a standalone IsotopeData (not part of the global isotope
 * table) whose own daughter names a (Z, A) that doesn't exist in the
 * real table -- DecayChain must throw std::runtime_error rather than
 * letting IsotopeTable's own std::out_of_range escape uncaught.
 */
inline auto testDecayChainUnknownDaughterThrows() -> int
{
    const elem::IsotopeData fake(
        {'X', '\0'}, 999U, 999U, 1.0, {{998U, 998U, 1.0}});

    try
    {
        const elem::DecayChain chain(fake);
        std::cerr << "testDecayChainUnknownDaughterThrows: expected "
            "construction to throw\n";
        return 1;
    }
    catch (const std::runtime_error&) { /* expected */ }

    return 0;
}

/**
 * @brief Unit tests for elem::DecayChain
 * @return 0 if the test passes, 1 if it fails.
 */
inline auto testDecayChain() -> int
{
    int result = 0;
    result += testDecayChainProductsNi56();
    result += testDecayChainYieldAtZero();
    result += testDecayChainYieldMatchesBateman();
    result += testDecayChainMassBalance();
    result += testDecayChainStableIsotope();
    result += testDecayChainUnknownDaughterThrows();
    return result;
}

#endif // TESTDECAYCHAIN_HPP
