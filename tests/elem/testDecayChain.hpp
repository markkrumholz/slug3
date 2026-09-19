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
#include "../../src/elem/ElemCommons.hpp"
#include "../../src/elem/IsotopeTable.hpp"
#include "../../src/utils/MiscUtils.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

/**
 * @brief Unit test for a simple, non-branching, two-step chain: Ni56 -> Co56 -> Fe56
 * @return 0 if the test passes, 1 if it fails.
 * @details
 * Cross-checks DecayChain::applyDecay() against the same independent
 * closed-form (textbook) two-step Bateman solution the old, per-isotope
 * Bateman-equation implementation of this class was checked against --
 * the new matrix-exponential solver must reproduce identical physics
 * for a chain simple enough that the closed form applies directly.
 * isotopes_ is laid out in Z-then-A order (Fe56, Co56, Ni56), matching
 * Yields::isotopes_'s own convention, not the decay order itself --
 * DecayChain no longer cares about "chain order" at all.
 */
inline auto testDecayChainNi56Co56Fe56() -> int
{
    const auto& fe56 = elem::isotopeTable(26U, 56U);
    const auto& co56 = elem::isotopeTable(27U, 56U);
    const auto& ni56 = elem::isotopeTable(28U, 56U);
    const elem::IsotopeList isotopes{ fe56, co56, ni56 };
    const elem::DecayChain chain(isotopes);

    const double l1 = 1.0 / ni56.lifetime();
    const double l2 = 1.0 / co56.lifetime();

    for (const double tMult : {0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0})
    {
        const double t = tMult * ni56.lifetime();
        std::vector<double> values{ 0.0, 0.0, 1.0 }; // all mass starts as Ni56
        chain.applyDecay(t, values);

        const double n1 = std::exp(-l1 * t);
        const double n2 = l1 / (l2 - l1) * (std::exp(-l1 * t) - std::exp(-l2 * t));
        const double n3 = 1.0 - n1 - n2;

        if (!utils::approxEqual(values[2], n1, 1e-9) ||
            !utils::approxEqual(values[1], n2, 1e-9) ||
            !utils::approxEqual(values[0], n3, 1e-9))
        {
            std::cerr << "testDecayChainNi56Co56Fe56: at t = " << tMult <<
                " * lifetime, values = [" << values[0] << ", " << values[1] << ", " <<
                values[2] << "], expected [" << n3 << ", " << n2 << ", " << n1 << "]\n";
            return 1;
        }
    }

    return 0;
}

/**
 * @brief Unit test for one parent with multiple decay modes into the same daughter
 * @return 0 if the test passes, 1 if it fails.
 * @details
 * Mn54's own real daughters() lists Cr54 twice (its dominant electron
 * capture mode, and a much rarer one) and Fe54 once -- exactly the
 * pattern that broke the old closed-form Bateman implementation's
 * per-isotope-index assumption (see DecayChain.cpp's own git history).
 * The matrix solver needs no special-casing for this at all: both Cr54
 * entries simply add into the same depletion-matrix element. Since Cr54
 * and Fe54 are both stable, this reduces to a simple, independently
 * verifiable closed form: N_Mn(t) = exp(-t/L), and each stable
 * daughter's own abundance grows as its own (summed) branching ratio
 * times (1 - exp(-t/L)).
 */
inline auto testDecayChainMn54MultipleModes() -> int
{
    const auto& cr54 = elem::isotopeTable(24U, 54U);
    const auto& mn54 = elem::isotopeTable(25U, 54U);
    const auto& fe54 = elem::isotopeTable(26U, 54U);
    const elem::IsotopeList isotopes{ cr54, mn54, fe54 };
    const elem::DecayChain chain(isotopes);

    double branchToCr54 = 0.0;
    double branchToFe54 = 0.0;
    for (const auto& daughter : mn54.daughters())
    {
        if (daughter.Z_ == 24U && daughter.A_ == 54U) { branchToCr54 += daughter.branchingRatio_; }
        if (daughter.Z_ == 26U && daughter.A_ == 54U) { branchToFe54 += daughter.branchingRatio_; }
    }
    if (mn54.daughters().size() < 2)
    {
        std::cerr << "testDecayChainMn54MultipleModes: test bug: expected Mn54 to list "
            "more than one decay mode in the real isotope table, got " <<
            mn54.daughters().size() << "\n";
        return 1;
    }

    const double lMn = 1.0 / mn54.lifetime();
    for (const double tMult : {0.0, 0.5, 1.0, 5.0})
    {
        const double t = tMult * mn54.lifetime();
        std::vector<double> values{ 0.0, 1.0, 0.0 }; // all mass starts as Mn54
        chain.applyDecay(t, values);

        const double nMn = std::exp(-lMn * t);
        const double nCr = branchToCr54 * (1.0 - nMn);
        const double nFe = branchToFe54 * (1.0 - nMn);

        if (!utils::approxEqual(values[1], nMn, 1e-9) ||
            !utils::approxEqual(values[0], nCr, 1e-9) ||
            !utils::approxEqual(values[2], nFe, 1e-9))
        {
            std::cerr << "testDecayChainMn54MultipleModes: at t = " << tMult <<
                " * lifetime, values = [" << values[0] << ", " << values[1] << ", " <<
                values[2] << "], expected [" << nCr << ", " << nMn << ", " << nFe << "]\n";
            return 1;
        }
    }

    return 0;
}

/**
 * @brief Unit test for a converging network: the same daughter produced at two different chain steps
 * @return 0 if the test passes, 1 if it fails.
 * @details
 * Sm148 alpha-decays to Nd144 (releasing one He4 nucleus per decay),
 * and Nd144 itself alpha-decays to Ce140 (releasing another He4). He4
 * is therefore produced by two different parents in the same chain --
 * a genuinely converging network the old, tree-only closed-form
 * Bateman implementation could not represent at all (see this class's
 * own header comment). Cross-checked against an independent analytic
 * solution of the resulting 4-isotope linear ODE system, derived
 * directly (not through this class): with l1 = 1/L(Sm148), l2 =
 * 1/L(Nd144),
 *
 *   Sm(t) = exp(-l1 t)
 *   Nd(t) = l1 / (l2 - l1) * (exp(-l1 t) - exp(-l2 t))   [same 2-step
 *       Bateman form as the Ni56 test above]
 *   Ce(t) = 1 - Sm(t) - Nd(t)                             [Sm+Nd+Ce is
 *       conserved along the heavy-nucleus chain, ignoring He4]
 *   He(t) = 2 - Sm(t) - [l2 * exp(-l1 t) - l1 * exp(-l2 t)] / (l2 - l1)
 *       [from integrating dHe/dt = l1 Sm(t) + l2 Nd(t) directly; He(0)
 *       = 0 and He(t -> infinity) = 2, one alpha particle from each of
 *       the two decay steps, as expected]
 */
inline auto testDecayChainSm148ConvergingNetwork() -> int
{
    const auto& he4 = elem::isotopeTable(2U, 4U);
    const auto& ce140 = elem::isotopeTable(58U, 140U);
    const auto& nd144 = elem::isotopeTable(60U, 144U);
    const auto& sm148 = elem::isotopeTable(62U, 148U);
    const elem::IsotopeList isotopes{ he4, ce140, nd144, sm148 };
    const elem::DecayChain chain(isotopes);

    const double l1 = 1.0 / sm148.lifetime();
    const double l2 = 1.0 / nd144.lifetime();

    for (const double tMult : {0.0, 0.5, 1.0, 2.0, 10.0})
    {
        const double t = tMult * sm148.lifetime();
        std::vector<double> values{ 0.0, 0.0, 0.0, 1.0 }; // all mass starts as Sm148
        chain.applyDecay(t, values);

        const double sm = std::exp(-l1 * t);
        const double nd = l1 / (l2 - l1) * (std::exp(-l1 * t) - std::exp(-l2 * t));
        const double ce = 1.0 - sm - nd;
        const double heExpected = 2.0 - sm - ((l2 * std::exp(-l1 * t)) - (l1 * std::exp(-l2 * t))) / (l2 - l1);

        if (!utils::approxEqual(values[3], sm, 1e-9) ||
            !utils::approxEqual(values[2], nd, 1e-9) ||
            !utils::approxEqual(values[1], ce, 1e-9) ||
            !utils::approxEqual(values[0], heExpected, 1e-9))
        {
            std::cerr << "testDecayChainSm148ConvergingNetwork: at t = " << tMult <<
                " * lifetime, values = [" << values[0] << ", " << values[1] << ", " <<
                values[2] << ", " << values[3] << "], expected [" << heExpected << ", " <<
                ce << ", " << nd << ", " << sm << "]\n";
            return 1;
        }

        // Independent, formula-free sanity check: the heavy-nucleus
        // chain alone (ignoring accumulated He4) conserves exactly 1
        // atom at every t -- each alpha decay converts one heavy
        // nucleus into exactly one lighter one, plus a separate He4
        // atom, so the heavy chain's own count never changes.
        const double heavyChainTotal = values[1] + values[2] + values[3];
        if (!utils::approxEqual(heavyChainTotal, 1.0, 1e-9))
        {
            std::cerr << "testDecayChainSm148ConvergingNetwork: at t = " << tMult <<
                " * lifetime, heavy-chain total = " << heavyChainTotal << ", expected 1\n";
            return 1;
        }
    }

    return 0;
}

/**
 * @brief Unit test that applyDecay() is a no-op when no isotope in the list is unstable
 * @return 0 if the test passes, 1 if it fails.
 */
inline auto testDecayChainNoUnstableIsotopes() -> int
{
    const auto& h1 = elem::isotopeTable(1U, 1U);
    const auto& fe56 = elem::isotopeTable(26U, 56U);
    const elem::IsotopeList isotopes{ h1, fe56 };
    const elem::DecayChain chain(isotopes);

    std::vector<double> values{ 3.0, 5.0 };
    chain.applyDecay(1e10, values);

    if (!utils::approxEqual(values[0], 3.0, 1e-12) || !utils::approxEqual(values[1], 5.0, 1e-12))
    {
        std::cerr << "testDecayChainNoUnstableIsotopes: values = [" << values[0] << ", " <<
            values[1] << "], expected unchanged [3, 5]\n";
        return 1;
    }

    return 0;
}

/**
 * @brief Unit test that applyDecay() rejects a negative dtDecay
 * @return 0 if the test passes, 1 if it fails.
 */
inline auto testDecayChainNegativeDtThrows() -> int
{
    const auto& fe56 = elem::isotopeTable(26U, 56U);
    const auto& co56 = elem::isotopeTable(27U, 56U);
    const auto& ni56 = elem::isotopeTable(28U, 56U);
    const elem::IsotopeList isotopes{ fe56, co56, ni56 };
    const elem::DecayChain chain(isotopes);

    std::vector<double> values{ 0.0, 0.0, 1.0 };
    try
    {
        chain.applyDecay(-1.0, values);
        std::cerr << "testDecayChainNegativeDtThrows: expected applyDecay(-1.0, ...) to throw\n";
        return 1;
    }
    catch (const std::invalid_argument&) { /* expected */ }

    return 0;
}

/**
 * @brief Unit test that construction rejects a daughter missing from the given isotope list
 * @return 0 if the test passes, 1 if it fails.
 * @details
 * Ni56's real daughters() names Co56, deliberately left out of the
 * isotope list passed here -- DecayChain itself only ever resolves
 * daughters() against the list it was given (see the constructor's own
 * comment), not the global elem::isotopeTable(); closing that list to
 * include every reachable isotope is Yields::rebuildYieldGrid()'s own
 * responsibility (see its own force-expansion comment), not this
 * class's.
 */
inline auto testDecayChainMissingDaughterThrows() -> int
{
    const auto& ni56 = elem::isotopeTable(28U, 56U);
    const elem::IsotopeList isotopes{ ni56 };

    try
    {
        const elem::DecayChain chain(isotopes);
        std::cerr << "testDecayChainMissingDaughterThrows: expected construction to throw\n";
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
    result += testDecayChainNi56Co56Fe56();
    result += testDecayChainMn54MultipleModes();
    result += testDecayChainSm148ConvergingNetwork();
    result += testDecayChainNoUnstableIsotopes();
    result += testDecayChainNegativeDtThrows();
    result += testDecayChainMissingDaughterThrows();
    return result;
}

#endif // TESTDECAYCHAIN_HPP
