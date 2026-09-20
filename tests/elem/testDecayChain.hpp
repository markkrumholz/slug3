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
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
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
 * directly (not through this class), in numbers of nuclei N (starting
 * from N = 1 Sm148 nucleus): with l1 = 1/L(Sm148), l2 = 1/L(Nd144),
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
 *
 * DecayChain::applyDecay() acts on masses, not numbers of nuclei, so
 * this test passes in the mass of one Sm148 nucleus (in units of the
 * atomic mass unit, i.e. its mass number A = 148) and compares the
 * result against each expected number of nuclei multiplied by that
 * isotope's own mass number. It also checks that the total mass is
 * conserved: 148 = 144 + 4 for the first alpha decay, and 144 = 140 + 4
 * for the second.
 */
inline auto testDecayChainSm148ConvergingNetwork() -> int
{
    const auto& he4 = elem::isotopeTable(2U, 4U);
    const auto& ce140 = elem::isotopeTable(58U, 140U);
    const auto& nd144 = elem::isotopeTable(60U, 144U);
    const auto& sm148 = elem::isotopeTable(62U, 148U);
    const elem::IsotopeList isotopes{ he4, ce140, nd144, sm148 };
    const elem::DecayChain chain(isotopes);

    const double aHe = static_cast<double>(he4.A());
    const double aCe = static_cast<double>(ce140.A());
    const double aNd = static_cast<double>(nd144.A());
    const double aSm = static_cast<double>(sm148.A());

    const double l1 = 1.0 / sm148.lifetime();
    const double l2 = 1.0 / nd144.lifetime();

    for (const double tMult : {0.0, 0.5, 1.0, 2.0, 10.0})
    {
        const double t = tMult * sm148.lifetime();
        std::vector<double> values{ 0.0, 0.0, 0.0, aSm }; // the mass of one Sm148 nucleus
        chain.applyDecay(t, values);

        const double sm = std::exp(-l1 * t);
        const double nd = l1 / (l2 - l1) * (std::exp(-l1 * t) - std::exp(-l2 * t));
        const double ce = 1.0 - sm - nd;
        const double heExpected = 2.0 - sm - ((l2 * std::exp(-l1 * t)) - (l1 * std::exp(-l2 * t))) / (l2 - l1);

        if (!utils::approxEqual(values[3], sm * aSm, 1e-9) ||
            !utils::approxEqual(values[2], nd * aNd, 1e-9) ||
            !utils::approxEqual(values[1], ce * aCe, 1e-9) ||
            !utils::approxEqual(values[0], heExpected * aHe, 1e-9))
        {
            std::cerr << "testDecayChainSm148ConvergingNetwork: at t = " << tMult <<
                " * lifetime, values = [" << values[0] << ", " << values[1] << ", " <<
                values[2] << ", " << values[3] << "], expected [" << heExpected * aHe << ", " <<
                ce * aCe << ", " << nd * aNd << ", " << sm * aSm << "]\n";
            return 1;
        }

        // Independent, formula-free sanity checks. First, the heavy-
        // nucleus chain alone (ignoring accumulated He4) conserves
        // exactly 1 nucleus at every t -- each alpha decay converts one
        // heavy nucleus into exactly one lighter one, plus a separate
        // He4 nucleus, so the heavy chain's own count never changes.
        const double heavyChainTotal = (values[1] / aCe) + (values[2] / aNd) + (values[3] / aSm);
        if (!utils::approxEqual(heavyChainTotal, 1.0, 1e-9))
        {
            std::cerr << "testDecayChainSm148ConvergingNetwork: at t = " << tMult <<
                " * lifetime, heavy-chain total = " << heavyChainTotal << " nuclei, expected 1\n";
            return 1;
        }

        // Second, total mass, counting every product including the
        // accumulated He4, is conserved.
        const double totalMass = values[0] + values[1] + values[2] + values[3];
        if (!utils::approxEqual(totalMass, aSm, 1e-9))
        {
            std::cerr << "testDecayChainSm148ConvergingNetwork: at t = " << tMult <<
                " * lifetime, total mass = " << totalMass << ", expected " << aSm << "\n";
            return 1;
        }
    }

    return 0;
}

/**
 * @brief Unit test that an alpha decay conserves total mass, and moves the right share of it to each product
 * @return 0 if the test passes, 1 if it fails.
 * @details
 * Sm147 alpha-decays to Nd143, releasing one He4 nucleus per decay. The
 * Bateman equations describe numbers of nuclei, but DecayChain::
 * applyDecay() acts on masses, so a decay that changes the mass number
 * must move mass A_daughter / A_parent (143/147) to the heavy daughter
 * and A_He4 / A_parent (4/147) to the He4, not the parent's whole mass
 * to each. Applying the branching ratios directly to masses instead
 * would put 100% of the decayed mass into the Nd143 *and* another 100%
 * into the He4 -- doubling the mass, and overstating the He4 by a
 * factor of A_parent / A_He4 = 36.75. Starts from 1 Msun of Sm147, and
 * compares against the closed-form single-step solution.
 */
inline auto testDecayChainAlphaDecayConservesMass() -> int
{
    const auto& he4 = elem::isotopeTable(2U, 4U);
    const auto& nd143 = elem::isotopeTable(60U, 143U);
    const auto& sm147 = elem::isotopeTable(62U, 147U);
    const elem::IsotopeList isotopes{ he4, nd143, sm147 };
    const elem::DecayChain chain(isotopes);

    const double aHe = static_cast<double>(he4.A());
    const double aNd = static_cast<double>(nd143.A());
    const double aSm = static_cast<double>(sm147.A());

    for (const double tMult : {0.0, 0.5, 1.0, 3.0, 20.0})
    {
        const double t = tMult * sm147.lifetime();
        std::vector<double> values{ 0.0, 0.0, 1.0 }; // 1 Msun, all Sm147
        chain.applyDecay(t, values);

        const double decayed = 1.0 - std::exp(-t / sm147.lifetime()); // mass fraction of Sm147 that has decayed
        const double smExpected = 1.0 - decayed;
        const double ndExpected = decayed * aNd / aSm;
        const double heExpected = decayed * aHe / aSm;

        if (!utils::approxEqual(values[2], smExpected, 1e-9) ||
            !utils::approxEqual(values[1], ndExpected, 1e-9) ||
            !utils::approxEqual(values[0], heExpected, 1e-9))
        {
            std::cerr << "testDecayChainAlphaDecayConservesMass: at t = " << tMult <<
                " * lifetime, values [He4, Nd143, Sm147] = [" << values[0] << ", " <<
                values[1] << ", " << values[2] << "], expected [" << heExpected << ", " <<
                ndExpected << ", " << smExpected << "]\n";
            return 1;
        }

        const double totalMass = values[0] + values[1] + values[2];
        if (!utils::approxEqual(totalMass, 1.0, 1e-9))
        {
            std::cerr << "testDecayChainAlphaDecayConservesMass: at t = " << tMult <<
                " * lifetime, total mass = " << totalMass << ", expected 1\n";
            return 1;
        }
    }

    return 0;
}

/**
 * @brief Unit test that a real decay chain mixing alpha and beta decays, with branch points, conserves total mass
 * @return 0 if the test passes, 1 if it fails.
 * @details
 * Follows Pb210 through its whole real decay chain to the stable end
 * point: beta decays, which leave the mass number unchanged, and an
 * alpha decay (Po210 -> Pb206 + He4), which releases a He4 and lowers
 * the mass number by four, with small branches to Tl206 and Hg206. The
 * isotope list is built by starting from Pb210 and repeatedly adding
 * every daughter named by daughters() until nothing new appears, i.e.
 * the same decay-closed list Yields builds. All of the mass starts as
 * Pb210, and at every later time the total over every isotope in the
 * chain, including the accumulated He4, must still be that same 1 Msun,
 * since decay only moves mass between isotopes.
 *
 * Deliberately not a longer chain such as U238's, whose 97 isotopes
 * have lifetimes spanning over thirty orders of magnitude, far too
 * stiff a system for a numerical matrix exponential to handle
 * reliably, independent of the mass/number conversion this test is
 * about.
 */
inline auto testDecayChainPb210ChainConservesMass() -> int
{
    const auto& pb210 = elem::isotopeTable(82U, 210U);
    elem::IsotopeList isotopes{ pb210 };
    for (std::size_t i = 0; i < isotopes.size(); ++i) // grows while iterating; index, not iterator
    {
        for (const auto& daughter : isotopes[i].get().daughters())
        {
            const auto& next = elem::isotopeTable(daughter.Z_, daughter.A_);
            const bool present = std::ranges::any_of(isotopes,
                [&next](const auto& iso) { return iso.get() == next; });
            if (!present) { isotopes.emplace_back(next); }
        }
    }
    if (isotopes.size() < 5)
    {
        std::cerr << "testDecayChainPb210ChainConservesMass: test bug: expected the Pb210 chain "
            "to contain several isotopes, got " << isotopes.size() << "\n";
        return 1;
    }

    const elem::DecayChain chain(isotopes);
    for (const double tMult : {0.0, 0.1, 1.0, 5.0})
    {
        const double t = tMult * pb210.lifetime();
        std::vector<double> values(isotopes.size(), 0.0);
        values[0] = 1.0; // 1 Msun of Pb210, which is isotopes[0]
        chain.applyDecay(t, values);

        // The tolerance is looser than for the analytic tests above
        // because the isotope data's own branching ratios are not
        // exactly normalized when a rare branch is listed: Bi210's
        // dominant beta branch has ratio 1.0, and its 1.3e-6 alpha
        // branch to Tl206 is listed on top of that, so the daughters
        // gain ~1e-6 more mass than the parent loses. That is orders of
        // magnitude smaller than the error this test guards against,
        // which is order unity for this chain.
        double totalMass = 0.0;
        for (const double v : values) { totalMass += v; }
        if (!utils::approxEqual(totalMass, 1.0, 1e-5))
        {
            std::cerr << std::setprecision(15) << "testDecayChainPb210ChainConservesMass: at t = " << tMult <<
                " * lifetime, total mass over the whole Pb210 chain = " << totalMass <<
                ", expected 1\n";
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
    result += testDecayChainAlphaDecayConservesMass();
    result += testDecayChainPb210ChainConservesMass();
    result += testDecayChainNoUnstableIsotopes();
    result += testDecayChainNegativeDtThrows();
    result += testDecayChainMissingDaughterThrows();
    return result;
}

#endif // TESTDECAYCHAIN_HPP
