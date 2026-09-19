/**
 * @file DecayChain.hpp
 * @author Mark Krumholz
 * @brief Solves the Bateman equation for the radioactive decay of one isotope
 * @date 2026-09-19
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#ifndef DECAYCHAIN_HPP
#define DECAYCHAIN_HPP

#include "ElemCommons.hpp"
#include "IsotopeData.hpp"
#include <vector>

namespace elem
{
    /**
     * @class DecayChain
     * @brief The full network of radioactive decay products of one
     *   isotope, and the analytic (Bateman equation) solution for how
     *   many atoms of each are present after any elapsed time
     * @details
     * Built once, from a single starting isotope, by following
     * IsotopeData::daughters() recursively until every branch reaches a
     * stable isotope (see the constructor's own comment) -- products()
     * then lists every isotope reachable this way (including the
     * starting one, at index 0), and yield(t) gives each one's own
     * abundance, in atoms per atom of the starting isotope initially
     * present, after time t has elapsed.
     *
     * The underlying math (see yield()'s own comment) is the standard
     * closed-form Bateman equation solution for a linear decay chain
     * N_1 -> N_2 -> ... -> N_n, generalized here only by reading the
     * i -> i+1 branching ratio directly off IsotopeData::daughters()
     * (1 for a simple, non-branching decay). This assumes every
     * isotope's own decay rate in the chain is distinct: two isotopes
     * sharing the same rate_ (in particular, two different *stable*
     * isotopes both appearing in the same chain -- only possible if the
     * starting isotope's own decay network branches into more than one
     * distinct stable end product) would divide by zero in yield()'s
     * own sum. Every isotope actually used for nucleosynthetic yields
     * in this codebase (see elem::isotopeTable()'s own data) decays
     * along a single, non-reconverging path, so this is not a practical
     * concern here, but it is not checked for either.
     */
    class DecayChain
    {
    public:

        /**
         * @brief Build the decay chain (and cache the Bateman equation's
         *   own prefactors) for one isotope
         * @param isotope The isotope to build the decay chain for; must
         *   be a reference into the single, global elem::isotopeTable()
         *   (or otherwise outlive this DecayChain), since products()
         *   returns references into the same table, resolved from
         *   isotope's own (and its descendants' own) daughters()
         * @throws std::runtime_error if daughters() names a (Z, A) pair
         *   not found in elem::isotopeTable()
         * @details
         * Building isotopes_ (see products()'s own comment) -- and
         * hence rates_/fac_, the two cached quantities yield() actually
         * uses -- happens in three steps:
         *
         * 1. isotopes_ starts as just isotope itself. Then, for each
         *    entry already in isotopes_, in order (including entries
         *    appended by this same step, so newly-added isotopes are
         *    themselves expanded in turn), every one of its own
         *    daughters() is resolved (via elem::isotopeTable()) and
         *    appended to isotopes_. This naturally terminates once
         *    every entry's own daughters() have been processed once --
         *    a stable isotope's own empty daughters() list contributes
         *    nothing further, so a branch stops growing exactly when it
         *    reaches one.
         *
         * 2. isotopes_ is then reordered until every entry's own
         *    daughters() (by (Z, A)) all appear later in isotopes_ than
         *    the entry itself: scanning for a violation (some entry i
         *    with a daughter that resolves to an entry at or before
         *    index i) and swapping the two elements whenever one is
         *    found, repeating until a full scan finds none. This always
         *    terminates -- there are no closed cycles in radioactive
         *    decay -- and, combined with step 1's own construction
         *    order, leaves isotopes_ topologically sorted: isotopes_[0]
         *    is always the original starting isotope.
         *
         * 3. rates_[i] is set to 1 / isotopes_[i]'s own lifetime()
         *    (0 for a stable isotope, matching lifetime()'s own
         *    convention). fac_[0] is 1; fac_[k] (k >= 1) is fac_[k-1]
         *    times rates_[k-1] times the branching ratio, read off
         *    isotopes_[k-1]'s own daughters(), for its decay
         *    specifically into isotopes_[k] (0 if isotopes_[k-1] does
         *    not decay directly into isotopes_[k] -- e.g. if the two
         *    are related only through some other isotope between them
         *    in the chain). This is exactly prod_{i=1}^{k-1} b_{i,i+1}
         *    rate_i in the Bateman equation's own 1-indexed notation.
         */
        explicit DecayChain(const IsotopeData& isotope);

        DecayChain(const DecayChain&) = default;
        auto operator=(const DecayChain&) -> DecayChain& = default;
        DecayChain(DecayChain&&) = default;
        auto operator=(DecayChain&&) -> DecayChain& = default;
        ~DecayChain() = default;

        /**
         * @brief Return every isotope in this decay chain
         * @return A const reference to isotopes_: the starting isotope
         *   (index 0) followed by every isotope reachable from it by
         *   following daughters() recursively, topologically sorted so
         *   that every entry's own daughters() appear later in the list
         *   than the entry itself -- see the constructor's own comment
         * @details
         * yield(t)'s own returned vector is in this same order and of
         * this same length, so products()[j] is the isotope
         * yield(t)[j] gives the abundance of.
         */
        [[nodiscard]] auto products() const noexcept -> const IsotopeList& { return isotopes_; }

        /**
         * @brief Return each product isotope's own abundance after time t
         * @param t Elapsed time since one atom of products()[0] (the
         *   starting isotope) was present, and no atoms of any other
         *   product yet, in the same time units as every isotope's own
         *   lifetime()
         * @return A vector of products().size() values, in products()'s
         *   own order: result[j] is the number of atoms of products()[j]
         *   expected to be present at time t, per atom of products()[0]
         *   present at t=0
         * @details
         * The standard closed-form Bateman equation solution for a
         * linear decay chain, generalized only by reading the
         * appropriate branching ratio into fac_ -- see the
         * constructor's own comment for how rates_/fac_ are built.
         * For product index n (0-indexed here; the Bateman equation
         * itself is usually written 1-indexed):
         *
         *   result[n] = fac_[n] * sum_{i=0}^{n} exp(-rates_[i] * t) /
         *       prod_{j=0, j != i}^{n} (rates_[j] - rates_[i])
         *
         * For n = 0 (the starting isotope itself), this reduces to
         * fac_[0] * exp(-rates_[0] * t) = exp(-rates_[0] * t) (fac_[0]
         * is always 1): simple exponential decay (or, if products()[0]
         * is itself stable, rates_[0] = 0 and result[0] is 1 for every
         * t, as expected).
         */
        [[nodiscard]] auto yield(double t) const -> std::vector<double>;

    private:
        IsotopeList isotopes_;      /**< Every isotope in the chain, topologically sorted -- see products()'s own comment */
        std::vector<double> rates_; /**< Decay rate (1/lifetime(), 0 if stable) of each isotopes_ entry, in the same order */
        std::vector<double> fac_;   /**< Cached Bateman equation prefactor for each isotopes_ entry, in the same order -- see the constructor's own comment */
    };

} // namespace elem

#endif // DECAYCHAIN_HPP
