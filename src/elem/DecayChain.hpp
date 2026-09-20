/**
 * @file DecayChain.hpp
 * @author Mark Krumholz
 * @brief Matrix-exponential solution for the radioactive decay of a set of isotopes
 * @date 2026-09-19
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#ifndef DECAYCHAIN_HPP
#define DECAYCHAIN_HPP

#include "ElemCommons.hpp"
#include <Eigen/Dense>
#include <cstddef>
#include <span>
#include <vector>

namespace elem
{
    /**
     * @class DecayChain
     * @brief The radioactive decay network among a set of isotopes, and
     *   the matrix-exponential solution for how their abundances evolve
     *   over any elapsed time
     * @details
     * Built once, from a whole list of isotopes (typically
     * Yields::isotopes_ -- see its own rebuildYieldGrid() comment for
     * why that list must already include every isotope reachable via
     * IsotopeData::daughters() from any unstable isotope in it, not just
     * the ones some yield channel directly tabulates). applyDecay()
     * then advances a per-isotope mass array, laid out in the exact same
     * order as the isotope list this was built from, forward by a given
     * elapsed time.
     *
     * The underlying math is the solution of the linear ODE system
     * dN/dt = M N, where N is the vector of abundances -- numbers of
     * nuclei, not masses -- and M (see depletionMatrix_'s own
     * comment) is the constant depletion matrix built once at
     * construction: N(t) = exp(M t) N(0). The branching ratios in M
     * are fractions of *nuclei*, so a decay that changes the mass
     * number (an alpha decay, which turns a parent of mass number A
     * into a daughter of A - 4 plus a He4) moves less mass into each
     * daughter than the parent loses in total, and only conserves
     * mass overall once every product is counted. applyDecay()
     * therefore takes masses, as its callers hold them, converts each
     * to a number of nuclei (proportional to mass / mass number)
     * before applying exp(M t), and converts back afterward -- see its
     * own comment. Unlike a
     * closed-form (Bateman equation) solution for a single linear decay
     * chain, this handles an arbitrary acyclic decay network with no
     * special-casing at all -- an isotope with more than one decay
     * daughter (branching out), two different isotopes that both decay
     * into the same daughter (converging, e.g. real alpha-decay chains
     * where the same daughter is produced at more than one step), and
     * an isotope that lists the same daughter more than once for
     * different decay modes (their branching ratios simply add, via the
     * same matrix entry) are all just particular shapes of M, not cases
     * this class needs to detect or reject.
     */
    class DecayChain
    {
    public:

        /**
         * @brief Build the depletion matrix for a list of isotopes
         * @param isotopes The full isotope list applyDecay()'s own
         *   values argument will be laid out over (typically
         *   Yields::isotopes_) -- must already include every isotope
         *   reachable via daughters() from any unstable entry in it (see
         *   the class's own comment); need not outlive this DecayChain,
         *   unlike the old per-isotope design -- only Z()/A()/lifetime()/
         *   daughters() are read here, at construction, nothing is
         *   retained by reference afterward
         * @throws std::runtime_error if some isotope's own daughters()
         *   names a (Z, A) pair not present in isotopes -- this should
         *   not happen if isotopes was built by force-expanding to
         *   decay-chain closure first, as Yields::rebuildYieldGrid()
         *   does; see its own comment
         * @details
         * Two passes over isotopes:
         *
         * 1. Identify the *relevant* subset: every unstable isotope, and
         *    every isotope reachable from one via daughters() (a
         *    worklist/BFS closure restricted to isotopes itself -- this
         *    does not go looking outside the given list, it only
         *    resolves daughters() against entries already present in
         *    it). A stable isotope with no unstable isotope decaying
         *    into it is *not* relevant: decay can never change its
         *    abundance, so it needs no row/column in the matrix at all.
         *    relevantIndices_ records these isotopes' own indices into
         *    isotopes, in ascending order.
         *
         * 2. Build depletionMatrix_, sized relevantIndices_.size()
         *    square, zero everywhere except: for each relevant, unstable
         *    isotope at matrix position p (i.e. relevantIndices_[p]),
         *    depletionMatrix_(p, p) -= 1 / lifetime_p (its own decay
         *    rate), and, for every entry in its own daughters() (its
         *    daughter's own matrix position q), depletionMatrix_(q, p)
         *    += branchingRatio / lifetime_p. Using += here, rather than
         *    =, is what lets an isotope list the same daughter more than
         *    once (summing to that daughter's own true total branching
         *    ratio) and what lets two different isotopes converge on
         *    the same daughter (each contributing its own inflow to the
         *    same matrix entry) -- both fall out automatically, with no
         *    separate handling.
         */
        explicit DecayChain(const IsotopeList& isotopes);

        DecayChain(const DecayChain&) = default;
        auto operator=(const DecayChain&) -> DecayChain& = default;
        DecayChain(DecayChain&&) = default;
        auto operator=(DecayChain&&) -> DecayChain& = default;
        ~DecayChain() = default;

        /**
         * @brief Advance a per-isotope mass array forward by dtDecay, in place
         * @param dtDecay Elapsed time, in yr (matching every isotope's
         *   own lifetime()), to advance values by
         * @param values One mass (Msun) per entry of the isotope list
         *   this was constructed from, in the same order; overwritten
         *   in place with the post-decay masses. Every entry not among
         *   the *relevant* isotopes identified at construction (see the
         *   constructor's own comment) is left completely untouched, on
         *   purpose: decay cannot change it.
         * @throws std::invalid_argument if dtDecay is negative --
         *   radioactive decay run backward in time is not a physically
         *   meaningful request, and can overflow the matrix exponential
         *   below to inf/NaN for a short-lived enough isotope
         * @details
         * Gathers values at the relevant isotopes' own positions into an
         * Eigen vector v, converting each mass to a number of nuclei
         * (up to a common constant factor) by dividing by that
         * isotope's mass number A -- depletionMatrix_'s own branching
         * ratios are fractions of nuclei, not of mass, so applying it
         * directly to masses would not conserve mass for a decay that
         * changes A (see the class comment). Computes the propagator
         * exp(depletionMatrix_ * dtDecay) (Eigen's own
         * scaling-and-squaring/Pade implementation,
         * unsupported/Eigen/MatrixFunctions -- well defined for any
         * matrix, including one with repeated or degenerate
         * eigenvalues, unlike a closed-form Bateman sum), multiplies it
         * by v, multiplies each entry of the result back by its
         * isotope's A to recover a mass, and scatters the result back.
         * The mass number A, rather than the true atomic mass, is used
         * for the conversion because it makes a decay chain conserve
         * total mass exactly: A is conserved by a beta decay, and
         * splits exactly as A_daughter + 4 by an alpha decay. A no-op
         * if there are no relevant isotopes at all (nothing in the given
         * list is unstable).
         *
         * This is exact, not an approximation, when applied
         * incrementally across multiple successive calls with the
         * abundances present at the start of each one (rather than the
         * ones originally present at t=0): exp(M dt1) * exp(M dt2) ==
         * exp(M (dt1 + dt2)) for any fixed matrix M, a basic property of
         * the matrix exponential -- see Cluster::computeYields()'s and
         * Galaxy::computeYields()'s own comments for why they rely on
         * this.
         */
        void applyDecay(double dtDecay, std::span<double> values) const;

        /**
         * @brief Compute the matrix-exponential propagator for a given elapsed time
         * @param dtDecay Elapsed time, in yr, to advance by -- same
         *   meaning as the other applyDecay() overload's own dtDecay
         * @returns exp(depletionMatrix_ * dtDecay), sized
         *   relevantIndices_.size() square
         * @throws std::invalid_argument if dtDecay is negative -- same
         *   condition and reason as the other applyDecay() overload
         * @details
         * Exposed separately from applyDecay() so a caller that needs to
         * advance more than one same-sized array by the *same* dtDecay
         * (e.g. Yields::yield()'s own per-channel loop, or its
         * decomposed applyDecay() overload) can compute this O(m^3)
         * matrix exponential once and reuse it via the applyDecay(const
         * Eigen::MatrixXd&, ...) overload below, rather than recomputing
         * an identical propagator on every call. The dtDecay-taking
         * applyDecay() overload above is exactly equivalent to
         * applyDecay(propagator(dtDecay), values).
         */
        [[nodiscard]] auto propagator(double dtDecay) const -> Eigen::MatrixXd; // NOLINT(misc-include-cleaner) -- Eigen::MatrixXd is provided by <Eigen/Dense> (already included above); clang-tidy's IWYU mapping just doesn't know that

        /**
         * @brief Advance a per-isotope mass array forward using an already-computed propagator, in place
         * @param propagator A matrix from propagator() (or
         *   exp(depletionMatrix_ * dtDecay) computed some other way),
         *   sized relevantIndices_.size() square
         * @param values Same meaning as the dtDecay-taking applyDecay()
         *   overload's own values parameter
         * @details
         * Skips computing the matrix exponential itself -- see
         * propagator()'s own comment for why this overload exists.
         * Does not itself validate that propagator actually corresponds
         * to a non-negative dtDecay; that guard lives in propagator().
         */
        void applyDecay(const Eigen::MatrixXd& propagator, std::span<double> values) const;

    private:
        std::vector<std::size_t> relevantIndices_; /**< Index, into the isotope list this was built from, of each row/column of depletionMatrix_, in that same order -- see the constructor's own comment */
        std::vector<double> massNumbers_; /**< Mass number A of each relevant isotope, in the same order as relevantIndices_ -- used by applyDecay() to convert between masses and numbers of nuclei */
        Eigen::MatrixXd depletionMatrix_; /**< The constant depletion matrix M in dN/dt = M N, sized relevantIndices_.size() square -- see the constructor's own comment for how it is filled */
    };

} // namespace elem

#endif // DECAYCHAIN_HPP
