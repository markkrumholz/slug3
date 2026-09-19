/**
 * @file DecayChain.cpp
 * @author Mark Krumholz
 * @brief Implementation of DecayChain
 * @date 2026-09-19
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#include "DecayChain.hpp"
#include "ElemCommons.hpp"
#include "IsotopeData.hpp"
#include <Eigen/Dense>
#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <unsupported/Eigen/MatrixFunctions>
#include <vector>

namespace elem
{
    namespace
    {
        /**
         * @brief Find an isotope's own index in a list, by (Z, A)
         * @param isotopes The list to search
         * @param z Atomic number to look for
         * @param a Mass number to look for
         * @returns The index of the first entry of isotopes matching
         *   (z, a), or nullopt if none does
         */
        auto findIndex(const IsotopeList& isotopes, //NOLINT(llvm-prefer-static-over-anonymous-namespace)
            const unsigned int z, const unsigned int a) -> std::optional<std::size_t>
        {
            for (std::size_t j = 0; j < isotopes.size(); ++j)
            {
                if (isotopes[j].get().Z() == z && isotopes[j].get().A() == a) { return j; } // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- j < isotopes.size() by construction (loop condition)
            }
            return std::nullopt;
        }

        /**
         * @brief Identify every isotope that decay can affect: unstable isotopes and their descendants
         * @param isotopes The full isotope list to scan
         * @returns The index, into isotopes, of every unstable entry and
         *   every entry reachable from one by following daughters()
         *   recursively, in ascending order -- see DecayChain::
         *   DecayChain()'s own comment, step 1
         * @throws std::runtime_error if some relevant isotope's own
         *   daughters() names a (Z, A) pair not present in isotopes
         */
        auto findRelevantIndices(const IsotopeList& isotopes) -> std::vector<std::size_t> //NOLINT(llvm-prefer-static-over-anonymous-namespace)
        {
            std::vector<bool> relevant(isotopes.size(), false);
            std::vector<std::size_t> worklist;
            for (std::size_t i = 0; i < isotopes.size(); ++i)
            {
                if (!isotopes[i].get().stable()) // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- i < isotopes.size() by loop bound
                {
                    relevant[i] = true; // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- see above
                    worklist.push_back(i);
                }
            }
            while (!worklist.empty())
            {
                const std::size_t i = worklist.back();
                worklist.pop_back();
                for (const auto& daughter : isotopes[i].get().daughters()) // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- i < isotopes.size(), pushed only from valid indices
                {
                    const auto j = findIndex(isotopes, daughter.Z_, daughter.A_);
                    if (!j.has_value())
                    {
                        throw std::runtime_error(
                            "DecayChain: " + isotopes[i].get().label() + // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- see above
                            "'s daughter (Z=" + std::to_string(daughter.Z_) + ", A=" +
                            std::to_string(daughter.A_) + ") is not present in the given isotope list");
                    }
                    if (!relevant[*j]) // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- *j < isotopes.size() == relevant.size() by construction
                    {
                        relevant[*j] = true; // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- see above
                        worklist.push_back(*j);
                    }
                }
            }

            std::vector<std::size_t> result;
            for (std::size_t i = 0; i < isotopes.size(); ++i)
            {
                if (relevant[i]) { result.push_back(i); } // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- i < isotopes.size() == relevant.size() by loop bound
            }
            return result;
        }
    } // namespace

    DecayChain::DecayChain(const IsotopeList& isotopes)
    {
        relevantIndices_ = findRelevantIndices(isotopes);
        const std::size_t m = relevantIndices_.size();

        // Reverse lookup: position within relevantIndices_ of each
        // relevant isotope's own index into isotopes. Entries for a
        // non-relevant isotope are left at this sentinel and never read
        // -- every lookup below is only ever performed for a daughter
        // of a relevant, unstable isotope, which findRelevantIndices()
        // itself already guarantees is relevant too.
        constexpr std::size_t notRelevant = std::numeric_limits<std::size_t>::max();
        std::vector<std::size_t> positionOf(isotopes.size(), notRelevant);
        for (std::size_t p = 0; p < m; ++p)
        {
            positionOf[relevantIndices_[p]] = p; // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- relevantIndices_[p] < isotopes.size() by construction
        }

        depletionMatrix_ = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(m), static_cast<Eigen::Index>(m));
        for (std::size_t p = 0; p < m; ++p)
        {
            const auto& parent = isotopes[relevantIndices_[p]].get(); // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- relevantIndices_[p] < isotopes.size() by construction
            if (parent.stable()) { continue; } // a relevant but stable isotope is a pure decay *product*, never a source
            const auto pIdx = static_cast<Eigen::Index>(p);
            depletionMatrix_(pIdx, pIdx) -= 1.0 / parent.lifetime();
            for (const auto& daughter : parent.daughters())
            {
                const auto j = findIndex(isotopes, daughter.Z_, daughter.A_);
                // j is guaranteed to have a value, and positionOf[*j] is
                // guaranteed relevant, by findRelevantIndices()'s own
                // construction: every daughter of a relevant, unstable
                // isotope was itself marked relevant there.
                const auto qIdx = static_cast<Eigen::Index>(positionOf[*j]); // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- see above
                depletionMatrix_(qIdx, pIdx) += daughter.branchingRatio_ / parent.lifetime();
            }
        }
    }

    void DecayChain::applyDecay(const double dtDecay, const std::span<double> values) const
    {
        applyDecay(propagator(dtDecay), values);
    }

    auto DecayChain::propagator(const double dtDecay) const -> Eigen::MatrixXd
    {
        if (dtDecay < 0.0)
        {
            throw std::invalid_argument(
                "DecayChain::propagator: dtDecay must be non-negative, got " + std::to_string(dtDecay));
        }
        // Eigen's own .exp() asserts on a 0x0 matrix ("you are using an
        // empty matrix") rather than just returning one -- guard it
        // here so a DecayChain with no relevant isotopes at all (no
        // unstable entries in the list it was built from) still works,
        // matching applyDecay()'s own no-op behavior for that case.
        if (depletionMatrix_.size() == 0) { return depletionMatrix_; }
        return (depletionMatrix_ * dtDecay).exp();
    }

    void DecayChain::applyDecay(const Eigen::MatrixXd& propagator, const std::span<double> values) const
    {
        const std::size_t m = relevantIndices_.size();
        if (m == 0) { return; }

        Eigen::VectorXd v(static_cast<Eigen::Index>(m));
        for (std::size_t k = 0; k < m; ++k)
        {
            v[static_cast<Eigen::Index>(k)] = values[relevantIndices_[k]]; // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- relevantIndices_[k] < values.size() == isotopes.size() by construction
        }

        const Eigen::VectorXd result = propagator * v;

        for (std::size_t k = 0; k < m; ++k)
        {
            values[relevantIndices_[k]] = result[static_cast<Eigen::Index>(k)]; // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- see above
        }
    }

} // namespace elem
