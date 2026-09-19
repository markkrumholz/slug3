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
#include "IsotopeTable.hpp"
#include <cmath>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace elem
{
    namespace
    {
        /**
         * @brief Breadth-first expansion: every isotope reachable from isotope via daughters()
         * @param isotope The starting isotope
         * @returns isotope itself (index 0) followed by every isotope
         *   reachable from it by following daughters() recursively --
         *   see DecayChain::DecayChain()'s own comment, step 1
         * @throws std::runtime_error if some isotope's own daughters()
         *   names a (Z, A) pair not found in the global isotope table
         */
        auto buildChain(const IsotopeData& isotope) -> IsotopeList //NOLINT(llvm-prefer-static-over-anonymous-namespace)
        {
            IsotopeList isotopes;
            isotopes.emplace_back(isotope);
            for (std::size_t i = 0; i < isotopes.size(); ++i)
            {
                for (const auto& daughter : isotopes[i].get().daughters()) // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- i < isotopes.size() by construction (loop condition)
                {
                    try
                    {
                        isotopes.emplace_back(isotopeTable(daughter.Z_, daughter.A_));
                    }
                    catch (const std::out_of_range&)
                    {
                        throw std::runtime_error(
                            "DecayChain: " + isotopes[i].get().label() + // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- see above
                            "'s daughter (Z=" + std::to_string(daughter.Z_) + ", A=" +
                            std::to_string(daughter.A_) + ") is not in the isotope table");
                    }
                }
            }
            return isotopes;
        }

        /**
         * @brief Find an isotope's own index in a chain, by (Z, A)
         * @param isotopes The chain to search
         * @param z Atomic number to look for
         * @param a Mass number to look for
         * @returns The index of the first entry of isotopes matching
         *   (z, a), or nullopt if none does -- used by sortChain() to
         *   locate where one isotope's own daughter currently sits
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
         * @brief Reorder isotopes in place until every entry's own daughters() appear later than it
         * @param isotopes The list to reorder, in place -- see
         *   DecayChain::DecayChain()'s own comment, step 2
         */
        void sortChain(IsotopeList& isotopes) //NOLINT(llvm-prefer-static-over-anonymous-namespace)
        {
            bool swappedAny = true;
            while (swappedAny)
            {
                swappedAny = false;
                for (std::size_t i = 0; i < isotopes.size() && !swappedAny; ++i)
                {
                    for (const auto& daughter : isotopes[i].get().daughters()) // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- i < isotopes.size() by construction (loop condition)
                    {
                        const auto j = findIndex(isotopes, daughter.Z_, daughter.A_);
                        if (j.has_value() && *j <= i)
                        {
                            std::swap(isotopes[i], isotopes[*j]); // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- i < isotopes.size() by construction, and *j <= i by the condition just checked
                            swappedAny = true;
                            break;
                        }
                    }
                }
            }
        }

        /**
         * @brief Compute each isotope's own decay rate (1/lifetime(), 0 if stable)
         * @param isotopes The (already sorted) chain
         * @returns One rate per entry of isotopes, in the same order --
         *   see DecayChain::DecayChain()'s own comment, step 3
         */
        auto computeRates(const IsotopeList& isotopes) -> std::vector<double> //NOLINT(llvm-prefer-static-over-anonymous-namespace)
        {
            std::vector<double> rates(isotopes.size());
            for (std::size_t i = 0; i < isotopes.size(); ++i)
            {
                const auto& iso = isotopes[i].get(); // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- i < isotopes.size() by construction
                rates[i] = iso.stable() ? 0.0 : 1.0 / iso.lifetime(); // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- see above
            }
            return rates;
        }

        /**
         * @brief Compute the cached Bateman equation prefactor for each isotope in the chain
         * @param isotopes The (already sorted) chain
         * @param rates isotopes' own decay rates, from computeRates()
         * @returns One prefactor per entry of isotopes, in the same
         *   order -- see DecayChain::DecayChain()'s own comment, step 3
         */
        auto computeFactors(const IsotopeList& isotopes, const std::vector<double>& rates) -> std::vector<double> //NOLINT(llvm-prefer-static-over-anonymous-namespace)
        {
            const std::size_t n = isotopes.size();
            std::vector<double> fac(n, 0.0);
            if (n > 0) { fac[0] = 1.0; }
            for (std::size_t k = 1; k < n; ++k)
            {
                double branchingRatio = 0.0;
                for (const auto& daughter : isotopes[k - 1].get().daughters()) // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- k - 1 < n == isotopes.size() since k < n
                {
                    if (daughter.Z_ == isotopes[k].get().Z() && daughter.A_ == isotopes[k].get().A()) // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- k < n == isotopes.size() by construction
                    {
                        branchingRatio = daughter.branchingRatio_;
                        break;
                    }
                }
                fac[k] = fac[k - 1] * branchingRatio * rates[k - 1]; // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- k < n, k - 1 < n by construction
            }
            return fac;
        }
    } // namespace

    DecayChain::DecayChain(const IsotopeData& isotope)
    {
        isotopes_ = buildChain(isotope);
        sortChain(isotopes_);
        rates_ = computeRates(isotopes_);
        fac_ = computeFactors(isotopes_, rates_);
    }

    auto DecayChain::yield(const double t) const -> std::vector<double>
    {
        const std::size_t n = isotopes_.size();
        std::vector<double> result(n, 0.0);
        for (std::size_t nIdx = 0; nIdx < n; ++nIdx)
        {
            double sum = 0.0;
            for (std::size_t i = 0; i <= nIdx; ++i)
            {
                double denom = 1.0;
                for (std::size_t j = 0; j <= nIdx; ++j)
                {
                    if (j != i) { denom *= (rates_[j] - rates_[i]); } // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- i, j <= nIdx < n == rates_.size() by construction
                }
                sum += std::exp(-rates_[i] * t) / denom; // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- i <= nIdx < n == rates_.size() by construction
            }
            result[nIdx] = fac_[nIdx] * sum; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access,cppcoreguidelines-pro-bounds-constant-array-index) -- nIdx < n == result.size() == fac_.size() by construction
        }
        return result;
    }

} // namespace elem
