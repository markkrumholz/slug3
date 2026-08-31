/**
 * @file IsotopeData.hpp
 * @author Mark Krumholz
 * @brief Isotope data class: mass number, radioactive lifetime, and decay daughters
 * @date 2026-08-31
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#ifndef ISOTOPEDATA_HPP
#define ISOTOPEDATA_HPP

#include "ElemData.hpp"
#include <array>
#include <utility>
#include <vector>

namespace elem
{
    /**
     * @struct IsotopeDecayData
     * @brief One decay channel: the daughter nuclide and its branching ratio
     */
    struct IsotopeDecayData
    {
        unsigned int Z_; // NOLINT(readability-identifier-naming) -- physics naming convention
        unsigned int A_; // NOLINT(readability-identifier-naming) -- physics naming convention
        double branchingRatio_;
    };

    /**
     * @class IsotopeData
     * @brief Isotopic data: mass number, radioactive decay lifetime, and decay daughters
     * @details Inherits atomic symbol and atomic number from ElemData.  A lifetime_ of
     * 0 indicates a stable isotope.
     */
    class IsotopeData : public ElemData
    {
    public:

        /**
         * @brief Construct an IsotopeData record
         * @param symbol Two-character atomic symbol; single-char symbols use '\0' as
         *   the second character (e.g. {'H', '\0'})
         * @param z Atomic number
         * @param a Mass number
         * @param lifetime Radioactive decay lifetime; 0 indicates a stable isotope
         * @param daughters List of decay channels (daughter nuclide and branching ratio)
         */
        IsotopeData(std::array<char, 2> symbol, unsigned int z, unsigned int a,
            double lifetime = 0.0, std::vector<IsotopeDecayData> daughters = {})
        : ElemData(symbol, z), A_(a), lifetime_(lifetime), daughters_(std::move(daughters)) {}

        IsotopeData(const IsotopeData&) = default;
        auto operator=(const IsotopeData&) -> IsotopeData& = default;
        IsotopeData(IsotopeData&&) = default;
        auto operator=(IsotopeData&&) -> IsotopeData& = default;
        ~IsotopeData() = default;

        /** @brief Return the mass number */
        [[nodiscard]] auto A() const noexcept // NOLINT(readability-identifier-naming)
            -> unsigned int { return A_; }

        /** @brief Return the radioactive decay lifetime; 0 indicates a stable isotope */
        [[nodiscard]] auto lifetime() const noexcept -> double { return lifetime_; }

        /** @brief Return the list of decay channels (daughter nuclide and branching ratio) */
        [[nodiscard]] auto daughters() const noexcept
            -> const std::vector<IsotopeDecayData>& { return daughters_; }

    protected:
        unsigned int A_; // NOLINT(readability-identifier-naming) -- physics naming convention
        double lifetime_; /**< Radioactive decay lifetime; 0 indicates a stable isotope */
        std::vector<IsotopeDecayData> daughters_; /**< Decay channels: daughter nuclide and branching ratio */
    };

} // namespace elem

#endif // ISOTOPEDATA_HPP
