/**
 * @file ElemData.hpp
 * @author Mark Krumholz
 * @brief Base class for compile-time elemental identity data
 * @date 2026-08-31
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#ifndef ELEMDATA_HPP
#define ELEMDATA_HPP

#include <array>

namespace elem
{
    /**
     * @class ElemData
     * @brief Compile-time elemental identity: atomic symbol and atomic number
     * @details All members, the constructor, observer methods, and comparison operators
     * are constexpr, allowing compile-time construction and use.  Comparison operators
     * compare on atomic number Z, enabling sorting of element lists by Z.  Serves as a
     * base class for more specific per-element data records, such as ionization
     * potentials or isotopic data.
     */
    class ElemData
    {
    public:

        /**
         * @brief Construct an ElemData record
         * @param symbol Two-character atomic symbol; single-char symbols use '\0' as
         *   the second character (e.g. {'H', '\0'})
         * @param z Atomic number
         */
        constexpr ElemData(std::array<char, 2> symbol, unsigned int z) noexcept
        : symbol_(symbol), Z_(z) {}

        ElemData(const ElemData&) = default;
        auto operator=(const ElemData&) -> ElemData& = default;
        ElemData(ElemData&&) = default;
        auto operator=(ElemData&&) -> ElemData& = default;
        ~ElemData() = default;

        /** @brief Return the two-character atomic symbol */
        [[nodiscard]] constexpr auto symbol() const noexcept
            -> const std::array<char, 2>& { return symbol_; }

        /** @brief Return the atomic number */
        [[nodiscard]] constexpr auto Z() const noexcept // NOLINT(readability-identifier-naming)
            -> unsigned int { return Z_; }

        constexpr auto operator==(const ElemData& o) const noexcept -> bool { return Z_ == o.Z_; }
        constexpr auto operator!=(const ElemData& o) const noexcept -> bool { return Z_ != o.Z_; }
        constexpr auto operator< (const ElemData& o) const noexcept -> bool { return Z_ <  o.Z_; }
        constexpr auto operator> (const ElemData& o) const noexcept -> bool { return Z_ >  o.Z_; }
        constexpr auto operator<=(const ElemData& o) const noexcept -> bool { return Z_ <= o.Z_; }
        constexpr auto operator>=(const ElemData& o) const noexcept -> bool { return Z_ >= o.Z_; }

    protected:
        std::array<char, 2> symbol_; /**< Two-character atomic symbol */
        unsigned int Z_; // NOLINT(readability-identifier-naming) -- physics naming convention
    };

} // namespace elem

#endif // ELEMDATA_HPP
