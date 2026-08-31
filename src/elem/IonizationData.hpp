/**
 * @file IonizationData.hpp
 * @author Mark Krumholz
 * @brief Elemental data class and compile-time table of ionization potentials
 * @date 2026-07-31
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#ifndef IONIZATIONDATA_HPP
#define IONIZATIONDATA_HPP

#include "ElemCommons.hpp"
#include <array>
#include <cstddef>
#include <limits>

namespace elem
{
    /** Maximum number of ionization potentials stored per element */
    constexpr std::size_t maxIP = 20;

    /**
     * @class ElemData
     * @brief Compile-time elemental data: symbol, atomic number, and ionization potentials
     * @details All members, the constructor, observer methods, and comparison operators
     * are constexpr, allowing compile-time construction and use.  Comparison operators
     * compare on atomic number Z, enabling sorting of element lists by Z.
     */
    class ElemData
    {
    public:

        /**
         * @brief Construct an ElemData record
         * @param symbol Two-character atomic symbol; single-char symbols use '\0' as
         *   the second character (e.g. {'H', '\0'})
         * @param z Atomic number
         * @param ionPot Ionization potentials in eV, from 1st to maxIP-th; entries
         *   beyond the element's actual ionization count are quiet_NaN()
         */
        constexpr ElemData(std::array<char, 2> symbol, unsigned int z,
            std::array<double, maxIP> ionPot) noexcept
        : symbol_(symbol), Z_(z), ionPot_(ionPot) {}

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

        /** @brief Return the array of ionization potentials in eV; entries beyond
         *   the element's count are quiet_NaN() */
        [[nodiscard]] constexpr auto ionPot() const noexcept
            -> const std::array<double, maxIP>& { return ionPot_; }

        constexpr auto operator==(const ElemData& o) const noexcept -> bool { return Z_ == o.Z_; }
        constexpr auto operator!=(const ElemData& o) const noexcept -> bool { return Z_ != o.Z_; }
        constexpr auto operator< (const ElemData& o) const noexcept -> bool { return Z_ <  o.Z_; }
        constexpr auto operator> (const ElemData& o) const noexcept -> bool { return Z_ >  o.Z_; }
        constexpr auto operator<=(const ElemData& o) const noexcept -> bool { return Z_ <= o.Z_; }
        constexpr auto operator>=(const ElemData& o) const noexcept -> bool { return Z_ >= o.Z_; }

    protected:
        std::array<char, 2> symbol_; /**< Two-character atomic symbol */
        unsigned int Z_; // NOLINT(readability-identifier-naming) -- physics naming convention
        std::array<double, maxIP> ionPot_; /**< Ionization potentials in eV */
    };

    namespace detail
    {
        /** Fill an array of maxIP ionization potentials from a C-style array of N values,
         *  padding the rest with quiet_NaN() */
        template<std::size_t N>
        constexpr auto makeIonPot(const double (&vals)[N]) noexcept // NOLINT(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays) -- C array ref is the only constexpr-compatible way to accept a braced-list of N values in C++17
            -> std::array<double, maxIP>
        {
            constexpr double nan = std::numeric_limits<double>::quiet_NaN();
            std::array<double, maxIP> arr{};
            for (std::size_t i = 0; i < maxIP; ++i) {
                arr[i] = (i < N) ? vals[i] : nan; // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- i is bounded by the loop condition
            }
            return arr;
        }

        /** Return an array of maxIP quiet_NaN values (for elements with no CRC data) */
        constexpr auto makeEmptyIonPot() noexcept -> std::array<double, maxIP>
        {
            constexpr double nan = std::numeric_limits<double>::quiet_NaN();
            std::array<double, maxIP> arr{};
            for (std::size_t i = 0; i < maxIP; ++i) {
                arr[i] = nan; // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- i is bounded by the loop condition
            }
            return arr;
        }
    } // namespace detail

    /**
     * @brief Compile-time table of elemental data, indexed by Symbols enum
     * @details Ionization potentials are CRC values in eV, from the data page
     * https://en.wikipedia.org/wiki/Ionization_energies_of_the_elements_(data_page).
     * Entries beyond an element's actual ionization count, or where CRC data are
     * unavailable, are quiet_NaN().
     */
    inline constexpr std::array<ElemData,
        static_cast<std::size_t>(Symbols::nElem)> elemData = {
        // Z=1-10
        ElemData({'H', '\0'}, 1U, detail::makeIonPot({13.59844})),
        ElemData({'H', 'e'}, 2U, detail::makeIonPot({24.58738, 54.41776})),
        ElemData({'L', 'i'}, 3U, detail::makeIonPot({5.39171, 75.64009, 122.45435})),
        ElemData({'B', 'e'}, 4U, detail::makeIonPot({9.32269, 18.21115, 153.89620, 217.71858})),
        ElemData({'B', '\0'}, 5U, detail::makeIonPot({8.29803, 25.15484, 37.93064, 259.37521, 340.22580})),
        ElemData({'C', '\0'}, 6U, detail::makeIonPot({11.26030, 24.38332, 47.8878, 64.4939, 392.087, 489.99334})),
        ElemData({'N', '\0'}, 7U, detail::makeIonPot({14.53414, 29.6013, 47.44924, 77.4735, 97.8902, 552.0718, 667.046})),
        ElemData({'O', '\0'}, 8U, detail::makeIonPot({13.61806, 35.11730, 54.9355, 77.41353, 113.8990, 138.1197, 739.29, 871.4101})),
        ElemData({'F', '\0'}, 9U, detail::makeIonPot({17.42282, 34.97082, 62.7084, 87.1398, 114.2428, 157.1651, 185.186, 953.9112, 1103.1176})),
        ElemData({'N', 'e'}, 10U, detail::makeIonPot({21.5646, 40.96328, 63.45, 97.12, 126.21, 157.93, 207.2759, 239.0989, 1195.8286, 1362.1995})),
        // Z=11-20
        ElemData({'N', 'a'}, 11U, detail::makeIonPot({5.13908, 47.2864, 71.6200, 98.91, 138.40, 172.18, 208.50, 264.25, 299.864, 1465.121, 1648.702})),
        ElemData({'M', 'g'}, 12U, detail::makeIonPot({7.64624, 15.03528, 80.1437, 109.2655, 141.27, 186.76, 225.02, 265.96, 328.06, 367.50, 1761.805, 1962.6650})),
        ElemData({'A', 'l'}, 13U, detail::makeIonPot({5.98577, 18.82856, 28.44765, 119.992, 153.825, 190.49, 241.76, 284.66, 330.13, 398.75, 442.00, 2085.98, 2304.1410})),
        ElemData({'S', 'i'}, 14U, detail::makeIonPot({8.15169, 16.34585, 33.49302, 45.14181, 166.767, 205.27, 246.5, 303.54, 351.12, 401.37, 476.36, 523.42, 2437.63, 2673.182})),
        ElemData({'P', '\0'}, 15U, detail::makeIonPot({10.48669, 19.7694, 30.2027, 51.4439, 65.0251, 220.421, 263.57, 309.60, 372.13, 424.4, 479.46, 560.8, 611.74, 2816.91, 3069.842})),
        ElemData({'S', '\0'}, 16U, detail::makeIonPot({10.36001, 23.3379, 34.79, 47.222, 72.5945, 88.0530, 280.948, 328.75, 379.55, 447.5, 504.8, 564.44, 652.2, 707.01, 3223.78, 3494.1892})),
        ElemData({'C', 'l'}, 17U, detail::makeIonPot({12.96764, 23.814, 39.61, 53.4652, 67.8, 97.03, 114.1958, 348.28, 400.06, 455.63, 529.28, 591.99, 656.71, 749.76, 809.40, 3658.521, 3946.2960})),
        ElemData({'A', 'r'}, 18U, detail::makeIonPot({15.75962, 27.62967, 40.74, 59.81, 75.02, 91.009, 124.323, 143.460, 422.45, 478.69, 538.96, 618.26, 686.10, 755.74, 854.77, 918.03, 4120.8857, 4426.2296})),
        ElemData({'K', '\0'}, 19U, detail::makeIonPot({4.34066, 31.63, 45.806, 60.91, 82.66, 99.4, 117.56, 154.88, 175.8174, 503.8, 564.7, 629.4, 714.6, 786.6, 861.1, 968.0, 1033.4, 4610.8, 4934.046})),
        ElemData({'C', 'a'}, 20U, detail::makeIonPot({6.11316, 11.87172, 50.9131, 67.27, 84.50, 108.78, 127.2, 147.24, 188.54, 211.275, 591.9, 657.2, 726.6, 817.6, 894.5, 974.0, 1087.0, 1157.8, 5128.8, 5469.864})),
        // Z=21-30
        ElemData({'S', 'c'}, 21U, detail::makeIonPot({6.5615, 12.79967, 24.75666, 73.4894, 91.65, 110.68, 138.0, 158.1, 180.03, 225.18, 249.798, 687.36, 756.7, 830.8, 927.5, 1009.0, 1094.0, 1213.0, 1287.97, 5674.8})),
        ElemData({'T', 'i'}, 22U, detail::makeIonPot({6.8281, 13.5755, 27.4917, 43.2672, 99.30, 119.53, 140.8, 170.4, 192.1, 215.92, 265.07, 291.500, 787.84, 863.1, 941.9, 1044.0, 1131.0, 1221.0, 1346.0, 1425.4})),
        ElemData({'V', '\0'}, 23U, detail::makeIonPot({6.7462, 14.66, 29.311, 46.709, 65.2817, 128.13, 150.6, 173.4, 205.8, 230.5, 255.7, 308.1, 336.277, 896.0, 976.0, 1060.0, 1168.0, 1260.0, 1355.0, 1486.0})),
        ElemData({'C', 'r'}, 24U, detail::makeIonPot({6.7665, 16.4857, 30.96, 49.16, 69.46, 90.6349, 160.18, 184.7, 209.3, 244.4, 270.8, 298.0, 354.8, 384.168, 1010.6, 1097.0, 1185.0, 1299.0, 1396.0, 1496.0})),
        ElemData({'M', 'n'}, 25U, detail::makeIonPot({7.43402, 15.63999, 33.668, 51.2, 72.4, 95.6, 119.203, 194.5, 221.8, 248.3, 286.0, 314.4, 343.6, 403.0, 435.163, 1134.7, 1224.0, 1317.0, 1437.0, 1539.0})),
        ElemData({'F', 'e'}, 26U, detail::makeIonPot({7.9024, 16.1878, 30.652, 54.8, 75.0, 99.1, 124.98, 151.06, 233.6, 262.1, 290.2, 330.8, 361.0, 392.2, 457.0, 489.256, 1266.0, 1358.0, 1456.0, 1582.0})),
        ElemData({'C', 'o'}, 27U, detail::makeIonPot({7.8810, 17.083, 33.50, 51.3, 79.5, 102.0, 128.9, 157.8, 186.13, 275.4, 305.0, 336.0, 379.0, 411.0, 444.0, 511.96, 546.58, 1397.2, 1504.6, 1603.0})),
        ElemData({'N', 'i'}, 28U, detail::makeIonPot({7.6398, 18.16884, 35.19, 54.9, 76.06, 108.0, 133.0, 162.0, 193.0, 224.6, 321.0, 352.0, 384.0, 430.0, 464.0, 499.0, 571.08, 607.06, 1541.0, 1648.0})),
        ElemData({'C', 'u'}, 29U, detail::makeIonPot({7.72638, 20.29240, 36.841, 57.38, 79.8, 103.0, 139.0, 166.0, 199.0, 232.0, 265.3, 369.0, 401.0, 435.0, 484.0, 520.0, 557.0, 633.0, 670.588, 1697.0})),
        ElemData({'Z', 'n'}, 30U, detail::makeIonPot({9.3942, 17.96440, 39.723, 59.4, 82.6, 108.0, 134.0, 174.0, 203.0, 238.0, 274.0, 310.8, 419.7, 454.0, 490.0, 542.0, 579.0, 619.0, 698.0, 738.0})),
        // Z=31-36
        ElemData({'G', 'a'}, 31U, detail::makeIonPot({5.99930, 20.5142, 30.71, 64.0})),
        ElemData({'G', 'e'}, 32U, detail::makeIonPot({7.8994, 15.93462, 34.2241, 45.7131, 93.5})),
        ElemData({'A', 's'}, 33U, detail::makeIonPot({9.7886, 18.633, 28.351, 50.13, 62.63, 127.6})),
        ElemData({'S', 'e'}, 34U, detail::makeIonPot({9.75238, 21.19, 30.8204, 42.9450, 68.3, 81.7, 155.4})),
        ElemData({'B', 'r'}, 35U, detail::makeIonPot({11.81381, 21.8, 36.0, 47.3, 59.7, 88.6, 103.0, 192.8})),
        ElemData({'K', 'r'}, 36U, detail::makeIonPot({13.99961, 24.35985, 36.950, 52.5, 64.7, 78.5, 111.0, 125.802, 230.85, 268.2, 308.0, 350.0, 391.0, 447.0, 492.0, 541.0, 592.0, 641.0, 786.0, 833.0})),
        // Z=37-48
        ElemData({'R', 'b'}, 37U, detail::makeIonPot({4.17713, 27.285, 40.0, 52.6, 71.0, 84.4, 99.2, 136.0, 150.0, 277.1})),
        ElemData({'S', 'r'}, 38U, detail::makeIonPot({5.6949, 11.03013, 42.89, 57.0, 71.6, 90.8, 106.0, 122.3, 162.0, 177.0, 324.1})),
        ElemData({'Y', '\0'}, 39U, detail::makeIonPot({6.2171, 12.24, 20.52, 60.597, 77.0, 93.0, 116.0, 129.0, 146.2, 191.0, 206.0, 374.0})),
        ElemData({'Z', 'r'}, 40U, detail::makeIonPot({6.63390, 13.13, 22.99, 34.34, 80.348})),
        ElemData({'N', 'b'}, 41U, detail::makeIonPot({6.75885, 14.32, 25.04, 38.3, 50.55, 102.057, 125.0})),
        ElemData({'M', 'o'}, 42U, detail::makeIonPot({7.09243, 16.16, 27.13, 46.4, 54.49, 68.8276, 125.664, 143.6, 164.12, 186.4, 209.3, 230.28, 279.1, 302.60, 544.0, 570.0, 636.0, 702.0, 767.0, 833.0})),
        ElemData({'T', 'c'}, 43U, detail::makeIonPot({7.28, 15.26, 29.54})),
        ElemData({'R', 'u'}, 44U, detail::makeIonPot({7.36050, 16.76, 28.47})),
        ElemData({'R', 'h'}, 45U, detail::makeIonPot({7.45890, 18.08, 31.06})),
        ElemData({'P', 'd'}, 46U, detail::makeIonPot({8.3369, 19.43, 32.93})),
        ElemData({'A', 'g'}, 47U, detail::makeIonPot({7.5762, 21.49, 34.83})),
        ElemData({'C', 'd'}, 48U, detail::makeIonPot({8.9938, 16.90832, 37.48})),
        // Z=49-54
        ElemData({'I', 'n'}, 49U, detail::makeIonPot({5.78636, 18.8698, 28.03, 54.0})),
        ElemData({'S', 'n'}, 50U, detail::makeIonPot({7.3439, 14.63225, 30.50260, 40.73502, 72.28})),
        ElemData({'S', 'b'}, 51U, detail::makeIonPot({8.6084, 16.53051, 25.3, 44.2, 56.0, 108.0})),
        ElemData({'T', 'e'}, 52U, detail::makeIonPot({9.0096, 18.6, 27.96, 37.41, 58.75, 70.7, 137.0})),
        ElemData({'I', '\0'}, 53U, detail::makeIonPot({10.45126, 19.1313, 33.0})),
        ElemData({'X', 'e'}, 54U, detail::makeIonPot({12.1298, 21.20979, 32.1230})),
        // Z=55-56
        ElemData({'C', 's'}, 55U, detail::makeIonPot({3.89390, 23.15745})),
        ElemData({'B', 'a'}, 56U, detail::makeIonPot({5.21170, 10.00390})),
        // Z=57-71 (lanthanides)
        ElemData({'L', 'a'}, 57U, detail::makeIonPot({5.5769, 11.060, 19.1773, 49.95, 61.6})),
        ElemData({'C', 'e'}, 58U, detail::makeIonPot({5.5387, 10.85, 20.198, 36.758, 65.55, 77.6})),
        ElemData({'P', 'r'}, 59U, detail::makeIonPot({5.473, 10.55, 21.624, 38.98, 57.53})),
        ElemData({'N', 'd'}, 60U, detail::makeIonPot({5.5250, 10.73, 22.1, 40.41})),
        ElemData({'P', 'm'}, 61U, detail::makeIonPot({5.582, 10.90, 22.3, 41.1})),
        ElemData({'S', 'm'}, 62U, detail::makeIonPot({5.6436, 11.07, 23.4, 41.4})),
        ElemData({'E', 'u'}, 63U, detail::makeIonPot({5.6704, 11.241, 24.92, 42.7})),
        ElemData({'G', 'd'}, 64U, detail::makeIonPot({6.1501, 12.09, 20.63, 44.0})),
        ElemData({'T', 'b'}, 65U, detail::makeIonPot({5.8638, 11.52, 21.91, 39.79})),
        ElemData({'D', 'y'}, 66U, detail::makeIonPot({5.9389, 11.67, 22.8, 41.47})),
        ElemData({'H', 'o'}, 67U, detail::makeIonPot({6.0215, 11.80, 22.84, 42.5})),
        ElemData({'E', 'r'}, 68U, detail::makeIonPot({6.1077, 11.93, 22.74, 42.7})),
        ElemData({'T', 'm'}, 69U, detail::makeIonPot({6.18431, 12.05, 23.68, 42.7})),
        ElemData({'Y', 'b'}, 70U, detail::makeIonPot({6.25416, 12.1761, 25.05, 43.56})),
        ElemData({'L', 'u'}, 71U, detail::makeIonPot({5.4259, 13.9, 20.9594, 45.25, 66.8})),
        // Z=72-80
        ElemData({'H', 'f'}, 72U, detail::makeIonPot({6.82507, 14.9, 23.3, 33.33})),
        ElemData({'T', 'a'}, 73U, detail::makeIonPot({7.5496})),
        ElemData({'W', '\0'}, 74U, detail::makeIonPot({7.8640})),
        ElemData({'R', 'e'}, 75U, detail::makeIonPot({7.8335})),
        ElemData({'O', 's'}, 76U, detail::makeIonPot({8.4382})),
        ElemData({'I', 'r'}, 77U, detail::makeIonPot({8.9670})),
        ElemData({'P', 't'}, 78U, detail::makeIonPot({8.9587, 18.563, 28.0})),
        ElemData({'A', 'u'}, 79U, detail::makeIonPot({9.2255, 20.5, 30.0})),
        ElemData({'H', 'g'}, 80U, detail::makeIonPot({10.43750, 18.756, 34.2})),
        // Z=81-86
        ElemData({'T', 'l'}, 81U, detail::makeIonPot({6.1082, 20.428, 29.83})),
        ElemData({'P', 'b'}, 82U, detail::makeIonPot({7.41666, 15.0322, 31.9373, 42.32, 68.8})),
        ElemData({'B', 'i'}, 83U, detail::makeIonPot({7.2856, 16.69, 25.56, 45.3, 56.0, 88.3})),
        ElemData({'P', 'o'}, 84U, detail::makeIonPot({8.417})),
        ElemData({'A', 't'}, 85U, detail::makeEmptyIonPot()),
        ElemData({'R', 'n'}, 86U, detail::makeIonPot({10.74850})),
        // Z=87-88
        ElemData({'F', 'r'}, 87U, detail::makeIonPot({4.0727})),
        ElemData({'R', 'a'}, 88U, detail::makeIonPot({5.2784, 10.14716})),
        // Z=89-103 (actinides)
        ElemData({'A', 'c'}, 89U, detail::makeIonPot({5.17, 12.1})),
        ElemData({'T', 'h'}, 90U, detail::makeIonPot({6.3067, 11.5, 20.0, 28.8})),
        ElemData({'P', 'a'}, 91U, detail::makeIonPot({5.89})),
        ElemData({'U', '\0'}, 92U, detail::makeIonPot({6.19405, 14.72})),
        ElemData({'N', 'p'}, 93U, detail::makeIonPot({6.2657})),
        ElemData({'P', 'u'}, 94U, detail::makeIonPot({6.0262})),
        ElemData({'A', 'm'}, 95U, detail::makeIonPot({5.9738})),
        ElemData({'C', 'm'}, 96U, detail::makeIonPot({5.9915})),
        ElemData({'B', 'k'}, 97U, detail::makeIonPot({6.1979})),
        ElemData({'C', 'f'}, 98U, detail::makeIonPot({6.2817})),
        ElemData({'E', 's'}, 99U, detail::makeIonPot({6.42})),
        ElemData({'F', 'm'}, 100U, detail::makeIonPot({6.50})),
        ElemData({'M', 'd'}, 101U, detail::makeIonPot({6.58})),
        ElemData({'N', 'o'}, 102U, detail::makeIonPot({6.65})),
        ElemData({'L', 'r'}, 103U, detail::makeIonPot({4.9})),
        // Z=104-118 (no CRC data available)
        ElemData({'R', 'f'}, 104U, detail::makeIonPot({6.0})),
        ElemData({'D', 'b'}, 105U, detail::makeEmptyIonPot()),
        ElemData({'S', 'g'}, 106U, detail::makeEmptyIonPot()),
        ElemData({'B', 'h'}, 107U, detail::makeEmptyIonPot()),
        ElemData({'H', 's'}, 108U, detail::makeEmptyIonPot()),
        ElemData({'M', 't'}, 109U, detail::makeEmptyIonPot()),
        ElemData({'D', 's'}, 110U, detail::makeEmptyIonPot()),
        ElemData({'R', 'g'}, 111U, detail::makeEmptyIonPot()),
        ElemData({'C', 'n'}, 112U, detail::makeEmptyIonPot()),
        ElemData({'N', 'h'}, 113U, detail::makeEmptyIonPot()),
        ElemData({'F', 'l'}, 114U, detail::makeEmptyIonPot()),
        ElemData({'M', 'c'}, 115U, detail::makeEmptyIonPot()),
        ElemData({'L', 'v'}, 116U, detail::makeEmptyIonPot()),
        ElemData({'T', 's'}, 117U, detail::makeEmptyIonPot()),
        ElemData({'O', 'g'}, 118U, detail::makeEmptyIonPot()),
    };

} // namespace elem

#endif // IONIZATIONDATA_HPP
