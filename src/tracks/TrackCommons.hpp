/**
 * @file TrackCommons.hpp
 * @author Mark Krumholz
 * @brief Common definitions used by track classes
 * @date 2024-07-09
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#ifndef TRACKCOMMONS_HPP
#define TRACKCOMMONS_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>

/**
 * @brief A namespace to hold items dealing with stellar tracks
*/
namespace tracks
{
    /**
     * @brief enum of properties stored in the tracks
     */
    enum class FieldIdx : std::uint8_t
    {
        mass,   /**< Present-day mass in Msun */
        mdot,   /**< Mass loss rate in Msun/yr */
        logL,   /**< Log luminosity in Lsun */
        logTe,  /**< Log effective temperature in K */
        hSurf,  /**< Surface H mass fraction */
        heSurf, /**< Surface He mass fraction */
        cSurf,  /**< Surface C mass fraction */
        nSurf,  /**< Surface N mass fraction */
        oSurf,  /**< Surface O mass fraction */
        nTrackQty /**< Number of quantities in the tracks */
    };

    constexpr std::array<std::string_view,
        static_cast<size_t>(FieldIdx::nTrackQty)> fieldStr{
        "mass",
        "mdot",
        "log_L",
        "log_Teff",
        "h_surf",
        "he_surf",
        "c_surf",
        "n_surf",
        "o_surf"
    };  /**< Names for each quantity in the files */

    inline static const std::string defaultRegistry = // NOLINT(bugprone-throwing-static-initialization,cert-err58-cpp) -- built from fixed string literals, so the (theoretically throwing) path conversion can never actually throw here
        (std::filesystem::path("data") / std::filesystem::path("tracks")
        / std::filesystem::path("tracks.toml")); /**< Default registry */

    inline constexpr double defaultVVcrit = 0.0; /**< Default v/vcrit value */
    inline constexpr double defaultAFe = 0.0;    /**< Default [alpha/Fe] value */

    /** FieldIdx positions of the five tracked surface mass fractions, X, Y, C, N, and O */
    inline constexpr std::array<FieldIdx, 5> surfaceAbundanceIdx = {
        FieldIdx::hSurf, FieldIdx::heSurf, FieldIdx::cSurf, FieldIdx::nSurf, FieldIdx::oSurf };

    /**
     * @brief Rescale a star's surface mass fractions so that they sum to at most 1
     * @param props A star's properties, as produced by evaluating a
     *   track or isochrone; its hSurf/heSurf/cSurf/nSurf/oSurf entries
     *   are modified in place, and every other entry is untouched
     * @details
     * Interpolating each surface mass fraction independently -- which
     * is how tracks and isochrones are evaluated, with a monotonicity-
     * preserving but nonlinear scheme -- does not preserve their sum:
     * between two sample points across which X falls and Y rises (as
     * when a massive star's surface becomes hydrogen-poor), X + Y can
     * exceed 1 even though every tabulated point is valid. Scanning the
     * isochrones of every track set slug ships finds interpolated sums
     * X + Y + C + N + O up to ~1.085 (MIST), although no tabulated MIST
     * point exceeds 1. A sum above 1 makes the implied metallicity
     * Z = 1 - X - Y negative (or smaller than the C + N + O it must
     * contain), which quantities that take log(Z) -- e.g. the
     * nugis_lamers_00 WC wind velocity -- cannot handle. If
     * X + Y + C + N + O exceeds 1, this scales all five down by the
     * same factor so that they sum to exactly 1 (so that
     * Z = 1 - X - Y = C + N + O >= 0), preserving their ratios. A sum at
     * or below 1, or a NaN sum, is left untouched. Corrupt track data,
     * as opposed to interpolation artifacts, is caught when the tracks
     * are loaded instead -- see checkSurfaceAbundances() in TrackUtils.hpp.
     */
    inline void renormalizeSurfaceAbundances(std::array<double, static_cast<std::size_t>(FieldIdx::nTrackQty)>& props)
    {
        double sum = 0.0;
        for (const auto f : surfaceAbundanceIdx) { sum += props.at(static_cast<std::size_t>(f)); }
        if (!(sum > 1.0)) { return; } // also leaves a NaN sum alone, for downstream checks to catch
        for (const auto f : surfaceAbundanceIdx) { props.at(static_cast<std::size_t>(f)) /= sum; }
    }

} // namespace tracks

#endif // TRACKCOMMONS_HPP