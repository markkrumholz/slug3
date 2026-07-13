/**
 * @file TrackCommons.hpp
 * @author Mark Krumholz
 * @brief Common definitions used by track classes
 * @date 2024-07-09
 */

#ifndef TRACKCOMMONS_HPP
#define TRACKCOMMONS_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
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

    inline static const std::string defaultRegistry =
        (std::filesystem::path("data") / std::filesystem::path("tracks")
        / std::filesystem::path("tracks.toml")); /**< Default registry */

    inline constexpr double defaultVVcrit = 0.0; /**< Default v/vcrit value */
    inline constexpr double defaultAFe = 0.0;    /**< Default [alpha/Fe] value */

} // namespace tracks

#endif // TRACKCOMMONS_HPP