/**
 * @file SpecsynCommons.hpp
 * @author Mark Krumholz
 * @brief Common definitions used by spectral synthesis classes
 * @date 2026-07-20
 */

#ifndef SPECSYNCOMMONS_HPP
#define SPECSYNCOMMONS_HPP

#include "../utils/Constants.hpp"
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

/**
 * @brief A namespace to hold items dealing with spectral synthesis
*/
namespace specsyn
{
    inline static const std::string defaultRegistry = // NOLINT(bugprone-throwing-static-initialization,cert-err58-cpp) -- built from fixed string literals, so the (theoretically throwing) path conversion can never actually throw here
        (std::filesystem::path("data") / std::filesystem::path("spectra")
        / std::filesystem::path("spectra.toml")); /**< Default registry */

    inline constexpr double defaultCFe = 0.0;        /**< Default [C/Fe] value */
    inline constexpr double defaultMicroTurb = 0.0;  /**< Default microturbulent velocity, in km/s */
    inline constexpr double defaultR = 500;          /**< Default spectral resolution */

    /**
     * @brief Default minimum output wavelength, in Angstrom
     * @details
     * The photon energy corresponding to 10 Rydberg (~91 Angstrom),
     * chosen as a floor deep enough into the extreme UV to capture
     * hydrogen- and helium-ionizing flux from hot/massive stars.
     */
    inline constexpr double defaultWlMin = (utils::h * utils::c / (10.0 * utils::Ryd)) / utils::Angstrom;

    inline constexpr double defaultWlMax = 1e5; /**< Default maximum output wavelength, in Angstrom (10 micron) */
    inline constexpr unsigned long defaultNWl = 2048; /**< Default number of output wavelength points */

    /**
     * @brief Standard chained set of stellar atmosphere models
     * @details
     * What spectra.model = "default" expands to in
     * SimControls::readSpectra() -- a reasonable general-purpose
     * chained library covering Wolf-Rayet through cool giant/dwarf
     * atmospheres, so users don't need to spell out the full list
     * themselves. TREMBLAY_DA and TREMBLAY_ELM are listed last, after
     * every ordinary (non-degenerate) atmosphere library, since white
     * dwarfs are compact, evolved end states no earlier library in
     * this list covers at all (none of POWR/TLUSTY/BOSZ/CK04/MARCS
     * extend anywhere near a white dwarf's log(g) ~ 7-9.5) -- mirrors
     * testSpecsynLibChained.cpp's own testChainWithWD, which chains an
     * ordinary library before a white dwarf one in that same order.
     * TREMBLAY_DA (log(g) 6.5-9.5, Teff up to 140000 K) is listed
     * before TREMBLAY_ELM (log(g) 4.0-9.5, but Teff only up to
     * 40000 K), so the hottest young white dwarfs -- outside ELM's own
     * Teff range entirely -- still resolve to DA rather than ELM's
     * final OOBPolicy::raise; the two grids otherwise overlap in
     * (log(g), Teff) coverage, and either would interpolate a
     * comparably reasonable spectrum there.
     */
    inline const std::vector<std::string> defaultModelList = { // NOLINT(cert-err58-cpp) -- built from fixed string literals, so construction can never actually throw here
        "POWR_WC", "POWR_WNE", "POWR_WNL_H20", "POWR_WNL_H40", "POWR_WNL_H60",
        "TLUSTY_O", "TLUSTY_B", "BOSZ", "CK04", "MARCS",
        "TREMBLAY_DA", "TREMBLAY_ELM"
    };

    /**
     * @brief Specifies how a Specsyn class handles out-of-bounds stars
     * @details
     * An out-of-bounds star is one whose properties (e.g. logg, Teff)
     * place it outside the domain a given Specsyn class can compute a
     * spectrum for -- for example, a star that falls outside a
     * SpecsynLib's (FeH, logg, Teff) tensor grid. This is a template
     * parameter, rather than a runtime flag, so that the choice of
     * behavior can be compiled directly into the hot spec() path
     * instead of requiring a runtime branch.
     */
    enum class OOBPolicy : std::uint8_t
    {
        raise,  /**< Throw a runtime error for an out-of-bounds star */
        silent, /**< Silently return a spectrum of size 0 for an out-of-bounds star */
        coerce  /**< Coerce an out-of-bounds star with at least one valid neighboring grid point to the nearest point it can be interpolated from, rather than treating it as out of bounds */
    };

} // namespace specsyn

#endif // SPECSYNCOMMONS_HPP
