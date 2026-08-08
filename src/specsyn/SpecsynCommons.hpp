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
     * own OOBPolicy::raise; the two grids otherwise overlap in
     * (log(g), Teff) coverage, and either would interpolate a
     * comparably reasonable spectrum there.
     *
     * RAUCH is listed after both Tremblay grids: MIST's brief post-AGB
     * ("phase 6") transit can reach Teff hotter than even TREMBLAY_DA's
     * own ceiling (140000 K) while log(g) is still only ~6.3-6.6 --
     * below TREMBLAY_DA's own log(g) floor (6.5) at the same time -- a
     * real gap neither Tremblay grid covers (see the WD/hot-atmosphere-
     * coverage project notes for how this gap was found). RAUCH's own
     * (log(g), Teff) coverage (5-8, 50000-190000 K) overlaps both
     * Tremblay grids' hot ends, and SpecsynLibChained's GridType::
     * wdGrid classification (see its own comment) picks whichever
     * chained WD-type library a given star's (log(g), Teff) actually
     * falls in reach of, so listing RAUCH after Tremblay only matters
     * for a star literally nothing earlier in the chain can serve at
     * all, even via a fallback clamp.
     *
     * RAUCH_H07 is listed last of all, after RAUCH: it reaches log(g)
     * up to 9 (vs. RAUCH's own 8) at the same hot temperatures RAUCH
     * covers, plugging a further gap -- a moderately massive white
     * dwarf, log(g) 8-9, cooling from a still-hot (140000-190000 K)
     * state -- that neither RAUCH nor either Tremblay grid covers on
     * its own (RAUCH's own log(g) ceiling is too low there; the
     * Tremblay grids' own Teff ceilings are too low). RAUCH_H07 is a
     * pure H/He, no-metals grid (see fetch_rauch.py's own docstring),
     * so it is less physically faithful than RAUCH's own solar-
     * abundance models -- listed last, rather than merged with RAUCH,
     * so RAUCH's more faithful models are always preferred wherever
     * they actually cover a star.
     */
    inline const std::vector<std::string> defaultModelList = { // NOLINT(cert-err58-cpp) -- built from fixed string literals, so construction can never actually throw here
        "POWR_WC", "POWR_WNE", "POWR_WNL_H20", "POWR_WNL_H40", "POWR_WNL_H60",
        "TLUSTY_O", "TLUSTY_B", "BOSZ", "CK04", "MARCS",
        "TREMBLAY_DA", "TREMBLAY_ELM", "RAUCH", "RAUCH_H07"
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

    /**
     * @brief Distinguishes the different kinds of chained spectral library by how they're parameterized
     * @details
     * Used by SpecsynLibChained to track separate log(Teff)/log(g)
     * clamp ranges per kind of library, rather than a single range
     * combined across every chained library regardless of type --
     * see SpecsynLibChained's own logTeffMin_/logTeffMax_/loggMin_/
     * loggMax_ members. nGridType is not itself a grid type: it is a
     * trailing sentinel giving the number of real enumerators above
     * it, used to size those arrays.
     */
    enum class GridType : std::uint8_t
    {
        wrGrid,     /**< A SpecsynLibWR library (WR_grid = true in the registry) */
        wdGrid,     /**< A SpecsynLibWD library (WD_grid = true in the registry) */
        normalGrid, /**< A SpecsynLibNoWind library (neither flag set) */
        nGridType   /**< Sentinel: the number of real GridType enumerators above */
    };

} // namespace specsyn

#endif // SPECSYNCOMMONS_HPP
