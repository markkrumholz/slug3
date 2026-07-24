/**
 * @file SpecsynCommons.hpp
 * @author Mark Krumholz
 * @brief Common definitions used by spectral synthesis classes
 * @date 2026-07-20
 */

#ifndef SPECSYNCOMMONS_HPP
#define SPECSYNCOMMONS_HPP

#include <cstdint>
#include <filesystem>
#include <string>

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
