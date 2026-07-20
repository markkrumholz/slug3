/**
 * @file SpecsynCommons.hpp
 * @author Mark Krumholz
 * @brief Common definitions used by spectral synthesis classes
 * @date 2026-07-20
 */

#ifndef SPECSYNCOMMONS_HPP
#define SPECSYNCOMMONS_HPP

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

} // namespace specsyn

#endif // SPECSYNCOMMONS_HPP
