/**
 * @file NebularCommons.hpp
 * @author Mark Krumholz
 * @brief Common definitions for nebular emission from a pre-computed cloudy grid
 * @date 2026-08-26
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#ifndef NEBULARCOMMONS_HPP
#define NEBULARCOMMONS_HPP

#include <filesystem>
#include <string>

/**
 * @brief A namespace to hold machinery for nebular emission from a pre-computed cloudy grid
 */
namespace nebular
{

    inline constexpr double defaultLogU = -2.5;      /**< Default log10 of the ionization parameter */
    inline constexpr double defaultCovFac = 0.5;     /**< Default nebular covering factor */
    inline constexpr double defaultLineWidth = 20.0; /**< Default assumed emission line width, in km/s */
    inline constexpr bool defaultComputeNeb = true;  /**< Default for whether nebular emission is computed at all */

    inline const std::string defaultTable = // NOLINT(bugprone-throwing-static-initialization,cert-err58-cpp) -- built from fixed string literals, so the (theoretically throwing) path conversion can never actually throw here
        (std::filesystem::path("data") / std::filesystem::path("nebular")
        / std::filesystem::path("nebular.h5")).string(); /**< Default nebular emission table */

    /**
     * @struct NebularControls
     * @brief Control parameters governing nebular emission from a pre-computed cloudy grid
     * @details
     * A plain aggregate of settings SimControls populates from an
     * input deck's own [nebular] stanza (see
     * SimControls::readNebular()), each defaulting to the
     * corresponding nebular::defaultXxx constant above if its own key
     * was not given.
     */
    struct NebularControls
    {
        double logU_ = defaultLogU;           /**< log10 of the ionization parameter */
        double covFac_ = defaultCovFac;       /**< Nebular covering factor */
        double lineWidth_ = defaultLineWidth; /**< Assumed width of emission lines, in km/s */
        bool computeNeb_ = defaultComputeNeb; /**< Whether nebular emission is computed at all; if false, SimControls::nebular() is left null */
    };

} // namespace nebular

#endif // NEBULARCOMMONS_HPP
