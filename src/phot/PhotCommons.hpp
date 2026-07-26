/**
 * @file PhotCommons.hpp
 * @author Mark Krumholz
 * @brief Common definitions used by photometry classes
 * @date 2026-07-26
 */

#ifndef PHOTCOMMONS_HPP
#define PHOTCOMMONS_HPP

#include <cstdint>

/**
 * @brief A namespace to hold items dealing with photometry
*/
namespace phot
{
    /**
     * @brief Specifies a photometric system
     * @details
     * A photometric system determines the units a computed
     * photometric value is expressed in -- either a physical flux
     * (Flambda, Fnu) or a magnitude on one of several magnitude
     * scales (ST, AB, Vega).
     */
    enum class PhotSystem : std::uint8_t // NOLINT(readability-identifier-naming) -- capitalized to match Flambda/Fnu/ST/AB/Vega's own fixed photometric-system naming below, rather than the project's usual camelBack enum-constant convention
    {
        Flambda, /**< Flux per unit wavelength */ // NOLINT(readability-identifier-naming) -- see PhotSystem
        Fnu,     /**< Flux per unit frequency */ // NOLINT(readability-identifier-naming) -- see PhotSystem
        ST,      /**< The ST magnitude system */ // NOLINT(readability-identifier-naming) -- see PhotSystem
        AB,      /**< The AB magnitude system */ // NOLINT(readability-identifier-naming) -- see PhotSystem
        Vega     /**< The Vega magnitude system */ // NOLINT(readability-identifier-naming) -- see PhotSystem
    };

    /**
     * @brief Specifies a photometric detector's counter type
     * @details
     * Distinguishes the two ways a detector can register incoming
     * light, which affects how its response is weighted across a
     * filter's transmission curve.
     */
    enum class CounterType : std::uint8_t // NOLINT(readability-identifier-naming) -- capitalized to match EnergyCounter/PhotonCounter's own fixed naming below, rather than the project's usual camelBack enum-constant convention
    {
        EnergyCounter, /**< A detector that measures energy */ // NOLINT(readability-identifier-naming) -- see CounterType
        PhotonCounter  /**< A detector that measures photons */ // NOLINT(readability-identifier-naming) -- see CounterType
    };

    /**
     * @brief Specifies a zero point definition system
     * @details
     * Describes the convention used to convert a physical flux into
     * a magnitude.
     */
    enum class ZPType : std::uint8_t // NOLINT(readability-identifier-naming) -- capitalized to match Pogson/Asinh/Linear's own fixed naming below, rather than the project's usual camelBack enum-constant convention
    {
        Pogson, /**< The Pogson (traditional logarithmic) magnitude system */ // NOLINT(readability-identifier-naming) -- see ZPType
        Asinh,  /**< The asinh magnitude system (Lupton, Gunn, & Szalay 1999, http://adsabs.harvard.edu/abs/1999AJ....118.1406L) */ // NOLINT(readability-identifier-naming) -- see ZPType
        Linear  /**< The linear (non-magnitude) system */ // NOLINT(readability-identifier-naming) -- see ZPType
    };

} // namespace phot

#endif // PHOTCOMMONS_HPP
