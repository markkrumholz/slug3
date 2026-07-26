/**
 * @file PhotCommons.hpp
 * @author Mark Krumholz
 * @brief Common definitions used by photometry classes
 * @date 2026-07-26
 */

#ifndef PHOTCOMMONS_HPP
#define PHOTCOMMONS_HPP

#include "../utils/Constants.hpp"
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

    inline constexpr double flux0AB = 3631.0;    /**< Zero point of the AB system, in Jy */
    inline constexpr double flux0ST = 3.631e-9;  /**< Zero point of the ST system, in erg/s/cm^2/Angstrom */

    /**
     * @brief Convert a physical flux from one photometric system to another
     * @tparam from The PhotSystem fluxIn is expressed in
     * @tparam to The PhotSystem to convert fluxIn to
     * @param fluxIn The input flux, in erg/s/cm^2/Angstrom (if from is
     *   Flambda) or Jy (if from is Fnu)
     * @param wl The wavelength at which fluxIn is evaluated, in Angstrom
     * @return fluxIn converted to to, in Jy (if to is Fnu) or
     *   erg/s/cm^2/Angstrom (if to is Flambda)
     * @details
     * Only declared here, not defined: only the Flambda <-> Fnu
     * conversions below are physical-flux conversions in the sense
     * this function handles, so only those two (from, to)
     * combinations are ever given a definition, via explicit
     * specialization. Converting to or from a magnitude system (ST,
     * AB, Vega) is a different kind of operation entirely, handled by
     * separate functions rather than by specializing this one for
     * those PhotSystem values.
     */
    template <PhotSystem from, PhotSystem to>
    constexpr auto PhotConvertPhys(double fluxIn, double wl) -> double; // NOLINT(readability-identifier-naming) -- capitalized to match this file's other fixed photometric naming (PhotSystem, CounterType, ZPType above), rather than the project's usual camelBack function convention

    /**
     * @brief Convert Flambda (erg/s/cm^2/Angstrom) to Fnu (Jy)
     * @details
     * F_nu = F_lambda * lambda^2 / c (with every quantity in cgs
     * units: F_lambda in erg/s/cm^2/cm, lambda and c in cm and cm/s
     * respectively, giving F_nu in erg/s/cm^2/Hz), rewritten so
     * F_lambda's own conversion from a per-Angstrom to a per-cm
     * density (dividing by utils::Angstrom) and one of lambda's two
     * powers' conversion from Angstrom to cm (multiplying by
     * utils::Angstrom) cancel, leaving a single overall factor of
     * utils::Angstrom -- confirmed against the standard F_nu[Jy] =
     * 3.336e4 * lambda[Angstrom]^2 * F_lambda[erg/s/cm^2/Angstrom]
     * conversion, since utils::Angstrom / utils::c / utils::Jy =
     * 3.336e4. The final division by utils::Jy converts the
     * erg/s/cm^2/Hz result to Jy.
     */
    template <>
    constexpr auto PhotConvertPhys<PhotSystem::Flambda, PhotSystem::Fnu>(const double fluxIn, const double wl) -> double // NOLINT(readability-identifier-naming) -- see the primary template above
    {
        return fluxIn * wl * wl * utils::Angstrom / utils::c / utils::Jy;
    }

    /**
     * @brief Convert Fnu (Jy) to Flambda (erg/s/cm^2/Angstrom)
     * @details
     * The algebraic inverse of the Flambda -> Fnu conversion above.
     */
    template <>
    constexpr auto PhotConvertPhys<PhotSystem::Fnu, PhotSystem::Flambda>(const double fluxIn, const double wl) -> double // NOLINT(readability-identifier-naming) -- see the primary template above
    {
        return fluxIn * utils::Jy * utils::c / (wl * wl * utils::Angstrom);
    }

} // namespace phot

#endif // PHOTCOMMONS_HPP
