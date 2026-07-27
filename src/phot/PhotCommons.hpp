/**
 * @file PhotCommons.hpp
 * @author Mark Krumholz
 * @brief Common definitions used by photometry classes
 * @date 2026-07-26
 */

#ifndef PHOTCOMMONS_HPP
#define PHOTCOMMONS_HPP

#include "../utils/Constants.hpp"
#include <cmath>
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

    inline constexpr double flux0AB = 3631.0;    /**< Zero point of the AB system, in Jy */
    inline constexpr double flux0ST = 3.631e-9;  /**< Zero point of the ST system, in erg/s/cm^2/Angstrom */

    /**
     * @brief Convert a flux or magnitude from one photometric system to another
     * @tparam from The PhotSystem fluxIn is expressed in
     * @tparam to The PhotSystem to convert fluxIn to
     * @param fluxIn The input value, in erg/s/cm^2/Angstrom (if from
     *   is Flambda), Jy (if from is Fnu), or a magnitude (if from is
     *   ST or AB)
     * @param wl The wavelength at which fluxIn is evaluated, in
     *   Angstrom; unused (present only for a uniform signature across
     *   every specialization) by any specialization whose from/to
     *   pair doesn't itself depend on wavelength
     * @return fluxIn converted to to, in Jy (if to is Fnu),
     *   erg/s/cm^2/Angstrom (if to is Flambda), or a magnitude (if to
     *   is ST or AB)
     * @details
     * Only declared here, not defined: only specific (from, to) pairs
     * are given a definition, via explicit specialization, below.
     * Vega -- a filter-based magnitude system with no absolute
     * flux/magnitude conversion of its own, unlike ST and AB -- is
     * handled separately, as a special case, rather than by
     * specializing this template.
     *
     * Declared constexpr here, but not every specialization below
     * actually is: only the Flambda <-> Fnu pair, which needs nothing
     * but arithmetic, can be. Every specialization touching a
     * magnitude system needs std::log10 or std::pow, neither of which
     * is a constexpr function until C++26 -- see PhotConvert<Fnu,
     * AB>'s own comment.
     */
    template <PhotSystem from, PhotSystem to>
    constexpr auto PhotConvert(double fluxIn, double wl) -> double; // NOLINT(readability-identifier-naming) -- capitalized to match this file's other fixed photometric naming (PhotSystem above), rather than the project's usual camelBack function convention

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
    constexpr auto PhotConvert<PhotSystem::Flambda, PhotSystem::Fnu>(const double fluxIn, const double wl) -> double // NOLINT(readability-identifier-naming) -- see the primary template above
    {
        return fluxIn * wl * wl * utils::Angstrom / utils::c / utils::Jy;
    }

    /**
     * @brief Convert Fnu (Jy) to Flambda (erg/s/cm^2/Angstrom)
     * @details
     * The algebraic inverse of the Flambda -> Fnu conversion above.
     */
    template <>
    constexpr auto PhotConvert<PhotSystem::Fnu, PhotSystem::Flambda>(const double fluxIn, const double wl) -> double // NOLINT(readability-identifier-naming) -- see the primary template above
    {
        return fluxIn * utils::Jy * utils::c / (wl * wl * utils::Angstrom);
    }

    /**
     * @brief Convert Fnu (Jy) to an AB magnitude
     * @details
     * The AB system's zero point (flux0AB) is a fixed flux in Jy,
     * independent of wavelength, so wl goes unused here -- see the
     * primary template's own comment. Not marked constexpr, unlike
     * the Flambda <-> Fnu conversions above: std::log10 isn't a
     * constexpr function in C++23 (only as of C++26), and while Clang
     * currently accepts it as an extension even in C++23 mode, GCC
     * cannot be relied on to be as permissive, so this only ever
     * actually needs to run at runtime anyway.
     */
    template <>
    auto PhotConvert<PhotSystem::Fnu, PhotSystem::AB>(const double fluxIn, double /*wl*/) -> double // NOLINT(readability-identifier-naming) -- see the primary template above
    {
        return -2.5 * std::log10(fluxIn / flux0AB);
    }

    /**
     * @brief Convert an AB magnitude to Fnu (Jy)
     * @details
     * The algebraic inverse of the Fnu -> AB conversion above; see
     * its own comment for why wl goes unused and why this isn't
     * marked constexpr (std::pow, used here, has the same C++23/C++26
     * constexpr status as std::log10).
     */
    template <>
    auto PhotConvert<PhotSystem::AB, PhotSystem::Fnu>(const double magIn, double /*wl*/) -> double // NOLINT(readability-identifier-naming) -- see the primary template above
    {
        return flux0AB * std::pow(10.0, magIn / -2.5);
    }

    /**
     * @brief Convert Flambda (erg/s/cm^2/Angstrom) to an ST magnitude
     * @details
     * The ST system's zero point (flux0ST) is a fixed flux in
     * erg/s/cm^2/Angstrom, independent of wavelength, so wl goes
     * unused here -- see PhotConvert<Fnu, AB>'s own comment for why,
     * and for why this isn't marked constexpr.
     */
    template <>
    auto PhotConvert<PhotSystem::Flambda, PhotSystem::ST>(const double fluxIn, double /*wl*/) -> double // NOLINT(readability-identifier-naming) -- see the primary template above
    {
        return -2.5 * std::log10(fluxIn / flux0ST);
    }

    /**
     * @brief Convert an ST magnitude to Flambda (erg/s/cm^2/Angstrom)
     * @details
     * The algebraic inverse of the Flambda -> ST conversion above.
     */
    template <>
    auto PhotConvert<PhotSystem::ST, PhotSystem::Flambda>(const double magIn, double /*wl*/) -> double // NOLINT(readability-identifier-naming) -- see the primary template above
    {
        return flux0ST * std::pow(10.0, magIn / -2.5);
    }

    // The remaining (from, to) combinations -- every one not directly
    // defined above, excluding any involving Vega (handled separately
    // as a special case rather than through this template -- see the
    // primary template's own comment) -- are each composed from the
    // six direct conversions above, chained through Flambda and/or Fnu
    // (the two physical-flux systems every magnitude system's own
    // conversion above already goes through). wl is genuinely used by
    // each of these, unlike PhotConvert<Fnu, AB>/<Flambda, ST> and
    // their inverses: it's threaded through to whichever of those
    // Flambda <-> Fnu steps the chain actually needs. None of these
    // are marked constexpr either, since each calls at least one of
    // the log10/pow-based conversions above that aren't -- see
    // PhotConvert<Fnu, AB>'s own comment for why.

    /**
     * @brief Convert Flambda (erg/s/cm^2/Angstrom) to an AB magnitude
     * @details
     * Flambda -> Fnu -> AB.
     */
    template <>
    auto PhotConvert<PhotSystem::Flambda, PhotSystem::AB>(const double fluxIn, const double wl) -> double // NOLINT(readability-identifier-naming) -- see the primary template above
    {
        return PhotConvert<PhotSystem::Fnu, PhotSystem::AB>(
            PhotConvert<PhotSystem::Flambda, PhotSystem::Fnu>(fluxIn, wl), wl);
    }

    /**
     * @brief Convert an AB magnitude to Flambda (erg/s/cm^2/Angstrom)
     * @details
     * AB -> Fnu -> Flambda, the algebraic inverse of the Flambda -> AB
     * conversion above.
     */
    template <>
    auto PhotConvert<PhotSystem::AB, PhotSystem::Flambda>(const double magIn, const double wl) -> double // NOLINT(readability-identifier-naming) -- see the primary template above
    {
        return PhotConvert<PhotSystem::Fnu, PhotSystem::Flambda>(
            PhotConvert<PhotSystem::AB, PhotSystem::Fnu>(magIn, wl), wl);
    }

    /**
     * @brief Convert Fnu (Jy) to an ST magnitude
     * @details
     * Fnu -> Flambda -> ST.
     */
    template <>
    auto PhotConvert<PhotSystem::Fnu, PhotSystem::ST>(const double fluxIn, const double wl) -> double // NOLINT(readability-identifier-naming) -- see the primary template above
    {
        return PhotConvert<PhotSystem::Flambda, PhotSystem::ST>(
            PhotConvert<PhotSystem::Fnu, PhotSystem::Flambda>(fluxIn, wl), wl);
    }

    /**
     * @brief Convert an ST magnitude to Fnu (Jy)
     * @details
     * ST -> Flambda -> Fnu, the algebraic inverse of the Fnu -> ST
     * conversion above.
     */
    template <>
    auto PhotConvert<PhotSystem::ST, PhotSystem::Fnu>(const double magIn, const double wl) -> double // NOLINT(readability-identifier-naming) -- see the primary template above
    {
        return PhotConvert<PhotSystem::Flambda, PhotSystem::Fnu>(
            PhotConvert<PhotSystem::ST, PhotSystem::Flambda>(magIn, wl), wl);
    }

    /**
     * @brief Convert an ST magnitude to an AB magnitude
     * @details
     * ST -> Flambda -> Fnu -> AB.
     */
    template <>
    auto PhotConvert<PhotSystem::ST, PhotSystem::AB>(const double magIn, const double wl) -> double // NOLINT(readability-identifier-naming) -- see the primary template above
    {
        return PhotConvert<PhotSystem::Fnu, PhotSystem::AB>(
            PhotConvert<PhotSystem::Flambda, PhotSystem::Fnu>(
                PhotConvert<PhotSystem::ST, PhotSystem::Flambda>(magIn, wl), wl), wl);
    }

    /**
     * @brief Convert an AB magnitude to an ST magnitude
     * @details
     * AB -> Fnu -> Flambda -> ST, the algebraic inverse of the ST ->
     * AB conversion above.
     */
    template <>
    auto PhotConvert<PhotSystem::AB, PhotSystem::ST>(const double magIn, const double wl) -> double // NOLINT(readability-identifier-naming) -- see the primary template above
    {
        return PhotConvert<PhotSystem::Flambda, PhotSystem::ST>(
            PhotConvert<PhotSystem::Fnu, PhotSystem::Flambda>(
                PhotConvert<PhotSystem::AB, PhotSystem::Fnu>(magIn, wl), wl), wl);
    }

} // namespace phot

#endif // PHOTCOMMONS_HPP
