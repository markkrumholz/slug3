/**
 * @file PhotCommons.hpp
 * @author Mark Krumholz
 * @brief Common definitions used by photometry classes
 * @date 2026-07-26
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#ifndef PHOTCOMMONS_HPP
#define PHOTCOMMONS_HPP

#include "../utils/Constants.hpp"
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <numbers>
#include <string>

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
     * @brief 4 pi (10 pc)^2, in cm^2 -- the standard distance magnitude systems assume
     * @details
     * ST, AB, and Vega magnitudes are all defined in terms of a flux,
     * not a luminosity, but the Flambda/Fnu-domain quantities SLUG
     * actually converts (e.g. FilterCollection::phot()'s own results)
     * are specific luminosities, with no particular distance baked in.
     * Converting one of these to a magnitude therefore requires first
     * adopting some fiducial distance to turn the luminosity into a
     * flux -- by convention, the same 10 pc the absolute-magnitude
     * scale itself is defined at. This has no analog for the Fnu
     * system's own physical units (there is no standard distance for
     * a bare flux-per-frequency density), which is why PhotConvert's
     * direct Flambda <-> Fnu conversion below never uses this at all.
     */
    inline constexpr double fourPiTenPcSq =
        4.0 * std::numbers::pi_v<double> * (10.0 * utils::pc) * (10.0 * utils::pc);

    inline static const std::string defaultVegaSpec = // NOLINT(bugprone-throwing-static-initialization,cert-err58-cpp) -- built from fixed string literals, so the (theoretically throwing) path conversion can never actually throw here
        (std::filesystem::path("data") / std::filesystem::path("spectra")
        / std::filesystem::path("vega.h5")); /**< Default Vega reference spectrum, for PhotSystem::Vega conversions */

    /**
     * @brief Convert a flux, specific luminosity, or magnitude from one photometric system to another
     * @tparam From The PhotSystem photIn is expressed in
     * @tparam To The PhotSystem to convert photIn to
     * @param photIn The input value -- named generically, rather than
     *   fluxIn, because it need not be a flux at all: in
     *   erg/s/cm^2/Angstrom (if From is Flambda) or erg/s/cm^2/Hz (if
     *   From is Fnu) if it genuinely is a flux, but just as often a
     *   specific luminosity in erg/s/Angstrom or erg/s/Hz respectively
     *   (e.g. FilterCollection::phot()'s own results, which carry no
     *   particular distance), or a magnitude (if From is ST or AB)
     * @param wl The wavelength at which photIn is evaluated, in
     *   Angstrom; unused (present only for a uniform signature across
     *   every specialization) by any specialization whose from/to
     *   pair doesn't itself depend on wavelength
     * @return photIn converted to To, in erg/s/[cm^2/]Hz (if To is
     *   Fnu), erg/s/[cm^2/]Angstrom (if To is Flambda), or a magnitude
     *   (if To is ST or AB)
     * @details
     * Only declared here, not defined: only specific (from, to) pairs
     * are given a definition, via explicit specialization, below.
     * Neither From nor To may be PhotSystem::Vega for this overload --
     * see the three-argument PhotConvert(photIn, wl, fluxVega)
     * overload below for any (from, to) pair involving Vega, since
     * Vega's zero point is filter-dependent (a given filter's own
     * fluxVega(), see Filter::fluxVega()) rather than a fixed
     * constant like flux0AB/flux0ST.
     *
     * Every specialization converting to or from one of the magnitude
     * systems (ST, AB) treats its Flambda/Fnu-side value as a specific
     * luminosity, not a flux already observed at some real distance:
     * it first divides by fourPiTenPcSq to convert to the flux that
     * would be seen at the standard distance of 10 pc, before applying
     * the magnitude system's own zero point -- see fourPiTenPcSq's own
     * comment. The direct Flambda <-> Fnu conversion, by contrast, is
     * purely a change of independent variable (wavelength to
     * frequency) and never touches fourPiTenPcSq at all, so it applies
     * identically whether photIn is a genuine flux or a specific
     * luminosity.
     *
     * Declared constexpr here, but not every specialization below
     * actually is: only the Flambda <-> Fnu pair, which needs nothing
     * but arithmetic, can be. Every specialization touching a
     * magnitude system needs std::log10 or std::pow, neither of which
     * is a constexpr function until C++26 -- see PhotConvert<Fnu,
     * AB>'s own comment.
     */
    template <PhotSystem From, PhotSystem To>
    constexpr auto PhotConvert(double photIn, double wl) -> double; // NOLINT(readability-identifier-naming) -- capitalized to match this file's other fixed photometric naming (PhotSystem above), rather than the project's usual camelBack function convention

    /**
     * @brief Convert a flux, specific luminosity, or magnitude to or from a Vega magnitude
     * @tparam From The PhotSystem photIn is expressed in
     * @tparam To The PhotSystem to convert photIn to
     * @param photIn The input value, in the units/system documented
     *   by the two-argument PhotConvert(photIn, wl) overload above
     * @param wl The wavelength at which photIn is evaluated, in
     *   Angstrom; see the two-argument overload above -- unused here
     *   too, since the direct Flambda <-> Vega conversions need only
     *   fluxVega, but kept for a uniform signature and because chained
     *   conversions through Flambda/Fnu still need it
     * @param fluxVega The filter-mean flux of Vega for the filter
     *   whose photometry is being converted, in erg/s/cm^2/Angstrom
     *   (see Filter::fluxVega()) -- Vega's own zero point, unlike
     *   AB/ST's fixed flux0AB/flux0ST, since "the Vega magnitude
     *   system" means different things in different filters. Unlike
     *   photIn, this is always a genuine flux (from the real,
     *   already-at-Earth Vega reference spectrum), never a specific
     *   luminosity -- see PhotConvert<Flambda, Vega>'s own comment for
     *   how the two direct Flambda <-> Vega conversions account for
     *   that difference.
     * @return photIn converted to To
     * @details
     * Overload of the two-argument PhotConvert(photIn, wl) above, for
     * exactly the (From, To) pairs where one of the two is
     * PhotSystem::Vega and the other is not (the pair (Vega, Vega) is
     * not defined, for the same reason (Flambda, Flambda) etc. are
     * not in the two-argument overload). Only declared here; see the
     * specializations below.
     */
    template <PhotSystem From, PhotSystem To>
    auto PhotConvert(double photIn, double wl, double fluxVega) -> double; // NOLINT(readability-identifier-naming) -- see the two-argument overload above

    /**
     * @brief Convert Flambda (erg/s/[cm^2/]Angstrom) to Fnu (erg/s/[cm^2/]Hz)
     * @details
     * F_nu = F_lambda * lambda^2 / c (with every quantity in cgs
     * units: F_lambda in erg/s/[cm^2/]cm, lambda and c in cm and cm/s
     * respectively, giving F_nu in erg/s/[cm^2/]Hz), rewritten so
     * F_lambda's own conversion from a per-Angstrom to a per-cm
     * density (dividing by utils::Angstrom) and one of lambda's two
     * powers' conversion from Angstrom to cm (multiplying by
     * utils::Angstrom) cancel, leaving a single overall factor of
     * utils::Angstrom. This is purely a change of independent variable
     * (wavelength to frequency), so it applies identically whether
     * photIn carries a per-area factor (a genuine flux) or not (a
     * specific luminosity) -- unlike a conversion to a magnitude
     * system, this one never needs to adopt any particular distance,
     * since Fnu (unlike ST/AB/Vega) has no standard distance of its
     * own to adopt in the first place.
     */
    template <>
    constexpr auto PhotConvert<PhotSystem::Flambda, PhotSystem::Fnu>(const double photIn, const double wl) -> double // NOLINT(readability-identifier-naming) -- see the primary template above
    {
        return photIn * wl * wl * utils::Angstrom / utils::c;
    }

    /**
     * @brief Convert Fnu (erg/s/[cm^2/]Hz) to Flambda (erg/s/[cm^2/]Angstrom)
     * @details
     * The algebraic inverse of the Flambda -> Fnu conversion above.
     */
    template <>
    constexpr auto PhotConvert<PhotSystem::Fnu, PhotSystem::Flambda>(const double photIn, const double wl) -> double // NOLINT(readability-identifier-naming) -- see the primary template above
    {
        return photIn * utils::c / (wl * wl * utils::Angstrom);
    }

    /**
     * @brief Convert Fnu (erg/s/Hz, a specific luminosity) to an AB magnitude
     * @details
     * AB magnitudes are defined in terms of a flux, not a luminosity,
     * so photIn is first divided by fourPiTenPcSq to convert it to the
     * flux that would be observed at the standard distance of 10 pc
     * (see fourPiTenPcSq's own comment), and then by utils::Jy to
     * convert that flux from erg/s/cm^2/Hz to Jy -- the units flux0AB
     * (the AB system's zero point) is itself expressed in -- before
     * applying the standard AB-magnitude formula. flux0AB is a fixed
     * value in Jy, independent of wavelength, so wl goes unused here.
     * Not marked constexpr, unlike the Flambda <-> Fnu conversions
     * above: std::log10 isn't a constexpr function in C++23 (only as
     * of C++26), and while Clang currently accepts it as an extension
     * even in C++23 mode, GCC cannot be relied on to be as permissive,
     * so this only ever actually needs to run at runtime anyway.
     * Explicitly inline, unlike the constexpr specializations above
     * (which are already implicitly inline): an explicit full
     * specialization that isn't constexpr is an ordinary function, and
     * defining an ordinary, non-inline function in a header risks an
     * ODR violation if the header is included from more than one
     * translation unit.
     */
    template <>
    inline auto PhotConvert<PhotSystem::Fnu, PhotSystem::AB>(const double photIn, double /*wl*/) -> double // NOLINT(readability-identifier-naming) -- see the primary template above
    {
        const double fluxIn = (photIn / fourPiTenPcSq) / utils::Jy;
        return -2.5 * std::log10(fluxIn / flux0AB);
    }

    /**
     * @brief Convert an AB magnitude to Fnu (erg/s/Hz, a specific luminosity)
     * @details
     * The algebraic inverse of the Fnu -> AB conversion above: recovers
     * a flux in Jy from the magnitude, converts it to erg/s/cm^2/Hz via
     * utils::Jy, then multiplies by fourPiTenPcSq to undo the standard-
     * distance assumption and recover a specific luminosity in erg/s/Hz
     * -- see PhotConvert<Fnu, AB>'s own comment for why wl goes unused,
     * why this isn't marked constexpr (std::pow, used here, has the
     * same C++23/C++26 constexpr status as std::log10), and why it is
     * marked inline.
     */
    template <>
    inline auto PhotConvert<PhotSystem::AB, PhotSystem::Fnu>(const double photIn, double /*wl*/) -> double // NOLINT(readability-identifier-naming) -- see the primary template above
    {
        const double fluxIn = flux0AB * std::pow(10.0, photIn / -2.5);
        return (fluxIn * utils::Jy) * fourPiTenPcSq;
    }

    /**
     * @brief Convert Flambda (erg/s/Angstrom, a specific luminosity) to an ST magnitude
     * @details
     * ST magnitudes are defined in terms of a flux, not a luminosity,
     * so photIn is first divided by fourPiTenPcSq to convert it to the
     * flux that would be observed at the standard distance of 10 pc
     * (see fourPiTenPcSq's own comment) before applying the standard
     * ST-magnitude formula against flux0ST (the ST system's own zero
     * point, likewise a fixed flux in erg/s/cm^2/Angstrom). flux0ST is
     * independent of wavelength, so wl goes unused here -- see
     * PhotConvert<Fnu, AB>'s own comment for why, why this isn't
     * marked constexpr, and why it is marked inline.
     */
    template <>
    inline auto PhotConvert<PhotSystem::Flambda, PhotSystem::ST>(const double photIn, double /*wl*/) -> double // NOLINT(readability-identifier-naming) -- see the primary template above
    {
        const double fluxIn = photIn / fourPiTenPcSq;
        return -2.5 * std::log10(fluxIn / flux0ST);
    }

    /**
     * @brief Convert an ST magnitude to Flambda (erg/s/Angstrom, a specific luminosity)
     * @details
     * The algebraic inverse of the Flambda -> ST conversion above:
     * recovers the flux flux0ST's own formula defines, then multiplies
     * by fourPiTenPcSq to undo the standard-distance assumption and
     * recover a specific luminosity in erg/s/Angstrom.
     */
    template <>
    inline auto PhotConvert<PhotSystem::ST, PhotSystem::Flambda>(const double photIn, double /*wl*/) -> double // NOLINT(readability-identifier-naming) -- see the primary template above
    {
        const double fluxIn = flux0ST * std::pow(10.0, photIn / -2.5);
        return fluxIn * fourPiTenPcSq;
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
    // PhotConvert<Fnu, AB>'s own comment for why. Each is marked
    // inline for the same ODR reason those non-constexpr conversions
    // are.

    /**
     * @brief Convert Flambda (erg/s/Angstrom, a specific luminosity) to an AB magnitude
     * @details
     * Flambda -> Fnu -> AB; the standard-distance conversion happens
     * once, inside the Fnu -> AB step (see its own comment) -- the
     * intermediate Flambda -> Fnu step is a pure change of variable
     * that never touches it.
     */
    template <>
    inline auto PhotConvert<PhotSystem::Flambda, PhotSystem::AB>(const double photIn, const double wl) -> double // NOLINT(readability-identifier-naming) -- see the primary template above
    {
        return PhotConvert<PhotSystem::Fnu, PhotSystem::AB>(
            PhotConvert<PhotSystem::Flambda, PhotSystem::Fnu>(photIn, wl), wl);
    }

    /**
     * @brief Convert an AB magnitude to Flambda (erg/s/Angstrom, a specific luminosity)
     * @details
     * AB -> Fnu -> Flambda, the algebraic inverse of the Flambda -> AB
     * conversion above.
     */
    template <>
    inline auto PhotConvert<PhotSystem::AB, PhotSystem::Flambda>(const double photIn, const double wl) -> double // NOLINT(readability-identifier-naming) -- see the primary template above
    {
        return PhotConvert<PhotSystem::Fnu, PhotSystem::Flambda>(
            PhotConvert<PhotSystem::AB, PhotSystem::Fnu>(photIn, wl), wl);
    }

    /**
     * @brief Convert Fnu (erg/s/Hz, a specific luminosity) to an ST magnitude
     * @details
     * Fnu -> Flambda -> ST; the standard-distance conversion happens
     * once, inside the Flambda -> ST step (see its own comment) -- the
     * intermediate Fnu -> Flambda step is a pure change of variable
     * that never touches it.
     */
    template <>
    inline auto PhotConvert<PhotSystem::Fnu, PhotSystem::ST>(const double photIn, const double wl) -> double // NOLINT(readability-identifier-naming) -- see the primary template above
    {
        return PhotConvert<PhotSystem::Flambda, PhotSystem::ST>(
            PhotConvert<PhotSystem::Fnu, PhotSystem::Flambda>(photIn, wl), wl);
    }

    /**
     * @brief Convert an ST magnitude to Fnu (erg/s/Hz, a specific luminosity)
     * @details
     * ST -> Flambda -> Fnu, the algebraic inverse of the Fnu -> ST
     * conversion above.
     */
    template <>
    inline auto PhotConvert<PhotSystem::ST, PhotSystem::Fnu>(const double photIn, const double wl) -> double // NOLINT(readability-identifier-naming) -- see the primary template above
    {
        return PhotConvert<PhotSystem::Flambda, PhotSystem::Fnu>(
            PhotConvert<PhotSystem::ST, PhotSystem::Flambda>(photIn, wl), wl);
    }

    /**
     * @brief Convert an ST magnitude to an AB magnitude
     * @details
     * ST -> Flambda -> Fnu -> AB. The ST -> Flambda step multiplies by
     * fourPiTenPcSq to undo ST's own standard-distance assumption
     * (recovering a specific luminosity), and the Fnu -> AB step
     * divides by it again to reapply AB's own -- so the two cancel,
     * exactly as they must for two magnitude systems that share the
     * same distance convention.
     */
    template <>
    inline auto PhotConvert<PhotSystem::ST, PhotSystem::AB>(const double photIn, const double wl) -> double // NOLINT(readability-identifier-naming) -- see the primary template above
    {
        return PhotConvert<PhotSystem::Fnu, PhotSystem::AB>(
            PhotConvert<PhotSystem::Flambda, PhotSystem::Fnu>(
                PhotConvert<PhotSystem::ST, PhotSystem::Flambda>(photIn, wl), wl), wl);
    }

    /**
     * @brief Convert an AB magnitude to an ST magnitude
     * @details
     * AB -> Fnu -> Flambda -> ST, the algebraic inverse of the ST ->
     * AB conversion above -- see its own comment for why the
     * intermediate standard-distance conversions cancel.
     */
    template <>
    inline auto PhotConvert<PhotSystem::AB, PhotSystem::ST>(const double photIn, const double wl) -> double // NOLINT(readability-identifier-naming) -- see the primary template above
    {
        return PhotConvert<PhotSystem::Flambda, PhotSystem::ST>(
            PhotConvert<PhotSystem::Fnu, PhotSystem::Flambda>(
                PhotConvert<PhotSystem::AB, PhotSystem::Fnu>(photIn, wl), wl), wl);
    }

    // ---- Vega magnitude conversions (three-argument overload) ----
    //
    // Flambda <-> Vega are direct: identical in form to Flambda <->
    // ST, but with flux0ST (a fixed constant) replaced by fluxVega (a
    // per-filter value) as the zero point -- so Vega itself (photIn
    // == fluxVega) evaluates to magnitude 0 by construction, exactly
    // as the Vega system is defined to work. Unlike photIn, fluxVega
    // is always a genuine flux (from the real, already-at-Earth Vega
    // reference spectrum -- see Filter::fluxVega()), never a specific
    // luminosity, so it needs no fourPiTenPcSq scaling of its own; only
    // photIn (when it's on the Flambda side) does, exactly as for
    // PhotConvert<Flambda, ST>/<ST, Flambda>. Every other (From, To)
    // pair involving Vega is built by chaining through Flambda: To
    // Vega, first convert From -> Flambda (via the two-argument
    // overload), then Flambda -> Vega; from Vega, first convert
    // Vega -> Flambda, then Flambda -> To (via the two-argument
    // overload again). None of these are constexpr, for the same
    // reason as their AB/ST counterparts above.

    /**
     * @brief Convert Flambda (erg/s/Angstrom, a specific luminosity) to a Vega magnitude
     * @details
     * Identical in form to PhotConvert<Flambda, ST>, but using this
     * filter's own Vega flux (fluxVega) as the zero point instead of
     * the fixed ST zero point (flux0ST): photIn is first divided by
     * fourPiTenPcSq to convert it to the flux that would be observed
     * at the standard distance of 10 pc (see fourPiTenPcSq's own
     * comment), before comparing it against fluxVega.
     */
    template <>
    inline auto PhotConvert<PhotSystem::Flambda, PhotSystem::Vega>(const double photIn, double /*wl*/, const double fluxVega) -> double // NOLINT(readability-identifier-naming) -- see the two-argument overload above
    {
        const double fluxIn = photIn / fourPiTenPcSq;
        return -2.5 * std::log10(fluxIn / fluxVega);
    }

    /**
     * @brief Convert a Vega magnitude to Flambda (erg/s/Angstrom, a specific luminosity)
     * @details
     * The algebraic inverse of the Flambda -> Vega conversion above,
     * identical in form to PhotConvert<ST, Flambda> with flux0ST
     * replaced by fluxVega: recovers the flux the Vega-magnitude
     * formula defines, then multiplies by fourPiTenPcSq to undo the
     * standard-distance assumption and recover a specific luminosity.
     */
    template <>
    inline auto PhotConvert<PhotSystem::Vega, PhotSystem::Flambda>(const double photIn, double /*wl*/, const double fluxVega) -> double // NOLINT(readability-identifier-naming) -- see the two-argument overload above
    {
        const double fluxIn = fluxVega * std::pow(10.0, photIn / -2.5);
        return fluxIn * fourPiTenPcSq;
    }

    /**
     * @brief Convert Fnu (erg/s/Hz, a specific luminosity) to a Vega magnitude
     * @details
     * Fnu -> Flambda (two-argument overload) -> Vega; the standard-
     * distance conversion happens once, inside the Flambda -> Vega
     * step above.
     */
    template <>
    inline auto PhotConvert<PhotSystem::Fnu, PhotSystem::Vega>(const double photIn, const double wl, const double fluxVega) -> double // NOLINT(readability-identifier-naming) -- see the two-argument overload above
    {
        return PhotConvert<PhotSystem::Flambda, PhotSystem::Vega>(
            PhotConvert<PhotSystem::Fnu, PhotSystem::Flambda>(photIn, wl), wl, fluxVega);
    }

    /**
     * @brief Convert a Vega magnitude to Fnu (erg/s/Hz, a specific luminosity)
     * @details
     * Vega -> Flambda -> Fnu (two-argument overload), the algebraic
     * inverse of the Fnu -> Vega conversion above.
     */
    template <>
    inline auto PhotConvert<PhotSystem::Vega, PhotSystem::Fnu>(const double photIn, const double wl, const double fluxVega) -> double // NOLINT(readability-identifier-naming) -- see the two-argument overload above
    {
        return PhotConvert<PhotSystem::Flambda, PhotSystem::Fnu>(
            PhotConvert<PhotSystem::Vega, PhotSystem::Flambda>(photIn, wl, fluxVega), wl);
    }

    /**
     * @brief Convert an ST magnitude to a Vega magnitude
     * @details
     * ST -> Flambda (two-argument overload) -> Vega. The ST -> Flambda
     * step multiplies by fourPiTenPcSq to undo ST's own standard-
     * distance assumption (recovering a specific luminosity), and the
     * Flambda -> Vega step divides by it again -- so the two cancel,
     * exactly as PhotConvert<ST, AB>'s own comment describes for the
     * ST <-> AB pair.
     */
    template <>
    inline auto PhotConvert<PhotSystem::ST, PhotSystem::Vega>(const double photIn, const double wl, const double fluxVega) -> double // NOLINT(readability-identifier-naming) -- see the two-argument overload above
    {
        return PhotConvert<PhotSystem::Flambda, PhotSystem::Vega>(
            PhotConvert<PhotSystem::ST, PhotSystem::Flambda>(photIn, wl), wl, fluxVega);
    }

    /**
     * @brief Convert a Vega magnitude to an ST magnitude
     * @details
     * Vega -> Flambda -> ST (two-argument overload), the algebraic
     * inverse of the ST -> Vega conversion above.
     */
    template <>
    inline auto PhotConvert<PhotSystem::Vega, PhotSystem::ST>(const double photIn, const double wl, const double fluxVega) -> double // NOLINT(readability-identifier-naming) -- see the two-argument overload above
    {
        return PhotConvert<PhotSystem::Flambda, PhotSystem::ST>(
            PhotConvert<PhotSystem::Vega, PhotSystem::Flambda>(photIn, wl, fluxVega), wl);
    }

    /**
     * @brief Convert an AB magnitude to a Vega magnitude
     * @details
     * AB -> Flambda (two-argument overload) -> Vega; the intermediate
     * standard-distance conversions cancel exactly as they do for
     * PhotConvert<ST, Vega> above.
     */
    template <>
    inline auto PhotConvert<PhotSystem::AB, PhotSystem::Vega>(const double photIn, const double wl, const double fluxVega) -> double // NOLINT(readability-identifier-naming) -- see the two-argument overload above
    {
        return PhotConvert<PhotSystem::Flambda, PhotSystem::Vega>(
            PhotConvert<PhotSystem::AB, PhotSystem::Flambda>(photIn, wl), wl, fluxVega);
    }

    /**
     * @brief Convert a Vega magnitude to an AB magnitude
     * @details
     * Vega -> Flambda -> AB (two-argument overload), the algebraic
     * inverse of the AB -> Vega conversion above.
     */
    template <>
    inline auto PhotConvert<PhotSystem::Vega, PhotSystem::AB>(const double photIn, const double wl, const double fluxVega) -> double // NOLINT(readability-identifier-naming) -- see the two-argument overload above
    {
        return PhotConvert<PhotSystem::Flambda, PhotSystem::AB>(
            PhotConvert<PhotSystem::Vega, PhotSystem::Flambda>(photIn, wl, fluxVega), wl);
    }

} // namespace phot

#endif // PHOTCOMMONS_HPP
