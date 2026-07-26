/**
 * @file testPhotCommons.hpp
 * @author Mark Krumholz
 * @brief Unit tests for the definitions in PhotCommons.hpp.
 * @date 2026-07-26
 */

#ifndef TESTPHOTCOMMONS_HPP
#define TESTPHOTCOMMONS_HPP

#include "../../src/phot/PhotCommons.hpp"
#include "../../src/utils/Constants.hpp"
#include <cmath>
#include <iostream>

namespace
{
    // A static_assert (rather than a runtime check) that
    // PhotConvert is actually usable in a constant expression --
    // proof that it and everything it depends on (the utils::
    // constants it's built from) are genuinely constexpr, not just
    // callable at runtime despite being marked so
    static_assert(phot::PhotConvert<phot::PhotSystem::Flambda, phot::PhotSystem::Fnu>(1.0, 5000.0) > 0.0);
    static_assert(phot::PhotConvert<phot::PhotSystem::Fnu, phot::PhotSystem::Flambda>(1.0, 5000.0) > 0.0);
} // namespace

/**
 * @brief Unit test for phot::PhotConvert<Flambda, Fnu> and <Fnu, Flambda>
 * @return 0 if the test passes, 1 if it fails.
 * @details
 * Checks the Flambda -> Fnu and Fnu -> Flambda conversions, at an
 * arbitrary (fluxIn, wl) point, against expected values computed
 * independently from the defining formulas (not by calling
 * PhotConvert itself), and checks that converting Flambda -> Fnu
 * -> Flambda recovers the original flux to within floating-point
 * roundoff.
 */
inline auto testPhotConvertFlambdaFnu() -> int
{
    constexpr double relTol = 1e-12;
    constexpr double wl = 5000.0; // Angstrom
    constexpr double flambda = 1e-15; // erg/s/cm^2/Angstrom

    // F_nu = F_lambda * lambda^2 / c, with every quantity converted to
    // cgs (cm, erg/s/cm^2/cm) before combining, and the result
    // converted from erg/s/cm^2/Hz to Jy at the end -- computed here
    // independently of PhotConvert's own implementation, from the
    // same defining formula, and cross-checked against the standard
    // F_nu[Jy] = 3.336e4 * lambda[Angstrom]^2 * F_lambda[erg/s/cm^2/Angstrom]
    // conversion (utils::Angstrom / utils::c / utils::Jy = 3.336e4)
    const double wlCm = wl * utils::Angstrom;
    const double flambdaCgs = flambda / utils::Angstrom; // erg/s/cm^2/cm (dividing, not multiplying, by the cm/Angstrom width converts a per-Angstrom density to a per-cm density)
    const double expectedFnu = (flambdaCgs * wlCm * wlCm / utils::c) / utils::Jy;

    const double fnu = phot::PhotConvert<phot::PhotSystem::Flambda, phot::PhotSystem::Fnu>(flambda, wl);
    const double relErrFnu = std::abs(fnu - expectedFnu) / std::abs(expectedFnu);
    if (relErrFnu > relTol)
    {
        std::cerr << "testPhotConvertFlambdaFnu: Flambda -> Fnu gave " << fnu
            << " Jy, expected " << expectedFnu << " Jy (relative error "
            << relErrFnu << ", tolerance " << relTol << ")\n";
        return 1;
    }

    // The inverse conversion, from the fnu just computed, should
    // recover flambda exactly (both are the same closed-form
    // expression solved for the other variable, so this isn't merely
    // an approximate round-trip)
    const double flambdaBack = phot::PhotConvert<phot::PhotSystem::Fnu, phot::PhotSystem::Flambda>(fnu, wl);
    const double relErrBack = std::abs(flambdaBack - flambda) / std::abs(flambda);
    if (relErrBack > relTol)
    {
        std::cerr << "testPhotConvertFlambdaFnu: Fnu -> Flambda round trip gave "
            << flambdaBack << " erg/s/cm^2/Angstrom, expected " << flambda
            << " (relative error " << relErrBack << ", tolerance "
            << relTol << ")\n";
        return 1;
    }

    return 0;
}

/**
 * @brief Unit test for phot::PhotConvert<Fnu, AB> and <AB, Fnu>
 * @return 0 if the test passes, 1 if it fails.
 * @details
 * Checks that flux0AB itself (by definition, the AB system's zero
 * point) converts to exactly magnitude 0, that a flux computed from
 * an arbitrary magnitude via the AB -> Fnu direction converts back to
 * that same magnitude via Fnu -> AB, and that the AB -> Fnu / Fnu ->
 * AB pair are algebraic inverses of each other (round trip from an
 * arbitrary flux, not just from flux0AB).
 */
inline auto testPhotConvertFnuAB() -> int
{
    constexpr double relTol = 1e-12;
    constexpr double wl = 5000.0; // Angstrom, unused by this conversion but required by PhotConvert's signature

    const double zeroPointMag = phot::PhotConvert<phot::PhotSystem::Fnu, phot::PhotSystem::AB>(phot::flux0AB, wl);
    if (std::abs(zeroPointMag) > relTol)
    {
        std::cerr << "testPhotConvertFnuAB: flux0AB converted to magnitude "
            << zeroPointMag << ", expected exactly 0\n";
        return 1;
    }

    constexpr double magIn = 18.5;
    const double fnu = phot::PhotConvert<phot::PhotSystem::AB, phot::PhotSystem::Fnu>(magIn, wl);
    const double expectedFnu = phot::flux0AB * std::pow(10.0, magIn / -2.5);
    const double relErrFnu = std::abs(fnu - expectedFnu) / std::abs(expectedFnu);
    if (relErrFnu > relTol)
    {
        std::cerr << "testPhotConvertFnuAB: AB -> Fnu gave " << fnu
            << " Jy, expected " << expectedFnu << " Jy (relative error "
            << relErrFnu << ", tolerance " << relTol << ")\n";
        return 1;
    }

    const double magBack = phot::PhotConvert<phot::PhotSystem::Fnu, phot::PhotSystem::AB>(fnu, wl);
    const double absErrBack = std::abs(magBack - magIn);
    if (absErrBack > relTol)
    {
        std::cerr << "testPhotConvertFnuAB: Fnu -> AB round trip gave "
            << magBack << " mag, expected " << magIn
            << " mag (absolute error " << absErrBack << ", tolerance "
            << relTol << ")\n";
        return 1;
    }

    return 0;
}

/**
 * @brief Unit test for phot::PhotConvert<Flambda, ST> and <ST, Flambda>
 * @return 0 if the test passes, 1 if it fails.
 * @details
 * The Flambda/ST analog of testPhotConvertFnuAB -- see its own
 * comment for what's checked and why.
 */
inline auto testPhotConvertFlambdaST() -> int
{
    constexpr double relTol = 1e-12;
    constexpr double wl = 5000.0; // Angstrom, unused by this conversion but required by PhotConvert's signature

    const double zeroPointMag = phot::PhotConvert<phot::PhotSystem::Flambda, phot::PhotSystem::ST>(phot::flux0ST, wl);
    if (std::abs(zeroPointMag) > relTol)
    {
        std::cerr << "testPhotConvertFlambdaST: flux0ST converted to magnitude "
            << zeroPointMag << ", expected exactly 0\n";
        return 1;
    }

    constexpr double magIn = 18.5;
    const double flambda = phot::PhotConvert<phot::PhotSystem::ST, phot::PhotSystem::Flambda>(magIn, wl);
    const double expectedFlambda = phot::flux0ST * std::pow(10.0, magIn / -2.5);
    const double relErrFlambda = std::abs(flambda - expectedFlambda) / std::abs(expectedFlambda);
    if (relErrFlambda > relTol)
    {
        std::cerr << "testPhotConvertFlambdaST: ST -> Flambda gave " << flambda
            << " erg/s/cm^2/Angstrom, expected " << expectedFlambda
            << " erg/s/cm^2/Angstrom (relative error " << relErrFlambda
            << ", tolerance " << relTol << ")\n";
        return 1;
    }

    const double magBack = phot::PhotConvert<phot::PhotSystem::Flambda, phot::PhotSystem::ST>(flambda, wl);
    const double absErrBack = std::abs(magBack - magIn);
    if (absErrBack > relTol)
    {
        std::cerr << "testPhotConvertFlambdaST: Flambda -> ST round trip gave "
            << magBack << " mag, expected " << magIn
            << " mag (absolute error " << absErrBack << ", tolerance "
            << relTol << ")\n";
        return 1;
    }

    return 0;
}

/**
 * @brief Unit tests for the definitions in PhotCommons.hpp
 * @return 0 if the test passes, 1 if it fails.
 */
inline auto testPhotCommons() -> int
{
    int result = 0;
    result += testPhotConvertFlambdaFnu();
    result += testPhotConvertFnuAB();
    result += testPhotConvertFlambdaST();
    return result;
}

#endif // TESTPHOTCOMMONS_HPP
