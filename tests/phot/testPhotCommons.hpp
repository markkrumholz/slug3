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
    // PhotConvertPhys is actually usable in a constant expression --
    // proof that it and everything it depends on (the utils::
    // constants it's built from) are genuinely constexpr, not just
    // callable at runtime despite being marked so
    static_assert(phot::PhotConvertPhys<phot::PhotSystem::Flambda, phot::PhotSystem::Fnu>(1.0, 5000.0) > 0.0);
    static_assert(phot::PhotConvertPhys<phot::PhotSystem::Fnu, phot::PhotSystem::Flambda>(1.0, 5000.0) > 0.0);
} // namespace

/**
 * @brief Unit test for phot::PhotConvertPhys
 * @return 0 if the test passes, 1 if it fails.
 * @details
 * Checks the Flambda -> Fnu and Fnu -> Flambda conversions, at an
 * arbitrary (fluxIn, wl) point, against expected values computed
 * independently from the defining formulas (not by calling
 * PhotConvertPhys itself), and checks that converting Flambda -> Fnu
 * -> Flambda recovers the original flux to within floating-point
 * roundoff.
 */
inline auto testPhotConvertPhys() -> int
{
    constexpr double relTol = 1e-12;
    constexpr double wl = 5000.0; // Angstrom
    constexpr double flambda = 1e-15; // erg/s/cm^2/Angstrom

    // F_nu = F_lambda * lambda^2 / c, with every quantity converted to
    // cgs (cm, erg/s/cm^2/cm) before combining, and the result
    // converted from erg/s/cm^2/Hz to Jy at the end -- computed here
    // independently of PhotConvertPhys's own implementation, from the
    // same defining formula, and cross-checked against the standard
    // F_nu[Jy] = 3.336e4 * lambda[Angstrom]^2 * F_lambda[erg/s/cm^2/Angstrom]
    // conversion (utils::Angstrom / utils::c / utils::Jy = 3.336e4)
    const double wlCm = wl * utils::Angstrom;
    const double flambdaCgs = flambda / utils::Angstrom; // erg/s/cm^2/cm (dividing, not multiplying, by the cm/Angstrom width converts a per-Angstrom density to a per-cm density)
    const double expectedFnu = (flambdaCgs * wlCm * wlCm / utils::c) / utils::Jy;

    const double fnu = phot::PhotConvertPhys<phot::PhotSystem::Flambda, phot::PhotSystem::Fnu>(flambda, wl);
    const double relErrFnu = std::abs(fnu - expectedFnu) / std::abs(expectedFnu);
    if (relErrFnu > relTol)
    {
        std::cerr << "testPhotConvertPhys: Flambda -> Fnu gave " << fnu
            << " Jy, expected " << expectedFnu << " Jy (relative error "
            << relErrFnu << ", tolerance " << relTol << ")\n";
        return 1;
    }

    // The inverse conversion, from the fnu just computed, should
    // recover flambda exactly (both are the same closed-form
    // expression solved for the other variable, so this isn't merely
    // an approximate round-trip)
    const double flambdaBack = phot::PhotConvertPhys<phot::PhotSystem::Fnu, phot::PhotSystem::Flambda>(fnu, wl);
    const double relErrBack = std::abs(flambdaBack - flambda) / std::abs(flambda);
    if (relErrBack > relTol)
    {
        std::cerr << "testPhotConvertPhys: Fnu -> Flambda round trip gave "
            << flambdaBack << " erg/s/cm^2/Angstrom, expected " << flambda
            << " (relative error " << relErrBack << ", tolerance "
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
    return testPhotConvertPhys();
}

#endif // TESTPHOTCOMMONS_HPP
