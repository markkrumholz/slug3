/**
 * @file testPhotCommons.hpp
 * @author Mark Krumholz
 * @brief Unit tests for the definitions in PhotCommons.hpp.
 * @date 2026-07-26
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#ifndef TESTPHOTCOMMONS_HPP
#define TESTPHOTCOMMONS_HPP

#include "../../src/phot/PhotCommons.hpp"
#include "../../src/utils/Constants.hpp"
#include <cmath>
#include <iostream>
#include <numbers>

namespace
{
    // 4 pi (10 pc)^2, in cm^2 -- computed independently of
    // phot::fourPiTenPcSq's own implementation, from the same
    // defining formula, so tests below cross-check that constant
    // rather than assuming it
    constexpr double testFourPiTenPcSq =
        4.0 * std::numbers::pi_v<double> * (10.0 * utils::pc) * (10.0 * utils::pc);

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
 * arbitrary (photIn, wl) point, against expected values computed
 * independently from the defining formulas (not by calling
 * PhotConvert itself), and checks that converting Flambda -> Fnu
 * -> Flambda recovers the original value to within floating-point
 * roundoff. flambda here is treated as a specific luminosity (no
 * particular distance baked in), but the conversion is purely a
 * change of independent variable (wavelength to frequency), so it
 * would give the identical result if flambda were instead a genuine
 * flux -- see PhotConvert<Flambda, Fnu>'s own comment.
 */
inline auto testPhotConvertFlambdaFnu() -> int
{
    constexpr double relTol = 1e-12;
    constexpr double wl = 5000.0; // Angstrom
    constexpr double flambda = 1e-15; // erg/s/Angstrom

    // F_nu = F_lambda * lambda^2 / c, with every quantity converted to
    // cgs (cm, erg/s/cm) before combining -- computed here
    // independently of PhotConvert's own implementation, from the
    // same defining formula
    const double wlCm = wl * utils::Angstrom;
    const double flambdaCgs = flambda / utils::Angstrom; // erg/s/cm (dividing, not multiplying, by the cm/Angstrom width converts a per-Angstrom density to a per-cm density)
    const double expectedFnu = flambdaCgs * wlCm * wlCm / utils::c;

    const double fnu = phot::PhotConvert<phot::PhotSystem::Flambda, phot::PhotSystem::Fnu>(flambda, wl);
    const double relErrFnu = std::abs(fnu - expectedFnu) / std::abs(expectedFnu);
    if (relErrFnu > relTol)
    {
        std::cerr << "testPhotConvertFlambdaFnu: Flambda -> Fnu gave " << fnu
            << " erg/s/Hz, expected " << expectedFnu << " erg/s/Hz (relative error "
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
            << flambdaBack << " erg/s/Angstrom, expected " << flambda
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
 * Checks that the specific luminosity equivalent to flux0AB (by
 * definition, the AB system's zero point) at the standard distance of
 * 10 pc converts to exactly magnitude 0, that a specific luminosity
 * computed from an arbitrary magnitude via the AB -> Fnu direction
 * converts back to that same magnitude via Fnu -> AB, and that the
 * AB -> Fnu / Fnu -> AB pair are algebraic inverses of each other
 * (round trip from an arbitrary luminosity, not just from the zero
 * point).
 */
inline auto testPhotConvertFnuAB() -> int
{
    constexpr double relTol = 1e-12;
    constexpr double wl = 5000.0; // Angstrom, unused by this conversion but required by PhotConvert's signature

    // The specific luminosity (erg/s/Hz) that, once divided by
    // fourPiTenPcSq and by utils::Jy (PhotConvert<Fnu, AB>'s own first
    // two steps), recovers flux0AB (in Jy) exactly
    const double zeroPointLum = (phot::flux0AB * utils::Jy) * testFourPiTenPcSq;
    const double zeroPointMag = phot::PhotConvert<phot::PhotSystem::Fnu, phot::PhotSystem::AB>(zeroPointLum, wl);
    if (std::abs(zeroPointMag) > relTol)
    {
        std::cerr << "testPhotConvertFnuAB: the luminosity equivalent to flux0AB "
            "converted to magnitude " << zeroPointMag << ", expected exactly 0\n";
        return 1;
    }

    constexpr double magIn = 18.5;
    const double fnu = phot::PhotConvert<phot::PhotSystem::AB, phot::PhotSystem::Fnu>(magIn, wl);
    const double expectedFlux = phot::flux0AB * std::pow(10.0, magIn / -2.5); // Jy
    const double expectedFnu = (expectedFlux * utils::Jy) * testFourPiTenPcSq; // erg/s/Hz
    const double relErrFnu = std::abs(fnu - expectedFnu) / std::abs(expectedFnu);
    if (relErrFnu > relTol)
    {
        std::cerr << "testPhotConvertFnuAB: AB -> Fnu gave " << fnu
            << " erg/s/Hz, expected " << expectedFnu << " erg/s/Hz (relative error "
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

    // The specific luminosity (erg/s/Angstrom) that, once divided by
    // fourPiTenPcSq (PhotConvert<Flambda, ST>'s own first step),
    // recovers flux0ST exactly
    const double zeroPointLum = phot::flux0ST * testFourPiTenPcSq;
    const double zeroPointMag = phot::PhotConvert<phot::PhotSystem::Flambda, phot::PhotSystem::ST>(zeroPointLum, wl);
    if (std::abs(zeroPointMag) > relTol)
    {
        std::cerr << "testPhotConvertFlambdaST: the luminosity equivalent to flux0ST "
            "converted to magnitude " << zeroPointMag << ", expected exactly 0\n";
        return 1;
    }

    constexpr double magIn = 18.5;
    const double flambda = phot::PhotConvert<phot::PhotSystem::ST, phot::PhotSystem::Flambda>(magIn, wl);
    const double expectedFlux = phot::flux0ST * std::pow(10.0, magIn / -2.5); // erg/s/cm^2/Angstrom
    const double expectedFlambda = expectedFlux * testFourPiTenPcSq; // erg/s/Angstrom
    const double relErrFlambda = std::abs(flambda - expectedFlambda) / std::abs(expectedFlambda);
    if (relErrFlambda > relTol)
    {
        std::cerr << "testPhotConvertFlambdaST: ST -> Flambda gave " << flambda
            << " erg/s/Angstrom, expected " << expectedFlambda
            << " erg/s/Angstrom (relative error " << relErrFlambda
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
 * @brief Unit test for phot::PhotConvert<Flambda, AB> and <AB, Flambda>
 * @return 0 if the test passes, 1 if it fails.
 * @details
 * These two are composed (Flambda -> Fnu -> AB and its inverse) from
 * the direct Flambda <-> Fnu and Fnu <-> AB conversions tested above,
 * rather than following any defining formula of their own, so this
 * checks the composed AB magnitude against an expected value computed
 * independently of PhotConvert entirely (i.e. not just independently
 * of this specific specialization, the way the other tests above
 * check against an independent formula -- there is no single formula
 * here to independently reimplement other than the same two-step
 * chain), and checks that Flambda -> AB -> Flambda recovers the
 * original value. flambda here is a specific luminosity, so the
 * intermediate Fnu-domain value is too, and picks up the standard-
 * distance/Jy conversion only inside the Fnu -> AB step, exactly as
 * PhotConvert<Fnu, AB> itself does.
 */
inline auto testPhotConvertFlambdaAB() -> int
{
    constexpr double relTol = 1e-9;
    constexpr double wl = 5000.0; // Angstrom
    constexpr double flambda = 1e-15; // erg/s/Angstrom

    const double fnu = (flambda / utils::Angstrom) * (wl * utils::Angstrom) * (wl * utils::Angstrom) / utils::c; // erg/s/Hz
    const double fluxJy = (fnu / testFourPiTenPcSq) / utils::Jy;
    const double expectedMagAB = -2.5 * std::log10(fluxJy / phot::flux0AB);

    const double magAB = phot::PhotConvert<phot::PhotSystem::Flambda, phot::PhotSystem::AB>(flambda, wl);
    const double absErr = std::abs(magAB - expectedMagAB);
    if (absErr > relTol)
    {
        std::cerr << "testPhotConvertFlambdaAB: Flambda -> AB gave " << magAB
            << " mag, expected " << expectedMagAB << " mag (absolute error "
            << absErr << ", tolerance " << relTol << ")\n";
        return 1;
    }

    const double flambdaBack = phot::PhotConvert<phot::PhotSystem::AB, phot::PhotSystem::Flambda>(magAB, wl);
    const double relErrBack = std::abs(flambdaBack - flambda) / std::abs(flambda);
    if (relErrBack > relTol)
    {
        std::cerr << "testPhotConvertFlambdaAB: AB -> Flambda round trip gave "
            << flambdaBack << " erg/s/Angstrom, expected " << flambda
            << " (relative error " << relErrBack << ", tolerance "
            << relTol << ")\n";
        return 1;
    }

    return 0;
}

/**
 * @brief Unit test for phot::PhotConvert<Fnu, ST> and <ST, Fnu>
 * @return 0 if the test passes, 1 if it fails.
 * @details
 * The Fnu/ST analog of testPhotConvertFlambdaAB (composed via Fnu ->
 * Flambda -> ST and its inverse) -- see its own comment for what's
 * checked and why.
 */
inline auto testPhotConvertFnuST() -> int
{
    constexpr double relTol = 1e-9;
    constexpr double wl = 5000.0; // Angstrom
    constexpr double fnu = 1e-3; // erg/s/Hz, a specific luminosity

    const double flambda = fnu * utils::c / (wl * wl * utils::Angstrom); // erg/s/Angstrom
    const double flux = flambda / testFourPiTenPcSq; // erg/s/cm^2/Angstrom
    const double expectedMagST = -2.5 * std::log10(flux / phot::flux0ST);

    const double magST = phot::PhotConvert<phot::PhotSystem::Fnu, phot::PhotSystem::ST>(fnu, wl);
    const double absErr = std::abs(magST - expectedMagST);
    if (absErr > relTol)
    {
        std::cerr << "testPhotConvertFnuST: Fnu -> ST gave " << magST
            << " mag, expected " << expectedMagST << " mag (absolute error "
            << absErr << ", tolerance " << relTol << ")\n";
        return 1;
    }

    const double fnuBack = phot::PhotConvert<phot::PhotSystem::ST, phot::PhotSystem::Fnu>(magST, wl);
    const double relErrBack = std::abs(fnuBack - fnu) / std::abs(fnu);
    if (relErrBack > relTol)
    {
        std::cerr << "testPhotConvertFnuST: ST -> Fnu round trip gave "
            << fnuBack << " erg/s/Hz, expected " << fnu << " erg/s/Hz (relative error "
            << relErrBack << ", tolerance " << relTol << ")\n";
        return 1;
    }

    return 0;
}

/**
 * @brief Unit test for phot::PhotConvert<ST, AB> and <AB, ST>
 * @return 0 if the test passes, 1 if it fails.
 * @details
 * The ST/AB analog of testPhotConvertFlambdaAB (composed via ST ->
 * Flambda -> Fnu -> AB and its inverse, the longest of the composed
 * chains) -- see its own comment for what's checked and why. The
 * standard-distance conversions ST -> Flambda applies and AB -> Fnu
 * later undoes are algebraic inverses of each other (both use the
 * same fourPiTenPcSq), so they cancel overall -- see PhotConvert<ST,
 * AB>'s own comment.
 */
inline auto testPhotConvertSTAB() -> int
{
    constexpr double relTol = 1e-9;
    constexpr double wl = 5000.0; // Angstrom
    constexpr double magST = 18.5;

    const double fluxST = phot::flux0ST * std::pow(10.0, magST / -2.5); // erg/s/cm^2/Angstrom
    const double flambda = fluxST * testFourPiTenPcSq; // erg/s/Angstrom, a specific luminosity
    const double fnu = (flambda / utils::Angstrom) * (wl * utils::Angstrom) * (wl * utils::Angstrom) / utils::c; // erg/s/Hz
    const double fluxAB = (fnu / testFourPiTenPcSq) / utils::Jy; // Jy
    const double expectedMagAB = -2.5 * std::log10(fluxAB / phot::flux0AB);

    const double magAB = phot::PhotConvert<phot::PhotSystem::ST, phot::PhotSystem::AB>(magST, wl);
    const double absErr = std::abs(magAB - expectedMagAB);
    if (absErr > relTol)
    {
        std::cerr << "testPhotConvertSTAB: ST -> AB gave " << magAB
            << " mag, expected " << expectedMagAB << " mag (absolute error "
            << absErr << ", tolerance " << relTol << ")\n";
        return 1;
    }

    const double magSTBack = phot::PhotConvert<phot::PhotSystem::AB, phot::PhotSystem::ST>(magAB, wl);
    const double absErrBack = std::abs(magSTBack - magST);
    if (absErrBack > relTol)
    {
        std::cerr << "testPhotConvertSTAB: AB -> ST round trip gave "
            << magSTBack << " mag, expected " << magST << " mag (absolute error "
            << absErrBack << ", tolerance " << relTol << ")\n";
        return 1;
    }

    return 0;
}

/**
 * @brief Unit test for phot::PhotConvert<Flambda, Vega> and <Vega, Flambda>
 * @return 0 if the test passes, 1 if it fails.
 * @details
 * The Flambda/Vega analog of testPhotConvertFlambdaST -- see its own
 * comment for what's checked and why. fluxVega plays the role
 * flux0ST plays there: converting the specific luminosity equivalent
 * to fluxVega at the standard distance of 10 pc gives magnitude 0 by
 * construction, but here that zero point is a per-call argument (a
 * representative real Vega V-band flux density, from CALSPEC) rather
 * than a fixed library constant, since Vega's own zero point is
 * filter-dependent. fluxVega itself is always a genuine flux, never
 * scaled by fourPiTenPcSq -- see PhotConvert<Flambda, Vega>'s own
 * comment.
 */
inline auto testPhotConvertFlambdaVega() -> int
{
    constexpr double relTol = 1e-12;
    constexpr double wl = 5000.0; // Angstrom, unused by this conversion but required by PhotConvert's signature
    constexpr double fluxVega = 3.44e-9; // erg/s/cm^2/Angstrom, representative real Vega V-band flux density (CALSPEC)

    const double zeroPointLum = fluxVega * testFourPiTenPcSq;
    const double zeroPointMag = phot::PhotConvert<phot::PhotSystem::Flambda, phot::PhotSystem::Vega>(zeroPointLum, wl, fluxVega);
    if (std::abs(zeroPointMag) > relTol)
    {
        std::cerr << "testPhotConvertFlambdaVega: the luminosity equivalent to fluxVega "
            "converted to magnitude " << zeroPointMag << ", expected exactly 0\n";
        return 1;
    }

    constexpr double magIn = 18.5;
    const double flambda = phot::PhotConvert<phot::PhotSystem::Vega, phot::PhotSystem::Flambda>(magIn, wl, fluxVega);
    const double expectedFlux = fluxVega * std::pow(10.0, magIn / -2.5); // erg/s/cm^2/Angstrom
    const double expectedFlambda = expectedFlux * testFourPiTenPcSq; // erg/s/Angstrom
    const double relErrFlambda = std::abs(flambda - expectedFlambda) / std::abs(expectedFlambda);
    if (relErrFlambda > relTol)
    {
        std::cerr << "testPhotConvertFlambdaVega: Vega -> Flambda gave " << flambda
            << " erg/s/Angstrom, expected " << expectedFlambda
            << " erg/s/Angstrom (relative error " << relErrFlambda
            << ", tolerance " << relTol << ")\n";
        return 1;
    }

    const double magBack = phot::PhotConvert<phot::PhotSystem::Flambda, phot::PhotSystem::Vega>(flambda, wl, fluxVega);
    const double absErrBack = std::abs(magBack - magIn);
    if (absErrBack > relTol)
    {
        std::cerr << "testPhotConvertFlambdaVega: Flambda -> Vega round trip gave "
            << magBack << " mag, expected " << magIn
            << " mag (absolute error " << absErrBack << ", tolerance "
            << relTol << ")\n";
        return 1;
    }

    return 0;
}

/**
 * @brief Sanity check: the same input converts to similar, but not identical, ST and Vega magnitudes
 * @return 0 if the test passes, 1 if it fails.
 * @details
 * flux0ST (the ST system's fixed zero point) was historically chosen
 * to be close to Vega's own flux, so converting an arbitrary Flambda
 * value to both systems should give magnitudes that agree to within
 * about a tenth of a magnitude. fluxVega here (a representative real
 * Vega V-band flux density from CALSPEC) is deliberately not
 * numerically equal to flux0ST, so the two magnitudes should still
 * differ by a small but distinctly nonzero amount -- a difference of
 * exactly 0 would indicate the Vega conversion is accidentally using
 * flux0ST (or some other fixed constant) rather than the fluxVega
 * argument actually passed in. magST - magVega is, algebraically,
 * -2.5*log10(fluxVega / flux0ST), independent of flambda's own actual
 * value (both conversions divide it by the same fourPiTenPcSq before
 * comparing against their own zero point, and that shared factor
 * cancels in the difference) -- so this comparison is unaffected by
 * whether flambda is a genuine flux or a specific luminosity, or by
 * how large it is.
 */
inline auto testPhotConvertVegaSanityCheck() -> int
{
    constexpr double wl = 5500.0; // Angstrom, roughly V band
    constexpr double flambda = 3.5e-9; // erg/s/Angstrom, arbitrary -- see this function's own comment for why its value doesn't matter here
    constexpr double fluxVega = 3.44e-9; // erg/s/cm^2/Angstrom, representative real Vega V-band flux density (CALSPEC)

    const double magST = phot::PhotConvert<phot::PhotSystem::Flambda, phot::PhotSystem::ST>(flambda, wl);
    const double magVega = phot::PhotConvert<phot::PhotSystem::Flambda, phot::PhotSystem::Vega>(flambda, wl, fluxVega);
    const double diff = std::abs(magST - magVega);

    constexpr double maxDiff = 0.1;
    if (diff > maxDiff)
    {
        std::cerr << "testPhotConvertVegaSanityCheck: ST mag " << magST
            << " and Vega mag " << magVega << " differ by " << diff
            << " mag, expected at most " << maxDiff << " mag\n";
        return 1;
    }
    constexpr double minDiff = 1e-6;
    if (diff < minDiff)
    {
        std::cerr << "testPhotConvertVegaSanityCheck: ST mag " << magST
            << " and Vega mag " << magVega << " are essentially identical "
            "(difference " << diff << " mag); expected a small but "
            "nonzero difference, since flux0ST and fluxVega are not "
            "the same value\n";
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
    result += testPhotConvertFlambdaAB();
    result += testPhotConvertFnuST();
    result += testPhotConvertSTAB();
    result += testPhotConvertFlambdaVega();
    result += testPhotConvertVegaSanityCheck();
    return result;
}

#endif // TESTPHOTCOMMONS_HPP
