/**
 * @file testFilterIdeal.hpp
 * @author Mark Krumholz
 * @brief Unit tests for phot::FilterIdeal
 * @date 2026-07-31
 */

#ifndef TESTFILTERIDEAL_HPP
#define TESTFILTERIDEAL_HPP

#include "../../src/elem/ElemData.hpp"
#include "../../src/phot/FilterIdeal.hpp"
#include "../../src/utils/Constants.hpp"
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace
{
    // Build a linear spectrum F(λ) = a + b·λ on a uniform grid of n points
    // over [wlLo, wlHi] Angstrom.  For this spectrum, the Steffen
    // interpolant is exact at all interior points (uniform-grid finite
    // differences reproduce the linear derivative exactly), so integration
    // errors are at floating-point machine precision rather than at the
    // O(h^4) truncation limit of cubic quadrature.
    auto makeLinearSpec(double wlLo, double wlHi, std::size_t n,
                        double a, double b)
        -> std::pair<std::vector<double>, std::vector<double>>
    {
        std::vector<double> wl(n), spec(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            wl[i]   = wlLo + (wlHi - wlLo)
                      * static_cast<double>(i) / static_cast<double>(n - 1);
            spec[i] = a + b * wl[i];
        }
        wl.back()   = wlHi;          // pin endpoint to avoid round-off
        spec.back() = a + b * wlHi;
        return {std::move(wl), std::move(spec)};
    }

    // Analytical mean flux density on a log-wavelength basis:
    // [∫ (a + b·λ)/λ dλ] / ln(lam1/lam0) = [a·ln(lam1/lam0) +
    // b·(lam1 - lam0)] / ln(lam1/lam0), from lam0 to lam1 -- matches
    // FilterIdeal::phot()'s own ln(wavelength)-domain mean convention
    // (see its own comment for why, mirroring FilterTabulated)
    auto energyExactLnMean(double a, double b, double lam0, double lam1) -> double
    {
        return a + b * (lam1 - lam0) / std::log(lam1 / lam0);
    }

    // Analytical photon-count integral ∫ (a + b·λ)·λ·Å/(h·c) dλ
    // from lam0 to lam1 (wavelengths in Angstrom)
    auto photExact(double a, double b, double lam0, double lam1) -> double
    {
        const double term = 0.5 * a * (lam1 * lam1 - lam0 * lam0)
                          + (b / 3.0) * (lam1 * lam1 * lam1
                                         - lam0 * lam0 * lam0);
        return utils::Angstrom / (utils::h * utils::c) * term;
    }
} // namespace

/**
 * @brief Test ideal_energy_X_Y filter against an analytical log-mean flux density
 * @return 0 on pass, 1 on failure
 * @details
 * Constructs a linear spectrum F(λ) = a + b·λ on [500, 2000] Å, then
 * checks that ideal_energy_700_1500 returns the mean flux density
 * over its passband on a log-wavelength basis -- [a·ln(1500/700) +
 * b·(1500−700)] / ln(1500/700) -- since phot() integrates in
 * ln(wavelength) (matching FilterTabulated's own convention), not
 * linear wavelength, to within 1e-8 relative tolerance.
 */
inline auto testFilterIdealEnergyMode() -> int
{
    constexpr double wlLo = 500.0, wlHi = 2000.0;
    constexpr double a = 2.0, b = 1.0e-3;
    constexpr std::size_t n = 1001;
    const auto [wl, spec] = makeLinearSpec(wlLo, wlHi, n, a, b);

    const phot::FilterIdeal f("ideal_energy_700_1500");
    const double got      = f.phot(wl, spec);
    const double expected = energyExactLnMean(a, b, 700.0, 1500.0);
    const double relErr   = std::abs(got - expected) / std::abs(expected);
    constexpr double tol  = 1e-8;
    if (relErr > tol)
    {
        std::cerr << "testFilterIdealEnergyMode: got " << got
                  << ", expected " << expected
                  << " (rel err " << relErr << ", tol " << tol << ")\n";
        return 1;
    }
    return 0;
}

/**
 * @brief Test ideal_phot_X_Y filter against an analytical photon-count integral
 * @return 0 on pass, 1 on failure
 * @details
 * Constructs the same linear spectrum as testFilterIdealEnergyMode, then
 * checks that ideal_phot_700_1500 integrates the photon-flux density
 * F(λ)·λ·Å/(h·c) to the analytical value to within 1e-8 relative tolerance.
 */
inline auto testFilterIdealPhotMode() -> int
{
    constexpr double wlLo = 500.0, wlHi = 2000.0;
    constexpr double a = 2.0, b = 1.0e-3;
    constexpr std::size_t n = 1001;
    const auto [wl, spec] = makeLinearSpec(wlLo, wlHi, n, a, b);

    const phot::FilterIdeal f("ideal_phot_700_1500");
    const double got      = f.phot(wl, spec);
    const double expected = photExact(a, b, 700.0, 1500.0);
    const double relErr   = std::abs(got - expected) / std::abs(expected);
    constexpr double tol  = 1e-8;
    if (relErr > tol)
    {
        std::cerr << "testFilterIdealPhotMode: got " << got
                  << ", expected " << expected
                  << " (rel err " << relErr << ", tol " << tol << ")\n";
        return 1;
    }
    return 0;
}

/**
 * @brief Test Q(*) filters against analytical photon-count integrals
 * @return 0 on pass, 1 on failure
 * @details
 * Checks Q(HI) and Q(HeII) by verifying:
 * (1) photCount() is true and wlMax() is infinity;
 * (2) wlMin() matches h·c/(IP·eV)/Å computed directly from elem::elemData;
 * (3) phot() on the linear test spectrum matches the analytical photon-count
 *     integral from wlMin() to the spectrum's upper bound (2000 Å) to
 *     within 1e-8 relative tolerance.
 */
inline auto testFilterIdealQMode() -> int
{
    constexpr double wlLo = 500.0, wlHi = 2000.0;
    constexpr double a = 2.0, b = 1.0e-3;
    constexpr std::size_t n = 1001;
    const auto [wl, spec] = makeLinearSpec(wlLo, wlHi, n, a, b);

    constexpr double wlTol = 1e-6; // Angstrom
    constexpr double tol   = 1e-8;

    // --- Q(HI): neutral hydrogen, first ionization ---
    const phot::FilterIdeal fHI("Q(HI)");
    if (!fHI.photCount())
    {
        std::cerr << "testFilterIdealQMode: Q(HI) photCount() should be true\n";
        return 1;
    }
    if (!std::isinf(fHI.wlMax()))
    {
        std::cerr << "testFilterIdealQMode: Q(HI) wlMax() should be infinity\n";
        return 1;
    }

    const double wlH1 = (utils::h * utils::c)
        / (elem::elemData[0].ionPot()[0] * utils::eV) / utils::Angstrom;
    if (std::abs(fHI.wlMin() - wlH1) > wlTol)
    {
        std::cerr << "testFilterIdealQMode: Q(HI) wlMin() = " << fHI.wlMin()
                  << " Ang, expected " << wlH1 << " Ang\n";
        return 1;
    }

    // wlMax = inf is clamped to spectrum end; wlMin is clamped to spectrum
    // start if the threshold falls below it -- mirror both clamps in the
    // expected value so the comparison is against the same effective range
    const double gotHI  = fHI.phot(wl, spec);
    const double expHI  = photExact(a, b, std::max(wlH1, wlLo), wlHi);
    const double relHI  = std::abs(gotHI - expHI) / std::abs(expHI);
    if (relHI > tol)
    {
        std::cerr << "testFilterIdealQMode: Q(HI) phot got " << gotHI
                  << ", expected " << expHI
                  << " (rel err " << relHI << ", tol " << tol << ")\n";
        return 1;
    }

    // --- Q(HeII): singly-ionized helium, second ionization ---
    // The threshold (~228 Å) lies below the spectrum's lower bound (500 Å),
    // so the effective integration range is [500, 2000] Å
    const phot::FilterIdeal fHeII("Q(HeII)");
    if (!fHeII.photCount())
    {
        std::cerr << "testFilterIdealQMode: Q(HeII) photCount() should be true\n";
        return 1;
    }

    const double wlHeII = (utils::h * utils::c)
        / (elem::elemData[1].ionPot()[1] * utils::eV) / utils::Angstrom;
    if (std::abs(fHeII.wlMin() - wlHeII) > wlTol)
    {
        std::cerr << "testFilterIdealQMode: Q(HeII) wlMin() = " << fHeII.wlMin()
                  << " Ang, expected " << wlHeII << " Ang\n";
        return 1;
    }

    const double gotHeII = fHeII.phot(wl, spec);
    const double expHeII = photExact(a, b, std::max(wlHeII, wlLo), wlHi);
    const double relHeII = std::abs(gotHeII - expHeII) / std::abs(expHeII);
    if (relHeII > tol)
    {
        std::cerr << "testFilterIdealQMode: Q(HeII) phot got " << gotHeII
                  << ", expected " << expHeII
                  << " (rel err " << relHeII << ", tol " << tol << ")\n";
        return 1;
    }

    return 0;
}

/**
 * @brief Test that filters with ranges entirely outside the spectrum return 0
 * @return 0 on pass, 1 on failure
 */
inline auto testFilterIdealOutOfRange() -> int
{
    constexpr double wlLo = 500.0, wlHi = 2000.0;
    constexpr std::size_t n = 101;
    const auto [wl, spec] = makeLinearSpec(wlLo, wlHi, n, 2.0, 1.0e-3);

    // Filter entirely above spectrum range
    const phot::FilterIdeal fHigh("ideal_energy_2500_3000");
    if (fHigh.phot(wl, spec) != 0.0)
    {
        std::cerr << "testFilterIdealOutOfRange: filter [2500,3000] above "
                     "spectrum [500,2000] should return 0\n";
        return 1;
    }

    // Filter entirely below spectrum range
    const phot::FilterIdeal fLow("ideal_phot_100_400");
    if (fLow.phot(wl, spec) != 0.0)
    {
        std::cerr << "testFilterIdealOutOfRange: filter [100,400] below "
                     "spectrum [500,2000] should return 0\n";
        return 1;
    }

    return 0;
}

/**
 * @brief Test that invalid Q(*) and ideal_* names throw std::runtime_error
 * @return 0 on pass, 1 on failure
 */
inline auto testFilterIdealErrors() -> int
{
    // Unknown element symbol
    try {
        const phot::FilterIdeal f("Q(ZzI)");
        std::cerr << "testFilterIdealErrors: Q(ZzI) should have thrown\n";
        return 1;
    } catch (const std::runtime_error&) {}

    // Missing ionization state (empty Roman numeral after element)
    try {
        const phot::FilterIdeal f("Q(H)");
        std::cerr << "testFilterIdealErrors: Q(H) should have thrown\n";
        return 1;
    } catch (const std::runtime_error&) {}

    // NaN ionization potential: Ta has only 1 CRC value, XX (20th) is NaN
    try {
        const phot::FilterIdeal f("Q(TaXX)");
        std::cerr << "testFilterIdealErrors: Q(TaXX) should have thrown\n";
        return 1;
    } catch (const std::runtime_error&) {}

    // Malformed ideal_energy name (too few underscore-separated tokens)
    try {
        const phot::FilterIdeal f("ideal_energy_500");
        std::cerr << "testFilterIdealErrors: ideal_energy_500 should have thrown\n";
        return 1;
    } catch (const std::runtime_error&) {}

    // Energy-flux filter with a non-positive wlMin: phot() integrates
    // in ln(wavelength), which is undefined at/below 0
    try {
        const phot::FilterIdeal f("ideal_energy_0_1000");
        std::cerr << "testFilterIdealErrors: ideal_energy_0_1000 should have thrown\n";
        return 1;
    } catch (const std::runtime_error&) {}

    // Energy-flux filter with an infinite wlMax: ln(wlMax/wlMin)
    // would be infinite, so the mean is undefined
    try {
        const phot::FilterIdeal f("ideal_energy_500_inf");
        std::cerr << "testFilterIdealErrors: ideal_energy_500_inf should have thrown\n";
        return 1;
    } catch (const std::runtime_error&) {}

    // The same two ranges are fine for a photon-count filter, where
    // phot() is a raw dλ integral of a count rate, not a log-mean
    // density -- neither should throw
    try {
        const phot::FilterIdeal f1("ideal_phot_0_1000");
        const phot::FilterIdeal f2("ideal_phot_500_inf");
    } catch (const std::runtime_error& e) {
        std::cerr << "testFilterIdealErrors: a photCount filter with wlMin = 0 "
            "or wlMax = inf should not have thrown, but got: " << e.what() << "\n";
        return 1;
    }

    // The direct (name, wlMin, wlMax, photCount) constructor enforces
    // the same restriction for photCount = false
    try {
        const phot::FilterIdeal f("direct_bad_range", 0.0, 1000.0, false);
        std::cerr << "testFilterIdealErrors: direct constructor with wlMin = 0, "
            "photCount = false should have thrown\n";
        return 1;
    } catch (const std::runtime_error&) {}

    return 0;
}

/**
 * @brief Run all FilterIdeal unit tests
 * @return 0 if all tests pass, positive count of failures otherwise
 */
inline auto testFilterIdeal() -> int
{
    int result = 0;
    result += testFilterIdealEnergyMode();
    result += testFilterIdealPhotMode();
    result += testFilterIdealQMode();
    result += testFilterIdealOutOfRange();
    result += testFilterIdealErrors();
    return result;
}

#endif // TESTFILTERIDEAL_HPP
