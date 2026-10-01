/**
 * @file Winds.cpp
 * @author Mark Krumholz
 * @brief Implementation of the Winds class
 * @date 2026-09-30
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#include "Winds.hpp"
#include "../io/SimControls.hpp"
#include "../specsyn/Specsyn.hpp"
#include "../tracks/TrackCommons.hpp"
#include "../utils/Constants.hpp"
#include "FeedbackCommons.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <stdexcept>

namespace feedback
{
    namespace
    {
        // Nugis & Lamers (2000) calibration sample bounds, converted from
        // km/s to cgs (cm/s) -- see vWindWR's own comment
        constexpr double kmToCm = 1e5;
        constexpr double vWindMin = 740.0 * kmToCm;  // cm/s
        constexpr double vWindMax = 5500.0 * kmToCm; // cm/s

        // Stellar radius, in cm, from the Stefan-Boltzmann law:
        // L = 4 pi R^2 sigma Teff^4
        auto stellarRadius(const double logL, const double logTeff) -> double // NOLINT(llvm-prefer-static-over-anonymous-namespace) -- see vWindLOverC's own identical NOLINT
        {
            const double lCgs = std::pow(10.0, logL) * utils::Lsun; // erg/s
            const double teff = std::pow(10.0, logTeff);            // K
            return std::sqrt(
                lCgs / (4.0 * std::numbers::pi * utils::sigmaSB * teff * teff * teff * teff)); // cm
        }

        // Electron-scattering Eddington factor,
        // Gamma_e = 7.66e-5 (L/Lsun) / (M/Msun) * 0.401 (X + Y/2 + Z/4),
        // with Z = 1 - X - Y -- i.e. sigma_e = 0.401 (X + Y/2 + Z/4)
        // cm^2/g, fully ionized gas (= 0.2 (1 + X) for Z = 0)
        auto eddingtonFactor(const specsyn::Specsyn::StarData& props) -> double // NOLINT(llvm-prefer-static-over-anonymous-namespace) -- see vWindLOverC's own identical NOLINT
        {
            // NOLINTBEGIN(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- StarData is a fixed-size std::array, and every index used here is compile-time-known
            const double mass = props[static_cast<std::size_t>(tracks::FieldIdx::mass)];     // Msun
            const double logL = props[static_cast<std::size_t>(tracks::FieldIdx::logL)];     // log10(L/Lsun)
            const double hSurf = props[static_cast<std::size_t>(tracks::FieldIdx::hSurf)];   // X
            const double heSurf = props[static_cast<std::size_t>(tracks::FieldIdx::heSurf)]; // Y
            // NOLINTEND(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)

            const double zSurf = 1.0 - hSurf - heSurf; // Z
            return 7.66e-5 * std::pow(10.0, logL) / mass * 0.401 *
                (hSurf + (0.5 * heSurf) + (0.25 * zSurf));
        }

        // Vink et al. (2001) bistability jump temperature, in K, at
        // metallicity [Fe/H] -- see vWindOB's own comment
        auto tJump(const double feh) -> double // NOLINT(llvm-prefer-static-over-anonymous-namespace) -- see vWindLOverC's own identical NOLINT
        {
            constexpr double kKToK = 1e3;
            const double logRho = -13.636 + (0.889 * feh);
            return (61.2 + (2.59 * logRho)) * kKToK;
        }

        // WRwindModel::lOverc_: v_wind = L / (mdot c), in cgs -- the
        // same computation as SpecsynLibWR::computeRawLogRt()'s own
        auto vWindLOverC(const specsyn::Specsyn::StarData& props) -> double // NOLINT(llvm-prefer-static-over-anonymous-namespace) -- kept in the same anonymous namespace as vWindMin/vWindMax above, which it shares with vWindNugisLamers00, rather than split across two visibility mechanisms
        {
            // NOLINTBEGIN(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- StarData is a fixed-size std::array, and every index used here is compile-time-known
            const double logL = props[static_cast<std::size_t>(tracks::FieldIdx::logL)];
            const double mdot = props[static_cast<std::size_t>(tracks::FieldIdx::mdot)]; // Msun/yr
            // NOLINTEND(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)

            const double lumCgs = std::pow(10.0, logL) * utils::Lsun; // erg/s
            const double mdotCgs = mdot * utils::Msun / utils::yr;    // g/s
            return lumCgs / (mdotCgs * utils::c);                     // cm/s
        }

        // WRwindModel::nugisLamers00_: see vWindWR's own comment
        auto vWindNugisLamers00(const specsyn::Specsyn::StarData& props) -> double // NOLINT(llvm-prefer-static-over-anonymous-namespace) -- see vWindLOverC's own identical NOLINT
        {
            // NOLINTBEGIN(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- StarData is a fixed-size std::array, and every index used here is compile-time-known
            const double mass = props[static_cast<std::size_t>(tracks::FieldIdx::mass)];  // Msun
            const double logL = props[static_cast<std::size_t>(tracks::FieldIdx::logL)];  // log10(L/Lsun)
            const double logTeff = props[static_cast<std::size_t>(tracks::FieldIdx::logTe)];
            const double hSurf = props[static_cast<std::size_t>(tracks::FieldIdx::hSurf)];   // X
            const double heSurf = props[static_cast<std::size_t>(tracks::FieldIdx::heSurf)]; // Y
            const double cSurf = props[static_cast<std::size_t>(tracks::FieldIdx::cSurf)];
            const double nSurf = props[static_cast<std::size_t>(tracks::FieldIdx::nSurf)];
            // NOLINTEND(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)

            const double zSurf = 1.0 - hSurf - heSurf; // Z
            const double gammaE = eddingtonFactor(props);

            // Gamma_e > 1 (equivalently, 1 - Gamma_e < 0): v_esc's own
            // radicand is negative, so v_esc is undefined -- skip straight
            // to the calibration sample's own lower bound, per instruction,
            // rather than let a NaN pass through std::clamp unchanged
            // below. See vWindWR's own comment.
            if (gammaE > 1.0) { return vWindMin; }

            const double radius = stellarRadius(logL, logTeff); // cm
            const double massCgs = mass * utils::Msun; // g
            const double vEsc = std::sqrt(utils::G * massCgs * (1.0 - gammaE) / radius); // cm/s

            // Nugis & Lamers give separate WN and WC calibrations; a WN
            // formula is used for both WNE and WNL stars (no separate WNL
            // calibration exists), selected by the same C/N surface
            // comparison SpecsynLibWR::getWRType uses to tell WNE/WNL apart
            // from WC (cSurf < nSurf: nitrogen sequence; otherwise: carbon
            // sequence)
            const bool isWC = cSurf >= nSurf;
            const double logRatio = isWC ?
                (-2.37 + (0.43 * logL) - (0.07 * std::log10(zSurf))) :
                (0.61 - (0.13 * logL) + (0.3 * std::log10(heSurf)));

            const double vWind = vEsc * std::pow(10.0, logRatio);

            return std::clamp(vWind, vWindMin, vWindMax);
        }

        // OBwindModel::vink01_: see vWindOB's own comment
        auto vWindVink01(const specsyn::Specsyn::StarData& props, const double feh) -> double // NOLINT(llvm-prefer-static-over-anonymous-namespace) -- see vWindLOverC's own identical NOLINT
        {
            // NOLINTBEGIN(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- StarData is a fixed-size std::array, and every index used here is compile-time-known
            const double mass = props[static_cast<std::size_t>(tracks::FieldIdx::mass)]; // Msun
            const double logL = props[static_cast<std::size_t>(tracks::FieldIdx::logL)]; // log10(L/Lsun)
            const double logTeff = props[static_cast<std::size_t>(tracks::FieldIdx::logTe)];
            // NOLINTEND(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)

            // Gamma_e >= 1: the effective mass M (1 - Gamma_e), and hence
            // the effective escape speed, is zero (or undefined), so
            // there is no wind -- see vWindOB's own comment
            const double gammaE = eddingtonFactor(props);
            if (gammaE >= 1.0) { return 0.0; }

            const double radius = stellarRadius(logL, logTeff); // cm
            const double vEsc = std::sqrt(
                2.0 * utils::G * mass * utils::Msun * (1.0 - gammaE) / radius); // cm/s
            const double ratio = (std::pow(10.0, logTeff) < tJump(feh)) ? 1.3 : 2.6;  // v_wind / v_esc
            return ratio * vEsc * std::pow(10.0, 0.13 * feh);                         // cm/s
        }

        // OBwindModel::vinkSander21_: see vWindOB's own comment
        auto vWindVinkSander21(const specsyn::Specsyn::StarData& props, const double feh) -> double // NOLINT(llvm-prefer-static-over-anonymous-namespace) -- see vWindLOverC's own identical NOLINT
        {
            // NOLINTBEGIN(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- StarData is a fixed-size std::array, and every index used here is compile-time-known
            const double logL = props[static_cast<std::size_t>(tracks::FieldIdx::logL)]; // log10(L/Lsun)
            const double logTeff = props[static_cast<std::size_t>(tracks::FieldIdx::logTe)];
            // NOLINTEND(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)

            const double logV = (std::pow(10.0, logTeff) < tJump(feh)) ?
                (-7.79 - (0.07 * logL) + (2.57 * logTeff) - (0.003 * feh)) :
                (0.39 - (0.04 * logL) + (0.74 * logTeff) + (0.19 * feh)); // log10(v_wind / km/s)
            return std::pow(10.0, logV) * kmToCm;                           // cm/s
        }
    } // namespace

    auto Winds::vWindWR(const specsyn::Specsyn::StarData& props) const -> double
    {
        switch (controls_.wrWindModel())
        {
            case WRwindModel::none_:         return 0.0;
            case WRwindModel::lOverc_:       return vWindLOverC(props);
            case WRwindModel::nugisLamers00_: return vWindNugisLamers00(props);
        }
        throw std::logic_error("Winds::vWindWR: unrecognized WR wind model"); // unreachable; exhaustive switch above
    }

    auto Winds::vWindOB(const specsyn::Specsyn::StarData& props, const double feh) const -> double
    {
        switch (controls_.obWindModel())
        {
            case OBwindModel::none_:         return 0.0;
            case OBwindModel::vink01_:       return vWindVink01(props, feh);
            case OBwindModel::vinkSander21_: return vWindVinkSander21(props, feh);
        }
        throw std::logic_error("Winds::vWindOB: unrecognized OB wind model"); // unreachable; exhaustive switch above
    }

} // namespace feedback
