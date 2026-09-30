/**
 * @file Winds.cpp
 * @author Mark Krumholz
 * @brief Implementation of the Winds class
 * @date 2026-09-30
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#include "Winds.hpp"
#include "../specsyn/Specsyn.hpp"
#include "../tracks/TrackCommons.hpp"
#include "../utils/Constants.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numbers>

namespace feedback
{
    namespace
    {
        // Nugis & Lamers (2000) calibration sample bounds, converted from
        // km/s to cgs (cm/s) -- see vWindWR's own comment
        constexpr double kmToCm = 1e5;
        constexpr double vWindMin = 740.0 * kmToCm;  // cm/s
        constexpr double vWindMax = 5500.0 * kmToCm; // cm/s
    } // namespace

    auto Winds::vWindWR(const specsyn::Specsyn::StarData& props) const -> double // NOLINT(readability-convert-member-functions-to-static) -- doesn't yet read controls_, but genuinely will once more than one wind-model choice exists for it to select among (see Winds.hpp's own comment on controls_); kept an instance method now rather than static, to avoid an API break when that lands
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

        const double lLsun = std::pow(10.0, logL);
        const double gammaE = 7.66e-5 * lLsun / mass * 0.401 *
            (hSurf + (0.5 * heSurf) + (0.25 * zSurf));

        // Gamma_e > 1 (equivalently, 1 - Gamma_e < 0): v_esc's own
        // radicand is negative, so v_esc is undefined -- skip straight
        // to the calibration sample's own lower bound, per instruction,
        // rather than let a NaN pass through std::clamp unchanged
        // below. See this method's own comment.
        if (gammaE > 1.0) { return vWindMin; }

        // Stellar radius from the Stefan-Boltzmann law: L = 4 pi R^2 sigma Teff^4
        const double lCgs = lLsun * utils::Lsun;               // erg/s
        const double teff = std::pow(10.0, logTeff);           // K
        const double radius = std::sqrt(
            lCgs / (4.0 * std::numbers::pi * utils::sigmaSB * teff * teff * teff * teff)); // cm

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

} // namespace feedback
