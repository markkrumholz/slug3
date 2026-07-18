/**
 * @file SpecsynBlackbody.cpp
 * @author Mark Krumholz
 * @brief Implementation of SpecsynBlackbody
 * @date 2026-07-18
 */

#include "SpecsynBlackbody.hpp"
#include "../tracks/TrackCommons.hpp"
#include "Specsyn.hpp"
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

// Unit conversions between cm and Angstrom (1 Angstrom = 1e-8 cm)
static constexpr double angstromToCm = 1e-8;
static constexpr double cmToAngstrom = 1e8;

specsyn::SpecsynBlackbody::SpecsynBlackbody(const double z) : Specsyn(z)
{
    // Wavelength range corresponding to photon energies from 10 Ry
    // down to 0.01 Ry, converted from cm to Angstrom
    const double wlMin = (planckH * speedOfLight / (10.0 * rydberg)) * cmToAngstrom;
    const double wlMax = (planckH * speedOfLight / (0.01 * rydberg)) * cmToAngstrom;

    wl_.resize(nWl);
    const double logWlMin = std::log(wlMin);
    const double logWlMax = std::log(wlMax);
    const double dLogWl = (logWlMax - logWlMin) / static_cast<double>(nWl - 1);
    for (std::size_t i = 0; i < nWl; ++i)
    {
        wl_.at(i) = std::exp(logWlMin + (static_cast<double>(i) * dLogWl));
    }
}

auto specsyn::SpecsynBlackbody::spec(
    const std::array<double, static_cast<std::size_t>(tracks::FieldIdx::nTrackQty)>& props
) const -> std::vector<double>
{
    const double logL = props.at(static_cast<std::size_t>(tracks::FieldIdx::logL));
    const double logTeff = props.at(static_cast<std::size_t>(tracks::FieldIdx::logTe));

    const double temperature = std::pow(10.0, logTeff);           // K
    const double luminosity = std::pow(10.0, logL) * solarLuminosity; // erg/s

    // Stellar radius and surface area, from L = 4 pi R^2 sigma T^4
    const double temperature4 = temperature * temperature * temperature * temperature;
    const double radius = std::sqrt(luminosity / (4.0 * pi * stefanBoltzmann * temperature4)); // cm
    const double area = 4.0 * pi * radius * radius; // cm^2

    std::vector<double> result(wl_.size());
    for (std::size_t i = 0; i < wl_.size(); ++i)
    {
        const double wlCm = wl_.at(i) * angstromToCm;
        const double x = planckH * speedOfLight / (wlCm * boltzmannK * temperature);
        const double bLambda = (2.0 * planckH * speedOfLight * speedOfLight) /
            ((wlCm * wlCm * wlCm * wlCm * wlCm) * (std::exp(x) - 1.0)); // erg/s/cm^2/cm
        result.at(i) = area * bLambda * angstromToCm; // erg/s/Angstrom
    }
    return result;
}
