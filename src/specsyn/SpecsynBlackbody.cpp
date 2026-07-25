/**
 * @file SpecsynBlackbody.cpp
 * @author Mark Krumholz
 * @brief Implementation of SpecsynBlackbody
 * @date 2026-07-18
 */

#include "SpecsynBlackbody.hpp"
#include "../tracks/TrackCommons.hpp"
#include "../utils/MiscUtils.hpp"
#include "Specsyn.hpp"
#include <cmath>
#include <cstddef>
#include <vector>

// Unit conversions between cm and Angstrom (1 Angstrom = 1e-8 cm)
static constexpr double angstromToCm = 1e-8;
static constexpr double cmToAngstrom = 1e8;

specsyn::SpecsynBlackbody::SpecsynBlackbody(
    double wlMin, double wlMax, std::size_t nWl, const double z) : Specsyn(z)
{
    if (nWl == 0)
    {
        // Wavelength range corresponding to photon energies from 10 Ry
        // down to 0.01 Ry, converted from cm to Angstrom
        wlMin = (planckH * speedOfLight / (10.0 * rydberg)) * cmToAngstrom;
        wlMax = (planckH * speedOfLight / (0.01 * rydberg)) * cmToAngstrom;
        nWl = nWlDefault;
    }

    wl_ = utils::logspace(wlMin, wlMax, nWl);
}

auto specsyn::SpecsynBlackbody::spec(const StarData& props, double /*feh*/) const -> std::vector<double>
{
    const double logTeff = props.at(static_cast<std::size_t>(tracks::FieldIdx::logTe));
    const double temperature = std::pow(10.0, logTeff); // K

    const double area = getSAandLogg(props).first; // cm^2

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
