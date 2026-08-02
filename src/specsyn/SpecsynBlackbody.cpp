/**
 * @file SpecsynBlackbody.cpp
 * @author Mark Krumholz
 * @brief Implementation of SpecsynBlackbody
 * @date 2026-07-18
 */

#include "SpecsynBlackbody.hpp"
#include "../io/SimControls.hpp"
#include "../tracks/TrackCommons.hpp"
#include "../utils/Constants.hpp"
#include "../utils/MiscUtils.hpp"
#include "Specsyn.hpp"
#include <cmath>
#include <cstddef>
#include <vector>

// Unit conversions between cm and Angstrom
static constexpr double angstromToCm = utils::Angstrom;
static constexpr double cmToAngstrom = 1.0 / utils::Angstrom;

specsyn::SpecsynBlackbody::SpecsynBlackbody(
    double wlMin, double wlMax, std::size_t nWl, const double z,
    const io::SimControls& controls) : Specsyn(controls, z)
{
    // nWl == 0 means neither a grid nor a point count was requested
    // at all; nWl != 0 with wlMin == 0 means only nWl was requested
    // (wlMin/wlMax are still at SimControls::readSpectra's "not
    // supplied" sentinel) -- either way, wlMin/wlMax fall back to
    // this class's own default range (photon energies from 10 Ry
    // down to 0.01 Ry, converted from cm to Angstrom); nWl only falls
    // back to nWlDefault in the former case, since supplying nWl
    // alone is a valid, meaningful request (this class's own default
    // range, but at a caller-chosen resolution) in its own right.
    if (nWl == 0 || wlMin == 0.0)
    {
        wlMin = (planckH * speedOfLight / (10.0 * rydberg)) * cmToAngstrom;
        wlMax = (planckH * speedOfLight / (0.01 * rydberg)) * cmToAngstrom;
        if (nWl == 0) { nWl = nWlDefault; }
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
