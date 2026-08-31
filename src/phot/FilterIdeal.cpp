/**
 * @file FilterIdeal.cpp
 * @author Mark Krumholz
 * @brief Implementation of FilterIdeal.hpp
 * @date 2026-07-31
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#include "FilterIdeal.hpp"
#include "../elem/IonizationData.hpp"
#include "../interpolation/Interpolator1D.hpp"
#include "../utils/Constants.hpp"
#include "../utils/MiscUtils.hpp"
#include "../utils/ParseUtils.hpp"
#include "Filter.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    // Return the index into elem::ionizationData whose symbol matches sym
    // (a 1- or 2-character string), or elem::ionizationData.size() if not found
    auto findElement(const std::string& sym) -> std::size_t
    {
        for (std::size_t i = 0; i < elem::ionizationData.size(); ++i)
        {
            const auto& s = elem::ionizationData[i].symbol(); // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- i bounded by ionizationData.size() loop guard
            if (s[0] != sym[0]) { continue; }
            if (sym.size() == 1 && s[1] == '\0') { return i; }
            if (sym.size() == 2 && s[1] == sym[1]) { return i; }
        }
        return elem::ionizationData.size();
    }
} // namespace

// Parse name against all recognized naming conventions; Filter(name)
// runs first via the member-initializer list, leaving photCount_ at
// its base-class default of false until this body determines and
// overwrites it
phot::FilterIdeal::FilterIdeal(std::string name) : Filter(name) // NOLINT(performance-unnecessary-value-param) -- kept by-value to match Filter's own by-value+move constructor and FilterIdeal's sibling direct constructor; this isn't a hot path (filters are built once, not per-evaluation), so the extra copy is immaterial
{
    // Q(<symbol><roman>) ionization-threshold convention
    if (name.size() >= 4 && name.front() == 'Q' && name[1] == '(' && name.back() == ')')
    {
        const std::string inner = name.substr(2, name.size() - 3);

        // Element symbol: first char always included; second char too
        // if it is lowercase (all two-letter symbols have a lowercase
        // second character, while Roman numeral chars are all uppercase)
        const std::size_t symLen =
            (inner.size() >= 2 && std::islower(static_cast<unsigned char>(inner[1])) != 0)
            ? 2U : 1U;
        const std::string sym   = inner.substr(0, symLen);
        const std::string roman = inner.substr(symLen);

        const std::size_t elemIdx = findElement(sym);
        if (elemIdx >= elem::ionizationData.size())
        {
            throw std::runtime_error(
                "FilterIdeal: in filter name '" + name + "', '" + sym +
                "' does not match any known element symbol");
        }

        const int ionState = utils::romanToInt(roman);
        if (ionState <= 0 || static_cast<std::size_t>(ionState) > elem::maxIP)
        {
            throw std::runtime_error(
                "FilterIdeal: in filter name '" + name + "', '" + roman +
                "' is not a valid Roman numeral in range I to " +
                std::to_string(elem::maxIP));
        }

        const double ip =
            elem::ionizationData[elemIdx].ionPot()[static_cast<std::size_t>(ionState - 1)]; // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- index validated to be in [0, maxIP) by the romanToInt range check above
        if (std::isnan(ip))
        {
            throw std::runtime_error(
                "FilterIdeal: in filter name '" + name +
                "', no CRC ionization potential data available for " +
                sym + " ionization state " + roman);
        }

        // Ionizing photons have energy above IP, i.e. wavelength BELOW
        // the threshold lambda = h*c / IP (converted to Angstrom) --
        // so the passband is [0, threshold], not [threshold, inf)
        wlMin_ = 0.0;
        wlMax_ = (utils::h * utils::c) / (ip * utils::eV) / utils::Angstrom;
        photCount_ = true;
        return;
    }

    const auto tokens = utils::splitOnChar(name, '_');
    const bool wellFormed = tokens.size() == 4 && tokens.at(0) == "ideal" &&
        (tokens.at(1) == "energy" || tokens.at(1) == "phot");
    if (!wellFormed)
    {
        throw std::runtime_error(
            "FilterIdeal: unable to parse filter name '" + name +
            "'; expected ideal_energy_X_Y or ideal_phot_X_Y, with X "
            "and Y wavelengths in Angstrom (Y may also be 'inf')");
    }

    try
    {
        wlMin_ = utils::stod(tokens.at(2));
        wlMax_ = utils::stod(tokens.at(3));
    }
    catch (const std::exception&)
    {
        throw std::runtime_error(
            "FilterIdeal: unable to parse filter name '" + name +
            "'; X and Y must be numeric wavelengths in Angstrom "
            "(Y may also be 'inf')");
    }

    photCount_ = (tokens.at(1) == "phot");

    // phot()'s energy-flux mode integrates in ln(wavelength), which
    // breaks for a non-positive wlMin or an infinite wlMax -- see
    // FilterIdeal.hpp's own constructor comment
    if (!photCount_ && (wlMin_ <= 0.0 || std::isinf(wlMax_)))
    {
        throw std::runtime_error(
            "FilterIdeal: in filter name '" + name + "', an energy-flux "
            "filter (ideal_energy_X_Y) must have a finite, positive "
            "wavelength range; got X = " + std::to_string(wlMin_) +
            ", Y = " + std::to_string(wlMax_));
    }
}

auto phot::FilterIdeal::phot(const std::vector<double>& wl,
    const std::vector<double>& spec) const -> double
{
    // Clamp integration range to the spectrum's wavelength coverage;
    // any portion of [wlMin_, wlMax_] outside the grid contributes 0
    const double x0 = std::max(wlMin_, wl.front());
    const double x1 = std::min(wlMax_, wl.back());
    if (x0 >= x1) { return 0.0; }

    if (!photCount_)
    {
        // Mean flux density over the passband on a log-wavelength
        // basis, matching FilterTabulated::phot()'s own convention:
        // integrate in ln(wavelength) directly (building the
        // interpolant on lnGrid(wl) rather than wl itself, exactly as
        // FilterTabulated::phot() does via its own lnWl_), then
        // divide by the ln-domain width -- the analog of dividing by
        // FilterTabulated's norm_, since this filter's implicit
        // top-hat response R = 1 on [wlMin_, wlMax_] integrates to
        // ln(wlMax_ / wlMin_) with respect to ln(wavelength) exactly.
        // Requires x0 > 0 and x1 finite, guaranteed for an
        // energy-flux filter by both constructors' own validation.
        const interp::Interpolator1D<> interp(lnGrid(wl), spec);
        return interp.integ(std::log(x0), std::log(x1)) / std::log(x1 / x0);
    }

    // Photon-count mode: convert F_lambda (erg/s/Å) to photon flux density
    // (photons/s/Å) by dividing by the photon energy h*c/lambda; wavelength
    // must be in cm to match h*c units (erg·cm), so multiply by Angstrom
    std::vector<double> photSpec(spec.size());
    for (std::size_t i = 0; i < spec.size(); ++i)
    {
        photSpec[i] = spec[i] * wl[i] * utils::Angstrom / (utils::h * utils::c); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- wl and spec are required to have the same size (Filter interface contract)
    }
    const interp::Interpolator1D<> interp(wl, photSpec);
    return interp.integ(x0, x1);
}
