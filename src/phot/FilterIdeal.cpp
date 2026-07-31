/**
 * @file FilterIdeal.cpp
 * @author Mark Krumholz
 * @brief Implementation of FilterIdeal.hpp
 * @date 2026-07-31
 */

#include "FilterIdeal.hpp"
#include "../elem/ElemData.hpp"
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
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    // Split str on every occurrence of '_', keeping empty tokens (e.g.
    // "a__b" -> {"a", "", "b"}) rather than collapsing them, so a
    // malformed name (an extra or missing underscore) fails to match
    // the expected token count downstream instead of silently merging
    // fields
    auto splitOnUnderscore(const std::string& str) -> std::vector<std::string>
    {
        std::vector<std::string> tokens;
        std::size_t start = 0;
        while (true)
        {
            const std::size_t pos = str.find('_', start);
            if (pos == std::string::npos)
            {
                tokens.push_back(str.substr(start));
                break;
            }
            tokens.push_back(str.substr(start, pos - start));
            start = pos + 1;
        }
        return tokens;
    }

    // Return the index into elem::elemData whose symbol matches sym
    // (a 1- or 2-character string), or elem::elemData.size() if not found
    auto findElement(const std::string& sym) -> std::size_t
    {
        for (std::size_t i = 0; i < elem::elemData.size(); ++i)
        {
            const auto& s = elem::elemData[i].symbol();
            if (s[0] != sym[0]) { continue; }
            if (sym.size() == 1 && s[1] == '\0') { return i; }
            if (sym.size() == 2 && s[1] == sym[1]) { return i; }
        }
        return elem::elemData.size();
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
            (inner.size() >= 2 && std::islower(static_cast<unsigned char>(inner[1])))
            ? 2U : 1U;
        const std::string sym   = inner.substr(0, symLen);
        const std::string roman = inner.substr(symLen);

        const std::size_t elemIdx = findElement(sym);
        if (elemIdx >= elem::elemData.size())
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
            elem::elemData[elemIdx].ionPot()[static_cast<std::size_t>(ionState - 1)];
        if (std::isnan(ip))
        {
            throw std::runtime_error(
                "FilterIdeal: in filter name '" + name +
                "', no CRC ionization potential data available for " +
                sym + " ionization state " + roman);
        }

        // Threshold wavelength: lambda = h*c / IP, converted to Angstrom
        wlMin_ = (utils::h * utils::c) / (ip * utils::eV) / utils::Angstrom;
        wlMax_ = std::numeric_limits<double>::infinity();
        photCount_ = true;
        return;
    }

    const auto tokens = splitOnUnderscore(name);
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
        const interp::Interpolator1D<> interp(wl, spec);
        return interp.integ(x0, x1);
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
