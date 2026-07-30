/**
 * @file FilterIdeal.cpp
 * @author Mark Krumholz
 * @brief Implementation of FilterIdeal.hpp
 * @date 2026-07-31
 */

#include "FilterIdeal.hpp"
#include "../utils/ParseUtils.hpp"
#include "Filter.hpp"
#include <cstddef>
#include <exception>
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
} // namespace

// Parse name against the ideal_energy_X_Y / ideal_phot_X_Y naming
// conventions (see this class's own header comment); Filter(name)
// runs first, via the member-initializer list, leaving photCount_ at
// its base-class default of false until this body determines and
// overwrites it
phot::FilterIdeal::FilterIdeal(std::string name) : Filter(name) // NOLINT(performance-unnecessary-value-param) -- kept by-value to match Filter's own by-value+move constructor and FilterIdeal's sibling direct constructor; this isn't a hot path (filters are built once, not per-evaluation), so the extra copy is immaterial
{
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
