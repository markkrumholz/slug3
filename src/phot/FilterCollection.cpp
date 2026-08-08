/**
 * @file FilterCollection.cpp
 * @author Mark Krumholz
 * @brief Implementation of FilterCollection.hpp
 * @date 2026-07-31
 */

#include "FilterCollection.hpp"
#include "../utils/Constants.hpp"
#include "../utils/ParseUtils.hpp"
#include "../utils/TOMLUtils.hpp"
#include "Filter.hpp"
#include "FilterIdeal.hpp"
#include "FilterTabulated.hpp"
#include "PhotCommons.hpp"
#include <cstddef>
#include <memory>
#include <numbers>
#include <stdexcept>
#include <string>
#include <toml.hpp>
#include <vector>

namespace
{
    // True if name matches one of FilterIdeal's own naming
    // conventions (ideal_energy_X_Y, ideal_phot_X_Y, or Q(...));
    // FilterIdeal's own constructor does the actual detailed parsing
    // and validation, this just decides which Filter subclass to build
    auto isIdealFilterName(const std::string& name) -> bool
    {
        if (name.starts_with("ideal_")) { return true; }
        return name.size() >= 4 && name.front() == 'Q' && name[1] == '(' && name.back() == ')';
    }

    // Resolve the single instrument belonging to facility, for the
    // instrument-omitted "facility.filter" tabulated-filter name form;
    // throws unless the registry lists exactly one instrument for
    // facility
    auto resolveSingleInstrument(const std::string& facility, const std::string& registryName) -> std::string
    {
        const auto [registry, registryPath] =
            utils::parseTOMLFile(registryName, "FilterCollection");

        const auto* instruments = registry[facility]["instruments"].as_array(); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- toml::table/node_view::operator[] is a keyed lookup, not a bounds-checkable container index; a missing key just yields a null node_view, handled below
        if (instruments == nullptr || instruments->empty())
        {
            throw std::runtime_error(
                "FilterCollection: facility '" + facility +
                "' has no instruments in registry " + registryPath.string());
        }
        if (instruments->size() != 1)
        {
            throw std::runtime_error(
                "FilterCollection: facility '" + facility +
                "' has more than one instrument; specify facility.instrument.filter explicitly");
        }

        std::string result;
        instruments->for_each([&result](auto&& el) -> void {
            if constexpr (toml::is_string<decltype(el)>) { result = std::string(el); }
        });
        if (result.empty())
        {
            throw std::runtime_error(
                "FilterCollection: facility '" + facility +
                "' has a non-string entry in its 'instruments' array in registry " +
                registryPath.string());
        }
        return result;
    }

    // Convert an energy-flux value (in Flambda) to the requested
    // PhotSystem at the given pivot wavelength (and, for Vega, this
    // filter's own fluxVega -- see Filter::fluxVega()); PhotConvert is
    // a compile-time-dispatched template, so this switch maps the
    // runtime PhotSystem to the right instantiation
    auto convertFlambda(const double value, const double wlPivot,
        const double fluxVega, const phot::PhotSystem to) -> double
    {
        switch (to)
        {
            case phot::PhotSystem::Flambda:
                return value;
            case phot::PhotSystem::Fnu:
                return phot::PhotConvert<phot::PhotSystem::Flambda, phot::PhotSystem::Fnu>(value, wlPivot);
            case phot::PhotSystem::ST:
            case phot::PhotSystem::AB:
            case phot::PhotSystem::Vega:
            {
                // ST, AB, and Vega magnitudes are defined in terms of
                // a flux, not a luminosity -- but Filter::phot()
                // returns a luminosity-like quantity (no distance
                // baked in), so convert it to the flux that would be
                // observed at the standard distance of 10 pc before
                // handing it to PhotConvert. fluxVega itself needs no
                // such scaling: it already comes from evaluating
                // phot() on the real (already-at-Earth) Vega
                // reference spectrum, not a simulated luminosity.
                constexpr double pi = std::numbers::pi_v<double>;
                constexpr double tenPc = 10.0 * utils::pc;
                const double flux = value / (4.0 * pi * tenPc * tenPc);
                if (to == phot::PhotSystem::ST)
                {
                    return phot::PhotConvert<phot::PhotSystem::Flambda, phot::PhotSystem::ST>(flux, wlPivot);
                }
                if (to == phot::PhotSystem::AB)
                {
                    return phot::PhotConvert<phot::PhotSystem::Flambda, phot::PhotSystem::AB>(flux, wlPivot);
                }
                return phot::PhotConvert<phot::PhotSystem::Flambda, phot::PhotSystem::Vega>(flux, wlPivot, fluxVega);
            }
        }
        throw std::runtime_error("FilterCollection: unrecognized PhotSystem value");
    }
} // namespace

phot::FilterCollection::FilterCollection(
    const std::vector<std::string>& filterNames,
    const PhotSystem photSystem,
    const std::string& registryName)
: photSystem_(photSystem)
{
    for (const auto& name : filterNames) { addFilter(name, registryName); }
}

void phot::FilterCollection::addFilter(const std::string& name, const std::string& registryName)
{
    if (isIdealFilterName(name))
    {
        filters_.push_back(std::make_unique<FilterIdeal>(name));
        return;
    }

    const auto tokens = utils::splitOnChar(name, '.');
    std::string facility;
    std::string instrument;
    std::string filter;
    if (tokens.size() == 3)
    {
        facility = tokens.at(0);
        instrument = tokens.at(1);
        filter = tokens.at(2);
    }
    else if (tokens.size() == 2)
    {
        facility = tokens.at(0);
        filter = tokens.at(1);
        instrument = resolveSingleInstrument(facility, registryName);
    }
    else
    {
        throw std::runtime_error(
            "FilterCollection: unable to parse filter name '" + name +
            "'; expected a tabulated filter name (facility.filter or "
            "facility.instrument.filter) or an idealized filter name "
            "(ideal_energy_X_Y, ideal_phot_X_Y, or Q(...))");
    }

    filters_.push_back(std::make_unique<FilterTabulated>(facility, instrument, filter, registryName));
}

auto phot::FilterCollection::phot(const std::vector<double>& wl,
    const std::vector<double>& spec) const -> std::vector<double>
{
    std::vector<double> result(filters_.size());
    for (std::size_t i = 0; i < filters_.size(); ++i)
    {
        const auto& filt = filters_.at(i);
        const double value = filt->phot(wl, spec);
        // fluxVega() lazily loads the (potentially large) global Vega
        // reference spectrum on its first call (see Filter::fluxVega()),
        // so only pay that cost when photSystem_ actually needs it;
        // convertFlambda() ignores its fluxVega argument for every
        // other PhotSystem anyway
        const double fluxVega = (photSystem_ == PhotSystem::Vega) ? filt->fluxVega() : 0.0;
        result.at(i) = filt->photCount()
            ? value
            : convertFlambda(value, filt->wlPivot(), fluxVega, photSystem_);
    }
    return result;
}

auto phot::FilterCollection::filterNames() const -> std::vector<std::string>
{
    std::vector<std::string> result;
    result.reserve(filters_.size());
    for (const auto& filt : filters_) { result.push_back(filt->name()); }
    return result;
}

auto phot::FilterCollection::filterUnits() const -> std::vector<std::string>
{
    std::vector<std::string> result;
    result.reserve(filters_.size());
    for (const auto& filt : filters_)
    {
        if (filt->photCount())
        {
            result.emplace_back("photon/s");
            continue;
        }
        switch (photSystem_)
        {
            case PhotSystem::Flambda: result.emplace_back("erg/(s Angstrom)"); break;
            case PhotSystem::Fnu:     result.emplace_back("Jy");             break;
            case PhotSystem::ST:      result.emplace_back("mag(ST)");       break;
            case PhotSystem::AB:      result.emplace_back("mag(AB)");       break;
            // astropy has no dedicated Vega-magnitude unit, so this
            // falls back to the generic "mag" -- callers that care
            // about the distinction need photSystem_/PhotSystem::Vega
            // itself, not this string
            case PhotSystem::Vega:    result.emplace_back("mag");           break;
        }
    }
    return result;
}

auto phot::FilterCollection::getFilter(const std::size_t i) const -> const Filter&
{
    if (i >= filters_.size())
    {
        throw std::out_of_range(
            "FilterCollection::getFilter: index " + std::to_string(i) +
            " is out of range for " + std::to_string(filters_.size()) + " filters");
    }
    return *filters_[i]; // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- i is checked against filters_.size() just above
}

auto phot::FilterCollection::getFilter(const std::string& name) const -> const Filter&
{
    for (const auto& filt : filters_)
    {
        if (filt->name() == name) { return *filt; }
    }
    throw std::runtime_error("FilterCollection::getFilter: no filter named \"" + name + "\"");
}
