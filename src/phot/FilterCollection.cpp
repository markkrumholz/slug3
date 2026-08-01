/**
 * @file FilterCollection.cpp
 * @author Mark Krumholz
 * @brief Implementation of FilterCollection.hpp
 * @date 2026-07-31
 */

#include "FilterCollection.hpp"
#include "../utils/Constants.hpp"
#include "../utils/MiscUtils.hpp"
#include "Filter.hpp"
#include "FilterIdeal.hpp"
#include "FilterTabulated.hpp"
#include "PhotCommons.hpp"
#include "hdf5.h" // NOLINT(misc-include-cleaner)
#include <cstddef>
#include <exception>
#include <memory>
#include <numbers>
#include <stdexcept>
#include <string>
#include <toml.hpp>
#include <utility>
#include <vector>

namespace
{
    // Split str on every occurrence of delim, keeping empty tokens
    // (e.g. "a..b" -> {"a", "", "b"}) rather than collapsing them, so
    // a malformed name fails to match the expected token count
    // downstream instead of silently merging fields
    auto splitOnChar(const std::string& str, const char delim) -> std::vector<std::string>
    {
        std::vector<std::string> tokens;
        std::size_t start = 0;
        while (true)
        {
            const std::size_t pos = str.find(delim, start);
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
        const auto registryPath = utils::getFilePath(registryName);
        if (registryPath.empty())
        {
            throw std::runtime_error(
                "FilterCollection: registry " + registryName + " not found");
        }
        toml::table registry;
        try
        {
            registry = toml::parse_file(registryPath.string());
        }
        catch (const std::exception&)
        {
            throw std::runtime_error(
                "FilterCollection: registry " + registryPath.string() +
                " is not a valid toml file");
        }

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

    // Disable linting for the next two functions -- including hdf5.h
    // wholesale (rather than individual headers) is the paradigm
    // HDF5 itself wants, which confuses misc-include-cleaner -- see
    // FilterTabulated.cpp's own identical suppression for its
    // equivalent readDataset1D
    // NOLINTBEGIN(misc-include-cleaner)

    // Read a 1D double dataset from an already-open HDF5 file
    auto readDataset1D(const hid_t file, const std::string& name) -> std::vector<double>
    {
        const hid_t dset = H5Dopen2(file, name.c_str(), H5P_DEFAULT);
        if (dset < 0)
        {
            throw std::runtime_error(
                "FilterCollection: unable to open dataset " + name);
        }
        const hid_t space = H5Dget_space(dset);
        hsize_t dims = 0;
        H5Sget_simple_extent_dims(space, &dims, nullptr);
        std::vector<double> data(dims);
        H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
        H5Sclose(space);
        H5Dclose(dset);
        return data;
    }

    // Load the Vega reference spectrum's wl/flux datasets from
    // vegaName -- see data/tools/fetch_vega.py for the file's schema
    auto loadVegaSpectrum(const std::string& vegaName) -> std::pair<std::vector<double>, std::vector<double>>
    {
        const auto vegaPath = utils::getFilePath(vegaName);
        if (vegaPath.empty())
        {
            throw std::runtime_error(
                "FilterCollection: Vega reference spectrum " + vegaName + " not found");
        }
        const hid_t file = H5Fopen(vegaPath.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        if (file < 0)
        {
            throw std::runtime_error(
                "FilterCollection: unable to open HDF5 file " + vegaPath.string());
        }
        std::vector<double> wlVega = readDataset1D(file, "wl");
        std::vector<double> fluxVega = readDataset1D(file, "flux");
        H5Fclose(file);
        return { std::move(wlVega), std::move(fluxVega) };
    }
    // NOLINTEND(misc-include-cleaner)
} // namespace

phot::FilterCollection::FilterCollection(
    const std::vector<std::string>& filterNames,
    const PhotSystem photSystem,
    const std::string& registryName,
    const std::string& vegaName)
: photSystem_(photSystem)
{
    for (const auto& name : filterNames)
    {
        if (isIdealFilterName(name))
        {
            filters_.push_back(std::make_unique<FilterIdeal>(name));
            continue;
        }

        const auto tokens = splitOnChar(name, '.');
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

    if (photSystem_ == PhotSystem::Vega)
    {
        const auto [wlVega, fluxVega] = loadVegaSpectrum(vegaName);
        for (const auto& filt : filters_)
        {
            // A photon-count filter's phot() value has no Flambda
            // meaning, so the Vega spectrum is meaningless for it;
            // leave its fluxVega() at Filter's own default of 0
            if (filt->photCount()) { continue; }
            filt->setFluxVega(wlVega, fluxVega);
        }
    }
}

auto phot::FilterCollection::phot(const std::vector<double>& wl,
    const std::vector<double>& spec) const -> std::vector<double>
{
    std::vector<double> result(filters_.size());
    for (std::size_t i = 0; i < filters_.size(); ++i)
    {
        const auto& filt = filters_.at(i);
        const double value = filt->phot(wl, spec);
        result.at(i) = filt->photCount()
            ? value
            : convertFlambda(value, filt->wlPivot(), filt->fluxVega(), photSystem_);
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
            result.emplace_back("photons/s");
            continue;
        }
        switch (photSystem_)
        {
            case PhotSystem::Flambda: result.emplace_back("erg/s/Angstrom"); break;
            case PhotSystem::Fnu:     result.emplace_back("erg/s/Hz");       break;
            case PhotSystem::ST:      result.emplace_back("STmag");         break;
            case PhotSystem::AB:      result.emplace_back("ABmag");         break;
            case PhotSystem::Vega:    result.emplace_back("VegaMag");       break;
        }
    }
    return result;
}
