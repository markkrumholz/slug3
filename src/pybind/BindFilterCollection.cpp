/**
 * @file BindFilterCollection.cpp
 * @author Mark Krumholz
 * @brief Python bindings for phot::FilterCollection and phot::PhotSystem
 * @date 2026-08-02
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#include "Bindings.hpp"
#include "../phot/Filter.hpp"
#include "../phot/FilterCollection.hpp"
#include "../phot/PhotCommons.hpp"
#include <cstddef>
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // NOLINT(misc-include-cleaner); this is needed for correct Python binding, even if clang-tidy can't recognize it
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// Numpy-style docstrings for the Python bindings below
static constexpr std::string_view photSystemDocstring = R"doc(A photometric system.

Determines the units a computed photometric value is expressed in --
either a physical flux (Flambda, Fnu) or a magnitude on one of several
magnitude scales (ST, AB, Vega).)doc";

static constexpr std::string_view constructorDocstring = R"doc(Construct a FilterCollection from a list of filter names.

Parameters
----------
filter_names : list of str
    Names of the filters to include; each must be either a
    tabulated-filter name (facility.filter or facility.instrument.
    filter, the latter's instrument omitted only for a facility with
    exactly one instrument in the registry -- see FilterTabulated) or
    an idealized-filter name (ideal_energy_X_Y, ideal_phot_X_Y, or
    Q(...) -- see FilterIdeal).
phot_system : PhotSystem
    The photometric system phot() converts energy-flux
    (photCount() == False) filters' results into.
registry : str, optional
    Name of the filter registry file to resolve tabulated filter
    names against.

Throws
------
RuntimeError
    If any entry of filter_names does not match a recognized
    tabulated- or idealized-filter naming convention, or if
    constructing the corresponding filter itself throws.

Details
-------
If phot_system is PhotSystem.Vega, each non-photCount() filter's Vega
zero point is computed lazily, from the global Vega reference
spectrum, the first time it is actually needed -- see
Filter.fluxVega().)doc";

static constexpr std::string_view photDocstring = R"doc(Compute the photometric response of every filter in this collection to a spectrum.

Parameters
----------
wl : list of float
    The wavelength grid, in Angstrom, on which spec is computed.
spec : list of float
    The spectrum to which to compute the photometric response.

Returns
-------
values : list of float
    One value per filter, in the same order as
    filterNames()/filterUnits(): for a photCount() filter, the raw
    photon-count value (PhotSystem conversions apply only to energy
    fluxes); for an energy-flux filter, the value converted to this
    collection's phot_system.)doc";

static constexpr std::string_view filterNamesDocstring = R"doc(Get the names of every filter in this collection.

Returns
-------
names : list of str
    The name of every filter, in the same order as
    phot()/filterUnits().)doc";

static constexpr std::string_view filterUnitsDocstring = R"doc(Get the units of every filter's phot() value.

Returns
-------
units : list of str
    For a photCount() filter: "photon/s"; for an energy-flux filter:
    "erg/(s Angstrom)" (Flambda), "Jy" (Fnu), "mag(ST)" (ST), "mag(AB)"
    (AB), or "mag" (Vega -- astropy has no dedicated Vega-magnitude
    unit), matching this collection's phot_system -- in the same order
    as phot()/filterNames().)doc";

static constexpr std::string_view getFilterByIndexDocstring = R"doc(Get a single filter by index.

Parameters
----------
i : int
    Index of the filter to get, in the same order as
    phot()/filterNames()/filterUnits().

Returns
-------
filter : FilterIdeal or FilterTabulated
    The i'th filter, as whichever concrete type it actually is.

Throws
------
IndexError
    If i is out of range.)doc";

static constexpr std::string_view getFilterByNameDocstring = R"doc(Get a single filter by name.

Parameters
----------
name : str
    Name to search for; must exactly match one entry of filterNames().

Returns
-------
filter : FilterIdeal or FilterTabulated
    The filter whose name() exactly matches name, as whichever
    concrete type it actually is.

Throws
------
RuntimeError
    If no filter's name() matches name.)doc";

static constexpr std::string_view filtersDocstring = R"doc(Get every filter in this collection.

Returns
-------
filters : list of FilterIdeal or FilterTabulated
    Every filter in this collection, each as whichever concrete type
    it actually is, in the same order as
    phot()/filterNames()/filterUnits().)doc";

static constexpr std::string_view addFilterByNameDocstring = R"doc(Parse a filter name and add the resulting filter to this collection.

Parameters
----------
name : str
    Name of the filter to add; see the constructor's own filter_names
    parameter for the recognized naming conventions.
registry : str, optional
    Name of the filter registry file to resolve a tabulated filter
    name against; unused for an idealized filter name.

Throws
------
RuntimeError
    If name does not match a recognized tabulated- or idealized-filter
    naming convention, or if constructing the corresponding filter
    itself throws.

Details
-------
Unlike a filter named in the constructor's own filter_names, a filter
added this way is not given a fluxVega() even if this collection's
phot_system is PhotSystem.Vega -- the constructor sets fluxVega() in a
second pass, after every filter named in filter_names has been built,
which this single-filter entry point has no equivalent of yet.)doc";

static constexpr std::string_view addFilterDirectDocstring = R"doc(Add an already-constructed filter to this collection.

Parameters
----------
filter : FilterIdeal or FilterTabulated
    The filter to add; ownership is transferred to this
    FilterCollection, so filter is no longer usable from Python after
    this call.

Details
-------
Lets a caller build a filter directly, via FilterIdeal's or
FilterTabulated's own constructor, and add it to this collection
rather than by name via the other addFilter() overload. Like that
overload, does not set the newly added filter's fluxVega() on a
PhotSystem.Vega collection -- see its own docstring.)doc";

// Disable linting for includes -- the pybind macro magic seems to confuse
// the linter
// NOLINTBEGIN(misc-include-cleaner)
void bindFilterCollection(py::module_& m)
{
    py::enum_<phot::PhotSystem>(m, "PhotSystem", photSystemDocstring.data())
        .value("Flambda", phot::PhotSystem::Flambda, "Flux per unit wavelength")
        .value("Fnu", phot::PhotSystem::Fnu, "Flux per unit frequency")
        .value("ST", phot::PhotSystem::ST, "The ST magnitude system")
        .value("AB", phot::PhotSystem::AB, "The AB magnitude system")
        .value("Vega", phot::PhotSystem::Vega, "The Vega magnitude system");

    py::class_<phot::FilterCollection, py::smart_holder>(m, "FilterCollection")
        .def(py::init<const std::vector<std::string>&, phot::PhotSystem,
                const std::string&>(),
                constructorDocstring.data(),
                py::arg("filter_names"), py::arg("phot_system"),
                py::arg("registry") = phot::defaultRegistry)
        .def("phot", &phot::FilterCollection::phot,
                photDocstring.data(),
                py::arg("wl"), py::arg("spec"))
        .def("filterNames", &phot::FilterCollection::filterNames,
                filterNamesDocstring.data())
        .def("filterUnits", &phot::FilterCollection::filterUnits,
                filterUnitsDocstring.data())
        .def("getFilter",
                [](const phot::FilterCollection& self, std::size_t i) -> const phot::Filter&
                {
                    return self.getFilter(i);
                },
                getFilterByIndexDocstring.data(),
                py::arg("i"), py::return_value_policy::reference_internal)
        .def("getFilter",
                [](const phot::FilterCollection& self, const std::string& name) -> const phot::Filter&
                {
                    return self.getFilter(name);
                },
                getFilterByNameDocstring.data(),
                py::arg("name"), py::return_value_policy::reference_internal)
        .def("filters",
                [](const phot::FilterCollection& self) -> std::vector<const phot::Filter*>
                {
                    std::vector<const phot::Filter*> result;
                    result.reserve(self.filters().size());
                    for (const auto& filt : self.filters()) { result.push_back(filt.get()); }
                    return result;
                },
                filtersDocstring.data(),
                py::return_value_policy::reference_internal)
        .def("addFilter",
                [](phot::FilterCollection& self, const std::string& name, const std::string& registry)
                {
                    self.addFilter(name, registry);
                },
                addFilterByNameDocstring.data(),
                py::arg("name"), py::arg("registry") = phot::defaultRegistry)
        .def("addFilter",
                [](phot::FilterCollection& self, std::unique_ptr<phot::Filter> filter)
                {
                    self.addFilter(std::move(filter));
                },
                addFilterDirectDocstring.data(),
                py::arg("filter"));
}
// NOLINTEND(misc-include-cleaner)
