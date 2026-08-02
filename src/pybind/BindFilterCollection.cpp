/**
 * @file BindFilterCollection.cpp
 * @author Mark Krumholz
 * @brief Python bindings for phot::FilterCollection and phot::PhotSystem
 * @date 2026-08-02
 */

#include "Bindings.hpp"
#include "../phot/Filter.hpp"
#include "../phot/FilterCollection.hpp"
#include "../phot/PhotCommons.hpp"
#include <cstddef>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // NOLINT(misc-include-cleaner); this is needed for correct Python binding, even if clang-tidy can't recognize it
#include <string>
#include <string_view>
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
vega : str, optional
    Name of the HDF5 file holding the Vega reference spectrum ("wl"
    and "flux" datasets); only read if phot_system is Vega, in which
    case every non-photCount() filter's fluxVega() is set from it.

Throws
------
RuntimeError
    If any entry of filter_names does not match a recognized
    tabulated- or idealized-filter naming convention, or if
    constructing the corresponding filter itself throws; also throws
    if phot_system is Vega and vega cannot be read.)doc";

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
    For a photCount() filter: "photons/s"; for an energy-flux filter:
    "erg/s/Angstrom" (Flambda), "Jy" (Fnu), "STmag" (ST), "ABmag" (AB),
    or "VegaMag" (Vega), matching this collection's phot_system -- in
    the same order as phot()/filterNames().)doc";

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
                const std::string&, const std::string&>(),
                constructorDocstring.data(),
                py::arg("filter_names"), py::arg("phot_system"),
                py::arg("registry") = phot::defaultRegistry,
                py::arg("vega") = phot::defaultVegaSpec)
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
                py::return_value_policy::reference_internal);
}
// NOLINTEND(misc-include-cleaner)
