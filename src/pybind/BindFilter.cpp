/**
 * @file BindFilter.cpp
 * @author Mark Krumholz
 * @brief Python bindings for phot::Filter
 * @date 2026-08-02
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 * @details
 * Filter is abstract (wlPivot() and phot() are pure virtual), so no
 * constructor is exposed here -- Python code only ever encounters a
 * Filter through a FilterIdeal or FilterTabulated, either constructed
 * directly or handed back by FilterCollection.getFilter()/filters().
 * Registering Filter as those two classes' Python base (see
 * BindFilterIdeal.cpp/BindFilterTabulated.cpp) is what lets pybind11
 * automatically wrap such a returned Filter reference as the correct
 * concrete Python type, rather than a bare Filter.
 */

#include "Bindings.hpp"
#include "../phot/Filter.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // NOLINT(misc-include-cleaner); this is needed for correct Python binding, even if clang-tidy can't recognize it
#include <string_view>

// Numpy-style docstrings for the Python bindings below
static constexpr std::string_view nameDocstring = R"doc(Get the name of this filter.

Returns
-------
name : str)doc";

static constexpr std::string_view photCountDocstring = R"doc(Get whether this filter returns photon counts.

Returns
-------
phot_count : bool
    True if this filter returns photon counts rather than F_lambda
    values, false otherwise.)doc";

static constexpr std::string_view wlPivotDocstring = R"doc(Get this filter's pivot wavelength.

Returns
-------
wl_pivot : float
    A single characteristic wavelength for this filter, in Angstrom --
    used, e.g., to convert a phot() value into other flux/magnitude
    systems, since phot() itself collapses a filter's response to a
    single number with no wavelength of its own. The exact definition
    depends on the concrete filter type -- see
    FilterIdeal.wlPivot()/FilterTabulated.wlPivot() for specifics.)doc";

static constexpr std::string_view photDocstring = R"doc(Compute the photometric response of this filter to a spectrum.

Parameters
----------
wl : list of float
    The wavelength grid, in Angstrom, on which spec is computed.
spec : list of float
    The spectrum to which to compute the photometric response.

Returns
-------
value : float
    The photometric response, in erg/s/cm^2/Angstrom if photCount() is
    False, or photons/s if photCount() is True. The exact formula
    depends on the concrete filter type -- see
    FilterIdeal.phot()/FilterTabulated.phot() for specifics.)doc";

static constexpr std::string_view fluxVegaDocstring = R"doc(Get this filter's mean Vega flux.

Returns
-------
flux_vega : float
    The filter-mean flux of Vega in this filter, in
    erg/s/cm^2/Angstrom, as populated when this filter was built as
    part of a FilterCollection with phot_system = PhotSystem.Vega (0
    if it was never populated).)doc";

// Disable linting for includes -- the pybind macro magic seems to confuse
// the linter
// NOLINTBEGIN(misc-include-cleaner)
void bindFilter(py::module_& m)
{
    py::class_<phot::Filter, py::smart_holder>(m, "Filter")
        .def("name", &phot::Filter::name,
                nameDocstring.data())
        .def("photCount", &phot::Filter::photCount,
                photCountDocstring.data())
        .def("wlPivot", &phot::Filter::wlPivot,
                wlPivotDocstring.data())
        .def("phot", &phot::Filter::phot,
                photDocstring.data(),
                py::arg("wl"), py::arg("spec"))
        .def("fluxVega", &phot::Filter::fluxVega,
                fluxVegaDocstring.data());
}
// NOLINTEND(misc-include-cleaner)
