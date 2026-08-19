/**
 * @file BindFilterIdeal.cpp
 * @author Mark Krumholz
 * @brief Python bindings for phot::FilterIdeal
 * @date 2026-08-02
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#include "Bindings.hpp"
#include "../phot/Filter.hpp"
#include "../phot/FilterIdeal.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // NOLINT(misc-include-cleaner); this is needed for correct Python binding, even if clang-tidy can't recognize it
#include <string_view>

// Numpy-style docstrings for the Python bindings below. Only the
// name-parsing constructor is exposed -- the direct (name, wlMin,
// wlMax, photCount) constructor is a lower-level entry point mainly
// useful to FilterCollection's own name-parsing logic in C++, which
// has no Python-facing counterpart of its own.
static constexpr std::string_view constructorDocstring = R"doc(Construct a FilterIdeal by parsing its name.

Parameters
----------
name : str
    Name of this filter; must match one of the recognized ideal-filter
    naming conventions:

    1. ideal_<kind>_X_Y, where X and Y are wavelengths in Angstrom (Y
       may also be "inf", meaning no upper bound) giving wlMin() and
       wlMax() respectively, and <kind> is either "energy" (an
       energy-flux filter, photCount() is False) or "phot" (a
       photon-counting filter, photCount() is True).
    2. Q(<spec>), where <spec> is an atomic symbol followed by a Roman
       numeral giving the ionization state in astronomical notation
       (e.g. Q(HI) = photons that ionize neutral hydrogen). Ionizing
       photons have wavelength below the ionization threshold
       lambda = h*c/IP, so the filter is set to wlMin() = 0 (no lower
       bound), wlMax() = h*c/IP (the ionization-threshold wavelength),
       and photCount() = True.

Throws
------
RuntimeError
    If name does not match either recognized convention, or if the
    requested ionization potential is not available.)doc";

static constexpr std::string_view wlMinDocstring = R"doc(Get the minimum wavelength of this filter's response.

Returns
-------
wl_min : float
    The minimum wavelength of this filter's response, in Angstrom.)doc";

static constexpr std::string_view wlMaxDocstring = R"doc(Get the maximum wavelength of this filter's response.

Returns
-------
wl_max : float
    The maximum wavelength of this filter's response, in Angstrom.)doc";

static constexpr std::string_view wlPivotDocstring = R"doc(Get this filter's pivot wavelength.

Returns
-------
wl_pivot : float
    The midpoint of [wlMin(), wlMax()]; wlMax() if wlMin() is 0 (as
    for a Q(*) ionization-threshold filter, whose passband has no
    finite lower end), or wlMin() if wlMax() is infinite (as for an
    ideal_phot_X_inf filter, whose passband has no finite upper
    end).)doc";

static constexpr std::string_view photDocstring = R"doc(Compute the photometric response of this filter to a spectrum.

Parameters
----------
wl : list of float
    The wavelength grid, in Angstrom.
spec : list of float
    The spectrum, in erg/s/Angstrom.

Returns
-------
value : float
    If photCount() is False: the mean value of spec over the filter's
    passband [wlMin(), wlMax()] on a log-wavelength basis, in
    erg/s/Angstrom, over the intersection of [wlMin(), wlMax()] with
    the supplied wavelength grid. If photCount() is True: the total
    photon-count rate, in photons/s, over the same range.)doc";

// Disable linting for includes -- the pybind macro magic seems to confuse
// the linter
// NOLINTBEGIN(misc-include-cleaner)
void bindFilterIdeal(py::module_& m)
{
    py::class_<phot::FilterIdeal, phot::Filter, py::smart_holder>(m, "FilterIdeal")
        .def(py::init<std::string>(),
                constructorDocstring.data(),
                py::arg("name"))
        .def("wlMin", &phot::FilterIdeal::wlMin,
                wlMinDocstring.data())
        .def("wlMax", &phot::FilterIdeal::wlMax,
                wlMaxDocstring.data())
        .def("wlPivot", &phot::FilterIdeal::wlPivot,
                wlPivotDocstring.data())
        .def("phot", &phot::FilterIdeal::phot,
                photDocstring.data(),
                py::arg("wl"), py::arg("spec"));
}
// NOLINTEND(misc-include-cleaner)
