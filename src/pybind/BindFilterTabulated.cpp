/**
 * @file BindFilterTabulated.cpp
 * @author Mark Krumholz
 * @brief Python bindings for phot::FilterTabulated
 * @date 2026-08-02
 */

#include "Bindings.hpp"
#include "../phot/Filter.hpp"
#include "../phot/FilterTabulated.hpp"
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // NOLINT(misc-include-cleaner); this is needed for correct Python binding, even if clang-tidy can't recognize it
#include <string>
#include <string_view>
#include <vector>

// Numpy-style docstrings for the Python bindings below. Both
// constructors are exposed via factory lambdas returning
// std::unique_ptr, rather than py::init<Args...>(), since
// FilterTabulated disallows copy and move (it holds an
// Interpolator1DScalar member, itself neither copyable nor movable --
// see FilterTabulated's own comment).
static constexpr std::string_view directConstructorDocstring = R"doc(Construct a FilterTabulated directly from a tabulated response.

Parameters
----------
name : str
    Name of this filter, for output purposes.
wl : list of float
    Wavelengths at which the response is tabulated, in Angstrom.
response : list of float
    Filter response at each wavelength in wl.
wl_pivot : float
    This filter's pivot wavelength, in Angstrom.

Details
-------
A FilterTabulated always returns an energy flux (photCount() is
always False).)doc";

static constexpr std::string_view registryConstructorDocstring = R"doc(Construct a FilterTabulated from an entry in a filter registry.

Parameters
----------
facility : str
    Facility that operates the instrument the filter belongs to
    (e.g. "HST").
instrument : str
    Instrument the filter belongs to (e.g. "ACS_WFC").
filter : str
    Name of the filter (e.g. "F435W").
registry : str, optional
    Name of the filter registry file to resolve facility/instrument/
    filter against.

Details
-------
name() is set to "facility.instrument.filter".

Throws
------
RuntimeError
    If the facility/instrument/filter is not found in the registry.)doc";

static constexpr std::string_view normDocstring = R"doc(Get the integral of the filter response with respect to ln(wavelength).

Returns
-------
norm : float)doc";

static constexpr std::string_view wlPivotDocstring = R"doc(Get this filter's pivot wavelength.

Returns
-------
wl_pivot : float
    The pivot wavelength, in Angstrom -- either supplied directly (the
    (name, wl, response, wl_pivot) constructor) or read from the
    registry entry's wl_ref field (the registry constructor).)doc";

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
    The photometric response, in erg/s/cm^2/Angstrom.)doc";

static constexpr std::string_view wlDocstring = R"doc(Get the wavelengths at which the response is tabulated.

Returns
-------
wl : list of float
    The wavelength grid, in Angstrom.)doc";

static constexpr std::string_view responseDataDocstring = R"doc(Get the tabulated filter response data.

Returns
-------
response_data : list of float
    The filter response at each wavelength in wl().)doc";

static constexpr std::string_view responseDocstring = R"doc(Get the interpolator built from the tabulated response.

Returns
-------
response : Interpolator1DScalar
    The interpolator built from ln(wl()) and responseData(), as used
    internally by phot().)doc";

// Disable linting for includes -- the pybind macro magic seems to confuse
// the linter
// NOLINTBEGIN(misc-include-cleaner)
void bindFilterTabulated(py::module_& m)
{
    py::class_<phot::FilterTabulated, phot::Filter, py::smart_holder>(m, "FilterTabulated")
        .def(py::init(
                [](std::string name, const std::vector<double>& wl,
                   const std::vector<double>& response, double wlPivot)
                    -> std::unique_ptr<phot::FilterTabulated>
                {
                    return std::make_unique<phot::FilterTabulated>(
                        std::move(name), wl, response, wlPivot);
                }),
                directConstructorDocstring.data(),
                py::arg("name"), py::arg("wl"), py::arg("response"),
                py::arg("wl_pivot"))
        .def(py::init(
                [](const std::string& facility, const std::string& instrument,
                   const std::string& filter, const std::string& registry)
                    -> std::unique_ptr<phot::FilterTabulated>
                {
                    return std::make_unique<phot::FilterTabulated>(
                        facility, instrument, filter, registry);
                }),
                registryConstructorDocstring.data(),
                py::arg("facility"), py::arg("instrument"), py::arg("filter"),
                py::arg("registry") = phot::defaultRegistry)
        .def("norm", &phot::FilterTabulated::norm,
                normDocstring.data())
        .def("wlPivot", &phot::FilterTabulated::wlPivot,
                wlPivotDocstring.data())
        .def("phot", &phot::FilterTabulated::phot,
                photDocstring.data(),
                py::arg("wl"), py::arg("spec"))
        .def("wl", &phot::FilterTabulated::wl,
                wlDocstring.data())
        .def("responseData", &phot::FilterTabulated::responseData,
                responseDataDocstring.data())
        .def("response", &phot::FilterTabulated::response,
                responseDocstring.data(),
                py::return_value_policy::reference_internal);
}
// NOLINTEND(misc-include-cleaner)
