/**
 * @file BindPhotConvert.cpp
 * @author Mark Krumholz
 * @brief Python bindings for phot::PhotConvert
 * @date 2026-08-02
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#include "Bindings.hpp"
#include "../phot/Filter.hpp"
#include "../phot/PhotCommons.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdexcept>
#include <string>
#include <string_view>

namespace
{
    // Map a Python-facing PhotSystem name string to phot::PhotSystem,
    // mirroring the phot.system string values io::SimControls::readFilters()
    // itself recognizes
    auto photSystemFromString(const std::string& name) -> phot::PhotSystem
    {
        if (name == "Flambda") { return phot::PhotSystem::Flambda; }
        if (name == "Fnu") { return phot::PhotSystem::Fnu; }
        if (name == "ST") { return phot::PhotSystem::ST; }
        if (name == "AB") { return phot::PhotSystem::AB; }
        if (name == "Vega") { return phot::PhotSystem::Vega; }
        throw std::runtime_error(
            "PhotConvert: '" + name + "' is not a recognized photometric "
            "system (expected Flambda, Fnu, ST, AB, or Vega)");
    }

    // Dispatch to the correctly-instantiated phot::PhotConvert<From, To>
    // template for a runtime (from, to) pair -- fluxVega is passed to
    // every call for a uniform signature, but is only actually read by
    // the instantiations where From or To is PhotSystem::Vega. from ==
    // to is handled by the caller (bindPhotConvert's lambda, below)
    // before this is ever reached, since PhotConvert has no defined
    // same-to-same instantiation (mirroring
    // FilterCollection.cpp's own convertFlambda(), which likewise
    // never calls PhotConvert<Flambda, Flambda>).
    auto photConvertDispatch(const phot::PhotSystem from, const phot::PhotSystem to,
        const double fluxIn, const double wl, const double fluxVega) -> double
    {
        using phot::PhotSystem;
        switch (from)
        {
            case PhotSystem::Flambda:
                switch (to)
                {
                    case PhotSystem::Fnu:
                        return phot::PhotConvert<PhotSystem::Flambda, PhotSystem::Fnu>(fluxIn, wl);
                    case PhotSystem::ST:
                        return phot::PhotConvert<PhotSystem::Flambda, PhotSystem::ST>(fluxIn, wl);
                    case PhotSystem::AB:
                        return phot::PhotConvert<PhotSystem::Flambda, PhotSystem::AB>(fluxIn, wl);
                    case PhotSystem::Vega:
                        return phot::PhotConvert<PhotSystem::Flambda, PhotSystem::Vega>(fluxIn, wl, fluxVega);
                    case PhotSystem::Flambda: break; // unreachable; from == to handled by caller
                }
                break;
            case PhotSystem::Fnu:
                switch (to)
                {
                    case PhotSystem::Flambda:
                        return phot::PhotConvert<PhotSystem::Fnu, PhotSystem::Flambda>(fluxIn, wl);
                    case PhotSystem::ST:
                        return phot::PhotConvert<PhotSystem::Fnu, PhotSystem::ST>(fluxIn, wl);
                    case PhotSystem::AB:
                        return phot::PhotConvert<PhotSystem::Fnu, PhotSystem::AB>(fluxIn, wl);
                    case PhotSystem::Vega:
                        return phot::PhotConvert<PhotSystem::Fnu, PhotSystem::Vega>(fluxIn, wl, fluxVega);
                    case PhotSystem::Fnu: break; // unreachable; from == to handled by caller
                }
                break;
            case PhotSystem::ST:
                switch (to)
                {
                    case PhotSystem::Flambda:
                        return phot::PhotConvert<PhotSystem::ST, PhotSystem::Flambda>(fluxIn, wl);
                    case PhotSystem::Fnu:
                        return phot::PhotConvert<PhotSystem::ST, PhotSystem::Fnu>(fluxIn, wl);
                    case PhotSystem::AB:
                        return phot::PhotConvert<PhotSystem::ST, PhotSystem::AB>(fluxIn, wl);
                    case PhotSystem::Vega:
                        return phot::PhotConvert<PhotSystem::ST, PhotSystem::Vega>(fluxIn, wl, fluxVega);
                    case PhotSystem::ST: break; // unreachable; from == to handled by caller
                }
                break;
            case PhotSystem::AB:
                switch (to)
                {
                    case PhotSystem::Flambda:
                        return phot::PhotConvert<PhotSystem::AB, PhotSystem::Flambda>(fluxIn, wl);
                    case PhotSystem::Fnu:
                        return phot::PhotConvert<PhotSystem::AB, PhotSystem::Fnu>(fluxIn, wl);
                    case PhotSystem::ST:
                        return phot::PhotConvert<PhotSystem::AB, PhotSystem::ST>(fluxIn, wl);
                    case PhotSystem::Vega:
                        return phot::PhotConvert<PhotSystem::AB, PhotSystem::Vega>(fluxIn, wl, fluxVega);
                    case PhotSystem::AB: break; // unreachable; from == to handled by caller
                }
                break;
            case PhotSystem::Vega:
                switch (to)
                {
                    case PhotSystem::Flambda:
                        return phot::PhotConvert<PhotSystem::Vega, PhotSystem::Flambda>(fluxIn, wl, fluxVega);
                    case PhotSystem::Fnu:
                        return phot::PhotConvert<PhotSystem::Vega, PhotSystem::Fnu>(fluxIn, wl, fluxVega);
                    case PhotSystem::ST:
                        return phot::PhotConvert<PhotSystem::Vega, PhotSystem::ST>(fluxIn, wl, fluxVega);
                    case PhotSystem::AB:
                        return phot::PhotConvert<PhotSystem::Vega, PhotSystem::AB>(fluxIn, wl, fluxVega);
                    case PhotSystem::Vega: break; // unreachable; from == to handled by caller
                }
                break;
        }
        throw std::runtime_error("PhotConvert: unreachable PhotSystem combination");
    }
} // namespace

// Numpy-style docstring for the Python binding below
static constexpr std::string_view photConvertDocstring = R"doc(Convert a flux or magnitude from one photometric system to another.

Parameters
----------
phot_from : str
    The photometric system flux_in is expressed in: one of "Flambda",
    "Fnu", "ST", "AB", or "Vega".
phot_to : str
    The photometric system to convert flux_in to; same recognized
    values as phot_from.
flux_in : float or array_like of float
    The input value(s), in erg/s/cm^2/Angstrom (if phot_from is
    "Flambda"), Jy (if phot_from is "Fnu"), or a magnitude (if
    phot_from is "ST", "AB", or "Vega").
wl : float or array_like of float
    The wavelength(s), in Angstrom, at which flux_in is evaluated;
    unused by conversions that don't depend on wavelength, but
    required for a uniform signature. Broadcast against flux_in, so
    e.g. a whole spectrum can be converted in one call by passing
    matching flux_in and wl arrays.
filter : FilterIdeal or FilterTabulated, optional
    A filter whose fluxVega() gives the Vega zero point to use;
    required if phot_from or phot_to is "Vega" (fluxVega() is computed
    lazily, so this may trigger loading the global Vega reference
    spectrum -- see Filter.fluxVega()), ignored otherwise.

Returns
-------
flux_out : float or numpy.ndarray of float
    flux_in converted to phot_to, in the units/magnitude convention
    phot_to specifies (see flux_in's own docstring). Returns flux_in
    unchanged if phot_from and phot_to are the same system.

Throws
------
RuntimeError
    If phot_from or phot_to is not a recognized photometric system, or
    if phot_from or phot_to is "Vega" and filter is not given.)doc";

// Disable linting for includes -- the pybind macro magic seems to confuse
// the linter
// NOLINTBEGIN(misc-include-cleaner)
void bindPhotConvert(py::module_& m)
{
    m.def("PhotConvert",
            py::vectorize(
                [](const std::string photFrom, const std::string photTo, // NOLINT(performance-unnecessary-value-param); pybind11's vectorize() requires non-vectorized arguments to be taken by value
                   const double fluxIn, const double wl, const py::object filter) -> double // NOLINT(performance-unnecessary-value-param); see above
                {
                    const auto from = photSystemFromString(photFrom);
                    const auto to = photSystemFromString(photTo);
                    if (from == to) { return fluxIn; }

                    double fluxVega = 0.0;
                    if (from == phot::PhotSystem::Vega || to == phot::PhotSystem::Vega)
                    {
                        if (filter.is_none())
                        {
                            throw std::runtime_error(
                                "PhotConvert: filter is required when phot_from "
                                "or phot_to is \"Vega\"");
                        }
                        fluxVega = py::cast<const phot::Filter&>(filter).fluxVega();
                    }

                    return photConvertDispatch(from, to, fluxIn, wl, fluxVega);
                }),
            photConvertDocstring.data(),
            py::arg("phot_from"), py::arg("phot_to"), py::arg("flux_in"),
            py::arg("wl"), py::arg("filter") = py::none());
}
// NOLINTEND(misc-include-cleaner)
