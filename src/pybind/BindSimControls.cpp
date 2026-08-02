/**
 * @file BindSimControls.cpp
 * @author Mark Krumholz
 * @brief Python bindings for io::SimControls
 * @date 2026-07-30
 */

#include "Bindings.hpp"
#include "../io/SimControls.hpp"
#include <cstddef>
#include <memory>
#include <pybind11/pybind11.h>
#include <string>
#include <string_view>
#include <toml.hpp>

static constexpr std::string_view fromPathDocstring = R"doc(Construct a SimControls object by parsing a slug input deck.

Parameters
----------
path : str
    Path to a slug TOML input deck. The deck must contain at least
    ``sim_type`` and one of the output-time specifications
    (``outputs.output_times``, ``outputs.output_time_dist``, or
    the ``outputs.start_time`` / ``outputs.end_time`` /
    ``outputs.ntime`` triplet). Optional integrator settings
    (``integrator.rel_tol``, ``integrator.abs_tol``,
    ``integrator.max_iter``) are read if present.

Throws
------
RuntimeError
    If the file cannot be parsed or any required key is missing.)doc";

static constexpr std::string_view fromTolerancesDocstring = R"doc(Construct a SimControls with default settings and optional integrator tolerances.

This constructor is intended for interactive use when building a
spectral synthesizer directly (rather than running a full simulation):
it sets every parameter to its default value and then applies whichever
integrator tolerances the caller supplies.

Parameters
----------
rel_tol : float, optional
    Relative convergence tolerance for the cubature integrator used
    when computing continuously-sampled stellar-population spectra.
    Default is 1e-2.
abs_tol : float, optional
    Absolute convergence tolerance for the same integrator. Useful when
    the integrand is near zero (e.g. outside a spectral library's
    wavelength coverage), where a pure relative tolerance is ill-defined.
    Default is 0 (rely on rel_tol alone).
max_iter : int, optional
    Maximum number of integrand evaluations before the integrator gives
    up. 0 (the default) means unlimited.)doc";

static constexpr std::string_view intRelTolDocstring = R"doc(The relative tolerance for PDF integration.

Relative convergence tolerance passed to the cubature integrator.)doc";

static constexpr std::string_view intAbsTolDocstring = R"doc(The absolute tolerance for PDF integration.

Absolute convergence tolerance passed to the cubature integrator.)doc";

static constexpr std::string_view intMaxIterDocstring = R"doc(The maximum number of evaluations for PDF integration.

Maximum number of integrand evaluations; 0 means unlimited.)doc";

// Disable linting for includes -- the pybind macro magic seems to confuse
// the linter
// NOLINTBEGIN(misc-include-cleaner)
void bindSimControls(py::module_& m)
{
    py::class_<io::SimControls, py::smart_holder>(m, "SimControls")
        .def(py::init(
                [](const std::string& path) -> std::unique_ptr<io::SimControls>
                {
                    const toml::table inputDeck = toml::parse_file(path);
                    return std::make_unique<io::SimControls>(inputDeck);
                }),
                fromPathDocstring.data(),
                py::arg("path"))
        .def(py::init(
                [](double relTol, double absTol, std::size_t maxIter)
                    -> std::unique_ptr<io::SimControls>
                {
                    auto controls = std::make_unique<io::SimControls>();
                    controls->setIntRelTol(relTol);
                    controls->setIntAbsTol(absTol);
                    controls->setIntMaxIter(maxIter);
                    return controls;
                }),
                fromTolerancesDocstring.data(),
                py::arg("rel_tol") = 1e-2,
                py::arg("abs_tol") = 0.0,
                py::arg("max_iter") = static_cast<std::size_t>(0))
        .def_property("intRelTol",
                &io::SimControls::intRelTol,
                &io::SimControls::setIntRelTol,
                intRelTolDocstring.data())
        .def_property("intAbsTol",
                &io::SimControls::intAbsTol,
                &io::SimControls::setIntAbsTol,
                intAbsTolDocstring.data())
        .def_property("intMaxIter",
                &io::SimControls::intMaxIter,
                &io::SimControls::setIntMaxIter,
                intMaxIterDocstring.data());
}
// NOLINTEND(misc-include-cleaner)
