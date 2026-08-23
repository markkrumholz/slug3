/**
 * @file BindSimGalaxy.cpp
 * @author Mark Krumholz
 * @brief Python bindings for core::SimGalaxy
 * @date 2026-08-10
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#include "Bindings.hpp"
#include "../core/SimGalaxy.hpp"
#include "../io/OutputManager.hpp"
#include "../io/SimControls.hpp"
#include <memory>
#include <pybind11/pybind11.h>
#include <string_view>
#include <utility>

// Numpy-style docstrings for the Python bindings below
static constexpr std::string_view constructorDocstring = R"doc(Construct a SimGalaxy to drive a galaxy-type simulation end to end.

Parameters
----------
sim_controls : SimControls
    Simulation controls (physics settings and control-flow settings
    together); sim_controls must outlive this SimGalaxy.
output_manager : OutputManagerH5 or OutputManagerAscii
    Output manager to which simulation results should be written;
    ownership is transferred to this SimGalaxy, so output_manager is
    no longer usable from Python after this call.)doc";

static constexpr std::string_view runDocstring = R"doc(Run the simulation.

Details
-------
Runs sim_controls.nTrial() independent trials -- each building a
Galaxy and advancing it through every one of sim_controls's own output
times, writing its properties, spectrum, and photometry (and, for each
currently-alive cluster, that cluster's own) to output_manager as it
goes (see SimGalaxy's own C++ documentation for the full per-trial
sequence) -- exactly as the slug command-line executable does for a
galaxy-type input deck. Blocks until every trial completes, which may
take anywhere from seconds to hours depending on n_trial and the
simulation's own physics settings; releases the GIL for the duration,
so this does not block other Python threads, and so the OpenMP
parallelization this loop uses internally (if slug was built with
OpenMP support) can actually make use of multiple cores.)doc";

static constexpr std::string_view trialsCompletedDocstring = R"doc(Return the number of trials completed so far.

Returns
-------
int
    Number of trials this simulation has finished running, out of
    sim_controls.nTrial() total.

Details
-------
Safe to call from another thread while run() is still executing (e.g.
to drive a progress bar), since run() releases the GIL for its own
entire duration and this itself is a fast, non-blocking read.)doc";

// Disable linting for includes -- the pybind macro magic seems to confuse
// the linter
// NOLINTBEGIN(misc-include-cleaner)
void bindSimGalaxy(py::module_& m)
{
    py::class_<core::SimGalaxy, py::smart_holder>(m, "SimGalaxy")
        .def(py::init(
                [](const io::SimControls& simControls, py::object outputManager)
                    -> std::unique_ptr<core::SimGalaxy>
                {
                    auto managerPtr = py::cast<std::unique_ptr<io::OutputManager>>(
                        std::move(outputManager));
                    return std::make_unique<core::SimGalaxy>(
                        simControls, std::move(managerPtr));
                }),
                constructorDocstring.data(),
                py::arg("sim_controls"), py::arg("output_manager"),
                // Keep sim_controls (argument index 2: 1 = self) alive
                // at least as long as this SimGalaxy -- see
                // BindSimCluster.cpp's own identical comment.
                py::keep_alive<1, 2>())
        .def("run", &core::SimGalaxy::run,
                runDocstring.data(),
                py::call_guard<py::gil_scoped_release>())
        .def("trialsCompleted", &core::SimGalaxy::trialsCompleted,
                trialsCompletedDocstring.data());
}
// NOLINTEND(misc-include-cleaner)
