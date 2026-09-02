/**
 * @file BindSimCluster.cpp
 * @author Mark Krumholz
 * @brief Python bindings for core::SimCluster
 * @date 2026-08-09
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#include "Bindings.hpp"
#include "../core/SimCluster.hpp"
#include "../io/OutputManager.hpp"
#include "../io/SimControls.hpp"
#include <memory>
#include <pybind11/pybind11.h>
#include <string_view>
#include <utility>

// Numpy-style docstrings for the Python bindings below
static constexpr std::string_view constructorDocstring = R"doc(Construct a SimCluster to drive a cluster-type simulation end to end.

Parameters
----------
sim_controls : SimControls
    Simulation controls (physics settings and control-flow settings
    together); sim_controls must outlive this SimCluster.
output_manager : OutputManagerH5 or OutputManagerAscii
    Output manager to which simulation results should be written;
    ownership is transferred to this SimCluster, so output_manager is
    no longer usable from Python after this call.
restart : bool, default False
    Whether this run is resuming a previous, interrupted run from its
    most recent checkpoint; see run()'s own docstring. output_manager
    must itself have already been constructed with its own equivalent
    restart flag set (see OutputManagerH5's own constructor); nothing
    here enforces that the two agree.)doc";

static constexpr std::string_view runDocstring = R"doc(Run the simulation.

Returns
-------
int
    0 if every trial completed normally; 143 (128 + SIGTERM) if a
    SIGTERM was caught and this stopped early instead, with fewer than
    sim_controls.nTrial() trials completed -- see trialsCompleted() to
    find out how many, and this SimCluster's own C++ run() comment for
    the full detail on when and how that can happen.

Details
-------
Runs sim_controls.nTrial() independent trials -- each drawing a
cluster from sim_controls's own cmf(), advancing it through every one
of sim_controls's own output times, and writing its properties,
spectrum, and photometry to output_manager as it goes (see
SimCluster's own C++ documentation for the full per-trial sequence) --
exactly as the slug command-line executable does for a cluster-type
input deck. Blocks until every trial completes, which may take
anywhere from seconds to hours depending on n_trial and the
simulation's own physics settings; releases the GIL for the duration,
so this does not block other Python threads, and so the OpenMP
parallelization this loop uses internally (if slug was built with
OpenMP support) can actually make use of multiple cores. If this
SimCluster was constructed with restart = True, starts from the trial
number output_manager reports already completed (its own
restartTrialsDone()) rather than from trial 0.)doc";

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
void bindSimCluster(py::module_& m)
{
    py::class_<core::SimCluster, py::smart_holder>(m, "SimCluster")
        .def(py::init(
                [](const io::SimControls& simControls, py::object outputManager,
                    const bool restart)
                    -> std::unique_ptr<core::SimCluster>
                {
                    auto managerPtr = py::cast<std::unique_ptr<io::OutputManager>>(
                        std::move(outputManager));
                    return std::make_unique<core::SimCluster>(
                        simControls, std::move(managerPtr), restart);
                }),
                constructorDocstring.data(),
                py::arg("sim_controls"), py::arg("output_manager"),
                py::arg("restart") = false,
                // Keep sim_controls (argument index 2: 1 = self) alive
                // at least as long as this SimCluster, since it stores
                // only a live reference to it -- see
                // BindOutputManagerH5.cpp's own identical comment.
                // output_manager needs no such protection: ownership of
                // it transfers into this SimCluster outright, via
                // py::cast<std::unique_ptr<...>> above, exactly as
                // BindSimControls.cpp's setSpecsyn()/setFilters()/
                // setTracks() already do for their own unique_ptr
                // arguments.
                py::keep_alive<1, 2>())
        .def("run", &core::SimCluster::run,
                runDocstring.data(),
                py::call_guard<py::gil_scoped_release>())
        .def("trialsCompleted", &core::SimCluster::trialsCompleted,
                trialsCompletedDocstring.data());
}
// NOLINTEND(misc-include-cleaner)
