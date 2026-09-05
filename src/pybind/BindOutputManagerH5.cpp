/**
 * @file BindOutputManagerH5.cpp
 * @author Mark Krumholz
 * @brief Python bindings for io::OutputManagerH5
 * @date 2026-08-09
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#include "Bindings.hpp"
#include "../io/OutputManagerH5.hpp"
#include "../io/SimControls.hpp"
#include <pybind11/pybind11.h>
#include <string_view>

// Numpy-style docstring for the Python binding below
static constexpr std::string_view constructorDocstring = R"doc(Open an HDF5 output file and write its header.

Parameters
----------
sim_controls : SimControls
    Simulation controls (physics settings and control-flow settings
    together); sim_controls must outlive this OutputManagerH5, and
    every value it reads from sim_controls (e.g. outDir()/modelName(),
    read once here, at construction) reflects sim_controls's state at
    this moment, not any later change. sim_controls.inputDeckStr() is
    written verbatim into the output file's own input_deck group, so a
    file this OutputManagerH5 produces always records exactly the deck
    that actually built sim_controls (empty if sim_controls was itself
    built with no input deck at all).
restart : bool, default False
    Whether this run is resuming a previous, interrupted run from its
    most recent checkpoint, rather than starting a new one from
    scratch. If True, scans sim_controls.outDir() for the most recent
    checkpoint left behind by a previous run of the same model_name/
    out_dir and resumes numbering checkpoints from just after it,
    instead of starting over at checkpoint 0 -- see the C++
    OutputManagerH5::restartSetup()'s own documentation for the full
    detail. Only meaningful with HDF5 output and a non-zero
    output.checkpoint_interval in sim_controls.

Throws
------
RuntimeError
    If the output file (sim_controls.outDir()/sim_controls.modelName()
    + ".h5") already exists, or if restart is True and
    sim_controls.outputMode() is ascii.)doc";

// Disable linting for includes -- the pybind macro magic seems to confuse
// the linter
// NOLINTBEGIN(misc-include-cleaner)
void bindOutputManagerH5(py::module_& m)
{
    py::class_<io::OutputManagerH5, io::OutputManager, py::smart_holder>(m, "OutputManagerH5")
        .def(py::init<const io::SimControls&, bool>(),
                constructorDocstring.data(),
                py::arg("sim_controls"), py::arg("restart") = false,
                // Keep sim_controls (argument index 2: 1 = self) alive
                // at least as long as this OutputManagerH5, since it
                // stores only a live reference to it, exactly as
                // Cluster's own constructor does for its own controls
                // argument -- see BindCluster.cpp's own comment.
                py::keep_alive<1, 2>());
}
// NOLINTEND(misc-include-cleaner)
