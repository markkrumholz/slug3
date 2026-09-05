/**
 * @file BindOutputManagerAscii.cpp
 * @author Mark Krumholz
 * @brief Python bindings for io::OutputManagerAscii
 * @date 2026-08-09
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#include "Bindings.hpp"
#include "../io/OutputManagerAscii.hpp"
#include "../io/SimControls.hpp"
#include <pybind11/pybind11.h>
#include <string_view>

// Numpy-style docstring for the Python binding below
static constexpr std::string_view constructorDocstring = R"doc(Open the ascii output files and write their headers.

Parameters
----------
sim_controls : SimControls
    Simulation controls (physics settings and control-flow settings
    together); sim_controls must outlive this OutputManagerAscii, and
    every value it reads from sim_controls (e.g. outDir()/modelName(),
    read once here, at construction) reflects sim_controls's state at
    this moment, not any later change. sim_controls.inputDeckStr() is
    written verbatim into the <model_name>_summary.txt file's own
    header, so a file this OutputManagerAscii produces always records
    exactly the deck that actually built sim_controls (empty if
    sim_controls was itself built with no input deck at all).

Throws
------
RuntimeError
    If any output file this would create (<model_name>_summary.txt,
    and -- depending on what sim_controls requested --
    <model_name>_clusters.txt/_cluster_spectra.txt/_cluster_phot.txt)
    already exists or cannot be opened.)doc";

// Disable linting for includes -- the pybind macro magic seems to confuse
// the linter
// NOLINTBEGIN(misc-include-cleaner)
void bindOutputManagerAscii(py::module_& m)
{
    py::class_<io::OutputManagerAscii, io::OutputManager, py::smart_holder>(m, "OutputManagerAscii")
        .def(py::init<const io::SimControls&>(),
                constructorDocstring.data(),
                py::arg("sim_controls"),
                // Keep sim_controls (argument index 2: 1 = self) alive
                // at least as long as this OutputManagerAscii -- see
                // BindOutputManagerH5.cpp's own identical comment.
                py::keep_alive<1, 2>());
}
// NOLINTEND(misc-include-cleaner)
