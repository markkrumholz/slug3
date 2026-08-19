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
#include <memory>
#include <pybind11/pybind11.h>
#include <string>
#include <string_view>
#include <toml.hpp>
#include <utility>

namespace
{
    // Owns the toml::table that OutputManagerAscii's own base class
    // (io::OutputManager) stores only a const reference to
    // (inputDeck_) -- see BindOutputManagerH5.cpp's InputDeckHolder,
    // which this is an exact copy of for OutputManagerAscii's own
    // base-before-member initialization needs.
    class InputDeckHolder
    {
    protected:
        explicit InputDeckHolder(toml::table inputDeck) : inputDeck_(std::move(inputDeck)) {}
        toml::table inputDeck_; // NOLINT(misc-non-private-member-variables-in-classes) -- only ever accessed by PyOutputManagerAscii below, which is what this class exists to support
    };

    // The concrete type actually constructed by the Python binding
    // below; returned as std::unique_ptr<io::OutputManagerAscii> --
    // see BindOutputManagerH5.cpp's PyOutputManagerH5 for why this
    // pattern is safe and why this class itself is never exposed to
    // Python.
    class PyOutputManagerAscii : private InputDeckHolder, public io::OutputManagerAscii
    {
    public:
        PyOutputManagerAscii(const io::SimControls& simControls, toml::table inputDeck) :
            InputDeckHolder(std::move(inputDeck)),
            io::OutputManagerAscii(simControls, InputDeckHolder::inputDeck_)
        {
        }
    };
} // namespace

// Numpy-style docstring for the Python binding below
static constexpr std::string_view constructorDocstring = R"doc(Open the ascii output files and write their headers.

Parameters
----------
sim_controls : SimControls
    Simulation controls (physics settings and control-flow settings
    together); sim_controls must outlive this OutputManagerAscii, and
    every value it reads from sim_controls (e.g. outDir()/modelName(),
    read once here, at construction) reflects sim_controls's state at
    this moment, not any later change.
input_deck : str
    Either the text of the slug TOML input deck that produced
    sim_controls, or a path to one on disk -- tried first as literal
    TOML text, then as a file path if that fails to parse (see
    SimControls's own constructor for the identical rule). Written
    verbatim into the <model_name>_summary.txt file's own header, so a
    file this OutputManagerAscii produces always records exactly the
    deck that was actually used.

Throws
------
RuntimeError
    If any output file this would create (<model_name>_summary.txt,
    and -- depending on what sim_controls requested --
    <model_name>_clusters.txt/_cluster_spectra.txt/_cluster_phot.txt)
    already exists or cannot be opened, or input_deck matches neither
    a literal TOML document nor a file containing one.)doc";

// Disable linting for includes -- the pybind macro magic seems to confuse
// the linter
// NOLINTBEGIN(misc-include-cleaner)
void bindOutputManagerAscii(py::module_& m)
{
    py::class_<io::OutputManagerAscii, io::OutputManager, py::smart_holder>(m, "OutputManagerAscii")
        .def(py::init(
                [](const io::SimControls& simControls, const std::string& inputDeck)
                    -> std::unique_ptr<io::OutputManagerAscii>
                {
                    return std::make_unique<PyOutputManagerAscii>(
                        simControls, parseTomlPathOrContent(inputDeck));
                }),
                constructorDocstring.data(),
                py::arg("sim_controls"), py::arg("input_deck"),
                // Keep sim_controls (argument index 2: 1 = self) alive
                // at least as long as this OutputManagerAscii -- see
                // BindOutputManagerH5.cpp's own identical comment.
                py::keep_alive<1, 2>());
}
// NOLINTEND(misc-include-cleaner)
