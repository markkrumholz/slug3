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
#include <memory>
#include <pybind11/pybind11.h>
#include <string>
#include <string_view>
#include <toml.hpp>
#include <utility>

namespace
{
    // Owns the toml::table that OutputManagerH5's own base class
    // (io::OutputManager) stores only a const reference to
    // (inputDeck_), since toml::table has no Python binding of its
    // own for a caller to keep alive independently. A private base
    // class rather than a member, specifically so it is guaranteed to
    // be constructed before the io::OutputManagerH5 base subobject
    // that needs a valid reference to it -- C++ always initializes
    // base classes, in declaration order, before a derived class's
    // own members, so listing this one first is what makes passing
    // inputDeck_ to io::OutputManagerH5's own constructor below safe.
    class InputDeckHolder
    {
    protected:
        explicit InputDeckHolder(toml::table inputDeck) : inputDeck_(std::move(inputDeck)) {}
        toml::table inputDeck_; // NOLINT(misc-non-private-member-variables-in-classes) -- only ever accessed by PyOutputManagerH5 below, which is what this class exists to support
    };

    // The concrete type actually constructed by the Python binding
    // below; returned as std::unique_ptr<io::OutputManagerH5> (pybind11
    // supports a py::init factory returning a unique_ptr to a subclass
    // of the exposed type), so this class itself is never exposed to
    // Python.
    class PyOutputManagerH5 : private InputDeckHolder, public io::OutputManagerH5
    {
    public:
        PyOutputManagerH5(const io::SimControls& simControls, toml::table inputDeck,
            const bool restart) :
            InputDeckHolder(std::move(inputDeck)),
            io::OutputManagerH5(simControls, InputDeckHolder::inputDeck_, restart)
        {
        }
    };
} // namespace

auto parseTomlPathOrContent(const std::string& pathOrContent) -> toml::table
{
    try
    {
        return toml::parse(pathOrContent);
    }
    catch (const toml::parse_error&) // NOLINT(bugprone-empty-catch) -- deliberately empty: this exception just means pathOrContent isn't literal TOML text, so fall through and treat it as a file path instead
    {
    }
    return toml::parse_file(pathOrContent);
}

// Numpy-style docstring for the Python binding below
static constexpr std::string_view constructorDocstring = R"doc(Open an HDF5 output file and write its header.

Parameters
----------
sim_controls : SimControls
    Simulation controls (physics settings and control-flow settings
    together); sim_controls must outlive this OutputManagerH5, and
    every value it reads from sim_controls (e.g. outDir()/modelName(),
    read once here, at construction) reflects sim_controls's state at
    this moment, not any later change.
input_deck : str
    Either the text of the slug TOML input deck that produced
    sim_controls, or a path to one on disk -- tried first as literal
    TOML text, then as a file path if that fails to parse (see
    SimControls's own constructor for the identical rule). Written
    verbatim into the output file's own input_deck group, so a file
    this OutputManagerH5 produces always records exactly the deck that
    was actually used.
restart : bool, default False
    Whether this run is resuming a previous, interrupted run from its
    most recent checkpoint, rather than starting a new one from
    scratch. If True, scans sim_controls.outDir() for the most recent
    checkpoint left behind by a previous run of the same model_name/
    out_dir and resumes numbering checkpoints from just after it,
    instead of starting over at checkpoint 0 -- see the C++
    OutputManagerH5::restartSetup()'s own documentation for the full
    detail. Only meaningful with HDF5 output and a non-zero
    outputs.checkpoint_interval in input_deck.

Throws
------
RuntimeError
    If the output file (sim_controls.outDir()/sim_controls.modelName()
    + ".h5") already exists, if input_deck matches neither a literal
    TOML document nor a file containing one, or if restart is True and
    sim_controls.outputMode() is ascii.)doc";

// Disable linting for includes -- the pybind macro magic seems to confuse
// the linter
// NOLINTBEGIN(misc-include-cleaner)
void bindOutputManagerH5(py::module_& m)
{
    py::class_<io::OutputManagerH5, io::OutputManager, py::smart_holder>(m, "OutputManagerH5")
        .def(py::init(
                [](const io::SimControls& simControls, const std::string& inputDeck,
                    const bool restart)
                    -> std::unique_ptr<io::OutputManagerH5>
                {
                    return std::make_unique<PyOutputManagerH5>(
                        simControls, parseTomlPathOrContent(inputDeck), restart);
                }),
                constructorDocstring.data(),
                py::arg("sim_controls"), py::arg("input_deck"),
                py::arg("restart") = false,
                // Keep sim_controls (argument index 2: 1 = self) alive
                // at least as long as this OutputManagerH5, since it
                // stores only a live reference to it, exactly as
                // Cluster's own constructor does for its own controls
                // argument -- see BindCluster.cpp's own comment.
                py::keep_alive<1, 2>());
}
// NOLINTEND(misc-include-cleaner)
