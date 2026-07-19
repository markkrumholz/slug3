/**
 * @file BindSimPhysics.cpp
 * @author Mark Krumholz
 * @brief Python bindings for io::SimPhysics
 * @date 2026-07-21
 */

#include "Bindings.hpp"
#include "../io/SimControls.hpp"
#include "../io/SimPhysics.hpp"
#include <memory>
#include <pybind11/pybind11.h>
#include <stdexcept>
#include <string>
#include <string_view>
#include <toml.hpp>

// Map the Python-facing sim_type string to SimControls::SimType,
// mirroring the sim_type key SimControls itself reads from a deck
// (see SimControls.cpp) -- SimControls is not otherwise exposed to
// Python, so this is done by hand here instead of constructing one
static auto simTypeFromString(const std::string& simType) -> io::SimControls::SimType
{
    if (simType == "cluster") { return io::SimControls::SimType::cluster; }
    if (simType == "galaxy") { return io::SimControls::SimType::galaxy; }
    throw std::runtime_error("SimPhysics: sim_type must be 'cluster' or 'galaxy'");
}

// Numpy-style docstring for the Python constructor binding. This is
// a convenience constructor, not a direct binding of SimPhysics's
// real (toml::table, SimControls::SimType) constructor -- toml++ is
// a vendored third-party library with no Python bindings of its own,
// so exposing it was not worth the surface area for what is, from
// Python, just a file to load.
static constexpr std::string_view constructorDocstring = R"doc(Construct a SimPhysics object by parsing a slug input deck.

Parameters
----------
path : str
    Path to a slug TOML input deck.
sim_type : str
    Either "cluster" or "galaxy"; controls whether galaxy-specific
    quantities (the CLF and SFR) are read from the deck.

Throws
------
RuntimeError
    If the file cannot be parsed, sim_type is not "cluster" or
    "galaxy", or the deck is otherwise invalid.)doc";

// Disable linting for includes -- the pybind macro magic seems to confuse
// the linter
// NOLINTBEGIN(misc-include-cleaner)
void bindSimPhysics(py::module_& m)
{
    py::class_<io::SimPhysics, py::smart_holder>(m, "SimPhysics")
        .def(py::init(
                [](const std::string& path, const std::string& simType)
                    -> std::unique_ptr<io::SimPhysics>
                {
                    const toml::table inputDeck = toml::parse_file(path);
                    return std::make_unique<io::SimPhysics>(
                        inputDeck, simTypeFromString(simType));
                }),
                constructorDocstring.data(),
                py::arg("path"), py::arg("sim_type"));
}
// NOLINTEND(misc-include-cleaner)
