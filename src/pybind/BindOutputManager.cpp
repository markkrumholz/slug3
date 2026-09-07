/**
 * @file BindOutputManager.cpp
 * @author Mark Krumholz
 * @brief Python bindings for io::OutputManager
 * @date 2026-08-09
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 * @details
 * OutputManager is abstract (writeCluster()/writeClusterSpec()/
 * writeClusterPhot() are pure virtual), so no constructor is exposed
 * here -- Python code only ever encounters an OutputManager through an
 * OutputManagerH5 or OutputManagerAscii, constructed directly. None of
 * the write*() methods are bound either, on any of the three classes:
 * writing is driven entirely from the C++ side (SimCluster.run()), not
 * called directly from Python. Registering OutputManager as
 * OutputManagerH5's/OutputManagerAscii's Python base (see
 * BindOutputManagerH5.cpp/BindOutputManagerAscii.cpp) is what lets
 * pybind11 accept either one wherever C++ expects a
 * std::unique_ptr<io::OutputManager> -- e.g. SimCluster's own
 * constructor -- exactly mirroring BindFilter.cpp's own comment for
 * FilterIdeal/FilterTabulated's relationship to Filter.
 */

#include "Bindings.hpp"
#include "../io/OutputManager.hpp"
#include <pybind11/pybind11.h>
#include <string_view>

static constexpr std::string_view classDocstring = R"doc(Abstract base class to manage writing simulation outputs to disk.

Defines the interface and behavior shared by both output formats; the
format-specific work is implemented by the OutputManagerH5 and
OutputManagerAscii subclasses. No constructor is exposed here -- Python
code only ever encounters an OutputManager through one of those two
concrete subclasses, constructed directly.)doc";

// Disable linting for includes -- the pybind macro magic seems to confuse
// the linter
// NOLINTBEGIN(misc-include-cleaner)
void bindOutputManager(py::module_& m)
{
    py::class_<io::OutputManager, py::smart_holder>(m, "OutputManager", classDocstring.data());
}
// NOLINTEND(misc-include-cleaner)
