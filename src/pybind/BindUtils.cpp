/**
 * @file BindUtils.cpp
 * @author Mark Krumholz
 * @brief Python bindings for utils::getFilePath
 * @date 2026-10-24
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#include "Bindings.hpp"
#include "../utils/MiscUtils.hpp"
#include <filesystem>
#include <optional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // NOLINT(misc-include-cleaner); needed for std::optional conversion
#include <pybind11/stl/filesystem.h> // NOLINT(misc-include-cleaner); needed for std::filesystem::path conversion
#include <string>
#include <string_view>

static constexpr std::string_view getFilePathDocstring =
    R"doc(Find a file the way slug finds its own data files.

Parameters
----------
fileName : str
    The file's name or path.
prefix : str, optional
    A directory, relative to each search directory below, in which to
    look for fileName (e.g. "data/imfs"); default none.

Returns
-------
path : pathlib.Path or None
    The first match found, searching in this order: fileName itself
    (relative to the current working directory, or absolute); and, if
    fileName is not absolute, prefix/fileName within the directory named
    by the SLUG_DIR environment variable, within each directory listed in
    the SLUG_DATA_PATH environment variable, in order (slugpy adds slug's
    per-user data directory and its own bundled data to it), and within
    the slug source tree this module was built from, if any. None if
    there is no match.)doc";

// Disable linting for includes -- the pybind macro magic seems to confuse
// the linter
// NOLINTBEGIN(misc-include-cleaner)
void bindUtils(py::module_& m)
{
    m.def("getFilePath",
            [](const std::string& fileName, const std::string& prefix) -> std::optional<std::filesystem::path>
            {
                auto path = utils::getFilePath(fileName, prefix);
                if (path.empty()) { return std::nullopt; }
                return path;
            },
            getFilePathDocstring.data(), py::arg("fileName"), py::arg("prefix") = "");
}
// NOLINTEND(misc-include-cleaner)
