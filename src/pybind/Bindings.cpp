/**
 * @file Bindings.cpp
 * @author Mark Krumholz
 * @brief Implement Python bindings for slug
 * @date 2026-07-11
 * @details
 * This module creates Python bindings that expose some of
 * the functionality of slug. The bindings for each class live in
 * their own translation unit (see Bindings.hpp); this file's
 * PYBIND11_MODULE block just calls all of them.
 */

#include "Bindings.hpp"
#include <pybind11/pybind11.h>

// Disable linting for includes -- the pybind macro magic seems to confuse
// the linter
// NOLINTBEGIN(misc-include-cleaner)
PYBIND11_MODULE(slug, m, py::mod_gil_not_used()) {
    m.doc() = "slug Python frontend"; // optional module docstring

    bindInterpolator1D(m);
    bindTracks3D(m);
    bindTracks2D(m);
}
// NOLINTEND(misc-include-cleaner)
