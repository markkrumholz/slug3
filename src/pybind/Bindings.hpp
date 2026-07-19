/**
 * @file Bindings.hpp
 * @author Mark Krumholz
 * @brief Shared declarations for slug's Python bindings
 * @date 2026-07-20
 * @details
 * Binding code for each bound class lives in its own translation
 * unit (BindInterpolator1D.cpp, BindTracks2D.cpp, BindTracks3D.cpp,
 * ...), each defining one bind*() function declared here.
 * Bindings.cpp's PYBIND11_MODULE block just calls all of them --
 * PYBIND11_MODULE itself can only appear once (it expands to the
 * module's single PyInit_slug entry point), so this is the pattern
 * pybind11 itself recommends for splitting binding code across
 * multiple files.
 */

#ifndef BINDINGS_HPP
#define BINDINGS_HPP

#include "../interpolation/Interpolator1D.hpp"
#include "../tracks/TrackCommons.hpp"
#include <cstddef>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// Number of quantities tabulated at each track point; this is the NF
// value Tracks3D uses internally for both Mesh3DInterpolator and the
// Interpolator1D objects its getTrack()/getIsochrone() methods return.
constexpr std::size_t nQty = static_cast<std::size_t>(tracks::FieldIdx::nTrackQty);

// pybind11 binds concrete types, not class templates, so
// Interpolator1D can't be bound directly. BindInterpolator1D.cpp
// provides the one explicit instantiation definition (forcing the
// compiler to generate the full class definition -- including every
// member function referenced in the bindings -- for pybind11 to bind
// against); every other translation unit that only uses Interp1D
// (e.g. to name a return type) declares it extern here instead, so
// it isn't redundantly instantiated in each of them.
extern template class interp::Interpolator1D<nQty>;
using Interp1D = interp::Interpolator1D<nQty>;

/**
 * @brief Bind interp::Interpolator1D<nQty> as Interpolator1D
 */
void bindInterpolator1D(py::module_& m);

/**
 * @brief Bind tracks::Tracks2D as Tracks2D
 */
void bindTracks2D(py::module_& m);

/**
 * @brief Bind tracks::Tracks3D as Tracks3D
 */
void bindTracks3D(py::module_& m);

/**
 * @brief Bind io::SimPhysics as SimPhysics
 */
void bindSimPhysics(py::module_& m);

/**
 * @brief Bind core::Cluster as Cluster
 */
void bindCluster(py::module_& m);

#endif // BINDINGS_HPP
