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
#include "../io/SimControls.hpp"
#include "../io/SimPhysics.hpp"
#include "../tracks/TrackCommons.hpp"
#include <cstddef>
#include <memory>
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

// A second explicit instantiation, for the single-quantity case (e.g.
// phot::FilterTabulated::response_) -- a genuinely different C++ type
// from Interp1D above (nQty != 1), so it needs its own pybind11 class
// registration; see BindInterpolator1DScalar.cpp
extern template class interp::Interpolator1D<1>;
using Interp1DScalar = interp::Interpolator1D<1>;

/**
 * @brief Bind interp::Interpolator1D<nQty> as Interpolator1D
 */
void bindInterpolator1D(py::module_& m);

/**
 * @brief Bind interp::Interpolator1D<1> as Interpolator1DScalar
 */
void bindInterpolator1DScalar(py::module_& m);

/**
 * @brief Bind tracks::Tracks2D as Tracks2D
 */
void bindTracks2D(py::module_& m);

/**
 * @brief Bind tracks::Tracks3D as Tracks3D
 */
void bindTracks3D(py::module_& m);

/**
 * @brief Bind io::SimControls as SimControls
 */
void bindSimControls(py::module_& m);

/**
 * @brief Bind io::SimPhysics as SimPhysics
 */
void bindSimPhysics(py::module_& m);

/**
 * @brief Bind Specsyn and its concrete subclasses
 */
void bindSpecsyn(py::module_& m);

/**
 * @brief Bind core::Cluster as Cluster
 */
void bindCluster(py::module_& m);

/**
 * @brief Bind phot::Filter as Filter
 * @details
 * Must run before bindFilterIdeal()/bindFilterTabulated(), since those
 * register phot::Filter as their Python base class -- this is what
 * lets pybind11 automatically hand back a FilterIdeal or
 * FilterTabulated (rather than a bare Filter) wherever C++ returns a
 * Filter reference/pointer whose most-derived type is one of them
 * (e.g. FilterCollection::getFilter()).
 */
void bindFilter(py::module_& m);

/**
 * @brief Bind phot::FilterIdeal as FilterIdeal
 */
void bindFilterIdeal(py::module_& m);

/**
 * @brief Bind phot::FilterTabulated as FilterTabulated
 */
void bindFilterTabulated(py::module_& m);

/**
 * @brief Bind phot::FilterCollection as FilterCollection, and phot::PhotSystem as PhotSystem
 */
void bindFilterCollection(py::module_& m);

/**
 * @brief Bind phot::PhotConvert as the module-level function PhotConvert
 */
void bindPhotConvert(py::module_& m);

/**
 * @brief Bind pdfs::PDF as PDF
 */
void bindPDF(py::module_& m);

/**
 * @brief Resolve an optional py::object SimControls argument to a real SimControls
 * @param controls The Python-facing controls argument; if py::none(),
 *   defaultControls is used instead
 * @param defaultControls Fallback used when controls is py::none();
 *   must outlive the returned reference, so callers should keep it
 *   alive (as a local variable) for the duration of the call using
 *   the returned reference
 * @return A const reference to either the SimControls held by
 *   controls, or defaultControls
 * @details
 * Shared by every binding that takes an optional SimControls (e.g.
 * SpecsynBlackbody's constructor, Cluster's constructor), so each
 * doesn't have to duplicate this same is_none()/py::cast dance.
 */
auto resolveControls(const py::object& controls,
    const io::SimControls& defaultControls) -> const io::SimControls&;

/**
 * @brief Build a fresh SimPhysics from slug's own bundled default deck
 * @return A new SimPhysics built from src/pybind/assets/PyDefaults.toml,
 *   independent of any other call's result
 * @throws std::runtime_error if the bundled default deck cannot be found
 * @details
 * Used directly by SimPhysics's own Python constructor, when path is
 * empty; see sharedDefaultSimPhysics() for the shared, non-owning
 * counterpart used by Cluster's own default physics argument.
 */
auto buildDefaultSimPhysics() -> std::unique_ptr<io::SimPhysics>;

/**
 * @brief Get slug's own default SimPhysics, shared for the life of the program
 * @return A const reference to a SimPhysics built (lazily, once) by
 *   buildDefaultSimPhysics()
 * @details
 * Used as Cluster's own default physics argument, so
 * slug.Cluster(mass) works without requiring a SimPhysics to be built
 * and kept alive by the caller. Unlike a SimPhysics a caller builds
 * itself, this instance is a function-local static -- its lifetime is
 * governed by the C++ runtime rather than Python's reference
 * counting, so a Cluster referencing it never needs py::keep_alive to
 * stay safe. It is shared (not rebuilt per Cluster), both to avoid
 * repeatedly reloading the real MIST tracks and spectral-library
 * chain PyDefaults.toml references, and because SimPhysics is
 * ordinarily meant to be shared across many Cluster objects anyway
 * (see SimCluster::run()).
 */
auto sharedDefaultSimPhysics() -> const io::SimPhysics&;

#endif // BINDINGS_HPP
