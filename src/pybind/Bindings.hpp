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
#include "../tracks/TrackCommons.hpp"
#include <cstddef>
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <string>
#include <toml.hpp>

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
 * @brief Bind io::OutputManager as OutputManager
 * @details
 * OutputManager is abstract (writeCluster()/writeClusterSpec()/
 * writeClusterPhot() are pure virtual), so no constructor is exposed
 * here, and none of those methods are bound either (see
 * BindOutputManagerH5.cpp's/BindOutputManagerAscii.cpp's own file
 * comments on why). Python code only ever encounters an OutputManager
 * through an OutputManagerH5 or OutputManagerAscii, and only to hand
 * one to SimCluster's own constructor. Must run before
 * bindOutputManagerH5()/bindOutputManagerAscii(), since those register
 * OutputManager as their Python base -- mirrors bindFilter()'s own
 * ordering requirement relative to bindFilterIdeal()/
 * bindFilterTabulated(), for the identical reason (see BindFilter.cpp).
 */
void bindOutputManager(py::module_& m);

/**
 * @brief Bind io::OutputManagerH5 as OutputManagerH5
 */
void bindOutputManagerH5(py::module_& m);

/**
 * @brief Bind io::OutputManagerAscii as OutputManagerAscii
 */
void bindOutputManagerAscii(py::module_& m);

/**
 * @brief Bind core::SimCluster as SimCluster
 */
void bindSimCluster(py::module_& m);

/**
 * @brief Parse a string as either literal TOML text or a path to a TOML file
 * @param pathOrContent Either the text of a TOML document, or a path
 *   to one on disk
 * @return The parsed toml::table
 * @throws toml::parse_error if pathOrContent parses as neither literal
 *   TOML text nor a path to a file that itself contains valid TOML
 * @details
 * Tried first as literal TOML text (via toml::parse); if that fails to
 * parse, tried again as a file path instead (via toml::parse_file) --
 * so e.g. a path like "deck.toml" (not valid TOML on its own) falls
 * through to being read as a file, while a string like "n_trial = 10"
 * (a path that could never exist) is used directly as the document's
 * own content. Shared by every binding that accepts a toml input deck
 * as a Python string (OutputManagerH5's and OutputManagerAscii's own
 * constructors) -- SimControls's own constructor needs to tell which
 * of the two interpretations actually matched (see its own comment in
 * BindSimControls.cpp), so it does not use this shared helper, even
 * though its logic is the same.
 */
auto parseTomlPathOrContent(const std::string& pathOrContent) -> toml::table;

/**
 * @brief Resolve an optional py::object SimControls argument to a real SimControls
 * @param controls The Python-facing controls argument; if py::none(),
 *   fallback is used instead
 * @param fallback Fallback used when controls is py::none(); must be
 *   a reference with a lifetime independent of this call (e.g. a
 *   function-local static, as sharedDefaultControls() and
 *   sharedMinimalControls() both return) -- never a stack-local
 *   temporary, since every binding using this stores the resolved
 *   reference live (see Specsyn's and Cluster's own controls_
 *   members) rather than copying out of it, so it must remain valid
 *   for as long as the object built from it does, not merely for the
 *   duration of this call
 * @return A const reference to either the SimControls held by
 *   controls, or fallback
 * @details
 * Shared by every binding that takes an optional SimControls (e.g.
 * SpecsynBlackbody's constructor, Cluster's constructor), so each
 * doesn't have to duplicate this same is_none()/py::cast dance.
 */
auto resolveControls(const py::object& controls,
    const io::SimControls& fallback) -> const io::SimControls&;

/**
 * @brief Build a fresh SimControls from slug's own bundled default deck
 * @return A new SimControls built from src/pybind/assets/PyDefaults.toml,
 *   independent of any other call's result
 * @throws std::runtime_error if the bundled default deck cannot be found
 * @details
 * Used directly by SimControls's own Python constructor, when path is
 * empty; see sharedDefaultControls() for the shared, non-owning
 * counterpart used by Cluster's own default controls argument.
 */
auto buildDefaultControls() -> std::unique_ptr<io::SimControls>;

/**
 * @brief Get slug's own default SimControls, shared for the life of the program
 * @return A const reference to a SimControls built (lazily, once) by
 *   buildDefaultControls()
 * @details
 * Used as Cluster's own default controls argument, so
 * slug.Cluster(mass) works without requiring a SimControls to be built
 * and kept alive by the caller. Unlike a SimControls a caller builds
 * itself, this instance is a function-local static -- its lifetime is
 * governed by the C++ runtime rather than Python's reference
 * counting, so a Cluster referencing it never needs py::keep_alive to
 * stay safe. It is shared (not rebuilt per Cluster), both to avoid
 * repeatedly reloading the real MIST tracks and spectral-library
 * chain PyDefaults.toml references, and because SimControls is
 * ordinarily meant to be shared across many Cluster objects anyway
 * (see SimCluster::run()).
 */
auto sharedDefaultControls() -> const io::SimControls&;

/**
 * @brief Get a minimal, all-C++-defaults SimControls, shared for the life of the program
 * @return A const reference to a default-constructed SimControls
 *   (io::SimControls{}: simType = none, every physics setting
 *   invalid/empty, integrator tolerances at their own C++ defaults)
 * @details
 * Used as the default controls argument for every Specsyn-derived
 * class's Python constructor, none of which need anything from
 * SimControls beyond the integrator tolerances. Deliberately distinct
 * from sharedDefaultControls(): that one loads slug's real bundled
 * physics deck (PyDefaults.toml), which needs the real MIST tracks
 * and spectral-library chain on disk -- overkill, and a dependency
 * this cheap fallback should not carry, for something that only ever
 * reads three scalar tolerances. Like sharedDefaultControls(), this is
 * a function-local static, so it needs no py::keep_alive protection.
 */
auto sharedMinimalControls() -> const io::SimControls&;

#endif // BINDINGS_HPP
