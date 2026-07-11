/**
 * @file Bindings.cpp
 * @author Mark Krumholz
 * @brief Implement Python bindings for slug
 * @date 2026-07-11
 * @details
 * This module creates Python bindings that expose some of
 * the functionality of slug.
 */

#include "../extern/pybind11/include/pybind11/numpy.h"
#include "../extern/pybind11/include/pybind11/pybind11.h"
#include "../extern/pybind11/include/pybind11/stl.h"
#include "../interpolation/Interpolator1D.hpp"
#include "../tracks/TrackCommons.hpp"
#include "../tracks/Tracks2D.hpp"
#include "../tracks/Tracks3D.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

// Number of quantities tabulated at each track point; this is the NF
// value Tracks3D uses internally for both Mesh3DInterpolator and the
// Interpolator1D objects its getTrack()/getIsochrone() methods return.
constexpr std::size_t nQty = static_cast<std::size_t>(tracks::FieldIdx::nTrackQty);

// pybind11 binds concrete types, not class templates, so
// Interpolator1D can't be bound directly. Explicitly instantiating it
// here, with the same NF value Tracks3D uses, forces the compiler to
// generate the full class definition -- including every member
// function referenced in the bindings below -- in this translation
// unit for pybind11 to bind against.
template class interp::Interpolator1D<nQty>;
using Interp1D = interp::Interpolator1D<nQty>;

// Disable linting for includes -- the pybind macro magic seems to confuse
// the linter
// NOLINTBEGIN(misc-include-cleaner)

PYBIND11_MODULE(slug, m, py::mod_gil_not_used()) { 
    m.doc() = "slug Python frontend"; // optional module docstring

    py::class_<Interp1D, py::smart_holder>(m, "Interpolator1D")
        .def(py::init<
                const std::vector<double>&,                   // x
                const std::array<std::vector<double>, nQty>&  // f
                >(),
                "Construct an Interpolator1D",
                py::arg("x"),
                py::arg("f")
        )
        .def("xMin", &Interp1D::xMin,
                "Get minimum allowed x")
        .def("xMax", &Interp1D::xMax,
                "Get maximum allowed x")
        .def("xRange", &Interp1D::xRange,
                "Get allowed range in x")
        .def("__call__",
                static_cast<std::array<double, nQty> (Interp1D::*)(double) const>(
                    &Interp1D::operator()),
                "Interpolate to a given point",
                py::arg("x"))
        .def("__call__",
                py::vectorize(
                    static_cast<double (Interp1D::*)(double, std::size_t) const>(
                        &Interp1D::operator())),
                "Interpolate a single quantity to a given point, or "
                "elementwise over numpy arrays of points and/or indices "
                "(which broadcast against each other, so e.g. an array of "
                "x with idx=range(9) returns all quantities for every x)",
                py::arg("x"), py::arg("idx"))
        .def("__call__",
                py::vectorize(
                    [](const Interp1D* self, const double x, const std::string name) -> double // NOLINT(performance-unnecessary-value-param); this has to be a value rather than a reference due to pybind11 limitations
                    {
                        const auto *const it = std::ranges::find(tracks::fieldStr, name);
                        if (it == tracks::fieldStr.end())
                        {
                            throw std::runtime_error(
                                "Interpolator1D: unrecognized field name " + name);
                        }
                        const auto idx = static_cast<std::size_t>(
                            std::distance(tracks::fieldStr.begin(), it));
                        return (*self)(x, idx);
                    }),
                "Interpolate a named quantity (one of tracks::fieldStr) to "
                "a given point, or elementwise over a numpy array of "
                "points; raises RuntimeError if name is not recognized",
                py::arg("x"), py::arg("name"));

    py::class_<tracks::Tracks3D, py::smart_holder>(m, "Tracks3D")
        .def(py::init<
                const std::string&, // trackName
                double,             // fehMin
                double,             // fehMax
                double,             // vvcrit
                double,             // afe
                const std::string&  // registryName
                >(),
                "Construct a Tracks3D object from tracks on disk",
                py::arg("trackName"),
                py::arg("fehMin"),
                py::arg("fehMax"),
                py::arg("vvcrit") = 0,
                py::arg("afe") = 0,
                py::arg("registryName") = tracks::defaultRegistry
        )
        .def("mMin", &tracks::Tracks3D::mMin,
                "Return the minimum mass in the tracks")
        .def("mMax", &tracks::Tracks3D::mMax,
                "Return the maximum mass in the tracks")
        .def("tMin", &tracks::Tracks3D::tMin,
                "Return the minimum time in the tracks")
        .def("tMax", &tracks::Tracks3D::tMax,
                "Return the maximum time in the tracks")
        .def("feH", &tracks::Tracks3D::feH,
                "Return the [Fe/H] values spanned by this set of tracks")
        .def("aFe", &tracks::Tracks3D::aFe,
                "Return the [alpha/Fe] value of this set of tracks")
        .def("vVcrit", &tracks::Tracks3D::vVcrit,
                "Return the v/vcrit value of this set of tracks")
        .def("getTrack", &tracks::Tracks3D::getTrack,
                "Return the track for a star of a given mass and [Fe/H]",
                py::arg("m"), py::arg("feh"))
        .def("getIsochrone", &tracks::Tracks3D::getIsochrone,
                "Return the isochrone at a given time and [Fe/H]",
                py::arg("t"), py::arg("feh"));

    py::class_<tracks::Tracks2D, py::smart_holder>(m, "Tracks2D")
        .def(py::init<
                const std::string&, // trackName
                double,             // feh
                double,             // vvcrit
                double,             // afe
                const std::string&, // registryName
                std::size_t         // ntMin
                >(),
                "Construct a Tracks2D object from tracks on disk",
                py::arg("trackName"),
                py::arg("feh") = 0,
                py::arg("vvcrit") = 0,
                py::arg("afe") = 0,
                py::arg("registryName") = tracks::defaultRegistry,
                py::arg("ntMin") = 0
        )
        .def("mMin", &tracks::Tracks2D::mMin,
                "Return the minimum mass in the tracks")
        .def("mMax", &tracks::Tracks2D::mMax,
                "Return the maximum mass in the tracks")
        .def("tMin", &tracks::Tracks2D::tMin,
                "Return the minimum time in the tracks")
        .def("tMax", &tracks::Tracks2D::tMax,
                "Return the maximum time in the tracks")
        .def("feH", &tracks::Tracks2D::feH,
                "Return the [Fe/H] value of this set of tracks")
        .def("aFe", &tracks::Tracks2D::aFe,
                "Return the [alpha/Fe] value of this set of tracks")
        .def("vVcrit", &tracks::Tracks2D::vVcrit,
                "Return the v/vcrit value of this set of tracks")
        .def("starLifetime", &tracks::Tracks2D::starLifetime,
                "Return the lifetime of a star of a given mass",
                py::arg("m"))
        .def("liveMassRange", &tracks::Tracks2D::liveMassRange,
                "Return the range of stellar masses alive at a given time",
                py::arg("t"))
        .def("getTrack", &tracks::Tracks2D::getTrack,
                "Return the track for a star of a given mass",
                py::arg("m"))
        .def("getIsochrone", &tracks::Tracks2D::getIsochrone,
                "Return the isochrone at a given time",
                py::arg("t"));
}

// NOLINTEND(misc-include-cleaner)
