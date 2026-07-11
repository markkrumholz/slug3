/**
 * @file Bindings.cpp
 * @author Mark Krumholz
 * @brief Implement Python bindings for slug
 * @date 2026-07-11
 * @details
 * This module creates Python bindings that expose some of
 * the functionality of slug.
 */

#include "../tracks/Tracks3D.hpp"
#include "../extern/pybind11/include/pybind11/pybind11.h"
#include "../extern/pybind11/include/pybind11/stl.h"
#include <string>

namespace py = pybind11;

PYBIND11_MODULE(slug, m, py::mod_gil_not_used()) {
    m.doc() = "slug Python frontend"; // optional module docstring
    py::class_<tracks::Tracks3D, py::smart_holder>(m, "Tracks3D")
        .def(py::init<
                const std::string&, // registryName
                const std::string&, // trackName
                double,             // fehMin
                double,             // fehMax
                double,             // vvcrit
                double              // afe
                >(),
                "Construct a Tracks3D object from tracks on disk",
                py::arg("registryName"),
                py::arg("trackName"),
                py::arg("fehMin"),
                py::arg("fehMax"),
                py::arg("vvcrit") = 0,
                py::arg("afe") = 0
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
                "Return the v/vcrit value of this set of tracks");
}