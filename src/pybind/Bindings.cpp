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
#include "../extern/pybind11/include/pybind11/stl.h" // NOLINT(misc-include-cleaner); this is needed for correct Python binding, even if clang-tidy can't recognize it
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
#include <utility>
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
                Interp1D::constructorDocstring.data(),
                py::arg("x"),
                py::arg("f")
        )
        .def("xMin", &Interp1D::xMin,
                Interp1D::xMinDocstring.data())
        .def("xMax", &Interp1D::xMax,
                Interp1D::xMaxDocstring.data())
        .def("xRange", &Interp1D::xRange,
                Interp1D::xRangeDocstring.data())
        .def("__call__",
                [](const Interp1D& self, const double x) -> std::array<double, nQty>
                {
                    if (x < self.xMin() || x > self.xMax())
                    {
                        throw std::runtime_error(
                            "Interpolator1D: x = " + std::to_string(x) +
                            " is outside the allowed range [" +
                            std::to_string(self.xMin()) + ", " +
                            std::to_string(self.xMax()) + "]");
                    }
                    return self(x);
                },
                Interp1D::callDocstring.data(),
                py::arg("x"))
        .def("__call__",
                py::vectorize(
                    [](const Interp1D* self, const double x, const std::size_t idx) -> double
                    {
                        if (x < self->xMin() || x > self->xMax())
                        {
                            throw std::runtime_error(
                                "Interpolator1D: x = " + std::to_string(x) +
                                " is outside the allowed range [" +
                                std::to_string(self->xMin()) + ", " +
                                std::to_string(self->xMax()) + "]");
                        }
                        return (*self)(x, idx);
                    }),
                Interp1D::callIdxDocstring.data(),
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
                        if (x < self->xMin() || x > self->xMax())
                        {
                            throw std::runtime_error(
                                "Interpolator1D: x = " + std::to_string(x) +
                                " is outside the allowed range [" +
                                std::to_string(self->xMin()) + ", " +
                                std::to_string(self->xMax()) + "]");
                        }
                        const auto idx = static_cast<std::size_t>(
                            std::distance(tracks::fieldStr.begin(), it));
                        return (*self)(x, idx);
                    }),
                Interp1D::callNameDocstring.data(),
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
                tracks::Tracks3D::constructorDocstring.data(),
                py::arg("trackName"),
                py::arg("fehMin"),
                py::arg("fehMax"),
                py::arg("vvcrit") = 0,
                py::arg("afe") = 0,
                py::arg("registryName") = tracks::defaultRegistry
        )
        .def("mMin", &tracks::Tracks3D::mMin,
                tracks::Tracks3D::mMinDocstring.data())
        .def("mMax", &tracks::Tracks3D::mMax,
                tracks::Tracks3D::mMaxDocstring.data())
        .def("logTMin", &tracks::Tracks3D::logTMin,
                tracks::Tracks3D::logTMinDocstring.data())
        .def("logTMax", &tracks::Tracks3D::logTMax,
                tracks::Tracks3D::logTMaxDocstring.data())
        .def("feH", &tracks::Tracks3D::feH,
                tracks::Tracks3D::feHDocstring.data())
        .def("aFe", &tracks::Tracks3D::aFe,
                tracks::Tracks3D::aFeDocstring.data())
        .def("vVcrit", &tracks::Tracks3D::vVcrit,
                tracks::Tracks3D::vVcritDocstring.data())
        .def("getTrack",
                [](const tracks::Tracks3D& self, const double m, const double feh)
                {
                    if (m < self.mMin() || m > self.mMax())
                    {
                        throw std::runtime_error(
                            "Tracks3D: mass = " + std::to_string(m) +
                            " is outside the allowed range [" +
                            std::to_string(self.mMin()) + ", " +
                            std::to_string(self.mMax()) + "]");
                    }
                    return self.getTrack(m, feh);
                },
                tracks::Tracks3D::getTrackDocstring.data(),
                py::arg("m"), py::arg("feh"))
        .def("getIsochrone",
                [](const tracks::Tracks3D& self, const double logT, const double feh)
                {
                    // Built up by hand, casting each segment individually,
                    // rather than binding tracks::Tracks3D::getIsochrone
                    // directly: pybind11's generic std::vector<T> caster
                    // can't move a std::unique_ptr<Interp1D> out of the
                    // vector when Interp1D is bound with py::smart_holder,
                    // even though casting a single std::unique_ptr<Interp1D>
                    // (as getTrack does) works fine
                    auto isochrone = self.getIsochrone(logT, feh);
                    py::list result;
                    for (auto& seg : isochrone) { result.append(py::cast(std::move(seg))); }
                    return result;
                },
                tracks::Tracks3D::getIsochroneDocstring.data(),
                py::arg("logT"), py::arg("feh"));

    py::class_<tracks::Tracks2D, py::smart_holder>(m, "Tracks2D")
        .def(py::init<
                const std::string&, // trackName
                double,             // feh
                double,             // vvcrit
                double,             // afe
                const std::string&  // registryName
                >(),
                tracks::Tracks2D::constructorDocstring.data(),
                py::arg("trackName"),
                py::arg("feh") = 0,
                py::arg("vvcrit") = 0,
                py::arg("afe") = 0,
                py::arg("registryName") = tracks::defaultRegistry
        )
        .def("mMin", &tracks::Tracks2D::mMin,
                tracks::Tracks2D::mMinDocstring.data())
        .def("mMax", &tracks::Tracks2D::mMax,
                tracks::Tracks2D::mMaxDocstring.data())
        .def("logTMin", &tracks::Tracks2D::logTMin,
                tracks::Tracks2D::logTMinDocstring.data())
        .def("logTMax", &tracks::Tracks2D::logTMax,
                tracks::Tracks2D::logTMaxDocstring.data())
        .def("feH", &tracks::Tracks2D::feH,
                tracks::Tracks2D::feHDocstring.data())
        .def("aFe", &tracks::Tracks2D::aFe,
                tracks::Tracks2D::aFeDocstring.data())
        .def("vVcrit", &tracks::Tracks2D::vVcrit,
                tracks::Tracks2D::vVcritDocstring.data())
        .def("starLifetime", &tracks::Tracks2D::starLifetime,
                tracks::Tracks2D::starLifetimeDocstring.data(),
                py::arg("m"))
        .def("liveMassRange", &tracks::Tracks2D::liveMassRange,
                tracks::Tracks2D::liveMassRangeDocstring.data(),
                py::arg("t"))
        .def("getTrack",
                [](const tracks::Tracks2D& self, const double m)
                {
                    if (m < self.mMin() || m > self.mMax())
                    {
                        throw std::runtime_error(
                            "Tracks2D: mass = " + std::to_string(m) +
                            " is outside the allowed range [" +
                            std::to_string(self.mMin()) + ", " +
                            std::to_string(self.mMax()) + "]");
                    }
                    return self.getTrack(m);
                },
                tracks::Tracks2D::getTrackDocstring.data(),
                py::arg("m"))
        .def("getIsochrone",
                [](const tracks::Tracks2D& self, const double logT)
                {
                    // See the comment on the Tracks3D getIsochrone binding
                    // above for why this can't just bind
                    // tracks::Tracks2D::getIsochrone directly
                    auto isochrone = self.getIsochrone(logT);
                    py::list result;
                    for (auto& seg : isochrone) { result.append(py::cast(std::move(seg))); }
                    return result;
                },
                tracks::Tracks2D::getIsochroneDocstring.data(),
                py::arg("logT"));
}

// NOLINTEND(misc-include-cleaner)
