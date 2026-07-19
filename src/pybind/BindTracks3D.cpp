/**
 * @file BindTracks3D.cpp
 * @author Mark Krumholz
 * @brief Python bindings for tracks::Tracks3D
 * @date 2026-07-20
 */

#include "Bindings.hpp"
#include "../tracks/TrackCommons.hpp"
#include "../tracks/Tracks3D.hpp"
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // NOLINT(misc-include-cleaner); this is needed for correct Python binding, even if clang-tidy can't recognize it
#include <stdexcept>
#include <string>
#include <utility>

// Disable linting for includes -- the pybind macro magic seems to confuse
// the linter
// NOLINTBEGIN(misc-include-cleaner)
void bindTracks3D(py::module_& m)
{
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
                    -> std::unique_ptr<Interp1D>
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
                    -> py::list
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
}
// NOLINTEND(misc-include-cleaner)
