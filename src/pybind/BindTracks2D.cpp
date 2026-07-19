/**
 * @file BindTracks2D.cpp
 * @author Mark Krumholz
 * @brief Python bindings for tracks::Tracks2D
 * @date 2026-07-20
 */

#include "Bindings.hpp"
#include "../tracks/TrackCommons.hpp"
#include "../tracks/Tracks2D.hpp"
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // NOLINT(misc-include-cleaner); this is needed for correct Python binding, even if clang-tidy can't recognize it
#include <stdexcept>
#include <string>
#include <utility>

// Disable linting for includes -- the pybind macro magic seems to confuse
// the linter
// NOLINTBEGIN(misc-include-cleaner)
void bindTracks2D(py::module_& m)
{
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
                    -> std::unique_ptr<Interp1D>
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
                    -> py::list
                {
                    // See the comment on Tracks3D's getIsochrone binding
                    // (BindTracks3D.cpp) for why this can't just bind
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
