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
#include <string_view>
#include <utility>

// Numpy-style docstrings for the Python bindings below
static constexpr std::string_view constructorDocstring = R"doc(Construct a Tracks3D object from tracks on disk.

Parameters
----------
trackName : str
    Name of the track set within the registry.
fehMin : float
    Minimum [Fe/H] value to include.
fehMax : float
    Maximum [Fe/H] value to include.
vvcrit : float, optional
    Rotation rate v/vcrit of the desired tracks. Default is 0.0.
afe : float, optional
    [alpha/Fe] value of the desired tracks. Default is 0.0.
registryName : str, optional
    Path to the track registry file. Default is the package's default
    registry (data/tracks/tracks.toml).

Throws
------
RuntimeError
    If no tracks in trackName match vvcrit and afe within the
    requested [Fe/H] range.)doc";

static constexpr std::string_view mMinDocstring = R"doc(Return the minimum mass spanned by this set of tracks.

Returns
-------
mmin : float
    Minimum mass, in Msun.)doc";

static constexpr std::string_view mMaxDocstring = R"doc(Return the maximum mass spanned by this set of tracks.

Returns
-------
mmax : float
    Maximum mass, in Msun.)doc";

static constexpr std::string_view logTMinDocstring = R"doc(Return the minimum log10(time) spanned by this set of tracks.

Returns
-------
logtmin : float
    Minimum log10(time / yr).)doc";

static constexpr std::string_view logTMaxDocstring = R"doc(Return the maximum log10(time) spanned by this set of tracks.

Returns
-------
logtmax : float
    Maximum log10(time / yr).)doc";

static constexpr std::string_view feHDocstring = R"doc(Return the [Fe/H] values spanned by this set of tracks.

Returns
-------
feh : list of float
    [Fe/H] values spanned by this set of tracks.)doc";

static constexpr std::string_view aFeDocstring = R"doc(Return the [alpha/Fe] value of this set of tracks.

Returns
-------
afe : float
    [alpha/Fe] value of this set of tracks, or NaN if not available.)doc";

static constexpr std::string_view vVcritDocstring = R"doc(Return the v/vcrit value of this set of tracks.

Returns
-------
vvcrit : float
    v/vcrit value of this set of tracks, or NaN if not available.)doc";

static constexpr std::string_view getTrackDocstring = R"doc(Return the track for a star of a given mass and [Fe/H].

Parameters
----------
m : float
    Stellar mass, in Msun.
feh : float
    [Fe/H] value of the tracks to use.

Returns
-------
interp : Interpolator1D
    An Interpolator1D object containing the interpolating track,
    parameterized by log10(time / yr).

Throws
------
RuntimeError
    If m is less than mMin() or greater than mMax().)doc";

static constexpr std::string_view getIsochroneDocstring = R"doc(Return the isochrone at a given log time and [Fe/H].

Parameters
----------
logT : float
    log10(time / yr) of the isochrone.
feh : float
    [Fe/H] value of the tracks to use.

Returns
-------
isochrone : list of Interpolator1D
    A list of Interpolator1D objects, each parameterized by mass,
    giving the isochrone. More than one entry indicates disjoint
    segments, which can occur for non-monotonic tracks; an empty list
    indicates that logT lies outside the tracks' time range, or that
    it touches the tracks at only a single mass (too few points to
    interpolate).)doc";

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
                constructorDocstring.data(),
                py::arg("trackName"),
                py::arg("fehMin"),
                py::arg("fehMax"),
                py::arg("vvcrit") = 0,
                py::arg("afe") = 0,
                py::arg("registryName") = tracks::defaultRegistry
        )
        .def("mMin", &tracks::Tracks3D::mMin,
                mMinDocstring.data())
        .def("mMax", &tracks::Tracks3D::mMax,
                mMaxDocstring.data())
        .def("logTMin", &tracks::Tracks3D::logTMin,
                logTMinDocstring.data())
        .def("logTMax", &tracks::Tracks3D::logTMax,
                logTMaxDocstring.data())
        .def("feH", &tracks::Tracks3D::feH,
                feHDocstring.data())
        .def("aFe", &tracks::Tracks3D::aFe,
                aFeDocstring.data())
        .def("vVcrit", &tracks::Tracks3D::vVcrit,
                vVcritDocstring.data())
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
                getTrackDocstring.data(),
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
                getIsochroneDocstring.data(),
                py::arg("logT"), py::arg("feh"));
}
// NOLINTEND(misc-include-cleaner)
