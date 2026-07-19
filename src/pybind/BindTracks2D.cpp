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
#include <string_view>
#include <utility>

// Numpy-style docstrings for the Python bindings below
static constexpr std::string_view constructorDocstring = R"doc(Construct a Tracks2D object from tracks on disk.

Parameters
----------
trackName : str
    Name of the track set within the registry.
feh : float, optional
    [Fe/H] value of the desired track. Default is 0.0.
vvcrit : float, optional
    Rotation rate v/vcrit of the desired track. Default is 0.0.
afe : float, optional
    [alpha/Fe] value of the desired track. Default is 0.0.
registryName : str, optional
    Path to the track registry file. Default is the package's default
    registry (data/tracks/tracks.toml).

Throws
------
RuntimeError
    If no track in trackName matches feh, vvcrit, and afe exactly.)doc";

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

static constexpr std::string_view feHDocstring = R"doc(Return the [Fe/H] value of this set of tracks.

Returns
-------
feh : float
    [Fe/H] value of this set of tracks.)doc";

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

static constexpr std::string_view starLifetimeDocstring = R"doc(Return the lifetime of a star of a given mass.

Parameters
----------
m : float
    Stellar mass, in Msun.

Returns
-------
logt : float
    log10(time / yr) at which a star of mass m ends this track.)doc";

static constexpr std::string_view liveMassRangeDocstring = R"doc(Return the range(s) of stellar mass alive at a given time.

Parameters
----------
logT : float
    log10(time / yr) at which to evaluate.

Returns
-------
ranges : list of tuple of float
    A list of (mMin, mMax) pairs giving the mass ranges alive at time
    t. Usually contains a single pair, but may contain more than one
    for non-monotonic tracks.)doc";

static constexpr std::string_view getTrackDocstring = R"doc(Return the track for a star of a given mass.

Parameters
----------
m : float
    Stellar mass, in Msun.

Returns
-------
interp : Interpolator1D
    An Interpolator1D object containing the interpolating track,
    parameterized by log10(time / yr).

Throws
------
RuntimeError
    If m is less than mMin() or greater than mMax().)doc";

static constexpr std::string_view getIsochroneDocstring = R"doc(Return the isochrone at a given log time.

Parameters
----------
logT : float
    log10(time / yr) of the isochrone.

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
                constructorDocstring.data(),
                py::arg("trackName"),
                py::arg("feh") = 0,
                py::arg("vvcrit") = 0,
                py::arg("afe") = 0,
                py::arg("registryName") = tracks::defaultRegistry
        )
        .def("mMin", &tracks::Tracks2D::mMin,
                mMinDocstring.data())
        .def("mMax", &tracks::Tracks2D::mMax,
                mMaxDocstring.data())
        .def("logTMin", &tracks::Tracks2D::logTMin,
                logTMinDocstring.data())
        .def("logTMax", &tracks::Tracks2D::logTMax,
                logTMaxDocstring.data())
        .def("feH", &tracks::Tracks2D::feH,
                feHDocstring.data())
        .def("aFe", &tracks::Tracks2D::aFe,
                aFeDocstring.data())
        .def("vVcrit", &tracks::Tracks2D::vVcrit,
                vVcritDocstring.data())
        .def("starLifetime", &tracks::Tracks2D::starLifetime,
                starLifetimeDocstring.data(),
                py::arg("m"))
        .def("liveMassRange", &tracks::Tracks2D::liveMassRange,
                liveMassRangeDocstring.data(),
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
                getTrackDocstring.data(),
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
                getIsochroneDocstring.data(),
                py::arg("logT"));
}
// NOLINTEND(misc-include-cleaner)
