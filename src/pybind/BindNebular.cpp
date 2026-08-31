/**
 * @file BindNebular.cpp
 * @author Mark Krumholz
 * @brief Python bindings for nebular::Nebular
 * @date 2026-08-31
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#include "Bindings.hpp"
#include "../nebular/Nebular.hpp"
#include "../tracks/TrackCommons.hpp"
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // NOLINT(misc-include-cleaner); needed for list/tuple/vector conversions
#include <string>
#include <string_view>

static constexpr std::string_view constructorDocstring = R"doc(Construct a Nebular object, loading one track set's own cloudy grid.

Parameters
----------
table_name : str
    Path to the nebular emission table to load this track set's own
    grid from (e.g. "data/nebular/nebular.h5").
track_name : str
    Name of the track set this nebular grid is for; used only during
    construction (to locate this track set's own cloudy grid data
    inside table_name), not retained afterward.
controls : SimControls
    Simulation controls (physics and control-flow settings); also the
    source of the simulation wavelength grid the continuum luminosity
    grids are resampled onto (controls.specsyn().wl()), so that grid
    need not be passed in separately. controls.specsyn() must
    therefore be set (non-None) before constructing a Nebular from it
    -- unlike Extinct's own controls argument, there is no optional/
    default fallback here (this mirrors the C++ constructor, which
    likewise gives table_name/track_name/controls no default of their
    own). Stored by reference: this SimControls must outlive the
    returned Nebular.
vvcrit : float, optional
    Rotation rate v/vcrit of the track set. Default is 0.0.

Throws
------
RuntimeError
    If table_name cannot be found, track_name has no group of its own
    in it, or -- for any of that group's own [Fe/H] values -- no
    v/vcrit subgroup matches vvcrit exactly, or controls's own
    nebular log(U) control falls outside the range of log(U) values
    actually tabulated for a given [Fe/H]/v/vcrit combination.)doc";

static constexpr std::string_view lineLabelDocstring = R"doc(Get the label of each of this Nebular's own nebular emission lines.

Returns
-------
line_label : list of str
    One label per line, in the same order as lineWl().)doc";

static constexpr std::string_view lineWlDocstring = R"doc(Get the wavelength of each of this Nebular's own nebular emission lines.

Returns
-------
line_wl : list of float
    Wavelength, in Angstrom, of each line, in the table's own line
    ordering (the same order as lineLabel()).)doc";

static constexpr std::string_view getGalaxyDocstring = R"doc(Add whole-galaxy nebular emission to an input stellar spectrum.

Q(HI) (the H-ionizing photon rate) is read off spec, then discounted
by controls's own nebular coverage fraction (the fraction of ionizing
photons actually absorbed by the nebula, rather than lost to dust
grains or escaping the observational aperture); every added quantity
scales with that discounted rate. The continuum is added everywhere;
the input spectrum itself is kept only above the H-ionization edge
(shorter wavelengths are zeroed, since a real HII region reprocesses
essentially all ionizing stellar flux); lines are deposited around
their own central wavelength.

Parameters
----------
spec : list of float
    Input stellar spectrum, on this object's own wavelength grid
    (i.e. the controls.specsyn().wl() this Nebular was built from), in
    erg/s/Angstrom.
feh : float
    [Fe/H] of the stellar population spec was computed for.

Returns
-------
spec_neb : list of float
    The stellar + nebular continuum + nebular line spectrum, on the
    same wavelength grid as spec, in erg/s/Angstrom.
line_lum : list of float
    Luminosity of each of lineWl()'s own lines, in erg/s.

Throws
------
RuntimeError
    If feh falls outside this Nebular's own tabulated [Fe/H] range.)doc";

static constexpr std::string_view getClusterDocstring = R"doc(Add one cluster's own nebular emission to an input stellar spectrum.

Same as getGalaxy(), but interpolated in cluster age as well as
[Fe/H] -- except that, for age above this table's own tabulated age
range, the returned line luminosities are all zero and the returned
spectrum is just spec with every bin at or below the H-ionization edge
zeroed (the same edge-zeroing getGalaxy() applies, with no continuum
or line emission added).

Parameters
----------
spec : list of float
    Input stellar spectrum, on this object's own wavelength grid, in
    erg/s/Angstrom.
feh : float
    [Fe/H] of the cluster spec was computed for.
age : float
    Cluster age, in yr.

Returns
-------
spec_neb : list of float
    The stellar + nebular continuum + nebular line spectrum, on the
    same wavelength grid as spec, in erg/s/Angstrom.
line_lum : list of float
    Luminosity of each of lineWl()'s own lines, in erg/s.

Throws
------
RuntimeError
    If feh falls outside this Nebular's own tabulated [Fe/H] range.
    (age has no such failure mode: below the table's own minimum age
    is pinned to that minimum, and above its own maximum takes the
    early-return path described above.))doc";

// Disable linting for includes -- the pybind macro magic seems to confuse
// the linter
// NOLINTBEGIN(misc-include-cleaner)
void bindNebular(py::module_& m)
{
    py::class_<nebular::Nebular, py::smart_holder>(m, "Nebular")
        .def(py::init(
                [](const std::string& tableName, const std::string& trackName,
                   const io::SimControls& controls, double vvcrit)
                    -> std::unique_ptr<nebular::Nebular>
                {
                    return std::make_unique<nebular::Nebular>(
                        tableName, trackName, controls, vvcrit);
                }),
                constructorDocstring.data(),
                py::arg("table_name"),
                py::arg("track_name"),
                py::arg("controls"),
                py::arg("vvcrit") = tracks::defaultVVcrit,
                // Keep controls (index 4: 1 = self, 2 = table_name,
                // 3 = track_name) alive at least as long as this
                // Nebular, which stores a live reference to it rather
                // than copying its specsyn()/nebControls() out -- see
                // Nebular's own simControls_ member.
                py::keep_alive<1, 4>())
        .def("lineLabel", &nebular::Nebular::lineLabel, lineLabelDocstring.data())
        .def("lineWl", &nebular::Nebular::lineWl, lineWlDocstring.data())
        .def("getGalaxy", &nebular::Nebular::getGalaxy,
                getGalaxyDocstring.data(),
                py::arg("spec"), py::arg("feh"))
        .def("getCluster", &nebular::Nebular::getCluster,
                getClusterDocstring.data(),
                py::arg("spec"), py::arg("feh"), py::arg("age"));
}
// NOLINTEND(misc-include-cleaner)
