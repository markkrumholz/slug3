/**
 * @file BindExtinct.cpp
 * @author Mark Krumholz
 * @brief Python bindings for extinct::Extinct
 * @date 2026-08-31
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#include "Bindings.hpp"
#include "../extinct/Extinct.hpp"
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // NOLINT(misc-include-cleaner); needed for list/vector conversions
#include <string>
#include <string_view>
#include <vector>

static constexpr std::string_view constructorDocstring = R"doc(Construct an Extinct from a named registry entry.

Parameters
----------
extinct_name : str
    Name of the extinction curve to load (e.g. "Calzetti_starburst").
wl : list of float
    Wavelength grid, in Angstrom, to interpolate the curve onto.
    Clipped to the native curve's own [min, max] wavelength coverage
    before interpolating -- see wl()'s own docstring.
controls : SimControls, optional
    Simulation controls this Extinct reads its redshift (see
    wlObs()) and, if a nebular emission grid was requested
    (SimControls.nebular), its field-star A_V distribution
    (SimControls.avDistField) from, live, for the rest of its
    lifetime. If None (the default), uses a minimal, all-C++-defaults
    SimControls -- z = 0, no nebular emission grid, an invalid
    avDistField() (treated as a delta at A_V = 0) -- rather than
    slug's bundled physics deck.
registry_name : str, optional
    Path to the extinction curve registry file. Default is the
    package's default registry (data/extinct/extinct.toml).

Throws
------
RuntimeError
    If extinct_name is not found in the registry, or the registry/
    HDF5 file cannot be read.)doc";

static constexpr std::string_view wlDatDocstring = R"doc(Get the native extinction curve wavelength grid.

Returns
-------
wl_dat : list of float
    Wavelength grid, in Angstrom, as read directly from the registry
    entry.)doc";

static constexpr std::string_view extinctDatDocstring = R"doc(Get the native extinction curve.

Returns
-------
extinct_dat : list of float
    Extinction curve, in arbitrary units, at each wavelength in
    wlDat().)doc";

static constexpr std::string_view wlDocstring = R"doc(Get the interpolated wavelength grid.

Returns
-------
wl : list of float
    Wavelength grid, in Angstrom, supplied to the constructor and
    clipped to wlDat()'s own [min, max] coverage.)doc";

static constexpr std::string_view extinctDocstring = R"doc(Get the interpolated extinction curve.

Returns
-------
extinct : list of float
    Extinction curve, in arbitrary units, interpolated onto wl() and
    normalized to a V-band extinction of 1 mag.)doc";

static constexpr std::string_view wlObsDocstring = R"doc(Get the observed-frame interpolated wavelength grid.

Returns
-------
wl_obs : list of float
    wl(), redshifted by (1 + z), with z read live from the
    SimControls this Extinct was built from.)doc";

static constexpr std::string_view wlOffsetDocstring = R"doc(Get the number of leading elements chopped off the constructor's own wl.

Returns
-------
wl_offset : int
    The number of leading elements of the wl passed to the
    constructor that fell below the native curve's own coverage and
    so are absent from wl()/extinct(). Lets a caller line up a
    spectrum tabulated on that original wl with wl()'s own, narrower
    grid, exactly as applyExtinction() does internally.)doc";

static constexpr std::string_view applyExtinctionDocstring = R"doc(Apply this extinction curve to a spectrum.

Parameters
----------
A_V : float
    V-band extinction to apply, in magnitudes.
spec : list of float
    Spectrum to extinguish, tabulated on exactly the same wavelength
    grid as the wl originally passed to the constructor.

Returns
-------
spec_ext : list of float
    The extinguished spectrum, on the wavelength grid returned by
    wl(). spec's first wlOffset() elements (falling outside this
    curve's own wavelength coverage) are discarded; each remaining
    element is multiplied by exp(-A_V * extinct()) at the
    corresponding wavelength.)doc";

static constexpr std::string_view applyExtinctionCtsDocstring =
    R"doc(Apply this extinction curve's own expected attenuation to a continuously-distributed population's spectrum.

Unlike applyExtinction(), which attenuates a single star (or cluster)
with one known A_V, this is for a population whose members are not
individually tracked, so there is no single A_V to apply -- instead,
each element of spec is multiplied by the expectation value of
exp(-A_V * extinct()) over the field-star A_V distribution
(SimControls.avDistField), precomputed once at construction.

Parameters
----------
spec : list of float
    Spectrum to extinguish, tabulated on exactly the same wavelength
    grid as the wl originally passed to the constructor -- see
    applyExtinction()'s own spec parameter.

Returns
-------
spec_ext : list of float
    The expected extinguished spectrum, on the wavelength grid
    returned by wl().)doc";

static constexpr std::string_view applyExtinctionLinesDocstring =
    R"doc(Apply this extinction curve to a set of nebular emission line luminosities.

Parameters
----------
A_V : float
    V-band extinction to apply, in magnitudes.
line_lum : list of float
    Luminosity of each of this Extinct's own SimControls's nebular
    emission grid's lines, in erg/s. Pass an empty list (getting back
    an empty result) if that SimControls has no nebular emission grid
    (its nebular property is None).

Returns
-------
line_lum_ext : list of float
    The extinguished line luminosities, in the same order as
    line_lum. Unlike applyExtinction(), no elements are dropped: a
    line outside the native curve's own wavelength coverage reads an
    extinction of 0 (no attenuation) rather than being excluded.)doc";

static constexpr std::string_view applyExtinctionCtsLinesDocstring =
    R"doc(Apply this extinction curve's own expected attenuation to a continuously-distributed population's line luminosities.

Line-luminosity analog of applyExtinctionCts() -- see its own
docstring; each element of line_lum is multiplied by the expectation
value of exp(-A_V * extinctLines) over the field-star A_V
distribution (SimControls.avDistField), precomputed once at
construction.

Parameters
----------
line_lum : list of float
    Luminosity of each line, in erg/s -- see applyExtinctionLines()'s
    own line_lum parameter.

Returns
-------
line_lum_ext : list of float
    The expected extinguished line luminosities, in the same order as
    line_lum.)doc";

// Disable linting for includes -- the pybind macro magic seems to confuse
// the linter
// NOLINTBEGIN(misc-include-cleaner)
void bindExtinct(py::module_& m)
{
    py::class_<extinct::Extinct, py::smart_holder>(m, "Extinct")
        .def(py::init(
                [](const std::string& extinctName, const std::vector<double>& wl,
                   const py::object& controls, const std::string& registryName)
                    -> std::unique_ptr<extinct::Extinct>
                {
                    return std::make_unique<extinct::Extinct>(
                        extinctName, wl,
                        resolveControls(controls, sharedMinimalControls()),
                        registryName);
                }),
                constructorDocstring.data(),
                py::arg("extinct_name"),
                py::arg("wl"),
                py::arg("controls") = py::none(),
                py::arg("registry_name") = extinct::defaultRegistry,
                // Keep controls (index 4: 1 = self, 2 = extinct_name,
                // 3 = wl) alive at least as long as this Extinct,
                // which stores a live reference to it rather than
                // copying its redshift/nebular grid/avDistField out --
                // see Extinct's own controls_ member. A harmless no-op
                // when controls is omitted: sharedMinimalControls()'s
                // own instance is a function-local static (see its
                // own comment in Bindings.hpp).
                py::keep_alive<1, 4>())
        .def("wlDat", &extinct::Extinct::wlDat, wlDatDocstring.data())
        .def("extinctDat", &extinct::Extinct::extinctDat, extinctDatDocstring.data())
        .def("wl", &extinct::Extinct::wl, wlDocstring.data())
        .def("extinct", &extinct::Extinct::extinct, extinctDocstring.data())
        .def("wlObs", &extinct::Extinct::wlObs, wlObsDocstring.data())
        .def("wlOffset", &extinct::Extinct::wlOffset, wlOffsetDocstring.data())
        .def("applyExtinction", &extinct::Extinct::applyExtinction,
                applyExtinctionDocstring.data(),
                py::arg("A_V"), py::arg("spec"))
        .def("applyExtinctionCts", &extinct::Extinct::applyExtinctionCts,
                applyExtinctionCtsDocstring.data(),
                py::arg("spec"))
        .def("applyExtinctionLines", &extinct::Extinct::applyExtinctionLines,
                applyExtinctionLinesDocstring.data(),
                py::arg("A_V"), py::arg("line_lum"))
        .def("applyExtinctionCtsLines", &extinct::Extinct::applyExtinctionCtsLines,
                applyExtinctionCtsLinesDocstring.data(),
                py::arg("line_lum"));
}
// NOLINTEND(misc-include-cleaner)
