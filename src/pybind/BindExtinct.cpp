/**
 * @file BindExtinct.cpp
 * @author Mark Krumholz
 * @brief Python bindings for extinct::Extinct
 * @date 2026-08-31
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#include "Bindings.hpp"
#include "../extinct/Extinct.hpp"
#include "../io/SimControls.hpp"
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // NOLINT(misc-include-cleaner); needed for list/vector conversions
#include <string>
#include <string_view>

static constexpr std::string_view classDocstring = R"doc(A dust extinction curve, interpolated onto its SimControls's own spectral-synthesis wavelength grid.

Built from a named curve in an extinction curve registry. The curve is
normalized so that a V-band extinction of A_V = 1 mag corresponds to
its own native shape; only the shape matters, since a caller scales it
by whatever A_V is actually applied.)doc";

static constexpr std::string_view constructorDocstring = R"doc(Construct an Extinct from a named registry entry.

Parameters
----------
extinct_name : str
    Name of the extinction curve to load (e.g. "Calzetti_starburst").
controls : SimControls, optional
    Simulation controls this Extinct reads its wavelength grid
    (controls.specsyn.wl, which controls.specsyn must therefore
    provide), redshift (see wlObs()), and, if a nebular emission grid
    was requested (SimControls.nebular), its field-star A_V
    distribution (SimControls.avDistField) from, live, for the rest of
    its lifetime. If None (the default), uses slug's own shared,
    bundled-default SimControls (built from
    src/pybind/assets/PyDefaults.toml the first time it is needed, and
    reused after that -- see SimControls()'s own default path), rather
    than a minimal, all-C++-defaults one: unlike most other classes'
    own optional controls argument, Extinct always needs a real
    spectral synthesizer to provide a wavelength grid, which a minimal
    SimControls does not have.
registry_name : str, optional
    Path to the extinction curve registry file. Default is the
    package's default registry (data/extinct/extinct.toml).

Throws
------
RuntimeError
    If extinct_name is not found in the registry, the registry/HDF5
    file cannot be read, or controls.specsyn is None.)doc";

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
    This Extinct's own SimControls.specsyn.wl, clipped to wlDat()'s
    own [min, max] coverage.)doc";

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

static constexpr std::string_view wlOffsetDocstring = R"doc(Get the number of leading elements chopped off this Extinct's own SimControls.specsyn.wl.

Returns
-------
wl_offset : int
    The number of leading elements of this Extinct's own
    SimControls.specsyn.wl that fell below the native curve's own
    coverage and so are absent from wl()/extinct(). Lets a caller line
    up a spectrum tabulated on that same wavelength grid with wl()'s
    own, narrower grid, exactly as applyExtinction() does internally.)doc";

static constexpr std::string_view applyExtinctionDocstring = R"doc(Apply this extinction curve to a spectrum.

Parameters
----------
A_V : float
    V-band extinction to apply, in magnitudes.
spec : list of float
    Spectrum to extinguish, tabulated on exactly this Extinct's own
    SimControls.specsyn.wl.

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
(SimControls.avDistField).

Parameters
----------
spec : list of float
    Spectrum to extinguish, tabulated on exactly this Extinct's own
    SimControls.specsyn.wl -- see applyExtinction()'s own spec
    parameter.

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
distribution (SimControls.avDistField).

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
    py::class_<extinct::Extinct, py::smart_holder>(m, "Extinct", classDocstring.data())
        .def(py::init(
                [](const std::string& extinctName,
                   const py::object& controls, const std::string& registryName)
                    -> std::unique_ptr<extinct::Extinct>
                {
                    // Deliberately a ternary, not resolveControls(controls,
                    // sharedDefaultControls()): sharedDefaultControls() is
                    // expensive (parses PyDefaults.toml, the real MIST
                    // tracks, and the full spectral-library chain) and can
                    // throw, so it must only run when controls is actually
                    // py::none() -- a plain function-call argument would
                    // evaluate it eagerly on every construction regardless
                    // -- see BindCluster.cpp's own identical comment.
                    // sharedMinimalControls() (used by every other
                    // Specsyn-derived class's own default controls
                    // argument) won't do here: it has no spectral
                    // synthesizer, and Extinct now always needs one, to
                    // provide the wavelength grid the curve is
                    // interpolated onto.
                    const io::SimControls& controlsRef = controls.is_none()
                        ? sharedDefaultControls()
                        : py::cast<const io::SimControls&>(controls);
                    return std::make_unique<extinct::Extinct>(
                        extinctName, controlsRef, registryName);
                }),
                constructorDocstring.data(),
                py::arg("extinct_name"),
                py::arg("controls") = py::none(),
                py::arg("registry_name") = extinct::defaultRegistry,
                // Keep controls (index 3: 1 = self, 2 = extinct_name)
                // alive at least as long as this Extinct, which stores
                // a live reference to it rather than copying its
                // wavelength grid/redshift/nebular grid/avDistField
                // out -- see Extinct's own controls_ member. A
                // harmless no-op when controls is omitted:
                // sharedDefaultControls()'s own instance is a
                // function-local static (see its own comment in
                // Bindings.hpp).
                py::keep_alive<1, 3>())
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
