/**
 * @file BindSpecsyn.cpp
 * @author Mark Krumholz
 * @brief Python bindings for the Specsyn class hierarchy
 * @date 2026-07-30
 * @details
 * Binds the abstract Specsyn base and all four concrete subclasses:
 * SpecsynBlackbody, SpecsynLibNoWind, SpecsynLibWR, and
 * SpecsynLibChained.  Only the OOBPolicy::raise specialization of the
 * two template classes is exposed, since that is the most useful policy
 * for interactive Python use (errors are visible rather than silently
 * returning empty spectra).
 */

#include "Bindings.hpp"
#include "../io/SimControls.hpp"
#include "../specsyn/Specsyn.hpp"
#include "../specsyn/SpecsynBlackbody.hpp"
#include "../specsyn/SpecsynCommons.hpp"
#include "../specsyn/SpecsynLibChained.hpp"
#include "../specsyn/SpecsynLibNoWind.hpp"
#include "../specsyn/SpecsynLibWR.hpp"
#include "../tracks/TrackCommons.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // NOLINT(misc-include-cleaner); needed for list/vector conversions
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

// -------------------------------------------------------------------------
// Specsyn (abstract base class)
// -------------------------------------------------------------------------

static constexpr std::string_view specsynWlDocstring = R"doc(Return the rest-frame wavelength grid.

Returns
-------
wl : list of float
    Wavelength grid, in Angstrom, on which spec() is evaluated.)doc";

static constexpr std::string_view specsynWlObsDocstring = R"doc(Return the observed-frame wavelength grid.

Returns
-------
wl_obs : list of float
    Wavelength grid, in Angstrom, redshifted by (1 + z).)doc";

static constexpr std::string_view specsynSpecDocstring = R"doc(Compute the spectrum of a single star.

Parameters
----------
props : list of float
    Stellar properties at this star's mass, as returned by evaluating
    an ``Interpolator1D`` track segment.  Must have exactly 9 elements,
    in the order: mass (Msun), mdot (Msun/yr), logL (Lsun), logTe (K),
    h_surf, he_surf, c_surf, n_surf, o_surf (all surface mass fractions).
feh : float
    [Fe/H] of the star.

Returns
-------
spec : list of float
    Specific luminosity of the star, in erg/s/Angstrom, on the wavelength
    grid returned by wl().

Throws
------
RuntimeError
    If props does not have exactly 9 elements, or if the star's
    parameters fall outside this synthesizer's domain and the
    out-of-bounds policy raises.)doc";

// -------------------------------------------------------------------------
// SpecsynBlackbody
// -------------------------------------------------------------------------

static constexpr std::string_view bbConstructorDocstring = R"doc(Construct a blackbody spectral synthesizer.

All arguments are optional; the wavelength range defaults to a
logarithmically spaced grid from roughly 10 to 10,000 Angstrom (the
range 10*Ry/hc to 0.01*Ry/hc), with 1000 points.

Parameters
----------
wl_min : float, optional
    Minimum wavelength of the output grid, in Angstrom. Ignored when
    nwl is 0 (falls back to the default range). Default is 0.
wl_max : float, optional
    Maximum wavelength of the output grid, in Angstrom. See wl_min.
    Default is 0.
nwl : int, optional
    Number of points in the output grid. 0 (the default) selects the
    built-in 1000-point default grid. If nonzero but wl_min is 0, uses
    nwl points spanning the default wavelength range.
z : float, optional
    Redshift. Default is 0.
controls : SimControls, optional
    Simulation controls; only the integrator tolerance settings
    (``intRelTol``, ``intAbsTol``, ``intMaxIter``) are used. If None
    (the default), uses ``SimControls()`` defaults: rel_tol = 1e-6,
    abs_tol = 0, max_iter = unlimited.)doc";

// -------------------------------------------------------------------------
// SpecsynLibNoWind
// -------------------------------------------------------------------------

static constexpr std::string_view nwConstructorDocstring = R"doc(Construct a no-wind spectral library synthesizer.

Covers libraries like BOSZ and TLUSTY, whose spectra sit on a
(FeH, logg, Teff) tensor grid.

Parameters
----------
spectra_name : str
    Name of the spectral model in the registry (e.g. ``"BOSZ"`` or
    ``"TLUSTY_OB"``).
feh_min : float
    Minimum [Fe/H] value spanned by the population.
feh_max : float
    Maximum [Fe/H] value spanned by the population. May equal feh_min
    for a single-metallicity population.
afe : float, optional
    [alpha/Fe] value. Default is 0.0.
cfe : float, optional
    [C/Fe] value. Default is 0.0.
micro_turb : float, optional
    Microturbulent velocity in km/s. NaN (the default) uses each
    library's own ``micro_default`` from the registry.
r : float, optional
    Spectral resolution. Default is 500.
registry_name : str, optional
    Path to the spectral library registry file. Default is the
    package's default registry (``data/spectra/spectra.toml``).
wl_min : float, optional
    Minimum output wavelength, in Angstrom. 0 (the default) falls
    back to the library's own native grid.
wl_max : float, optional
    Maximum output wavelength, in Angstrom. See wl_min.
nwl : int, optional
    Number of output wavelength points. 0 (the default) uses the
    library's own native wavelength grid.
z : float, optional
    Redshift. Default is 0.
controls : SimControls, optional
    Simulation controls; only integrator tolerance settings are used.
    None (the default) uses ``SimControls()`` defaults.

Throws
------
RuntimeError
    If spectra_name is not found in the registry, or the library
    file cannot be read.)doc";

// -------------------------------------------------------------------------
// SpecsynLibWR
// -------------------------------------------------------------------------

static constexpr std::string_view wrConstructorDocstring = R"doc(Construct a Wolf-Rayet spectral library synthesizer.

Covers the Potsdam Wolf-Rayet (PoWR) model grids: spectra sit on a
(FeH, log_Rt, log_Teff) tensor grid, where log_Rt is the
log-transformed radius.

Parameters
----------
spectra_name : str
    Name of the spectral model in the registry. Must identify a WR
    library; currently ``"POWR_WNE"``, ``"POWR_WNL_H20"``,
    ``"POWR_WNL_H40"``, ``"POWR_WNL_H60"``, or ``"POWR_WC"``.
feh_min : float
    Minimum [Fe/H] value spanned by the population.
feh_max : float
    Maximum [Fe/H] value spanned by the population.
registry_name : str, optional
    Path to the spectral library registry file. Default is the
    package's default registry (``data/spectra/spectra.toml``).
wl_min : float, optional
    Minimum output wavelength, in Angstrom. 0 (the default) falls
    back to the library's own native wavelength coverage.
wl_max : float, optional
    Maximum output wavelength, in Angstrom. See wl_min.
nwl : int, optional
    Number of output wavelength points. 0 (the default) uses the
    library's own common wavelength grid.
z : float, optional
    Redshift. Default is 0.
controls : SimControls, optional
    Simulation controls; only integrator tolerance settings are used.
    None (the default) uses ``SimControls()`` defaults.

Throws
------
RuntimeError
    If spectra_name does not identify a recognized WR subtype, is not
    found in the registry, or the library file cannot be read.)doc";

// -------------------------------------------------------------------------
// SpecsynLibChained
// -------------------------------------------------------------------------

static constexpr std::string_view chainedConstructorDocstring = R"doc(Construct a chained spectral synthesizer from several libraries.

Each star is assigned a spectrum by trying the chained libraries in
priority order; the first library able to produce a spectrum for that
star is used.  This allows combining complementary libraries that cover
different regions of stellar parameter space (e.g. TLUSTY for hot OB
stars and BOSZ for cooler ones).

Parameters
----------
spectra_names : list of str
    Names of the spectral models to chain, in priority order.
feh_min : float
    Minimum [Fe/H] value spanned by the population.
feh_max : float
    Maximum [Fe/H] value spanned by the population.
afe : float, optional
    [alpha/Fe] value. Default is 0.0.
cfe : float, optional
    [C/Fe] value. Default is 0.0.
micro_turb : list of float, optional
    Microturbulent velocity in km/s, one value per library. An empty
    list (the default) uses each library's own ``micro_default`` from
    the registry.
r : float, optional
    Spectral resolution. Default is 500.
registry_name : str, optional
    Path to the spectral library registry file. Default is the
    package's default registry.
wl_min : float, optional
    Minimum output wavelength, in Angstrom. 0 (the default) falls
    back to each library's own native wavelength coverage.
wl_max : float, optional
    Maximum output wavelength, in Angstrom. See wl_min.
nwl : int, optional
    Number of output wavelength points. 0 (the default) uses a common
    wavelength grid derived from all chained libraries.
z : float, optional
    Redshift. Default is 0.
t_clamp : bool, optional
    If True (the default), clamp each star's log(Teff) to the range
    spanned by the chained libraries before calling spec(), so a star
    with a track-derived Teff outside any library's coverage is handled
    by the nearest library grid point rather than raising an error.
controls : SimControls, optional
    Simulation controls; only integrator tolerance settings are used.
    None (the default) uses ``SimControls()`` defaults.

Throws
------
RuntimeError
    If spectra_names is empty, micro_turb is non-empty and does not
    match spectra_names in length, or any library cannot be loaded.)doc";

// -------------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------------

// Convert props list to StarData, checking length
static auto toStarData(const std::vector<double>& props) -> specsyn::Specsyn::StarData
{
    constexpr std::size_t nQty = static_cast<std::size_t>(tracks::FieldIdx::nTrackQty);
    if (props.size() != nQty)
    {
        throw std::runtime_error(
            "Specsyn.spec: props must have exactly " + std::to_string(nQty) +
            " elements (mass, mdot, logL, logTe, h_surf, he_surf, c_surf, n_surf, o_surf)");
    }
    specsyn::Specsyn::StarData starData{};
    std::copy(props.begin(), props.end(), starData.begin());
    return starData;
}

// Return a const SimControls& from a py::object that is either None
// (giving the default-constructed SimControls) or an actual SimControls.
// The defaultControls local must outlive the returned reference; callers
// must keep it alive for the duration of the call.
static auto resolveControls(const py::object& controls,
    const io::SimControls& defaultControls) -> const io::SimControls&
{
    if (controls.is_none()) { return defaultControls; }
    return py::cast<const io::SimControls&>(controls);
}

// Disable linting for includes -- the pybind macro magic seems to confuse
// the linter
// NOLINTBEGIN(misc-include-cleaner)
void bindSpecsyn(py::module_& m)
{
    // Abstract base class: no constructor, just wl/wlObs/spec
    py::class_<specsyn::Specsyn, py::smart_holder>(m, "Specsyn")
        .def("wl",
                [](const specsyn::Specsyn& self) -> std::vector<double>
                { return self.wl(); },
                specsynWlDocstring.data())
        .def("wlObs",
                [](const specsyn::Specsyn& self) -> std::vector<double>
                { return self.wlObs(); },
                specsynWlObsDocstring.data())
        .def("spec",
                [](const specsyn::Specsyn& self,
                   const std::vector<double>& props, double feh)
                    -> std::vector<double>
                { return self.spec(toStarData(props), feh); },
                specsynSpecDocstring.data(),
                py::arg("props"), py::arg("feh"));

    // SpecsynBlackbody
    py::class_<specsyn::SpecsynBlackbody, specsyn::Specsyn, py::smart_holder>(
            m, "SpecsynBlackbody")
        .def(py::init(
                [](double wlMin, double wlMax, std::size_t nWl, double z,
                   const py::object& controls)
                    -> std::unique_ptr<specsyn::SpecsynBlackbody>
                {
                    const io::SimControls defaultControls{};
                    return std::make_unique<specsyn::SpecsynBlackbody>(
                        wlMin, wlMax, nWl, z, resolveControls(controls, defaultControls));
                }),
                bbConstructorDocstring.data(),
                py::arg("wl_min") = 0.0,
                py::arg("wl_max") = 0.0,
                py::arg("nwl") = static_cast<std::size_t>(0),
                py::arg("z") = 0.0,
                py::arg("controls") = py::none());

    // SpecsynLibNoWind<OOBPolicy::raise>
    py::class_<specsyn::SpecsynLibNoWind<specsyn::OOBPolicy::raise>,
               specsyn::Specsyn, py::smart_holder>(m, "SpecsynLibNoWind")
        .def(py::init(
                [](const std::string& spectraName,
                   double fehMin, double fehMax,
                   double afe, double cfe, double microTurb, double r,
                   const std::string& registryName,
                   double wlMin, double wlMax, std::size_t nWl, double z,
                   const py::object& controls)
                    -> std::unique_ptr<specsyn::SpecsynLibNoWind<specsyn::OOBPolicy::raise>>
                {
                    const io::SimControls defaultControls{};
                    return std::make_unique<
                        specsyn::SpecsynLibNoWind<specsyn::OOBPolicy::raise>>(
                            spectraName, fehMin, fehMax,
                            afe, cfe, microTurb, r,
                            registryName, wlMin, wlMax, nWl, z,
                            resolveControls(controls, defaultControls));
                }),
                nwConstructorDocstring.data(),
                py::arg("spectra_name"),
                py::arg("feh_min"),
                py::arg("feh_max"),
                py::arg("afe") = tracks::defaultAFe,
                py::arg("cfe") = specsyn::defaultCFe,
                py::arg("micro_turb") = std::numeric_limits<double>::quiet_NaN(),
                py::arg("r") = specsyn::defaultR,
                py::arg("registry_name") = specsyn::defaultRegistry,
                py::arg("wl_min") = 0.0,
                py::arg("wl_max") = 0.0,
                py::arg("nwl") = static_cast<std::size_t>(0),
                py::arg("z") = 0.0,
                py::arg("controls") = py::none());

    // SpecsynLibWR<OOBPolicy::raise>
    py::class_<specsyn::SpecsynLibWR<specsyn::OOBPolicy::raise>,
               specsyn::Specsyn, py::smart_holder>(m, "SpecsynLibWR")
        .def(py::init(
                [](const std::string& spectraName,
                   double fehMin, double fehMax,
                   const std::string& registryName,
                   double wlMin, double wlMax, std::size_t nWl, double z,
                   const py::object& controls)
                    -> std::unique_ptr<specsyn::SpecsynLibWR<specsyn::OOBPolicy::raise>>
                {
                    const io::SimControls defaultControls{};
                    return std::make_unique<
                        specsyn::SpecsynLibWR<specsyn::OOBPolicy::raise>>(
                            spectraName, fehMin, fehMax,
                            registryName, wlMin, wlMax, nWl, z,
                            resolveControls(controls, defaultControls));
                }),
                wrConstructorDocstring.data(),
                py::arg("spectra_name"),
                py::arg("feh_min"),
                py::arg("feh_max"),
                py::arg("registry_name") = specsyn::defaultRegistry,
                py::arg("wl_min") = 0.0,
                py::arg("wl_max") = 0.0,
                py::arg("nwl") = static_cast<std::size_t>(0),
                py::arg("z") = 0.0,
                py::arg("controls") = py::none());

    // SpecsynLibChained
    py::class_<specsyn::SpecsynLibChained, specsyn::Specsyn, py::smart_holder>(
            m, "SpecsynLibChained")
        .def(py::init(
                [](const std::vector<std::string>& spectraNames,
                   double fehMin, double fehMax,
                   double afe, double cfe,
                   const std::vector<double>& microTurb,
                   double r, const std::string& registryName,
                   double wlMin, double wlMax, std::size_t nWl, double z,
                   bool tClamp, const py::object& controls)
                    -> std::unique_ptr<specsyn::SpecsynLibChained>
                {
                    const io::SimControls defaultControls{};
                    return std::make_unique<specsyn::SpecsynLibChained>(
                        spectraNames, fehMin, fehMax,
                        afe, cfe, microTurb, r,
                        registryName, wlMin, wlMax, nWl, z, tClamp,
                        resolveControls(controls, defaultControls));
                }),
                chainedConstructorDocstring.data(),
                py::arg("spectra_names"),
                py::arg("feh_min"),
                py::arg("feh_max"),
                py::arg("afe") = tracks::defaultAFe,
                py::arg("cfe") = specsyn::defaultCFe,
                py::arg("micro_turb") = std::vector<double>{},
                py::arg("r") = specsyn::defaultR,
                py::arg("registry_name") = specsyn::defaultRegistry,
                py::arg("wl_min") = 0.0,
                py::arg("wl_max") = 0.0,
                py::arg("nwl") = static_cast<std::size_t>(0),
                py::arg("z") = 0.0,
                py::arg("t_clamp") = true,
                py::arg("controls") = py::none());
}
// NOLINTEND(misc-include-cleaner)
