/**
 * @file BindSimPhysics.cpp
 * @author Mark Krumholz
 * @brief Python bindings for io::SimPhysics
 * @date 2026-07-21
 */

#include "Bindings.hpp"
#include "../io/SimControls.hpp"
#include "../io/SimPhysics.hpp"
#include "../specsyn/Specsyn.hpp"
#include "../tracks/Tracks3D.hpp"
#include "../utils/MiscUtils.hpp"
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // NOLINT(misc-include-cleaner); this is needed for correct Python binding, even if clang-tidy can't recognize it
#include <stdexcept>
#include <string>
#include <string_view>
#include <toml.hpp>
#include <vector>

// Map the Python-facing sim_type string to SimControls::SimType,
// mirroring the sim_type key SimControls itself reads from a deck
// (see SimControls.cpp) -- SimControls is not otherwise exposed to
// Python, so this is done by hand here instead of constructing one
static auto simTypeFromString(const std::string& simType) -> io::SimControls::SimType
{
    if (simType == "cluster") { return io::SimControls::SimType::cluster; }
    if (simType == "galaxy") { return io::SimControls::SimType::galaxy; }
    throw std::runtime_error("SimPhysics: sim_type must be 'cluster' or 'galaxy'");
}

// Declared in Bindings.hpp (shared with BindCluster.cpp -- see its own
// comment there). The bundled deck specifies no output-time
// information, so controls is built from SimControls's own
// all-defaults constructor instead of parsing it out of the deck
// (which would throw, since SimControls otherwise requires one of the
// output-time options to be set).
auto buildDefaultSimPhysics() -> std::unique_ptr<io::SimPhysics>
{
    const auto defaultsPath = utils::getFilePath("PyDefaults.toml", "src/pybind/assets");
    if (defaultsPath.empty())
    {
        throw std::runtime_error(
            "SimPhysics: default input deck PyDefaults.toml not found");
    }
    const toml::table inputDeck = toml::parse_file(defaultsPath.string());
    const io::SimControls controls;
    return std::make_unique<io::SimPhysics>(inputDeck, controls);
}

// Declared in Bindings.hpp (shared with BindCluster.cpp -- see its own
// comment there)
auto sharedDefaultSimPhysics() -> const io::SimPhysics&
{
    static const std::unique_ptr<io::SimPhysics> instance = buildDefaultSimPhysics();
    return *instance;
}

// Numpy-style docstring for the Python constructor binding. This is
// a convenience constructor, not a direct binding of SimPhysics's
// real (toml::table, SimControls::SimType) constructor -- toml++ is
// a vendored third-party library with no Python bindings of its own,
// so exposing it was not worth the surface area for what is, from
// Python, just a file to load.
static constexpr std::string_view constructorDocstring = R"doc(Construct a SimPhysics object by parsing a slug input deck.

Parameters
----------
path : str, optional
    Path to a slug TOML input deck. If omitted (or empty), loads
    slug's own bundled default deck (src/pybind/assets/PyDefaults.toml)
    instead, letting a caller build a usable SimPhysics for interactive
    work -- e.g. sp = slug.SimPhysics() -- without touching an input
    deck at all. That bundled deck specifies no output-time
    information (it is not meant to drive a full simulation), so in
    this case SimPhysics is built using an interactive-use SimControls
    (all defaults, see SimControls's own default constructor) rather
    than one parsed from the deck.
sim_type : str, optional
    Either "cluster" or "galaxy"; controls whether galaxy-specific
    quantities (the CLF and SFR) are read from the deck. Defaults to
    "cluster", matching the bundled default deck's own sim_type.

Throws
------
RuntimeError
    If the file cannot be parsed, sim_type is not "cluster" or
    "galaxy", the deck is otherwise invalid, or (only if path is
    empty) the bundled default deck cannot be found.)doc";

static constexpr std::string_view wlDocstring = R"doc(Return the rest-frame wavelength grid of the spectral synthesizer.

Returns
-------
wl : list of float
    The wavelength grid, in Angstrom.

Throws
------
RuntimeError
    If no spectral synthesizer was requested (spectra.model was not
    set in the input deck).)doc";

static constexpr std::string_view wlObsDocstring = R"doc(Return the observed-frame wavelength grid of the spectral synthesizer.

Returns
-------
wl_obs : list of float
    The wavelength grid, in Angstrom, redshifted by (1 + z). This is
    the wavelength grid that a Cluster's spec() is evaluated on.

Throws
------
RuntimeError
    If no spectral synthesizer was requested (spectra.model was not
    set in the input deck).)doc";

// Both wl() and wlObs() below throw this if no spectral synthesizer
// was requested, rather than letting a null specsyn() dereference
// crash
static void checkSpecsyn(const io::SimPhysics& self)
{
    if (self.specsyn() == nullptr)
    {
        throw std::runtime_error(
            "SimPhysics: no spectral synthesizer was requested "
            "(spectra.model was not set in the input deck)");
    }
}

static constexpr std::string_view setComputeLbolDocstring = R"doc(Set whether the bolometric luminosity should be computed as an output.

Lets a caller request or suppress Lbol output on an already-constructed
SimPhysics -- e.g. from Python, where there is no phot.filters input-
deck entry to set "Lbol" through -- so any Cluster built from this
SimPhysics picks up the new setting the next time it advances.

Parameters
----------
value : bool
    New value for computeLbol().)doc";

static constexpr std::string_view setIMFDocstring = R"doc(Set the initial mass function.

Parameters
----------
imf : str
    A numerical value (interpreted as a delta-function IMF at that
    mass) or the name of an IMF PDF file, resolved relative to
    SLUG_DIR/REPO_DIR under data/imfs -- the same way stars.IMF is
    resolved when parsing an input deck.

Throws
------
RuntimeError
    If imf is not numeric and does not name a file that can be found.)doc";

static constexpr std::string_view setCMFDocstring = R"doc(Set the cluster mass function.

Parameters
----------
cmf : str
    A numerical value (interpreted as a delta-function CMF at that
    mass) or the name of a CMF PDF file.

Throws
------
RuntimeError
    If cmf is not numeric and does not name a file that can be found.)doc";

static constexpr std::string_view setFeHDocstring = R"doc(Set the [Fe/H] distribution.

Parameters
----------
feh : str
    A numerical value (interpreted as a fixed [Fe/H]) or the name of a
    [Fe/H] PDF file.

Throws
------
RuntimeError
    If feh is not numeric and does not name a file that can be found.)doc";

static constexpr std::string_view setCLFDocstring = R"doc(Set the cluster lifetime function.

Parameters
----------
clf : str
    A numerical value (interpreted as a delta-function CLF at that
    lifetime) or the name of a CLF PDF file.

Throws
------
RuntimeError
    If clf is not numeric and does not name a file that can be found.)doc";

static constexpr std::string_view setSFRDocstring = R"doc(Set the star formation rate.

Parameters
----------
sfr : str
    A numerical value (interpreted as a constant star formation rate,
    in Msun/yr) or the name of an SFR PDF file.

Throws
------
RuntimeError
    If sfr is neither numeric nor names a file that can be parsed as a
    PDF descriptor.

Details
-------
Unlike setIMF()/setCMF()/setFeH()/setCLF(), a numerical value is not
interpreted as a delta function, but as the normalization of a
non-normalized PDF that is constant in time -- mirroring how
galaxy.sfr itself is handled when parsing an input deck. Also unlike
those four, a file name is not resolved relative to SLUG_DIR/REPO_DIR;
it is used as given.)doc";

static constexpr std::string_view setSpecsynDocstring = R"doc(Set the spectral synthesizer.

Parameters
----------
specsyn : Specsyn
    The spectral synthesizer to use (e.g. a SpecsynBlackbody,
    SpecsynLibNoWind, SpecsynLibWR, or SpecsynLibChained); ownership is
    transferred to this SimPhysics, so specsyn is no longer usable
    from Python after this call.

Details
-------
Lets a caller build its own spectral synthesizer and install it on an
already-constructed SimPhysics, without needing an input deck.)doc";

static constexpr std::string_view setFiltersDocstring = R"doc(Set the photometric filter collection.

Parameters
----------
filters : FilterCollection
    The filter collection to use; ownership is transferred to this
    SimPhysics, so filters is no longer usable from Python after this
    call.

Details
-------
Lets a caller build its own FilterCollection (e.g. via addFilter())
and install it on an already-constructed SimPhysics, without needing
an input deck.)doc";

static constexpr std::string_view setTracksDocstring = R"doc(Set the stellar tracks.

Parameters
----------
tracks : Tracks3D
    The stellar tracks to use; ownership is transferred to this
    SimPhysics, so tracks is no longer usable from Python after this
    call.

Details
-------
Lets a caller build its own Tracks3D and install it on an
already-constructed SimPhysics, without needing an input deck. If
constFeH() is True, also recomputes tracks2D() (the [Fe/H]-sliced
cache) from the new tracks, mirroring setFeH()'s own equivalent
recomputation.)doc";

static constexpr std::string_view setMinStochMassDocstring = R"doc(Set the minimum mass for fully stochastic treatment.

Parameters
----------
min_stoch_mass : float
    New minimum mass for fully stochastic treatment.

Details
-------
Also recomputes the fraction of stellar mass being treated
stochastically, as imf().integral(min_stoch_mass, imf().getMax()),
exactly as the constructor does when stars.min_stoch_mass is given.)doc";

static constexpr std::string_view imfPropertyDocstring = R"doc(The initial mass function.

Reading returns a PDF; assigning a str sets a new one via setIMF() (a
numerical value is interpreted as a delta-function IMF at that mass,
otherwise the value is interpreted as the name of an IMF PDF file --
see setIMF()'s own docstring for the exact rules).)doc";

static constexpr std::string_view cmfPropertyDocstring = R"doc(The cluster mass function.

Reading returns a PDF; assigning a str sets a new one via setCMF() --
see its own docstring for the exact rules.)doc";

static constexpr std::string_view feHPropertyDocstring = R"doc(The [Fe/H] distribution.

Reading returns a PDF; assigning a str sets a new one via setFeH() --
see its own docstring for the exact rules, including the tracks2D()
cache rebuild that happens if constFeH() is True afterward.)doc";

static constexpr std::string_view clfPropertyDocstring = R"doc(The cluster lifetime function.

Reading returns a PDF; assigning a str sets a new one via setCLF() --
see its own docstring for the exact rules.)doc";

static constexpr std::string_view sfrPropertyDocstring = R"doc(The star formation rate.

Reading returns a PDF; assigning a str sets a new one via setSFR() --
see its own docstring for the exact rules, which differ from
imf/cmf/feH/clf's.)doc";

static constexpr std::string_view computeLbolPropertyDocstring = R"doc(Whether the bolometric luminosity is computed as an output.

True if "Lbol" was included in phot.filters in the input deck, or
this property (or setComputeLbol()) has since been set to True.)doc";

static constexpr std::string_view specsynPropertyDocstring = R"doc(The spectral synthesizer, or None if none was requested.

Reading returns the Specsyn requested via spectra.model (or None if
spectra.model was not given). Assigning a Specsyn transfers its
ownership to this SimPhysics, so it is no longer usable from Python
after assignment -- see setSpecsyn()'s own docstring.)doc";

static constexpr std::string_view filtersPropertyDocstring = R"doc(The photometric filter collection, or None if none was requested.

Reading returns the FilterCollection requested via phot.filters (or
None if phot.filters was not given). Assigning a FilterCollection
transfers its ownership to this SimPhysics, so it is no longer usable
from Python after assignment -- see setFilters()'s own docstring.)doc";

static constexpr std::string_view tracksPropertyDocstring = R"doc(The stellar tracks.

Assigning a Tracks3D transfers its ownership to this SimPhysics, so it
is no longer usable from Python after assignment -- see setTracks()'s
own docstring, including the tracks2D() cache rebuild that happens if
constFeH() is True.)doc";

static constexpr std::string_view minStochMassPropertyDocstring = R"doc(The minimum mass for fully stochastic treatment.

Assigning a value also recomputes the fraction of stellar mass being
treated stochastically -- see setMinStochMass()'s own docstring.)doc";

static constexpr std::string_view intRelTolPropertyDocstring = R"doc(The relative tolerance for the spectral synthesizer's PDF integration.

Throws
------
RuntimeError
    On read or assignment, if no spectral synthesizer was requested
    (spectra.model was not set in the input deck).)doc";

static constexpr std::string_view intAbsTolPropertyDocstring = R"doc(The absolute tolerance for the spectral synthesizer's PDF integration.

Throws
------
RuntimeError
    On read or assignment, if no spectral synthesizer was requested
    (spectra.model was not set in the input deck).)doc";

static constexpr std::string_view intMaxIterPropertyDocstring = R"doc(The maximum number of evaluations for the spectral synthesizer's PDF integration.

0 means unlimited.

Throws
------
RuntimeError
    On read or assignment, if no spectral synthesizer was requested
    (spectra.model was not set in the input deck).)doc";

static constexpr std::string_view setIntRelTolDocstring = R"doc(Set the relative tolerance for the spectral synthesizer's PDF integration.

Changes the tolerance on the spectral synthesizer this SimPhysics
already holds, so any Cluster built from this SimPhysics picks up the
new tolerance the next time it computes a spectrum -- no need to
construct a new SimPhysics or Specsyn.

Parameters
----------
rel_tol : float
    New relative convergence tolerance.

Throws
------
RuntimeError
    If no spectral synthesizer was requested (spectra.model was not
    set in the input deck).)doc";

static constexpr std::string_view setIntAbsTolDocstring = R"doc(Set the absolute tolerance for the spectral synthesizer's PDF integration.

Changes the tolerance on the spectral synthesizer this SimPhysics
already holds, so any Cluster built from this SimPhysics picks up the
new tolerance the next time it computes a spectrum -- no need to
construct a new SimPhysics or Specsyn.

Parameters
----------
abs_tol : float
    New absolute convergence tolerance.

Throws
------
RuntimeError
    If no spectral synthesizer was requested (spectra.model was not
    set in the input deck).)doc";

static constexpr std::string_view setIntMaxIterDocstring = R"doc(Set the maximum number of evaluations for the spectral synthesizer's PDF integration.

Changes the setting on the spectral synthesizer this SimPhysics
already holds, so any Cluster built from this SimPhysics picks up the
new value the next time it computes a spectrum -- no need to
construct a new SimPhysics or Specsyn.

Parameters
----------
max_iter : int
    New maximum number of integrand evaluations; 0 means unlimited.

Throws
------
RuntimeError
    If no spectral synthesizer was requested (spectra.model was not
    set in the input deck).)doc";

// Disable linting for includes -- the pybind macro magic seems to confuse
// the linter
// NOLINTBEGIN(misc-include-cleaner)
void bindSimPhysics(py::module_& m)
{
    py::class_<io::SimPhysics, py::smart_holder>(m, "SimPhysics")
        .def(py::init(
                [](const std::string& path, const std::string& simType)
                    -> std::unique_ptr<io::SimPhysics>
                {
                    simTypeFromString(simType); // validate; SimControls reads it from deck

                    if (!path.empty())
                    {
                        const toml::table inputDeck = toml::parse_file(path);
                        const io::SimControls controls(inputDeck);
                        return std::make_unique<io::SimPhysics>(inputDeck, controls);
                    }
                    return buildDefaultSimPhysics();
                }),
                constructorDocstring.data(),
                py::arg("path") = "", py::arg("sim_type") = "cluster")
        .def("wl",
                [](const io::SimPhysics& self) -> std::vector<double>
                {
                    checkSpecsyn(self);
                    return self.specsyn()->wl();
                },
                wlDocstring.data())
        .def("wlObs",
                [](const io::SimPhysics& self) -> std::vector<double>
                {
                    checkSpecsyn(self);
                    return self.specsyn()->wlObs();
                },
                wlObsDocstring.data())
        .def("setComputeLbol", &io::SimPhysics::setComputeLbol,
                setComputeLbolDocstring.data(), py::arg("value"))
        .def("setIMF", &io::SimPhysics::setIMF,
                setIMFDocstring.data(), py::arg("imf"))
        .def("setCMF", &io::SimPhysics::setCMF,
                setCMFDocstring.data(), py::arg("cmf"))
        .def("setFeH", &io::SimPhysics::setFeH,
                setFeHDocstring.data(), py::arg("feh"))
        .def("setCLF", &io::SimPhysics::setCLF,
                setCLFDocstring.data(), py::arg("clf"))
        .def("setSFR", &io::SimPhysics::setSFR,
                setSFRDocstring.data(), py::arg("sfr"))
        .def("setSpecsyn", &io::SimPhysics::setSpecsyn,
                setSpecsynDocstring.data(), py::arg("specsyn"))
        .def("setFilters", &io::SimPhysics::setFilters,
                setFiltersDocstring.data(), py::arg("filters"))
        .def("setTracks",
                [](io::SimPhysics& self, std::unique_ptr<tracks::Tracks3D> tracks)
                {
                    self.setTracks(std::move(*tracks));
                },
                setTracksDocstring.data(), py::arg("tracks"))
        .def("setMinStochMass", &io::SimPhysics::setMinStochMass,
                setMinStochMassDocstring.data(), py::arg("min_stoch_mass"))
        .def("setIntRelTol", &io::SimPhysics::setIntRelTol,
                setIntRelTolDocstring.data(), py::arg("rel_tol"))
        .def("setIntAbsTol", &io::SimPhysics::setIntAbsTol,
                setIntAbsTolDocstring.data(), py::arg("abs_tol"))
        .def("setIntMaxIter", &io::SimPhysics::setIntMaxIter,
                setIntMaxIterDocstring.data(), py::arg("max_iter"))
        // Properties: alternative, attribute-style access to the same
        // getters/setters bound as plain methods above (e.g.
        // sp.imf = "20.0" instead of sp.setIMF("20.0")). Getters that
        // return a reference (imf, cmf, feH, clf, sfr, specsyn,
        // filters, tracks) use def_property's own default
        // return_value_policy::reference_internal, tying the
        // returned object's lifetime to this SimPhysics.
        .def_property("imf",
                &io::SimPhysics::imf,
                [](io::SimPhysics& self, const std::string& imf) { self.setIMF(imf); },
                imfPropertyDocstring.data())
        .def_property("cmf",
                &io::SimPhysics::cmf,
                [](io::SimPhysics& self, const std::string& cmf) { self.setCMF(cmf); },
                cmfPropertyDocstring.data())
        .def_property("feH",
                &io::SimPhysics::fehDist,
                [](io::SimPhysics& self, const std::string& feH) { self.setFeH(feH); },
                feHPropertyDocstring.data())
        .def_property("clf",
                &io::SimPhysics::clf,
                [](io::SimPhysics& self, const std::string& clf) { self.setCLF(clf); },
                clfPropertyDocstring.data())
        .def_property("sfr",
                &io::SimPhysics::sfr,
                [](io::SimPhysics& self, const std::string& sfr) { self.setSFR(sfr); },
                sfrPropertyDocstring.data())
        .def_property("computeLbol",
                &io::SimPhysics::computeLbol,
                &io::SimPhysics::setComputeLbol,
                computeLbolPropertyDocstring.data())
        .def_property("specsyn",
                &io::SimPhysics::specsyn,
                &io::SimPhysics::setSpecsyn,
                specsynPropertyDocstring.data())
        .def_property("filters",
                &io::SimPhysics::filters,
                &io::SimPhysics::setFilters,
                filtersPropertyDocstring.data())
        .def_property("tracks",
                &io::SimPhysics::tracks,
                [](io::SimPhysics& self, std::unique_ptr<tracks::Tracks3D> tracks)
                { self.setTracks(std::move(*tracks)); },
                tracksPropertyDocstring.data())
        .def_property("minStochMass",
                &io::SimPhysics::minStochMass,
                &io::SimPhysics::setMinStochMass,
                minStochMassPropertyDocstring.data())
        .def_property("intRelTol",
                &io::SimPhysics::intRelTol,
                &io::SimPhysics::setIntRelTol,
                intRelTolPropertyDocstring.data())
        .def_property("intAbsTol",
                &io::SimPhysics::intAbsTol,
                &io::SimPhysics::setIntAbsTol,
                intAbsTolPropertyDocstring.data())
        .def_property("intMaxIter",
                &io::SimPhysics::intMaxIter,
                &io::SimPhysics::setIntMaxIter,
                intMaxIterPropertyDocstring.data());
}
// NOLINTEND(misc-include-cleaner)
