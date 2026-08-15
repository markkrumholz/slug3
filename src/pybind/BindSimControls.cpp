/**
 * @file BindSimControls.cpp
 * @author Mark Krumholz
 * @brief Python bindings for io::SimControls
 * @date 2026-07-30
 * @details
 * io::SimControls holds both simulation control-flow settings (IO,
 * trial count, output timing) and physics settings (IMF, tracks,
 * spectral synthesis, ...) -- originally two separate classes,
 * SimControls and SimPhysics, merged into one; this file's own
 * bindings were originally split the same way, across
 * BindSimControls.cpp and BindSimPhysics.cpp.
 */

#include "Bindings.hpp"
#include "../io/SimControls.hpp"
#include "../phot/FilterCollection.hpp"
#include "../specsyn/Specsyn.hpp"
#include "../tracks/Tracks3D.hpp"
#include "../utils/MiscUtils.hpp"
#include <cstddef>
#include <memory>
#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h> // NOLINT(misc-include-cleaner); this is needed for correct Python binding, even if clang-tidy can't recognize it
#include <stdexcept>
#include <string>
#include <string_view>
#include <toml.hpp>
#include <utility>
#include <vector>

// Map the Python-facing sim_type string to SimControls::SimType,
// validating it against the same vocabulary the deck's own sim_type
// key must use (see SimControls.cpp) -- the deck's own sim_type key
// is what actually determines simType(), not this argument; this just
// gives a clearer, earlier error for an unrecognized value
static void simTypeFromString(const std::string& simType)
{
    if (simType != "cluster" && simType != "galaxy")
    {
        throw std::runtime_error("SimControls: sim_type must be 'cluster' or 'galaxy'");
    }
}

// Declared in Bindings.hpp (shared with BindCluster.cpp -- see its own
// comment there). The bundled deck specifies no output-time
// information beyond a single placeholder time, since it is meant for
// interactive Cluster exploration rather than driving a full
// simulation.
auto buildDefaultControls() -> std::unique_ptr<io::SimControls>
{
    const auto defaultsPath = utils::getFilePath("PyDefaults.toml", "src/pybind/assets");
    if (defaultsPath.empty())
    {
        throw std::runtime_error(
            "SimControls: default input deck PyDefaults.toml not found");
    }
    const toml::table inputDeck = toml::parse_file(defaultsPath.string());
    return std::make_unique<io::SimControls>(inputDeck);
}

// Declared in Bindings.hpp (shared with BindCluster.cpp -- see its own
// comment there)
auto sharedDefaultControls() -> const io::SimControls&
{
    static const std::unique_ptr<io::SimControls> instance = buildDefaultControls();
    return *instance;
}

// Declared in Bindings.hpp (shared with BindSpecsyn.cpp -- see its own
// comment there)
auto sharedMinimalControls() -> const io::SimControls&
{
    static const io::SimControls instance;
    return instance;
}

// Numpy-style docstring for the Python constructor binding. This is
// a convenience constructor, not a direct binding of SimControls's
// real toml::table constructor -- toml++ is a vendored third-party
// library with no Python bindings of its own, so exposing it was not
// worth the surface area for what is, from Python, just a file to
// load.
static constexpr std::string_view constructorDocstring = R"doc(Construct a SimControls object by parsing a slug input deck.

Parameters
----------
path : str, optional
    Either the text of a slug TOML input deck, or a path to one on
    disk. Tried first as literal TOML text (via toml::parse); if that
    fails to parse, tried again as a file path instead (via
    toml::parse_file) -- so e.g. a path like "deck.toml" (not valid
    TOML on its own) falls through to being read as a file, while a
    string like "n_trial = 10" (a path that could never exist) is used
    directly as the deck's own content. If path parses successfully as
    literal TOML text, every other argument below is ignored -- there
    is no sim_type key to cross-check outside of the deck's own text,
    and none of imf/cmf/... would have any given deck left to
    override. If path is omitted (or empty), loads slug's own bundled
    default deck (src/pybind/assets/PyDefaults.toml) instead, letting
    a caller build a usable SimControls for interactive work -- e.g.
    sc = slug.SimControls() -- without touching an input deck at all.
sim_type : str, optional
    Either "cluster" or "galaxy"; validated against the deck's own
    sim_type key (which is what actually determines the simulation
    type) purely to give a clearer, earlier error for an unrecognized
    value. Defaults to "cluster", matching the bundled default deck's
    own sim_type.
imf, cmf, feH, clf, sfr : str, optional
computeLbol : bool, optional
specsyn : Specsyn, optional
filters : FilterCollection, optional
tracks : Tracks3D, optional
minStochMass, intRelTol, intAbsTol : float, optional
intMaxIter : int, optional
    Each, if given, is applied via the corresponding property's own
    setter (setIMF(), setCMF(), ..., setIntMaxIter()) after the
    SimControls described above (from path, or the bundled default
    deck) is otherwise fully built -- e.g.
    SimControls(imf="20.0") is equivalent to
    SimControls() followed by sc.imf = "20.0". specsyn/filters/tracks
    transfer ownership exactly as their own setter/property does, so
    the object passed in is no longer usable from Python afterward.
    See each property's own docstring for further details.

Throws
------
RuntimeError
    If the file cannot be parsed, sim_type is not "cluster" or
    "galaxy", the deck is otherwise invalid, (only if path is empty)
    the bundled default deck cannot be found, or any of the
    property-setting keyword arguments above would itself raise.)doc";

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

static constexpr std::string_view simTypeGetterDocstring = R"doc(Return the simulation type.

Returns
-------
sim_type : SimControls.SimType)doc";

static constexpr std::string_view outputModeGetterDocstring = R"doc(Return the output mode.

Returns
-------
output_mode : SimControls.OutputMode)doc";

static constexpr std::string_view modelNameGetterDocstring = R"doc(Return the model name.

Returns
-------
model_name : str
    Base name of this simulation's output file(s) -- e.g. for HDF5
    output, model_name + ".h5".)doc";

static constexpr std::string_view outDirGetterDocstring = R"doc(Return the output directory.

Returns
-------
out_dir : str
    Directory into which output will be written. An empty string (the
    default) means output will be written into the current working
    directory.)doc";

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
static void checkSpecsyn(const io::SimControls& self)
{
    if (self.specsyn() == nullptr)
    {
        throw std::runtime_error(
            "SimControls: no spectral synthesizer was requested "
            "(spectra.model was not set in the input deck)");
    }
}

static constexpr std::string_view setComputeLbolDocstring = R"doc(Set whether the bolometric luminosity should be computed as an output.

Lets a caller request or suppress Lbol output on an already-constructed
SimControls -- e.g. from Python, where there is no phot.filters input-
deck entry to set "Lbol" through -- so any Cluster built from this
SimControls picks up the new setting the next time it advances.

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
    transferred to this SimControls, so specsyn is no longer usable
    from Python after this call.

Details
-------
Lets a caller build its own spectral synthesizer and install it on an
already-constructed SimControls, without needing an input deck.)doc";

static constexpr std::string_view setFiltersDocstring = R"doc(Set the photometric filter collection.

Parameters
----------
filters : FilterCollection
    The filter collection to use; ownership is transferred to this
    SimControls, so filters is no longer usable from Python after this
    call.

Details
-------
Lets a caller build its own FilterCollection (e.g. via addFilter())
and install it on an already-constructed SimControls, without needing
an input deck.)doc";

static constexpr std::string_view setTracksDocstring = R"doc(Set the stellar tracks.

Parameters
----------
tracks : Tracks3D
    The stellar tracks to use; ownership is transferred to this
    SimControls, so tracks is no longer usable from Python after this
    call.

Details
-------
Lets a caller build its own Tracks3D and install it on an
already-constructed SimControls, without needing an input deck. If
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

static constexpr std::string_view fClusterPropertyDocstring = R"doc(The fraction of stellar mass formed in stochastically-treated clusters.

The fraction of a galaxy simulation's stellar mass that forms in
clusters treated stochastically (as individual Cluster objects); the
remaining (1 - fCluster) is treated as a stellar population continuous
in both mass and time. Defaults to 1.0 (every star forms in a
stochastic cluster); only meaningful for, and only read from, a
galaxy-type simulation's own clusters.f_cluster.)doc";

static constexpr std::string_view computeLbolPropertyDocstring = R"doc(Whether the bolometric luminosity is computed as an output.

True if "Lbol" was included in phot.filters in the input deck, or
this property (or setComputeLbol()) has since been set to True.)doc";

static constexpr std::string_view specsynPropertyDocstring = R"doc(The spectral synthesizer, or None if none was requested.

Reading returns the Specsyn requested via spectra.model (or None if
spectra.model was not given). Assigning a Specsyn transfers its
ownership to this SimControls, so it is no longer usable from Python
after assignment -- see setSpecsyn()'s own docstring.)doc";

static constexpr std::string_view filtersPropertyDocstring = R"doc(The photometric filter collection, or None if none was requested.

Reading returns the FilterCollection requested via phot.filters (or
None if phot.filters was not given). Assigning a FilterCollection
transfers its ownership to this SimControls, so it is no longer usable
from Python after assignment -- see setFilters()'s own docstring.)doc";

static constexpr std::string_view tracksPropertyDocstring = R"doc(The stellar tracks.

Assigning a Tracks3D transfers its ownership to this SimControls, so
it is no longer usable from Python after assignment -- see
setTracks()'s own docstring, including the tracks2D() cache rebuild
that happens if constFeH() is True.)doc";

static constexpr std::string_view minStochMassPropertyDocstring = R"doc(The minimum mass for fully stochastic treatment.

Assigning a value also recomputes the fraction of stellar mass being
treated stochastically -- see setMinStochMass()'s own docstring.)doc";

static constexpr std::string_view intRelTolPropertyDocstring = R"doc(The relative tolerance for PDF integration.

Any spectral synthesizer built by this SimControls reads this value
live, not a snapshot, so assigning a new value takes effect the next
time it integrates -- no need to rebuild the synthesizer.)doc";

static constexpr std::string_view intAbsTolPropertyDocstring = R"doc(The absolute tolerance for PDF integration.

See intRelTol's own docstring on live (not snapshotted) effect.)doc";

static constexpr std::string_view intMaxIterPropertyDocstring = R"doc(The maximum number of evaluations for PDF integration.

0 means unlimited. See intRelTol's own docstring on live (not
snapshotted) effect.)doc";

static constexpr std::string_view zPropertyDocstring = R"doc(The redshift.

Applied by every Specsyn's and Extinct's own wlObs(). Any such object
built by this SimControls reads this value live, not a snapshot, so
assigning a new value takes effect the next time wlObs() is called --
no need to rebuild anything.)doc";

// Apply each property-named constructor keyword argument that was
// actually given (not py::none()) to an already-built sc, via the
// same setter its property uses -- factored out of the constructor
// lambda below purely to keep bindSimControls()'s own cognitive
// complexity down; see constructorDocstring for the user-facing
// contract this implements
static void applyConstructorProperties(io::SimControls& sc,
    const py::object& imf, const py::object& cmf, const py::object& feH,
    const py::object& clf, const py::object& sfr, const py::object& computeLbol,
    py::object specsynArg, py::object filtersArg, py::object tracksArg,
    const py::object& minStochMass, const py::object& intRelTol,
    const py::object& intAbsTol, const py::object& intMaxIter)
{
    if (!imf.is_none()) { sc.setIMF(py::cast<std::string>(imf)); }
    if (!cmf.is_none()) { sc.setCMF(py::cast<std::string>(cmf)); }
    if (!feH.is_none()) { sc.setFeH(py::cast<std::string>(feH)); }
    if (!clf.is_none()) { sc.setCLF(py::cast<std::string>(clf)); }
    if (!sfr.is_none()) { sc.setSFR(py::cast<std::string>(sfr)); }
    if (!computeLbol.is_none()) { sc.setComputeLbol(py::cast<bool>(computeLbol)); }
    if (!specsynArg.is_none())
    {
        sc.setSpecsyn(py::cast<std::unique_ptr<specsyn::Specsyn>>(std::move(specsynArg)));
    }
    if (!filtersArg.is_none())
    {
        sc.setFilters(
            py::cast<std::unique_ptr<phot::FilterCollection>>(std::move(filtersArg)));
    }
    if (!tracksArg.is_none())
    {
        auto tracksPtr = py::cast<std::unique_ptr<tracks::Tracks3D>>(std::move(tracksArg));
        sc.setTracks(std::move(*tracksPtr));
    }
    if (!minStochMass.is_none()) { sc.setMinStochMass(py::cast<double>(minStochMass)); }
    if (!intRelTol.is_none()) { sc.setIntRelTol(py::cast<double>(intRelTol)); }
    if (!intAbsTol.is_none()) { sc.setIntAbsTol(py::cast<double>(intAbsTol)); }
    if (!intMaxIter.is_none()) { sc.setIntMaxIter(py::cast<std::size_t>(intMaxIter)); }
}

// Disable linting for includes -- the pybind macro magic seems to confuse
// the linter
// NOLINTBEGIN(misc-include-cleaner)
void bindSimControls(py::module_& m)
{
    py::class_<io::SimControls, py::smart_holder> simControlsClass(m, "SimControls");

    py::enum_<io::SimControls::SimType>(simControlsClass, "SimType",
            "The type of simulation this SimControls describes.")
        .value("cluster", io::SimControls::SimType::cluster, "Cluster simulation")
        .value("galaxy", io::SimControls::SimType::galaxy, "Galaxy simulation")
        .value("none", io::SimControls::SimType::none,
                "Dummy value, before a real simulation type has been set");

    py::enum_<io::SimControls::OutputMode>(simControlsClass, "OutputMode",
            "The format this SimControls's own output will be written in.")
        .value("h5", io::SimControls::OutputMode::h5, "HDF5 output")
        .value("ascii", io::SimControls::OutputMode::ascii, "ASCII output");

    simControlsClass
        .def(py::init(
                [](const std::string& path, const std::string& simType,
                   const py::object& imf, const py::object& cmf, const py::object& feH,
                   const py::object& clf, const py::object& sfr,
                   const py::object& computeLbol, py::object specsynArg,
                   py::object filtersArg, py::object tracksArg,
                   const py::object& minStochMass, const py::object& intRelTol,
                   const py::object& intAbsTol, const py::object& intMaxIter)
                    -> std::unique_ptr<io::SimControls>
                {
                    // path may be literal TOML text rather than a
                    // file path (see constructorDocstring); try that
                    // interpretation first, and if it succeeds, return
                    // immediately -- every other argument here only
                    // makes sense relative to a deck this path doesn't
                    // name, so there is nothing left for them to do.
                    if (!path.empty())
                    {
                        try
                        {
                            const toml::table inputDeck = toml::parse(path);
                            return std::make_unique<io::SimControls>(inputDeck);
                        }
                        catch (const toml::parse_error&) // NOLINT(bugprone-empty-catch) -- deliberately empty: this exception just means path isn't literal TOML text, so fall through and treat it as a file path instead
                        {
                        }
                    }

                    simTypeFromString(simType); // validate; the deck's own sim_type key is authoritative

                    std::unique_ptr<io::SimControls> sc;
                    if (!path.empty())
                    {
                        const toml::table inputDeck = toml::parse_file(path);
                        sc = std::make_unique<io::SimControls>(inputDeck);
                    }
                    else
                    {
                        sc = buildDefaultControls();
                    }

                    applyConstructorProperties(*sc, imf, cmf, feH, clf, sfr, computeLbol,
                        std::move(specsynArg), std::move(filtersArg), std::move(tracksArg),
                        minStochMass, intRelTol, intAbsTol, intMaxIter);

                    return sc;
                }),
                constructorDocstring.data(),
                py::arg("path") = "", py::arg("sim_type") = "cluster",
                py::arg("imf") = py::none(), py::arg("cmf") = py::none(),
                py::arg("feH") = py::none(), py::arg("clf") = py::none(),
                py::arg("sfr") = py::none(), py::arg("computeLbol") = py::none(),
                py::arg("specsyn") = py::none(), py::arg("filters") = py::none(),
                py::arg("tracks") = py::none(), py::arg("minStochMass") = py::none(),
                py::arg("intRelTol") = py::none(), py::arg("intAbsTol") = py::none(),
                py::arg("intMaxIter") = py::none())
        .def("wl",
                [](const io::SimControls& self) -> std::vector<double>
                {
                    checkSpecsyn(self);
                    return self.specsyn()->wl();
                },
                wlDocstring.data())
        .def("wlObs",
                [](const io::SimControls& self) -> std::vector<double>
                {
                    checkSpecsyn(self);
                    return self.specsyn()->wlObs();
                },
                wlObsDocstring.data())
        .def("simType", &io::SimControls::simType,
                simTypeGetterDocstring.data())
        .def("outputMode", &io::SimControls::outputMode,
                outputModeGetterDocstring.data())
        .def("modelName", &io::SimControls::modelName,
                modelNameGetterDocstring.data())
        .def("outDir", &io::SimControls::outDir,
                outDirGetterDocstring.data())
        .def("setComputeLbol", &io::SimControls::setComputeLbol,
                setComputeLbolDocstring.data(), py::arg("value"))
        .def("setIMF", &io::SimControls::setIMF,
                setIMFDocstring.data(), py::arg("imf"))
        .def("setCMF", &io::SimControls::setCMF,
                setCMFDocstring.data(), py::arg("cmf"))
        .def("setFeH", &io::SimControls::setFeH,
                setFeHDocstring.data(), py::arg("feh"))
        .def("setCLF", &io::SimControls::setCLF,
                setCLFDocstring.data(), py::arg("clf"))
        .def("setSFR", &io::SimControls::setSFR,
                setSFRDocstring.data(), py::arg("sfr"))
        .def("setSpecsyn", &io::SimControls::setSpecsyn,
                setSpecsynDocstring.data(), py::arg("specsyn"))
        .def("setFilters", &io::SimControls::setFilters,
                setFiltersDocstring.data(), py::arg("filters"))
        .def("setTracks",
                [](io::SimControls& self, std::unique_ptr<tracks::Tracks3D> tracks)
                {
                    self.setTracks(std::move(*tracks));
                },
                setTracksDocstring.data(), py::arg("tracks"))
        .def("setMinStochMass", &io::SimControls::setMinStochMass,
                setMinStochMassDocstring.data(), py::arg("min_stoch_mass"))
        .def("setIntRelTol", &io::SimControls::setIntRelTol,
                intRelTolPropertyDocstring.data(), py::arg("rel_tol"))
        .def("setIntAbsTol", &io::SimControls::setIntAbsTol,
                intAbsTolPropertyDocstring.data(), py::arg("abs_tol"))
        .def("setIntMaxIter", &io::SimControls::setIntMaxIter,
                intMaxIterPropertyDocstring.data(), py::arg("max_iter"))
        .def("setZ", &io::SimControls::setZ,
                zPropertyDocstring.data(), py::arg("z"))
        .def("setFCluster", &io::SimControls::setFCluster,
                fClusterPropertyDocstring.data(), py::arg("f_cluster"))
        // Properties: alternative, attribute-style access to the same
        // getters/setters bound as plain methods above (e.g.
        // sc.imf = "20.0" instead of sc.setIMF("20.0")). Getters that
        // return a reference (imf, cmf, feH, clf, sfr, specsyn,
        // filters, tracks) use def_property's own default
        // return_value_policy::reference_internal, tying the
        // returned object's lifetime to this SimControls.
        .def_property("imf",
                &io::SimControls::imf,
                [](io::SimControls& self, const std::string& imf) { self.setIMF(imf); },
                imfPropertyDocstring.data())
        .def_property("cmf",
                &io::SimControls::cmf,
                [](io::SimControls& self, const std::string& cmf) { self.setCMF(cmf); },
                cmfPropertyDocstring.data())
        .def_property("feH",
                &io::SimControls::fehDist,
                [](io::SimControls& self, const std::string& feH) { self.setFeH(feH); },
                feHPropertyDocstring.data())
        .def_property("clf",
                &io::SimControls::clf,
                [](io::SimControls& self, const std::string& clf) { self.setCLF(clf); },
                clfPropertyDocstring.data())
        .def_property("sfr",
                &io::SimControls::sfr,
                [](io::SimControls& self, const std::string& sfr) { self.setSFR(sfr); },
                sfrPropertyDocstring.data())
        .def_property("fCluster",
                &io::SimControls::fCluster,
                &io::SimControls::setFCluster,
                fClusterPropertyDocstring.data())
        .def_property("computeLbol",
                &io::SimControls::computeLbol,
                &io::SimControls::setComputeLbol,
                computeLbolPropertyDocstring.data())
        .def_property("specsyn",
                &io::SimControls::specsyn,
                &io::SimControls::setSpecsyn,
                specsynPropertyDocstring.data())
        .def_property("filters",
                &io::SimControls::filters,
                &io::SimControls::setFilters,
                filtersPropertyDocstring.data())
        .def_property("tracks",
                &io::SimControls::tracks,
                [](io::SimControls& self, std::unique_ptr<tracks::Tracks3D> tracks)
                { self.setTracks(std::move(*tracks)); },
                tracksPropertyDocstring.data())
        .def_property("minStochMass",
                &io::SimControls::minStochMass,
                &io::SimControls::setMinStochMass,
                minStochMassPropertyDocstring.data())
        .def_property("intRelTol",
                &io::SimControls::intRelTol,
                &io::SimControls::setIntRelTol,
                intRelTolPropertyDocstring.data())
        .def_property("intAbsTol",
                &io::SimControls::intAbsTol,
                &io::SimControls::setIntAbsTol,
                intAbsTolPropertyDocstring.data())
        .def_property("intMaxIter",
                &io::SimControls::intMaxIter,
                &io::SimControls::setIntMaxIter,
                intMaxIterPropertyDocstring.data())
        .def_property("z",
                &io::SimControls::z,
                &io::SimControls::setZ,
                zPropertyDocstring.data());
}
// NOLINTEND(misc-include-cleaner)
