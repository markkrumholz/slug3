/**
 * @file BindSimControls.cpp
 * @author Mark Krumholz
 * @brief Python bindings for io::SimControls
 * @date 2026-07-30
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 * @details
 * io::SimControls holds both simulation control-flow settings (IO,
 * trial count, output timing) and physics settings (IMF, tracks,
 * spectral synthesis, ...) -- originally two separate classes,
 * SimControls and SimPhysics, merged into one; this file's own
 * bindings were originally split the same way, across
 * BindSimControls.cpp and BindSimPhysics.cpp.
 */

#include "Bindings.hpp"
#include "../extinct/Extinct.hpp"
#include "../feedback/FeedbackCommons.hpp"
#include "../feedback/Winds.hpp"
#include "../io/SimControls.hpp"
#include "../nebular/Nebular.hpp"
#include "../phot/FilterCollection.hpp"
#include "../specsyn/Specsyn.hpp"
#include "../tracks/Tracks3D.hpp"
#include "../utils/MiscUtils.hpp"
#include "../utils/TrackedDeck.hpp"
#include "../yields/Yields.hpp"
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
static constexpr std::string_view classDocstring = R"doc(Holds simulation control-flow information and physics settings.

Bundles together both control-flow settings (I/O, trial count, output
timing) and physics choices (IMF, tracks, spectral synthesis) for a
simulation -- almost every other object slug builds (Cluster, Galaxy,
SimCluster, SimGalaxy, the output managers, every spectral
synthesizer) needs both together, and reads many of its own settings
(e.g. the integrator tolerances) live from this object rather than a
snapshot taken at construction time, so a SimControls must outlive
anything built from it.)doc";

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
    SimControls() followed by sc.imf = "20.0" (except that if imf
    and tracks are both given, they are applied in whichever order
    lets each pass the other's mass-range check, so that a valid
    pair is accepted regardless of the deck's own IMF and tracks).
    specsyn/filters/tracks
    transfer ownership exactly as their own setter/property does, so
    the object passed in is no longer usable from Python afterward.
    specsyn, unlike filters/tracks, cannot usefully be passed here in
    practice: a Specsyn must have been constructed with controls=this
    same SimControls (see setSpecsyn()'s own docstring), which does not
    yet exist at the point a value for this keyword argument would
    need to be built -- prefer setting it after construction instead
    (sc = SimControls(...); sc.specsyn = SpecsynBlackbody(...,
    controls=sc)). See each property's own docstring for further
    details.

Throws
------
RuntimeError
    If the file cannot be parsed, sim_type is not "cluster" or
    "galaxy", the deck is otherwise invalid, (only if path is empty)
    the bundled default deck cannot be found, or any of the
    property-setting keyword arguments above would itself raise a
    RuntimeError.
ValueError
    If specsyn is given and was constructed against a different
    SimControls -- see specsyn's own description above.)doc";

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

static constexpr std::string_view nTrialGetterDocstring = R"doc(Return the number of trials this simulation will run.

Returns
-------
n_trial : int)doc";

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
    If imf is not numeric and does not name a file that can be found.
ValueError
    If the new IMF's maximum mass exceeds the current stellar tracks'
    maximum mass. This SimControls is then left unchanged.

Details
-------
Like construction from an input deck, also prints a warning if the
new IMF extends below the tracks' minimum mass; stars in that range
are treated as having zero luminosity.)doc";

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
    If feh is not numeric and does not name a file that can be found,
    or if its own [min, max] range is broader than the [Fe/H] range the
    current stellar tracks were loaded over -- widening past it risks
    interpolating outside the range of data actually loaded -- or
    extends beyond the requestedFehMin()/requestedFehMax() range of the
    installed spectral synthesizer or Yields, if any. Narrowing, or
    otherwise staying within, those ranges is always accepted. On
    rejection, this SimControls is left unchanged.)doc";

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

static constexpr std::string_view setAVDistDocstring = R"doc(Set the distribution of V-band extinction for clustered stars.

Parameters
----------
av_dist : str
    A numerical value (interpreted as a delta-function A_V at that
    value) or the name of an A_V PDF file.

Throws
------
RuntimeError
    If av_dist is not numeric and does not name a file that can be
    found.

Details
-------
If this SimControls already has an extinction curve (its extinct
property is not None), also rebuilds that Extinct's own cached
quantities (Extinct.rebuildCache()) -- av_dist itself is never
actually read by Extinct (only av_dist_field is), so this is harmless
rather than strictly necessary, but kept for symmetry with
set_av_dist_field()'s own identical behavior, where it is actually
needed.)doc";

static constexpr std::string_view setAVDistFieldDocstring = R"doc(Set the distribution of V-band extinction for field stars.

Parameters
----------
av_dist_field : str
    A numerical value (interpreted as a delta-function A_V at that
    value) or the name of an A_V PDF file.

Throws
------
RuntimeError
    If av_dist_field is not numeric and does not name a file that can
    be found.

Details
-------
If this SimControls already has an extinction curve (its extinct
property is not None), also rebuilds that Extinct's own cached
quantities (Extinct.rebuildCache()), so its cached expectation values
(used by Extinct.applyExtinctionCts()/applyExtinctionCtsLines()) stay
in sync with the new distribution.)doc";

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
it is used as given. Also clears sfrDist back to invalid/empty -- see
setSFRDist()'s own docstring for why.)doc";

static constexpr std::string_view setSFRDistDocstring = R"doc(Set the distribution from which a single, constant star formation rate is drawn.

Parameters
----------
sfr_dist : str
    A numerical value (interpreted as a delta function at that rate)
    or the name of an SFR-distribution PDF file.

Throws
------
RuntimeError
    If sfr_dist is neither numeric nor names a file that can be found.

Details
-------
Unlike setSFR(), this is a plain PDF-valued setter (a numerical value
becomes a delta function, exactly as setCLF()/setCMF()/etc. already
work): sfrDist() is a genuine distribution over a single, scalar rate
value, not itself a rate as a function of time the way sfr is. Also
clears sfr back to invalid/empty: only one of sfr/sfrDist is ever
meant to be valid at a time (mirroring the exclusive-or rule the
constructor enforces between galaxy.sfr/galaxy.sfr_dist when parsing a
real input deck), so setting one via its own setter invalidates the
other, regardless of which was set first. Each Galaxy built from a
SimControls where sfrDist is the valid one draws its own, independent
constant rate from it, once, at construction.)doc";

static constexpr std::string_view setSpecsynDocstring = R"doc(Set the spectral synthesizer.

Parameters
----------
specsyn : Specsyn, optional
    The spectral synthesizer to use (e.g. a SpecsynBlackbody,
    SpecsynLibNoWind, SpecsynLibWR, or SpecsynLibChained); ownership is
    transferred to this SimControls, so specsyn is no longer usable
    from Python after this call. May be None, to remove the current
    one.

Throws
------
ValueError
    If specsyn is not None and was constructed with a controls
    argument other than this same SimControls -- a Specsyn stores a
    live reference to whichever SimControls it was built against, for
    the rest of its lifetime, and this method cannot re-bind it, so
    specsyn must already have been constructed with controls=this
    SimControls (e.g. slug.SpecsynBlackbody(..., controls=sc);
    sc.setSpecsyn(specsyn)). This also means specsyn cannot usefully be
    passed as a constructor keyword argument (SimControls(...,
    specsyn=...)): the SimControls being constructed does not exist
    yet at the point specsyn would need to be built against it.
    Also raised if specsyn's requestedFehMin()/requestedFehMax() range
    does not contain the currently installed synthesizer's own (or, if
    there is none, the current [Fe/H] distribution): a synthesizer
    built for a narrower [Fe/H] range could not cover every [Fe/H] the
    simulation may ask about. Either way, this SimControls is left
    unchanged and specsyn stays usable from Python.

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
    call. Must not be None.

Throws
------
ValueError
    If tracks is None, or if the [Fe/H] range tracks was loaded over
    does not cover the current [Fe/H] distribution (the feH
    property's own [min, max]), or if the tracks' maximum mass is
    below the current IMF's maximum mass. This SimControls is then
    left unchanged, and tracks stays usable from Python.

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
cache rebuild that happens if constFeH() is True afterward, and the
RuntimeError raised if the new range is broader than the current
one.)doc";

static constexpr std::string_view clfPropertyDocstring = R"doc(The cluster lifetime function.

Reading returns a PDF; assigning a str sets a new one via setCLF() --
see its own docstring for the exact rules.)doc";

static constexpr std::string_view sfrPropertyDocstring = R"doc(The star formation rate.

Reading returns a PDF; assigning a str sets a new one via setSFR() --
see its own docstring for the exact rules, which differ from
imf/cmf/feH/clf's.)doc";

static constexpr std::string_view sfrDistPropertyDocstring = R"doc(The distribution from which a single, constant star formation rate is drawn.

Reading returns a PDF; assigning a str sets a new one via
setSFRDist() -- see its own docstring for the exact rules.)doc";

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
spectra.model was not given); a Specsyn read this way stays fully
valid even after a later assignment replaces it, though it's simply no
longer the one this property itself returns. Assigning a Specsyn (or
None, to remove one already present) transfers its ownership to this
SimControls, so it is no longer usable from Python after assignment --
see setSpecsyn()'s own docstring, including the ValueError raised if it
was built against a different SimControls.)doc";

static constexpr std::string_view filtersPropertyDocstring = R"doc(The photometric filter collection, or None if none was requested.

Reading returns the FilterCollection requested via phot.filters (or
None if phot.filters was not given); a FilterCollection read this way
stays fully valid even after a later assignment replaces it. Assigning
a FilterCollection transfers its ownership to this SimControls, so it
is no longer usable from Python after assignment -- see setFilters()'s
own docstring.)doc";

static constexpr std::string_view extinctPropertyDocstring = R"doc(The extinction curve, or None if none was requested.

Reading returns the Extinct requested via extinct.model (or None if
neither extinct.AV nor extinct.AV_field was given in the input deck),
built once, at construction, or later installed via setExtinct(); an
Extinct read this way stays fully valid even after a later assignment
replaces it. Assigning an Extinct (or None, to remove one already
present) transfers its ownership to this SimControls, so it is no
longer usable from Python after assignment -- see setExtinct()'s own
docstring, including the ValueError raised if it was built against a
different SimControls.)doc";

static constexpr std::string_view nebularPropertyDocstring = R"doc(The nebular emission grid, or None if none was requested.

Reading returns the Nebular built from nebular.table/stars.tracks if
nebular emission is being computed, and None otherwise. Nebular emission
requires a spectral synthesizer, so nebular.compute_neb defaults to true
if the input deck requested one (a deck that never mentions [nebular] at
all still builds a Nebular), and to false if it did not; explicitly
setting nebular.compute_neb = true without a spectral synthesizer is an
error, and setting it false always leaves this None. Built once, at
construction, or later installed via setNebular(); a Nebular read this way stays
fully valid even after a later assignment replaces it. Assigning a
Nebular (or None, to remove one already present) transfers its
ownership to this SimControls, so it is no longer usable from Python
after assignment -- see setNebular()'s own docstring, including the
ValueError raised if it was built against a different SimControls.)doc";

static constexpr std::string_view setExtinctDocstring = R"doc(Set the extinction curve.

Parameters
----------
extinct : Extinct, optional
    The extinction curve to use; ownership is transferred to this
    SimControls, so extinct is no longer usable from Python after this
    call. May be None, to remove the current one.

Throws
------
ValueError
    If extinct is not None and was constructed with a controls
    argument other than this same SimControls.

Details
-------
Lets a caller build its own Extinct and install it on an
already-constructed SimControls, without needing an input deck --
including installing one for the first time on a SimControls whose
extinct property was previously None, or passing None to remove one
already present.

extinct must have been constructed with its own controls argument set
to this same SimControls (e.g. extinct = slug.Extinct(name,
controls=sc); sc.setExtinct(extinct)) -- an Extinct stores a live
reference to whichever SimControls it was built against, for the rest
of its lifetime, and this method cannot re-bind it.)doc";

static constexpr std::string_view setYieldsDocstring = R"doc(Set the Yields built from yieldChannels.

Parameters
----------
yields : Yields, optional
    The Yields to use; ownership is transferred to this SimControls,
    so yields is no longer usable from Python after this call. May be
    None, to remove the current one.

Throws
------
ValueError
    If yields is not None and was constructed with a controls argument
    other than this same SimControls, or if its
    requestedFehMin()/requestedFehMax() range does not contain the
    currently installed Yields' own (or, if there is none, the current
    [Fe/H] distribution). Either way, this SimControls is left
    unchanged and yields stays usable from Python.

Details
-------
Lets a caller build its own Yields and install it on an
already-constructed SimControls, without needing an input deck --
including installing one for the first time on a SimControls whose
yields property was previously None, or passing None to remove one
already present. Does not touch yieldChannels either way, so the two
can end up disagreeing if set independently.

yields must have been constructed with its own controls argument set
to this same SimControls (e.g. yields = slug.Yields(controls=sc);
sc.setYields(yields)) -- a Yields stores a live reference to whichever
SimControls it was built against, for the rest of its lifetime, and
this method cannot re-bind it.

A Yields object previously read back from the yields property stays
fully valid even after this method replaces (or, with None, clears)
this SimControls's own copy -- it's simply no longer the one the
yields property itself returns afterward.)doc";

static constexpr std::string_view setNebularDocstring = R"doc(Set the nebular emission grid.

Parameters
----------
nebular : Nebular, optional
    The nebular emission grid to use; ownership is transferred to this
    SimControls, so nebular is no longer usable from Python after this
    call. May be None, to remove the current one.

Throws
------
ValueError
    If nebular is not None and was constructed with a controls
    argument other than this same SimControls.

Details
-------
Lets a caller build its own Nebular and install it on an
already-constructed SimControls, without needing an input deck --
including installing one for the first time on a SimControls whose
nebular property was previously None, or passing None to remove one
already present. Does not touch nebControls().compute_neb either way,
so that flag and the nebular property's own None-ness can end up
disagreeing if set independently -- see setNebControls()'s own
docstring.

nebular must have been constructed with its own controls argument set
to this same SimControls, for the same reason described in
setExtinct()'s own docstring -- a Nebular stores an identical live
reference to whichever SimControls it was built against.)doc";

static constexpr std::string_view tracksPropertyDocstring = R"doc(The stellar tracks.

Reading returns the Tracks3D loaded via stars.tracks; a Tracks3D read
this way stays fully valid even after a later assignment replaces it.
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

static constexpr std::string_view intMaxIterPropertyDocstring = R"doc(The maximum number of bisection iterations for PDF integration.

Not a count of raw integrand evaluations, which each iteration costs
several of. 0 means unlimited. See intRelTol's own docstring on live
(not snapshotted) effect.)doc";

static constexpr std::string_view zPropertyDocstring = R"doc(The redshift.

Applied by every Specsyn's and Extinct's own wlObs(). Any such object
built by this SimControls reads this value live, not a snapshot, so
assigning a new value takes effect the next time wlObs() is called --
no need to rebuild anything.)doc";

static constexpr std::string_view modelNamePropertyDocstring =
R"doc(The model name.

Base name of this simulation's output file(s) -- e.g. for HDF5 output,
modelName + ".h5". Defaults to "slug_sim" when no output.model_name was
given in the input deck.

Like the write* flags, this is read by an OutputManager at its own
construction, not live: assigning this only affects an OutputManager
built from this SimControls afterward, not one already built from it.)doc";

static constexpr std::string_view outDirPropertyDocstring =
R"doc(The output directory.

Directory into which output files are written. An empty string (the
default when no output.out_dir was given in the input deck) means the
current working directory.

Like the write* flags, this is read by an OutputManager at its own
construction, not live: assigning this only affects an OutputManager
built from this SimControls afterward, not one already built from it.)doc";

static constexpr std::string_view nTrialPropertyDocstring =
R"doc(The number of trials this simulation will run.

Defaults to 1 when no n_trial was given in the input deck. Assigning
this before calling SimCluster.run() / SimGalaxy.run() takes effect
immediately -- no need to rebuild the simulation object.)doc";

static constexpr std::string_view outputModePropertyDocstring =
R"doc(The output mode.

One of SimControls.OutputMode.h5 (the default), .h5divided, or .ascii.
Determines what kind of OutputManager is built from this SimControls, so
like the write* flags, assigning this only affects an OutputManager built
afterward, not one already built from it.)doc";

static constexpr std::string_view verbosityPropertyDocstring =
R"doc(The verbosity level.

Controls how much diagnostic output is printed during a simulation
run. 0 (the default) means no extra output; higher values produce
progressively more. Read live by every object that checks this
SimControls's verbosity() -- no need to rebuild anything after
assigning.)doc";

static constexpr std::string_view strictInputPropertyDocstring =
R"doc(Whether an input-deck key that is never used is an error.

Read-only. This is the value of the optional top-level strict_input
key of the deck this SimControls was built from, False by default.
If False, a key that nothing reads (a typo, or a key in the wrong
section) only produces a warning, and is listed in unusedKeys. If
True, constructing the SimControls raises an error instead.

Returns
-------
strict_input : bool)doc";

static constexpr std::string_view unusedKeysPropertyDocstring =
R"doc(Input-deck keys that were never used.

Read-only. Filled in when the SimControls is constructed from a deck:
every key the deck contains is recorded, and any that nothing read is
listed here -- typically a misspelled key, or one given in the wrong
section (e.g. n_trial under [output] rather than at the top level).
Each such key also produces a warning when the deck is read. Empty for
a deck with no stray keys, and for a default-constructed SimControls.

Returns
-------
keys : list of dict
    One dict per key, in order of path, with entries 'path' (str, the
    dotted path of the key, e.g. "output.n_trial"), 'line' (int, its
    line in the deck, or 0 if the deck was not parsed from text),
    'column' (int, likewise), and 'hint' (str, a suggestion of what the
    key was probably meant to be, e.g. "did you mean 'n_trial'?", or an
    empty string if there is none).)doc";

static constexpr std::string_view ignoredKeysPropertyDocstring =
R"doc(Input-deck keys that were valid but deliberately not read.

Read-only. Some keys are fine but have no effect for a particular
simulation, e.g. nebular.log_U when nebular.compute_neb is false, or
clusters.CLF in a cluster simulation. Unlike unusedKeys, these are not
mistakes, so they produce no warning (only a note at verbosity 1 or
higher).

Returns
-------
keys : list of dict
    One dict per key, in order of path, with entries 'path', 'line' and
    'column' (as for unusedKeys) and 'reason' (str, why the key was not
    read, e.g. "nebular.compute_neb is false").)doc";

// Convert a list of input-deck key reports to a Python list of dicts.
// Each dict has the key's path, line, and column, plus its hint (for an
// unused key) or its reason (for an ignored one)
static auto deckKeyReportsToPython(const std::vector<utils::DeckKeyReport>& reports, const bool ignored) -> py::list
{
    py::list result;
    for (const auto& report : reports)
    {
        py::dict entry;
        entry["path"] = report.path_;
        entry["line"] = report.line_;
        entry["column"] = report.column_;
        if (ignored) { entry["reason"] = report.reason_; }
        else { entry["hint"] = report.hint_; }
        result.append(entry);
    }
    return result;
}

static constexpr std::string_view outTimesPropertyDocstring =
R"doc(The output times, in years.

Reading returns the output times as a list of floats -- either the
explicit array from the input deck, or a single time drawn from the
output time distribution if a distribution was specified instead.
Assigning a list sets an explicit array and clears any distribution
that was previously set: once assigned, outTimes always returns the
assigned list, never a draw from a distribution.)doc";

static constexpr std::string_view checkpointIntervalPropertyDocstring =
R"doc(The number of trials between checkpoints.

0 (the default) disables checkpointing. Only supported with HDF5
output (output_mode "h5" or "h5divided"), not ascii: assigning a
non-zero value here does not itself check output_mode the way
setting output.checkpoint_interval in an input deck does, so a
SimControls built with ascii output that later has this set to a
non-zero value here will raise a RuntimeError the first time a
checkpoint is actually attempted, rather than immediately.)doc";

static constexpr std::string_view writeClusterPropertyDocstring =
R"doc(Whether the clusters group/file is written.

True (the default) unless output.write_cluster was set to false in
the input deck. Unlike intRelTol/z/etc., this is not read live by an
already-built OutputManagerH5/OutputManagerAscii: each one decides
once, at its own construction, which groups/files to create at all,
so assigning this only affects an OutputManagerH5/OutputManagerAscii
built from this SimControls afterward, not one already built from
it.)doc";

static constexpr std::string_view writeClusterSpecPropertyDocstring =
R"doc(Whether the cluster_spectra group/file is written.

True (the default) unless output.write_cluster_spec was set to false
in the input deck. See writeCluster's own docstring on when assigning
this does (and does not) take effect.)doc";

static constexpr std::string_view writeClusterPhotPropertyDocstring =
R"doc(Whether the cluster_phot group/file is written.

True (the default) unless output.write_cluster_phot was set to false
in the input deck. See writeCluster's own docstring on when assigning
this does (and does not) take effect.)doc";

static constexpr std::string_view writeGalaxyPropertyDocstring =
R"doc(Whether the galaxy group/file is written.

True (the default) unless output.write_galaxy was set to false in the
input deck. Only meaningful for a galaxy-type simulation. See
writeCluster's own docstring on when assigning this does (and does
not) take effect.)doc";

static constexpr std::string_view writeGalaxySpecPropertyDocstring =
R"doc(Whether the galaxy_spectra group/file is written.

True (the default) unless output.write_galaxy_spec was set to false
in the input deck. Only meaningful for a galaxy-type simulation. See
writeCluster's own docstring on when assigning this does (and does
not) take effect.)doc";

static constexpr std::string_view writeGalaxyPhotPropertyDocstring =
R"doc(Whether the galaxy_phot group/file is written.

True (the default) unless output.write_galaxy_phot was set to false
in the input deck. Only meaningful for a galaxy-type simulation. See
writeCluster's own docstring on when assigning this does (and does
not) take effect.)doc";

static constexpr std::string_view writeClusterYieldsPropertyDocstring =
R"doc(Whether the cluster_yields group/file is written.

True (the default) unless output.write_cluster_yields was set to false
in the input deck. See writeCluster's own docstring on when assigning
this does (and does not) take effect.)doc";

static constexpr std::string_view writeGalaxyYieldsPropertyDocstring =
R"doc(Whether the galaxy_yields group/file is written.

True (the default) unless output.write_galaxy_yields was set to false
in the input deck. Only meaningful for a galaxy-type simulation. See
writeCluster's own docstring on when assigning this does (and does
not) take effect.)doc";

static constexpr std::string_view yieldChannelsPropertyDocstring =
R"doc(The nucleosynthetic yield channels requested via yields.channel1, yields.channel2, etc.

Read-only; an empty list if no yields.channelN table was given at all.)doc";

static constexpr std::string_view yieldsPropertyDocstring =
R"doc(The Yields built from yieldChannels, or None if none was requested.

Reading returns the Yields built once, at construction, or later
installed via setYields() -- see Yields's own class docstring; a Yields
read this way stays fully valid even after a later assignment replaces
it. Assigning a Yields (or None, to remove one already present) transfers
its ownership to this SimControls, so it is no longer usable from
Python after assignment -- see setYields()'s own docstring, including
the ValueError raised if it was built against a different
SimControls.)doc";

static constexpr std::string_view yieldsChannelDecomposedPropertyDocstring =
R"doc(Whether yields should be reported decomposed by channel.

True (the default) unless yields.channel_decomposed was set to false
in the input deck. Meaningful only if yields is not None. Unlike
writeCluster/etc., this is read live by Cluster.computeYields()/
Galaxy.computeYields() and OutputManagerH5's own cluster_yields/
galaxy_yields group creation every time each runs, not cached once --
so assigning this takes effect immediately, the same as z/intRelTol,
rather than only affecting an object built afterward.)doc";

static constexpr std::string_view noDecayPropertyDocstring =
R"doc(Whether radioactive decay should be excluded from computed yields.

False (the default) unless yields.no_decay was set to true in the
input deck. Meaningful only if yields is not None. False means
Yields.yield_()/yieldSum() report the amount of each isotope actually
present at the requested output time, including whatever radioactive
decay has occurred since it was produced; True means they instead
report the cumulative amount of each isotope ever produced, regardless
of whether some of it has since decayed into something else. Like
yieldsChannelDecomposed, this is read live every time
Yields.yield_()/yieldSum() runs, not cached once, so assigning this
takes effect immediately.)doc";

static constexpr std::string_view minIsotopeLifetimePropertyDocstring =
R"doc(Minimum isotope lifetime, in yr, below which isotopes decay instantly.

1e4 unless yields.min_isotope_lifetime was set in the input deck.
Meaningful only if yields is not None. Every unstable isotope with a
shorter lifetime is dropped from yields.isotopes and treated as
decaying the instant it is produced: its tabulated yield goes to its
first longer-lived descendants, and decays into it go straight on to
those descendants too. This is accurate for yields at times long
compared to this lifetime, and makes decay much cheaper to compute. 0
skips nothing; +inf makes every unstable isotope decay instantly.

Assigning any non-negative float rebuilds yields (if not None) at once,
keeping its current isotope restriction (Yields.requestedIsotopes).

Raises
------
ValueError
    If the assigned value is negative or NaN. The old value is then
    kept.
RuntimeError
    If the rebuild fails, e.g. because every requested isotope would be
    skipped. The old value is then restored.)doc";

static constexpr std::string_view snMassLimitsPropertyDocstring =
R"doc(The stellar mass limits over which supernovae occur.

A list of masses in Msun, read from feedback.sn_mass_range in the
input deck (empty if that key was not given). Consecutive pairs
(lower, upper) each give one mass interval, inclusive of both ends, in
which stars end their lives as supernovae; more than one pair
describes disjoint intervals. If empty, hasSN() defers to the
core-collapse supernova yield channels in yields instead.

Raises
------
ValueError
    On assignment (or setSNMassLimits()), if the list has an odd
    number of elements or its elements are not strictly increasing
    (which also rejects NaN); the existing limits are left unchanged
    in this case.)doc";

static constexpr std::string_view hasSNDocstring =
R"doc(Check whether a star of a given mass ends its life as a supernova.

Parameters
----------
mass : float
    Stellar mass (Msun).

Returns
-------
has_sn : bool
    If snMassLimits is non-empty, True if mass lies within any of its
    (lower, upper) intervals, inclusive of both ends. Otherwise, if
    yields is not None, whether mass lies within the mass range of any
    core-collapse supernova (ccsn) yield channel loaded. Otherwise,
    False.)doc";

static constexpr std::string_view hasSNFeHDocstring =
R"doc(Check whether a star of a given mass and [Fe/H] ends its life as a supernova.

Parameters
----------
mass : float
    Stellar mass (Msun).
feh : float
    [Fe/H] of the star.

Returns
-------
has_sn : bool
    If snMassLimits is non-empty, True if mass lies within any of its
    (lower, upper) intervals, inclusive of both ends (feh is ignored).
    Otherwise, if yields is not None,
    yields.hasYield(mass, feh, YieldChannelType.ccsn) -- which, unlike
    hasSN(mass), also catches failed supernovae within the ccsn yield
    channels' own mass ranges. Otherwise, False.)doc";

static constexpr std::string_view wrWindModelPropertyDocstring =
R"doc(The Wolf-Rayet wind velocity model.

One of "none", "l_over_c", or "nugis_lamers_00", read from
feedback.wr_winds in the input deck ("nugis_lamers_00" if that key was
not given). Read live by Winds.vWindWR() on every call -- see its own
docstring for what each model does -- so assigning a new value (or
calling setWRWindModel()) takes effect immediately, with no need to
rebuild winds.

Raises
------
ValueError
    On assignment (or setWRWindModel()), if the value is not one of
    the three names above; the existing model is left unchanged in
    this case.)doc";

static constexpr std::string_view obWindModelPropertyDocstring =
R"doc(The O and B star wind velocity model.

One of "none", "vink_01", or "vink_sander_21", read from
feedback.ob_winds in the input deck ("vink_sander_21" if that key was
not given). Read live by Winds.vWindOB() on every call -- see its own
docstring for what each model does -- so assigning a new value (or
calling setOBWindModel()) takes effect immediately, with no need to
rebuild winds.

Raises
------
ValueError
    On assignment (or setOBWindModel()), if the value is not one of
    the three names above; the existing model is left unchanged in
    this case.)doc";

static constexpr std::string_view agbWindModelPropertyDocstring =
R"doc(The AGB star wind velocity model.

One of "none" or "slug2", read from feedback.agb_winds in the input
deck ("slug2" if that key was not given). Read live by Winds.vWindAGB()
on every call -- see its own docstring for what each model does -- so
assigning a new value (or calling setAGBWindModel()) takes effect
immediately, with no need to rebuild winds.

Raises
------
ValueError
    On assignment (or setAGBWindModel()), if the value is not one of
    the two names above; the existing model is left unchanged in this
    case.)doc";

static constexpr std::string_view otherWindModelPropertyDocstring =
R"doc(The wind velocity model for all stars not covered by wrWindModel,
obWindModel, or agbWindModel.

One of "none" or "vesc", read from feedback.other_winds in the input
deck ("vesc" if that key was not given). Read live by
Winds.vWindOther() on every call -- see its own docstring for what
each model does -- so assigning a new value (or calling
setOtherWindModel()) takes effect immediately, with no need to rebuild
winds.

Raises
------
ValueError
    On assignment (or setOtherWindModel()), if the value is not one of
    the two names above; the existing model is left unchanged in this
    case.)doc";

static constexpr std::string_view windsPropertyDocstring =
R"doc(The stellar wind calculator, or None if none is installed.

Reading returns the Winds built at construction, or later installed via
setWinds(); None only for a SimControls that was never given one.
Assigning a Winds (or None, to remove one already present) transfers
ownership in exactly the same way as setWinds() -- see its own
docstring.)doc";

static constexpr std::string_view setWRWindModelDocstring = R"doc(Set the Wolf-Rayet wind velocity model.

Equivalent to assigning the wrWindModel property -- see its own
docstring for what each model does.

Parameters
----------
model : str
    One of "none", "l_over_c", or "nugis_lamers_00".

Raises
------
ValueError
    If model is not one of the three names above; the existing model
    is left unchanged in this case.)doc";

static constexpr std::string_view setOBWindModelDocstring = R"doc(Set the O and B star wind velocity model.

Equivalent to assigning the obWindModel property -- see its own
docstring for what each model does.

Parameters
----------
model : str
    One of "none", "vink_01", or "vink_sander_21".

Raises
------
ValueError
    If model is not one of the three names above; the existing model
    is left unchanged in this case.)doc";

static constexpr std::string_view setAGBWindModelDocstring = R"doc(Set the AGB star wind velocity model.

Equivalent to assigning the agbWindModel property -- see its own
docstring for what each model does.

Parameters
----------
model : str
    One of "none" or "slug2".

Raises
------
ValueError
    If model is not one of the two names above; the existing model is
    left unchanged in this case.)doc";

static constexpr std::string_view setOtherWindModelDocstring = R"doc(Set the wind velocity model for all other stars.

Equivalent to assigning the otherWindModel property -- see its own
docstring for what each model does.

Parameters
----------
model : str
    One of "none" or "vesc".

Raises
------
ValueError
    If model is not one of the two names above; the existing model is
    left unchanged in this case.)doc";

static constexpr std::string_view setWindsDocstring = R"doc(Set the stellar wind calculator.

Parameters
----------
winds : Winds, optional
    The Winds to use; ownership is transferred to this SimControls,
    so winds is no longer usable from Python after this call. May be
    None, to remove the one already present.

Raises
------
ValueError
    If winds is not None and was constructed with a controls argument
    other than this same SimControls -- a Winds stores a live reference
    to whichever SimControls it was built against, and reads its
    wind models (wrWindModel, obWindModel, etc.) from there. The current
    winds is left unchanged, and winds stays usable from Python, in this
    case.)doc";

static constexpr std::string_view inputDeckStrPropertyDocstring =
R"doc(The input deck's own text.

The toml table this SimControls was constructed from, re-serialized
back to text -- not necessarily byte-identical to whatever text/file
originally produced that table, but parses back to an equivalent
table. Empty if this SimControls was built with no input deck at all.
Read-only: there is no legitimate reason for a caller to override what
deck actually built this SimControls.)doc";

// Apply each property-named constructor keyword argument that was
// actually given (not py::none()) to an already-built sc, via the
// same setter its property uses -- factored out of the constructor
// lambda below purely to keep bindSimControls()'s own cognitive
// complexity down; see constructorDocstring for the user-facing
// contract this implements
// setTracks(), for Python: checks that the new tracks cover the
// current [Fe/H] distribution and IMF mass range
// (SimControls::checkTracksCoverFeH()/checkTracksCoverIMF()) through
// a borrowed reference first, so that a rejected Tracks3D
// stays usable from Python, rather than having its ownership moved
// into setTracks()'s own argument, and then destroyed, before the
// check even runs. None is passed straight through, so setTracks()
// reports it as usual.
static void setTracksKeepOnFailure(io::SimControls& sc, py::object tracksArg)
{
    if (!tracksArg.is_none())
    {
        const auto& tracks = py::cast<const tracks::Tracks3D&>(tracksArg);
        sc.checkTracksCoverFeH(tracks);
        sc.checkTracksCoverIMF(tracks);
    }
    sc.setTracks(py::cast<std::unique_ptr<tracks::Tracks3D>>(std::move(tracksArg)));
}

// setSpecsyn()/setYields(), for Python: run SimControls'
// checkSpecsynReplacement()/checkYieldsReplacement() through a
// borrowed reference first, so that a rejected object stays usable
// from Python -- see setTracksKeepOnFailure()'s own comment. None is
// passed straight through, to remove the current object.
static void setSpecsynKeepOnFailure(io::SimControls& sc, py::object specsynArg)
{
    if (!specsynArg.is_none()) { sc.checkSpecsynReplacement(py::cast<const specsyn::Specsyn&>(specsynArg)); }
    sc.setSpecsyn(py::cast<std::unique_ptr<specsyn::Specsyn>>(std::move(specsynArg)));
}

static void setYieldsKeepOnFailure(io::SimControls& sc, py::object yieldsArg)
{
    if (!yieldsArg.is_none()) { sc.checkYieldsReplacement(py::cast<const yields::Yields&>(yieldsArg)); }
    sc.setYields(py::cast<std::unique_ptr<yields::Yields>>(std::move(yieldsArg)));
}

static void setWindsKeepOnFailure(io::SimControls& sc, py::object windsArg)
{
    if (!windsArg.is_none()) { sc.checkWindsReplacement(py::cast<const feedback::Winds&>(windsArg)); }
    sc.setWinds(py::cast<std::unique_ptr<feedback::Winds>>(std::move(windsArg)));
}

// wrWindModel, for Python: translated to/from its input-deck name (see
// feedback::wrWindModelStr), matching the PDF sampling property's own
// string-valued convention
static void setWRWindModelFromString(io::SimControls& sc, const std::string& model)
{
    sc.setWRWindModel(feedback::wrWindModelFromString(model));
}

static auto wrWindModelAsString(const io::SimControls& sc) -> std::string
{
    return std::string(feedback::wrWindModelToString(sc.wrWindModel()));
}

// obWindModel, for Python: likewise, via feedback::obWindModelStr
static void setOBWindModelFromString(io::SimControls& sc, const std::string& model)
{
    sc.setOBWindModel(feedback::obWindModelFromString(model));
}

static auto obWindModelAsString(const io::SimControls& sc) -> std::string
{
    return std::string(feedback::obWindModelToString(sc.obWindModel()));
}

// agbWindModel and otherWindModel, for Python: likewise, via
// feedback::agbWindModelStr and feedback::otherWindModelStr
static void setAGBWindModelFromString(io::SimControls& sc, const std::string& model)
{
    sc.setAGBWindModel(feedback::agbWindModelFromString(model));
}

static auto agbWindModelAsString(const io::SimControls& sc) -> std::string
{
    return std::string(feedback::agbWindModelToString(sc.agbWindModel()));
}

static void setOtherWindModelFromString(io::SimControls& sc, const std::string& model)
{
    sc.setOtherWindModel(feedback::otherWindModelFromString(model));
}

static auto otherWindModelAsString(const io::SimControls& sc) -> std::string
{
    return std::string(feedback::otherWindModelToString(sc.otherWindModel()));
}

static void applyConstructorProperties(io::SimControls& sc,
    const py::object& imf, const py::object& cmf, const py::object& feH,
    const py::object& clf, const py::object& sfr, const py::object& computeLbol,
    py::object specsynArg, py::object filtersArg, py::object tracksArg,
    const py::object& minStochMass, const py::object& intRelTol,
    const py::object& intAbsTol, const py::object& intMaxIter)
{
    // imf and tracks are each checked against the other's current
    // value when set (see setIMF()/setTracks()), so if both are given,
    // set tracks first only if they already cover the current IMF;
    // otherwise the new IMF must lie within the current tracks for
    // the pair to be valid at all, so set it first
    const bool tracksFirst = !imf.is_none() && !tracksArg.is_none() &&
        py::cast<const tracks::Tracks3D&>(tracksArg).mMax() >= sc.imf().getMax();
    if (!imf.is_none() && !tracksFirst) { sc.setIMF(py::cast<std::string>(imf)); }
    if (!cmf.is_none()) { sc.setCMF(py::cast<std::string>(cmf)); }
    if (!feH.is_none()) { sc.setFeH(py::cast<std::string>(feH)); }
    if (!clf.is_none()) { sc.setCLF(py::cast<std::string>(clf)); }
    if (!sfr.is_none()) { sc.setSFR(py::cast<std::string>(sfr)); }
    if (!computeLbol.is_none()) { sc.setComputeLbol(py::cast<bool>(computeLbol)); }
    if (!specsynArg.is_none())
    {
        setSpecsynKeepOnFailure(sc, std::move(specsynArg));
    }
    if (!filtersArg.is_none())
    {
        sc.setFilters(
            py::cast<std::unique_ptr<phot::FilterCollection>>(std::move(filtersArg)));
    }
    if (!tracksArg.is_none()) { setTracksKeepOnFailure(sc, std::move(tracksArg)); }
    if (tracksFirst) { sc.setIMF(py::cast<std::string>(imf)); }
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
    py::class_<io::SimControls, py::smart_holder> simControlsClass(m, "SimControls", classDocstring.data());

    py::enum_<io::SimControls::SimType>(simControlsClass, "SimType",
            "The type of simulation this SimControls describes.")
        .value("cluster", io::SimControls::SimType::cluster, "Cluster simulation")
        .value("galaxy", io::SimControls::SimType::galaxy, "Galaxy simulation")
        .value("none", io::SimControls::SimType::none,
                "Dummy value, before a real simulation type has been set");

    py::enum_<io::SimControls::OutputMode>(simControlsClass, "OutputMode",
            "The format this SimControls's own output will be written in.")
        .value("h5", io::SimControls::OutputMode::h5, "HDF5 output")
        .value("h5divided", io::SimControls::OutputMode::h5divided,
                "Same as h5, but skips the final consolidation step when "
                "built with OpenMP, leaving one HDF5 file per thread")
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
        .def("setOutputMode", &io::SimControls::setOutputMode,
                outputModePropertyDocstring.data(), py::arg("mode"))
        .def("setModelName", &io::SimControls::setModelName,
                modelNamePropertyDocstring.data(), py::arg("name"))
        .def("setOutDir", &io::SimControls::setOutDir,
                outDirPropertyDocstring.data(), py::arg("dir"))
        .def("setNTrial", &io::SimControls::setNTrial,
                nTrialPropertyDocstring.data(), py::arg("n"))
        .def("setVerbosity", &io::SimControls::setVerbosity,
                verbosityPropertyDocstring.data(), py::arg("verbosity"))
        .def("setOutTimes", &io::SimControls::setOutTimes,
                outTimesPropertyDocstring.data(), py::arg("times"))
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
        .def("setAVDist", &io::SimControls::setAVDist,
                setAVDistDocstring.data(), py::arg("av_dist"))
        .def("setAVDistField", &io::SimControls::setAVDistField,
                setAVDistFieldDocstring.data(), py::arg("av_dist_field"))
        .def("setSFR", &io::SimControls::setSFR,
                setSFRDocstring.data(), py::arg("sfr"))
        .def("setSFRDist", &io::SimControls::setSFRDist,
                setSFRDistDocstring.data(), py::arg("sfr_dist"))
        .def("setSpecsyn", &setSpecsynKeepOnFailure,
                setSpecsynDocstring.data(), py::arg("specsyn"))
        .def("setFilters", &io::SimControls::setFilters,
                setFiltersDocstring.data(), py::arg("filters"))
        .def("setTracks", &setTracksKeepOnFailure,
                setTracksDocstring.data(), py::arg("tracks"))
        .def("setExtinct", &io::SimControls::setExtinct,
                setExtinctDocstring.data(), py::arg("extinct"))
        .def("setNebular", &io::SimControls::setNebular,
                setNebularDocstring.data(), py::arg("nebular"))
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
        .def("setCheckpointInterval", &io::SimControls::setCheckpointInterval,
                checkpointIntervalPropertyDocstring.data(), py::arg("interval"))
        .def("setWriteCluster", &io::SimControls::setWriteCluster,
                writeClusterPropertyDocstring.data(), py::arg("value"))
        .def("setWriteClusterSpec", &io::SimControls::setWriteClusterSpec,
                writeClusterSpecPropertyDocstring.data(), py::arg("value"))
        .def("setWriteClusterPhot", &io::SimControls::setWriteClusterPhot,
                writeClusterPhotPropertyDocstring.data(), py::arg("value"))
        .def("setWriteGalaxy", &io::SimControls::setWriteGalaxy,
                writeGalaxyPropertyDocstring.data(), py::arg("value"))
        .def("setWriteGalaxySpec", &io::SimControls::setWriteGalaxySpec,
                writeGalaxySpecPropertyDocstring.data(), py::arg("value"))
        .def("setWriteGalaxyPhot", &io::SimControls::setWriteGalaxyPhot,
                writeGalaxyPhotPropertyDocstring.data(), py::arg("value"))
        .def("setWriteClusterYields", &io::SimControls::setWriteClusterYields,
                writeClusterYieldsPropertyDocstring.data(), py::arg("value"))
        .def("setWriteGalaxyYields", &io::SimControls::setWriteGalaxyYields,
                writeGalaxyYieldsPropertyDocstring.data(), py::arg("value"))
        .def("setYieldsChannelDecomposed", &io::SimControls::setYieldsChannelDecomposed,
                yieldsChannelDecomposedPropertyDocstring.data(), py::arg("value"))
        .def("setNoDecay", &io::SimControls::setNoDecay,
                noDecayPropertyDocstring.data(), py::arg("value"))
        .def("setMinIsotopeLifetime", &io::SimControls::setMinIsotopeLifetime,
                minIsotopeLifetimePropertyDocstring.data(), py::arg("value"))
        .def("setSNMassLimits", &io::SimControls::setSNMassLimits,
                snMassLimitsPropertyDocstring.data(), py::arg("limits"))
        .def("hasSN", py::overload_cast<double>(&io::SimControls::hasSN, py::const_),
                hasSNDocstring.data(), py::arg("mass"))
        .def("hasSN", py::overload_cast<double, double>(&io::SimControls::hasSN, py::const_),
                hasSNFeHDocstring.data(), py::arg("mass"), py::arg("feh"))
        .def("setYields", &setYieldsKeepOnFailure,
                setYieldsDocstring.data(), py::arg("yields"))
        .def("setWRWindModel", &setWRWindModelFromString,
                setWRWindModelDocstring.data(), py::arg("model"))
        .def("setOBWindModel", &setOBWindModelFromString,
                setOBWindModelDocstring.data(), py::arg("model"))
        .def("setAGBWindModel", &setAGBWindModelFromString,
                setAGBWindModelDocstring.data(), py::arg("model"))
        .def("setOtherWindModel", &setOtherWindModelFromString,
                setOtherWindModelDocstring.data(), py::arg("model"))
        .def("setWinds", &setWindsKeepOnFailure,
                setWindsDocstring.data(), py::arg("winds"))
        // Properties: alternative, attribute-style access to the same
        // getters/setters bound as plain methods above (e.g.
        // sc.imf = "20.0" instead of sc.setIMF("20.0")). Getters that
        // return a reference (imf, cmf, feH, clf, sfr, sfrDist, specsyn,
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
        .def_property("sfrDist",
                &io::SimControls::sfrDist,
                [](io::SimControls& self, const std::string& sfrDist) { self.setSFRDist(sfrDist); },
                sfrDistPropertyDocstring.data())
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
                &setSpecsynKeepOnFailure,
                specsynPropertyDocstring.data())
        .def_property("filters",
                &io::SimControls::filters,
                &io::SimControls::setFilters,
                filtersPropertyDocstring.data())
        .def_property("extinct",
                &io::SimControls::extinct,
                &io::SimControls::setExtinct,
                extinctPropertyDocstring.data())
        .def_property("nebular",
                &io::SimControls::nebular,
                &io::SimControls::setNebular,
                nebularPropertyDocstring.data())
        .def_property("tracks",
                &io::SimControls::tracks,
                &setTracksKeepOnFailure,
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
                zPropertyDocstring.data())
        .def_property("outputMode",
                &io::SimControls::outputMode,
                &io::SimControls::setOutputMode,
                outputModePropertyDocstring.data())
        .def_property("modelName",
                &io::SimControls::modelName,
                &io::SimControls::setModelName,
                modelNamePropertyDocstring.data())
        .def_property("outDir",
                &io::SimControls::outDir,
                &io::SimControls::setOutDir,
                outDirPropertyDocstring.data())
        .def_property("nTrial",
                &io::SimControls::nTrial,
                &io::SimControls::setNTrial,
                nTrialPropertyDocstring.data())
        .def_property("verbosity",
                &io::SimControls::verbosity,
                &io::SimControls::setVerbosity,
                verbosityPropertyDocstring.data())
        .def_property("outTimes",
                &io::SimControls::outTimes,
                &io::SimControls::setOutTimes,
                outTimesPropertyDocstring.data())
        .def_property("checkpointInterval",
                &io::SimControls::checkpointInterval,
                &io::SimControls::setCheckpointInterval,
                checkpointIntervalPropertyDocstring.data())
        .def_property("writeCluster",
                &io::SimControls::writeCluster,
                &io::SimControls::setWriteCluster,
                writeClusterPropertyDocstring.data())
        .def_property("writeClusterSpec",
                &io::SimControls::writeClusterSpec,
                &io::SimControls::setWriteClusterSpec,
                writeClusterSpecPropertyDocstring.data())
        .def_property("writeClusterPhot",
                &io::SimControls::writeClusterPhot,
                &io::SimControls::setWriteClusterPhot,
                writeClusterPhotPropertyDocstring.data())
        .def_property("writeGalaxy",
                &io::SimControls::writeGalaxy,
                &io::SimControls::setWriteGalaxy,
                writeGalaxyPropertyDocstring.data())
        .def_property("writeGalaxySpec",
                &io::SimControls::writeGalaxySpec,
                &io::SimControls::setWriteGalaxySpec,
                writeGalaxySpecPropertyDocstring.data())
        .def_property("writeGalaxyPhot",
                &io::SimControls::writeGalaxyPhot,
                &io::SimControls::setWriteGalaxyPhot,
                writeGalaxyPhotPropertyDocstring.data())
        .def_property("writeClusterYields",
                &io::SimControls::writeClusterYields,
                &io::SimControls::setWriteClusterYields,
                writeClusterYieldsPropertyDocstring.data())
        .def_property("writeGalaxyYields",
                &io::SimControls::writeGalaxyYields,
                &io::SimControls::setWriteGalaxyYields,
                writeGalaxyYieldsPropertyDocstring.data())
        .def_property("yieldsChannelDecomposed",
                &io::SimControls::yieldsChannelDecomposed,
                &io::SimControls::setYieldsChannelDecomposed,
                yieldsChannelDecomposedPropertyDocstring.data())
        .def_property("noDecay",
                &io::SimControls::noDecay,
                &io::SimControls::setNoDecay,
                noDecayPropertyDocstring.data())
        .def_property("minIsotopeLifetime",
                &io::SimControls::minIsotopeLifetime,
                &io::SimControls::setMinIsotopeLifetime,
                minIsotopeLifetimePropertyDocstring.data())
        .def_property("snMassLimits",
                &io::SimControls::snMassLimits,
                &io::SimControls::setSNMassLimits,
                snMassLimitsPropertyDocstring.data())
        .def_property_readonly("yieldChannels",
                &io::SimControls::yieldChannels,
                yieldChannelsPropertyDocstring.data())
        .def_property("yields",
                &io::SimControls::yields,
                &setYieldsKeepOnFailure,
                yieldsPropertyDocstring.data(), py::return_value_policy::reference_internal)
        .def_property("wrWindModel",
                &wrWindModelAsString,
                &setWRWindModelFromString,
                wrWindModelPropertyDocstring.data())
        .def_property("obWindModel",
                &obWindModelAsString,
                &setOBWindModelFromString,
                obWindModelPropertyDocstring.data())
        .def_property("agbWindModel",
                &agbWindModelAsString,
                &setAGBWindModelFromString,
                agbWindModelPropertyDocstring.data())
        .def_property("otherWindModel",
                &otherWindModelAsString,
                &setOtherWindModelFromString,
                otherWindModelPropertyDocstring.data())
        .def_property("winds",
                &io::SimControls::winds,
                &setWindsKeepOnFailure,
                windsPropertyDocstring.data(), py::return_value_policy::reference_internal)
        .def_property_readonly("inputDeckStr",
                &io::SimControls::inputDeckStr,
                inputDeckStrPropertyDocstring.data())
        .def_property_readonly("strictInput",
                &io::SimControls::strictInput,
                strictInputPropertyDocstring.data())
        .def_property_readonly("unusedKeys",
                [](const io::SimControls& self) { return deckKeyReportsToPython(self.unusedKeys(), false); },
                unusedKeysPropertyDocstring.data())
        .def_property_readonly("ignoredKeys",
                [](const io::SimControls& self) { return deckKeyReportsToPython(self.ignoredKeys(), true); },
                ignoredKeysPropertyDocstring.data());
}
// NOLINTEND(misc-include-cleaner)
