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

// Numpy-style docstring for the Python constructor binding. This is
// a convenience constructor, not a direct binding of SimPhysics's
// real (toml::table, SimControls::SimType) constructor -- toml++ is
// a vendored third-party library with no Python bindings of its own,
// so exposing it was not worth the surface area for what is, from
// Python, just a file to load.
static constexpr std::string_view constructorDocstring = R"doc(Construct a SimPhysics object by parsing a slug input deck.

Parameters
----------
path : str
    Path to a slug TOML input deck.
sim_type : str
    Either "cluster" or "galaxy"; controls whether galaxy-specific
    quantities (the CLF and SFR) are read from the deck.

Throws
------
RuntimeError
    If the file cannot be parsed, sim_type is not "cluster" or
    "galaxy", or the deck is otherwise invalid.)doc";

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

static constexpr std::string_view computeLbolDocstring = R"doc(Check whether the bolometric luminosity was requested as an output.

Returns
-------
compute_lbol : bool
    True if "Lbol" was included in phot.filters in the input deck, or
    setComputeLbol(True) has since been called.)doc";

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

static constexpr std::string_view intRelTolDocstring = R"doc(Return the relative tolerance for the spectral synthesizer's PDF integration.

Returns
-------
rel_tol : float
    Relative convergence tolerance passed to the cubature integrator
    used by the spectral synthesizer's specCts().

Throws
------
RuntimeError
    If no spectral synthesizer was requested (spectra.model was not
    set in the input deck).)doc";

static constexpr std::string_view intAbsTolDocstring = R"doc(Return the absolute tolerance for the spectral synthesizer's PDF integration.

Returns
-------
abs_tol : float
    Absolute convergence tolerance passed to the cubature integrator
    used by the spectral synthesizer's specCts().

Throws
------
RuntimeError
    If no spectral synthesizer was requested (spectra.model was not
    set in the input deck).)doc";

static constexpr std::string_view intMaxIterDocstring = R"doc(Return the maximum number of evaluations for the spectral synthesizer's PDF integration.

Returns
-------
max_iter : int
    Maximum number of integrand evaluations used by the spectral
    synthesizer's specCts(); 0 means unlimited.

Throws
------
RuntimeError
    If no spectral synthesizer was requested (spectra.model was not
    set in the input deck).)doc";

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
                    const toml::table inputDeck = toml::parse_file(path);
                    const io::SimControls controls(inputDeck);
                    return std::make_unique<io::SimPhysics>(inputDeck, controls);
                }),
                constructorDocstring.data(),
                py::arg("path"), py::arg("sim_type"))
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
        .def("computeLbol", &io::SimPhysics::computeLbol,
                computeLbolDocstring.data())
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
        .def("intRelTol", &io::SimPhysics::intRelTol,
                intRelTolDocstring.data())
        .def("intAbsTol", &io::SimPhysics::intAbsTol,
                intAbsTolDocstring.data())
        .def("intMaxIter", &io::SimPhysics::intMaxIter,
                intMaxIterDocstring.data())
        .def("setIntRelTol", &io::SimPhysics::setIntRelTol,
                setIntRelTolDocstring.data(), py::arg("rel_tol"))
        .def("setIntAbsTol", &io::SimPhysics::setIntAbsTol,
                setIntAbsTolDocstring.data(), py::arg("abs_tol"))
        .def("setIntMaxIter", &io::SimPhysics::setIntMaxIter,
                setIntMaxIterDocstring.data(), py::arg("max_iter"));
}
// NOLINTEND(misc-include-cleaner)
