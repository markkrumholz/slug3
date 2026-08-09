/**
 * @file BindCluster.cpp
 * @author Mark Krumholz
 * @brief Python bindings for core::Cluster
 * @date 2026-07-21
 */

#include "Bindings.hpp"
#include "../core/Cluster.hpp"
#include "../io/SimControls.hpp"
#include "../utils/RngThread.hpp"
#include <algorithm>
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h> // NOLINT(misc-include-cleaner); this is needed for correct Python binding, even if clang-tidy can't recognize it
#include <stdexcept>
#include <string>
#include <string_view>

// Numpy-style docstrings for the Python bindings below
static constexpr std::string_view constructorDocstring = R"doc(Construct a Cluster.

Parameters
----------
mass : float
    Target cluster mass, in Msun.
uid : int, optional
    Unique identifier for this cluster. Defaults to 0.
time : float, optional
    Cluster formation time, in yr. Defaults to 0.
controls : SimControls, optional
    Simulation controls (physics settings and control-flow settings
    together); controls must outlive this Cluster, and every value
    this Cluster reads from it (e.g. its integrator tolerances) is
    read live, not snapshotted, so changing controls after this
    Cluster is built takes effect immediately. Defaults to slug's own
    shared, bundled-default SimControls (built from
    src/pybind/assets/PyDefaults.toml the first time it is needed, and
    reused after that -- see SimControls()'s own default path),
    letting a caller do e.g. cl = slug.Cluster(1e3) without building a
    SimControls at all.
rng_state : bytes, optional
    Serialized pcg64 rng state (see Cluster.rngState() in the C++ API;
    from Python, typically a value previously read back from an output
    file's clusters/rng dataset) to draw this cluster's star masses
    (and feh/aV, if applicable) from, in place of the live rng stream
    -- so the resulting starMasses()/birthMass() come out bitwise
    identical to whatever cluster originally produced this state. If
    omitted, mass/feh/aV are all drawn live instead, as normal.

Throws
------
RuntimeError
    If rng_state is longer than the fixed-width buffer a serialized
    rng state is stored in (128 bytes) -- in practice, only possible
    if rng_state did not actually come from a previous Cluster's own
    rngState().)doc";

static constexpr std::string_view uidDocstring = R"doc(Return the cluster's unique identifier.

Returns
-------
uid : int)doc";

static constexpr std::string_view targetMassDocstring = R"doc(Return the cluster's target mass.

Returns
-------
target_mass : float
    Target cluster mass, in Msun.)doc";

static constexpr std::string_view birthMassDocstring = R"doc(Return the cluster's actual mass at birth.

Returns
-------
birth_mass : float
    Actual cluster mass at birth, in Msun.)doc";

static constexpr std::string_view formTimeDocstring = R"doc(Return the cluster's formation time.

Returns
-------
form_time : float
    Cluster formation time, in yr.)doc";

static constexpr std::string_view feHDocstring = R"doc(Return the cluster's [Fe/H].

Returns
-------
feh : float)doc";

static constexpr std::string_view starMassesDocstring = R"doc(Return the current list of living stellar masses.

Returns
-------
masses : list of float
    Masses of currently alive stars, in Msun.)doc";

static constexpr std::string_view deadStarMassesDocstring = R"doc(Return the list of dead stellar masses.

Returns
-------
masses : list of float
    Masses of dead stars, in Msun.)doc";

static constexpr std::string_view tracksDocstring = R"doc(Return the stellar tracks at this cluster's [Fe/H].

Returns
-------
tracks : Tracks2D)doc";

static constexpr std::string_view specDocstring = R"doc(Return the cluster's continuously-sampled spectrum.

Returns
-------
spec : list of float
    The spectrum of the non-stochastically-sampled part of the
    population, on the wavelength grid of the simulation's spectral
    synthesizer, or an empty list if no spectral synthesizer was
    requested.)doc";

static constexpr std::string_view photDocstring = R"doc(Return the cluster's photometry.

Returns
-------
phot : list of float
    The photometric value computed from spec() by each filter in
    SimControls.filters(), in the same order as
    FilterCollection.filterNames()/filterUnits(), or an empty list if
    no filter collection was requested (SimControls.filters() is
    None).)doc";

static constexpr std::string_view lbolDocstring = R"doc(Return the cluster's bolometric luminosity.

Returns
-------
lbol : float
    The population's total bolometric luminosity, in Lsun, at the
    current time, or 0 if no bolometric luminosity has ever been
    computed (SimControls.computeLbol() is False).)doc";

static constexpr std::string_view isDisruptedDocstring = R"doc(Return whether the cluster has disrupted.

Returns
-------
disrupted : bool)doc";

static constexpr std::string_view advanceDocstring = R"doc(Advance the cluster in time.

Parameters
----------
t : float
    Time to which to advance, in yr.)doc";

// Convert a Python bytes object (typically a value read straight back
// from an output file's clusters/rng dataset) to a utils::RngState --
// a fixed-width, null-padded buffer, matching the convention
// RngThread::getState() itself uses to build one
static auto pyBytesToRngState(const py::bytes& rngState) -> utils::RngState
{
    const std::string state = rngState;
    if (state.size() > utils::rngStateWidth)
    {
        throw std::runtime_error(
            "Cluster: rng_state is too long to fit in the fixed-width RngState buffer");
    }
    utils::RngState buf{};
    std::ranges::copy(state, buf.begin());
    return buf;
}

// Disable linting for includes -- the pybind macro magic seems to confuse
// the linter
// NOLINTBEGIN(misc-include-cleaner)
void bindCluster(py::module_& m)
{
    py::class_<core::Cluster, py::smart_holder>(m, "Cluster")
        .def(py::init(
                [](double mass, unsigned long uid, double time,
                   const py::object& controls, const py::object& rngState)
                    -> std::unique_ptr<core::Cluster>
                {
                    // Deliberately a ternary, not
                    // resolveControls(controls, sharedDefaultControls()):
                    // sharedDefaultControls() is expensive (parses
                    // PyDefaults.toml, the real MIST tracks, and the
                    // full spectral-library chain) and can throw, so
                    // it must only run when controls is actually
                    // py::none() -- a plain function-call argument
                    // would evaluate it eagerly on every construction
                    // regardless.
                    const io::SimControls& controlsRef = controls.is_none()
                        ? sharedDefaultControls()
                        : py::cast<const io::SimControls&>(controls);
                    if (rngState.is_none())
                    {
                        return std::make_unique<core::Cluster>(uid, mass, time, controlsRef);
                    }
                    return std::make_unique<core::Cluster>(uid, mass, time, controlsRef,
                        pyBytesToRngState(py::cast<py::bytes>(rngState)));
                }),
                constructorDocstring.data(),
                py::arg("mass"), py::arg("uid") = 0UL, py::arg("time") = 0.0,
                py::arg("controls") = py::none(), py::arg("rng_state") = py::none(),
                // Keep the controls argument (index 5: 1 = self, 2-4 =
                // mass/uid/time) alive at least as long as this
                // Cluster, since Cluster stores only a live reference
                // to it rather than copying anything out of it. When
                // controls is omitted (py::none()), this is a
                // harmless no-op -- sharedDefaultControls()'s own
                // instance is a function-local static, needing no
                // keep_alive protection at all (see its own comment
                // in Bindings.hpp).
                py::keep_alive<1, 5>())
        .def("uid", &core::Cluster::uid,
                uidDocstring.data())
        .def("targetMass", &core::Cluster::targetMass,
                targetMassDocstring.data())
        .def("birthMass", &core::Cluster::birthMass,
                birthMassDocstring.data())
        .def("formTime", &core::Cluster::formTime,
                formTimeDocstring.data())
        .def("feH", &core::Cluster::feH,
                feHDocstring.data())
        .def("starMasses", &core::Cluster::starMasses,
                starMassesDocstring.data())
        .def("deadStarMasses", &core::Cluster::deadStarMasses,
                deadStarMassesDocstring.data())
        .def("tracks", &core::Cluster::tracks,
                tracksDocstring.data(),
                py::return_value_policy::reference_internal)
        .def("spec", &core::Cluster::spec,
                specDocstring.data())
        .def("phot", &core::Cluster::phot,
                photDocstring.data())
        .def("lbol", &core::Cluster::lbol,
                lbolDocstring.data())
        .def("isDisrupted", &core::Cluster::isDisrupted,
                isDisruptedDocstring.data())
        .def("advance", &core::Cluster::advance,
                advanceDocstring.data(),
                py::arg("t"));
}
// NOLINTEND(misc-include-cleaner)
