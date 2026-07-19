/**
 * @file BindCluster.cpp
 * @author Mark Krumholz
 * @brief Python bindings for core::Cluster
 * @date 2026-07-21
 */

#include "Bindings.hpp"
#include "../core/Cluster.hpp"
#include "../io/SimPhysics.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // NOLINT(misc-include-cleaner); this is needed for correct Python binding, even if clang-tidy can't recognize it
#include <string_view>

// Numpy-style docstrings for the Python bindings below
static constexpr std::string_view constructorDocstring = R"doc(Construct a Cluster.

Parameters
----------
uid : int
    Unique identifier for this cluster.
mass : float
    Target cluster mass, in Msun.
time : float
    Cluster formation time, in yr.
physics : SimPhysics
    Simulation physics settings; physics must outlive this Cluster.)doc";

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

static constexpr std::string_view isDisruptedDocstring = R"doc(Return whether the cluster has disrupted.

Returns
-------
disrupted : bool)doc";

static constexpr std::string_view advanceDocstring = R"doc(Advance the cluster in time.

Parameters
----------
t : float
    Time to which to advance, in yr.)doc";

// Disable linting for includes -- the pybind macro magic seems to confuse
// the linter
// NOLINTBEGIN(misc-include-cleaner)
void bindCluster(py::module_& m)
{
    py::class_<core::Cluster, py::smart_holder>(m, "Cluster")
        .def(py::init<unsigned long, double, double, const io::SimPhysics&>(),
                constructorDocstring.data(),
                py::arg("uid"), py::arg("mass"), py::arg("time"), py::arg("physics"),
                // Keep the physics argument (index 5: 1 = self, 2-4 =
                // uid/mass/time) alive at least as long as this
                // Cluster, since Cluster stores only a reference to
                // it rather than its own copy
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
        .def("isDisrupted", &core::Cluster::isDisrupted,
                isDisruptedDocstring.data())
        .def("advance", &core::Cluster::advance,
                advanceDocstring.data(),
                py::arg("t"));
}
// NOLINTEND(misc-include-cleaner)
