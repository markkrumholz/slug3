/**
 * @file BindGalaxy.cpp
 * @author Mark Krumholz
 * @brief Python bindings for core::Galaxy
 * @date 2026-08-10
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#include "Bindings.hpp"
#include "../core/Cluster.hpp"
#include "../core/Galaxy.hpp"
#include "../io/SimControls.hpp"
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // NOLINT(misc-include-cleaner); this is needed for correct Python binding, even if clang-tidy can't recognize it
#include <string_view>
#include <vector>

// Numpy-style docstrings for the Python bindings below
static constexpr std::string_view constructorDocstring = R"doc(Construct a Galaxy.

Parameters
----------
controls : SimControls, optional
    Simulation controls (physics settings and control-flow settings
    together); controls must outlive this Galaxy, and every value this
    Galaxy reads from it (e.g. sfr()/cmf()) is read live, not
    snapshotted, so changing controls after this Galaxy is built takes
    effect the next time advance() runs. Defaults to slug's own
    shared, bundled-default SimControls (built from
    src/pybind/assets/PyDefaults.toml the first time it is needed, and
    reused after that -- see SimControls()'s own default path),
    letting a caller do e.g. gal = slug.Galaxy() without building a
    SimControls at all.

Details
-------
curTime()/lbol()/targetMass()/actualMass() all start at 0, and
clusters()/disruptedClusters()/spec()/specExtinct()/phot()/
photExtinct() all start empty -- no clusters exist, and no
spectrum/photometry has been computed, until the first call to
advance().)doc";

static constexpr std::string_view curTimeDocstring = R"doc(Return the galaxy's current time.

Returns
-------
cur_time : float
    Current simulation time, in yr.)doc";

static constexpr std::string_view clustersDocstring = R"doc(Return the galaxy's currently-alive (non-disrupted) clusters.

Returns
-------
clusters : list of Cluster
    The clusters formed so far that have not yet disrupted. Each
    Cluster in the returned list is a live reference into this Galaxy
    (not a copy), so mutating one (e.g. via advance()) is reflected
    here, and the returned Clusters must not outlive this Galaxy.)doc";

static constexpr std::string_view disruptedClustersDocstring = R"doc(Return the galaxy's disrupted clusters.

Returns
-------
clusters : list of Cluster
    The clusters formed so far that have already disrupted -- see
    clusters()'s own comment on the returned Clusters' lifetime.)doc";

static constexpr std::string_view specDocstring = R"doc(Return the galaxy's continuously-sampled spectrum.

Returns
-------
spec : list of float
    The sum of spec() over every cluster in clusters() and
    disruptedClusters(), on the wavelength grid of the simulation's
    spectral synthesizer, or an empty list if no spectral synthesizer
    was requested.)doc";

static constexpr std::string_view specExtinctDocstring = R"doc(Return the galaxy's extincted spectrum.

Returns
-------
spec_extinct : list of float
    The sum of specExtinct() over every cluster in clusters() and
    disruptedClusters(); an empty list if no extinction curve was
    requested, or spec() itself is empty.)doc";

static constexpr std::string_view photDocstring = R"doc(Return the galaxy's photometry.

Returns
-------
phot : list of float
    The photometric value computed from spec() by each filter in
    SimControls.filters(), in the same order as
    FilterCollection.filterNames()/filterUnits(), or an empty list if
    no filter collection was requested.)doc";

static constexpr std::string_view photExtinctDocstring = R"doc(Return the galaxy's extincted photometry.

Returns
-------
phot_extinct : list of float
    The photometric value computed from specExtinct() by each filter
    in SimControls.filters(), in the same order as phot(); an empty
    list if no extinction curve or filter collection was requested.)doc";

static constexpr std::string_view lbolDocstring = R"doc(Return the galaxy's bolometric luminosity.

Returns
-------
lbol : float
    The sum of lbol() over every cluster in clusters() and
    disruptedClusters(), in Lsun, at the current time, or 0 if no
    bolometric luminosity has ever been computed
    (SimControls.computeLbol() is False).)doc";

static constexpr std::string_view targetMassDocstring = R"doc(Return the total target mass of clusters formed so far.

Returns
-------
target_mass : float
    The sum, over every advance() call so far, of the target mass of
    clusters that should have formed, in Msun -- may differ from
    actualMass() due to stochastic sampling from SimControls.cmf().)doc";

static constexpr std::string_view actualMassDocstring = R"doc(Return the total actual mass of clusters formed so far.

Returns
-------
actual_mass : float
    The sum, over every advance() call so far, of the target mass of
    every cluster actually drawn from SimControls.cmf() during that
    call, in Msun.)doc";

static constexpr std::string_view advanceDocstring = R"doc(Advance the galaxy in time.

Parameters
----------
t : float
    Time to which to advance, in yr; must be >= curTime().)doc";

// Disable linting for includes -- the pybind macro magic seems to confuse
// the linter
// NOLINTBEGIN(misc-include-cleaner)
void bindGalaxy(py::module_& m)
{
    py::class_<core::Galaxy, py::smart_holder>(m, "Galaxy")
        .def(py::init(
                [](const py::object& controls) -> std::unique_ptr<core::Galaxy>
                {
                    // See BindCluster.cpp's own identical comment on
                    // why this is a ternary, not
                    // resolveControls(controls, sharedDefaultControls()).
                    const io::SimControls& controlsRef = controls.is_none()
                        ? sharedDefaultControls()
                        : py::cast<const io::SimControls&>(controls);
                    return std::make_unique<core::Galaxy>(controlsRef);
                }),
                constructorDocstring.data(),
                py::arg("controls") = py::none(),
                // Keep the controls argument (index 2: 1 = self) alive
                // at least as long as this Galaxy -- see BindCluster.cpp's
                // own identical comment.
                py::keep_alive<1, 2>())
        .def("curTime", &core::Galaxy::curTime,
                curTimeDocstring.data())
        .def("clusters",
                [](core::Galaxy& self) -> std::vector<core::Cluster*>
                {
                    std::vector<core::Cluster*> result;
                    result.reserve(self.clusters().size());
                    for (auto& cluster : self.clusters()) { result.push_back(&cluster); }
                    return result;
                },
                clustersDocstring.data(),
                py::return_value_policy::reference_internal)
        .def("disruptedClusters",
                [](core::Galaxy& self) -> std::vector<core::Cluster*>
                {
                    std::vector<core::Cluster*> result;
                    result.reserve(self.disruptedClusters().size());
                    for (auto& cluster : self.disruptedClusters()) { result.push_back(&cluster); }
                    return result;
                },
                disruptedClustersDocstring.data(),
                py::return_value_policy::reference_internal)
        .def("spec", &core::Galaxy::spec,
                specDocstring.data())
        .def("specExtinct", &core::Galaxy::specExtinct,
                specExtinctDocstring.data())
        .def("phot", &core::Galaxy::phot,
                photDocstring.data())
        .def("photExtinct", &core::Galaxy::photExtinct,
                photExtinctDocstring.data())
        .def("lbol", &core::Galaxy::lbol,
                lbolDocstring.data())
        .def("targetMass", &core::Galaxy::targetMass,
                targetMassDocstring.data())
        .def("actualMass", &core::Galaxy::actualMass,
                actualMassDocstring.data())
        .def("advance", &core::Galaxy::advance,
                advanceDocstring.data(),
                py::arg("t"));
}
// NOLINTEND(misc-include-cleaner)
