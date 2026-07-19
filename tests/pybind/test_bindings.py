"""
Unit tests for the Python bindings exposed by src/pybind/Bindings.cpp.

These tests exercise Tracks2D and Tracks3D against the small MIST_test
track set in tests/tracks/assets/tracks.toml, the same fixture used by
the C++ tests in tests/tracks, so that this suite can run without
access to the full-size track files under data/tracks. Interpolator1D
has no exposed constructor (it is only ever returned by Tracks2D and
Tracks3D methods), so it is exercised indirectly through the
Interpolator1D objects returned by getTrack() and getIsochrone().
SimPhysics and Cluster are exercised against
tests/core/assets/testCluster.in, the same fixture the C++ tests in
tests/core use, which also points at the MIST_test track set and
requests spectra.model = "blackbody".

This file is run via pytest, invoked as a CTest test from CMakeLists.txt
(see the test_PythonBindings target), so `ctest` alone runs both the
C++ and Python sides of the test suite. It requires the SLUG_DIR
environment variable to be set to the repo root (see the
test_PythonBindings target in CMakeLists.txt), since SimPhysics
resolves stars.IMF = "chabrier.toml" via SLUG_DIR + "data/imfs".
"""

import gc
import pathlib

import numpy as np
import pytest

import slug

# Registry and track set used throughout this file: a reduced, 5-group
# fixture with afe = -0.2, vvcrit = 0.0, and feh = -1.0, -0.5, -0.25,
# 0.0, and 0.5 (see tests/tracks/assets/tracks.toml and the docstrings
# in tests/tracks/testTracksAll.cpp for the same metadata used by the
# C++ tests).
REGISTRY = "tests/tracks/assets/tracks.toml"
TRACK_SET = "MIST_test"
KNOWN_FEH = -0.25
KNOWN_AFE = -0.2
KNOWN_VVCRIT = 0.0

# Masses that land exactly on the MIST_test mass grid: 0.1 (== mMin()),
# 1, 5, 20, 100, and 300 (== mMax()) Msun.
GRID_MASSES = (0.1, 1.0, 5.0, 20.0, 100.0, 300.0)

# logT values used for cross-checking getTrack() against getIsochrone();
# kept low enough to stay within every grid mass's real (non-padded)
# time coverage.
SAFE_LOG_TIMES = (0.0, 1.0, 2.0, 3.0)

# Input deck used for the SimPhysics/Cluster tests below: a cluster-type
# deck pointing at the MIST_test track set, with a fixed target mass
# (clusters.CMF = 1e3) and [Fe/H] (stars.FeH = 0.0), and
# spectra.model = "blackbody".
CLUSTER_DECK = "tests/core/assets/testCluster.in"
CLUSTER_TARGET_MASS = 1e3


@pytest.fixture(scope="module")
def tracks2d():
    """A Tracks2D object built from the known MIST_test slice at KNOWN_FEH."""
    return slug.Tracks2D(
        TRACK_SET, KNOWN_FEH, KNOWN_VVCRIT, KNOWN_AFE, registryName=REGISTRY
    )


@pytest.fixture(scope="module")
def tracks3d():
    """A Tracks3D object spanning the full MIST_test feh range."""
    return slug.Tracks3D(TRACK_SET, -1.0, 0.5, KNOWN_VVCRIT, KNOWN_AFE, REGISTRY)


@pytest.fixture(scope="module")
def sim_physics():
    """A SimPhysics object built from CLUSTER_DECK."""
    return slug.SimPhysics(CLUSTER_DECK, "cluster")


# ---------------------------------------------------------------------
# Tracks2D
# ---------------------------------------------------------------------


def test_tracks2d_getters(tracks2d):
    """feH(), aFe(), and vVcrit() should report the known fixture metadata."""
    assert tracks2d.feH() == pytest.approx(KNOWN_FEH)
    assert tracks2d.aFe() == pytest.approx(KNOWN_AFE)
    assert tracks2d.vVcrit() == pytest.approx(KNOWN_VVCRIT)
    assert tracks2d.mMin() == pytest.approx(0.1)
    assert tracks2d.mMax() == pytest.approx(300.0)
    assert tracks2d.logTMax() > 0.0


def test_tracks2d_gettrack_out_of_range_mass_raises(tracks2d):
    """getTrack() should raise, not assert-crash or silently misbehave,
    for a mass outside [mMin(), mMax()]."""
    with pytest.raises(RuntimeError):
        tracks2d.getTrack(tracks2d.mMax() + 1.0)
    with pytest.raises(RuntimeError):
        tracks2d.getTrack(tracks2d.mMin() - 1.0)


@pytest.mark.parametrize("mass", GRID_MASSES)
def test_tracks2d_gettrack_selects_correct_mass(tracks2d, mass):
    """The tabulated 'mass' field of the track returned for a given grid
    mass should closely match that mass at the earliest evaluable time,
    confirming both that getTrack() selects the right track and that
    fields are read in the correct column order."""
    track = tracks2d.getTrack(mass)
    reported_mass = track(track.xMin(), "mass")
    assert reported_mass == pytest.approx(mass, rel=1e-3)


@pytest.mark.parametrize("mass", GRID_MASSES)
@pytest.mark.parametrize("logT", SAFE_LOG_TIMES)
def test_tracks2d_gettrack_matches_isochrone(tracks2d, mass, logT):
    """getTrack(mass) evaluated at a given logT and getIsochrone(logT)
    evaluated at that mass are two different slices through the same
    underlying interpolated mesh, so they should agree."""
    track = tracks2d.getTrack(mass)
    value_from_track = track(logT, "log_L")

    isochrone = tracks2d.getIsochrone(logT)
    matching = [seg for seg in isochrone if seg.xMin() <= mass <= seg.xMax()]
    assert matching, "no isochrone segment spans the requested mass"
    value_from_isochrone = matching[0](mass, "log_L")

    assert value_from_track == pytest.approx(value_from_isochrone, abs=1e-6)


def test_tracks2d_gettrack_at_mass_boundaries(tracks2d):
    """Regression test: getTrack() at exactly mMin() or mMax() used to
    crash. yIdx()'s cell-search clamps its returned index at the upper
    mass boundary so it always refers to a valid cell, which makes the
    mass offset within that cell nonzero there (unlike every interior
    grid mass, where the offset is naturally zero); combined with a
    degenerate (collapsed) last cell -- which happens for real MIST
    data, since different masses cross a given age at very different
    rows -- this corrupted Mesh2DGrid's traversal and tripped
    Interpolator1D's monotonicity check."""
    for mass in (tracks2d.mMin(), tracks2d.mMax()):
        track = tracks2d.getTrack(mass)
        assert track(track.xMin(), "mass") == pytest.approx(mass, rel=1e-3)


def test_tracks2d_isochrone_covers_mass_range(tracks2d):
    """The union of the isochrone's segments should span the track
    set's full mass range."""
    isochrone = tracks2d.getIsochrone(1.0)
    assert len(isochrone) >= 1
    assert min(seg.xMin() for seg in isochrone) == pytest.approx(tracks2d.mMin())
    assert max(seg.xMax() for seg in isochrone) == pytest.approx(tracks2d.mMax())


def test_tracks2d_isochrone_at_time_boundaries(tracks2d):
    """Regression test: getIsochrone() at exactly logTMax() used to
    crash (an uncaught GSL abort). Only the lowest mass (0.1 Msun,
    with by far the longest lifetime in this fixture) actually reaches
    logTMax(), so the line of constant time is tangent to the mesh at
    a single point there; since an isochrone needs at least 2 points
    to interpolate over, this should now cleanly report zero segments
    rather than raising or crashing. logTMin() is the opposite
    extreme: every mass shares that sentinel starting time, so the
    isochrone there should span the full mass range in one segment."""
    isochrone_max = tracks2d.getIsochrone(tracks2d.logTMax())
    assert not isochrone_max

    isochrone_min = tracks2d.getIsochrone(tracks2d.logTMin())
    assert len(isochrone_min) == 1
    assert isochrone_min[0].xMin() == pytest.approx(tracks2d.mMin())
    assert isochrone_min[0].xMax() == pytest.approx(tracks2d.mMax())


# ---------------------------------------------------------------------
# Tracks3D
# ---------------------------------------------------------------------


def test_tracks3d_getters(tracks3d):
    """feH() should report all 5 feh values in the fixture (loading a
    3D mesh requires enough feh points to support the default spline
    type, so all of them get loaded regardless of the requested
    range); aFe()/vVcrit() should match the known fixture metadata."""
    assert list(tracks3d.feH()) == pytest.approx([-1.0, -0.5, -0.25, 0.0, 0.5])
    assert tracks3d.aFe() == pytest.approx(KNOWN_AFE)
    assert tracks3d.vVcrit() == pytest.approx(KNOWN_VVCRIT)
    assert tracks3d.mMin() == pytest.approx(0.1)
    assert tracks3d.mMax() == pytest.approx(300.0)
    assert tracks3d.logTMax() > 0.0


def test_tracks3d_gettrack_out_of_range_mass_raises(tracks3d):
    """getTrack() should raise for a mass outside [mMin(), mMax()],
    regardless of feh."""
    with pytest.raises(RuntimeError):
        tracks3d.getTrack(tracks3d.mMax() + 1.0, KNOWN_FEH)
    with pytest.raises(RuntimeError):
        tracks3d.getTrack(tracks3d.mMin() - 1.0, KNOWN_FEH)


@pytest.mark.parametrize("mass", GRID_MASSES)
def test_tracks3d_gettrack_selects_correct_mass(tracks3d, mass):
    """As with Tracks2D, the tabulated 'mass' field should closely
    match the requested grid mass at the earliest evaluable time."""
    track = tracks3d.getTrack(mass, KNOWN_FEH)
    reported_mass = track(track.xMin(), "mass")
    assert reported_mass == pytest.approx(mass, rel=1e-3)


@pytest.mark.parametrize("mass", GRID_MASSES)
@pytest.mark.parametrize("logT", SAFE_LOG_TIMES)
def test_tracks3d_gettrack_matches_isochrone(tracks3d, mass, logT):
    """Same cross-check as test_tracks2d_gettrack_matches_isochrone,
    but for the 3D (mass, logT, feh) mesh at a fixed feh."""
    track = tracks3d.getTrack(mass, KNOWN_FEH)
    value_from_track = track(logT, "log_L")

    isochrone = tracks3d.getIsochrone(logT, KNOWN_FEH)
    matching = [seg for seg in isochrone if seg.xMin() <= mass <= seg.xMax()]
    assert matching, "no isochrone segment spans the requested mass"
    value_from_isochrone = matching[0](mass, "log_L")

    assert value_from_track == pytest.approx(value_from_isochrone, abs=1e-6)


def test_tracks3d_gettrack_at_mass_boundaries(tracks3d):
    """Same regression as test_tracks2d_gettrack_at_mass_boundaries,
    for Tracks3D."""
    for mass in (tracks3d.mMin(), tracks3d.mMax()):
        track = tracks3d.getTrack(mass, KNOWN_FEH)
        assert track(track.xMin(), "mass") == pytest.approx(mass, rel=1e-3)


def test_tracks3d_isochrone_covers_mass_range(tracks3d):
    """The union of the isochrone's segments should span the track
    set's full mass range, at a fixed feh."""
    isochrone = tracks3d.getIsochrone(1.0, KNOWN_FEH)
    assert len(isochrone) >= 1
    assert min(seg.xMin() for seg in isochrone) == pytest.approx(tracks3d.mMin())
    assert max(seg.xMax() for seg in isochrone) == pytest.approx(tracks3d.mMax())


def test_tracks3d_isochrone_at_time_boundaries(tracks3d):
    """Same regression as test_tracks2d_isochrone_at_time_boundaries,
    for Tracks3D."""
    isochrone_max = tracks3d.getIsochrone(tracks3d.logTMax(), KNOWN_FEH)
    assert not isochrone_max

    isochrone_min = tracks3d.getIsochrone(tracks3d.logTMin(), KNOWN_FEH)
    assert len(isochrone_min) == 1
    assert isochrone_min[0].xMin() == pytest.approx(tracks3d.mMin())
    assert isochrone_min[0].xMax() == pytest.approx(tracks3d.mMax())


# ---------------------------------------------------------------------
# Interpolator1D (indirectly, via Tracks2D.getTrack())
# ---------------------------------------------------------------------


def test_interpolator1d_call_variants_agree(tracks2d):
    """The three __call__ overloads (full array, indexed, and named)
    should all agree with each other for the same (x, quantity)."""
    track = tracks2d.getTrack(5.0)
    x = 1.0

    full = track(x)
    by_index = track(x, 0)  # 0 == FieldIdx::mass
    by_name = track(x, "mass")

    assert full[0] == pytest.approx(by_index)
    assert by_index == pytest.approx(by_name)


def test_interpolator1d_call_out_of_range_raises(tracks2d):
    """All three __call__ overloads should raise for x outside
    [xMin(), xMax()], rather than asserting or returning junk."""
    track = tracks2d.getTrack(5.0)
    bad_x = track.xMax() + 1000.0

    with pytest.raises(RuntimeError):
        track(bad_x)
    with pytest.raises(RuntimeError):
        track(bad_x, 0)
    with pytest.raises(RuntimeError):
        track(bad_x, "mass")


def test_interpolator1d_call_unknown_name_raises(tracks2d):
    """The named-quantity __call__ overload should raise for an
    unrecognized field name."""
    track = tracks2d.getTrack(5.0)
    with pytest.raises(RuntimeError):
        track(1.0, "not_a_real_field")


def test_interpolator1d_vectorized_call(tracks2d):
    """The indexed and named __call__ overloads should broadcast
    elementwise over a numpy array of x values."""
    track = tracks2d.getTrack(5.0)
    xs = np.array([0.0, 1.0, 2.0])

    by_index = track(xs, 0)
    by_name = track(xs, "mass")

    assert by_index.shape == xs.shape
    assert by_index == pytest.approx(by_name)
    for i, x in enumerate(xs):
        assert by_index[i] == pytest.approx(track(x, "mass"))


# ---------------------------------------------------------------------
# SimPhysics
# ---------------------------------------------------------------------


def test_simphysics_construction(sim_physics):
    """Constructing from CLUSTER_DECK should succeed (the fixture
    itself would already have failed the whole module if not); this
    just documents the expectation explicitly."""
    assert sim_physics is not None


def test_simphysics_invalid_sim_type_raises():
    """sim_type must be 'cluster' or 'galaxy'."""
    with pytest.raises(RuntimeError):
        slug.SimPhysics(CLUSTER_DECK, "not_a_sim_type")


def test_simphysics_missing_file_raises():
    """A nonexistent input deck path should raise, not crash."""
    with pytest.raises(RuntimeError):
        slug.SimPhysics("tests/core/assets/does_not_exist.in", "cluster")


def test_simphysics_wl_and_wlobs(sim_physics):
    """wl() and wlObs() should return equal-length, non-empty lists;
    at z = 0 (the default SpecsynBlackbody redshift) they should be
    identical, since wlObs() is just wl() redshifted by (1 + z)."""
    wl = sim_physics.wl()
    wl_obs = sim_physics.wlObs()

    assert len(wl) > 0
    assert len(wl_obs) == len(wl)
    assert list(wl) == pytest.approx(list(wl_obs))


def test_simphysics_wl_without_specsyn_raises(tmp_path):
    """wl()/wlObs() should raise, not crash, if spectra.model was not
    set in the input deck (so SimPhysics.specsyn() is null)."""
    deck_text = pathlib.Path(CLUSTER_DECK).read_text()
    stripped = deck_text.replace('[spectra]\nmodel = "blackbody"\n\n', "")
    assert stripped != deck_text, "expected to find and strip a [spectra] section"
    deck_path = tmp_path / "no_spectra.in"
    deck_path.write_text(stripped)

    physics = slug.SimPhysics(str(deck_path), "cluster")
    with pytest.raises(RuntimeError):
        physics.wl()
    with pytest.raises(RuntimeError):
        physics.wlObs()


# ---------------------------------------------------------------------
# Cluster
# ---------------------------------------------------------------------


def test_cluster_construction(sim_physics):
    """A freshly constructed Cluster should report the uid, target
    mass, and formation time it was given; birth mass should be
    within 5% of the target (stochastic IMF sampling, same tolerance
    used by the C++ testCluster.cpp); feh should match the deck's
    fixed stars.FeH; it should not be disrupted, and should have no
    spectrum yet (advance() has not been called)."""
    cluster = slug.Cluster(1, CLUSTER_TARGET_MASS, 0.0, sim_physics)

    assert cluster.uid() == 1
    assert cluster.targetMass() == pytest.approx(CLUSTER_TARGET_MASS)
    assert cluster.birthMass() == pytest.approx(CLUSTER_TARGET_MASS, rel=0.05)
    assert cluster.formTime() == pytest.approx(0.0)
    assert cluster.feH() == pytest.approx(0.0)
    assert not cluster.isDisrupted()
    assert len(cluster.starMasses()) > 0
    assert len(cluster.deadStarMasses()) == 0
    assert len(cluster.spec()) == 0


def test_cluster_advance_populates_spec(sim_physics):
    """advance() should populate spec() (empty beforehand, since
    spectra.model = "blackbody" is set in CLUSTER_DECK) and can move
    stars from starMasses() to deadStarMasses() as the population
    ages."""
    cluster = slug.Cluster(2, CLUSTER_TARGET_MASS, 0.0, sim_physics)
    assert len(cluster.spec()) == 0

    cluster.advance(5.0)

    assert len(cluster.spec()) > 0
    assert len(cluster.starMasses()) + len(cluster.deadStarMasses()) > 0


def test_cluster_spec_matches_wl_length(sim_physics):
    """Cluster.spec() should be evaluated on the same wavelength grid
    as SimPhysics.wl()/wlObs()."""
    cluster = slug.Cluster(7, CLUSTER_TARGET_MASS, 0.0, sim_physics)
    cluster.advance(5.0)

    assert len(cluster.spec()) == len(sim_physics.wl())
    assert len(cluster.spec()) == len(sim_physics.wlObs())


def test_cluster_advance_backwards_raises(sim_physics):
    """advance() to a time before the cluster's current time should
    raise, not silently misbehave."""
    cluster = slug.Cluster(3, CLUSTER_TARGET_MASS, 0.0, sim_physics)
    cluster.advance(5.0)
    with pytest.raises(RuntimeError):
        cluster.advance(1.0)


def test_cluster_tracks_returns_tracks2d(sim_physics):
    """tracks() should return a usable Tracks2D spanning the
    MIST_test mass grid."""
    cluster = slug.Cluster(4, CLUSTER_TARGET_MASS, 0.0, sim_physics)
    cluster_tracks = cluster.tracks()

    assert cluster_tracks.mMin() == pytest.approx(0.1)
    assert cluster_tracks.mMax() == pytest.approx(300.0)


def test_cluster_tracks_reference_survives_cluster_deletion(sim_physics):
    """tracks() returns a reference tied to the owning Cluster's
    lifetime (py::return_value_policy::reference_internal); dropping
    every other reference to the Cluster should not invalidate a
    still-live Tracks2D object obtained from it."""
    cluster = slug.Cluster(5, CLUSTER_TARGET_MASS, 0.0, sim_physics)
    cluster_tracks = cluster.tracks()
    del cluster
    gc.collect()

    assert cluster_tracks.mMin() == pytest.approx(0.1)


def test_cluster_keeps_physics_alive():
    """Cluster stores only a reference to the SimPhysics it was
    constructed with, so the binding must keep_alive its physics
    argument; dropping every other reference to the SimPhysics object
    used to construct a Cluster should not leave that Cluster with a
    dangling reference."""
    physics = slug.SimPhysics(CLUSTER_DECK, "cluster")
    cluster = slug.Cluster(6, CLUSTER_TARGET_MASS, 0.0, physics)
    del physics
    gc.collect()

    # advance() reads physics_ internally; this would be a
    # use-after-free (and likely crash) if keep_alive were missing
    cluster.advance(5.0)
    assert len(cluster.spec()) > 0
