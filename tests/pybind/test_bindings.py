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
requests spectra.model = "blackbody". FilterIdeal, FilterTabulated,
and FilterCollection are exercised against the same small filter
registry (tests/phot/assets/filters_test.toml) the C++ tests in
tests/phot use.

This file is run via pytest, invoked as a CTest test from CMakeLists.txt
(see the test_PythonBindings target), so `ctest` alone runs both the
C++ and Python sides of the test suite. It requires the SLUG_DIR
environment variable to be set to the repo root (see the
test_PythonBindings target in CMakeLists.txt), since SimPhysics
resolves stars.IMF = "chabrier.toml" via SLUG_DIR + "data/imfs".
"""

import gc
import pathlib
import tomllib

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

# Input decks used for the Cluster.phot()/lbol() tests below.
# PHOT_DECK requests two real/idealized filters plus "Lbol"; LBOL_DECK
# requests "Lbol" alone, so SimPhysics.filters() stays None there while
# SimPhysics.computeLbol() is True (see the deck's own comment) -- both
# are otherwise identical to CLUSTER_DECK and are also used by the C++
# tests in tests/core/testCluster.cpp.
PHOT_DECK = "tests/core/assets/testClusterPhot.in"
PHOT_DECK_NFILTERS = 2  # SLUGTEST.CAM1.G500, ideal_phot_700_1500 ("Lbol" is not a filter)
LBOL_DECK = "tests/core/assets/testClusterLbol.in"

# Otherwise identical to CLUSTER_DECK, but with stars.FeH set to a
# uniform distribution over [-0.5, 0.5] rather than a fixed value, so
# SimPhysics.constFeH() is False at construction (see the deck's own
# comment) -- used by the setFeH() tests below to exercise the
# variable-to-fixed transition.
VAR_FEH_DECK = "tests/core/assets/testClusterVarFeH.in"

# slug's own bundled default deck, used by SimPhysics() when path is
# omitted/empty (see the SimPhysics tests below). Unlike every other
# deck used in this file, it references the real MIST tracks and a
# real spectral-library chain (data/tracks, data/spectra), rather than
# the small MIST_test/blackbody test fixtures -- see fetch_mist.py
# etc. -- which this test environment does not guarantee are fetched
# (CI in particular never fetches them).
PY_DEFAULTS_PATH = pathlib.Path("src/pybind/assets/PyDefaults.toml")

# Filter registry used by the FilterIdeal/FilterTabulated/FilterCollection
# tests below: a single tabulated filter, SLUGTEST.CAM1.G500 (a
# synthetic Gaussian response peaked at 5000 Angstrom), on a facility
# with exactly one instrument -- same fixture used by
# tests/phot/testFilterCollection.hpp et al.
FILTER_REGISTRY = "tests/phot/assets/filters_test.toml"


def _make_const_spec(wl_lo, wl_hi, n, f0):
    """A constant-F_lambda spectrum on a uniform grid of n points over
    [wl_lo, wl_hi], mirroring tests/phot/testFilterCollection.hpp's own
    makeConstSpec(): a constant spectrum's filter-integrated value
    doesn't depend on the filter's response shape, giving an exact
    closed form (== f0) for any energy-flux filter without needing to
    duplicate any filter's own integration machinery."""
    wl = [wl_lo + (wl_hi - wl_lo) * i / (n - 1) for i in range(n)]
    wl[-1] = wl_hi
    return wl, [f0] * n


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


@pytest.fixture(scope="module")
def sim_controls():
    """A SimControls object built from CLUSTER_DECK."""
    return slug.SimControls(CLUSTER_DECK)


@pytest.fixture(scope="module")
def phot_physics():
    """A SimPhysics object built from PHOT_DECK."""
    return slug.SimPhysics(PHOT_DECK, "cluster")


@pytest.fixture(scope="module")
def phot_controls():
    """A SimControls object built from PHOT_DECK."""
    return slug.SimControls(PHOT_DECK)


@pytest.fixture(scope="module")
def lbol_physics():
    """A SimPhysics object built from LBOL_DECK."""
    return slug.SimPhysics(LBOL_DECK, "cluster")


@pytest.fixture(scope="module")
def lbol_controls():
    """A SimControls object built from LBOL_DECK."""
    return slug.SimControls(LBOL_DECK)


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


def test_simphysics_default_deck_file_contents():
    """The bundled default deck should exist and contain the expected
    keys. Checked directly against the file (independent of whether
    slug.SimPhysics() itself can fully construct in this environment,
    see test_simphysics_default_construction_finds_bundled_deck)."""
    assert PY_DEFAULTS_PATH.is_file()
    with PY_DEFAULTS_PATH.open("rb") as f:
        deck = tomllib.load(f)

    assert deck["sim_type"] == "cluster"
    assert deck["n_trial"] == 1
    assert deck["stars"]["IMF"] == "chabrier.toml"
    assert deck["stars"]["tracks"] == "MIST"
    assert deck["stars"]["v_vcrit"] == pytest.approx(0.4)
    assert deck["stars"]["alphaFe"] == pytest.approx(0.0)
    assert deck["stars"]["FeH"] == pytest.approx(0.0)
    assert deck["spectra"]["model"] == [
        "POWR_WC", "POWR_WNE", "POWR_WNL_H20", "POWR_WNL_H40", "POWR_WNL_H60",
        "TLUSTY_O", "TLUSTY_B", "BOSZ", "CK04", "MARCS"]


def test_simphysics_default_construction_finds_bundled_deck():
    """SimPhysics() with an empty (or omitted) path should locate and
    parse the bundled default deck, rather than raising "file not
    found" the way it would for a genuinely missing path -- and,
    unlike an explicit path, should not require sim_type either. Since
    the bundled deck references real MIST/spectral-library data this
    environment does not guarantee is fetched (see PY_DEFAULTS_PATH's
    own comment), this only checks that the *deck itself* was found
    (any failure must come from further downstream, e.g. missing
    track/spectral data, not from the deck-loading step itself), not
    that construction fully succeeds everywhere this suite runs."""
    for kwargs in ({}, {"sim_type": "cluster"}, {"path": ""}):
        try:
            physics = slug.SimPhysics(**kwargs)
        except RuntimeError as e:
            assert "PyDefaults.toml" not in str(e)
        else:
            assert physics is not None


def test_simphysics_compute_lbol_default_false(sim_physics):
    """CLUSTER_DECK has no [phot] section, so computeLbol() should
    default to False."""
    assert not sim_physics.computeLbol()


def test_simphysics_set_compute_lbol_toggles():
    """setComputeLbol() should be reflected back by computeLbol(). Uses
    its own SimPhysics, rather than the shared sim_physics fixture, so
    a failed assertion here can't leave module-scoped state behind for
    other tests to trip over."""
    physics = slug.SimPhysics(CLUSTER_DECK, "cluster")
    assert not physics.computeLbol()

    physics.setComputeLbol(True)
    assert physics.computeLbol()

    physics.setComputeLbol(False)
    assert not physics.computeLbol()


def test_simphysics_set_compute_lbol_enables_lbol_computation():
    """setComputeLbol(True) on a SimPhysics built from a deck with no
    [phot] section should, on its own, be enough to make advance()
    populate a Cluster's lbol() -- the Python-only path for requesting
    Lbol output without a phot.filters deck entry."""
    physics = slug.SimPhysics(CLUSTER_DECK, "cluster")
    physics.setComputeLbol(True)
    controls = slug.SimControls(CLUSTER_DECK)
    cluster = slug.Cluster(CLUSTER_TARGET_MASS, 30, 0.0, physics, controls)
    assert cluster.lbol() == pytest.approx(0.0)

    cluster.advance(5.0)

    assert cluster.lbol() > 0.0


def test_simphysics_set_imf_numeric_reflected_in_cluster_stars():
    """setIMF() with a numeric argument should install a delta-function
    IMF; every star in a Cluster built from the resulting SimPhysics
    should then have exactly that mass, since there is no other mass
    the IMF could ever draw."""
    physics = slug.SimPhysics(CLUSTER_DECK, "cluster")
    physics.setIMF("20.0")
    controls = slug.SimControls(CLUSTER_DECK)
    cluster = slug.Cluster(CLUSTER_TARGET_MASS, 31, 0.0, physics, controls)

    assert len(cluster.starMasses()) > 0
    for mass in cluster.starMasses():
        assert mass == pytest.approx(20.0)


def test_simphysics_set_imf_file():
    """setIMF() with a file name should load an IMF from that file,
    exercising the non-numeric branch of setIMF()/initPDFFromString(),
    including its data/imfs prefix resolution (mirroring how stars.IMF
    itself is resolved when parsing an input deck)."""
    physics = slug.SimPhysics(CLUSTER_DECK, "cluster")
    physics.setIMF("chabrier.toml")
    controls = slug.SimControls(CLUSTER_DECK)
    cluster = slug.Cluster(CLUSTER_TARGET_MASS, 32, 0.0, physics, controls)

    assert len(cluster.starMasses()) > 0


def test_simphysics_set_imf_invalid_raises():
    """setIMF() with neither a numeric value nor a findable file name
    should raise, not crash."""
    physics = slug.SimPhysics(CLUSTER_DECK, "cluster")
    with pytest.raises(RuntimeError):
        physics.setIMF("not_numeric_or_a_real_file")


def test_simphysics_set_feh_rebuilds_tracks2d_cache():
    """setFeH() pinning a previously-variable [Fe/H] to a fixed value
    must rebuild tracks2D() (constFeHTracks_) from the current
    tracks_, not just fehDist_ -- otherwise Cluster (which switches to
    reading tracks2D() once constFeH() is true, see Cluster.cpp) would
    be left with a stale or invalid cache. VAR_FEH_DECK's [Fe/H] is a
    uniform distribution over [-0.5, 0.5] (constFeH() is False at
    construction, so the constructor itself never builds this cache),
    so this exercises setFeH() building the cache for the first time,
    including advancing the resulting Cluster far enough to require
    tracks2D() actually being usable."""
    physics = slug.SimPhysics(VAR_FEH_DECK, "cluster")
    physics.setFeH("-0.25")
    controls = slug.SimControls(VAR_FEH_DECK)
    cluster = slug.Cluster(CLUSTER_TARGET_MASS, 34, 0.0, physics, controls)
    assert cluster.feH() == pytest.approx(-0.25)

    cluster.advance(5.0)

    assert len(cluster.spec()) > 0
    assert all(np.isfinite(v) for v in cluster.spec())


def test_simphysics_set_feh_invalid_raises():
    """setFeH() with neither a numeric value nor a findable file name
    should raise, not crash."""
    physics = slug.SimPhysics(CLUSTER_DECK, "cluster")
    with pytest.raises(RuntimeError):
        physics.setFeH("not_numeric_or_a_real_file")


def test_simphysics_set_cmf_clf_sfr_numeric():
    """setCMF()/setCLF()/setSFR() should all accept a numeric argument
    without raising. cmf_/clf_/sfr_ are only read by simulation-driver
    machinery (SimCluster/SimGalaxy) that has no Python binding of its
    own, so this only exercises that the bindings are wired up and
    accept valid input, not their downstream effect."""
    physics = slug.SimPhysics(CLUSTER_DECK, "cluster")
    physics.setCMF("2e3")
    physics.setCLF("1e7")
    physics.setSFR("0.5")


@pytest.mark.parametrize("setter", ["setCMF", "setCLF", "setSFR"])
def test_simphysics_set_cmf_clf_sfr_invalid_raises(setter):
    """setCMF()/setCLF()/setSFR() should each raise, not crash, for a
    value that is neither numeric nor a findable file name."""
    physics = slug.SimPhysics(CLUSTER_DECK, "cluster")
    with pytest.raises(RuntimeError):
        getattr(physics, setter)("not_numeric_or_a_real_file")


def test_simphysics_set_specsyn_installs_new_synthesizer(tmp_path):
    """setSpecsyn() should install a working spectral synthesizer on a
    SimPhysics built from a deck with no spectra.model (so wl() raises
    beforehand), transferring ownership of the Python-built Specsyn
    (mirroring FilterCollection.addFilter()'s own ownership-transfer
    behavior for a directly-built Filter)."""
    deck_text = pathlib.Path(CLUSTER_DECK).read_text()
    stripped = deck_text.replace('[spectra]\nmodel = "blackbody"\n\n', "")
    assert stripped != deck_text, "expected to find and strip a [spectra] section"
    deck_path = tmp_path / "no_spectra.in"
    deck_path.write_text(stripped)

    physics = slug.SimPhysics(str(deck_path), "cluster")
    with pytest.raises(RuntimeError):
        physics.wl()

    specsyn = slug.SpecsynBlackbody(3000.0, 9000.0, 50)
    physics.setSpecsyn(specsyn)

    assert list(physics.wl()) == pytest.approx(list(physics.wlObs()))
    assert len(physics.wl()) == 50

    with pytest.raises(ValueError):
        specsyn.wl()


def test_simphysics_set_filters_installs_new_collection():
    """setFilters() should install a working FilterCollection on a
    SimPhysics built from a deck with no phot.filters (so Cluster.phot()
    stays empty), reflected in a Cluster built afterward."""
    physics = slug.SimPhysics(CLUSTER_DECK, "cluster")
    controls = slug.SimControls(CLUSTER_DECK)

    fc = slug.FilterCollection([], slug.PhotSystem.Flambda)
    fc.addFilter("Q(HI)")
    physics.setFilters(fc)

    cluster = slug.Cluster(CLUSTER_TARGET_MASS, 35, 0.0, physics, controls)
    assert len(cluster.phot()) == 0

    cluster.advance(5.0)

    assert len(cluster.phot()) == 1
    assert cluster.phot()[0] > 0.0


def test_simphysics_set_tracks_installs_new_tracks():
    """setTracks() should install new stellar tracks, transferring
    ownership of the Python-built Tracks3D, and a Cluster built
    afterward should work normally with them. CLUSTER_DECK's [Fe/H] is
    fixed (constFeH() is True), so this also exercises setTracks()'s
    own recomputation of tracks2D() (constFeHTracks_) from the new
    tracks -- the new Tracks3D spans the full MIST_test grid (unlike
    CLUSTER_DECK's own tracks_, loaded as a single degenerate slice
    at exactly [Fe/H] = 0.0, see testClusterVarFeH.in's own comment),
    so the recomputed slice at [Fe/H] = 0.0 comes from genuinely
    different (interpolated, not just copied) track data than before."""
    physics = slug.SimPhysics(CLUSTER_DECK, "cluster")
    controls = slug.SimControls(CLUSTER_DECK)

    tracks = slug.Tracks3D(TRACK_SET, -1.0, 0.5, KNOWN_VVCRIT, KNOWN_AFE, REGISTRY)
    physics.setTracks(tracks)

    with pytest.raises(ValueError):
        tracks.mMin()

    cluster = slug.Cluster(CLUSTER_TARGET_MASS, 36, 0.0, physics, controls)
    cluster.advance(5.0)

    assert len(cluster.starMasses()) + len(cluster.deadStarMasses()) > 0
    assert len(cluster.spec()) > 0
    assert all(np.isfinite(v) for v in cluster.spec())


def test_simphysics_set_min_stoch_mass_disables_stochastic_sampling():
    """setMinStochMass() with a value above the IMF's own maximum mass
    should drive the stochastic mass fraction (recomputed internally
    as imf().integral(min_stoch_mass, imf().getMax())) to zero, so no
    stars in a Cluster built from the resulting SimPhysics are drawn
    individually (starMasses() empty), while the continuously-sampled
    population still produces a valid spectrum."""
    physics = slug.SimPhysics(CLUSTER_DECK, "cluster")
    physics.setMinStochMass(1e6)
    controls = slug.SimControls(CLUSTER_DECK)

    cluster = slug.Cluster(CLUSTER_TARGET_MASS, 37, 0.0, physics, controls)
    assert len(cluster.starMasses()) == 0

    cluster.advance(5.0)

    assert len(cluster.spec()) > 0
    assert all(np.isfinite(v) for v in cluster.spec())


# ---------------------------------------------------------------------
# Cluster
# ---------------------------------------------------------------------


def test_cluster_construction(sim_physics, sim_controls):
    """A freshly constructed Cluster should report the uid, target
    mass, and formation time it was given; birth mass should be
    within 5% of the target (stochastic IMF sampling, same tolerance
    used by the C++ testCluster.cpp); feh should match the deck's
    fixed stars.FeH; it should not be disrupted, and should have no
    spectrum yet (advance() has not been called)."""
    cluster = slug.Cluster(CLUSTER_TARGET_MASS, 1, 0.0, sim_physics, sim_controls)

    assert cluster.uid() == 1
    assert cluster.targetMass() == pytest.approx(CLUSTER_TARGET_MASS)
    assert cluster.birthMass() == pytest.approx(CLUSTER_TARGET_MASS, rel=0.05)
    assert cluster.formTime() == pytest.approx(0.0)
    assert cluster.feH() == pytest.approx(0.0)
    assert not cluster.isDisrupted()
    assert len(cluster.starMasses()) > 0
    assert len(cluster.deadStarMasses()) == 0
    assert len(cluster.spec()) == 0


def test_cluster_mass_only_defaults(sim_physics, sim_controls):
    """Supplying only mass (by position) and physics/controls (by
    keyword) should leave uid and time at their documented defaults
    (0 and 0.0), confirming mass is now the first positional
    parameter and that uid/time/physics/controls are all independently
    optional."""
    cluster = slug.Cluster(CLUSTER_TARGET_MASS, physics=sim_physics, controls=sim_controls)

    assert cluster.uid() == 0
    assert cluster.targetMass() == pytest.approx(CLUSTER_TARGET_MASS)
    assert cluster.formTime() == pytest.approx(0.0)
    assert cluster.feH() == pytest.approx(0.0)  # from CLUSTER_DECK's stars.FeH


def test_cluster_fully_default_construction():
    """Cluster(mass) -- with uid, time, physics, and controls all
    omitted -- should construct a Cluster with uid() == 0,
    formTime() == 0.0, and the given mass, using slug's own shared
    default SimPhysics (built from PY_DEFAULTS_PATH). Since that deck
    references real MIST/spectral-library data this environment does
    not guarantee is fetched (see PY_DEFAULTS_PATH's own comment),
    this only checks that any failure is not from the deck-loading
    step itself, not that construction fully succeeds everywhere this
    suite runs."""
    try:
        cluster = slug.Cluster(750.0)
    except RuntimeError as e:
        assert "PyDefaults.toml" not in str(e)
        return

    assert cluster.uid() == 0
    assert cluster.targetMass() == pytest.approx(750.0)
    assert cluster.formTime() == pytest.approx(0.0)


def test_cluster_advance_populates_spec(sim_physics, sim_controls):
    """advance() should populate spec() (empty beforehand, since
    spectra.model = "blackbody" is set in CLUSTER_DECK) and can move
    stars from starMasses() to deadStarMasses() as the population
    ages."""
    cluster = slug.Cluster(CLUSTER_TARGET_MASS, 2, 0.0, sim_physics, sim_controls)
    assert len(cluster.spec()) == 0

    cluster.advance(5.0)

    assert len(cluster.spec()) > 0
    assert len(cluster.starMasses()) + len(cluster.deadStarMasses()) > 0


def test_cluster_spec_matches_wl_length(sim_physics, sim_controls):
    """Cluster.spec() should be evaluated on the same wavelength grid
    as SimPhysics.wl()/wlObs()."""
    cluster = slug.Cluster(CLUSTER_TARGET_MASS, 7, 0.0, sim_physics, sim_controls)
    cluster.advance(5.0)

    assert len(cluster.spec()) == len(sim_physics.wl())
    assert len(cluster.spec()) == len(sim_physics.wlObs())


def test_cluster_advance_backwards_raises(sim_physics, sim_controls):
    """advance() to a time before the cluster's current time should
    raise, not silently misbehave."""
    cluster = slug.Cluster(CLUSTER_TARGET_MASS, 3, 0.0, sim_physics, sim_controls)
    cluster.advance(5.0)
    with pytest.raises(RuntimeError):
        cluster.advance(1.0)


def test_cluster_tracks_returns_tracks2d(sim_physics, sim_controls):
    """tracks() should return a usable Tracks2D spanning the
    MIST_test mass grid."""
    cluster = slug.Cluster(CLUSTER_TARGET_MASS, 4, 0.0, sim_physics, sim_controls)
    cluster_tracks = cluster.tracks()

    assert cluster_tracks.mMin() == pytest.approx(0.1)
    assert cluster_tracks.mMax() == pytest.approx(300.0)


def test_cluster_tracks_reference_survives_cluster_deletion(sim_physics, sim_controls):
    """tracks() returns a reference tied to the owning Cluster's
    lifetime (py::return_value_policy::reference_internal); dropping
    every other reference to the Cluster should not invalidate a
    still-live Tracks2D object obtained from it."""
    cluster = slug.Cluster(CLUSTER_TARGET_MASS, 5, 0.0, sim_physics, sim_controls)
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
    controls = slug.SimControls(CLUSTER_DECK)
    cluster = slug.Cluster(CLUSTER_TARGET_MASS, 6, 0.0, physics, controls)
    del physics
    gc.collect()

    # advance() reads physics_ internally; this would be a
    # use-after-free (and likely crash) if keep_alive were missing
    cluster.advance(5.0)
    assert len(cluster.spec()) > 0


# ---------------------------------------------------------------------
# FilterIdeal
# ---------------------------------------------------------------------


def test_filterideal_energy_filter():
    """An ideal_energy_X_Y filter should report the parsed range and
    photCount() == False, and phot() of a constant spectrum should
    return that same constant (see _make_const_spec's own docstring)."""
    filt = slug.FilterIdeal("ideal_energy_700_1500")
    assert filt.name() == "ideal_energy_700_1500"
    assert not filt.photCount()
    assert filt.wlMin() == pytest.approx(700.0)
    assert filt.wlMax() == pytest.approx(1500.0)
    assert filt.wlPivot() == pytest.approx(1100.0)

    wl, spec = _make_const_spec(500.0, 2000.0, 2000, 3.5)
    assert filt.phot(wl, spec) == pytest.approx(3.5, rel=1e-6)


def test_filterideal_phot_filter():
    """An ideal_phot_X_Y filter should report photCount() == True and a
    positive photon-count rate for a positive spectrum."""
    filt = slug.FilterIdeal("ideal_phot_700_1500")
    assert filt.photCount()
    wl, spec = _make_const_spec(500.0, 2000.0, 2000, 3.5)
    assert filt.phot(wl, spec) > 0.0


def test_filterideal_ionization_threshold():
    """A Q(<elem><ion>) filter should have wlMax() == inf and a finite,
    positive wlMin() set from the ionization threshold, with
    photCount() == True."""
    filt = slug.FilterIdeal("Q(HI)")
    assert filt.photCount()
    assert filt.wlMin() > 0.0
    assert np.isinf(filt.wlMax())
    assert filt.wlPivot() == pytest.approx(filt.wlMin())


def test_filterideal_invalid_name_raises():
    """A name matching neither recognized convention should raise, not
    crash."""
    with pytest.raises(RuntimeError):
        slug.FilterIdeal("not_a_valid_name")


# ---------------------------------------------------------------------
# FilterTabulated
# ---------------------------------------------------------------------


def test_filtertabulated_registry_constructor():
    """The registry constructor should resolve SLUGTEST.CAM1.G500 from
    FILTER_REGISTRY, reporting the full facility.instrument.filter name,
    photCount() == False, and a finite, positive wlPivot()/norm()."""
    filt = slug.FilterTabulated("SLUGTEST", "CAM1", "G500", FILTER_REGISTRY)
    assert filt.name() == "SLUGTEST.CAM1.G500"
    assert not filt.photCount()
    assert filt.wlPivot() == pytest.approx(5000.0)
    assert filt.norm() > 0.0


def test_filtertabulated_direct_constructor():
    """The direct (name, wl, response, wl_pivot) constructor should
    round-trip its wl/response arrays through wl()/responseData(), and
    report the supplied wlPivot()."""
    wl = [1000.0, 2000.0, 3000.0]
    response = [0.0, 1.0, 0.0]
    filt = slug.FilterTabulated("my_filter", wl, response, 2000.0)

    assert filt.name() == "my_filter"
    assert not filt.photCount()
    assert filt.wlPivot() == pytest.approx(2000.0)
    assert list(filt.wl()) == pytest.approx(wl)
    assert list(filt.responseData()) == pytest.approx(response)


def test_filtertabulated_response_interpolator():
    """response() should return an Interpolator1DScalar spanning
    ln(wl()) that evaluates to responseData()'s peak value at the
    corresponding ln(wavelength)."""
    wl = [1000.0, 2000.0, 3000.0]
    response = [0.0, 1.0, 0.0]
    filt = slug.FilterTabulated("my_filter", wl, response, 2000.0)

    interp = filt.response()
    assert interp.xMin() == pytest.approx(np.log(1000.0))
    assert interp.xMax() == pytest.approx(np.log(3000.0))
    assert interp(np.log(2000.0)) == pytest.approx(1.0)


def test_filtertabulated_phot_matches_const_spectrum():
    """phot() of a constant spectrum should return that same constant,
    same closed-form argument as test_filterideal_energy_filter."""
    filt = slug.FilterTabulated("SLUGTEST", "CAM1", "G500", FILTER_REGISTRY)
    wl, spec = _make_const_spec(2000.0, 8000.0, 5000, 3.5)
    assert filt.phot(wl, spec) == pytest.approx(3.5, rel=1e-6)


def test_filtertabulated_unknown_facility_raises():
    """An unrecognized facility should raise, not crash."""
    with pytest.raises(RuntimeError):
        slug.FilterTabulated("BOGUS", "CAM1", "G500", FILTER_REGISTRY)


# ---------------------------------------------------------------------
# FilterCollection
# ---------------------------------------------------------------------


def test_filtercollection_mixed_construction():
    """A FilterCollection built from one tabulated and three idealized
    filter names should report filterNames()/filterUnits() in the
    supplied order, and phot() should agree with the same filters'
    own phot(), matching testFilterCollectionMixedConstruction in
    tests/phot/testFilterCollection.hpp."""
    names = ["SLUGTEST.CAM1.G500", "ideal_energy_700_1500", "ideal_phot_700_1500", "Q(HI)"]
    fc = slug.FilterCollection(names, slug.PhotSystem.Flambda, FILTER_REGISTRY)

    assert fc.filterNames() == names
    assert fc.filterUnits() == [
        "erg/s/Angstrom", "erg/s/Angstrom", "photons/s", "photons/s"]

    wl, spec = _make_const_spec(500.0, 9000.0, 5000, 3.5)

    ref_tab = slug.FilterTabulated("SLUGTEST", "CAM1", "G500", FILTER_REGISTRY)
    ref_energy = slug.FilterIdeal("ideal_energy_700_1500")
    ref_phot = slug.FilterIdeal("ideal_phot_700_1500")
    ref_qhi = slug.FilterIdeal("Q(HI)")
    expected = [
        ref_tab.phot(wl, spec), ref_energy.phot(wl, spec),
        ref_phot.phot(wl, spec), ref_qhi.phot(wl, spec)]

    assert list(fc.phot(wl, spec)) == pytest.approx(expected, rel=1e-9)


def test_filtercollection_phot_system_conversion():
    """filterUnits() should report the requested photSystem's unit for
    every energy-flux filter, leaving photon-count filters
    unconverted."""
    names = ["SLUGTEST.CAM1.G500", "ideal_phot_700_1500"]
    fc = slug.FilterCollection(names, slug.PhotSystem.AB, FILTER_REGISTRY)
    assert fc.filterUnits() == ["ABmag", "photons/s"]


def test_filtercollection_instrument_omitted():
    """FILTER_REGISTRY's SLUGTEST facility has exactly one instrument
    (CAM1), so the instrument-omitted "SLUGTEST.G500" name should
    resolve to the same filter as "SLUGTEST.CAM1.G500"."""
    fc = slug.FilterCollection(["SLUGTEST.G500"], slug.PhotSystem.Flambda, FILTER_REGISTRY)
    assert fc.filterNames() == ["SLUGTEST.CAM1.G500"]


def test_filtercollection_invalid_name_raises():
    """An unparseable filter name should raise, not crash."""
    with pytest.raises(RuntimeError):
        slug.FilterCollection(["not_a_valid_name"], slug.PhotSystem.Flambda, FILTER_REGISTRY)


def test_filtercollection_getfilter_by_index_downcasts():
    """getFilter(i) should hand back each filter as its actual
    concrete type (FilterTabulated or FilterIdeal), not a bare Filter
    -- pybind11's automatic polymorphic downcasting, since FilterIdeal
    and FilterTabulated are both registered as Filter subclasses."""
    names = ["SLUGTEST.CAM1.G500", "ideal_phot_700_1500", "Q(HI)"]
    fc = slug.FilterCollection(names, slug.PhotSystem.Flambda, FILTER_REGISTRY)

    tab = fc.getFilter(0)
    assert isinstance(tab, slug.FilterTabulated)
    assert isinstance(tab, slug.Filter)
    assert tab.name() == "SLUGTEST.CAM1.G500"
    assert tab.norm() > 0.0  # FilterTabulated-only method

    ideal = fc.getFilter(1)
    assert isinstance(ideal, slug.FilterIdeal)
    assert ideal.name() == "ideal_phot_700_1500"
    assert ideal.wlMin() == pytest.approx(700.0)  # FilterIdeal-only method


def test_filtercollection_getfilter_by_name():
    """getFilter(name) should return the filter whose name() exactly
    matches, agreeing with the same filter fetched by index."""
    names = ["SLUGTEST.CAM1.G500", "ideal_phot_700_1500", "Q(HI)"]
    fc = slug.FilterCollection(names, slug.PhotSystem.Flambda, FILTER_REGISTRY)

    by_name = fc.getFilter("Q(HI)")
    by_index = fc.getFilter(2)
    assert type(by_name) is type(by_index)
    assert by_name.name() == by_index.name() == "Q(HI)"


def test_filtercollection_getfilter_bad_index_raises_indexerror():
    """An out-of-range index should raise IndexError specifically, not
    just any RuntimeError."""
    fc = slug.FilterCollection(["Q(HI)"], slug.PhotSystem.Flambda, FILTER_REGISTRY)
    with pytest.raises(IndexError):
        fc.getFilter(1)


def test_filtercollection_getfilter_bad_name_raises_runtimeerror():
    """A name with no matching filter should raise RuntimeError, not
    IndexError."""
    fc = slug.FilterCollection(["Q(HI)"], slug.PhotSystem.Flambda, FILTER_REGISTRY)
    with pytest.raises(RuntimeError):
        fc.getFilter("not_in_this_collection")


def test_filtercollection_filters_matches_names_and_order():
    """filters() should return one filter per filterNames() entry, in
    the same order, each downcast to its concrete type, agreeing with
    the same filters fetched one at a time via getFilter()."""
    names = ["SLUGTEST.CAM1.G500", "ideal_energy_700_1500", "ideal_phot_700_1500", "Q(HI)"]
    fc = slug.FilterCollection(names, slug.PhotSystem.Flambda, FILTER_REGISTRY)

    filts = fc.filters()
    assert [f.name() for f in filts] == names
    assert [type(f).__name__ for f in filts] == [
        "FilterTabulated", "FilterIdeal", "FilterIdeal", "FilterIdeal"]
    for i, filt in enumerate(filts):
        assert type(filt) is type(fc.getFilter(i))
        assert filt.name() == fc.getFilter(i).name()


def test_filtercollection_filter_outlives_collection():
    """A filter obtained from getFilter()/filters() must stay usable
    even after every other reference to its owning FilterCollection is
    dropped -- FilterCollection owns the underlying Filter objects, so
    the binding must keep the collection alive as long as any filter
    obtained from it survives (mirrors
    test_cluster_tracks_reference_survives_cluster_deletion)."""
    fc = slug.FilterCollection(["Q(HI)"], slug.PhotSystem.Flambda, FILTER_REGISTRY)
    filt = fc.getFilter(0)
    filts = fc.filters()
    del fc
    gc.collect()

    assert filt.name() == "Q(HI)"
    assert filts[0].name() == "Q(HI)"


def test_filtercollection_addfilter_by_name_matches_constructor():
    """Building a FilterCollection incrementally via addFilter(name)
    should be equivalent to passing the same names to the constructor
    up front: same filterNames() and same phot() values."""
    names = ["SLUGTEST.CAM1.G500", "ideal_phot_700_1500", "Q(HI)"]

    fc_ctor = slug.FilterCollection(names, slug.PhotSystem.Flambda, FILTER_REGISTRY)

    fc_incremental = slug.FilterCollection([], slug.PhotSystem.Flambda)
    for name in names:
        fc_incremental.addFilter(name, FILTER_REGISTRY)

    assert fc_incremental.filterNames() == fc_ctor.filterNames() == names

    wl, spec = _make_const_spec(500.0, 9000.0, 5000, 3.5)
    assert list(fc_incremental.phot(wl, spec)) == pytest.approx(
        list(fc_ctor.phot(wl, spec)), rel=1e-9)


def test_filtercollection_addfilter_by_name_invalid_raises():
    """addFilter(name) should raise, not crash, for an unparseable name."""
    fc = slug.FilterCollection([], slug.PhotSystem.Flambda)
    with pytest.raises(RuntimeError):
        fc.addFilter("not_a_valid_name")


def test_filtercollection_addfilter_direct():
    """addFilter(filter) should accept a directly-constructed
    FilterIdeal or FilterTabulated and append it, preserving each
    filter's own concrete type and values."""
    fc = slug.FilterCollection([], slug.PhotSystem.Flambda)

    ideal = slug.FilterIdeal("Q(HeI)")
    fc.addFilter(ideal)

    tab = slug.FilterTabulated("SLUGTEST", "CAM1", "G500", FILTER_REGISTRY)
    fc.addFilter(tab)

    assert fc.filterNames() == ["Q(HeI)", "SLUGTEST.CAM1.G500"]
    assert isinstance(fc.getFilter(0), slug.FilterIdeal)
    assert isinstance(fc.getFilter(1), slug.FilterTabulated)
    assert fc.getFilter(1).norm() > 0.0


def test_filtercollection_addfilter_direct_transfers_ownership():
    """addFilter(filter) transfers filter's underlying C++ object into
    the FilterCollection; the Python wrapper that was passed in should
    no longer be usable afterwards, rather than silently aliasing an
    object now owned (and possibly later destroyed alongside) the
    collection."""
    fc = slug.FilterCollection([], slug.PhotSystem.Flambda)
    ideal = slug.FilterIdeal("Q(HeI)")
    fc.addFilter(ideal)

    with pytest.raises(ValueError):
        ideal.name()

    # the collection's own copy is unaffected
    assert fc.getFilter(0).name() == "Q(HeI)"


def test_cluster_phot_empty_without_filters(sim_physics, sim_controls):
    """CLUSTER_DECK has no [phot] section, so phot() should stay empty
    even after advance()."""
    cluster = slug.Cluster(CLUSTER_TARGET_MASS, 8, 0.0, sim_physics, sim_controls)
    cluster.advance(5.0)
    assert len(cluster.phot()) == 0


def test_cluster_advance_populates_phot(phot_physics, phot_controls):
    """phot() should be empty before advance() and, afterwards, should
    hold one positive value per non-"Lbol" filter in PHOT_DECK's
    phot.filters."""
    cluster = slug.Cluster(CLUSTER_TARGET_MASS, 9, 0.0, phot_physics, phot_controls)
    assert len(cluster.phot()) == 0

    cluster.advance(5.0)

    assert len(cluster.phot()) == PHOT_DECK_NFILTERS
    assert all(value > 0.0 for value in cluster.phot())


def test_cluster_lbol_zero_before_advance(lbol_physics, lbol_controls):
    """lbol() should be 0 for a freshly constructed Cluster, before
    advance() has ever run."""
    cluster = slug.Cluster(CLUSTER_TARGET_MASS, 10, 0.0, lbol_physics, lbol_controls)
    assert cluster.lbol() == pytest.approx(0.0)


def test_cluster_advance_populates_lbol(lbol_physics, lbol_controls):
    """advance() should populate lbol() with a finite, positive value
    when computeLbol() is True -- LBOL_DECK sets stars.min_stoch_mass,
    so this exercises both the stochastic and continuously-sampled
    code paths in Cluster's bolometric-luminosity computation."""
    assert lbol_physics.computeLbol()
    cluster = slug.Cluster(CLUSTER_TARGET_MASS, 11, 0.0, lbol_physics, lbol_controls)

    cluster.advance(5.0)

    assert np.isfinite(cluster.lbol())
    assert cluster.lbol() > 0.0


# ---------------------------------------------------------------------
# PhotConvert
# ---------------------------------------------------------------------


@pytest.mark.parametrize("system", ["Flambda", "Fnu", "ST", "AB", "Vega"])
def test_photconvert_identity(system):
    """Converting a system to itself should return flux_in unchanged,
    even for Vega (where no filter is needed, since the identity case
    is handled before fluxVega() would ever be read)."""
    assert slug.PhotConvert(system, system, 5.0, 1000.0) == pytest.approx(5.0)


def test_photconvert_flambda_fnu_roundtrip():
    """Flambda -> Fnu -> Flambda should round-trip to the original value."""
    flambda_in = 1e-15
    wl = 5000.0
    fnu = slug.PhotConvert("Flambda", "Fnu", flambda_in, wl)
    back = slug.PhotConvert("Fnu", "Flambda", fnu, wl)
    assert back == pytest.approx(flambda_in, rel=1e-9)


@pytest.mark.parametrize("phot_to", ["ST", "AB"])
def test_photconvert_flambda_magnitude_roundtrip(phot_to):
    """Flambda -> {ST, AB} -> Flambda should round-trip, exercising the
    magnitude-system conversions that don't need a filter."""
    flambda_in = 1e-15
    wl = 5000.0
    mag = slug.PhotConvert("Flambda", phot_to, flambda_in, wl)
    back = slug.PhotConvert(phot_to, "Flambda", mag, wl)
    assert back == pytest.approx(flambda_in, rel=1e-9)


def test_photconvert_unknown_system_raises():
    """An unrecognized phot_from/phot_to should raise, not crash."""
    with pytest.raises(RuntimeError):
        slug.PhotConvert("bogus", "AB", 1.0, 1000.0)
    with pytest.raises(RuntimeError):
        slug.PhotConvert("AB", "bogus", 1.0, 1000.0)


def test_photconvert_vega_without_filter_raises():
    """Converting to or from Vega without a filter should raise."""
    with pytest.raises(RuntimeError):
        slug.PhotConvert("Flambda", "Vega", 1e-15, 5000.0)
    with pytest.raises(RuntimeError):
        slug.PhotConvert("Vega", "Flambda", 10.0, 5000.0)


def test_photconvert_vega_matches_filter_fluxvega():
    """Flambda -> Vega should match -2.5*log10(flux_in / filter.fluxVega()),
    the definition of a Vega magnitude, and round-trip back to
    flux_in. This also exercises fluxVega()'s lazy computation being
    triggered as a side effect of the conversion."""
    filt = slug.FilterTabulated("SLUGTEST", "CAM1", "G500", FILTER_REGISTRY)
    flambda_in = 1e-15
    wl = filt.wlPivot()

    vegamag = slug.PhotConvert("Flambda", "Vega", flambda_in, wl, filt)

    assert filt.fluxVega() > 0.0
    assert vegamag == pytest.approx(-2.5 * np.log10(flambda_in / filt.fluxVega()))

    back = slug.PhotConvert("Vega", "Flambda", vegamag, wl, filt)
    assert back == pytest.approx(flambda_in, rel=1e-9)


def test_photconvert_vega_photcount_filter_gives_negative_infinity():
    """A photCount() filter's fluxVega() is always 0 (see
    Filter.fluxVega()); PhotConvert should still accept it (it just
    forwards fluxVega() through, without raising), giving
    -2.5*log10(flux_in / 0) == -inf -- a well-defined, if degenerate,
    IEEE double rather than a crash or NaN."""
    filt = slug.FilterIdeal("Q(HI)")
    assert filt.photCount()

    result = slug.PhotConvert("Flambda", "Vega", 1e-15, filt.wlPivot(), filt)
    assert result == float("-inf")
