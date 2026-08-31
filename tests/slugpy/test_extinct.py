"""
Unit tests for the Extinct Python binding (slugpy._slug.Extinct) and
SimControls's own "extinct" property.

Uses the committed data/extinct/extinct.h5/.toml fixture (real Draine
2003 Milky Way / Calzetti extinction curves, small enough to be
committed directly -- see data/tools/extinct/) and the small,
committed MIST_test-based fixtures tests/core/assets/testCluster.in
and testClusterExtinct.in (the same ones test_run_sim.py and
test_readers.py rely on) rather than slug's real bundled physics deck,
so this needs no data fetched separately and can stay a "quick" test.
Run via pytest with WORKING_DIRECTORY set to the repo root (see the
test_Slugpy CMakeLists.txt target), so every path below resolves
without needing SLUG_DIR.

Physics-level correctness (interpolation, V-band normalization,
degenerate-avDistField handling, ...) is already covered by the C++
unit tests in tests/extinct/testExtinct.hpp; the tests here instead
focus on the binding itself: argument order/defaults, the optional
``controls`` argument's fallback and keep-alive behavior, exception
propagation, and SimControls's own read-only "extinct" property.

:copyright: Copyright (c) 2026 Mark Krumholz
"""

import gc
import pathlib

import h5py
import pytest

from slugpy._slug import Extinct, SimControls

REPO_ROOT = pathlib.Path.cwd()
CLUSTER_DECK = str(REPO_ROOT / "tests" / "core" / "assets" / "testCluster.in")
CLUSTER_EXTINCT_DECK = str(REPO_ROOT / "tests" / "core" / "assets" / "testClusterExtinct.in")
EXTINCT_REGISTRY = "data/extinct/extinct.toml"
EXTINCT_H5 = "data/extinct/extinct.h5"
CURVE_NAME = "Calzetti_starburst"


@pytest.fixture(scope="module")
def raw_curve():
    """The Calzetti curve's native (wavelength, kappa), read directly from the HDF5 file."""
    with h5py.File(EXTINCT_H5, "r") as f:
        grp = f[CURVE_NAME]
        return list(grp["wavelength"][:]), list(grp["kappa"][:])


@pytest.fixture
def wide_wl(raw_curve):
    """A request grid extending well past both ends of the native curve's own coverage."""
    wl_raw, _ = raw_curve
    lo, hi = wl_raw[0] - 500.0, wl_raw[-1] + 500.0
    n = 200
    step = (hi - lo) / (n - 1)
    return [lo + i * step for i in range(n)]


def test_construction_matches_raw_hdf5_data(raw_curve, wide_wl):
    """wlDat()/extinctDat() reproduce the registry's own raw data exactly."""
    wl_raw, kappa_raw = raw_curve
    ext = Extinct(CURVE_NAME, wide_wl, registry_name=EXTINCT_REGISTRY)
    assert list(ext.wlDat()) == wl_raw
    assert list(ext.extinctDat()) == kappa_raw


def test_wl_truncated_to_native_coverage(raw_curve, wide_wl):
    """wl()/extinct() are clipped to the native curve's own [min, max] coverage."""
    wl_raw, _ = raw_curve
    ext = Extinct(CURVE_NAME, wide_wl, registry_name=EXTINCT_REGISTRY)

    assert ext.wl()[0] >= wl_raw[0]
    assert ext.wl()[-1] <= wl_raw[-1]
    assert len(ext.wl()) == len(ext.extinct())

    expected_kept = sum(1 for w in wide_wl if wl_raw[0] <= w <= wl_raw[-1])
    assert len(ext.wl()) == expected_kept
    assert ext.wlOffset() == next(i for i, w in enumerate(wide_wl) if w >= wl_raw[0])


def test_default_registry_matches_explicit(wide_wl):
    """Omitting registry_name falls back to data/extinct/extinct.toml."""
    default = Extinct(CURVE_NAME, wide_wl)
    explicit = Extinct(CURVE_NAME, wide_wl, registry_name=EXTINCT_REGISTRY)
    assert list(default.wl()) == list(explicit.wl())
    assert list(default.extinct()) == list(explicit.extinct())


def test_unknown_curve_raises(wide_wl):
    """An unrecognized curve name raises RuntimeError, not a crash."""
    with pytest.raises(RuntimeError):
        Extinct("NotARealCurve", wide_wl, registry_name=EXTINCT_REGISTRY)


def test_wlobs_matches_wl_when_controls_omitted(wide_wl):
    """With no controls argument, wlObs() falls back to a minimal SimControls with z = 0."""
    ext = Extinct(CURVE_NAME, wide_wl, registry_name=EXTINCT_REGISTRY)
    assert list(ext.wlObs()) == list(ext.wl())


def test_wlobs_reads_z_live_from_controls(wide_wl):
    """wlObs() re-reads controls.z live, not a value snapshotted at construction."""
    controls = SimControls(CLUSTER_DECK)
    ext = Extinct(CURVE_NAME, wide_wl, controls=controls, registry_name=EXTINCT_REGISTRY)

    assert list(ext.wlObs()) == list(ext.wl())

    controls.z = 0.25
    wl_obs = ext.wlObs()
    assert all(wl_obs[i] == pytest.approx(ext.wl()[i] * 1.25) for i in range(len(wl_obs)))


def test_controls_keep_alive_across_del(wide_wl):
    """The Extinct built from a Python-local SimControls survives that SimControls being deleted."""
    controls = SimControls(CLUSTER_DECK)
    controls.z = 0.1
    ext = Extinct(CURVE_NAME, wide_wl, controls=controls, registry_name=EXTINCT_REGISTRY)

    del controls
    gc.collect()

    # Would crash (dangling reference) rather than raise if keep_alive
    # were missing/broken, since wlObs() reads controls_.z() live
    wl_obs = ext.wlObs()
    assert all(wl_obs[i] == pytest.approx(ext.wl()[i] * 1.1) for i in range(len(wl_obs)))


def test_apply_extinction_attenuates_flux(wide_wl):
    """applyExtinction() with A_V > 0 uniformly reduces a flat spectrum."""
    ext = Extinct(CURVE_NAME, wide_wl, registry_name=EXTINCT_REGISTRY)
    # spec must be tabulated on the *original* wl passed to the
    # constructor (wide_wl), not the truncated ext.wl() -- see
    # applyExtinction()'s own docstring
    spec = [1.0] * len(wide_wl)

    result = ext.applyExtinction(1.0, spec)
    assert len(result) == len(ext.wl())
    assert all(0.0 < value < 1.0 for value in result)


def test_apply_extinction_zero_av_is_a_no_op(wide_wl):
    """applyExtinction() with A_V = 0 leaves a spectrum unchanged (exp(0) = 1)."""
    ext = Extinct(CURVE_NAME, wide_wl, registry_name=EXTINCT_REGISTRY)
    spec = [1.0] * len(wide_wl)

    result = ext.applyExtinction(0.0, spec)
    assert all(value == pytest.approx(1.0) for value in result)


def test_apply_extinction_cts_matches_unattenuated_with_invalid_avdistfield(wide_wl):
    """With no controls (an invalid avDistField()), applyExtinctionCts() is a no-op (A_V = 0)."""
    ext = Extinct(CURVE_NAME, wide_wl, registry_name=EXTINCT_REGISTRY)
    spec = [1.0] * len(wide_wl)

    result = ext.applyExtinctionCts(spec)
    assert all(value == pytest.approx(1.0) for value in result)


def test_apply_extinction_lines_empty_without_nebular_grid(wide_wl):
    """With no nebular emission grid requested, both line-extinction methods return empty lists."""
    ext = Extinct(CURVE_NAME, wide_wl, registry_name=EXTINCT_REGISTRY)

    assert ext.applyExtinctionLines(1.0, []) == []
    assert ext.applyExtinctionCtsLines([]) == []


def test_simcontrols_extinct_property_none_without_extinct_config():
    """SimControls.extinct is None when the deck has no [extinct] section."""
    controls = SimControls(CLUSTER_DECK)
    assert controls.extinct is None


def test_simcontrols_extinct_property_returns_extinct_instance():
    """SimControls.extinct returns a real Extinct when extinct.AV/model were given."""
    controls = SimControls(CLUSTER_EXTINCT_DECK)
    ext = controls.extinct
    assert isinstance(ext, Extinct)
    assert len(ext.wl()) > 0
    assert len(ext.wl()) == len(ext.extinct())


def test_simcontrols_extinct_property_is_read_only():
    """SimControls.extinct has no setter -- assigning to it raises."""
    controls = SimControls(CLUSTER_EXTINCT_DECK)
    with pytest.raises(AttributeError):
        controls.extinct = None


def test_simcontrols_extinct_property_reads_z_live():
    """The Extinct returned by SimControls.extinct reads the same live controls.z as controls itself."""
    controls = SimControls(CLUSTER_EXTINCT_DECK)
    ext = controls.extinct

    assert list(ext.wlObs()) == list(ext.wl())
    controls.z = 0.4
    wl_obs = ext.wlObs()
    assert all(wl_obs[i] == pytest.approx(ext.wl()[i] * 1.4) for i in range(len(wl_obs)))


def test_simcontrols_extinct_property_keeps_controls_alive():
    """The Extinct returned by SimControls.extinct survives that SimControls object being deleted."""
    controls = SimControls(CLUSTER_EXTINCT_DECK)
    controls.z = 0.2
    ext = controls.extinct

    del controls
    gc.collect()

    wl_obs = ext.wlObs()
    assert all(wl_obs[i] == pytest.approx(ext.wl()[i] * 1.2) for i in range(len(wl_obs)))
