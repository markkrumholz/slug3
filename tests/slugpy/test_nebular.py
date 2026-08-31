"""
Unit tests for the Nebular Python binding (slugpy._slug.Nebular) and
SimControls's own "nebular" property.

Uses the small, committed tests/nebular/assets/nebular_test.h5 fixture
(built by data/tools/cloudy/make_nebular_test_fixture.py) via
tests/nebular/assets/testNebular.in, the same fixture the C++ unit
tests in tests/nebular/testNebular.hpp check against -- see that
file's own module comment for the exact analytic formulas every
continuum/line value below is checked against. This needs no data
fetched separately and can stay a "quick" test. Run via pytest with
WORKING_DIRECTORY set to the repo root (see the test_Slugpy
CMakeLists.txt target), so every path below resolves without needing
SLUG_DIR.

Physics-level correctness ([Fe/H]/age interpolation, line deposition,
edge zeroing, ...) is already covered by the C++ unit tests; the tests
here instead focus on the binding itself: argument order/defaults, the
mandatory (no fallback) ``controls`` argument's keep-alive behavior,
exception propagation, and SimControls's own read-only "nebular"
property -- while still cross-checking a handful of values against the
fixture's own known-analytic formulas, both directly against Nebular
and via SimControls.nebular, so the binding is exercised the same way
a real caller would use it either way.

:copyright: Copyright (c) 2026 Mark Krumholz
"""

import gc
import pathlib

import pytest

from slugpy._slug import FilterIdeal, Nebular, SimControls

REPO_ROOT = pathlib.Path.cwd()
CLUSTER_DECK = str(REPO_ROOT / "tests" / "core" / "assets" / "testCluster.in")
NEBULAR_DECK = str(REPO_ROOT / "tests" / "nebular" / "assets" / "testNebular.in")
NEBULAR_TABLE = "tests/nebular/assets/nebular_test.h5"
TRACK_NAME = "MIST_test"

CTM_GAL0 = 1.0e-20
LINE_GAL0 = 1.0e-18
CTM_CLUS0 = 1.0e-21
LINE_CLUS0 = 1.0e-19
N_LINE = 3
WL_NATIVE_MIN = 500.0
WL_NATIVE_MAX = 20000.0
LINE_CENTERS = [4000.0, 6000.0, 9000.0]
LINE_HALF_WIDTH = 200.0
DEFAULT_COV_FAC = 0.5  # nebular::defaultCovFac; testNebular.in doesn't override it


def _qhi(controls, spec):
    """Q(HI), discounted by the default covering factor -- mirrors testNebular.hpp's own independent cross-check."""
    wl = controls.specsyn.wl()
    qhi_filter = FilterIdeal("Q(HI)")
    return DEFAULT_COV_FAC * qhi_filter.phot(wl, spec)


def _edge_wl():
    return FilterIdeal("Q(HI)").wlMax()


def _near_any_line(wl):
    return any(abs(wl - center) < LINE_HALF_WIDTH for center in LINE_CENTERS)


@pytest.fixture
def nebular_controls():
    """A SimControls built from testNebular.in, with nebular emission on."""
    return SimControls(NEBULAR_DECK)


def test_simcontrols_nebular_none_without_compute_neb():
    """SimControls.nebular is None for a deck that sets nebular.compute_neb = false."""
    controls = SimControls(CLUSTER_DECK)
    assert controls.nebular is None


def test_simcontrols_nebular_returns_instance(nebular_controls):
    """SimControls.nebular returns a real Nebular when nebular.compute_neb = true."""
    assert isinstance(nebular_controls.nebular, Nebular)


def test_simcontrols_nebular_is_read_only(nebular_controls):
    """SimControls.nebular has no setter -- assigning to it raises."""
    with pytest.raises(AttributeError):
        nebular_controls.nebular = None


def test_simcontrols_nebular_keeps_controls_alive(nebular_controls):
    """The Nebular returned by SimControls.nebular survives that SimControls being deleted."""
    neb = nebular_controls.nebular
    wl = nebular_controls.specsyn.wl()
    spec = [1.0] * len(wl)

    del nebular_controls
    gc.collect()

    # Would crash (dangling simControls_ reference) rather than return,
    # if reference_internal's keep-alive were missing/broken -- getGalaxy()
    # reads nebControls().covFac_ live at call time
    spec_neb, line_lum = neb.getGalaxy(spec, 0.0)
    assert len(spec_neb) == len(wl)
    assert len(line_lum) == N_LINE


def test_line_label_and_wl(nebular_controls):
    """lineLabel()/lineWl() match the fixture's own line_label/line_wl exactly."""
    neb = nebular_controls.nebular
    assert neb.lineLabel() == ["LINE1", "LINE2", "LINE3"]
    assert list(neb.lineWl()) == pytest.approx([4000.0, 6000.0, 9000.0])


def test_direct_construction_matches_simcontrols_nebular(nebular_controls):
    """Constructing a Nebular directly from the same table/track/controls agrees with SimControls.nebular."""
    neb_direct = Nebular(NEBULAR_TABLE, TRACK_NAME, nebular_controls)
    neb_via_sc = nebular_controls.nebular

    assert list(neb_direct.lineWl()) == list(neb_via_sc.lineWl())
    assert neb_direct.lineLabel() == neb_via_sc.lineLabel()


def test_direct_construction_vvcrit_default_matches_explicit(nebular_controls):
    """Omitting vvcrit falls back to 0.0, matching the fixture's own v/vcrit = 0 grid."""
    default = Nebular(NEBULAR_TABLE, TRACK_NAME, nebular_controls)
    explicit = Nebular(NEBULAR_TABLE, TRACK_NAME, nebular_controls, 0.0)
    assert list(default.lineWl()) == list(explicit.lineWl())


def test_direct_construction_unknown_track_raises(nebular_controls):
    """An unrecognized track name raises RuntimeError, not a crash."""
    with pytest.raises(RuntimeError):
        Nebular(NEBULAR_TABLE, "NotARealTrack", nebular_controls)


def test_direct_construction_controls_keep_alive_across_del():
    """A Nebular built from a Python-local SimControls survives that SimControls being deleted."""
    controls = SimControls(NEBULAR_DECK)
    wl = controls.specsyn.wl()
    spec = [1.0] * len(wl)

    neb = Nebular(NEBULAR_TABLE, TRACK_NAME, controls)

    del controls
    gc.collect()

    # Would crash (dangling simControls_ reference) if keep_alive were
    # missing/broken, since getGalaxy() reads specsyn()/nebControls() live
    spec_neb, line_lum = neb.getGalaxy(spec, 0.0)
    assert len(spec_neb) == len(wl)
    assert len(line_lum) == N_LINE


def test_get_galaxy_exact_grid_hit_line_luminosities(nebular_controls):
    """getGalaxy() at feh = 0.0 (an exact grid point) reproduces the fixture's own linear line-luminosity formula."""
    neb = nebular_controls.nebular
    wl = nebular_controls.specsyn.wl()
    spec = [1.0] * len(wl)
    qhi = _qhi(nebular_controls, spec)
    assert qhi > 0.0

    feh = 0.0
    _spec_neb, line_lum = neb.getGalaxy(spec, feh)
    assert len(line_lum) == N_LINE

    for ell in range(N_LINE):
        expected = qhi * LINE_GAL0 * (ell + 1) * (1.0 + feh)
        assert line_lum[ell] == pytest.approx(expected, rel=1e-4)


def test_get_galaxy_continuum_away_from_lines(nebular_controls):
    """getGalaxy()'s returned spectrum matches the fixture's own linear continuum formula, away from every line's own deposit window."""
    neb = nebular_controls.nebular
    wl = nebular_controls.specsyn.wl()
    spec = [1.0] * len(wl)
    qhi = _qhi(nebular_controls, spec)
    edge_wl = _edge_wl()

    feh = 0.0
    spec_neb, _line_lum = neb.getGalaxy(spec, feh)
    assert len(spec_neb) == len(wl)

    expected_ctm = qhi * CTM_GAL0 * (1.0 + feh)
    for k, w in enumerate(wl):
        if _near_any_line(w):
            continue
        stellar_part = spec[k] if w > edge_wl else 0.0
        in_native = WL_NATIVE_MIN <= w <= WL_NATIVE_MAX
        ctm_part = expected_ctm if in_native else 0.0
        assert spec_neb[k] == pytest.approx(stellar_part + ctm_part, rel=1e-4)


def test_get_galaxy_out_of_range_feh_raises(nebular_controls):
    """getGalaxy() raises RuntimeError for a [Fe/H] outside the fixture's own tabulated range."""
    neb = nebular_controls.nebular
    wl = nebular_controls.specsyn.wl()
    spec = [1.0] * len(wl)
    with pytest.raises(RuntimeError):
        neb.getGalaxy(spec, 10.0)


def test_get_cluster_exact_grid_hit(nebular_controls):
    """getCluster() at feh = 0, age = 1e7 yr (both exact grid points) reproduces the fixture's own linear formula."""
    neb = nebular_controls.nebular
    wl = nebular_controls.specsyn.wl()
    spec = [1.0] * len(wl)
    qhi = _qhi(nebular_controls, spec)

    feh, age = 0.0, 1.0e7
    spec_neb, line_lum = neb.getCluster(spec, feh, age)
    assert len(spec_neb) == len(wl)

    for ell in range(N_LINE):
        expected = qhi * LINE_CLUS0 * (ell + 1) * (1.0 + feh) * (age / 1.0e6)
        assert line_lum[ell] == pytest.approx(expected, rel=1e-4)


def test_get_cluster_off_grid_bilinear_interpolation(nebular_controls):
    """getCluster() at an off-grid (feh, age) pair bilinearly interpolates both axes together."""
    neb = nebular_controls.nebular
    wl = nebular_controls.specsyn.wl()
    spec = [1.0] * len(wl)
    qhi = _qhi(nebular_controls, spec)
    edge_wl = _edge_wl()

    feh, age = 0.25, 5.0e6
    spec_neb, line_lum = neb.getCluster(spec, feh, age)

    for ell in range(N_LINE):
        expected = qhi * LINE_CLUS0 * (ell + 1) * (1.0 + feh) * (age / 1.0e6)
        assert line_lum[ell] == pytest.approx(expected, rel=1e-4)

    expected_ctm = qhi * CTM_CLUS0 * (1.0 + feh) * (age / 1.0e6)
    for k, w in enumerate(wl):
        if _near_any_line(w):
            continue
        stellar_part = spec[k] if w > edge_wl else 0.0
        in_native = WL_NATIVE_MIN <= w <= WL_NATIVE_MAX
        ctm_part = expected_ctm if in_native else 0.0
        assert spec_neb[k] == pytest.approx(stellar_part + ctm_part, rel=1e-4)


def test_get_cluster_above_age_range_falls_back(nebular_controls):
    """getCluster() above the fixture's own tabulated age range returns all-zero lines and only the edge-zeroed input spectrum."""
    neb = nebular_controls.nebular
    wl = nebular_controls.specsyn.wl()
    spec = [1.0] * len(wl)
    edge_wl = _edge_wl()

    feh, very_old_age = 0.0, 1.0e9
    spec_neb, line_lum = neb.getCluster(spec, feh, very_old_age)

    assert all(lum == 0.0 for lum in line_lum)
    for k, w in enumerate(wl):
        expected = 0.0 if w <= edge_wl else spec[k]
        assert spec_neb[k] == pytest.approx(expected, abs=1e-6)
