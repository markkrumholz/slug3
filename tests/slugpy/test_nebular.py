"""
Unit tests for the Nebular Python binding (slugpy._slug.Nebular) and
SimControls's own "nebular" property.

Uses the small, committed tests/nebular/assets/nebular_test.h5 fixture
(built by data/tools/cloudy/make_nebular_test_fixture.py) via
tests/nebular/assets/testNebular.in, the same fixture the C++ unit
tests in tests/nebular/testNebular.hpp check against -- see that
file's own module comment for the exact analytic formulas every
continuum/line value below is checked against. This needs no data
fetched separately and can stay a "quick" test. Paths below resolve
whichever directory pytest is started from: deck paths are absolute,
built from REPO_ROOT (found from this file's own location -- see
test_run_sim.py's own comment), while NEBULAR_TABLE is resolved by
slug itself, which falls back to the repository directory compiled
into the module (see utils::getFilePath).

Physics-level correctness ([Fe/H]/age interpolation, line deposition,
edge zeroing, ...) is already covered by the C++ unit tests; the tests
here instead focus on the binding itself: argument order/defaults, the
mandatory (no fallback) ``controls`` argument's keep-alive behavior,
exception propagation, and SimControls's own settable "nebular"
property -- while still cross-checking a handful of values against the
fixture's own known-analytic formulas, both directly against Nebular
and via SimControls.nebular, so the binding is exercised the same way
a real caller would use it either way.

:copyright: Copyright (c) 2026 Mark Krumholz
"""

import gc
import math
import pathlib

import h5py
import numpy as np
import pytest
from slugpy._slug import Cluster, FilterIdeal, Nebular, SimControls

from slugpy import run_sim
from slugpy.slug_reader import slug_reader

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CLUSTER_DECK = str(REPO_ROOT / "tests" / "core" / "assets" / "testCluster.in")
NEBULAR_DECK = str(REPO_ROOT / "tests" / "nebular" / "assets" / "testNebular.in")
COV_FAC_UNIFORM = str(REPO_ROOT / "tests" / "nebular" / "assets" / "covFacUniform.toml")  # uniform on [0.2, 0.8]
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


def _qhi(controls: SimControls, spec: list[float], cov_fac: float = DEFAULT_COV_FAC) -> float:
    """Q(HI), discounted by cov_fac (default: the deck's own default covering factor) -- mirrors testNebular.hpp's own independent cross-check."""
    wl = controls.specsyn.wl()
    qhi_filter = FilterIdeal("Q(HI)")
    return cov_fac * qhi_filter.phot(wl, spec)


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


def test_simcontrols_nebular_is_settable(nebular_controls):
    """SimControls.nebular can be assigned a Nebular (transferring its
    ownership, exactly like setNebular() -- see its own docstring), or
    None, to remove one already present."""
    nebular_controls.nebular = None
    assert nebular_controls.nebular is None

    nebular_controls.nebular = Nebular(NEBULAR_TABLE, TRACK_NAME, nebular_controls)
    assert nebular_controls.nebular is not None
    assert len(nebular_controls.nebular.lineWl()) == N_LINE


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
    # Not the deck's own (default) covering factor, so this also checks
    # that getCluster() uses its cov_fac argument
    cov_fac = 0.3
    qhi = _qhi(nebular_controls, spec, cov_fac)

    feh, age = 0.0, 1.0e7
    spec_neb, line_lum = neb.getCluster(spec, feh, age, cov_fac)
    assert len(spec_neb) == len(wl)

    for ell in range(N_LINE):
        expected = qhi * LINE_CLUS0 * (ell + 1) * (1.0 + feh) * (age / 1.0e6)
        assert line_lum[ell] == pytest.approx(expected, rel=1e-4)


def test_get_cluster_off_grid_bilinear_interpolation(nebular_controls):
    """getCluster() at an off-grid (feh, age) pair bilinearly interpolates both axes together."""
    neb = nebular_controls.nebular
    wl = nebular_controls.specsyn.wl()
    spec = [1.0] * len(wl)
    cov_fac = 0.7
    qhi = _qhi(nebular_controls, spec, cov_fac)
    edge_wl = _edge_wl()

    feh, age = 0.25, 5.0e6
    spec_neb, line_lum = neb.getCluster(spec, feh, age, cov_fac=cov_fac)

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
    spec_neb, line_lum = neb.getCluster(spec, feh, very_old_age, DEFAULT_COV_FAC)

    assert all(lum == 0.0 for lum in line_lum)
    for k, w in enumerate(wl):
        expected = 0.0 if w <= edge_wl else spec[k]
        assert spec_neb[k] == pytest.approx(expected, abs=1e-6)


def _nebular_deck_with_cov_fac(cov_fac: str) -> str:
    """testNebular.in's text, with nebular.cov_fac set to the given (TOML-literal) value."""
    text = pathlib.Path(NEBULAR_DECK).read_text()
    return text.replace("[nebular]\n", f"[nebular]\ncov_fac = {cov_fac}\n", 1)


def test_simcontrols_cov_fac_from_deck_pdf_file() -> None:
    """nebular.cov_fac naming a PDF file sets SimControls.covFac to that distribution."""
    controls = SimControls(_nebular_deck_with_cov_fac(f'"{COV_FAC_UNIFORM}"'))
    assert controls.covFac.getMin() == pytest.approx(0.2)
    assert controls.covFac.getMax() == pytest.approx(0.8)
    assert controls.covFac.expectationValue() == pytest.approx(0.5)


def test_simcontrols_cov_fac_out_of_range_deck_raises() -> None:
    """A nebular.cov_fac outside [0, 1] raises RuntimeError at construction."""
    with pytest.raises(RuntimeError):
        SimControls(_nebular_deck_with_cov_fac("1.5"))


def test_cluster_cov_fac_nan_without_nebular() -> None:
    """Cluster.covFac() is NaN when nebular emission is off."""
    controls = SimControls(CLUSTER_DECK)
    cluster = Cluster(1e3, controls=controls)
    assert math.isnan(cluster.covFac())


def test_cluster_cov_fac_default(nebular_controls: SimControls) -> None:
    """Cluster.covFac() is the default 0.5 when nebular.cov_fac is not given."""
    cluster = Cluster(1e3, controls=nebular_controls)
    assert cluster.covFac() == 0.5


def test_cluster_cov_fac_drawn_from_distribution() -> None:
    """Each Cluster draws its own covFac() from SimControls.covFac, and
    setCovFac() changes what clusters built afterward draw."""
    controls = SimControls(_nebular_deck_with_cov_fac(f'"{COV_FAC_UNIFORM}"'))
    draws = [Cluster(1e3, uid=i, controls=controls).covFac() for i in range(20)]
    assert all(0.2 <= d <= 0.8 for d in draws)
    assert len(set(draws)) > 1

    controls.setCovFac("0.9")
    assert Cluster(1e3, controls=controls).covFac() == pytest.approx(0.9)


def _run_nebular_deck(deck_text: str, out_dir: pathlib.Path) -> slug_reader:
    """Run deck_text (with n_trial = 4) in out_dir and return a reader on its output."""
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(out_dir)
        data = run_sim("n_trial = 4\n" + deck_text, progress=False)
    assert isinstance(data, slug_reader)
    return data


def test_get_cluster_replays_cov_fac(tmp_path: pathlib.Path) -> None:
    """For a run with per-cluster covering factors, slug_reader.get_cluster()
    reproduces each cluster's recorded cov_fac and birth mass exactly."""
    run = _run_nebular_deck(_nebular_deck_with_cov_fac(f'"{COV_FAC_UNIFORM}"'), tmp_path)
    clusters = run.clusters
    assert clusters is not None
    assert "cov_fac" in clusters
    cov_fac = np.asarray(clusters["cov_fac"])
    birth_mass = np.asarray(clusters["birth_mass"])
    uids = np.asarray(clusters["uid"])
    assert len(set(cov_fac.tolist())) > 1
    for i, uid in enumerate(uids):
        cluster = run.get_cluster(int(uid))
        assert cluster.covFac() == cov_fac[i]
        assert cluster.birthMass() == birth_mass[i]


def test_get_cluster_legacy_file_without_cov_fac(tmp_path: pathlib.Path) -> None:
    """A file with no cov_fac dataset (as written before covering factors
    were drawn per cluster) whose rebuilt controls nonetheless have nebular
    emission on -- e.g. a deck that relied on compute_neb's default --
    still reproduces every cluster's birth mass exactly, with covFac() at
    the default: drawing from a delta covering factor consumes no rng
    state, so the replayed draw sequence matches the original's."""
    off_deck = pathlib.Path(NEBULAR_DECK).read_text().replace(
        "compute_neb = true", "compute_neb = false", 1)
    run = _run_nebular_deck(off_deck, tmp_path)
    clusters = run.clusters
    assert clusters is not None
    assert "cov_fac" not in clusters
    birth_mass = np.asarray(clusters["birth_mass"])
    uids = np.asarray(clusters["uid"])

    # Turn nebular emission back on in the stored input deck, so the
    # rebuilt controls have it on although the run itself did not
    on_toml = "n_trial = 4\n" + pathlib.Path(NEBULAR_DECK).read_text()
    for path in run._file:
        with h5py.File(path, "r+") as f:
            group = f["input_deck"]
            assert isinstance(group, h5py.Group)
            del group["toml"]
            group.create_dataset("toml", data=on_toml)
    patched = slug_reader(run._file[0])
    assert patched.controls.nebular is not None

    for i, uid in enumerate(uids):
        cluster = patched.get_cluster(int(uid))
        assert cluster.birthMass() == birth_mass[i]
        assert cluster.covFac() == pytest.approx(DEFAULT_COV_FAC)
