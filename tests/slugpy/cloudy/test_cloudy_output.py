"""
Unit tests for slugpy.cloudy.cloudy_output.write_cloudy_h5_results.

Exercises the HDF5 group-writing logic directly with hand-built
CloudyRunResult objects (rather than going through slug_reader.
run_cloudy and a fake cloudy subprocess), so the extensible-dataset
creation/append/wavelength-grid-growth/backfill logic can be checked
precisely and independent of the process-launching machinery (already
covered in test_run_cloudy.py's own end-to-end integration tests).
"""

import h5py
import numpy as np
import pytest
from astropy import units as u

from slugpy.cloudy.cloudy_output import CloudyRunResult, write_cloudy_h5_results


def _result(id_val, time, nII=100.0, r0=1e18, r1=2e18, U=1e-3, U0=2e-3, Omega=0.5,
    continuum=None, lines=None):
    return CloudyRunResult(
        id_val=id_val, time=time,
        nII=nII / u.cm ** 3, r0=r0 * u.cm, r1=r1 * u.cm,
        U=U * u.dimensionless_unscaled, U0=U0 * u.dimensionless_unscaled,
        Omega=Omega * u.dimensionless_unscaled, continuum=continuum, lines=lines)


def _continuum(wl_aa, inc, trans=None, emit=None, trans_emit=None):
    wl_aa = np.asarray(wl_aa, dtype=float)
    inc = np.asarray(inc, dtype=float)
    zeros = np.zeros_like(wl_aa)
    trans = zeros if trans is None else np.asarray(trans, dtype=float)
    emit = zeros if emit is None else np.asarray(emit, dtype=float)
    trans_emit = zeros if trans_emit is None else np.asarray(trans_emit, dtype=float)
    unit = u.erg / u.s / u.AA
    return (wl_aa * u.AA, inc * unit, trans * unit, emit * unit, trans_emit * unit)


def _lines(wl_aa, labels, lum):
    return (np.asarray(wl_aa, dtype=float) * u.AA, list(labels), np.asarray(lum, dtype=float) * u.erg / u.s)


@pytest.fixture
def h5_path(tmp_path):
    path = tmp_path / "test.h5"
    with h5py.File(path, "w"):
        pass
    return path


# ---------------------------------------------------------------------
# Scalar parameters, no continuum
# ---------------------------------------------------------------------

def test_creates_group_with_scalar_datasets(h5_path):
    write_cloudy_h5_results(h5_path, "cluster_cloudy", "uid", [_result(1, 1e6)])
    with h5py.File(h5_path, "r") as f:
        g = f["cluster_cloudy"]
        assert g["uid"][()].tolist() == [1]
        assert g["time"][()] == pytest.approx([1e6])
        assert g["nII"][()] == pytest.approx([100.0])
        assert g["r0"][()] == pytest.approx([1e18])
        assert g["r1"][()] == pytest.approx([2e18])
        assert g["U"][()] == pytest.approx([1e-3])
        assert g["U0"][()] == pytest.approx([2e-3])
        assert g["Omega"][()] == pytest.approx([0.5])
        assert "wl" not in g


def test_units_attributes(h5_path):
    write_cloudy_h5_results(h5_path, "cluster_cloudy", "uid", [_result(1, 1e6)])
    with h5py.File(h5_path, "r") as f:
        g = f["cluster_cloudy"]
        assert u.Unit(g["time"].attrs["units"]) == u.yr
        assert u.Unit(g["nII"].attrs["units"]) == u.cm ** -3
        assert u.Unit(g["r0"].attrs["units"]) == u.cm
        assert g["uid"].attrs["units"] == ""
        assert g["U"].attrs["units"] == ""


def test_appending_extends_rather_than_overwrites(h5_path):
    write_cloudy_h5_results(h5_path, "cluster_cloudy", "uid", [_result(1, 1e6)])
    write_cloudy_h5_results(h5_path, "cluster_cloudy", "uid", [_result(2, 2e6), _result(3, 3e6)])
    with h5py.File(h5_path, "r") as f:
        g = f["cluster_cloudy"]
        assert g["uid"][()].tolist() == [1, 2, 3]
        assert g["time"][()] == pytest.approx([1e6, 2e6, 3e6])


def test_galaxy_group_uses_trial_id_key(h5_path):
    write_cloudy_h5_results(h5_path, "galaxy_cloudy", "trial", [_result(0, 1e6)])
    with h5py.File(h5_path, "r") as f:
        assert "cluster_cloudy" not in f
        assert f["galaxy_cloudy"]["trial"][()].tolist() == [0]


def test_empty_results_is_a_noop(h5_path):
    write_cloudy_h5_results(h5_path, "cluster_cloudy", "uid", [])
    with h5py.File(h5_path, "r") as f:
        assert "cluster_cloudy" not in f


# ---------------------------------------------------------------------
# Continuum: creation, appending, no-continuum-in-batch backfill
# ---------------------------------------------------------------------

def test_continuum_created_alongside_scalars(h5_path):
    continuum = _continuum([100.0, 200.0, 300.0], [1.0, 2.0, 3.0])
    write_cloudy_h5_results(h5_path, "cluster_cloudy", "uid", [_result(1, 1e6, continuum=continuum)])
    with h5py.File(h5_path, "r") as f:
        g = f["cluster_cloudy"]
        assert g["wl"][()] == pytest.approx([100.0, 200.0, 300.0])
        assert g["spec_inc"][()] == pytest.approx(np.array([[1.0, 2.0, 3.0]]))
        assert u.Unit(g["wl"].attrs["units"]) == u.AA
        assert u.Unit(g["spec_inc"].attrs["units"]) == u.erg / u.s / u.AA


def test_result_without_continuum_in_batch_gets_zero_row(h5_path):
    """Within one call, a result with no continuum still gets a same-shape all-zero row, staying aligned with the scalar datasets."""
    continuum = _continuum([100.0, 200.0], [1.0, 2.0])
    write_cloudy_h5_results(h5_path, "cluster_cloudy", "uid",
        [_result(1, 1e6, continuum=continuum), _result(2, 2e6, continuum=None)])
    with h5py.File(h5_path, "r") as f:
        g = f["cluster_cloudy"]
        assert g["uid"][()].tolist() == [1, 2]
        assert g["spec_inc"][()] == pytest.approx(np.array([[1.0, 2.0], [0.0, 0.0]]))


def test_no_continuum_in_any_result_creates_no_continuum_datasets(h5_path):
    write_cloudy_h5_results(h5_path, "cluster_cloudy", "uid", [_result(1, 1e6), _result(2, 2e6)])
    with h5py.File(h5_path, "r") as f:
        g = f["cluster_cloudy"]
        assert "wl" not in g
        assert "spec_inc" not in g


def test_backfill_when_continuum_introduced_on_a_later_call(h5_path):
    """Rows written before continuum existed get backfilled with an all-zero row once a later call introduces it, so every dataset in the group stays the same length."""
    write_cloudy_h5_results(h5_path, "cluster_cloudy", "uid", [_result(1, 1e6), _result(2, 2e6)])
    continuum = _continuum([100.0, 200.0], [5.0, 6.0])
    write_cloudy_h5_results(h5_path, "cluster_cloudy", "uid", [_result(3, 3e6, continuum=continuum)])

    with h5py.File(h5_path, "r") as f:
        g = f["cluster_cloudy"]
        assert g["uid"][()].tolist() == [1, 2, 3]
        assert g["spec_inc"].shape == (3, 2)
        assert g["spec_inc"][()] == pytest.approx(np.array([[0.0, 0.0], [0.0, 0.0], [5.0, 6.0]]))


# ---------------------------------------------------------------------
# Wavelength grid growth (longer grid on a later write) and the
# corresponding low-wavelength zero-padding
# ---------------------------------------------------------------------

def test_shorter_new_row_padded_to_existing_grid(h5_path):
    """A later result with a SHORTER wl array than what's already stored is zero-padded at the low-wavelength (front) end to match."""
    c1 = _continuum([100.0, 200.0, 300.0], [1.0, 2.0, 3.0])
    write_cloudy_h5_results(h5_path, "cluster_cloudy", "uid", [_result(1, 1e6, continuum=c1)])
    c2 = _continuum([300.0], [9.0])  # only the shared max wavelength, spacing consistent
    write_cloudy_h5_results(h5_path, "cluster_cloudy", "uid", [_result(2, 2e6, continuum=c2)])

    with h5py.File(h5_path, "r") as f:
        g = f["cluster_cloudy"]
        assert g["wl"][()] == pytest.approx([100.0, 200.0, 300.0])
        assert g["spec_inc"][()] == pytest.approx(np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 9.0]]))


def test_longer_new_row_grows_grid_and_rewrites_existing_rows(h5_path):
    """A later result with a LONGER wl array grows the shared grid, and every already-stored row is rewritten, zero-padded at the front to stay aligned to the same (shared) maximum wavelength."""
    c1 = _continuum([200.0, 300.0], [1.0, 2.0])
    write_cloudy_h5_results(h5_path, "cluster_cloudy", "uid", [_result(1, 1e6, continuum=c1)])
    c2 = _continuum([100.0, 200.0, 300.0], [7.0, 8.0, 9.0])
    write_cloudy_h5_results(h5_path, "cluster_cloudy", "uid", [_result(2, 2e6, continuum=c2)])

    with h5py.File(h5_path, "r") as f:
        g = f["cluster_cloudy"]
        assert g["wl"][()] == pytest.approx([100.0, 200.0, 300.0])
        assert g["spec_inc"].shape == (2, 3)
        # row 0 (uid=1) rewritten: zero-padded at the front to match the new, longer grid
        assert g["spec_inc"][0, :] == pytest.approx([0.0, 1.0, 2.0])
        # row 1 (uid=2) uses the new grid directly, no padding needed
        assert g["spec_inc"][1, :] == pytest.approx([7.0, 8.0, 9.0])


def test_growth_rewrites_all_continuum_fields_not_just_spec_inc(h5_path):
    """Grid growth and the resulting shift must apply identically to spec_trans/spec_emit/spec_trans_emit, not just spec_inc."""
    c1 = _continuum([200.0, 300.0], [1.0, 2.0], trans=[10.0, 20.0], emit=[100.0, 200.0], trans_emit=[110.0, 220.0])
    write_cloudy_h5_results(h5_path, "cluster_cloudy", "uid", [_result(1, 1e6, continuum=c1)])
    c2 = _continuum([100.0, 200.0, 300.0], [7.0, 8.0, 9.0])
    write_cloudy_h5_results(h5_path, "cluster_cloudy", "uid", [_result(2, 2e6, continuum=c2)])

    with h5py.File(h5_path, "r") as f:
        g = f["cluster_cloudy"]
        assert g["spec_trans"][0, :] == pytest.approx([0.0, 10.0, 20.0])
        assert g["spec_emit"][0, :] == pytest.approx([0.0, 100.0, 200.0])
        assert g["spec_trans_emit"][0, :] == pytest.approx([0.0, 110.0, 220.0])


def test_grid_growth_within_a_single_batch(h5_path):
    """Growth logic also applies when a longer and a shorter grid both appear within the same call's own results."""
    c_short = _continuum([200.0, 300.0], [1.0, 2.0])
    c_long = _continuum([100.0, 200.0, 300.0], [7.0, 8.0, 9.0])
    write_cloudy_h5_results(h5_path, "cluster_cloudy", "uid",
        [_result(1, 1e6, continuum=c_short), _result(2, 2e6, continuum=c_long)])

    with h5py.File(h5_path, "r") as f:
        g = f["cluster_cloudy"]
        assert g["wl"][()] == pytest.approx([100.0, 200.0, 300.0])
        assert g["spec_inc"][0, :] == pytest.approx([0.0, 1.0, 2.0])
        assert g["spec_inc"][1, :] == pytest.approx([7.0, 8.0, 9.0])


# ---------------------------------------------------------------------
# Lines: shared, sorted wavelength list + padded 2-D luminosity matrix
# ---------------------------------------------------------------------

def test_lines_created_with_padded_matrix(h5_path):
    lines = _lines([1000.0, 2000.0, 3000.0], ["H  1", "He 2", "O  3"], [1e40, 2e40, 3e40])
    write_cloudy_h5_results(h5_path, "cluster_cloudy", "uid", [_result(1, 1e6, lines=lines)])
    with h5py.File(h5_path, "r") as f:
        g = f["cluster_cloudy"]
        assert g["line_wl"][()] == pytest.approx([1000.0, 2000.0, 3000.0])
        assert [x.decode() for x in g["line_label"][()]] == ["H  1", "He 2", "O  3"]
        assert g["line_lum"][()] == pytest.approx(np.array([[1e40, 2e40, 3e40]]))
        assert u.Unit(g["line_wl"].attrs["units"]) == u.AA
        assert u.Unit(g["line_lum"].attrs["units"]) == u.erg / u.s


def test_lines_sorted_by_wavelength_even_if_input_is_not(h5_path):
    """A single run's own lines are sorted by wavelength before being stored, whatever order the parser returned them in."""
    lines = _lines([3000.0, 1000.0, 2000.0], ["O  3", "H  1", "He 2"], [3e40, 1e40, 2e40])
    write_cloudy_h5_results(h5_path, "cluster_cloudy", "uid", [_result(1, 1e6, lines=lines)])
    with h5py.File(h5_path, "r") as f:
        g = f["cluster_cloudy"]
        assert g["line_wl"][()] == pytest.approx([1000.0, 2000.0, 3000.0])
        assert [x.decode() for x in g["line_label"][()]] == ["H  1", "He 2", "O  3"]
        assert g["line_lum"][()] == pytest.approx(np.array([[1e40, 2e40, 3e40]]))


def test_lines_result_without_lines_in_batch_gets_zero_row(h5_path):
    """Within one call, a result with no lines of its own still gets a same-shape all-zero row, staying aligned with the scalar datasets."""
    lines1 = _lines([1000.0], ["H  1"], [1e40])
    write_cloudy_h5_results(h5_path, "cluster_cloudy", "uid",
        [_result(1, 1e6, lines=lines1), _result(2, 2e6, lines=None)])
    with h5py.File(h5_path, "r") as f:
        g = f["cluster_cloudy"]
        assert g["line_wl"].shape == (1,)
        assert g["line_lum"][()] == pytest.approx(np.array([[1e40], [0.0]]))


def test_lines_no_lines_in_any_result_creates_no_line_datasets(h5_path):
    write_cloudy_h5_results(h5_path, "cluster_cloudy", "uid", [_result(1, 1e6), _result(2, 2e6)])
    with h5py.File(h5_path, "r") as f:
        g = f["cluster_cloudy"]
        assert "line_wl" not in g
        assert "line_lum" not in g


def test_lines_backfill_when_introduced_on_a_later_call(h5_path):
    """Rows written before line data existed get an all-zero row once a later call introduces it, so line_lum stays aligned with the scalar datasets."""
    write_cloudy_h5_results(h5_path, "cluster_cloudy", "uid", [_result(1, 1e6), _result(2, 2e6)])
    lines = _lines([500.0, 600.0], ["Ne 3", "S  2"], [5e39, 6e39])
    write_cloudy_h5_results(h5_path, "cluster_cloudy", "uid", [_result(3, 3e6, lines=lines)])

    with h5py.File(h5_path, "r") as f:
        g = f["cluster_cloudy"]
        assert g["line_wl"][()] == pytest.approx([500.0, 600.0])
        assert g["line_lum"].shape == (3, 2)
        assert g["line_lum"][()] == pytest.approx(np.array([[0.0, 0.0], [0.0, 0.0], [5e39, 6e39]]))


def test_lines_disjoint_wavelengths_across_calls_grow_and_pad(h5_path):
    """A later call whose lines don't overlap the ones already stored grows line_wl to their union, zero-padding both the already-stored row and the new one at the columns each doesn't have."""
    lines1 = _lines([100.0, 200.0], ["H  1", "He 2"], [1e40, 2e40])
    write_cloudy_h5_results(h5_path, "cluster_cloudy", "uid", [_result(1, 1e6, lines=lines1)])
    lines2 = _lines([300.0], ["O  3"], [3e40])
    write_cloudy_h5_results(h5_path, "cluster_cloudy", "uid", [_result(2, 2e6, lines=lines2)])

    with h5py.File(h5_path, "r") as f:
        g = f["cluster_cloudy"]
        assert g["line_wl"][()] == pytest.approx([100.0, 200.0, 300.0])
        assert [x.decode() for x in g["line_label"][()]] == ["H  1", "He 2", "O  3"]
        assert g["line_lum"].shape == (2, 3)
        assert g["line_lum"][()] == pytest.approx(np.array([[1e40, 2e40, 0.0], [0.0, 0.0, 3e40]]))


def test_lines_overlapping_wavelength_reuses_the_same_column(h5_path):
    """A wavelength shared between two calls lands in the same column rather than being duplicated."""
    lines1 = _lines([100.0, 200.0], ["H  1", "He 2"], [1e40, 2e40])
    write_cloudy_h5_results(h5_path, "cluster_cloudy", "uid", [_result(1, 1e6, lines=lines1)])
    lines2 = _lines([200.0, 300.0], ["He 2", "O  3"], [9e40, 3e40])
    write_cloudy_h5_results(h5_path, "cluster_cloudy", "uid", [_result(2, 2e6, lines=lines2)])

    with h5py.File(h5_path, "r") as f:
        g = f["cluster_cloudy"]
        assert g["line_wl"][()] == pytest.approx([100.0, 200.0, 300.0])
        assert g["line_lum"].shape == (2, 3)
        assert g["line_lum"][()] == pytest.approx(np.array([[1e40, 2e40, 0.0], [0.0, 9e40, 3e40]]))


def test_lines_mixed_batch_within_one_call(h5_path):
    """Within one call, results with different (and partly overlapping) line sets each land in their own row, correctly padded against the union of every wavelength in the batch."""
    lines1 = _lines([100.0, 200.0], ["H  1", "He 2"], [1e40, 2e40])
    lines2 = _lines([200.0, 300.0], ["He 2", "O  3"], [9e40, 3e40])
    write_cloudy_h5_results(h5_path, "cluster_cloudy", "uid",
        [_result(1, 1e6, lines=lines1), _result(2, 2e6, lines=lines2), _result(3, 3e6, lines=None)])
    with h5py.File(h5_path, "r") as f:
        g = f["cluster_cloudy"]
        assert g["line_wl"][()] == pytest.approx([100.0, 200.0, 300.0])
        assert g["line_lum"].shape == (3, 3)
        assert g["line_lum"][()] == pytest.approx(
            np.array([[1e40, 2e40, 0.0], [0.0, 9e40, 3e40], [0.0, 0.0, 0.0]]))
