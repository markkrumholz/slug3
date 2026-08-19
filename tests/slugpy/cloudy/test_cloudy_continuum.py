"""
Unit tests for slugpy.cloudy.cloudy_continuum.read_cloudy_continuum.

Uses a small, hand-written cloudy "save last continuum" file (matching
the real format's column layout and comment/header conventions) so
every expected value can be computed independently and checked
exactly.
"""

import astropy.units as u
import numpy as np
import pytest

from slugpy.cloudy.cloudy_continuum import read_cloudy_continuum

_C_CGS = 2.99792458e10  # cm/s, independent of the module's own constant
_RY_ERG = 2.1798723611030e-11  # 1 Rydberg in erg, independent reference value


def _make_continuum_file(tmp_path, rows):
    """rows: list of (energy_ry, incident, trans, diffout, nettrans) tuples, written in file order."""
    path = tmp_path / "test.con"
    lines = ["#Cont\tnu\tincident\ttrans\tDiffOut\tnet trans\treflc\ttotal"]
    for energy_ry, inc, trans, diffout, nettrans in rows:
        lines.append(f"{energy_ry:.8e}\t{inc:.6e}\t{trans:.6e}\t{diffout:.6e}\t{nettrans:.6e}\t0.0\t0.0")
    path.write_text("\n".join(lines) + "\n")
    return path


def test_parses_expected_number_of_rows(tmp_path):
    """One data row per non-comment line."""
    path = _make_continuum_file(tmp_path, [
        (1e-2, 1e30, 0.0, 1e30, 1e30),
        (1e-1, 2e30, 0.0, 2e30, 2e30),
        (1e0, 3e30, 0.0, 3e30, 3e30),
    ])
    wl, inc, trans, emit, trans_emit = read_cloudy_continuum(path)
    assert len(wl) == 3
    assert len(inc) == len(trans) == len(emit) == len(trans_emit) == 3


def test_wavelength_ascending_and_correct_values(tmp_path):
    """Wavelength is returned ascending, converted from Rydberg energy via E = hc/lambda."""
    energies = [1e-2, 1e-1, 1e0]
    path = _make_continuum_file(tmp_path, [(e, 1e30, 0.0, 1e30, 1e30) for e in energies])
    wl, *_ = read_cloudy_continuum(path)

    # Independent computation: wl[Angstrom] = hc/E, with E in erg
    expected_wl_cm = [(_planck_h_cgs() * _C_CGS) / (e * _RY_ERG) for e in energies]
    expected_wl_aa = sorted(w * 1e8 for w in expected_wl_cm)

    assert np.all(np.diff(wl.to_value(u.AA)) > 0)
    assert wl.to_value(u.AA) == pytest.approx(expected_wl_aa, rel=1e-4)


def _planck_h_cgs() -> float:
    return 6.62607015e-27  # erg s, independent reference value


def test_flux_conversion_matches_nu_lnu_over_wl(tmp_path):
    """L_lambda = (nu*L_nu) / wl[Angstrom], with nu*L_nu taken directly from the incident/trans/DiffOut/net-trans columns."""
    path = _make_continuum_file(tmp_path, [
        (1e-2, 5e29, 1e28, 2e29, 3e29),
        (5e-2, 6e29, 2e28, 3e29, 4e29),
    ])
    wl, inc, trans, emit, trans_emit = read_cloudy_continuum(path)

    wl_aa = wl.to_value(u.AA)
    assert inc.to_value(u.erg / u.s / u.AA) == pytest.approx(
        np.array([6e29, 5e29]) / wl_aa, rel=1e-6)
    assert trans.to_value(u.erg / u.s / u.AA) == pytest.approx(
        np.array([2e28, 1e28]) / wl_aa, rel=1e-6)
    assert emit.to_value(u.erg / u.s / u.AA) == pytest.approx(
        np.array([3e29, 2e29]) / wl_aa, rel=1e-6)
    assert trans_emit.to_value(u.erg / u.s / u.AA) == pytest.approx(
        np.array([4e29, 3e29]) / wl_aa, rel=1e-6)


def test_units_are_flux_per_wavelength(tmp_path):
    """The returned quantities are physically dimensioned as a luminosity per unit wavelength."""
    path = _make_continuum_file(tmp_path, [(1e-2, 1e30, 0.0, 1e30, 1e30)])
    wl, inc, *_ = read_cloudy_continuum(path)
    assert wl.unit.physical_type == "length"
    inc.to(u.erg / u.s / u.AA)  # doesn't raise -- confirms convertibility


def test_comment_lines_skipped(tmp_path):
    """Comment lines (# prefix) anywhere in the file, not just the header, are skipped."""
    path = tmp_path / "test.con"
    path.write_text(
        "#Cont\tnu\tincident\ttrans\tDiffOut\tnet trans\n"
        "1.0000e-02\t1.000e+30\t0.000e+00\t1.000e+30\t1.000e+30\t0.0\t0.0\n"
        "# a stray interior comment line\n"
        "1.0000e-01\t2.000e+30\t0.000e+00\t2.000e+30\t2.000e+30\t0.0\t0.0\n"
    )
    wl, *_ = read_cloudy_continuum(path)
    assert len(wl) == 2


def test_extra_trailing_columns_ignored(tmp_path):
    """Columns beyond the first 5 (reflc, total, reflin, outlin, lineID, cont, nLine in real cloudy output) don't affect parsing."""
    path = tmp_path / "test.con"
    path.write_text(
        "#Cont\tnu\tincident\ttrans\tDiffOut\tnet trans\treflc\ttotal\treflin\toutlin\tlineID\tcont\tnLine\n"
        "1.0000e-02\t1.000e+30\t0.000e+00\t1.000e+30\t1.000e+30\t0.0\t0.0\t0.0\t0.0\t    \t    \t0.00\n"
    )
    wl, inc, *_ = read_cloudy_continuum(path)
    assert len(wl) == 1
    assert inc.to_value(u.erg / u.s / u.AA) == pytest.approx(1e30 / wl.to_value(u.AA)[0], rel=1e-6)
