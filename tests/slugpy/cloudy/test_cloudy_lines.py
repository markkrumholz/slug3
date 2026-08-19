"""
Unit tests for slugpy.cloudy.cloudy_lines.read_cloudy_linearr.

Uses a small, hand-written cloudy "save last line array" file matching
the real format exactly (tab-separated, one header line, a combined
9-character-label + human-readable-wavelength field) -- verified
against a real line array file produced by an actual cloudy run during
development -- so every expected value can be computed independently
and checked exactly.
"""

import astropy.units as u
import pytest

from slugpy.cloudy.cloudy_lines import read_cloudy_linearr

_HEADER = "#enr\tID\tI(intrinsic)\tI(emergent)\ttype\n"


def _row(wl_aa: float, label9: str, log_intrinsic: float, log_emergent: float, wl_human: str = "1.00000A",
    line_type: str = "i") -> str:
    """Build one real-format data row. label9 must be exactly 9 characters."""
    assert len(label9) == 9
    idfield = f"{label9}{wl_human} "
    return f" {wl_aa:.5e}\t{idfield}\t  {log_intrinsic:.3f}\t {log_emergent:.3f} \t{line_type}\n"


def _make_linearr_file(tmp_path, rows: list[str]):
    path = tmp_path / "test.linearr"
    path.write_text(_HEADER + "".join(rows))
    return path


def test_keeps_only_blank_suffix_and_above_threshold(tmp_path):
    """Only rows with a blank 5-character label suffix AND emergent luminosity > 1e10 erg/s are kept."""
    rows = [
        _row(1000.0, "H  1     ", 40.0, 41.0),  # kept: blank suffix, 10**41 > 1e10
        _row(2000.0, "H  1 Coll", 40.0, 41.0),  # dropped: non-blank suffix (a sub-process breakdown)
        _row(3000.0, "He 2     ", 5.0, 5.0),    # dropped: below luminosity threshold (10**5 << 1e10)
        _row(4000.0, "O  3     ", 42.0, 42.0),  # kept
    ]
    path = _make_linearr_file(tmp_path, rows)
    line_wl, line_label, line_lum = read_cloudy_linearr(path)

    assert line_label == ["H  1", "O  3"]
    assert line_wl.to_value(u.AA) == pytest.approx([1000.0, 4000.0])
    assert line_lum.to_value(u.erg / u.s) == pytest.approx([10.0 ** 41.0, 10.0 ** 42.0])


def test_luminosity_is_delogged_from_emergent_column(tmp_path):
    """line_lum is 10**(emergent column), not the intrinsic column and not left as a log value."""
    rows = [_row(1000.0, "H  1     ", log_intrinsic=99.0, log_emergent=15.5)]
    path = _make_linearr_file(tmp_path, rows)
    _, _, line_lum = read_cloudy_linearr(path)
    assert line_lum.to_value(u.erg / u.s) == pytest.approx([10.0 ** 15.5])


def test_label_is_first_four_characters_only(tmp_path):
    """The saved label is exactly the first 4 characters of the 9-character field, not the padded 9 or the human-readable wavelength that follows it."""
    rows = [_row(1000.0, "Ne 3     ", 40.0, 41.0)]
    path = _make_linearr_file(tmp_path, rows)
    _, line_label, _ = read_cloudy_linearr(path)
    assert line_label == ["Ne 3"]
    assert all(len(lbl) == 4 for lbl in line_label)


def test_threshold_is_strict(tmp_path):
    """Exactly 1e10 erg/s (log10 = 10.0) is not kept -- the threshold is a strict '>'."""
    rows = [
        _row(1000.0, "H  1     ", 10.0, 10.0),       # exactly at threshold: dropped
        _row(2000.0, "H  1     ", 10.001, 10.001),   # just above: kept
    ]
    path = _make_linearr_file(tmp_path, rows)
    _, line_label, line_lum = read_cloudy_linearr(path)
    assert line_label == ["H  1"]
    assert line_lum.to_value(u.erg / u.s) == pytest.approx([10.0 ** 10.001])


def test_empty_file_returns_empty(tmp_path):
    """A line array file with only a header (no lines survived any run at all) returns empty arrays, not an error."""
    path = _make_linearr_file(tmp_path, [])
    line_wl, line_label, line_lum = read_cloudy_linearr(path)
    assert len(line_wl) == 0
    assert line_label == []
    assert len(line_lum) == 0


def test_units(tmp_path):
    rows = [_row(1000.0, "H  1     ", 40.0, 41.0)]
    path = _make_linearr_file(tmp_path, rows)
    line_wl, _, line_lum = read_cloudy_linearr(path)
    assert line_wl.unit.physical_type == "length"
    line_wl.to(u.AA)
    line_lum.to(u.erg / u.s)
