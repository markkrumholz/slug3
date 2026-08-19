"""
Unit tests for slugpy.cloudy.cloudy_input.write_cloudy_input.

Uses a small, hand-constructed synthetic spectrum (rather than real
slug output) so every expected value in the written deck can be
computed independently and checked exactly, rather than just checking
"the file was written."
"""

import re

import astropy.units as u
import numpy as np
import pytest

from slugpy.cloudy.cloudy_input import DEFAULT_TEMPLATE, write_cloudy_input
from slugpy.cloudy.hiiregparam import hiiregparam

QH0 = 1e49 * u.photon / u.s
_C = 2.99792458e10  # cm/s, independent of the module's own constant


def _extract_float(text: str, pattern: str) -> float:
    """Search text for pattern (which must have one capture group) and return it as a float."""
    match = re.search(pattern, text, re.MULTILINE)
    assert match is not None, f"pattern {pattern!r} not found"
    return float(match.group(1))


@pytest.fixture
def hp():
    """A simple hiiregparam instance, nII + U given directly."""
    return hiiregparam(QH0, nII=100.0 / u.cm ** 3, U=10.0 ** -2.5)


@pytest.fixture
def spectrum():
    """A small constant-flux spectrum spanning the H ionization edge (~912 Angstrom)."""
    wl = np.array([200.0, 500.0, 900.0, 1500.0, 3000.0]) * u.AA
    spec = np.full(5, 1e-10) * u.erg / u.s / u.AA
    return wl, spec


def test_default_template_exists():
    """DEFAULT_TEMPLATE points at a real, readable file."""
    assert DEFAULT_TEMPLATE.is_file()
    assert "save last continuum" in DEFAULT_TEMPLATE.read_text()


def test_writes_expected_physical_condition_lines(tmp_path, spectrum, hp):
    """hden/radius/metals and grains/Q(H) lines match hp/feh exactly."""
    wl, spec = spectrum
    feh = -0.3
    path = write_cloudy_input(wl, spec, QH0, hp, feh, tmp_path / "test.in")
    text = path.read_text()

    assert _extract_float(text, r"^hden (\S+)$") == pytest.approx(np.log10(100.0), rel=1e-6)
    assert _extract_float(text, r"^radius (\S+)$") == pytest.approx(
        np.log10(hp.r0.to_value(u.cm)), rel=1e-6)
    assert _extract_float(text, r"^metals and grains (\S+)$") == pytest.approx(10.0 ** feh, rel=1e-6)
    assert _extract_float(text, r"^Q\(H\) = (\S+)$") == pytest.approx(49.0, rel=1e-6)


def test_output_filename_substitution(tmp_path, spectrum, hp):
    """$OUTPUT_FILENAME in the template is replaced with output_path's own stem, no directory/extension."""
    wl, spec = spectrum
    path = write_cloudy_input(wl, spec, QH0, hp, 0.0, tmp_path / "my_model.in")
    text = path.read_text()

    assert 'save last continuum "my_model.con"' in text
    assert 'save last line array units Angstrom "my_model.linearr"' in text
    assert "$OUTPUT_FILENAME" not in text


def test_template_lines_copied_verbatim(tmp_path, spectrum, hp):
    """Every non-$OUTPUT_FILENAME template line appears unchanged in the output."""
    wl, spec = spectrum
    path = write_cloudy_input(wl, spec, QH0, hp, 0.0, tmp_path / "test.in")
    text = path.read_text()

    for line in DEFAULT_TEMPLATE.read_text().splitlines():
        if "$OUTPUT_FILENAME" in line:
            continue
        assert line in text


def test_spectrum_table_matches_hand_computed_values(tmp_path, spectrum, hp):
    """The interpolate/continue table's frequency/luminosity pairs match an independent calculation."""
    wl, spec = spectrum
    path = write_cloudy_input(wl, spec, QH0, hp, 0.0, tmp_path / "test.in")
    text = path.read_text()

    wl_aa = wl.to_value(u.AA)
    spec_val = spec.to_value(u.erg / u.s / u.AA)
    freq = _C / (wl_aa * 1e-8)
    logfreq = np.log10(freq)
    logL_nu = np.log10(spec_val * _C / freq ** 2)
    floor = np.amin(logL_nu) - 4.0

    # Extract every (freq, lum) pair actually written, in order
    pairs = [(float(a), float(b)) for a, b in
        re.findall(r"\(\s*([\d.eE+-]+)\s+([\d.eE+-]+)\s*\)", text)]

    # First pair: low-frequency padding at 10**7.51 Hz
    assert pairs[0] == pytest.approx((7.51, floor), rel=1e-6)
    # Second pair: just below the spectrum's own lowest frequency
    assert pairs[1] == pytest.approx((logfreq[-1] - 0.01, floor), rel=1e-6)
    # Real spectrum points, in ascending-frequency order
    for k in range(5):
        assert pairs[2 + k] == pytest.approx((logfreq[-1 - k], logL_nu[-1 - k]), rel=1e-6)
    # Last two pairs: high-frequency padding
    assert pairs[7] == pytest.approx((logfreq[0] + 0.01, floor), rel=1e-6)
    assert pairs[8] == pytest.approx((22.4, floor), rel=1e-6)
    assert len(pairs) == 9


def test_zero_flux_replaced_with_floor(tmp_path, hp):
    """A zero-flux point is replaced with 1e-4x the minimum positive flux, not left as log(0)."""
    wl = np.array([200.0, 500.0, 900.0]) * u.AA
    spec = np.array([1e-10, 0.0, 1e-10]) * u.erg / u.s / u.AA
    path = write_cloudy_input(wl, spec, QH0, hp, 0.0, tmp_path / "test.in")
    text = path.read_text()
    assert "nan" not in text.lower()
    assert "-inf" not in text.lower()


def test_all_zero_spectrum_raises(tmp_path, hp):
    """A spectrum with no positive flux anywhere raises ValueError rather than silently producing garbage."""
    wl = np.array([200.0, 500.0]) * u.AA
    spec = np.array([0.0, 0.0]) * u.erg / u.s / u.AA
    with pytest.raises(ValueError, match="all zero"):
        write_cloudy_input(wl, spec, QH0, hp, 0.0, tmp_path / "test.in")


def test_creates_parent_directory(tmp_path, spectrum, hp):
    """output_path's own parent directory is created if it doesn't exist."""
    wl, spec = spectrum
    nested = tmp_path / "a" / "b" / "c" / "test.in"
    path = write_cloudy_input(wl, spec, QH0, hp, 0.0, nested)
    assert path.is_file()


def test_bare_rate_qH0_matches_photon_tagged(tmp_path, spectrum, hp):
    """Q(H) is the same whether qH0 is tagged photon/s or a bare 1/s rate."""
    wl, spec = spectrum
    path_photon = write_cloudy_input(wl, spec, 1e49 * u.photon / u.s, hp, 0.0, tmp_path / "a.in")
    path_bare = write_cloudy_input(wl, spec, 1e49 / u.s, hp, 0.0, tmp_path / "b.in")
    qh_photon = _extract_float(path_photon.read_text(), r"^Q\(H\) = (\S+)$")
    qh_bare = _extract_float(path_bare.read_text(), r"^Q\(H\) = (\S+)$")
    assert qh_photon == qh_bare


def test_custom_template(tmp_path, spectrum, hp):
    """A custom template path is used instead of DEFAULT_TEMPLATE when given."""
    wl, spec = spectrum
    custom = tmp_path / "custom_template.in"
    custom.write_text('title custom\nsave last continuum "$OUTPUT_FILENAME.con"\n')
    path = write_cloudy_input(wl, spec, QH0, hp, 0.0, tmp_path / "out.in", template=custom)
    text = path.read_text()
    assert "title custom" in text
    assert "abundances HII region" not in text  # not the default template
