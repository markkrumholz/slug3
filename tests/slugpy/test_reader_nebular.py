"""
Unit tests for the slugpy reader's own nebular-emission support:
slug_spectra_reader (line_labels/line_index/line_luminosity, and
spec_neb/spec_neb_extinct access) and slug_phot_reader's "_neb"/
"_neb_ex" suffixes (phot_neb/phot_neb_extinct access).

Drives a real simulation via run_sim (see test_run_sim.py's own
identical rationale) from tests/slugpy/assets/testReaderNebular.in --
photometry, extinction, and nebular emission (from the small,
committed tests/nebular/assets/nebular_test.h5 fixture) all enabled
together, so this needs no data fetched separately and can stay a
"quick" test. Every test below chdir's into tmp_path (via
monkeypatch.chdir) before calling run_sim; NEBULAR_DECK is captured as
an absolute path at collection time -- see test_run_sim.py's own
identical comment for why.

:copyright: Copyright (c) 2026 Mark Krumholz
"""

import pathlib
from typing import cast

import numpy as np
import pytest
from astropy import units as u

from slugpy import run_sim
from slugpy.slug_group_reader import slug_group_reader
from slugpy.slug_phot_reader import slug_phot_reader
from slugpy.slug_reader import slug_reader
from slugpy.slug_spectra_reader import slug_spectra_reader

REPO_ROOT = pathlib.Path.cwd()
NEBULAR_DECK = str(REPO_ROOT / "tests" / "slugpy" / "assets" / "testReaderNebular.in")
CLUSTER_EXTINCT_DECK = str(REPO_ROOT / "tests" / "core" / "assets" / "testClusterExtinct.in")


@pytest.fixture
def nebular_reader(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> slug_reader:
    """A slug_reader for a fresh run of NEBULAR_DECK (phot + extinct + nebular all enabled)."""
    monkeypatch.chdir(tmp_path)
    reader = run_sim(NEBULAR_DECK, progress=False)
    assert reader is not None
    return reader


@pytest.fixture
def no_nebular_reader(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> slug_reader:
    """A slug_reader for a fresh run of CLUSTER_EXTINCT_DECK (phot + extinct, but no nebular emission)."""
    monkeypatch.chdir(tmp_path)
    reader = run_sim(CLUSTER_EXTINCT_DECK, progress=False)
    assert reader is not None
    return reader


def test_cluster_spectra_is_a_spectra_reader(nebular_reader: slug_reader):
    """slug_reader.cluster_spectra is a slug_spectra_reader."""
    assert isinstance(nebular_reader.cluster_spectra, slug_spectra_reader)


def test_cluster_phot_is_a_phot_reader(nebular_reader: slug_reader):
    """slug_reader.cluster_phot is still a slug_phot_reader."""
    assert isinstance(nebular_reader.cluster_phot, slug_phot_reader)


def test_line_labels_decoded_to_str(nebular_reader: slug_reader):
    """line_labels is a plain list of Python str, matching the fixture's own line_label."""
    cs = cast(slug_spectra_reader, nebular_reader.cluster_spectra)
    labels = cs.line_labels
    assert labels == ["LINE1", "LINE2", "LINE3"]
    assert labels is not None
    assert all(isinstance(lbl, str) for lbl in labels)


def test_getitem_line_label_matches_line_labels(nebular_reader: slug_reader):
    """cluster_spectra["line_label"] agrees with the line_labels property."""
    cs = cast(slug_spectra_reader, nebular_reader.cluster_spectra)
    assert list(cs["line_label"]) == cs.line_labels


def test_line_wl_matches_fixture(nebular_reader: slug_reader):
    """line_wl matches the fixture's own known line wavelengths."""
    cs = cast(slug_spectra_reader, nebular_reader.cluster_spectra)
    line_wl = cast(u.Quantity, cs["line_wl"])
    assert line_wl.unit == u.AA
    assert list(line_wl.to_value(u.AA)) == pytest.approx([4000.0, 6000.0, 9000.0])


def test_line_index_matches_position(nebular_reader: slug_reader):
    """line_index(label, wl) returns the position matching both label and line_wl."""
    cs = cast(slug_spectra_reader, nebular_reader.cluster_spectra)
    assert cs.line_index("LINE1", 4000.0) == 0
    assert cs.line_index("LINE2", 6000.0) == 1
    assert cs.line_index("LINE3", 9000.0) == 2


def test_line_index_accepts_quantity_wl(nebular_reader: slug_reader):
    """line_index accepts wl as an astropy Quantity, not just a bare float."""
    cs = cast(slug_spectra_reader, nebular_reader.cluster_spectra)
    assert cs.line_index("LINE2", 6000.0 * u.AA) == 1
    assert cs.line_index("LINE2", 0.6 * u.um) == 1


def test_line_index_wrong_wl_for_label_raises(nebular_reader: slug_reader):
    """A label that exists, but at the wrong wavelength, still raises KeyError."""
    cs = cast(slug_spectra_reader, nebular_reader.cluster_spectra)
    with pytest.raises(KeyError):
        cs.line_index("LINE1", 6000.0)


def test_line_index_unknown_label_raises(nebular_reader: slug_reader):
    """An unrecognized label raises KeyError."""
    cs = cast(slug_spectra_reader, nebular_reader.cluster_spectra)
    with pytest.raises(KeyError):
        cs.line_index("NotARealLine", 4000.0)


def test_line_luminosity_matches_raw_column(nebular_reader: slug_reader):
    """line_luminosity(label, wl) matches the corresponding column of neb_lines directly."""
    cs = cast(slug_spectra_reader, nebular_reader.cluster_spectra)
    neb_lines = cast(u.Quantity, cs["neb_lines"])
    index = cs.line_index("LINE2", 6000.0)

    lum = cast(u.Quantity, cs.line_luminosity("LINE2", 6000.0))
    assert lum.unit == u.erg / u.s
    assert np.array_equal(lum.to_value(u.erg / u.s), neb_lines[:, index].to_value(u.erg / u.s))


def test_line_luminosity_extinct_matches_raw_column(nebular_reader: slug_reader):
    """line_luminosity(..., extinct=True) matches the corresponding column of neb_lines_extinct."""
    cs = cast(slug_spectra_reader, nebular_reader.cluster_spectra)
    neb_lines_extinct = cast(u.Quantity, cs["neb_lines_extinct"])
    index = cs.line_index("LINE2", 6000.0)

    lum_ex = cast(u.Quantity, cs.line_luminosity("LINE2", 6000.0, extinct=True))
    assert np.array_equal(
        lum_ex.to_value(u.erg / u.s), neb_lines_extinct[:, index].to_value(u.erg / u.s))


def test_line_luminosity_differs_with_and_without_extinction(nebular_reader: slug_reader):
    """The extincted line luminosity is strictly fainter than the unextincted one."""
    cs = cast(slug_spectra_reader, nebular_reader.cluster_spectra)
    lum = cast(u.Quantity, cs.line_luminosity("LINE2", 6000.0))
    lum_ex = cast(u.Quantity, cs.line_luminosity("LINE2", 6000.0, extinct=True))
    assert np.all(lum_ex.to_value(u.erg / u.s) < lum.to_value(u.erg / u.s))


def test_getitem_label_wl_tuple_matches_line_luminosity(nebular_reader: slug_reader):
    """spectra["LINE2", 6000.0] (filter-name-style access) is exactly equivalent to line_luminosity()."""
    cs = cast(slug_spectra_reader, nebular_reader.cluster_spectra)
    via_tuple = cast(u.Quantity, cs["LINE2", 6000.0])
    via_method = cast(u.Quantity, cs.line_luminosity("LINE2", 6000.0))
    assert np.array_equal(via_tuple.to_value(u.erg / u.s), via_method.to_value(u.erg / u.s))


def test_getitem_label_wl_tuple_ex_suffix_matches_extincted_line_luminosity(nebular_reader: slug_reader):
    """spectra["LINE2_ex", 6000.0] matches line_luminosity(..., extinct=True), mirroring slug_phot_reader's own "_ex" convention."""
    cs = cast(slug_spectra_reader, nebular_reader.cluster_spectra)
    via_tuple = cast(u.Quantity, cs["LINE2_ex", 6000.0])
    via_method = cast(u.Quantity, cs.line_luminosity("LINE2", 6000.0, extinct=True))
    assert np.array_equal(via_tuple.to_value(u.erg / u.s), via_method.to_value(u.erg / u.s))


def test_getitem_label_wl_tuple_caches_result(nebular_reader: slug_reader):
    """A (label, wl) lookup is cached: the second access returns the identical cached array."""
    cs = cast(slug_spectra_reader, nebular_reader.cluster_spectra)
    first = cs["LINE1", 4000.0]
    second = cs["LINE1", 4000.0]
    assert first is second


def test_getitem_label_wl_tuple_unknown_line_raises(nebular_reader: slug_reader):
    """An unrecognized (label, wl) pair raises KeyError, matching line_index's own contract."""
    cs = cast(slug_spectra_reader, nebular_reader.cluster_spectra)
    with pytest.raises(KeyError):
        cs["NotARealLine", 4000.0]


def test_spec_neb_and_spec_neb_extinct_accessible(nebular_reader: slug_reader):
    """spec_neb/spec_neb_extinct are readable, with the expected shapes and units."""
    cs = cast(slug_spectra_reader, nebular_reader.cluster_spectra)
    wl = cast(u.Quantity, cs["wl"])
    wl_extinct = cast(u.Quantity, cs["wl_extinct"])
    spec_neb = cast(u.Quantity, cs["spec_neb"])
    spec_neb_extinct = cast(u.Quantity, cs["spec_neb_extinct"])

    assert spec_neb.shape[1] == wl.shape[0]
    assert spec_neb_extinct.shape[1] == wl_extinct.shape[0]
    assert spec_neb.unit == u.erg / u.s / u.AA
    assert spec_neb_extinct.unit == u.erg / u.s / u.AA


def test_phot_neb_matches_raw_dataset(nebular_reader: slug_reader):
    """cluster_phot["Filter_neb"] matches the raw phot_neb dataset's own column."""
    cp = cast(slug_phot_reader, nebular_reader.cluster_phot)
    filter_name = "SLUGTEST.CAM1.G500"
    index = cp.filters.index(filter_name)
    # Read phot_neb as a whole via the base class's own generic
    # __getitem__, independent of slug_phot_reader's per-filter logic
    # -- a plain ndarray, not a Quantity, since phot_neb's own "units"
    # attribute is one string per column, not a single string (see
    # slug_group_reader's own docstring)
    raw = cast(np.ndarray, slug_group_reader.__getitem__(cp, "phot_neb"))
    assert np.array_equal(cast(u.Quantity, cp[filter_name + "_neb"]).to_value(), raw[:, index])


def test_phot_neb_extinct_matches_raw_dataset(nebular_reader: slug_reader):
    """cluster_phot["Filter_neb_ex"] matches the raw phot_neb_extinct dataset's own column."""
    cp = cast(slug_phot_reader, nebular_reader.cluster_phot)
    filter_name = "SLUGTEST.CAM1.G500"
    index = cp.filters.index(filter_name)
    raw = cast(np.ndarray, slug_group_reader.__getitem__(cp, "phot_neb_extinct"))
    assert np.array_equal(cast(u.Quantity, cp[filter_name + "_neb_ex"]).to_value(), raw[:, index])


def test_phot_neb_lbol_raises_indexerror(nebular_reader: slug_reader):
    """phot_neb has no column for the synthetic Lbol entry -- requesting it raises IndexError."""
    cp = cast(slug_phot_reader, nebular_reader.cluster_phot)
    assert "Lbol" in cp.filters
    with pytest.raises(IndexError):
        cp["Lbol_neb"]
    with pytest.raises(IndexError):
        cp["Lbol_neb_ex"]


def test_phot_convert_also_converts_neb_variants(nebular_reader: slug_reader):
    """phot_convert() converts cached phot_neb/phot_neb_extinct entries too, not just phot/phot_extinct."""
    cp = cast(slug_phot_reader, nebular_reader.cluster_phot)
    filter_name = "SLUGTEST.CAM1.G500"

    # Trigger caching of both nebular variants before converting
    before_neb = cast(u.Quantity, cp[filter_name + "_neb"])
    before_neb_ex = cast(u.Quantity, cp[filter_name + "_neb_ex"])
    assert before_neb.unit == u.erg / u.s / u.AA
    assert before_neb_ex.unit == u.erg / u.s / u.AA

    cp.phot_convert("AB")

    after_neb = cast(u.Quantity, cp[filter_name + "_neb"])
    after_neb_ex = cast(u.Quantity, cp[filter_name + "_neb_ex"])
    assert after_neb.unit == u.mag(u.AB)
    assert after_neb_ex.unit == u.mag(u.AB)


def test_no_nebular_data_line_labels_is_none(no_nebular_reader: slug_reader):
    """line_labels is None for a file whose simulation did not request nebular emission."""
    cs = cast(slug_spectra_reader, no_nebular_reader.cluster_spectra)
    assert cs.line_labels is None


def test_no_nebular_data_getitem_line_label_raises(no_nebular_reader: slug_reader):
    """cluster_spectra["line_label"] raises KeyError when there is no nebular line data at all."""
    cs = cast(slug_spectra_reader, no_nebular_reader.cluster_spectra)
    with pytest.raises(KeyError):
        cs["line_label"]


def test_no_nebular_data_line_index_raises(no_nebular_reader: slug_reader):
    """line_index() raises KeyError when there is no nebular line data at all."""
    cs = cast(slug_spectra_reader, no_nebular_reader.cluster_spectra)
    with pytest.raises(KeyError):
        cs.line_index("LINE1", 4000.0)


def test_no_nebular_data_no_spec_neb_dataset(no_nebular_reader: slug_reader):
    """spec_neb is absent entirely (not just empty) when nebular emission was not requested."""
    cs = cast(slug_spectra_reader, no_nebular_reader.cluster_spectra)
    with pytest.raises(KeyError):
        cs["spec_neb"]
