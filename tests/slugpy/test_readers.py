"""
Unit tests for slugpy's HDF5 output readers (slug_reader,
slug_group_reader, slug_phot_reader), exercised against a real
committed output file, examples/clusterlib/clusterlib.h5, rather than
a small synthetic fixture: the reader classes have no logic of their
own to react to a particular deck, so a real file is both simpler to
obtain (it's already committed, for exactly this kind of use) and a
more representative check that the classes agree with the actual
layout OutputManagerH5 writes.

This file is run via pytest, invoked as a CTest test from
CMakeLists.txt (see the test_Slugpy target), with WORKING_DIRECTORY
set to the repo root, so CLUSTERLIB_H5 below resolves without needing
SLUG_DIR or any other environment variable.
"""

import numpy as np
import pytest
from astropy import units as u

from slugpy.slug_group_reader import slug_group_reader
from slugpy.slug_phot_reader import slug_phot_reader
from slugpy.slug_reader import slug_reader

CLUSTERLIB_H5 = "examples/clusterlib/clusterlib.h5"


@pytest.fixture(scope="module")
def reader():
    """A slug_reader opened on the committed clusterlib.h5 example output."""
    return slug_reader(CLUSTERLIB_H5)


@pytest.fixture(scope="module")
def clusters(reader):
    """reader.clusters, the clusters group's slug_group_reader."""
    return reader.clusters


@pytest.fixture(scope="module")
def cluster_spectra(reader):
    """reader.cluster_spectra, the cluster_spectra group's slug_group_reader."""
    return reader.cluster_spectra


@pytest.fixture(scope="module")
def cluster_phot(reader):
    """reader.cluster_phot, the cluster_phot group's slug_phot_reader."""
    return reader.cluster_phot


# ---------------------------------------------------------------------
# slug_reader: top-level metadata
# ---------------------------------------------------------------------

def test_top_level_metadata(reader):
    """slug_hash/date/time/rng_state are all non-empty strings."""
    assert isinstance(reader.slug_hash, str) and len(reader.slug_hash) > 0
    assert isinstance(reader.date, str) and len(reader.date) > 0
    assert isinstance(reader.time, str) and len(reader.time) > 0
    assert isinstance(reader.rng_state, str) and len(reader.rng_state) > 0


# ---------------------------------------------------------------------
# slug_reader.input_deck
# ---------------------------------------------------------------------

def test_input_deck_contents(reader):
    """input_deck parses to a mapping with this example's own known values."""
    deck = reader.input_deck
    assert deck["sim_type"] == "cluster"
    assert deck["n_trial"] == 2000
    assert deck["output"]["model_name"] == "clusterlib"


def test_input_deck_cached(reader):
    """input_deck is read once and the same object returned thereafter."""
    assert reader.input_deck is reader.input_deck


def test_input_deck_readonly(reader):
    """Assigning to input_deck raises AttributeError."""
    with pytest.raises(AttributeError):
        reader.input_deck = None


# ---------------------------------------------------------------------
# slug_reader.clusters / cluster_spectra / cluster_phot: presence,
# laziness, caching, and read-only-ness, common to all three
# ---------------------------------------------------------------------

@pytest.mark.parametrize("attr", ["clusters", "cluster_spectra", "cluster_phot"])
def test_group_property_present_and_cached(reader, attr):
    """Each group property returns a non-None reader, cached across accesses."""
    first = getattr(reader, attr)
    assert first is not None
    assert getattr(reader, attr) is first


@pytest.mark.parametrize("attr", ["clusters", "cluster_spectra", "cluster_phot"])
def test_group_property_readonly(reader, attr):
    """Assigning to any of the group properties raises AttributeError."""
    with pytest.raises(AttributeError):
        setattr(reader, attr, None)


def test_clusters_is_group_reader(clusters):
    """clusters is a plain slug_group_reader (not the phot subclass)."""
    assert isinstance(clusters, slug_group_reader)
    assert not isinstance(clusters, slug_phot_reader)


def test_cluster_spectra_is_group_reader(cluster_spectra):
    """cluster_spectra is a plain slug_group_reader."""
    assert isinstance(cluster_spectra, slug_group_reader)


def test_cluster_phot_is_phot_reader(cluster_phot):
    """cluster_phot is specifically a slug_phot_reader."""
    assert isinstance(cluster_phot, slug_phot_reader)


# ---------------------------------------------------------------------
# slug_group_reader, via clusters
# ---------------------------------------------------------------------

def test_clusters_keys(clusters):
    """keys() lists every dataset actually written to the clusters group."""
    expected = {"trial", "uid", "target_mass", "birth_mass", "form_time", "feh", "rng", "A_V"}
    assert set(clusters.keys()) == expected


def test_clusters_unitful_dataset(clusters):
    """A dataset with a real physical unit comes back as a positive-valued Quantity."""
    masses = clusters["target_mass"]
    assert isinstance(masses, u.Quantity)
    assert masses.unit == u.Msun
    assert len(masses) == 2000
    assert np.all(masses > 0)


def test_clusters_empty_unit_dataset(clusters):
    """A dataset with an empty units attribute comes back as a plain array."""
    feh = clusters["feh"]
    assert isinstance(feh, np.ndarray)
    assert not isinstance(feh, u.Quantity)
    assert len(feh) == 2000


def test_clusters_non_numeric_dataset(clusters):
    """The non-numeric, empty-unit rng dataset also comes back as a plain array."""
    rng = clusters["rng"]
    assert isinstance(rng, np.ndarray)
    assert not isinstance(rng, u.Quantity)
    assert rng.dtype.kind == "S"


def test_clusters_getitem_cached(clusters):
    """Repeated access to the same dataset returns the identical cached object."""
    assert clusters["target_mass"] is clusters["target_mass"]


def test_clusters_getitem_unknown_key(clusters):
    """An unknown dataset name raises KeyError."""
    with pytest.raises(KeyError):
        clusters["not_a_real_dataset"]


# ---------------------------------------------------------------------
# slug_group_reader, via cluster_spectra (2D datasets)
# ---------------------------------------------------------------------

def test_cluster_spectra_keys(cluster_spectra):
    """keys() lists every dataset actually written to the cluster_spectra group."""
    expected = {"trial", "time", "uid", "wl", "spec", "spec_extinct"}
    assert set(cluster_spectra.keys()) == expected


def test_cluster_spectra_wl(cluster_spectra):
    """wl is a 1D wavelength Quantity in Angstrom."""
    wl = cluster_spectra["wl"]
    assert isinstance(wl, u.Quantity)
    assert wl.unit == u.AA
    assert len(wl) == 1024
    assert np.all(np.diff(wl.value) > 0)


def test_cluster_spectra_spec_2d(cluster_spectra, reader):
    """spec is a 2D flux-density Quantity, one row per trial, one column per wl point."""
    spec = cluster_spectra["spec"]
    wl = cluster_spectra["wl"]
    assert isinstance(spec, u.Quantity)
    assert spec.unit.is_equivalent(u.erg / u.s / u.AA)
    assert spec.shape == (reader.input_deck["n_trial"], len(wl))
    assert np.all(spec.value >= 0)
    assert not np.any(np.isnan(spec.value))


# ---------------------------------------------------------------------
# slug_phot_reader, via cluster_phot
# ---------------------------------------------------------------------

def test_cluster_phot_filters_match_input_deck(cluster_phot, reader):
    """filters/filter_units match this deck's own phot.filters, in order."""
    assert cluster_phot.filters == list(reader.input_deck["phot"]["filters"])
    assert len(cluster_phot.filter_units) == len(cluster_phot.filters)


def test_cluster_phot_magnitude_filter(cluster_phot):
    """A Vega-system real filter comes back as a magnitude Quantity."""
    phot = cluster_phot["HST.WFC3_UVIS1.F555W"]
    assert isinstance(phot, u.Quantity)
    assert phot.unit == u.mag
    assert len(phot) == 2000


def test_cluster_phot_photon_count_filter(cluster_phot):
    """An idealized photon-count filter comes back in photon/s."""
    phot = cluster_phot["Q(HI)"]
    assert isinstance(phot, u.Quantity)
    assert phot.unit.is_equivalent(u.photon / u.s)
    assert np.all(phot.value >= 0)


def test_cluster_phot_lbol_filter(cluster_phot):
    """Lbol comes back as a luminosity Quantity."""
    lbol = cluster_phot["Lbol"]
    assert isinstance(lbol, u.Quantity)
    assert lbol.unit.is_equivalent(u.Lsun)
    assert np.all(lbol.value > 0)


def test_cluster_phot_ext_suffix(cluster_phot):
    """Appending _ext to a real filter name returns its extincted counterpart."""
    plain = cluster_phot["HST.WFC3_UVIS1.F555W"]
    extinct = cluster_phot["HST.WFC3_UVIS1.F555W_ext"]
    assert isinstance(extinct, u.Quantity)
    assert extinct.unit == plain.unit
    assert len(extinct) == len(plain)
    assert not np.array_equal(extinct.value, plain.value)


def test_cluster_phot_lbol_has_no_extincted_counterpart(cluster_phot):
    """Lbol_ext isn't a real column (phot_extinct excludes Lbol) and fails."""
    with pytest.raises(Exception):
        cluster_phot["Lbol_ext"]


def test_cluster_phot_getitem_cached(cluster_phot):
    """Repeated per-filter access returns the identical cached object."""
    assert cluster_phot["Lbol"] is cluster_phot["Lbol"]


def test_cluster_phot_fallthrough_to_group_reader(cluster_phot):
    """A non-filter key (e.g. trial) falls through to slug_group_reader.__getitem__."""
    trial = cluster_phot["trial"]
    assert isinstance(trial, np.ndarray)
    assert not isinstance(trial, u.Quantity)
    assert len(trial) == 2000


def test_cluster_phot_bulk_raw_access(cluster_phot):
    """The raw phot/phot_extinct datasets (per-column units) come back as plain arrays."""
    phot = cluster_phot["phot"]
    phot_extinct = cluster_phot["phot_extinct"]
    assert isinstance(phot, np.ndarray) and not isinstance(phot, u.Quantity)
    assert isinstance(phot_extinct, np.ndarray) and not isinstance(phot_extinct, u.Quantity)
    assert phot.shape == (2000, len(cluster_phot.filters))
    # phot_extinct excludes Lbol, the only filter with no extincted counterpart
    assert phot_extinct.shape == (2000, len(cluster_phot.filters) - 1)


def test_cluster_phot_unknown_key(cluster_phot):
    """A key that's neither a filter name nor a dataset name raises KeyError."""
    with pytest.raises(KeyError):
        cluster_phot["not_a_filter_or_dataset"]
