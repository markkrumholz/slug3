"""
Unit tests for slug_reader.cluster_feedback/galaxy_feedback, the lazy
readers for supernova and stellar wind feedback output.

Drives two real simulations via run_sim, reusing test_yields_reader.py's
own decks (see its own docstring): TESTREADERYIELDS_DECK (galaxy-type)
and TESTREADERYIELDSNOTDECOMPOSED_DECK (cluster-type). Both request a
ccsn yield channel, so stars in its mass range explode as supernovae,
and both run out to 5 Myr, late enough for the most massive stars to
have done so. Every test below chdir's into tmp_path (via
monkeypatch.chdir) before calling run_sim; the deck paths themselves
are captured as absolute paths at collection time -- see
test_run_sim.py's own identical comment for why.

:copyright: Copyright (c) 2026 Mark Krumholz
"""

import pathlib
from typing import cast

import numpy as np
import pytest
from astropy import units as u

from slugpy import run_sim
from slugpy.slug_group_reader import slug_group_reader
from slugpy.slug_reader import slug_reader

REPO_ROOT = pathlib.Path.cwd()
GALAXY_DECK = str(REPO_ROOT / "tests" / "slugpy" / "assets" / "testReaderYields.in")
CLUSTER_DECK = str(REPO_ROOT / "tests" / "slugpy" / "assets" / "testReaderYieldsNotDecomposed.in")

FEEDBACK_UNITS: dict[str, u.UnitBase] = {
    "mdot_wind": u.Msun / u.yr,
    "pdot_wind": u.Msun * u.km / u.s / u.yr,
    "edot_wind": u.Lsun,
}


@pytest.fixture
def galaxy_reader(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> slug_reader:
    """A slug_reader for a fresh run of the galaxy-type deck."""
    monkeypatch.chdir(tmp_path)
    reader = run_sim(GALAXY_DECK, progress=False)
    assert reader is not None
    return reader


@pytest.fixture
def cluster_reader(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> slug_reader:
    """A slug_reader for a fresh run of the cluster-type deck."""
    monkeypatch.chdir(tmp_path)
    reader = run_sim(CLUSTER_DECK, progress=False)
    assert reader is not None
    return reader


def _group(reader_group: slug_group_reader | None) -> slug_group_reader:
    """Return reader_group, asserting that it is present."""
    assert isinstance(reader_group, slug_group_reader)
    return reader_group


def _plain(group: slug_group_reader, key: str) -> np.ndarray:
    """Return a unitless dataset from group as a plain numpy array."""
    data = group[key]
    assert not isinstance(data, u.Quantity)
    return data


def _quantity(group: slug_group_reader, key: str) -> u.Quantity:
    """Return a dataset with a physical unit from group as a Quantity."""
    data = group[key]
    assert isinstance(data, u.Quantity)
    return cast(u.Quantity, data)


def test_cluster_feedback_datasets(cluster_reader: slug_reader) -> None:
    """cluster_feedback holds trial/time/uid plus the four feedback datasets."""
    group = _group(cluster_reader.cluster_feedback)
    assert set(group.keys()) == {
        "trial", "time", "uid", "n_sn", "mdot_wind", "pdot_wind", "edot_wind"}


def test_galaxy_feedback_datasets(galaxy_reader: slug_reader) -> None:
    """galaxy_feedback holds the same datasets as cluster_feedback, minus uid."""
    group = _group(galaxy_reader.galaxy_feedback)
    assert set(group.keys()) == {
        "trial", "time", "n_sn", "mdot_wind", "pdot_wind", "edot_wind"}


def test_cluster_type_sim_has_no_galaxy_feedback(cluster_reader: slug_reader) -> None:
    """A cluster-type simulation has no galaxy_feedback group at all."""
    assert cluster_reader.galaxy_feedback is None


@pytest.mark.parametrize("which", ["cluster_feedback", "galaxy_feedback"])
def test_feedback_units(galaxy_reader: slug_reader, which: str) -> None:
    """n_sn is unitless; the wind fluxes come back as Quantities in Msun/yr, Msun km/s/yr, and Lsun."""
    group = _group(getattr(galaxy_reader, which))
    n_sn = _plain(group, "n_sn")
    assert np.all(n_sn >= 0.0)
    for key, unit in FEEDBACK_UNITS.items():
        data = _quantity(group, key)
        assert data.unit == unit
        assert np.all(data.value >= 0.0)


def test_cluster_n_sn_nondecreasing(cluster_reader: slug_reader) -> None:
    """Each cluster's cumulative supernova count never decreases with time."""
    group = _group(cluster_reader.cluster_feedback)
    uid = _plain(group, "uid")
    time = _quantity(group, "time").to_value(u.yr)
    n_sn = _plain(group, "n_sn")
    for this_uid in np.unique(uid):
        sel = uid == this_uid
        order = np.argsort(time[sel])
        assert np.all(np.diff(n_sn[sel][order]) >= 0.0)


def test_galaxy_n_sn_at_least_sum_of_clusters(galaxy_reader: slug_reader) -> None:
    """At every (trial, time), the galaxy's own n_sn is at least the sum over its currently-alive clusters' rows.

    Whether any supernova has happened by the last output time (5 Myr)
    is itself stochastic for these decks -- with the MIST_test tracks
    only stars above ~60 Msun have exploded by then -- so a nonzero
    count is not asserted here; testWriteClusterFeedbackH5 (in
    tests/io/testOutputManager.cpp) checks one deterministically.
    """
    gal = _group(galaxy_reader.galaxy_feedback)
    clu = _group(galaxy_reader.cluster_feedback)
    gal_trial = _plain(gal, "trial")
    gal_time = _quantity(gal, "time").to_value(u.yr)
    gal_n_sn = _plain(gal, "n_sn")
    clu_trial = _plain(clu, "trial")
    clu_time = _quantity(clu, "time").to_value(u.yr)
    clu_n_sn = _plain(clu, "n_sn")
    for trial, time, n_sn in zip(gal_trial, gal_time, gal_n_sn, strict=True):
        sel = (clu_trial == trial) & (clu_time == time)
        assert n_sn >= np.sum(clu_n_sn[sel]) * (1.0 - 1e-12)


def test_feedback_attributes_are_read_only(galaxy_reader: slug_reader) -> None:
    """cluster_feedback/galaxy_feedback raise AttributeError on assignment."""
    with pytest.raises(AttributeError):
        galaxy_reader.cluster_feedback = None
    with pytest.raises(AttributeError):
        galaxy_reader.galaxy_feedback = None
