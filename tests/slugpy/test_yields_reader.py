"""
Unit tests for slug_yields_reader (slug_reader.cluster_yields/
galaxy_yields/isotopes), the lazy reader for nucleosynthetic yield
output.

Drives two real simulations via run_sim (see test_run_sim.py's own
identical rationale): TESTREADERYIELDS_DECK (a galaxy-type simulation,
decomposed by channel -- the default -- with two real ccsn channels,
sukhbold16 and kobayashi06_11, deliberately both the same channel type
so a wildcard query has more than one matching column to actually
exercise) and TESTREADERYIELDSNOTDECOMPOSED_DECK (a cluster-type
simulation with yields.channel_decomposed = false). Both use the real,
small, committed yield model files under data/yields/ and the small
MIST_test track fixture, so this needs no data fetched separately and
can stay a "quick" test. Every test below chdir's into tmp_path (via
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
from slugpy.slug_reader import slug_reader
from slugpy.slug_yields_reader import slug_yields_reader

REPO_ROOT = pathlib.Path.cwd()
TESTREADERYIELDS_DECK = str(REPO_ROOT / "tests" / "slugpy" / "assets" / "testReaderYields.in")
TESTREADERYIELDSNOTDECOMPOSED_DECK = str(
    REPO_ROOT / "tests" / "slugpy" / "assets" / "testReaderYieldsNotDecomposed.in")


@pytest.fixture
def yields_reader(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> slug_reader:
    """A slug_reader for a fresh run of TESTREADERYIELDS_DECK (decomposed, galaxy-type)."""
    monkeypatch.chdir(tmp_path)
    reader = run_sim(TESTREADERYIELDS_DECK, progress=False)
    assert reader is not None
    return reader


@pytest.fixture
def not_decomposed_reader(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> slug_reader:
    """A slug_reader for a fresh run of TESTREADERYIELDSNOTDECOMPOSED_DECK (not decomposed, cluster-type)."""
    monkeypatch.chdir(tmp_path)
    reader = run_sim(TESTREADERYIELDSNOTDECOMPOSED_DECK, progress=False)
    assert reader is not None
    return reader


def test_cluster_yields_is_a_yields_reader(yields_reader: slug_reader):
    """slug_reader.cluster_yields is a slug_yields_reader."""
    assert isinstance(yields_reader.cluster_yields, slug_yields_reader)


def test_galaxy_yields_is_a_yields_reader(yields_reader: slug_reader):
    """slug_reader.galaxy_yields is a slug_yields_reader."""
    assert isinstance(yields_reader.galaxy_yields, slug_yields_reader)


def test_cluster_type_sim_has_no_galaxy_yields(not_decomposed_reader: slug_reader):
    """A cluster-type simulation has no galaxy_yields group at all."""
    assert not_decomposed_reader.galaxy_yields is None


def test_isotopes_alias_matches_cluster_yields(yields_reader: slug_reader):
    """slug_reader.isotopes aliases cluster_yields.isotopes, mirroring the filters alias."""
    cy = cast(slug_yields_reader, yields_reader.cluster_yields)
    assert yields_reader.isotopes == cy.isotopes
    assert yields_reader.isotopes is not None
    assert "Fe56" in yields_reader.isotopes


def test_channels_and_models_match_deck(yields_reader: slug_reader):
    """channels/models match the deck's own two ccsn channels, in order."""
    cy = cast(slug_yields_reader, yields_reader.cluster_yields)
    assert cy.channels == ["ccsn", "ccsn"]
    assert cy.models == ["sukhbold16", "kobayashi06_11"]


def test_decomposed_true_by_default(yields_reader: slug_reader):
    """decomposed is True for a deck that never sets yields.channel_decomposed."""
    cy = cast(slug_yields_reader, yields_reader.cluster_yields)
    assert cy.decomposed is True


def test_decomposed_false_when_requested(not_decomposed_reader: slug_reader):
    """decomposed is False for a deck with yields.channel_decomposed = false."""
    cy = cast(slug_yields_reader, not_decomposed_reader.cluster_yields)
    assert cy.decomposed is False


def test_attributes_are_read_only(yields_reader: slug_reader):
    """isotopes/channels/models/decomposed all raise AttributeError on assignment."""
    cy = cast(slug_yields_reader, yields_reader.cluster_yields)
    with pytest.raises(AttributeError):
        cy.isotopes = []
    with pytest.raises(AttributeError):
        cy.channels = []
    with pytest.raises(AttributeError):
        cy.models = []
    with pytest.raises(AttributeError):
        cy.decomposed = False
    with pytest.raises(AttributeError):
        yields_reader.cluster_yields = None
    with pytest.raises(AttributeError):
        yields_reader.galaxy_yields = None
    with pytest.raises(AttributeError):
        yields_reader.isotopes = []


def test_getitem_isotope_lookup_is_case_and_space_insensitive(yields_reader: slug_reader):
    """'fe56', 'FE56', 'Fe 56', and 'fe 56' all resolve to the same cached 'Fe56' entry."""
    cy = cast(slug_yields_reader, yields_reader.cluster_yields)
    canonical = cast(u.Quantity, cy["Fe56"])
    for spelling in ("fe56", "FE56", "Fe 56", "fe 56"):
        assert cy[spelling] is canonical


def test_getitem_unknown_isotope_raises_keyerror(yields_reader: slug_reader):
    """An isotope name that matches nothing in isotopes raises KeyError."""
    cy = cast(slug_yields_reader, yields_reader.cluster_yields)
    with pytest.raises(KeyError):
        cy["Xx999"]


def test_getitem_plain_string_decomposed_returns_full_2d_array(yields_reader: slug_reader):
    """A bare isotope name, when decomposed, returns one column per channel."""
    cy = cast(slug_yields_reader, yields_reader.cluster_yields)
    fe56 = cast(u.Quantity, cy["Fe56"])
    assert fe56.unit == u.Msun
    assert fe56.ndim == 2
    assert fe56.shape[1] == len(cy.channels) == 2


def test_getitem_plain_string_not_decomposed_returns_1d_array(not_decomposed_reader: slug_reader):
    """A bare isotope name, when not decomposed, returns a single 1D column."""
    cy = cast(slug_yields_reader, not_decomposed_reader.cluster_yields)
    fe56 = cast(u.Quantity, cy["Fe56"])
    assert fe56.unit == u.Msun
    assert fe56.ndim == 1


def test_getitem_triple_key_not_decomposed_raises_typeerror(not_decomposed_reader: slug_reader):
    """A (channel, model, isotope) triple is meaningless when decomposed is False."""
    cy = cast(slug_yields_reader, not_decomposed_reader.cluster_yields)
    with pytest.raises(TypeError):
        cy[("ccsn", "", "Fe56")]


def test_getitem_triple_exact_match_returns_1d_matching_first_column(yields_reader: slug_reader):
    """(channel, model, isotope), fully specified, returns the one matching column as 1D."""
    cy = cast(slug_yields_reader, yields_reader.cluster_yields)
    full = cast(u.Quantity, cy["Fe56"])
    exact = cast(u.Quantity, cy[("ccsn", "sukhbold16", "fe56")])
    assert exact.ndim == 1
    assert np.array_equal(exact.to_value(u.Msun), full[:, 0].to_value(u.Msun))


def test_getitem_triple_wildcard_channel_matches_every_column(yields_reader: slug_reader):
    """(channel, "", isotope) matches every column of that channel type -- both, here."""
    cy = cast(slug_yields_reader, yields_reader.cluster_yields)
    full = cast(u.Quantity, cy["Fe56"])
    wildcard = cast(u.Quantity, cy[("ccsn", "", "Fe56")])
    assert wildcard.shape == full.shape
    assert np.array_equal(wildcard.to_value(u.Msun), full.to_value(u.Msun))


def test_getitem_triple_wildcard_model_matches_single_column(yields_reader: slug_reader):
    """("", model, isotope) matches just the one channel using that model."""
    cy = cast(slug_yields_reader, yields_reader.cluster_yields)
    full = cast(u.Quantity, cy["Fe56"])
    via_model = cast(u.Quantity, cy[("", "kobayashi06_11", "Fe56")])
    assert via_model.ndim == 1
    assert np.array_equal(via_model.to_value(u.Msun), full[:, 1].to_value(u.Msun))


def test_getitem_triple_no_match_raises_keyerror(yields_reader: slug_reader):
    """A (channel, model) combination that matches nothing raises KeyError."""
    cy = cast(slug_yields_reader, yields_reader.cluster_yields)
    with pytest.raises(KeyError):
        cy[("massive_star_winds", "", "Fe56")]
    with pytest.raises(KeyError):
        cy[("ccsn", "not_a_real_model", "Fe56")]


def test_getitem_triple_unknown_isotope_raises_keyerror(yields_reader: slug_reader):
    """A triple naming an unrecognized isotope raises KeyError, same as the bare-string case."""
    cy = cast(slug_yields_reader, yields_reader.cluster_yields)
    with pytest.raises(KeyError):
        cy[("ccsn", "", "Xx999")]


def test_getitem_invalid_key_type_raises_typeerror(yields_reader: slug_reader):
    """A key that is neither a string nor a 3-element sequence of strings raises TypeError."""
    cy = cast(slug_yields_reader, yields_reader.cluster_yields)
    with pytest.raises(TypeError):
        cy[123]  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError):
        cy[("ccsn", "sukhbold16")]
    with pytest.raises(TypeError):
        cy[("ccsn", "sukhbold16", "Fe56", "extra")]


def test_galaxy_yields_matches_cluster_yields_channels(yields_reader: slug_reader):
    """galaxy_yields shares the same channels/models/isotopes as cluster_yields."""
    cy = cast(slug_yields_reader, yields_reader.cluster_yields)
    gy = cast(slug_yields_reader, yields_reader.galaxy_yields)
    assert gy.channels == cy.channels
    assert gy.models == cy.models
    assert gy.isotopes == cy.isotopes
    assert gy.decomposed == cy.decomposed


def test_galaxy_yields_getitem_shape(yields_reader: slug_reader):
    """galaxy_yields indexes the same way as cluster_yields, just with fewer (one per trial/time) rows."""
    gy = cast(slug_yields_reader, yields_reader.galaxy_yields)
    fe56 = cast(u.Quantity, gy["Fe56"])
    assert fe56.unit == u.Msun
    assert fe56.shape[1] == 2
