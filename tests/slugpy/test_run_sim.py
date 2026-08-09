"""
Unit tests for slugpy.run_sim, which drives an entire slug simulation
end to end from Python, exactly as the slug command-line executable
does.

Uses the same small, committed MIST_test-based fixtures the C++ core
tests use (tests/core/assets/testCluster.in, testGalaxy.in), so this
needs no data fetched separately and can stay a "quick" test. Every
test below chdir's into tmp_path (via monkeypatch.chdir) before
calling run_sim, so the output files a simulation run writes don't
land in the repo itself; the deck paths themselves are captured as
absolute paths at collection time (while cwd is still the repo root),
since SimControls's own path argument has no SLUG_DIR/REPO_DIR
fallback the way paths *inside* a deck do (see test_readers.py's
identical CLUSTERLIB_DIR/get_cluster comment for the same rule).
"""

import pathlib
from typing import cast

import pytest
from astropy import units as u

from slugpy import run_sim
from slugpy.slug_reader import slug_reader

REPO_ROOT = pathlib.Path.cwd()
CLUSTER_DECK = str(REPO_ROOT / "tests" / "core" / "assets" / "testCluster.in")
GALAXY_DECK = str(REPO_ROOT / "tests" / "core" / "assets" / "testGalaxy.in")

# A deck otherwise identical to testCluster.in, but requesting ASCII
# output; built as literal text (rather than a fixture file) purely to
# keep the ASCII-specific test self-contained.
ASCII_DECK = f"""
sim_type = "cluster"
n_trial = 1

[stars]
IMF = "chabrier.toml"
track_registry = "{(REPO_ROOT / "tests" / "tracks" / "assets" / "tracks.toml").as_posix()}"
tracks = "MIST_test"
FeH = 0.0
alphaFe = -0.2

[spectra]
model = "blackbody"

[clusters]
CMF = 1e3

[outputs]
start_time = 0.0
end_time = 10.0
ntime = 3
output_mode = "ascii"
"""


def test_run_sim_cluster_deck(tmp_path, monkeypatch):
    """A cluster-type deck runs to completion and returns a slug_reader
    with the expected data (one cluster, at CLUSTER_DECK's own fixed
    clusters.CMF, advanced through its own three output times)."""
    monkeypatch.chdir(tmp_path)
    data = run_sim(CLUSTER_DECK)

    assert isinstance(data, slug_reader)
    assert data.clusters is not None
    assert list(cast(u.Quantity, data.clusters["target_mass"]).value) == [1000.0]
    assert data.cluster_spectra is not None
    assert data.cluster_spectra["spec"].shape[0] == 3


def test_run_sim_accepts_literal_toml_content(tmp_path, monkeypatch):
    """input_deck may be the deck's own text, not just a path to it --
    same rule as SimControls's own constructor."""
    deck_text = pathlib.Path(CLUSTER_DECK).read_text()
    monkeypatch.chdir(tmp_path)
    data = run_sim(deck_text)

    assert isinstance(data, slug_reader)
    assert data.clusters is not None
    assert list(cast(u.Quantity, data.clusters["target_mass"]).value) == [1000.0]


def test_run_sim_galaxy_deck_is_a_noop(tmp_path, monkeypatch):
    """A galaxy-type deck still produces a valid, readable output file
    (matching main.cpp's own current behavior), but with no clusters
    written, since the galaxy branch is not yet implemented."""
    monkeypatch.chdir(tmp_path)
    data = run_sim(GALAXY_DECK)

    assert isinstance(data, slug_reader)
    assert data.clusters is not None
    assert len(data.clusters["uid"]) == 0


def test_run_sim_ascii_deck_warns_and_returns_none(tmp_path, monkeypatch):
    """An ASCII-output deck still runs to completion and writes its own
    output files, but run_sim warns and returns None instead of
    attempting to read back a nonexistent HDF5 file."""
    monkeypatch.chdir(tmp_path)

    with pytest.warns(UserWarning, match="ASCII output"):
        result = run_sim(ASCII_DECK)

    assert result is None
    assert (tmp_path / "slug_sim_summary.txt").exists()
    assert (tmp_path / "slug_sim_clusters.txt").exists()
