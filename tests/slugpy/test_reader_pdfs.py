"""
Unit tests for slug_reader.pdfs, which rebuilds every PDF a run used
from its output file's own PDFs group (one toml descriptor per PDF,
written by OutputManagerH5 via PDF.toToml()), without needing the PDF
files the input deck names.

Exact, draw-for-draw equality of a PDF read back this way is tested
in C++ (tests/pdfs/testPDFSerialize.hpp), since slug's rng cannot be
seeded from Python. Runs a small simulation with the same committed MIST_test-based
fixture deck (tests/core/assets/testCluster.in) that test_run_sim.py
uses, so it needs no data fetched separately. Every test that runs a
simulation chdir's into tmp_path first, so the output lands there
rather than in the repository.

:copyright: Copyright (c) 2026 Mark Krumholz
"""

import pathlib
import tomllib

import h5py
import pytest
from slugpy._slug import PDF, SimControls

from slugpy import run_sim
from slugpy.slug_reader import slug_reader

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CLUSTER_DECK = str(REPO_ROOT / "tests" / "core" / "assets" / "testCluster.in")
OLD_OUTPUT_H5 = str(REPO_ROOT / "examples" / "clusterlib" / "clusterlib.h5")


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> slug_reader:
    """
    Run CLUSTER_DECK once, in a temporary directory, and return a
    reader on its output.

    Parameters
    ----------
    tmp_path_factory : pytest.TempPathFactory
        pytest's own factory for module-scoped temporary directories.

    Returns
    -------
    slug_reader
        Reader on the run's own output file.
    """
    out_dir = tmp_path_factory.mktemp("reader_pdfs")
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(out_dir)
        data = run_sim(CLUSTER_DECK, progress=False)
    assert isinstance(data, slug_reader)
    return data


def _same_pdf(a: PDF, b: PDF) -> bool:
    """
    Check that two PDFs are identical: same toml text, weights, and limits.

    Parameters
    ----------
    a, b : PDF
        The PDFs to compare.

    Returns
    -------
    bool
        True if a and b serialize to the same text and have identical
        weights and limits.
    """
    return (a.toToml() == b.toToml() and a.getWeights() == b.getWeights()
        and a.getMin() == b.getMin() and a.getMax() == b.getMax())


def test_pdfs_keys(run: slug_reader) -> None:
    """The run's valid PDFs are all present, under their accessor names."""
    assert {"imf", "cmf", "fehDist", "avNebFac"} <= set(run.pdfs)
    assert set(run.pdfs) <= {"imf", "cmf", "fehDist", "sfr", "sfrDist", "clf",
        "avDist", "avDistField", "avNebFac", "outTimeDist"}
    assert all(isinstance(p, PDF) for p in run.pdfs.values())


def test_pdfs_match_controls(run: slug_reader) -> None:
    """Each rebuilt PDF is identical to the one SimControls builds from
    the input deck (and its PDF files) directly."""
    controls = SimControls(CLUSTER_DECK)
    assert _same_pdf(run.pdfs["imf"], controls.imf)
    assert _same_pdf(run.pdfs["cmf"], controls.cmf)
    assert _same_pdf(run.pdfs["fehDist"], controls.feH)
    assert _same_pdf(run.pdfs["avNebFac"], controls.avNebFac)


def test_pdfs_cached_and_readonly(run: slug_reader) -> None:
    """pdfs is built once and the same object returned thereafter, and
    cannot be assigned to."""
    assert run.pdfs is run.pdfs
    with pytest.raises(AttributeError):
        run.pdfs = {}
    # The returned mapping is a read-only view of the cache, so it
    # cannot be modified in place either
    with pytest.raises(TypeError):
        run.pdfs["imf"] = run.pdfs["cmf"]  # pyright: ignore[reportIndexIssue]


def test_pdfs_group_is_self_contained(run: slug_reader) -> None:
    """Every stored descriptor is advanced format, so it names no
    breakpoints or files and needs nothing outside the output file."""
    with h5py.File(run._file[0], "r") as f:
        group = f["PDFs"]
        assert isinstance(group, h5py.Group)
        for value in group.attrs.values():
            text = value.decode() if isinstance(value, bytes) else str(value)
            descriptor = tomllib.loads(text)
            assert descriptor["format"] == "advanced"
            assert "breakpoints" not in descriptor


def test_pdfs_empty_for_older_files() -> None:
    """A file written before the PDFs group existed gives an empty dict."""
    with h5py.File(OLD_OUTPUT_H5, "r") as f:
        assert "PDFs" not in f
    assert slug_reader(OLD_OUTPUT_H5).pdfs == {}


# ---------------------------------------------------------------------
# slug_reader.controls built from the stored PDFs
# ---------------------------------------------------------------------

FEH_DESCRIPTOR = """\
breakpoints = [-0.5, 0.2]
[segment1]
type = "powerlaw"
slope = 0.0
"""


def _run_with_feh_file(run_dir: pathlib.Path) -> slug_reader:
    """
    Run CLUSTER_DECK in run_dir with stars.FeH read from a PDF file.

    Parameters
    ----------
    run_dir : pathlib.Path
        Directory to run in; feh.toml is written there, and the deck
        names it by that relative path.

    Returns
    -------
    slug_reader
        Reader on the run's own output file.
    """
    (run_dir / "feh.toml").write_text(FEH_DESCRIPTOR)
    deck = pathlib.Path(CLUSTER_DECK).read_text().replace("FeH = 0.0", 'FeH = "feh.toml"')
    assert 'FeH = "feh.toml"' in deck
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(run_dir)
        data = run_sim(deck, progress=False)
    assert isinstance(data, slug_reader)
    return data


def test_controls_without_pdf_files(tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch) -> None:
    """controls builds from the stored PDFs once the PDF file the deck
    names is gone, matches them exactly, leaves input_deck untouched,
    and lets get_cluster reproduce the recorded [Fe/H]."""
    data = _run_with_feh_file(tmp_path)
    (tmp_path / "feh.toml").rename(tmp_path / "feh_moved.toml")
    monkeypatch.chdir(tmp_path)

    reader = slug_reader(data._file[0])
    assert _same_pdf(reader.controls.feH, reader.pdfs["fehDist"])
    assert reader.controls.feH.getMin() == -0.5
    assert reader.input_deck["stars"]["FeH"] == "feh.toml"

    clusters = reader.clusters
    assert clusters is not None
    uid = int(clusters["uid"][0])
    assert reader.get_cluster(uid).feH() == float(clusters["feh"][0])


def test_controls_unchanged_without_stored_pdfs(tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch) -> None:
    """For a file with no PDFs group, controls is built from the input
    deck as-is: it needs the PDF file the deck names, and fails without
    it, exactly as before stored PDFs existed."""
    data = _run_with_feh_file(tmp_path)
    with h5py.File(data._file[0], "a") as f:
        del f["PDFs"]
    monkeypatch.chdir(tmp_path)

    (tmp_path / "feh.toml").rename(tmp_path / "feh_moved.toml")
    with pytest.raises(RuntimeError):
        _ = slug_reader(data._file[0]).controls

    (tmp_path / "feh_moved.toml").rename(tmp_path / "feh.toml")
    reader = slug_reader(data._file[0])
    assert reader.pdfs == {}
    assert reader.controls.feH.getMin() == -0.5
