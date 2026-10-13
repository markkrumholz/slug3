"""
Unit tests for consolidate_checkpoints: finding a run's checkpoints,
refusing to merge checkpoints that differ in structure or were never
closed cleanly, concatenating per-trial rows while taking metadata from
the final checkpoint, and optionally cleaning up afterward.

Uses small, hand-built HDF5 fixtures shaped like OutputManagerH5's own
output, as test_reader_checkpointed.py does.

:copyright: Copyright (c) 2026 Mark Krumholz
"""

import importlib
from pathlib import Path
from typing import cast

import h5py
import numpy as np
import pytest

from slugpy.consolidate_checkpoints import consolidate_checkpoints
from slugpy.slug_reader import slug_reader

# The module itself, for monkeypatching -- "import
# slugpy.consolidate_checkpoints as cc" would give the function, since
# slugpy/__init__.py rebinds that name to it
cc = importlib.import_module("slugpy.consolidate_checkpoints")


def _write_output_file(path: Path, *, trials: list[int], trials_completed: int | None,
    deck: str = "n_trial = 1\n", wl: tuple[float, float] = (1000.0, 2000.0),
    extra_group: bool = False) -> None:
    """
    Write one slug-shaped HDF5 output file at path.

    Parameters
    ----------
    path : pathlib.Path
        Where to write the file.
    trials : list of int
        One clusters/cluster_spectra row is written per entry; uid is
        set to 10 * trial, and each spectrum row to [trial, NaN].
    trials_completed : int or None
        Value of the trials_completed attribute, or None to omit it
        (as for a checkpoint that was never closed cleanly);
        restart_uid and max_trial are written alongside it.
    deck : str, optional
        Contents of the input_deck/toml dataset.
    wl : tuple of float, optional
        Contents of the fixed cluster_spectra/wl dataset.
    extra_group : bool, optional
        If True, also write an extra galaxy group, so the file's
        structure differs from the default.
    """
    with h5py.File(path, "w") as f:
        f.attrs["slug-hash"] = "deadbeef"
        f.attrs["date"] = "2026-01-01"
        f.attrs["time"] = "00:00:00"
        f.attrs["rng_state"] = f"state-{path.name}"
        if trials_completed is not None:
            f.attrs["trials_completed"] = np.uint64(trials_completed)
            f.attrs["restart_uid"] = np.uint64(10 * trials_completed)
            f.attrs["max_trial"] = np.uint64(trials_completed - 1)
        f.create_group("input_deck").create_dataset("toml", data=deck.encode())

        n = len(trials)
        g = f.create_group("clusters")
        g.attrs["note"] = f"from {path.name}"
        trial = g.create_dataset("trial", data=np.array(trials, dtype="u8"),
            maxshape=(None,), chunks=(1,))
        trial.attrs["units"] = ""
        g.create_dataset("uid", data=10 * np.array(trials, dtype="u8"),
            maxshape=(None,), chunks=(1,))

        cs = f.create_group("cluster_spectra")
        wl_dset = cs.create_dataset("wl", data=np.array(wl))
        wl_dset.attrs["units"] = "Angstrom"
        spec = np.array([[float(t), np.nan] for t in trials]).reshape(n, 2)
        spec_dset = cs.create_dataset("spec", data=spec, maxshape=(None, 2), chunks=(1, 2))
        spec_dset.attrs["units"] = "erg/s/Angstrom"

        if extra_group:
            f.create_group("galaxy").create_dataset(
                "trial", data=np.array(trials, dtype="u8"), maxshape=(None,))


def _make_mixed_run(tmp_path: Path) -> None:
    """
    Write a three-checkpoint run: a consolidated model_chk00000.h5, a
    per-thread model_chk00001/ directory, and a per-thread
    model_chk00002/ directory holding the final checkpoint, whose
    deck and wl differ from the earlier ones.
    """
    _write_output_file(tmp_path / "model_chk00000.h5", trials=[0, 1], trials_completed=2)
    d1 = tmp_path / "model_chk00001"
    d1.mkdir()
    _write_output_file(d1 / "thread_0000.h5", trials=[2], trials_completed=4)
    _write_output_file(d1 / "thread_0001.h5", trials=[3], trials_completed=4)
    d2 = tmp_path / "model_chk00002"
    d2.mkdir()
    _write_output_file(d2 / "thread_0000.h5", trials=[4], trials_completed=6,
        deck="n_trial = 6\n# restarted\n", wl=(1500.0, 2500.0))
    _write_output_file(d2 / "thread_0001.h5", trials=[], trials_completed=6,
        deck="n_trial = 6\n# restarted\n", wl=(1500.0, 2500.0))


def test_rows_concatenated_in_checkpoint_order(tmp_path: Path) -> None:
    """Every extensible dataset's rows are concatenated across every
    checkpoint file, in checkpoint (then thread) order, NaNs intact."""
    _make_mixed_run(tmp_path)
    dest = consolidate_checkpoints("model", tmp_path)
    assert dest == tmp_path / "model.h5"
    with h5py.File(dest, "r") as f:
        assert f["clusters/trial"][()].tolist() == [0, 1, 2, 3, 4]
        assert f["clusters/uid"][()].tolist() == [0, 10, 20, 30, 40]
        spec = cast(np.ndarray, f["cluster_spectra/spec"][()])
        assert spec.shape == (5, 2)
        assert spec[:, 0].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
        assert np.all(np.isnan(spec[:, 1]))
        assert cast(h5py.Dataset, f["clusters/trial"]).maxshape == (None,)
        assert cast(h5py.Dataset, f["cluster_spectra/spec"]).maxshape == (None, 2)


def test_metadata_from_final_checkpoint(tmp_path: Path) -> None:
    """Top-level attributes, fixed datasets, the input deck, and group/
    dataset attributes all come from the final checkpoint."""
    _make_mixed_run(tmp_path)
    dest = consolidate_checkpoints("model", tmp_path)
    with h5py.File(dest, "r") as f:
        assert f.attrs["trials_completed"] == 6
        assert f.attrs["restart_uid"] == 60
        assert f.attrs["max_trial"] == 5
        assert f.attrs["rng_state"] == "state-thread_0000.h5"
        assert f["input_deck/toml"][()] == b"n_trial = 6\n# restarted\n"
        assert f["cluster_spectra/wl"][()].tolist() == [1500.0, 2500.0]
        assert f["cluster_spectra/wl"].attrs["units"] == "Angstrom"
        assert f["cluster_spectra/spec"].attrs["units"] == "erg/s/Angstrom"
        assert f["clusters"].attrs["note"] == "from thread_0000.h5"


def test_pdfs_group_copied(tmp_path: Path) -> None:
    """The PDFs group -- string attributes only, no datasets -- is
    carried over from the final checkpoint, so slug_reader.pdfs works
    on the consolidated file."""
    _make_mixed_run(tmp_path)
    descriptor = ('format = "advanced"\nmethod = "stop_nearest"\nnormalized = true\n'
        '\n[segment1]\nmax = 0.5\nmin = 0.5\ntype = "delta"\nweight = 1.0\n')
    for path in sorted(tmp_path.rglob("*.h5")):
        with h5py.File(path, "a") as f:
            f.create_group("PDFs").attrs["fehDist"] = descriptor
    dest = consolidate_checkpoints("model", tmp_path)
    with h5py.File(dest, "r") as f:
        assert f["PDFs"].attrs["fehDist"] == descriptor
    pdfs = slug_reader(str(dest)).pdfs
    assert list(pdfs) == ["fehDist"]
    assert pdfs["fehDist"].getMin() == 0.5


def test_result_readable_by_slug_reader(tmp_path: Path) -> None:
    """slug_reader reads the consolidated file back as one file."""
    _make_mixed_run(tmp_path)
    dest = consolidate_checkpoints("model", tmp_path)
    reader = slug_reader(str(dest))
    assert reader.trials_completed == 6
    clusters = reader.clusters
    assert clusters is not None
    assert cast(np.ndarray, clusters["trial"]).tolist() == [0, 1, 2, 3, 4]


def test_keeps_checkpoints_by_default(tmp_path: Path) -> None:
    """Without cleanup, every checkpoint is left in place."""
    _make_mixed_run(tmp_path)
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))
    consolidate_checkpoints("model", tmp_path)
    after = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))
    assert after == sorted([*before, Path("model.h5")])


def test_cleanup_removes_checkpoints(tmp_path: Path) -> None:
    """With cleanup=True, every checkpoint file and directory is
    removed, leaving just the consolidated file."""
    _make_mixed_run(tmp_path)
    consolidate_checkpoints("model", tmp_path, cleanup=True)
    assert [p.name for p in tmp_path.iterdir()] == ["model.h5"]


def test_cleanup_leaves_directory_with_other_files(tmp_path: Path) -> None:
    """A checkpoint directory holding files other than slug output has
    its slug output removed but is itself left in place, with a
    warning."""
    _make_mixed_run(tmp_path)
    (tmp_path / "model_chk00001" / "notes.txt").write_text("keep me")
    with pytest.warns(UserWarning, match="model_chk00001"):
        consolidate_checkpoints("model", tmp_path, cleanup=True)
    assert [p.name for p in (tmp_path / "model_chk00001").iterdir()] == ["notes.txt"]
    assert not (tmp_path / "model_chk00002").exists()


def test_cleanup_skips_directory_shadowed_by_file(tmp_path: Path) -> None:
    """When a checkpoint exists both as a file and a directory, the
    file is the one consolidated (as on restart), so cleanup never
    touches the directory's contents."""
    _write_output_file(tmp_path / "model_chk00000.h5", trials=[0], trials_completed=1)
    shadowed = tmp_path / "model_chk00000"
    shadowed.mkdir()
    _write_output_file(shadowed / "thread_0000.h5", trials=[7], trials_completed=1)
    dest = consolidate_checkpoints("model", tmp_path, cleanup=True)
    with h5py.File(dest, "r") as f:
        assert f["clusters/trial"][()].tolist() == [0]
    assert (shadowed / "thread_0000.h5").is_file()
    assert not (tmp_path / "model_chk00000.h5").exists()


def test_ignores_non_matching_names(tmp_path: Path) -> None:
    """Only model_chkNNNNN (exactly five digits) is a checkpoint of
    model -- other models' checkpoints and wrong-width numbers are
    ignored, as they are on restart."""
    _write_output_file(tmp_path / "model_chk00000.h5", trials=[0], trials_completed=1)
    _write_output_file(tmp_path / "model_chk0001.h5", trials=[8], trials_completed=2)
    _write_output_file(tmp_path / "model2_chk00001.h5", trials=[9], trials_completed=2)
    dest = consolidate_checkpoints("model", tmp_path)
    with h5py.File(dest, "r") as f:
        assert f["clusters/trial"][()].tolist() == [0]


def test_default_out_dir_is_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With no out_dir, checkpoints are looked for in (and the result
    written to) the current working directory."""
    _make_mixed_run(tmp_path)
    monkeypatch.chdir(tmp_path)
    dest = consolidate_checkpoints("model")
    assert dest == tmp_path.resolve() / "model.h5"
    assert dest.is_file()


def test_refuses_structure_mismatch(tmp_path: Path) -> None:
    """Checkpoints with different HDF5 groups are refused, and nothing
    is written or deleted."""
    _write_output_file(tmp_path / "model_chk00000.h5", trials=[0], trials_completed=1)
    _write_output_file(tmp_path / "model_chk00001.h5", trials=[1], trials_completed=2,
        extra_group=True)
    with pytest.raises(ValueError, match="galaxy"):
        consolidate_checkpoints("model", tmp_path, cleanup=True)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "model_chk00000.h5", "model_chk00001.h5"]


def test_refuses_extensible_dtype_mismatch(tmp_path: Path) -> None:
    """Extensible datasets whose element type differs between
    checkpoints cannot be concatenated, so are refused."""
    _write_output_file(tmp_path / "model_chk00000.h5", trials=[0], trials_completed=1)
    _write_output_file(tmp_path / "model_chk00001.h5", trials=[1], trials_completed=2)
    with h5py.File(tmp_path / "model_chk00001.h5", "a") as f:
        del f["clusters/uid"]
        f["clusters"].create_dataset("uid", data=np.array([1.0]), maxshape=(None,))
    with pytest.raises(ValueError, match="clusters/uid"):
        consolidate_checkpoints("model", tmp_path)
    assert not (tmp_path / "model.h5").exists()


def test_refuses_unclosed_checkpoint(tmp_path: Path) -> None:
    """A checkpoint file without trials_completed (never closed
    cleanly) is refused."""
    _write_output_file(tmp_path / "model_chk00000.h5", trials=[0], trials_completed=1)
    _write_output_file(tmp_path / "model_chk00001.h5", trials=[1], trials_completed=None)
    with pytest.raises(ValueError, match="trials_completed"):
        consolidate_checkpoints("model", tmp_path)
    assert not (tmp_path / "model.h5").exists()


def test_refuses_existing_destination(tmp_path: Path) -> None:
    """An existing model.h5 is never overwritten."""
    _write_output_file(tmp_path / "model_chk00000.h5", trials=[0], trials_completed=1)
    (tmp_path / "model.h5").write_text("precious")
    with pytest.raises(FileExistsError):
        consolidate_checkpoints("model", tmp_path, cleanup=True)
    assert (tmp_path / "model.h5").read_text() == "precious"
    assert (tmp_path / "model_chk00000.h5").is_file()


def test_refuses_destination_created_during_consolidation(tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch) -> None:
    """A model.h5 that appears while consolidation is running is never
    overwritten, and the checkpoints are not cleaned up."""
    _write_output_file(tmp_path / "model_chk00000.h5", trials=[0], trials_completed=1)
    real_same_rows = cc._same_rows

    def same_rows_creating_dest(a: np.ndarray, b: np.ndarray) -> bool:
        (tmp_path / "model.h5").write_text("precious")
        return real_same_rows(a, b)

    monkeypatch.setattr(cc, "_same_rows", same_rows_creating_dest)
    with pytest.raises(FileExistsError, match="while consolidating"):
        consolidate_checkpoints("model", tmp_path, cleanup=True)
    assert (tmp_path / "model.h5").read_text() == "precious"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["model.h5", "model_chk00000.h5"]


def test_raises_without_checkpoints(tmp_path: Path) -> None:
    """A model with no checkpoints at all raises FileNotFoundError."""
    _write_output_file(tmp_path / "model.h5", trials=[0], trials_completed=1)
    with pytest.raises(FileNotFoundError):
        consolidate_checkpoints("other", tmp_path)


def test_failed_verification_writes_and_deletes_nothing(tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch) -> None:
    """If the read-back check fails, no model.h5 (or temporary file) is
    left behind, and cleanup does not run."""
    _make_mixed_run(tmp_path)
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))
    monkeypatch.setattr(cc, "_same_rows", lambda a, b: False)
    with pytest.raises(RuntimeError, match="does not match"):
        consolidate_checkpoints("model", tmp_path, cleanup=True)
    after = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))
    assert after == before


def test_copies_in_bounded_blocks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Rows are copied and verified correctly when a dataset spans
    several blocks."""
    _write_output_file(tmp_path / "model_chk00000.h5", trials=list(range(7)), trials_completed=7)
    _write_output_file(tmp_path / "model_chk00001.h5", trials=list(range(7, 12)),
        trials_completed=12)
    monkeypatch.setattr(cc, "_BLOCK_BYTES", 16)
    dest = consolidate_checkpoints("model", tmp_path)
    with h5py.File(dest, "r") as f:
        assert f["clusters/trial"][()].tolist() == list(range(12))
        assert cast(np.ndarray, f["cluster_spectra/spec"][()])[:, 0].tolist() == \
            [float(t) for t in range(12)]
