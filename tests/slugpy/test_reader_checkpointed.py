"""
Unit tests for slug_reader's support for checkpointed and/or
h5divided output: resolving a run's own model name (rather than a
single .h5 path) to every underlying file it left behind, aggregating
extensible datasets across them while reading fixed ones (e.g. "wl")
from just one, and reading trials_completed/restart_uid from the most
recent checkpoint specifically.

Uses small, hand-built HDF5 fixtures (mirroring test_readers.py's own
_make_galaxy_test_file/_make_phot_convert_test_file pattern) rather
than a real multi-checkpoint slug run, since the reader-side logic
under test only cares about the on-disk file layout and attribute/
dataset shapes OutputManagerH5 produces, not the physics that filled
them in.

:copyright: Copyright (c) 2026 Mark Krumholz
"""

from pathlib import Path
from typing import cast

import h5py
import numpy as np
import pytest
from astropy import units as u

from slugpy.slug_reader import slug_reader


def _write_output_file(path: Path, *, trials: list[int], uids: list[int],
    trials_completed: int | None = None, restart_uid: int | None = None) -> None:
    """
    Write one slug-shaped HDF5 output file at path: top-level
    attributes (slug-hash/date/time/rng_state always; trials_completed/
    restart_uid only if given -- omitting them mirrors a checkpoint
    that was still open when read, see OutputManagerH5::
    closeOutputFile()'s own comment), an input_deck group, and a
    clusters group with one row per entry of trials/uids -- trial and
    uid are extensible (maxshape=(None,), matching
    createExtensible1dDataset() on the C++ side) so slug_reader
    aggregates them across every file it spans; target_mass is
    likewise extensible. A cluster_spectra group is also written, with
    a *fixed* "wl" dataset (no maxshape, matching writeFixed1dDataset()
    on the C++ side -- physically identical in every file, so must be
    read from one file only, not concatenated) and an extensible 2D
    "spec" dataset, one row per trial.
    """
    with h5py.File(path, "w") as f:
        f.attrs["slug-hash"] = "deadbeef"
        f.attrs["date"] = "2026-01-01"
        f.attrs["time"] = "00:00:00"
        f.attrs["rng_state"] = "x"
        if trials_completed is not None:
            f.attrs["trials_completed"] = trials_completed
        if restart_uid is not None:
            f.attrs["restart_uid"] = restart_uid
        f.create_group("input_deck").create_dataset(
            "toml", data="n_trial = 1\nsim_type = \"cluster\"\n")

        n = len(trials)
        g = f.create_group("clusters")
        trial = g.create_dataset("trial", data=np.array(trials, dtype="u8"), maxshape=(None,))
        trial.attrs["units"] = ""
        uid = g.create_dataset("uid", data=np.array(uids, dtype="u8"), maxshape=(None,))
        uid.attrs["units"] = ""
        target_mass = g.create_dataset(
            "target_mass", data=np.array([1e3] * n), maxshape=(None,))
        target_mass.attrs["units"] = "Msun"

        cs = f.create_group("cluster_spectra")
        wl = cs.create_dataset("wl", data=np.array([1000.0, 2000.0]))  # fixed: no maxshape
        wl.attrs["units"] = "Angstrom"
        spec = cs.create_dataset(
            "spec", data=np.array([[1.0, 2.0]] * n), maxshape=(None, 2))
        spec.attrs["units"] = "erg/s/Angstrom"


# ---------------------------------------------------------------------
# File discovery: plain (non-checkpointed) output
# ---------------------------------------------------------------------

def test_model_name_resolves_plain_h5(tmp_path: Path) -> None:
    """A model name with only model.h5 on disk (no checkpointing, no
    h5divided) resolves to that single file -- the ordinary case."""
    path = tmp_path / "model.h5"
    _write_output_file(path, trials=[0], uids=[0], trials_completed=1, restart_uid=1)

    reader = slug_reader(str(tmp_path / "model"))
    assert reader.slug_hash == "deadbeef"
    assert reader.trials_completed == 1
    assert reader.restart_uid == 1
    clusters = reader.clusters
    assert clusters is not None
    assert cast(np.ndarray, clusters["trial"]).tolist() == [0]


def test_model_name_resolves_thread_dir_without_checkpointing(tmp_path: Path) -> None:
    """h5divided output without checkpointing (model/thread_NNNN.h5,
    never consolidated) is found and its per-thread files aggregated."""
    thread_dir = tmp_path / "model"
    thread_dir.mkdir()
    _write_output_file(thread_dir / "thread_0000.h5", trials=[0], uids=[0],
        trials_completed=2, restart_uid=2)
    _write_output_file(thread_dir / "thread_0001.h5", trials=[1], uids=[1],
        trials_completed=2, restart_uid=2)

    reader = slug_reader(str(tmp_path / "model"))
    assert len(reader._file) == 2  # noqa: SLF001 -- exercising the discovery/aggregation contract directly
    clusters = reader.clusters
    assert clusters is not None
    assert sorted(cast(np.ndarray, clusters["trial"]).tolist()) == [0, 1]


def test_model_name_raises_when_nothing_matches(tmp_path: Path) -> None:
    """A model name with no matching output at all raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        slug_reader(str(tmp_path / "nonexistent_model"))


def test_literal_h5_path_raises_when_missing(tmp_path: Path) -> None:
    """A literal .h5 path that doesn't exist still raises FileNotFoundError (not silently treated as a model name)."""
    with pytest.raises(FileNotFoundError):
        slug_reader(str(tmp_path / "nonexistent.h5"))


# ---------------------------------------------------------------------
# File discovery + aggregation: checkpointed output
# ---------------------------------------------------------------------

def test_checkpoints_consolidated_files_aggregate_and_order(tmp_path: Path) -> None:
    """Three consolidated checkpoint files (model_chkNNNNN.h5) are all
    found, in ascending checkpoint order, and their clusters rows
    pooled together."""
    _write_output_file(tmp_path / "model_chk00000.h5", trials=[0, 1], uids=[0, 1],
        trials_completed=2, restart_uid=2)
    _write_output_file(tmp_path / "model_chk00001.h5", trials=[2, 3], uids=[2, 3],
        trials_completed=4, restart_uid=4)
    _write_output_file(tmp_path / "model_chk00002.h5", trials=[4], uids=[4],
        trials_completed=5, restart_uid=5)

    reader = slug_reader(str(tmp_path / "model"))
    assert reader._file == [  # noqa: SLF001
        str(tmp_path / "model_chk00000.h5"),
        str(tmp_path / "model_chk00001.h5"),
        str(tmp_path / "model_chk00002.h5"),
    ]
    clusters = reader.clusters
    assert clusters is not None
    assert sorted(cast(np.ndarray, clusters["trial"]).tolist()) == [0, 1, 2, 3, 4]


def test_checkpoints_thread_dirs_aggregate(tmp_path: Path) -> None:
    """An unconsolidated checkpoint (model_chkNNNNN/thread_NNNN.h5)
    alongside a consolidated one: every thread file of the
    unconsolidated checkpoint, plus the consolidated one, all
    contribute their own rows."""
    chk0_dir = tmp_path / "model_chk00000"
    chk0_dir.mkdir()
    _write_output_file(chk0_dir / "thread_0000.h5", trials=[0], uids=[0],
        trials_completed=2, restart_uid=2)
    _write_output_file(chk0_dir / "thread_0001.h5", trials=[1], uids=[1],
        trials_completed=2, restart_uid=2)
    _write_output_file(tmp_path / "model_chk00001.h5", trials=[2], uids=[2],
        trials_completed=3, restart_uid=3)

    reader = slug_reader(str(tmp_path / "model"))
    assert len(reader._file) == 3  # noqa: SLF001
    clusters = reader.clusters
    assert clusters is not None
    assert sorted(cast(np.ndarray, clusters["trial"]).tolist()) == [0, 1, 2]


def test_trials_completed_and_restart_uid_from_last_checkpoint(tmp_path: Path) -> None:
    """trials_completed/restart_uid come from the highest-numbered
    checkpoint, not the first -- unlike slug_hash/date/time/rng_state,
    they differ from one checkpoint to the next."""
    _write_output_file(tmp_path / "model_chk00000.h5", trials=[0], uids=[0],
        trials_completed=5, restart_uid=5)
    _write_output_file(tmp_path / "model_chk00001.h5", trials=[1], uids=[1],
        trials_completed=17, restart_uid=23)

    reader = slug_reader(str(tmp_path / "model"))
    assert reader.trials_completed == 17
    assert reader.restart_uid == 23


def test_trials_completed_and_restart_uid_none_when_last_checkpoint_incomplete(tmp_path: Path) -> None:
    """If the most recent checkpoint has no trials_completed/
    restart_uid attribute at all (e.g. it was still open when this was
    read), both come back None rather than raising -- see
    OutputManagerH5::closeOutputFile()'s own comment on when these are
    written."""
    _write_output_file(tmp_path / "model_chk00000.h5", trials=[0], uids=[0],
        trials_completed=5, restart_uid=5)
    _write_output_file(tmp_path / "model_chk00001.h5", trials=[1], uids=[1])  # no attrs written

    reader = slug_reader(str(tmp_path / "model"))
    assert reader.trials_completed is None
    assert reader.restart_uid is None


def test_fixed_dataset_read_once_not_duplicated(tmp_path: Path) -> None:
    """wl (fixed, not extensible) is read from one file only -- pooling
    it across every checkpoint the way extensible datasets are pooled
    would wrongly duplicate it once per checkpoint."""
    _write_output_file(tmp_path / "model_chk00000.h5", trials=[0], uids=[0],
        trials_completed=1, restart_uid=1)
    _write_output_file(tmp_path / "model_chk00001.h5", trials=[1], uids=[1],
        trials_completed=2, restart_uid=2)

    reader = slug_reader(str(tmp_path / "model"))
    spectra = reader.cluster_spectra
    assert spectra is not None
    wl = cast(u.Quantity, spectra["wl"])
    assert wl.value.tolist() == [1000.0, 2000.0]  # not duplicated to length 4


def test_extensible_2d_dataset_aggregated_by_row(tmp_path: Path) -> None:
    """spec (extensible, 2D) gets one row per trial pooled across every
    checkpoint, each row's own columns intact."""
    _write_output_file(tmp_path / "model_chk00000.h5", trials=[0], uids=[0],
        trials_completed=1, restart_uid=1)
    _write_output_file(tmp_path / "model_chk00001.h5", trials=[1, 2], uids=[1, 2],
        trials_completed=3, restart_uid=3)

    reader = slug_reader(str(tmp_path / "model"))
    spectra = reader.cluster_spectra
    assert spectra is not None
    spec = cast(u.Quantity, spectra["spec"])
    assert spec.shape == (3, 2)
    assert spec.value.tolist() == [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]


def test_no_checkpoint_beyond_the_last_is_included(tmp_path: Path) -> None:
    """A stray file that doesn't match the _chkNNNNN pattern at all
    (e.g. a leftover from an unrelated model) is not swept in."""
    _write_output_file(tmp_path / "model_chk00000.h5", trials=[0], uids=[0],
        trials_completed=1, restart_uid=1)
    _write_output_file(tmp_path / "other_model_chk00000.h5", trials=[99], uids=[99],
        trials_completed=1, restart_uid=1)

    reader = slug_reader(str(tmp_path / "model"))
    assert reader._file == [str(tmp_path / "model_chk00000.h5")]  # noqa: SLF001


# ---------------------------------------------------------------------
# run_cloudy() and a multi-file reader
# ---------------------------------------------------------------------

def test_run_cloudy_raises_for_multi_file_reader(tmp_path: Path) -> None:
    """run_cloudy() refuses outright when this reader spans more than
    one underlying file, rather than guessing which one to write new
    cluster_cloudy rows into."""
    _write_output_file(tmp_path / "model_chk00000.h5", trials=[0], uids=[0],
        trials_completed=1, restart_uid=1)
    _write_output_file(tmp_path / "model_chk00001.h5", trials=[1], uids=[1],
        trials_completed=2, restart_uid=2)

    reader = slug_reader(str(tmp_path / "model"))
    with pytest.raises(RuntimeError, match="more than one"):
        reader.run_cloudy("cluster")


# ---------------------------------------------------------------------
# Never more than one file open at once
# ---------------------------------------------------------------------

def test_never_more_than_one_file_open_at_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reading an aggregated dataset across several checkpoints opens
    (and closes) one file at a time -- never two concurrently -- per
    this reader's own design (see slug_reader.__init__'s own comment).

    h5py.File.__init__ also fires internally for lightweight, already-
    open-file wrapper objects (e.g. while iterating a group's own
    members) that are never independently closed -- identifiable by
    their own first argument being something other than a path (a
    GroupID, in practice) -- so only path-argument inits, the ones
    that actually open a new OS-level file handle, are counted here;
    every close() call does correspond to one of those (verified: this
    assertion would trivially pass by never exceeding 0 if that were
    not so, since nothing would ever increment).
    """
    for i in range(4):
        _write_output_file(tmp_path / f"model_chk{i:05d}.h5", trials=[i], uids=[i],
            trials_completed=i + 1, restart_uid=i + 1)

    real_file_init = h5py.File.__init__
    open_count = 0
    max_open_count = 0

    def _tracking_init(self, *args, **kwargs):
        nonlocal open_count, max_open_count
        real_file_init(self, *args, **kwargs)
        if args and isinstance(args[0], (str, Path)):
            open_count += 1
            max_open_count = max(max_open_count, open_count)

    real_file_close = h5py.File.close

    def _tracking_close(self, *args, **kwargs):
        nonlocal open_count
        open_count -= 1
        real_file_close(self, *args, **kwargs)

    monkeypatch.setattr(h5py.File, "__init__", _tracking_init)
    monkeypatch.setattr(h5py.File, "close", _tracking_close)

    reader = slug_reader(str(tmp_path / "model"))
    clusters = reader.clusters
    assert clusters is not None
    assert sorted(cast(np.ndarray, clusters["trial"]).tolist()) == [0, 1, 2, 3]

    assert max_open_count == 1
    assert open_count == 0
