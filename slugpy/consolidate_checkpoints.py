"""
consolidate_checkpoints.py

Implements consolidate_checkpoints, which merges the checkpoints a
checkpointed slug run left behind into a single output file.

:copyright: Copyright (c) 2026 Mark Krumholz
"""

import os
import warnings
from pathlib import Path
from typing import cast

import h5py
import numpy as np

from .slug_reader import _find_checkpoints

# Upper bound on how much of one dataset is held in memory at once
# while copying or verifying rows, in bytes
_BLOCK_BYTES = 1 << 26

# One entry per HDF5 object path in a file (excluding the root group):
# ("group",) for a group, ("fixed",) for a dataset with no unlimited
# dimension, or ("extensible", dtype, trailing shape) for a dataset that
# has one, where trailing shape is every dimension but the first
_Signature = dict[str, tuple[object, ...]]


def _is_extensible(dset: h5py.Dataset) -> bool:
    """
    Check whether a dataset holds per-trial rows.

    Parameters
    ----------
    dset : h5py.Dataset
        The dataset to check.

    Returns
    -------
    bool
        True if any of dset's dimensions is unlimited -- the same rule
        utils::isExtensibleDataset() uses on the C++ side to decide
        which datasets OutputManagerH5::consolidateFiles() appends,
        rather than treating as shared metadata.
    """
    maxshape = cast(tuple[int | None, ...] | None, dset.maxshape)
    return maxshape is not None and any(m is None for m in maxshape)


def _signature(path: str) -> tuple[_Signature, dict[str, int]]:
    """
    Describe the group/dataset structure of one HDF5 file.

    Parameters
    ----------
    path : str
        The file to describe.

    Returns
    -------
    (signature, rows) : tuple of (dict, dict of str to int)
        signature : the file's structure (see _Signature). Fixed
        datasets record only that they exist, not their type or shape,
        since only the final checkpoint's copy of them is kept (and
        e.g. the input deck's string length can legitimately change
        across a restart); extensible datasets also record their dtype
        and trailing shape, which must match for their rows to be
        concatenated.
        rows : the number of rows in each extensible dataset, keyed by
        object path.
    """
    signature: _Signature = {}
    rows: dict[str, int] = {}

    def visit(name: str, obj: object) -> None:
        if isinstance(obj, h5py.Group):
            signature[name] = ("group",)
        elif isinstance(obj, h5py.Dataset):
            dset = cast(h5py.Dataset, obj)
            shape = cast(tuple[int, ...], dset.shape)
            if _is_extensible(dset):
                signature[name] = ("extensible", np.dtype(dset.dtype), shape[1:])
                rows[name] = shape[0]
            else:
                signature[name] = ("fixed",)

    with h5py.File(path, "r") as f:
        f.visititems(visit)
    return signature, rows


def _describe_mismatch(sig: _Signature, ref: _Signature) -> str:
    """
    Summarize how one file's signature differs from a reference one.

    Parameters
    ----------
    sig : _Signature
        Signature of the file being checked.
    ref : _Signature
        Signature of the reference file.

    Returns
    -------
    str
        A short, human-readable description of the first few
        differences.
    """
    problems: list[str] = []
    for name in sorted(ref.keys() - sig.keys()):
        problems.append(f"missing {name}")
    for name in sorted(sig.keys() - ref.keys()):
        problems.append(f"unexpected {name}")
    for name in sorted(sig.keys() & ref.keys()):
        if sig[name] != ref[name]:
            problems.append(f"{name} is {sig[name]}, expected {ref[name]}")
    return "; ".join(problems[:5]) + ("; ..." if len(problems) > 5 else "")


def _row_blocks(n_rows: int, dset: h5py.Dataset) -> list[tuple[int, int]]:
    """
    Split a dataset's rows into blocks of bounded size.

    Parameters
    ----------
    n_rows : int
        Number of rows to split.
    dset : h5py.Dataset
        A dataset whose row size sets the block size.

    Returns
    -------
    list of (int, int)
        (start, stop) row ranges covering [0, n_rows), each at most
        roughly _BLOCK_BYTES in size.
    """
    shape = cast(tuple[int, ...], dset.shape)
    row_bytes = max(1, np.dtype(dset.dtype).itemsize * int(np.prod(shape[1:], dtype=np.int64)))
    step = max(1, _BLOCK_BYTES // row_bytes)
    return [(start, min(start + step, n_rows)) for start in range(0, n_rows, step)]


def _same_rows(a: np.ndarray, b: np.ndarray) -> bool:
    """
    Check whether two blocks of rows are bitwise identical.

    Parameters
    ----------
    a, b : numpy.ndarray
        The blocks to compare.

    Returns
    -------
    bool
        True if a and b have the same shape, dtype, and contents --
        compared byte for byte, so NaNs in the same places compare
        equal, unlike with ==.
    """
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype.kind == "O":
        return a.tolist() == b.tolist()
    return a.tobytes() == b.tobytes()


def consolidate_checkpoints(model_name: str, out_dir: str | os.PathLike[str] | None = None,
    cleanup: bool = False) -> Path:
    """
    Merge the checkpoints of a checkpointed slug run into one file.

    Parameters
    ----------
    model_name : str
        The run's model name (output.model_name in its input deck).
    out_dir : str or os.PathLike, optional
        The directory holding the run's output (output.out_dir in its
        input deck). Defaults to the current working directory.
    cleanup : bool, optional
        If True, delete the individual checkpoints once the
        consolidated file has been written and verified. Defaults to
        False.

    Returns
    -------
    pathlib.Path
        Path of the consolidated file, out_dir/model_name.h5.

    Raises
    ------
    FileNotFoundError
        If out_dir holds no checkpoints of model_name.
    FileExistsError
        If out_dir/model_name.h5 already exists, or is created by
        something else while this function is running.
    ValueError
        If the checkpoint files do not all have the same HDF5 group
        and dataset structure, or if any of them lacks the
        trials_completed attribute slug writes when it closes a
        checkpoint (meaning the file was never closed cleanly -- e.g.
        the run is still going, or crashed while writing it).
    RuntimeError
        If the consolidated file fails verification against the
        checkpoints. Nothing is written to out_dir/model_name.h5, and
        nothing is deleted, in that case.

    Notes
    -----
    Checkpoints are found the same way slug finds them when restarting
    a run (see OutputManagerH5::restartSetup()): every
    model_name_chkNNNNN.h5 file, or model_name_chkNNNNN/ directory of
    per-thread/per-rank files, in out_dir, with the file taking
    precedence if both exist for the same NNNNN.

    Every dataset with an unlimited dimension holds one row per output
    record (cluster, galaxy, time, ...), and its rows from every file
    are concatenated, in checkpoint order. Everything else -- top-level
    attributes (trials_completed, restart_uid, rng_state, ...), the
    input deck, fixed datasets such as wavelength grids, and every
    group's and dataset's attributes -- is taken from the final
    checkpoint.

    The consolidated file is written under a temporary name and only
    renamed to model_name.h5 after every row of it has been read back
    and compared against the checkpoints. Note that slug_reader, given
    a model name, prefers checkpoints to model_name.h5 whenever any
    exist, so it reads the consolidated file in preference to the
    checkpoints only once they are removed (cleanup=True), or if it is
    given the path to model_name.h5 directly. Likewise, once the
    checkpoints are removed, a run can no longer be restarted from
    them.
    """
    prefix = (Path(out_dir) if out_dir is not None else Path.cwd()).resolve() / model_name
    checkpoints = _find_checkpoints(prefix)
    if not checkpoints:
        raise FileNotFoundError(
            f"no checkpoints {model_name}_chkNNNNN.h5 or {model_name}_chkNNNNN/ "
            f"found in {prefix.parent}")
    dest = prefix.parent / f"{model_name}.h5"
    if dest.exists():
        raise FileExistsError(f"{dest} already exists")

    files = [f for num in sorted(checkpoints) for f in checkpoints[num]]
    final = checkpoints[max(checkpoints)][0]

    # Check that every file is complete and shares the same structure,
    # recording each extensible dataset's row count per file
    ref_sig, _ = _signature(final)
    rows: list[dict[str, int]] = []
    for path in files:
        with h5py.File(path, "r") as f:
            if "trials_completed" not in f.attrs:
                raise ValueError(
                    f"{path} has no trials_completed attribute, so was never "
                    "closed cleanly (the run may still be in progress, or may "
                    "have crashed); refusing to consolidate")
        sig, file_rows = _signature(path)
        if sig != ref_sig:
            raise ValueError(
                f"{path} does not have the same HDF5 structure as {final}: "
                f"{_describe_mismatch(sig, ref_sig)}")
        rows.append(file_rows)
    extensible = sorted(name for name, entry in ref_sig.items() if entry[0] == "extensible")
    offsets = {name: np.cumsum([0] + [r[name] for r in rows]).tolist() for name in extensible}

    # Write to a temporary file first, so that a failure part way
    # through never leaves a partial model_name.h5 behind; the name
    # deliberately does not match the _chkNNNNN pattern
    tmp = prefix.parent / f".{model_name}.h5.consolidating"
    try:
        with h5py.File(tmp, "w") as dst:
            with h5py.File(final, "r") as src:
                for key, value in src.attrs.items():
                    dst.attrs[key] = value
                for name in sorted(ref_sig):
                    kind = ref_sig[name][0]
                    if kind == "group":
                        group = dst.require_group(name)
                        for key, value in cast(h5py.Group, src[name]).attrs.items():
                            group.attrs[key] = value
                    elif kind == "fixed":
                        src.copy(src[name], dst, name=name)
                    else:
                        tmpl = cast(h5py.Dataset, src[name])
                        tmpl_shape = cast(tuple[int, ...], tmpl.shape)
                        dset = dst.create_dataset(
                            name, shape=(offsets[name][-1], *tmpl_shape[1:]),
                            maxshape=tmpl.maxshape, dtype=tmpl.dtype, chunks=True,
                            compression=tmpl.compression,
                            compression_opts=tmpl.compression_opts,
                            shuffle=tmpl.shuffle)
                        for key, value in tmpl.attrs.items():
                            dset.attrs[key] = value

            for i, path in enumerate(files):
                with h5py.File(path, "r") as src:
                    for name in extensible:
                        src_dset = cast(h5py.Dataset, src[name])
                        dst_dset = cast(h5py.Dataset, dst[name])
                        base = offsets[name][i]
                        for start, stop in _row_blocks(rows[i][name], src_dset):
                            dst_dset[base + start:base + stop] = src_dset[start:stop]

        # Verify: same structure as the final checkpoint, and every
        # row matches the checkpoint it came from
        dst_sig, dst_rows = _signature(str(tmp))
        if dst_sig != ref_sig:
            raise RuntimeError(
                "consolidated file does not have the same HDF5 structure as "
                f"{final}: {_describe_mismatch(dst_sig, ref_sig)}")
        with h5py.File(tmp, "r") as dst:
            for name in extensible:
                if dst_rows[name] != offsets[name][-1]:
                    raise RuntimeError(
                        f"consolidated {name} has {dst_rows[name]} rows, "
                        f"expected {offsets[name][-1]}")
            for i, path in enumerate(files):
                with h5py.File(path, "r") as src:
                    for name in extensible:
                        src_dset = cast(h5py.Dataset, src[name])
                        dst_dset = cast(h5py.Dataset, dst[name])
                        base = offsets[name][i]
                        for start, stop in _row_blocks(rows[i][name], src_dset):
                            if not _same_rows(np.asarray(src_dset[start:stop]),
                                np.asarray(dst_dset[base + start:base + stop])):
                                raise RuntimeError(
                                    f"consolidated {name} does not match {path} "
                                    f"in rows {start}-{stop - 1}")
        # Link rather than rename, since os.link fails instead of
        # overwriting if dest was created while this was running; fall
        # back to a (racy) check-then-rename only on filesystems that
        # do not support hard links
        try:
            os.link(tmp, dest)
        except FileExistsError:
            raise FileExistsError(f"{dest} was created while consolidating") from None
        except OSError:
            if dest.exists():
                raise FileExistsError(f"{dest} was created while consolidating") from None
            tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)

    if cleanup:
        for num in sorted(checkpoints):
            chk_dir = prefix.parent / f"{model_name}_chk{num:05d}"
            for path in checkpoints[num]:
                Path(path).unlink()
            # Only remove the directory if its files were the ones
            # consolidated (a model_name_chkNNNNN.h5 takes precedence
            # over it when both exist; see _checkpoint_files())
            if Path(checkpoints[num][0]).parent == chk_dir:
                try:
                    chk_dir.rmdir()
                except OSError:
                    warnings.warn(
                        f"left {chk_dir} in place, since it still holds files "
                        "other than slug output", stacklevel=2)

    return dest
