"""
cloudy_output.py

Implements the machinery to write cloudy run results (the physical
conditions used, and -- where available -- the emergent continuum and
emission lines) into a slug HDF5 output file's own cluster_cloudy/
galaxy_cloudy group.

:copyright: Copyright (c) 2026 Mark Krumholz
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from astropy import units as u

# The four continuum fields, in the order cloudy's own "save last
# continuum" output lists them
_CONTINUUM_FIELDS = ("spec_inc", "spec_trans", "spec_emit", "spec_trans_emit")

_FLUX_UNIT = u.erg / u.s / u.AA


@dataclass
class CloudyRunResult:
    """
    The result of one successfully-completed cloudy run, ready to be
    written into a slug HDF5 output file's cluster_cloudy/galaxy_cloudy
    group.

    Attributes
    ----------
    id_val : int
        uid (cluster mode) or trial (galaxy mode) this run belongs to.
    time : float
        Output time, in yr.
    nII, r0, r1, U, U0, Omega : astropy.units.Quantity
        The physical conditions of the HII region used for this run
        (see hiiregparam).
    continuum : tuple of astropy.units.Quantity, optional
        (wl, inc, trans, emit, trans_emit), as returned by
        read_cloudy_continuum, or None if this run's deck had no
        "save last continuum" output.
    lines : tuple, optional
        (line_wl, line_label, line_lum), as returned by
        read_cloudy_linearr, or None if this run's deck had no "save
        last line array" output.
    """

    id_val: int
    time: float
    nII: u.Quantity
    r0: u.Quantity
    r1: u.Quantity
    U: u.Quantity
    U0: u.Quantity
    Omega: u.Quantity
    continuum: tuple[u.Quantity, u.Quantity, u.Quantity, u.Quantity, u.Quantity] | None
    lines: tuple[u.Quantity, list[str], u.Quantity] | None


def write_cloudy_h5_results(h5_path: str | Path, group_name: str, id_key: str,
    results: Sequence[CloudyRunResult]) -> None:
    """
    Append cloudy run results to an HDF5 file's own cluster_cloudy or
    galaxy_cloudy group, creating the group (and its datasets) if this
    is the first time it's been written to.

    Parameters
    ----------
    h5_path : str or pathlib.Path
        Path to the slug HDF5 output file to write into. Opened and
        closed by this function -- the caller is responsible for not
        holding its own handle to the same file open at the same time.
    group_name : str
        Name of the group to write into: "cluster_cloudy" or
        "galaxy_cloudy".
    id_key : str
        Name of the id dataset: "uid" or "trial".
    results : sequence of CloudyRunResult
        The results to append, one row per entry. A no-op if empty.

    Details
    -------
    Each result becomes one row in id_key/time/nII/r0/r1/U/U0/Omega,
    all extensible 1-D datasets. If any result (in this call, or from
    an earlier one) has continuum data, the group also gets wl (the
    shared wavelength grid) and spec_inc/spec_trans/spec_emit/
    spec_trans_emit (one row per result, aligned with the datasets
    above -- results with no continuum of their own get an all-zero
    row). Cloudy's own continuum grid always shares the same maximum
    wavelength and grid spacing across runs, but the minimum
    wavelength can vary; if a result's own grid is shorter than what's
    already stored, its row is zero-padded at the low-wavelength end
    to align with the stored (longer) grid. If a result's grid is
    instead longer than what's already stored, the wavelength grid and
    every already-stored row are rewritten to the new, longer grid
    (zero-padded the same way) before appending.

    If any result (in this call, or an earlier one) has line data, the
    group also gets line_start and line_count -- one entry per result,
    aligned with the datasets above -- plus flat, append-only
    line_wl/line_label/line_lum arrays holding every kept line from
    every result with line data, in order; a given result's own lines
    are the slice line_wl[line_start:line_start + line_count] (etc.).
    This is a ragged (variable count per result) format, unlike
    continuum's fixed-width one, so it needs no padding or rewriting:
    a result with no line data of its own simply gets line_count = 0.
    """
    if not results:
        return

    with h5py.File(h5_path, "a") as f:
        group = f.require_group(group_name)
        n_old_rows = group[id_key].shape[0] if id_key in group else 0

        _append_scalar(group, id_key, np.array([r.id_val for r in results], dtype=np.uint64), "")
        _append_scalar(group, "time", np.array([r.time for r in results], dtype=np.float64), str(u.yr))
        _append_scalar(group, "nII",
            np.array([r.nII.to_value(u.cm ** -3) for r in results]), str(u.cm ** -3))
        _append_scalar(group, "r0", np.array([r.r0.to_value(u.cm) for r in results]), str(u.cm))
        _append_scalar(group, "r1", np.array([r.r1.to_value(u.cm) for r in results]), str(u.cm))
        _append_scalar(group, "U",
            np.array([r.U.to_value(u.dimensionless_unscaled) for r in results]), "")
        _append_scalar(group, "U0",
            np.array([r.U0.to_value(u.dimensionless_unscaled) for r in results]), "")
        _append_scalar(group, "Omega",
            np.array([r.Omega.to_value(u.dimensionless_unscaled) for r in results]), "")

        has_continuum_group = "wl" in group
        any_new_continuum = any(r.continuum is not None for r in results)
        if has_continuum_group or any_new_continuum:
            _append_continuum(group, results, n_old_rows)

        has_lines_group = "line_start" in group
        any_new_lines = any(r.lines is not None for r in results)
        if has_lines_group or any_new_lines:
            _append_lines(group, results, n_old_rows)


def _append_scalar(group: h5py.Group, name: str, values: np.ndarray, units: str) -> None:
    """Append values as new rows of group's own extensible 1-D dataset name, creating it first if needed."""
    if name not in group:
        dset = group.create_dataset(name, shape=(0,), maxshape=(None,), chunks=True, dtype=values.dtype)
        dset.attrs["units"] = units
    dset = group[name]
    old_len = dset.shape[0]
    new_len = old_len + len(values)
    dset.resize((new_len,))
    dset[old_len:new_len] = values


def _append_continuum(group: h5py.Group, results: Sequence[CloudyRunResult], n_old_rows: int) -> None:
    """Append (or backfill with zero) one continuum row per result, growing and rewriting the shared wavelength grid first if needed."""
    old_wl_len = group["wl"].shape[0] if "wl" in group else 0

    new_ref: u.Quantity | None = None
    for r in results:
        if r.continuum is not None and (new_ref is None or len(r.continuum[0]) > len(new_ref)):
            new_ref = r.continuum[0]
    new_max_len = len(new_ref) if new_ref is not None else 0

    final_len = max(old_wl_len, new_max_len)
    if new_max_len > old_wl_len:
        assert new_ref is not None
        _grow_wavelength_grid(group, new_ref, old_wl_len)

    n_new = len(results)
    for name in _CONTINUUM_FIELDS:
        if name not in group:
            dset = group.create_dataset(name, shape=(n_old_rows, final_len),
                maxshape=(None, None), chunks=True, dtype="f8")
            dset.attrs["units"] = str(_FLUX_UNIT)
        else:
            dset = group[name]
        dset.resize((n_old_rows + n_new, final_len))

    for i, r in enumerate(results):
        row = n_old_rows + i
        if r.continuum is None:
            for name in _CONTINUUM_FIELDS:
                group[name][row, :] = 0.0
            continue
        wl_r, inc, trans, emit, trans_emit = r.continuum
        offset = final_len - len(wl_r)
        for name, arr in zip(_CONTINUUM_FIELDS, (inc, trans, emit, trans_emit), strict=True):
            dset = group[name]
            if offset > 0:
                dset[row, :offset] = 0.0
            dset[row, offset:] = arr.to_value(_FLUX_UNIT)


def _grow_wavelength_grid(group: h5py.Group, new_wl: u.Quantity, old_len: int) -> None:
    """Replace group's own wl dataset with the longer new_wl grid, shifting every already-stored continuum row right to stay aligned to the shared maximum wavelength."""
    new_len = len(new_wl)
    offset = new_len - old_len

    if "wl" not in group:
        dset = group.create_dataset("wl", shape=(new_len,), maxshape=(None,), chunks=True, dtype="f8")
        dset.attrs["units"] = str(u.AA)
    else:
        dset = group["wl"]
        dset.resize((new_len,))
    dset[:] = new_wl.to_value(u.AA)

    if offset == 0:
        return
    for name in _CONTINUUM_FIELDS:
        if name not in group:
            continue
        dset = group[name]
        n_rows = dset.shape[0]
        old_data = dset[()]
        dset.resize((n_rows, new_len))
        if n_rows > 0:
            padded = np.zeros((n_rows, new_len))
            padded[:, offset:] = old_data
            dset[:, :] = padded


def _append_lines(group: h5py.Group, results: Sequence[CloudyRunResult], n_old_rows: int) -> None:
    """Append one (line_start, line_count) index entry per result, plus that result's own kept lines onto the group's flat, append-only line_wl/line_label/line_lum arrays."""
    if "line_wl" not in group:
        wl_dset = group.create_dataset("line_wl", shape=(0,), maxshape=(None,), chunks=True, dtype="f8")
        wl_dset.attrs["units"] = str(u.AA)
        group.create_dataset("line_label", shape=(0,), maxshape=(None,),
            chunks=True, dtype=h5py.string_dtype(encoding="ascii", length=4))
        lum_dset = group.create_dataset("line_lum", shape=(0,), maxshape=(None,), chunks=True, dtype="f8")
        lum_dset.attrs["units"] = str(u.erg / u.s)
    if "line_start" not in group:
        # Backfill a zero-count index entry for any rows already
        # written before line data was ever introduced, so line_start/
        # line_count stay aligned with the scalar datasets
        start_dset = group.create_dataset(
            "line_start", shape=(n_old_rows,), maxshape=(None,), chunks=True, dtype=np.uint64)
        count_dset = group.create_dataset(
            "line_count", shape=(n_old_rows,), maxshape=(None,), chunks=True, dtype=np.uint64)
        start_dset.attrs["units"] = ""
        count_dset.attrs["units"] = ""

    wl_dset = group["line_wl"]
    label_dset = group["line_label"]
    lum_dset = group["line_lum"]
    flat_old_len = wl_dset.shape[0]

    starts = np.empty(len(results), dtype=np.uint64)
    counts = np.empty(len(results), dtype=np.uint64)
    all_wl: list[float] = []
    all_label: list[str] = []
    all_lum: list[float] = []
    running = flat_old_len
    for i, r in enumerate(results):
        starts[i] = running
        if r.lines is None:
            counts[i] = 0
            continue
        wl_r, label_r, lum_r = r.lines
        counts[i] = len(label_r)
        running += len(label_r)
        all_wl.extend(wl_r.to_value(u.AA))
        all_label.extend(label_r)
        all_lum.extend(lum_r.to_value(u.erg / u.s))

    flat_new_len = flat_old_len + len(all_wl)
    wl_dset.resize((flat_new_len,))
    label_dset.resize((flat_new_len,))
    lum_dset.resize((flat_new_len,))
    if all_wl:
        wl_dset[flat_old_len:flat_new_len] = all_wl
        label_dset[flat_old_len:flat_new_len] = all_label
        lum_dset[flat_old_len:flat_new_len] = all_lum

    _append_scalar(group, "line_start", starts, "")
    _append_scalar(group, "line_count", counts, "")
