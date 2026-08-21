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

# Every dataset in a cluster_cloudy/galaxy_cloudy group that's aligned
# one-entry-per-row with id_key/time (as opposed to a shared column
# header like wl or line_wl/line_label), split by dimensionality --
# id_key itself is handled separately since its name ("uid" or
# "trial") varies by group
_ROW_ALIGNED_1D = ("time", "nII", "r0", "r1", "U", "U0", "Omega")
_ROW_ALIGNED_2D = (*_CONTINUUM_FIELDS, "line_lum")


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
    group also gets line_wl and line_label (the shared, ascending-
    sorted list of every distinct line wavelength seen from any result
    with line data, one label per wavelength) and line_lum (one row
    per result, aligned with the datasets above, giving that result's
    own luminosity at every wavelength in line_wl -- zero at any
    wavelength that result's own run didn't report, including an
    all-zero row for a result with no line data of its own). Different
    cloudy runs need not report the same set of lines, so -- exactly
    like the wavelength grid in _grow_wavelength_grid -- if a result's
    own lines include a wavelength not already in line_wl, line_wl (and
    every already-stored line_lum row) is grown and remapped onto the
    new, merged wavelength list before appending.
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

        has_lines_group = "line_wl" in group
        any_new_lines = any(r.lines is not None for r in results)
        if has_lines_group or any_new_lines:
            _append_lines(group, results, n_old_rows)


def delete_cloudy_h5_rows(h5_path: str | Path, group_name: str, id_key: str,
    row_indices: Sequence[int]) -> None:
    """
    Permanently remove one or more rows (by absolute row index) from
    an HDF5 file's own cluster_cloudy or galaxy_cloudy group.

    Parameters
    ----------
    h5_path : str or pathlib.Path
        Path to the slug HDF5 output file to modify. Opened and closed
        by this function -- the caller is responsible for not holding
        its own handle to the same file open at the same time.
    group_name : str
        Name of the group to modify: "cluster_cloudy" or
        "galaxy_cloudy".
    id_key : str
        Name of the id dataset: "uid" or "trial".
    row_indices : sequence of int
        Absolute row indices to remove. A no-op if empty.

    Details
    -------
    Every dataset aligned one-entry-per-row with id_key/time --
    id_key/time/nII/r0/r1/U/U0/Omega and, where present,
    spec_inc/spec_trans/spec_emit/spec_trans_emit/line_lum -- is
    shrunk to drop exactly those rows, shifting every later row down
    to close the gap. Shared column-header datasets (wl, line_wl,
    line_label) are left untouched: removing a row never removes a
    wavelength or line other rows still reference.
    """
    if not row_indices:
        return

    with h5py.File(h5_path, "a") as f:
        group = f[group_name]
        n_rows = group[id_key].shape[0]
        keep = np.ones(n_rows, dtype=bool)
        keep[list(row_indices)] = False

        for name in (id_key, *_ROW_ALIGNED_1D):
            if name not in group:
                continue
            dset = group[name]
            new_data = dset[()][keep]
            dset.resize((len(new_data),))
            dset[:] = new_data

        for name in _ROW_ALIGNED_2D:
            if name not in group:
                continue
            dset = group[name]
            new_data = dset[()][keep, :]
            dset.resize(new_data.shape)
            dset[:] = new_data


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
    """Append (or backfill with zero) one line-luminosity row per result, growing and remapping the shared, sorted line_wl/line_label list first if needed."""
    # This call's own combined line table: the union of every new
    # result's own (wavelength, label) pairs, sorted by wavelength,
    # plus a matching per-result luminosity matrix (one row per
    # result, zero in any column that result's own run didn't report)
    new_lines: dict[float, str] = {}
    for r in results:
        if r.lines is None:
            continue
        wl_r, label_r, _ = r.lines
        for wl_val, label_val in zip(wl_r.to_value(u.AA), label_r, strict=True):
            new_lines.setdefault(wl_val, label_val)
    new_wl = np.array(sorted(new_lines), dtype=np.float64)
    new_label = [new_lines[w] for w in new_wl]
    new_col = {w: i for i, w in enumerate(new_wl)}

    n_new = len(results)
    new_lum = np.zeros((n_new, len(new_wl)), dtype=np.float64)
    for i, r in enumerate(results):
        if r.lines is None:
            continue
        wl_r, _, lum_r = r.lines
        for wl_val, lum_val in zip(wl_r.to_value(u.AA), lum_r.to_value(u.erg / u.s), strict=True):
            new_lum[i, new_col[wl_val]] = lum_val

    if "line_wl" not in group:
        wl_dset = group.create_dataset("line_wl", shape=(len(new_wl),), maxshape=(None,), chunks=True, dtype="f8")
        wl_dset.attrs["units"] = str(u.AA)
        wl_dset[:] = new_wl

        label_dset = group.create_dataset("line_label", shape=(len(new_label),), maxshape=(None,),
            chunks=True, dtype=h5py.string_dtype(encoding="ascii", length=4))
        label_dset[:] = new_label

        lum_dset = group.create_dataset("line_lum", shape=(n_old_rows + n_new, len(new_wl)),
            maxshape=(None, None), chunks=True, dtype="f8")
        lum_dset.attrs["units"] = str(u.erg / u.s)
        if n_old_rows > 0:
            # Backfill an all-zero row for any result written before
            # line data was ever introduced, so line_lum stays aligned
            # with the scalar datasets
            lum_dset[:n_old_rows, :] = 0.0
        lum_dset[n_old_rows:, :] = new_lum
        return

    # Line data already exists: merge the stored wavelength/label list
    # with this call's own, remapping both the already-stored
    # luminosity rows and this call's new rows onto the merged list
    old_wl_dset = group["line_wl"]
    old_label_dset = group["line_label"]
    old_lum_dset = group["line_lum"]

    old_wl = old_wl_dset[()]
    old_label = [lbl.decode() if isinstance(lbl, bytes) else lbl for lbl in old_label_dset[()]]
    old_lum = old_lum_dset[()]

    merged: dict[float, str] = dict(zip(old_wl, old_label, strict=True))
    for w, lbl in zip(new_wl, new_label, strict=True):
        merged.setdefault(w, lbl)
    merged_wl = np.array(sorted(merged), dtype=np.float64)
    merged_label = [merged[w] for w in merged_wl]
    merged_col = {w: i for i, w in enumerate(merged_wl)}

    old_wl_dset.resize((len(merged_wl),))
    old_wl_dset[:] = merged_wl
    old_label_dset.resize((len(merged_wl),))
    old_label_dset[:] = merged_label

    old_lum_padded = np.zeros((old_lum.shape[0], len(merged_wl)), dtype=np.float64)
    for j, w in enumerate(old_wl):
        old_lum_padded[:, merged_col[w]] = old_lum[:, j]
    new_lum_padded = np.zeros((n_new, len(merged_wl)), dtype=np.float64)
    for j, w in enumerate(new_wl):
        new_lum_padded[:, merged_col[w]] = new_lum[:, j]

    old_lum_dset.resize((old_lum.shape[0] + n_new, len(merged_wl)))
    old_lum_dset[:old_lum.shape[0], :] = old_lum_padded
    old_lum_dset[old_lum.shape[0]:, :] = new_lum_padded
