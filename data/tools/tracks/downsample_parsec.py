"""
Script to downsample PARSEC evolutionary tracks that are far more
densely time-sampled than slug needs.

PARSEC's adaptive time-stepping produces some tracks -- especially at
the high-mass end of the VMS grid -- with tens of thousands of points,
the vast majority of which change the track's physical state by only
~1e-6 from one point to the next. Tracks3D interpolates in mass,
[Fe/H], and (for the rotating grid) v/vcrit across every point of
every track it loads, so this oversampling directly drives its memory
footprint without buying any real accuracy: slug's own use of these
tracks (stellar evolution during cluster/galaxy simulation) does not
need age resolution anywhere near this fine.

This script thins each track using a simple greedy decimation: walk
forward from the first point, keeping a new point only once it has
diverged from the *last kept point* (not the previous point) by more
than a threshold in any tracked field, then always keep the track's
last point. Thresholds are relative for mass and mdot (both of which
range over orders of magnitude across a track) and absolute for
log_L/log_Teff (already log quantities, so an absolute dex threshold
is the natural choice) and the surface abundance fields (mass
fractions of order unity, so an absolute threshold is directly
meaningful). All of these thresholds have defaults but can be
overridden from the command line.

Run standalone against an existing PARSEC HDF5 track file (VMS or
rotating grid) to downsample it in place:
    python3 downsample_parsec.py /path/to/parsec_vms.h5
or import downsample_parsec() to call it as part of another script.

:copyright: Copyright (c) 2026 Mark Krumholz
"""

import argparse
import os

import h5py
import numpy as np

# Default decimation thresholds
DEFAULT_MASS_TOL = 1.0e-2     # relative, applied to the 'mass' field
DEFAULT_MDOT_TOL = 0.25       # relative, applied to the 'mdot' field; much
                               # looser than the other fields because mdot
                               # is dominated by short-timescale noise (up to
                               # ~1000x at literally the same age, in some
                               # duplicate-age PARSEC substep rows) that
                               # barely registers in mass, log_L, log_Teff,
                               # or the surface abundances -- see the
                               # downsample_parsec commit message for the
                               # trigger-field analysis behind this value
DEFAULT_LOGL_TOL = 1.0e-2     # absolute dex, applied to 'log_L'
DEFAULT_LOGTEFF_TOL = 1.0e-2  # absolute dex, applied to 'log_Teff'
DEFAULT_SURF_TOL = 1.0e-2     # absolute, applied to every '*_surf' field

# Fields compared with a relative threshold; every other tracked field
# (log_L, log_Teff, and any field ending in '_surf', identified below
# by name rather than listed explicitly since PARSEC track files vary
# in which elements they tabulate) is compared with an absolute one.
RELATIVE_FIELDS = ("mass", "mdot")
ABSOLUTE_DEX_FIELDS = ("log_L", "log_Teff")


def _build_specs(
    field_names: list[str],
    mass_tol: float,
    mdot_tol: float,
    logl_tol: float,
    logteff_tol: float,
    surf_tol: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split field_names into (relative_idx, relative_tol, absolute_idx,
    absolute_tol): parallel column-index/threshold arrays, using
    RELATIVE_FIELDS/ABSOLUTE_DEX_FIELDS by name and every
    '_surf'-suffixed field for the surface-abundance group.
    """
    relative_tol = {"mass": mass_tol, "mdot": mdot_tol}
    absolute_tol = {"log_L": logl_tol, "log_Teff": logteff_tol}

    relative_fields = [f for f in RELATIVE_FIELDS if f in field_names]
    absolute_fields = [f for f in ABSOLUTE_DEX_FIELDS if f in field_names]
    surf_fields = [f for f in field_names if f.endswith("_surf")]

    relative_idx = np.array([field_names.index(f) for f in relative_fields], dtype=int)
    relative_tol_arr = np.array([relative_tol[f] for f in relative_fields])
    absolute_idx = np.array(
        [field_names.index(f) for f in absolute_fields + surf_fields], dtype=int)
    absolute_tol_arr = np.array(
        [absolute_tol[f] for f in absolute_fields] + [surf_tol] * len(surf_fields))

    return relative_idx, relative_tol_arr, absolute_idx, absolute_tol_arr


def _exceed_mask(
    rel_data: np.ndarray | None,
    abs_data: np.ndarray | None,
    relative_tol: np.ndarray,
    absolute_tol: np.ndarray,
    anchor: int,
    lo: int,
    hi: int,
) -> np.ndarray:
    """
    Boolean mask over rows [lo, hi) of rel_data/abs_data that diverge
    from row anchor by more than threshold in at least one field.

    Every candidate row's divergence from the anchor is expressed as a
    threshold-normalized difference (so a value >1 means "exceeds"),
    which lets every relative field (each divided by its own tolerance
    in relative_tol) and every absolute field (absolute_tol) be
    compared to its own threshold with one vectorized numpy call each,
    rather than a Python-level loop over fields.

    A relative-field comparison against a last-kept value of exactly
    zero treats any nonzero value as exceeding the threshold (the
    relative change is otherwise undefined), since a field like mdot
    going from exactly zero to nonzero is itself a real change in the
    star's state; dividing by a zero denominator produces +inf (a
    real change, correctly > 1) or nan (0/0, no change, mapped to 0).
    """
    row_max = np.full(hi - lo, -np.inf)

    if rel_data is not None:
        base = rel_data[anchor]
        diff = np.abs(rel_data[lo:hi] - base)
        with np.errstate(divide="ignore", invalid="ignore"):
            normalized = diff / (relative_tol * np.abs(base))
        normalized = np.where(np.isnan(normalized), 0.0, normalized)
        row_max = np.maximum(row_max, np.max(normalized, axis=1))

    if abs_data is not None:
        base = abs_data[anchor]
        normalized = np.abs(abs_data[lo:hi] - base) / absolute_tol
        row_max = np.maximum(row_max, np.max(normalized, axis=1))

    return row_max > 1.0


def downsample_track(
    data: np.ndarray,
    relative_idx: np.ndarray,
    relative_tol: np.ndarray,
    absolute_idx: np.ndarray,
    absolute_tol: np.ndarray,
    window0: int = 64,
) -> np.ndarray:
    """
    Return the row indices of data to keep under the greedy decimation
    algorithm: keep row 0, then repeatedly keep the next row whose
    divergence from the last *kept* row exceeds threshold in any
    tracked field (see _exceed_mask()), then always keep the last row.

    Real PARSEC tracks are extremely non-uniform: a track can spend
    its first few percent of rows changing fast enough that nearly
    every row gets kept, then go essentially flat for the rest of its
    length (kept only every few thousand rows). Searching for the next
    kept row by masking the *entire* remainder of the track from each
    anchor -- despite np.argmax's own vectorized speed -- means that
    for every one of the many closely-spaced anchors in the fast
    region, most of that O(n) mask is wasted work re-scanning the flat
    tail, which empirically costs far more than the Python-level loop
    over fields it might look like the hot path (that loop was
    replaced by _exceed_mask()'s vectorized version with no measurable
    speedup, since it was never the bottleneck). Instead, each anchor
    searches an exponentially growing window (starting at window0
    rows, doubling on each miss) until it finds an exceeding row or
    reaches the end of the track, so a nearby match (the common case
    in the fast-changing region) costs O(1) amortized rather than
    O(n), while a distant match (the flat region) still costs at most
    ~2x the distance actually skipped.
    """
    n = data.shape[0]
    if n <= 2:
        return np.arange(n)

    rel_data = data[:, relative_idx] if relative_idx.size else None
    abs_data = data[:, absolute_idx] if absolute_idx.size else None

    keep = [0]
    anchor = 0
    while anchor < n - 1:
        window = window0
        lo = anchor + 1
        found = None
        while True:
            hi = min(anchor + 1 + window, n)
            mask = _exceed_mask(rel_data, abs_data, relative_tol, absolute_tol, anchor, lo, hi)
            if mask.any():
                found = lo + int(np.argmax(mask))
                break
            if hi >= n:
                break
            lo = hi
            window *= 2

        if found is None:
            break
        anchor = found
        keep.append(anchor)

    if keep[-1] != n - 1:
        keep.append(n - 1)

    return np.array(keep)


def downsample_parsec(
    h5_filename: str,
    mass_tol: float = DEFAULT_MASS_TOL,
    mdot_tol: float = DEFAULT_MDOT_TOL,
    logl_tol: float = DEFAULT_LOGL_TOL,
    logteff_tol: float = DEFAULT_LOGTEFF_TOL,
    surf_tol: float = DEFAULT_SURF_TOL,
    verbose: bool = False,
) -> dict:
    """
    Downsample every track_m* dataset in every group of h5_filename,
    using the greedy decimation algorithm in downsample_track(), and
    replace h5_filename with the result.

    mass_tol/mdot_tol are relative thresholds (0.01 = 1%); logl_tol
    and logteff_tol are absolute thresholds in dex; surf_tol is an
    absolute threshold applied to every field whose name ends in
    '_surf'. A group's 'ntime' attribute (informational only; nothing
    in slug reads it back) is updated to match the downsampled length
    of its first (by dataset name) mass's track, mirroring the
    convention used when these files are first fetched/combined.

    The result is built up in a fresh sibling file rather than
    deleting and recreating each track_m* dataset in place: HDF5 does
    not reclaim a deleted dataset's space within an existing file (it
    becomes internal free space, reused only by later writes that fit
    it), so modifying every dataset in the file in place leaves it
    close to its original size regardless of how much smaller the
    data actually got. Writing a new file and then replacing the
    original with it (os.replace, atomic on the same filesystem)
    avoids that fragmentation entirely and leaves no partially-written
    file if interrupted.

    Returns a dict of aggregate statistics: {'n_tracks', 'n_points_before',
    'n_points_after'}.
    """
    n_tracks = 0
    n_points_before = 0
    n_points_after = 0

    tmp_filename = h5_filename + ".downsample_tmp"
    with h5py.File(h5_filename, "r") as h5in, h5py.File(tmp_filename, "w") as h5out:
        for attr_name, attr_val in h5in.attrs.items():
            h5out.attrs[attr_name] = attr_val

        for grp_name in h5in:
            grp_in = h5in[grp_name]
            if "field_names" not in grp_in.attrs:
                h5in.copy(grp_in, h5out, name=grp_name)
                continue
            field_names = list(grp_in.attrs["field_names"])
            relative_idx, relative_tol, absolute_idx, absolute_tol = _build_specs(
                field_names, mass_tol, mdot_tol, logl_tol, logteff_tol, surf_tol)

            track_keys = sorted(k for k in grp_in.keys() if k.startswith("track_m"))
            if not track_keys:
                h5in.copy(grp_in, h5out, name=grp_name)
                continue

            grp_out = h5out.create_group(grp_name)
            for attr_name, attr_val in grp_in.attrs.items():
                grp_out.attrs[attr_name] = attr_val
            if "masses" in grp_in:
                grp_out.create_dataset("masses", data=grp_in["masses"][:])

            first_track_len = None
            for track_key in track_keys:
                data = grp_in[track_key][:]
                n_before = data.shape[0]

                keep = downsample_track(
                    data, relative_idx, relative_tol, absolute_idx, absolute_tol)
                new_data = data[keep]

                grp_out.create_dataset(track_key, data=new_data, compression="gzip")

                n_tracks += 1
                n_points_before += n_before
                n_points_after += new_data.shape[0]
                if first_track_len is None:
                    first_track_len = new_data.shape[0]

                if verbose:
                    print(f"{grp_name}/{track_key}: {n_before} -> {new_data.shape[0]} points")

            grp_out.attrs["ntime"] = first_track_len

    os.replace(tmp_filename, h5_filename)

    return {
        "n_tracks": n_tracks,
        "n_points_before": n_points_before,
        "n_points_after": n_points_after,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Downsample over-sampled PARSEC tracks in an HDF5 track "
                    "file, in place")
    parser.add_argument("h5file", help="PARSEC HDF5 track file to downsample in place")
    parser.add_argument("--mass_tol", type=float, default=DEFAULT_MASS_TOL,
                        help=f"relative threshold for the 'mass' field "
                             f"(default {DEFAULT_MASS_TOL:g})")
    parser.add_argument("--mdot_tol", type=float, default=DEFAULT_MDOT_TOL,
                        help=f"relative threshold for the 'mdot' field "
                             f"(default {DEFAULT_MDOT_TOL:g})")
    parser.add_argument("--logl_tol", type=float, default=DEFAULT_LOGL_TOL,
                        help=f"absolute dex threshold for 'log_L' "
                             f"(default {DEFAULT_LOGL_TOL:g})")
    parser.add_argument("--logteff_tol", type=float, default=DEFAULT_LOGTEFF_TOL,
                        help=f"absolute dex threshold for 'log_Teff' "
                             f"(default {DEFAULT_LOGTEFF_TOL:g})")
    parser.add_argument("--surf_tol", type=float, default=DEFAULT_SURF_TOL,
                        help=f"absolute threshold for every '*_surf' field "
                             f"(default {DEFAULT_SURF_TOL:g})")
    parser.add_argument("--verbose", action="store_true",
                        help="print the point-count reduction for every track")
    args = parser.parse_args()

    stats = downsample_parsec(
        args.h5file,
        mass_tol=args.mass_tol,
        mdot_tol=args.mdot_tol,
        logl_tol=args.logl_tol,
        logteff_tol=args.logteff_tol,
        surf_tol=args.surf_tol,
        verbose=args.verbose,
    )
    before, after = stats["n_points_before"], stats["n_points_after"]
    print(f"Downsampled {stats['n_tracks']} tracks in {args.h5file}: "
          f"{before} -> {after} points "
          f"({100.0 * (1.0 - after / before):.1f}% reduction)")
