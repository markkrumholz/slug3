"""
Script to prune PARSEC evolutionary tracks of any row whose age does
not strictly advance relative to the last kept row.

PARSEC's own raw output occasionally backtracks in age -- not just the
duplicate-age substep rows interpolate_parsec_rot.py's dedup_by_age()
already collapses when synthesizing a donor track for a missing mass,
but a genuine several-percent decrease in the middle of a real,
natively-fetched track (e.g. parsec_rot.h5's feh_-2.18298_vvcrit_0.00,
m=0.96 Msun: age drops from 6.372e9 to 6.364e9 at row 245, then only
partially recovers to 6.366e9 by row 247 -- still below the row-245
peak). dedup_by_age() would not fully fix even this specific case if
it were applied here (it only compares each row to its immediate
predecessor in the raw array, not to the last row that actually
survived the filter, so the partial-recovery row at 247 would still
pass), and in any case it never runs on this track at all: it is only
ever invoked on a *synthesized* donor track for a mass that is missing
from a group, and this mass is not missing anywhere -- it is native,
already-fetched data.

Tracks3D's exact-[Fe/H]-match fast path (used whenever a query's
[Fe/H] lands precisely on a grid point) hands a track's raw age column
straight to Mesh2DGrid's strict monotonicity check with no
interpolation to smooth it over, so this kind of backtrack crashes
slug outright the moment a simulation requests that exact [Fe/H] (see
cloudy_grid_genuine_slug_failures_report.txt's "FAILURE CLASS A" for
the reproduction and root-cause trace: PARSEC_comp [Fe/H]=-2.182985,
v/vcrit=0.0, m=0.96 Msun -- inherited from parsec_rot.h5 via
combine_parsec.py, since masses below its mass_break are taken
verbatim from the rotating grid's own v/vcrit=0 subset).

This script walks each track forward from its first point, keeping a
row only once its age exceeds the age of the last *kept* row. This
both removes exact duplicate/plateaued ages and any genuine backward
jump (fully, unlike a naive adjacent-row comparison -- see above), at
the cost of discarding the (typically few) rows in between --
effectively cutting out the offending segment rather than trying to
repair it.

Run standalone against an existing PARSEC HDF5 track file (VMS or
rotating grid):
    python3 prune_parsec_age_regressions.py /path/to/parsec_vms.h5
Run it against both parsec_vms.h5 and parsec_rot.h5, then re-run
combine_parsec.py (with --overwrite) to regenerate parsec_composite.h5
from the pruned inputs. Or import prune_parsec_age_regressions() to
call it as part of another script.

:copyright: Copyright (c) 2026 Mark Krumholz
"""

import argparse
import os

import h5py
import numpy as np


def prune_age_regressions(data: np.ndarray) -> np.ndarray:
    """
    Return the rows of data (age in column 0) whose age strictly
    increases relative to the age of the last returned row, always
    keeping the first row.

    Vectorized via the identity that the running maximum of every raw
    age value up to row i-1 equals the age of the last row that would
    be kept up to that point: a row that fails to exceed the running
    maximum is dropped, and dropping it cannot itself change the
    running maximum, so the two are self-consistent.
    """
    if data.shape[0] == 0:
        return data
    age = data[:, 0]
    running_max_before = np.concatenate(([-np.inf], np.maximum.accumulate(age[:-1])))
    keep = age > running_max_before
    keep[0] = True
    return data[keep]


def prune_parsec_age_regressions(h5_filename: str, verbose: bool = False) -> dict:
    """
    Prune every track_m* dataset in every group of h5_filename of any
    row that does not strictly advance in age relative to the last
    kept row, and replace h5_filename with the result.

    Builds the result in a fresh sibling file rather than deleting and
    recreating datasets in place, then replaces the original with it
    via os.replace (atomic on the same filesystem) -- see
    downsample_parsec.py's own version of this same pattern for why:
    HDF5 does not reclaim a deleted dataset's space within an existing
    file, so modifying datasets in place leaves the file close to its
    original size regardless of how much smaller the data actually
    got.

    Returns a dict of aggregate statistics: {'n_tracks', 'n_points_before',
    'n_points_after', 'n_tracks_changed'}.
    """
    n_tracks = 0
    n_tracks_changed = 0
    n_points_before = 0
    n_points_after = 0

    tmp_filename = h5_filename + ".prune_age_tmp"
    with h5py.File(h5_filename, "r") as h5in, h5py.File(tmp_filename, "w") as h5out:
        for attr_name, attr_val in h5in.attrs.items():
            h5out.attrs[attr_name] = attr_val

        for grp_name in h5in:
            grp_in = h5in[grp_name]
            track_keys = sorted(k for k in grp_in.keys() if k.startswith("track_m"))
            if "field_names" not in grp_in.attrs or not track_keys:
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

                new_data = prune_age_regressions(data)

                grp_out.create_dataset(track_key, data=new_data, compression="gzip")

                n_tracks += 1
                n_points_before += n_before
                n_points_after += new_data.shape[0]
                if new_data.shape[0] != n_before:
                    n_tracks_changed += 1
                if first_track_len is None:
                    first_track_len = new_data.shape[0]

                if verbose and new_data.shape[0] != n_before:
                    print(f"{grp_name}/{track_key}: {n_before} -> "
                          f"{new_data.shape[0]} points "
                          f"({n_before - new_data.shape[0]} pruned)")

            grp_out.attrs["ntime"] = first_track_len

    os.replace(tmp_filename, h5_filename)

    return {
        "n_tracks": n_tracks,
        "n_tracks_changed": n_tracks_changed,
        "n_points_before": n_points_before,
        "n_points_after": n_points_after,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prune non-advancing/regressing age rows from a PARSEC "
                    "HDF5 track file, in place")
    parser.add_argument("h5file", help="PARSEC HDF5 track file to prune in place")
    parser.add_argument("--verbose", action="store_true",
                        help="print every track whose point count changed")
    args = parser.parse_args()

    stats = prune_parsec_age_regressions(args.h5file, verbose=args.verbose)
    print(f"Pruned {args.h5file}: {stats['n_tracks_changed']} of "
          f"{stats['n_tracks']} tracks had rows removed "
          f"({stats['n_points_before']} -> {stats['n_points_after']} points total)")
