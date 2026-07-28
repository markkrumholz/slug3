"""
Script to identify and fill gaps in the MIST stellar track library.

The MIST tracks are supposed to contain the same set of initial masses for
every combination of [Fe/H], [A/Fe], and v/vCrit, but a small number of
models did not converge and are absent from the file. This script locates
those gaps by comparing each group's mass list against the canonical
(maximum-count) mass grid, then fills them by linear interpolation in mass
between the two nearest available tracks.

Gaps fall into two cases:
  Case 1  – Both neighboring tracks have the same number of time steps.
             The missing track is produced by direct linear interpolation
             in mass at each time step.
  Case 2  – The neighboring tracks differ in length because they do not
             cover the same set of evolutionary phases.  Only the phases
             present in both neighbors are used; within each common phase
             the track with more time steps is truncated to the shorter
             count.  The resulting tracks are then interpolated as in
             Case 1.  This handles both minor differences (one neighbor
             has an extra leading/trailing phase) and severe ones (at very
             low metallicity the low-mass neighbor may only have pre-MS and
             MS phases).  In the latter case the interpolated track will be
             missing post-MS evolution, but this is a limitation of the
             available MIST data, not of the interpolation scheme.

Interpolated tracks are written back into the same HDF5 file with the same
dataset layout as the original tracks, plus a boolean attribute
'interpolated = True' to distinguish them.
"""

import argparse
import shutil
from collections import defaultdict

import h5py
import numpy as np

MIST_FILE = shutil.os.path.join("..", "tracks", "mist.h5")
MASS_TOL = 1e-6


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_phases(track_data: np.ndarray) -> frozenset[int]:
    """Return the set of unique integer phase values in *track_data*."""
    return frozenset(int(round(p)) for p in track_data[:, 10])


def find_track_key(group: h5py.Group, mass: float) -> str | None:
    """Return the dataset key for the track at *mass* in *group*, or None."""
    for k in group.keys():
        if k.startswith("track_m"):
            try:
                if abs(float(k[7:]) - mass) < MASS_TOL:
                    return k
            except ValueError:
                pass
    return None


def track_dataset_name(mass: float) -> str:
    return f"track_m{mass:.3f}"


def interpolate_gap(
    lo_data: np.ndarray,
    hi_data: np.ndarray,
    lo_mass: float,
    hi_mass: float,
    target_mass: float,
) -> tuple[str, np.ndarray | None]:
    """
    Compute an interpolated track at *target_mass* from neighbors at
    *lo_mass* and *hi_mass*.

    Returns (case, data) where *case* is 'case1', 'case2', or 'no_common'.
    *data* is the interpolated 2-D array for case1/case2, None for no_common.
    """
    alpha = (target_mass - lo_mass) / (hi_mass - lo_mass)

    if len(lo_data) == len(hi_data):
        return "case1", (1.0 - alpha) * lo_data + alpha * hi_data

    # Align phase by phase: keep only phases present in both tracks, and
    # within each common phase take min(lo_count, hi_count) rows.
    lo_phases = get_phases(lo_data)
    hi_phases = get_phases(hi_data)
    common = sorted(lo_phases & hi_phases)
    lo_parts: list[np.ndarray] = []
    hi_parts: list[np.ndarray] = []
    for ph in common:
        ph_mask_lo = np.round(lo_data[:, 10]).astype(int) == ph
        ph_mask_hi = np.round(hi_data[:, 10]).astype(int) == ph
        lo_rows = lo_data[ph_mask_lo]
        hi_rows = hi_data[ph_mask_hi]
        n = min(len(lo_rows), len(hi_rows))
        lo_parts.append(lo_rows[:n])
        hi_parts.append(hi_rows[:n])

    lo_aligned = np.concatenate(lo_parts)
    hi_aligned = np.concatenate(hi_parts)

    return "case2", (1.0 - alpha) * lo_aligned + alpha * hi_aligned


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Identify and fill missing MIST mass tracks by interpolation")
parser.add_argument("--input", default=MIST_FILE,
                    help="Path to the MIST HDF5 track file (default: %(default)s)")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Read phase: collect group masses without modifying the file.
# ---------------------------------------------------------------------------

with h5py.File(args.input, "r") as h5:
    group_nmass: dict[str, int] = {
        g: int(h5[g].attrs["nmass"]) for g in h5.keys()
    }
    group_masses: dict[str, np.ndarray] = {
        g: h5[g]["masses"][:] for g in h5.keys()
    }

n_max = max(group_nmass.values())
ref_group = next(g for g, n in group_nmass.items() if n == n_max)
ref_masses = group_masses[ref_group]

# Build per-group work lists: (target_mass, lo_mass, hi_mass)
group_work: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
for grp, masses in group_masses.items():
    if len(masses) == n_max:
        continue
    missing = [m for m in ref_masses if not np.any(np.abs(masses - m) < MASS_TOL)]
    for m in missing:
        below = [pm for pm in masses if pm < m - MASS_TOL]
        above = [pm for pm in masses if pm > m + MASS_TOL]
        if below and above:
            group_work[grp].append((float(m), float(below[-1]), float(above[0])))

if not group_work:
    print(f"All {len(group_nmass)} groups already have the full set of "
          f"{n_max} mass tracks. Nothing to do.")
    raise SystemExit(0)

total_missing = sum(len(v) for v in group_work.values())
print(f"Found {len(group_work)} group(s) with {total_missing} missing tracks; "
      f"beginning interpolation.\n")

# ---------------------------------------------------------------------------
# Interpolation and write phase.
# ---------------------------------------------------------------------------

n_interpolated = 0
n_skipped_exists = 0

with h5py.File(args.input, "a") as h5:
    for grp in sorted(group_work):
        new_masses: list[float] = []
        for target_mass, lo_mass, hi_mass in group_work[grp]:
            lo_key = find_track_key(h5[grp], lo_mass)
            hi_key = find_track_key(h5[grp], hi_mass)
            if lo_key is None or hi_key is None:
                print(f"  WARNING: neighbor track missing for "
                      f"{target_mass:.4g} M_sun in {grp}; skipping")
                continue

            ds_name = track_dataset_name(target_mass)
            if ds_name in h5[grp]:
                n_skipped_exists += 1
                continue

            lo_data = h5[grp][lo_key][:]
            hi_data = h5[grp][hi_key][:]
            case, new_data = interpolate_gap(
                lo_data, hi_data, lo_mass, hi_mass, target_mass)

            ds = h5[grp].create_dataset(ds_name, data=new_data)
            ds.attrs["interpolated"] = True
            new_masses.append(target_mass)
            n_interpolated += 1

        # Update the group's masses array and nmass attribute.
        if new_masses:
            old_masses = h5[grp]["masses"][:]
            updated = np.sort(np.append(old_masses, new_masses))
            del h5[grp]["masses"]
            h5[grp].create_dataset("masses", data=updated)
            h5[grp].attrs["nmass"] = len(updated)

print(f"Interpolated : {n_interpolated} tracks")
if n_skipped_exists:
    print(f"Skipped (already present) : {n_skipped_exists}")
