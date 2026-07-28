"""
Script to identify and fill gaps in the MIST stellar track library.

The MIST tracks are supposed to contain the same set of initial masses for
every combination of [Fe/H], [A/Fe], and v/vCrit, but a small number of
models did not converge and are absent from the file. This script locates
those gaps by comparing each group's mass list against the canonical
(maximum-count) mass grid, then reports which masses are missing from which
groups. A later pass will interpolate the missing tracks from their
nearest-mass neighbours.
"""

import argparse
import shutil
import h5py
import numpy as np

# Default path to the MIST track file
MIST_FILE = shutil.os.path.join("..", "tracks", "mist.h5")

# Tolerance for floating-point comparison of mass values
MASS_TOL = 1e-6

# Parse command line arguments
parser = argparse.ArgumentParser(
    description="Identify (and later fill) missing MIST mass tracks")
parser.add_argument("--input", default=MIST_FILE,
                    help="Path to the MIST HDF5 track file (default: %(default)s)")
args = parser.parse_args()

# Open the file and read the nmass attribute and mass array for every group
with h5py.File(args.input, "r") as h5:
    group_nmass: dict[str, int] = {
        g: int(h5[g].attrs["nmass"]) for g in h5.keys()
    }
    group_masses: dict[str, np.ndarray] = {
        g: h5[g]["masses"][:] for g in h5.keys()
    }

# The canonical mass grid is taken from a complete group (one whose nmass
# equals the maximum). All complete groups have identical mass arrays.
n_max = max(group_nmass.values())
ref_group = next(g for g, n in group_nmass.items() if n == n_max)
ref_masses = group_masses[ref_group]

# Identify groups that are missing one or more mass tracks
incomplete: dict[str, list[float]] = {}
for grp, masses in group_masses.items():
    if len(masses) == n_max:
        continue
    missing = [
        float(m) for m in ref_masses
        if not np.any(np.abs(masses - m) < MASS_TOL)
    ]
    if missing:
        incomplete[grp] = missing

if not incomplete:
    print(f"All {len(group_nmass)} groups have the full set of {n_max} mass "
          "tracks. Nothing to do.")
else:
    n_missing_total = sum(len(v) for v in incomplete.values())
    print(f"Found {len(incomplete)} group(s) with missing mass tracks "
          f"({n_missing_total} missing tracks in total, out of a canonical "
          f"grid of {n_max} masses):\n")
    print(f"  {'Group':<45s}  {'Have':>5s}  {'Missing masses (M_sun)'}")
    print(f"  {'-'*45}  {'-'*5}  {'-'*40}")
    for grp_name in sorted(incomplete):
        missing = incomplete[grp_name]
        n_have = group_nmass[grp_name]
        missing_str = ", ".join(f"{m:.4g}" for m in missing)
        print(f"  {grp_name:<45s}  {n_have:>5d}  {missing_str}")
