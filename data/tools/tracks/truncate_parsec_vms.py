"""
Script to truncate PARSEC VMS (very-massive-star) tracks down to the
mass range shared by every [Fe/H] group, discarding the high-mass tail
that only exists for the most metal-poor compositions.

The PARSEC VMS release only computed models above 600 Msun for its
three most metal-poor [Fe/H] groups (FeH <= -2.18); every other [Fe/H]
group stops at 600 Msun. Unlike the scattered, independent convergence
dropouts found in the rotating grid (see interpolate_parsec_rot.py),
this is a clean split along [Fe/H] with no partial coverage in
between: the masses present in the smaller (88-mass) groups are an
exact subset of those in the larger (98-mass) groups, all the way up
to 600 Msun, and the ten extra masses (650-2000 Msun) in the
metal-poor groups have no counterpart anywhere else -- consistent with
a genuine boundary in what PARSEC computed (very massive stars are
astrophysically relevant mainly at low metallicity, where weak winds
let them survive as such long enough to be worth tabulating), not a
convergence defect to patch over.

Tracks3D requires every group matched by a single [Fe/H] query to
share one common mass grid, so this script truncates every group down
to the masses common to ALL of them -- discarding the metal-poor
groups' own >600 Msun tail -- rather than trying to interpolate or
donate data no metallicity-adjacent group actually has.

Run standalone against an existing PARSEC VMS HDF5 track file:
    python3 truncate_parsec_vms.py /path/to/parsec_vms.h5
or import truncate_parsec_vms() to call it as part of another script
(see fetch_parsec_vms.py, which calls this automatically after
fetching new tracks, unless run with --no_truncate).

:copyright: Copyright (c) 2026 Mark Krumholz
"""

import argparse

import h5py
import numpy as np


def truncate_parsec_vms(h5_filename: str) -> float:
    """
    Truncate every group in h5_filename down to the mass grid common
    to all groups, in place.

    Computes the intersection of every group's own masses array, then
    -- for any group that has masses beyond that common set -- deletes
    the corresponding track_m* datasets and rewrites that group's own
    masses/nmass/m_max attributes. A group whose own masses are
    already a subset of the common set is left untouched.

    Returns the common maximum mass every group ends up sharing.
    """
    with h5py.File(h5_filename, "r+") as h5file:
        group_masses = {
            grp_name: set(np.round(h5file[grp_name]["masses"][:], 3))
            for grp_name in h5file
        }
        common_masses = set.intersection(*group_masses.values())
        if not common_masses:
            raise RuntimeError(
                f"{h5_filename}: no mass value is common to every group; "
                "nothing to truncate to")
        m_common_max = max(common_masses)

        for grp_name, masses in group_masses.items():
            extra = sorted(masses - common_masses)
            if not extra:
                continue
            grp = h5file[grp_name]
            for m in extra:
                ds_name = f"track_m{m:.3f}"
                if ds_name in grp:
                    del grp[ds_name]
            remaining = np.array(sorted(masses & common_masses))
            del grp["masses"]
            grp.create_dataset("masses", data=remaining)
            grp.attrs["nmass"] = len(remaining)
            grp.attrs["m_max"] = float(remaining.max())

    return m_common_max


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Truncate PARSEC VMS tracks down to the mass range "
                    "common to every [Fe/H] group")
    parser.add_argument("h5file", help="PARSEC VMS HDF5 track file to truncate in place")
    args = parser.parse_args()
    m_max = truncate_parsec_vms(args.h5file)
    print(f"Truncated {args.h5file} to the mass grid common to every "
          f"[Fe/H] group (max mass {m_max:.3f} Msun).")
