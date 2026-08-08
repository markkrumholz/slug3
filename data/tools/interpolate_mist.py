"""
Script to identify and fill gaps in the MIST stellar track library.

The MIST tracks are supposed to contain the same set of initial masses for
every combination of [Fe/H], [A/Fe], and v/vCrit, but a small number of
models did not converge and are absent from the file. This script locates
those gaps by comparing each group's mass list against the canonical
(maximum-count) mass grid, then fills them by linear interpolation in mass
between the two nearest available tracks.

Some tracks are not missing outright but *stalled*: the underlying MIST
model run failed to converge partway through and was released as-is, far
short of any real evolutionary endpoint. For example, the 0.40 Msun track
at [Fe/H]=+0.50, [a/Fe]=+0.4 stops after only 2 EEPs / 202 points, still
sitting on the unremarkable main sequence, while its 0.45 Msun neighbor
runs the full 454 points out to ~100 Gyr -- confirmed against the raw
MIST v2.5 EEP file itself (`N_EEP = 2` in the file header), so this is a
genuine defect in the released MIST data, not a bug in this fetch/parse
pipeline. Because more massive stars must have shorter (or equal, modulo
minor real non-monotonicity) post-MS lifetimes than less massive ones, a
track whose final tabulated age is far shorter than some *more massive*
track's final age in the same group is physically impossible unless the
less massive one stalled. Such tracks are detected (see
`find_stalled_masses`) and treated exactly like a missing track: deleted
from their group and re-filled by the same neighbor-interpolation logic
used for gaps that were missing outright.

Gaps (both kinds) fall into two cases:
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
STALL_DEX_THRESHOLD = 1.0  # flag a track as stalled if some more massive
                           # track in the same group has a final tabulated
                           # age more than this many dex (log10) larger


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


def find_stalled_masses(
    h5: h5py.File, dex_threshold: float = STALL_DEX_THRESHOLD
) -> dict[str, list[float]]:
    """
    Identify tracks whose MIST model run stalled/failed to converge.

    For every group, sorts tracks by mass and compares each track's final
    tabulated age against the largest final age among all *more massive*
    tracks in the same group. Since final age must be non-increasing with
    mass (barring minor real non-monotonicity), any track whose final age
    falls more than *dex_threshold* dex short of a more massive track's
    final age is flagged as stalled -- unless it ends at phase 6, which is
    a legitimate, deliberately short post-AGB/white-dwarf-cooling
    truncation (see mist_truncate_wd.py), not a stall. Returns
    {group_name: [stalled masses]}.
    """
    stalled: dict[str, list[float]] = defaultdict(list)
    for grp_name in h5:
        grp = h5[grp_name]
        field_names = list(grp.attrs["field_names"])
        age_idx = field_names.index("age")
        phase_idx = field_names.index("phase")
        masses = grp["masses"][:]
        order = np.argsort(masses)
        masses_s = masses[order]

        final_logage = np.empty(len(masses_s))
        final_phase = np.empty(len(masses_s))
        for i, m in enumerate(masses_s):
            track = grp[track_dataset_name(m)][:]
            final_logage[i] = np.log10(track[-1, age_idx])
            final_phase[i] = track[-1, phase_idx]

        # Running max of final_logage among strictly more massive tracks.
        suffix_max = np.full(len(masses_s), -np.inf)
        running_max = -np.inf
        for i in range(len(masses_s) - 1, -1, -1):
            suffix_max[i] = running_max
            running_max = max(running_max, final_logage[i])

        for i, m in enumerate(masses_s):
            if final_phase[i] == 6.0:
                continue
            if suffix_max[i] - final_logage[i] > dex_threshold:
                stalled[grp_name].append(float(m))

    return stalled


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


def find_afe_donor(
    group_attrs: dict[str, tuple[float, float, float]],
    group_masses: dict[str, np.ndarray],
    grp: str,
    target_mass: float,
) -> str | None:
    """
    Find the best donor group to borrow *target_mass* from when *grp* has
    no usable same-group mass bracket for it (e.g. an entire low-mass
    block stalled with nothing below the bottom of the grid to
    interpolate from).

    Searches all other groups sharing the same [Fe/H] and v/vcrit as
    *grp* for ones that currently have *target_mass*, and returns the one
    closest in [a/Fe] (alpha/Fe has a much weaker effect on low-mass
    main-sequence structure than extrapolating past the edge of the mass
    grid, or borrowing from a different Fe/H entirely). Returns None if no
    such donor exists.
    """
    feh, afe, vvcrit = group_attrs[grp]
    candidates: list[tuple[float, str]] = []
    for other_grp, (o_feh, o_afe, o_vvcrit) in group_attrs.items():
        if other_grp == grp:
            continue
        if abs(o_feh - feh) > MASS_TOL or abs(o_vvcrit - vvcrit) > MASS_TOL:
            continue
        if np.any(np.abs(group_masses[other_grp] - target_mass) < MASS_TOL):
            candidates.append((abs(o_afe - afe), other_grp))
    if not candidates:
        return None
    candidates.sort(key=lambda c: c[0])
    return candidates[0][1]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Identify and fill missing MIST mass tracks by interpolation")
parser.add_argument("--input", default=MIST_FILE,
                    help="Path to the MIST HDF5 track file (default: %(default)s)")
parser.add_argument("--stall_dex_threshold", type=float, default=STALL_DEX_THRESHOLD,
                    help="Flag a track as stalled if some more massive track "
                         "in the same group has a final tabulated age more "
                         "than this many dex (log10) larger (default: %(default)s)")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Stalled-track pass: find tracks whose MIST run stalled/failed to
# converge, and delete them so they get treated as gaps and re-filled by
# the same interpolation logic used for tracks that were missing outright.
# ---------------------------------------------------------------------------

with h5py.File(args.input, "r") as h5:
    stalled = find_stalled_masses(h5, args.stall_dex_threshold)

n_stalled = sum(len(v) for v in stalled.values())
if n_stalled:
    print(f"Found {n_stalled} stalled track(s) in {len(stalled)} group(s); "
          f"removing them so they are re-filled by interpolation.\n")
    with h5py.File(args.input, "a") as h5:
        for grp_name, masses in stalled.items():
            grp = h5[grp_name]
            for m in masses:
                key = find_track_key(grp, m)
                if key is not None:
                    del grp[key]
            remaining = np.array(
                [mm for mm in grp["masses"][:]
                 if not any(abs(mm - sm) < MASS_TOL for sm in masses)]
            )
            del grp["masses"]
            grp.create_dataset("masses", data=remaining)
            grp.attrs["nmass"] = len(remaining)
else:
    print("No stalled tracks found.\n")

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

# ---------------------------------------------------------------------------
# Alpha/Fe fallback pass: masses that still have no same-group mass
# bracket (e.g. an entire low-mass block stalled with nothing below the
# bottom of the grid to interpolate from) are filled by borrowing the
# track for that mass from the nearest [a/Fe] value at the same [Fe/H]
# and v/vcrit, rather than left missing. These are tagged
# 'alpha/Fe_replaced', not 'interpolated', since they are not an
# interpolation in mass at all -- they're a different composition's track
# substituted in outright.
# ---------------------------------------------------------------------------

with h5py.File(args.input, "r") as h5:
    group_attrs = {
        g: (float(h5[g].attrs["feh"]), float(h5[g].attrs["afe"]), float(h5[g].attrs["vvcrit"]))
        for g in h5.keys()
    }
    group_masses_now = {g: h5[g]["masses"][:] for g in h5.keys()}

afe_work: dict[str, list[float]] = defaultdict(list)
for grp, masses in group_masses_now.items():
    still_missing = [m for m in ref_masses if not np.any(np.abs(masses - m) < MASS_TOL)]
    if still_missing:
        afe_work[grp] = still_missing

n_replaced = 0
n_no_donor = 0

if afe_work:
    total_afe_missing = sum(len(v) for v in afe_work.values())
    print(f"\nFound {len(afe_work)} group(s) with {total_afe_missing} track(s) "
          f"lacking a same-group mass bracket; searching for [a/Fe] donors.\n")

    with h5py.File(args.input, "a") as h5:
        for grp in sorted(afe_work):
            new_masses: list[float] = []
            for target_mass in afe_work[grp]:
                donor_grp = find_afe_donor(group_attrs, group_masses_now, grp, target_mass)
                if donor_grp is None:
                    print(f"  WARNING: no [a/Fe] donor available for "
                          f"{target_mass:.4g} M_sun in {grp}; leaving missing")
                    n_no_donor += 1
                    continue

                ds_name = track_dataset_name(target_mass)
                if ds_name in h5[grp]:
                    continue

                donor_key = find_track_key(h5[donor_grp], target_mass)
                donor_data = h5[donor_grp][donor_key][:]

                ds = h5[grp].create_dataset(ds_name, data=donor_data)
                ds.attrs["alpha/Fe_replaced"] = True
                new_masses.append(target_mass)
                n_replaced += 1

            if new_masses:
                old_masses = h5[grp]["masses"][:]
                updated = np.sort(np.append(old_masses, new_masses))
                del h5[grp]["masses"]
                h5[grp].create_dataset("masses", data=updated)
                h5[grp].attrs["nmass"] = len(updated)

    print(f"\n[a/Fe] replaced : {n_replaced} tracks")
    if n_no_donor:
        print(f"No donor found : {n_no_donor} tracks left missing")
