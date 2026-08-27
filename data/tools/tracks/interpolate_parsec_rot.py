"""
Script to identify and fill gaps in the PARSEC rotating (v/vcrit) track
library.

Every v/vcrit slice of the PARSEC rotating grid is supposed to contain
the same set of initial masses across its own 13 [Fe/H] groups (this is
a Tracks3D requirement: findMatchingTracks brackets a query in [Fe/H] at
a single, fixed v/vcrit, and Tracks3D::Tracks3D() then requires every
group it reads to share one common mass grid), but PARSEC's own
per-metallicity convergence dropouts mean that in practice no two [Fe/H]
groups at a given v/vcrit necessarily agree on which masses converged.
Unlike the analogous MIST gap (see interpolate_mist.py), this is not a
small number of one-off holes against an otherwise-shared canonical
grid: at v/vcrit=0.0, for example, the union of masses across the 13
[Fe/H] groups is 116, but the single most-complete group only has 95 of
them, and every other group is missing a different subset. So the
canonical mass grid this script fills every group up to is the *union*
of masses present anywhere in that group's own v/vcrit slice, not
(as for MIST) whichever single group happens to have the most tracks.

Gaps are filled in three passes, each operating on the state left by
the previous one (mirroring interpolate_mist.py's own staged design):

  Pass 1 (mass interpolation) -- for each mass missing from a group, if
    that group has surviving tracks on both sides of it, linearly
    interpolate a new track in mass from those two neighbors (see
    interpolate_gap() below for how this differs from MIST's own
    version). This is the bulk of the fix (~65% of all gaps in this
    grid), but cannot help masses missing at the very edge of a
    group's own mass range, since there is no same-group neighbor on
    one side to interpolate from -- these are overwhelmingly (>99%)
    missing at the *low*-mass edge, and increasingly so at higher
    rotation rates, consistent with low-mass models becoming harder to
    converge as rotation increases.

  Pass 2 (v/vcrit donor) -- for a gap pass 1 could not fill, walk down
    through this group's own [Fe/H]'s other v/vcrit values, nearest
    first, and copy the track for this mass verbatim from the first
    one that has it (post-pass-1, so a donor's own interpolated tracks
    count too). Physically: if e.g. a 1.2 Msun model fails to converge
    at v/vcrit=0.3, the v/vcrit=0.0 (non-rotating) track at the same
    mass and [Fe/H] is a much better substitute than leaving the mass
    missing outright, since rotation's structural effect on low-mass
    stars is weak compared to just not having a model at all. This
    closes the large majority (~99.6%) of pass 1's leftover gaps.

  Pass 3 ([Fe/H] donor) -- for a gap pass 2 also could not fill (only
    arises at v/vcrit=0.0 itself, where there is no lower v/vcrit left
    to walk down to), walk through this group's own v/vcrit's other
    [Fe/H] values, nearest first in |ΔFe/H|, and copy the track for
    this mass verbatim from the first one that has it. Closes the
    handful of gaps (2, in the current grid) that pass 2 cannot reach.

Pass 2/3 donor tracks are marked with a 'vvcrit_donor'/'feh_donor'
boolean attribute (rather than 'interpolated') to keep their different
provenance visible, mirroring how interpolate_mist.py distinguishes its
own 'interpolated' and 'alpha/Fe_replaced' tracks.

The mass-interpolation itself (pass 1) has to work differently than
MIST's, though, because PARSEC tracks carry no discrete evolutionary
-phase marker to align mismatched-length neighbors on (MIST's Case 2).
PARSEC tracks almost never have equal lengths even between
adjacent-mass neighbors -- empirically, *every* bracketed gap in this
grid needs some form of length reconciliation, since PARSEC's own
adaptive time-stepping picks a different point count for every track
independently. What PARSEC tracks do carry is a genuine 'age' column
(index 0), and adjacent-mass neighbors' own age ranges overlap well in
practice (median ~99%, worst observed ~71%, log-scale) -- so age
stands in as the common coordinate: the shorter of the two neighbor
tracks (by point count) supplies the common age grid, and every other
field of the longer track is 1D-interpolated (np.interp) onto those
ages before the usual mass-fraction blend. A handful of PARSEC tracks
repeat the same age value across several consecutive rows
(rapid-transition substeps in the raw model output); since np.interp
needs a strictly increasing x array, repeated ages are collapsed to
their first occurrence before use.

:copyright: Copyright (c) 2026 Mark Krumholz
"""

import argparse
import shutil
from collections import defaultdict

import h5py
import numpy as np

PARSEC_ROT_FILE = shutil.os.path.join("..", "..", "tracks", "parsec_rot.h5")
MASS_TOL = 1e-4  # PARSEC mass values/dataset names are only good to 1e-3 Msun
FEH_TOL = 1e-6   # tolerance for matching [Fe/H] between groups in the same file
VVCRIT_TOL = 1e-6


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def track_dataset_name(mass: float) -> str:
    return f"track_m{mass:.3f}"


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


def dedup_by_age(data: np.ndarray) -> np.ndarray:
    """
    Collapse consecutive rows sharing the same age (column 0) down to
    their first occurrence, so the result's own age column is strictly
    increasing -- required before using it as (or interpolating onto)
    a common age grid with np.interp.
    """
    age = data[:, 0]
    keep = np.concatenate(([True], np.diff(age) > 0))
    return data[keep]


def interpolate_gap(
    lo_data: np.ndarray,
    hi_data: np.ndarray,
    lo_mass: float,
    hi_mass: float,
    target_mass: float,
) -> np.ndarray:
    """
    Compute an interpolated track at *target_mass* from neighbors at
    *lo_mass* and *hi_mass*.

    If the two neighbors happen to share the same number of points,
    they are blended directly by mass fraction (as for MIST's Case 1).
    Otherwise, the shorter (deduplicated) track's own age grid is used
    as the common time axis, the longer track's every other field is
    1D-interpolated onto it, and the two same-length results are then
    blended by mass fraction.
    """
    alpha = (target_mass - lo_mass) / (hi_mass - lo_mass)

    lo_dedup = dedup_by_age(lo_data)
    hi_dedup = dedup_by_age(hi_data)

    if len(lo_dedup) == len(hi_dedup) and np.array_equal(lo_dedup[:, 0], hi_dedup[:, 0]):
        return (1.0 - alpha) * lo_dedup + alpha * hi_dedup

    if len(lo_dedup) <= len(hi_dedup):
        short, long_ = lo_dedup, hi_dedup
    else:
        short, long_ = hi_dedup, lo_dedup

    common_age = short[:, 0]
    long_resampled = np.empty_like(short)
    long_resampled[:, 0] = common_age
    for j in range(1, short.shape[1]):
        long_resampled[:, j] = np.interp(common_age, long_[:, 0], long_[:, j])

    if len(lo_dedup) <= len(hi_dedup):
        lo_aligned, hi_aligned = short, long_resampled
    else:
        lo_aligned, hi_aligned = long_resampled, short

    return (1.0 - alpha) * lo_aligned + alpha * hi_aligned


def find_vvcrit_donor(
    group_attrs: dict[str, tuple[float, float]],
    group_masses: dict[str, np.ndarray],
    grp: str,
    target_mass: float,
) -> str | None:
    """
    Find the best donor group to borrow *target_mass* from when *grp*
    has no usable same-group mass bracket for it.

    Searches this group's own [Fe/H] at every *lower* v/vcrit (nearest
    first) for one that currently has *target_mass*. Returns None if no
    such donor exists (only possible when grp is already at v/vcrit=0,
    the bottom of the rotation axis).
    """
    feh, vvcrit = group_attrs[grp]
    candidates: list[tuple[float, str]] = []
    for other_grp, (o_feh, o_vvcrit) in group_attrs.items():
        if other_grp == grp:
            continue
        if abs(o_feh - feh) > FEH_TOL:
            continue
        if o_vvcrit >= vvcrit - VVCRIT_TOL:
            continue
        if np.any(np.abs(group_masses[other_grp] - target_mass) < MASS_TOL):
            candidates.append((vvcrit - o_vvcrit, other_grp))
    if not candidates:
        return None
    candidates.sort(key=lambda c: c[0])
    return candidates[0][1]


def find_feh_donor(
    group_attrs: dict[str, tuple[float, float]],
    group_masses: dict[str, np.ndarray],
    grp: str,
    target_mass: float,
) -> str | None:
    """
    Find the best donor group to borrow *target_mass* from when *grp*
    has no usable same-group mass bracket for it, and find_vvcrit_donor()
    also found nothing (i.e. grp is already at v/vcrit=0).

    Searches this group's own v/vcrit at every other [Fe/H] (nearest
    first in |delta Fe/H|) for one that currently has *target_mass*.
    Returns None if no such donor exists.
    """
    feh, vvcrit = group_attrs[grp]
    candidates: list[tuple[float, str]] = []
    for other_grp, (o_feh, o_vvcrit) in group_attrs.items():
        if other_grp == grp:
            continue
        if abs(o_vvcrit - vvcrit) > VVCRIT_TOL:
            continue
        if np.any(np.abs(group_masses[other_grp] - target_mass) < MASS_TOL):
            candidates.append((abs(o_feh - feh), other_grp))
    if not candidates:
        return None
    candidates.sort(key=lambda c: c[0])
    return candidates[0][1]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Identify and fill missing PARSEC rotating-grid mass tracks, "
                "independently within each v/vcrit slice, by mass "
                "interpolation and (for gaps that can't be interpolated) "
                "donation from a lower v/vcrit or, failing that, a "
                "neighboring [Fe/H]")
parser.add_argument("--input", default=PARSEC_ROT_FILE,
                    help="Path to the PARSEC rotating-grid HDF5 track file "
                         "(default: %(default)s)")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Pass 1: mass interpolation.
# ---------------------------------------------------------------------------

with h5py.File(args.input, "r") as h5:
    groups_by_vvcrit: dict[float, list[str]] = defaultdict(list)
    group_masses: dict[str, np.ndarray] = {}
    for g in h5.keys():
        vvcrit = float(h5[g].attrs["vvcrit"])
        groups_by_vvcrit[vvcrit].append(g)
        group_masses[g] = h5[g]["masses"][:]

group_work: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
group_unbracketed: dict[str, list[float]] = defaultdict(list)

for vvcrit, grps in groups_by_vvcrit.items():
    union_masses = np.array(sorted(
        {round(float(m), 4) for g in grps for m in group_masses[g]}))
    for g in grps:
        masses = group_masses[g]
        missing = [m for m in union_masses
                  if not np.any(np.abs(masses - m) < MASS_TOL)]
        for m in missing:
            below = masses[masses < m - MASS_TOL]
            above = masses[masses > m + MASS_TOL]
            if len(below) and len(above):
                group_work[g].append((float(m), float(below.max()), float(above.min())))
            else:
                group_unbracketed[g].append(m)

total_work = sum(len(v) for v in group_work.values())
total_unbracketed = sum(len(v) for v in group_unbracketed.values())
print(f"Pass 1 (mass interpolation): {total_work} fillable gap(s) in "
      f"{len(group_work)} group(s); {total_unbracketed} gap(s) in "
      f"{len(group_unbracketed)} group(s) have no same-group mass bracket "
      f"and will be handled by passes 2/3.\n")

n_interpolated = 0
n_skipped_exists = 0

if group_work:
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
                new_data = interpolate_gap(lo_data, hi_data, lo_mass, hi_mass, target_mass)

                ds = h5[grp].create_dataset(ds_name, data=new_data, compression="gzip")
                ds.attrs["interpolated"] = True
                new_masses.append(target_mass)
                n_interpolated += 1

            if new_masses:
                old_masses = h5[grp]["masses"][:]
                updated = np.sort(np.append(old_masses, new_masses))
                del h5[grp]["masses"]
                h5[grp].create_dataset("masses", data=updated)
                h5[grp].attrs["nmass"] = len(updated)

print(f"Pass 1 interpolated : {n_interpolated} tracks")
if n_skipped_exists:
    print(f"Pass 1 skipped (already present) : {n_skipped_exists}")

# ---------------------------------------------------------------------------
# Pass 2: v/vcrit donor, for gaps pass 1 could not bracket.
# ---------------------------------------------------------------------------

with h5py.File(args.input, "r") as h5:
    group_attrs: dict[str, tuple[float, float]] = {
        g: (float(h5[g].attrs["feh"]), float(h5[g].attrs["vvcrit"]))
        for g in h5.keys()
    }
    group_masses_now: dict[str, np.ndarray] = {g: h5[g]["masses"][:] for g in h5.keys()}

vvcrit_work: dict[str, list[tuple[float, str]]] = defaultdict(list)
still_missing_after_2: dict[str, list[float]] = defaultdict(list)

for grp, missing in group_unbracketed.items():
    for m in missing:
        donor = find_vvcrit_donor(group_attrs, group_masses_now, grp, m)
        if donor is not None:
            vvcrit_work[grp].append((m, donor))
        else:
            still_missing_after_2[grp].append(m)

total_vvcrit_work = sum(len(v) for v in vvcrit_work.values())
print(f"\nPass 2 (v/vcrit donor): {total_vvcrit_work} gap(s) in "
      f"{len(vvcrit_work)} group(s) have a lower-v/vcrit donor at the same "
      f"[Fe/H]; {sum(len(v) for v in still_missing_after_2.values())} gap(s) "
      f"do not (already at v/vcrit=0) and will be tried in pass 3.\n")

n_vvcrit_donated = 0

if vvcrit_work:
    with h5py.File(args.input, "a") as h5:
        for grp in sorted(vvcrit_work):
            new_masses: list[float] = []
            for target_mass, donor_grp in vvcrit_work[grp]:
                ds_name = track_dataset_name(target_mass)
                if ds_name in h5[grp]:
                    continue
                donor_key = find_track_key(h5[donor_grp], target_mass)
                if donor_key is None:
                    print(f"  WARNING: donor {donor_grp} no longer has "
                          f"{target_mass:.4g} M_sun for {grp}; skipping")
                    continue
                donor_data = h5[donor_grp][donor_key][:]
                ds = h5[grp].create_dataset(ds_name, data=donor_data, compression="gzip")
                ds.attrs["vvcrit_donor"] = True
                new_masses.append(target_mass)
                n_vvcrit_donated += 1

            if new_masses:
                old_masses = h5[grp]["masses"][:]
                updated = np.sort(np.append(old_masses, new_masses))
                del h5[grp]["masses"]
                h5[grp].create_dataset("masses", data=updated)
                h5[grp].attrs["nmass"] = len(updated)

print(f"Pass 2 donated : {n_vvcrit_donated} tracks")

# ---------------------------------------------------------------------------
# Pass 3: [Fe/H] donor, for gaps pass 2 also could not fill.
# ---------------------------------------------------------------------------

with h5py.File(args.input, "r") as h5:
    group_attrs = {
        g: (float(h5[g].attrs["feh"]), float(h5[g].attrs["vvcrit"]))
        for g in h5.keys()
    }
    group_masses_now = {g: h5[g]["masses"][:] for g in h5.keys()}

feh_work: dict[str, list[tuple[float, str]]] = defaultdict(list)
still_missing_after_3: dict[str, list[float]] = defaultdict(list)

for grp, missing in still_missing_after_2.items():
    for m in missing:
        donor = find_feh_donor(group_attrs, group_masses_now, grp, m)
        if donor is not None:
            feh_work[grp].append((m, donor))
        else:
            still_missing_after_3[grp].append(m)

total_feh_work = sum(len(v) for v in feh_work.values())
n_still_missing = sum(len(v) for v in still_missing_after_3.values())
print(f"\nPass 3 ([Fe/H] donor): {total_feh_work} gap(s) in "
      f"{len(feh_work)} group(s) have a neighboring-[Fe/H] donor at the "
      f"same v/vcrit; {n_still_missing} gap(s) remain unfillable.\n")

n_feh_donated = 0

if feh_work:
    with h5py.File(args.input, "a") as h5:
        for grp in sorted(feh_work):
            new_masses: list[float] = []
            for target_mass, donor_grp in feh_work[grp]:
                ds_name = track_dataset_name(target_mass)
                if ds_name in h5[grp]:
                    continue
                donor_key = find_track_key(h5[donor_grp], target_mass)
                if donor_key is None:
                    print(f"  WARNING: donor {donor_grp} no longer has "
                          f"{target_mass:.4g} M_sun for {grp}; skipping")
                    continue
                donor_data = h5[donor_grp][donor_key][:]
                ds = h5[grp].create_dataset(ds_name, data=donor_data, compression="gzip")
                ds.attrs["feh_donor"] = True
                new_masses.append(target_mass)
                n_feh_donated += 1

            if new_masses:
                old_masses = h5[grp]["masses"][:]
                updated = np.sort(np.append(old_masses, new_masses))
                del h5[grp]["masses"]
                h5[grp].create_dataset("masses", data=updated)
                h5[grp].attrs["nmass"] = len(updated)

print(f"Pass 3 donated : {n_feh_donated} tracks")
if still_missing_after_3:
    print(f"\nStill missing after all three passes:")
    for grp, masses in sorted(still_missing_after_3.items()):
        print(f"  {grp}: {masses}")

# ---------------------------------------------------------------------------
# Final report: is every v/vcrit slice now uniform?
# ---------------------------------------------------------------------------

with h5py.File(args.input, "r") as h5:
    remaining_nonuniform = 0
    for vvcrit, grps in groups_by_vvcrit.items():
        counts = {g: len(h5[g]["masses"][:]) for g in grps}
        if len(set(counts.values())) > 1:
            remaining_nonuniform += 1
            print(f"\nv/vcrit={vvcrit}: STILL non-uniform "
                  f"(nmass ranges {min(counts.values())}-{max(counts.values())} "
                  f"across its {len(grps)} [Fe/H] groups)")

if remaining_nonuniform == 0:
    print("\nEvery v/vcrit slice now has a uniform mass grid across its own "
          "[Fe/H] groups.")
else:
    print(f"\n{remaining_nonuniform} v/vcrit slice(s) still non-uniform; "
          f"Tracks3D will still fail for those.")
