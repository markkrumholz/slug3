"""
Post-processing script for Geneva/SYCLIST HDF5 track files produced by
fetch_geneva.py.  Applies three fixes and then splits the data into the
files slug expects:

Fix 1 – Repeated-time pruning
    Geneva track files contain runs of rows whose age is identical to the
    previous row (phase-boundary duplicates for massive stars; terminal-
    state padding for low-mass stars).  We walk each track forward and
    keep only the first occurrence at each age, i.e. rows whose age
    strictly exceeds the running maximum.

Fix 2 – Mass monotonicity
    A handful of tracks have isolated one-step increases in present-day
    mass (numerical noise at phase restarts).  We apply a cumulative
    minimum to the mass column so it is guaranteed non-increasing.

Fix 3 – Mass-grid rectification and file split
    Slug's 3-D track interpolator requires a rectangular grid in mass ×
    [Fe/H].  The raw download has:
      • Z=0.0004 ([Fe/H]=−1.54): only 17 masses (no low-mass stars)
      • Z=0.002  ([Fe/H]=−0.85): 24 masses  ← common grid
      • Z=0.006  ([Fe/H]=−0.37): 24 masses  ← common grid
      • Z=0.014  ([Fe/H]= 0.00): 31 masses (7 extra intermediate masses)

    Output:
      geneva.h5        – Z=0.002, 0.006, 0.014 with the common 24-mass
                         grid (rectangular); Z=0.0004 omitted.
      geneva_Z0004.h5  – Z=0.0004 only, all 17 masses.
      geneva_Z014.h5   – Z=0.014 only, all 31 masses.

    All three output files get fixes 1 and 2 applied.

The registry (tracks.toml) is updated to reflect all three files.

Usage:
    python3 prune_geneva.py [--input geneva.h5] [--registry tracks.toml]
                            [--verbose]

:copyright: Copyright (c) 2026 Mark Krumholz
"""

import argparse
import os

import h5py
import numpy as np
import tomlkit

# ── command-line interface ────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="Post-process Geneva tracks: prune repeated ages, fix mass "
                "monotonicity, and split into rectangular-grid files")
parser.add_argument("--input",
                    default=os.path.join("..", "..", "tracks", "geneva.h5"),
                    help="Input HDF5 file written by fetch_geneva.py")
parser.add_argument("--registry",
                    default=os.path.join("..", "..", "tracks", "tracks.toml"),
                    help="tracks.toml registry to update")
parser.add_argument("--verbose", action="store_true",
                    help="Print per-track pruning statistics")
args = parser.parse_args()

out_dir       = os.path.dirname(args.input)
main_out      = args.input                                         # overwritten in-place
z0004_out     = os.path.join(out_dir, "geneva_Z0004.h5")
z014_out      = os.path.join(out_dir, "geneva_Z014.h5")

# ── core fix routines ─────────────────────────────────────────────────────────

def prune_repeated_ages(arr: np.ndarray) -> np.ndarray:
    """
    Return rows of arr (age in column 0) whose age strictly exceeds the
    running maximum of all previously kept rows.  Always keeps row 0.

    The cumulative-max identity makes this O(N): a row is kept iff
    age[i] > max(age[:i]).  Dropping a non-advancing row does not change
    the running max, so the two criteria are self-consistent.
    """
    if arr.shape[0] == 0:
        return arr
    age = arr[:, 0]
    running_max_before = np.concatenate(([-np.inf],
                                         np.maximum.accumulate(age[:-1])))
    keep = age > running_max_before
    keep[0] = True
    return arr[keep]


def fix_mass_monotonicity(arr: np.ndarray) -> np.ndarray:
    """
    Return arr with column 1 (present-day mass) replaced by its
    cumulative minimum, guaranteeing the mass is non-increasing.
    """
    out = arr.copy()
    out[:, 1] = np.minimum.accumulate(arr[:, 1])
    return out


def fix_track(arr: np.ndarray) -> np.ndarray:
    """Apply both fixes to a single track array."""
    arr = prune_repeated_ages(arr)
    arr = fix_mass_monotonicity(arr)
    return arr

# ── discover the common mass grid ─────────────────────────────────────────────

# Read all groups and classify them by their [Fe/H] rounded to 2 d.p.
with h5py.File(args.input, "r") as src:
    group_feh   = {}   # grp_name → rounded feh
    group_masses = {}  # grp_name → frozenset of rounded masses

    for gname in src.keys():
        g   = src[gname]
        feh = round(float(g.attrs["feh"]), 2)
        ms  = frozenset(round(float(m), 4) for m in g["masses"][:])
        group_feh[gname]    = feh
        group_masses[gname] = ms

# The "main" metallicities that go into geneva.h5
MAIN_FEH = {round(np.log10(z / 0.014), 2) for z in (0.002, 0.006, 0.014)}
SOLO_Z0004_FEH = round(np.log10(0.0004 / 0.014), 2)
SOLO_Z014_FEH  = round(np.log10(0.014  / 0.014), 2)

# Common mass grid = intersection of masses across all main-file metallicities
# (one entry per vvcrit, but both vvcrit share the same mass set per feh,
# so we just intersect over all groups that belong to MAIN_FEH)
main_groups = [g for g, f in group_feh.items() if f in MAIN_FEH]
common_masses = frozenset.intersection(*(group_masses[g] for g in main_groups))
common_masses_sorted = sorted(common_masses)
if args.verbose:
    print(f"Common mass grid ({len(common_masses_sorted)} masses): "
          f"{common_masses_sorted}")

# ── helper: copy one group into an open HDF5 output file ─────────────────────

def write_group(grp_in, h5out, grp_name,
                mass_filter=None, verbose=False):
    """
    Copy grp_in into h5out under grp_name, applying fix_track to every
    track dataset.  If mass_filter is a set, only copy tracks whose
    initial mass (rounded to 4 d.p.) is in that set and update the
    'masses' dataset and related attrs accordingly.

    Returns (n_tracks, pts_before, pts_after, n_changed).
    """
    masses_in = grp_in["masses"][:]

    # Decide which masses to include
    if mass_filter is None:
        masses_out = masses_in
    else:
        masses_out = np.array([m for m in masses_in
                               if round(float(m), 4) in mass_filter])

    grp_out = h5out.create_group(grp_name)
    for k, v in grp_in.attrs.items():
        grp_out.attrs[k] = v
    grp_out.attrs["nmass"] = len(masses_out)
    grp_out.attrs["m_min"] = float(masses_out.min())
    grp_out.attrs["m_max"] = float(masses_out.max())

    grp_out.create_dataset("masses", data=masses_out)

    n_tracks = pts_before = pts_after = n_changed = 0
    max_ntime = 0

    for mi in masses_out:
        key  = f"track_m{mi:.3f}"
        data = grp_in[key][:]
        nb   = data.shape[0]
        fixed = fix_track(data)
        na    = fixed.shape[0]

        grp_out.create_dataset(key, data=fixed, compression="gzip")

        n_tracks   += 1
        pts_before += nb
        pts_after  += na
        max_ntime   = max(max_ntime, na)
        if na != nb:
            n_changed += 1
            if verbose:
                print(f"  {grp_name}/{key}: {nb} → {na} pts "
                      f"({nb - na} pruned)")

    grp_out.attrs["ntime"] = max_ntime
    return n_tracks, pts_before, pts_after, n_changed

# ── write the three output files ──────────────────────────────────────────────

totals = {"n_tracks": 0, "pts_before": 0, "pts_after": 0, "n_changed": 0}

def accumulate(t, n, pb, pa, nc):
    t["n_tracks"]  += n
    t["pts_before"] += pb
    t["pts_after"]  += pa
    t["n_changed"]  += nc

# We build each output into a sibling temp file first, then replace
# atomically so the source file is never left in a partial state.

tmp_main  = main_out  + ".prune_tmp"
tmp_z0004 = z0004_out + ".prune_tmp"
tmp_z014  = z014_out  + ".prune_tmp"

with h5py.File(args.input, "r") as src, \
     h5py.File(tmp_main,   "w") as out_main, \
     h5py.File(tmp_z0004,  "w") as out_z0004, \
     h5py.File(tmp_z014,   "w") as out_z014:

    # Copy top-level file attributes (references, reference_urls)
    for attr_name, attr_val in src.attrs.items():
        out_main.attrs[attr_name]  = attr_val
        out_z0004.attrs[attr_name] = attr_val
        out_z014.attrs[attr_name]  = attr_val

    for gname in sorted(src.keys()):
        g   = src[gname]
        feh = group_feh[gname]

        if feh == SOLO_Z0004_FEH:
            # Goes into geneva_Z0004.h5 only (all masses)
            n, pb, pa, nc = write_group(g, out_z0004, gname,
                                        verbose=args.verbose)
            accumulate(totals, n, pb, pa, nc)
            if args.verbose:
                print(f"[Z0004] {gname}: {n} tracks")

        elif feh == SOLO_Z014_FEH:
            # Goes into geneva_Z014.h5 (all masses) AND into
            # geneva.h5 (trimmed to common_masses)
            n, pb, pa, nc = write_group(g, out_z014, gname,
                                        verbose=args.verbose)
            accumulate(totals, n, pb, pa, nc)
            if args.verbose:
                print(f"[Z014 full] {gname}: {n} tracks")

            n, pb, pa, nc = write_group(g, out_main, gname,
                                        mass_filter=common_masses,
                                        verbose=args.verbose)
            accumulate(totals, n, pb, pa, nc)
            if args.verbose:
                print(f"[main, trimmed] {gname}: {n} tracks")

        elif feh in MAIN_FEH:
            # Z002 and Z006: go into geneva.h5 with all their masses
            n, pb, pa, nc = write_group(g, out_main, gname,
                                        verbose=args.verbose)
            accumulate(totals, n, pb, pa, nc)
            if args.verbose:
                print(f"[main] {gname}: {n} tracks")

        else:
            print(f"Warning: unrecognised [Fe/H]={feh} in {gname}; skipping")

# Atomic replace
os.replace(tmp_main,  main_out)
os.replace(tmp_z0004, z0004_out)
os.replace(tmp_z014,  z014_out)

print(f"Written {main_out}")
print(f"Written {z0004_out}")
print(f"Written {z014_out}")
print(f"Totals: {totals['n_changed']} of {totals['n_tracks']} tracks modified "
      f"({totals['pts_before']} → {totals['pts_after']} points)")

# ── update tracks.toml ────────────────────────────────────────────────────────

if os.path.exists(args.registry):
    with open(args.registry, "r") as fp:
        registry = tomlkit.parse(fp.read())
else:
    registry = {"name": "Registry of track sets"}

# Collect reference strings from the source HDF5
with h5py.File(main_out, "r") as h5:
    references     = list(h5.attrs.get("references",     []))
    reference_urls = list(h5.attrs.get("reference_urls", []))

def make_geneva_tab(h5_filename, track_set_name):
    """Build a tomlkit table entry for one Geneva HDF5 file."""
    tab = tomlkit.table()
    tab["file"]           = os.path.basename(h5_filename)
    tab["references"]     = references
    tab["reference_urls"] = reference_urls

    with h5py.File(h5_filename, "r") as h5:
        for qty, attr in (("Fe_H", "feh"), ("v_vcrit", "vvcrit")):
            seen = []
            for gname in h5.keys():
                v = float(h5[gname].attrs[attr])
                if not any(abs(v - s) < 1e-6 for s in seen):
                    seen.append(v)
            seen.sort()
            tab[qty] = seen

    return tab

# Remove old Geneva entry; add all three
for name in ("Geneva", "Geneva_Z0004", "Geneva_Z014"):
    if name in registry:
        registry.pop(name)

registry["Geneva"]       = make_geneva_tab(main_out,  "Geneva")
registry["Geneva_Z0004"] = make_geneva_tab(z0004_out, "Geneva_Z0004")
registry["Geneva_Z014"]  = make_geneva_tab(z014_out,  "Geneva_Z014")

# Ensure all three appear in track_sets
for name in ("Geneva", "Geneva_Z0004", "Geneva_Z014"):
    if "track_sets" not in registry:
        registry["track_sets"] = [name]
    elif name not in registry["track_sets"]:
        registry["track_sets"].append(name)

with open(args.registry, "w") as fp:
    fp.write(tomlkit.dumps(registry))

print(f"Updated {args.registry}")
