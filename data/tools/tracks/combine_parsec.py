"""
Script to build a composite PARSEC (PAdova and TRieste Stellar Evolution
Code) track grid with full stellar mass coverage.

The tracks released by the PARSEC group come in two separate sets that
were fetched by fetch_parsec_rot.py and fetch_parsec_vms.py:
  - "rot": a range of rotation rates (v/vcrit) for low- to
    intermediate-mass stars.
  - "vms": non-rotating tracks for intermediate- to very-massive stars.

Neither set covers the full mass range on its own, and each has denser
mass coverage than the other over part of the range covered by both (the
rotating grid below ~6 Msun, the VMS grid at and above it). This script
splices the two together at a given mass, for every [Fe/H] value present
in the non-rotating (v/vcrit = 0) subset of the rotating grid and also
present in the VMS grid, producing a single composite HDF5 file formatted
exactly like the VMS file (same group naming and attribute conventions).
"""

# Imports
import argparse
import h5py
import numpy as np
import shutil
import tomlkit

# Magic strings
PARSEC_version = "v2.0"
PARSEC_ROT_URL = "https://stev.oapd.inaf.it/PARSEC/Database/PARSECv2.0_ROT_2025/"
PARSEC_VMS_URL = "https://stev.oapd.inaf.it/PARSEC/Database/PARSECv2.0_VMS/"
MASS_BREAK = 6.0    # Msun; tracks below this mass are taken from the
                    # rotating (v/vcrit=0) grid, tracks at or above it
                    # are taken from the VMS grid
FEH_TOL = 1e-4      # tolerance for matching [Fe/H] between the two grids

# Parse command line arguments
parser = argparse.ArgumentParser(
    description="Combine PARSEC rotating (v/vcrit=0) and VMS tracks into "
                "a composite grid with full mass coverage")
parser.add_argument("--rot",
                    default=shutil.os.path.join("..", "..", "tracks", "parsec_rot.h5"),
                    help="Input HDF5 file containing the rotating PARSEC tracks")
parser.add_argument("--vms",
                    default=shutil.os.path.join("..", "..", "tracks", "parsec_vms.h5"),
                    help="Input HDF5 file containing the VMS PARSEC tracks")
parser.add_argument("--output",
                    default=shutil.os.path.join("..", "..", "tracks", "parsec_composite.h5"),
                    help="Output file for the composite HDF5 tracks")
parser.add_argument("--registry",
                    default=shutil.os.path.join("..", "..", "tracks", "tracks.toml"),
                    help="Output file for the registry")
parser.add_argument("--mass_break", type=float, default=MASS_BREAK,
                    help="Initial mass (Msun) at which to switch from the "
                         "rotating grid to the VMS grid")
parser.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing output file")
parser.add_argument("--verbose", action="store_true",
                    help="Print verbose output")
args = parser.parse_args()

if shutil.os.path.exists(args.output) and not args.overwrite:
    raise RuntimeError(f"Output file {args.output} already exists; "
                       "pass --overwrite to regenerate it")


def merge_unique(list1: list, list2: list) -> list:
    """
    Merge two lists, preserving order and dropping exact duplicates.

    Parameters
    ----------
    list1 : list
        First list; all its elements are kept, in order.
    list2 : list
        Second list; only elements not already in *list1* are appended,
        in order.

    Returns
    -------
    list
        The merged list.
    """
    merged = list(list1)
    for item in list2:
        if item not in merged:
            merged.append(item)
    return merged


def merge_references(refs1: list, urls1: list, refs2: list, urls2: list) -> tuple[list, list]:
    """
    Merge two parallel (reference, reference_url) list pairs.

    Parameters
    ----------
    refs1, urls1 : list
        References and corresponding URLs from the first source.
    refs2, urls2 : list
        References and corresponding URLs from the second source.

    Returns
    -------
    tuple[list, list]
        The merged references and reference URLs, with exact-duplicate
        (reference, url) pairs dropped and order preserved.
    """
    pairs = merge_unique(list(zip(refs1, urls1)), list(zip(refs2, urls2)))
    refs = [p[0] for p in pairs]
    urls = [p[1] for p in pairs]
    return refs, urls


def track_dataset_name(mass: float) -> str:
    """Return the HDF5 dataset name used for the track at *mass*."""
    return f"track_m{mass:.3f}"


# Open the two source files
rot_file = h5py.File(args.rot, 'r')
vms_file = h5py.File(args.vms, 'r')

# Identify the non-rotating (v/vcrit = 0) groups in the rotating file
rot_groups = { grp: float(rot_file[grp].attrs['feh']) for grp in rot_file.keys()
              if abs(rot_file[grp].attrs['vvcrit']) < 1e-8 }
vms_groups = { grp: float(vms_file[grp].attrs['feh']) for grp in vms_file.keys() }

# Match rotating (v/vcrit=0) groups to VMS groups by [Fe/H]
matches = []    # list of (rot_group, vms_group, feh)
matched_vms = set()
for rgrp, rfeh in rot_groups.items():
    vgrp = next((vg for vg, vfeh in vms_groups.items()
                if abs(rfeh - vfeh) < FEH_TOL), None)
    if vgrp is None:
        if args.verbose:
            print(f"No VMS match for rotating group {rgrp} "
                  f"([Fe/H]={rfeh:.6g}); skipping.")
        continue
    matches.append((rgrp, vgrp, rfeh))
    matched_vms.add(vgrp)

if args.verbose:
    for vgrp, vfeh in vms_groups.items():
        if vgrp not in matched_vms:
            print(f"No rotating (v/vcrit=0) match for VMS group {vgrp} "
                  f"([Fe/H]={vfeh:.6g}); skipping.")

print(f"Matched {len(matches)} of {len(rot_groups)} rotating (v/vcrit=0) "
      f"groups to {len(vms_groups)} VMS groups.")

# Build the composite file
with h5py.File(args.output, 'w') as out_file:

    # Combine top-level provenance attributes from both source files
    references, reference_urls = merge_references(
        list(rot_file.attrs['references']), list(rot_file.attrs['reference_urls']),
        list(vms_file.attrs['references']), list(vms_file.attrs['reference_urls']))
    download_urls = merge_unique([PARSEC_ROT_URL], [PARSEC_VMS_URL])
    out_file.attrs['references'] = references
    out_file.attrs['reference_urls'] = reference_urls
    out_file.attrs['download_urls'] = download_urls

    for rgrp, vgrp, feh in matches:

        rot_grp = rot_file[rgrp]
        vms_grp = vms_file[vgrp]

        # Both sides of a matched pair must agree on the field layout,
        # since track datasets are copied through verbatim
        rot_fields = list(rot_grp.attrs['field_names'])
        vms_fields = list(vms_grp.attrs['field_names'])
        if rot_fields != vms_fields:
            raise ValueError(f"Field name mismatch between {rgrp} in {args.rot} "
                             f"and {vgrp} in {args.vms}: {rot_fields} != {vms_fields}")

        # y_init/z_init should agree between matched groups; warn (but do
        # not fail) if they do not, since this only affects provenance
        # metadata, not the track data itself
        if abs(rot_grp.attrs['y_init'] - vms_grp.attrs['y_init']) > 1e-6 or \
           abs(rot_grp.attrs['z_init'] - vms_grp.attrs['z_init']) > 1e-6:
            print(f"WARNING: y_init/z_init mismatch between {rgrp} "
                  f"(y={rot_grp.attrs['y_init']}, z={rot_grp.attrs['z_init']}) "
                  f"and {vgrp} (y={vms_grp.attrs['y_init']}, "
                  f"z={vms_grp.attrs['z_init']}); using VMS values")

        # Select masses: rotating grid below the break, VMS grid at/above it
        rot_masses = rot_grp['masses'][:]
        vms_masses = vms_grp['masses'][:]
        low_masses = np.sort(rot_masses[rot_masses < args.mass_break])
        high_masses = np.sort(vms_masses[vms_masses >= args.mass_break])
        combined_masses = np.concatenate([low_masses, high_masses])

        # Create the output group, using the same naming convention as
        # the VMS file (no v/vcrit component, since this is only ever
        # built from the non-rotating subset of the rotating grid)
        out_grp_name = f"feh_{feh:.6g}"
        out_grp = out_file.create_group(out_grp_name)

        # Copy the selected track datasets through verbatim, preserving
        # their compression
        for m in low_masses:
            name = track_dataset_name(m)
            rot_file.copy(rot_grp[name], out_grp, name=name)
        for m in high_masses:
            name = track_dataset_name(m)
            vms_file.copy(vms_grp[name], out_grp, name=name)

        # Write the group's metadata and mass list
        out_grp.attrs['feh'] = feh
        out_grp.attrs['y_init'] = vms_grp.attrs['y_init']
        out_grp.attrs['z_init'] = vms_grp.attrs['z_init']
        out_grp.attrs['ntime'] = out_grp[track_dataset_name(combined_masses[0])].shape[0]
        out_grp.attrs['nmass'] = len(combined_masses)
        out_grp.attrs['m_min'] = float(combined_masses.min())
        out_grp.attrs['m_max'] = float(combined_masses.max())
        out_grp.create_dataset('masses', data=combined_masses)
        out_grp.attrs['field_names'] = rot_fields

        if args.verbose:
            print(f"Wrote group {out_grp_name}: {len(low_masses)} tracks "
                  f"from {rgrp}, {len(high_masses)} tracks from {vgrp} "
                  f"({len(combined_masses)} total).")

rot_file.close()
vms_file.close()

# Read existing registry file if it exists, otherwise create a new one
if shutil.os.path.exists(args.registry):
    with open(args.registry, 'r') as f:
        registry = tomlkit.parse(f.read())
else:
    registry = { "name" : "Registry of track sets" }

# Add PARSEC_comp to list of track sets
if "track_sets" in registry.keys():
    if "PARSEC_comp" not in registry["track_sets"]:
        registry["track_sets"].append("PARSEC_comp")
else:
    registry["track_sets"] = [ "PARSEC_comp" ]

# Generate registry entry for PARSEC_comp
if "PARSEC_comp" in registry.keys():
    registry.pop("PARSEC_comp")
parsec_tab = tomlkit.table()
parsec_tab["file"] = shutil.os.path.basename(args.output)
parsec_tab["version"] = PARSEC_version
parsec_tab["references"] = references
parsec_tab["reference_urls"] = reference_urls
parsec_tab["download_urls"] = download_urls
with h5py.File(args.output, 'r') as h5file:
    fehs = sorted({ float(h5file[grp].attrs['feh']) for grp in h5file.keys() })
parsec_tab["Fe_H"] = fehs
registry["PARSEC_comp"] = parsec_tab

# Write registry back to file
with open(args.registry, 'w') as fp:
    fp.write(tomlkit.dumps(registry))

print(f"Wrote {len(matches)} groups to {args.output} and updated {args.registry}.")
