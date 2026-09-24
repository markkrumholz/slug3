"""
This is a script to import the legacy starburst99 stellar tracks that
slug2 shipped (in slug2's lib/tracks/sb99 directory), and write them
to HDF5 files in slug's track format, along with matching entries in
the tracks registry.

The track families imported are:

* ``modc*`` -- "old" (pre-SYCLIST) Geneva models with standard mass loss
* ``mode*`` -- "old" Geneva models with twice the standard mass loss
* ``modp*`` -- "old" (pre-PARSEC) Padova models with TP-AGB stars
* ``mods*`` -- "old" Padova models without TP-AGB stars

Each file ``mod{family}{digits}.dat`` holds one metallicity, Z =
0.{digits} (e.g. modc020.dat -> Z = 0.020, modp0004.dat -> Z = 0.0004).
All the metallicities of a family are written to a single HDF5 file
``sb99_mod{family}.h5``, one group per metallicity, with a single
registry entry ``sb99_mod{family}``. [Fe/H] is computed as log10(Z /
0.02), since Z = 0.02 is solar for all of these model sets. None of
the models include rotation, so v/vcrit = 0 for all of them.

File format: a one-line description, a blank line, a line giving the
number of masses and number of time points per track, then one block
per mass (from most to least massive). Each block starts with a line
giving the initial mass followed by an optional track type code, then
a blank line, then one row per time point. Rows are Fortran
fixed-width records whose fields can run together with no separating
whitespace (e.g. "0.9000-0.027"), so they are parsed by column
position rather than by splitting on whitespace. The fields are: row
index, time (yr), present-day mass (Msun), log L (Lsun), log Teff (K),
and surface mass fractions of H, He, C, N, and O, followed by
additional fields that depend on the track type code:

* ``WR`` -- a second log Teff (the hydrostatic, rather than
  wind-corrected, value; not used), then log mass loss rate
  (Msun/yr); a mass loss rate written as exactly ``0.000`` means a
  mass loss rate of zero, not 1 Msun/yr
* ``ML`` -- log mass loss rate (Msun/yr)
* no code -- no further fields; mass loss rate is zero

The column positions and the track type code conventions follow those
used by slug2's own reader for these files.

slug requires every metallicity in a track set to share one mass grid,
but the Geneva families' mass grids differ slightly between
metallicities: some carry an extra m = 1.701 track (a variant of the
1.70 Msun track that undergoes the He flash), and Z = 0.008 has a 10
Msun track in place of the 9 Msun track every other metallicity has.
To put every metallicity on one common grid, this script:

* drops the m = 1.701 tracks entirely; and
* synthesizes any track missing from a metallicity by interpolating,
  row by row, between the nearest bracketing masses present at that
  same metallicity (see synthesize_track). This is possible because
  every track in a starburst99 file has the same number of rows, with
  row i of every track marking the same evolutionary phase.
  Synthesized masses are recorded in each group's
  ``synthesized_masses`` attribute.

Finally, any row whose age does not strictly exceed that of every
previous row is dropped, since slug requires strictly increasing
ages. Most such rows are padding at the ends of low-mass tracks, which
repeat the final age for all remaining rows; the rest are fast phases
(e.g. first dredge-up in the Padova models) whose ages differ by less
than the 7 significant figures to which ages are written.

:copyright: Copyright (c) 2026 Mark Krumholz
"""

import argparse
import os
import re

import h5py
import numpy as np
import tomlkit

# Constants / metadata
SB99_ZSUN = 0.02   # Solar metallicity for the legacy starburst99 track sets
SB99_FAMILIES = ("c", "e", "p", "s")
SB99_REFERENCES = [
    "Leitherer, C., Schaerer, D., Goldader, J. D., et al. 1999, ApJS, 123, 3",
    "Vazquez, G. A., Leitherer, C. 2005, ApJ, 621, 695",
]
SB99_REFERENCE_URLS = [
    "https://ui.adsabs.harvard.edu/abs/1999ApJS..123....3L",
    "https://ui.adsabs.harvard.edu/abs/2005ApJ...621..695V",
]

# Fixed-width column boundaries for each track type, as used by slug2
# (src/tracks/slug_tracks_sb99.cpp); field i occupies characters
# [breaks[i], breaks[i+1]) of each data row
COL_BREAKS = {
    "WR": [0, 2, 16, 25, 31, 37, 46, 55, 64, 73, 82, 89, 96],
    "ML": [0, 2, 16, 25, 31, 37, 46, 55, 64, 73, 82, 89],
    "":   [0, 2, 16, 25, 31, 37, 46, 55, 64, 73, 82],
}
# Index of the log mass loss rate field for each track type, or None
# if that track type has no mass loss rate field
MDOT_FIELD = {"WR": 11, "ML": 10, "": None}

# Order of fields in the output HDF5 datasets
FIELDS = ['age', 'mass', 'mdot', 'log_L', 'log_Teff',
          'h_surf', 'he_surf', 'c_surf', 'n_surf', 'o_surf']

FILE_PATTERN = re.compile(r'^mod([' + ''.join(SB99_FAMILIES) + r'])(\d{3,4})\.dat$')


def z_from_filename(filename):
    """Return the metallicity Z encoded in a starburst99 track filename.

    Parameters
    ----------
    filename : str
        Base name of the track file, e.g. ``modc020.dat``

    Returns
    -------
    float
        The metallicity Z, e.g. 0.02 for ``modc020.dat``

    Raises
    ------
    ValueError
        If the filename does not follow the ``mod{family}{digits}.dat``
        convention
    """
    m = FILE_PATTERN.match(filename)
    if not m:
        raise ValueError(f"Cannot parse starburst99 filename: {filename}")
    return float("0." + m.group(2))


def parse_row(line, track_type):
    """Parse one fixed-width data row of a starburst99 track file.

    Parameters
    ----------
    line : str
        The data row
    track_type : str
        The track type code of the block the row belongs to
        (``"WR"``, ``"ML"``, or ``""``)

    Returns
    -------
    dict
        Mapping from each name in FIELDS to its value in this row, with
        the mass loss rate converted from log to linear
    """
    brk = COL_BREAKS[track_type]

    def field(i):
        return line[brk[i]:brk[i + 1]].strip()

    row = {
        'age':      float(field(1)),
        'mass':     float(field(2)),
        'log_L':    float(field(3)),
        'log_Teff': float(field(4)),
        'h_surf':   float(field(5)),
        'he_surf':  float(field(6)),
        'c_surf':   float(field(7)),
        'n_surf':   float(field(8)),
        'o_surf':   float(field(9)),
    }
    mdot_idx = MDOT_FIELD[track_type]
    if mdot_idx is None:
        row['mdot'] = 0.0
    else:
        s = field(mdot_idx)
        # For WR tracks, a value of exactly 0.000 is a sentinel for zero
        # mass loss, not log mdot = 0
        if track_type == "WR" and s == "0.000":
            row['mdot'] = 0.0
        else:
            row['mdot'] = 10.0**float(s)
    return row


def read_sb99_file(path):
    """Read a starburst99 track file.

    Parameters
    ----------
    path : str
        Path to the track file

    Returns
    -------
    description : str
        The descriptive text from the first line of the file
    tracks : list of (float, numpy.ndarray)
        One entry per track, sorted by increasing initial mass; each is
        a pair of the initial mass and an array of shape (ntime,
        len(FIELDS)) holding the track data, with columns in the order
        given by FIELDS

    Raises
    ------
    ValueError
        If the file is not in the expected format
    """
    with open(path, 'r') as fp:
        lines = fp.read().splitlines()

    description = lines[0].strip()
    header = lines[2].split()
    ntrack, ntime = int(header[0]), int(header[1])

    tracks = []
    pos = 3
    for _ in range(ntrack):
        # Skip blank lines, then read the mass / track type line
        while not lines[pos].strip():
            pos += 1
        tokens = lines[pos].split()
        m_init = float(tokens[0])
        track_type = tokens[1] if len(tokens) > 1 else ""
        if track_type not in COL_BREAKS:
            raise ValueError(f"{path}: unknown track type '{track_type}' "
                             f"for mass {m_init}")
        pos += 1

        # Skip blank lines, then read ntime data rows
        while not lines[pos].strip():
            pos += 1
        rows = [parse_row(l, track_type) for l in lines[pos:pos + ntime]]
        if len(rows) != ntime:
            raise ValueError(f"{path}: track for mass {m_init} has "
                             f"{len(rows)} rows, expected {ntime}")
        pos += ntime

        arr = np.array([[r[f] for f in FIELDS] for r in rows])
        tracks.append((m_init, arr))

    tracks.sort(key=lambda x: x[0])
    return description, tracks


def prune_repeated_ages(arr):
    """Drop rows whose age does not strictly exceed all previous ages.

    Parameters
    ----------
    arr : numpy.ndarray
        Track data of shape (ntime, nfield), with age in column 0

    Returns
    -------
    numpy.ndarray
        The rows of arr whose age strictly exceeds the running maximum
        of all previous rows' ages; row 0 is always kept
    """
    age = arr[:, 0]
    running_max_before = np.concatenate(([-np.inf],
                                         np.maximum.accumulate(age[:-1])))
    return arr[age > running_max_before]



# Initial masses dropped from every track set; see the module docstring
DROP_MASSES = (1.701,)

# Surface abundance fields, which must never all be zero in a valid row
ABUNDANCE_FIELDS = ('h_surf', 'he_surf', 'c_surf', 'n_surf', 'o_surf')

# Fields interpolated in log space when synthesizing a track; all
# others are interpolated linearly
LOG_FIELDS = ('age', 'mass')


def synthesize_track(m, m_lo, arr_lo, m_hi, arr_hi):
    """Synthesize a track by interpolating between two bracketing tracks.

    Parameters
    ----------
    m : float
        Initial mass of the track to synthesize
    m_lo, m_hi : float
        Initial masses of the bracketing tracks, with m_lo < m < m_hi
    arr_lo, arr_hi : numpy.ndarray
        Data for the bracketing tracks, each of shape (ntime,
        len(FIELDS)), with columns in the order given by FIELDS

    Returns
    -------
    numpy.ndarray
        The synthesized track, of the same shape as arr_lo and arr_hi

    Notes
    -----
    Row i of the result is interpolated from row i of each bracketing
    track, with interpolation weights linear in log initial mass. Age
    and present-day mass are interpolated in log space, and mass loss
    rate in log space too where it is positive in both bracketing rows
    (and linearly otherwise); every other field (log L, log Teff, and
    the surface abundances) is interpolated linearly.
    """
    if arr_lo.shape != arr_hi.shape:
        raise ValueError("bracketing tracks must have the same shape")
    w = np.log(m / m_lo) / np.log(m_hi / m_lo)
    out = (1.0 - w) * arr_lo + w * arr_hi
    for f in LOG_FIELDS:
        c = FIELDS.index(f)
        out[:, c] = np.exp((1.0 - w) * np.log(arr_lo[:, c]) +
                           w * np.log(arr_hi[:, c]))
    c = FIELDS.index('mdot')
    pos = (arr_lo[:, c] > 0.0) & (arr_hi[:, c] > 0.0)
    out[pos, c] = np.exp((1.0 - w) * np.log(arr_lo[pos, c]) +
                         w * np.log(arr_hi[pos, c]))
    return out


def rectify_mass_grids(tracks_by_z, verbose=False):
    """Put every metallicity of a family on one common mass grid.

    Parameters
    ----------
    tracks_by_z : dict
        Mapping from metallicity Z to that metallicity's tracks, as a
        list of (initial mass, track array) pairs as returned by
        read_sb99_file
    verbose : bool
        If True, report every track dropped or synthesized

    Returns
    -------
    masses : list of float
        The common mass grid, sorted in increasing order
    rectified : dict
        Mapping from metallicity Z to a pair of (list of track arrays,
        one per mass in masses; list of synthesized masses)

    Raises
    ------
    ValueError
        If a missing mass is not bracketed by masses present at its
        metallicity, so it cannot be synthesized
    """
    def keep(m):
        return not any(abs(m - d) < 1.0e-6 for d in DROP_MASSES)

    kept = {z: {round(m, 3): arr for m, arr in tracks if keep(m)}
            for z, tracks in tracks_by_z.items()}
    if verbose:
        for z, tracks in tracks_by_z.items():
            for m, _ in tracks:
                if not keep(m):
                    print(f"  Z = {z}: dropping m = {m:.3f}")
    masses = sorted(set().union(*kept.values()))

    rectified = {}
    for z, present in kept.items():
        have = sorted(present)
        arrs, synth = [], []
        for m in masses:
            if m in present:
                arrs.append(present[m])
                continue
            lo = [h for h in have if h < m]
            hi = [h for h in have if h > m]
            if not lo or not hi:
                raise ValueError(f"Z = {z}: cannot synthesize m = {m}, "
                                 "which is not bracketed by available masses")
            m_lo, m_hi = lo[-1], hi[0]
            if verbose:
                print(f"  Z = {z}: synthesizing m = {m:.3f} from "
                      f"m = {m_lo:.3f} and {m_hi:.3f}")
            arrs.append(synthesize_track(m, m_lo, present[m_lo],
                                         m_hi, present[m_hi]))
            synth.append(m)
        rectified[z] = (arrs, synth)
    return masses, rectified


def check_abundances(arr, label):
    """Check that no row of a track has all-zero surface abundances.

    Parameters
    ----------
    arr : numpy.ndarray
        Track data of shape (ntime, len(FIELDS)), with columns in the
        order given by FIELDS
    label : str
        Description of the track, used in the error message

    Raises
    ------
    ValueError
        If any row's surface abundances are all exactly zero, which
        indicates missing data in the source file (as in the m = 1.701
        track of modc020.dat, which is dropped for other reasons)
    """
    cols = [FIELDS.index(f) for f in ABUNDANCE_FIELDS]
    bad = np.all(arr[:, cols] == 0.0, axis=1)
    if bad.any():
        raise ValueError(f"{label}: {int(bad.sum())} rows have all-zero "
                         "surface abundances")


def write_family_h5(out_path, descriptions, masses, rectified,
                    verbose=False):
    """Write one family of starburst99 tracks to an HDF5 file in slug format.

    Parameters
    ----------
    out_path : str
        Path to the output HDF5 file; overwritten if it exists
    descriptions : dict
        Mapping from metallicity Z to the descriptive text of that
        metallicity's source file, stored as a group attribute
    masses : list of float
        Common mass grid, as returned by rectify_mass_grids
    rectified : dict
        Rectified tracks, as returned by rectify_mass_grids
    verbose : bool
        If True, report any tracks from which rows were pruned

    Returns
    -------
    list of float
        The [Fe/H] values of the groups written, in increasing order
    """
    fehs = []
    with h5py.File(out_path, 'w') as h5file:
        h5file.attrs['references']     = SB99_REFERENCES
        h5file.attrs['reference_urls'] = SB99_REFERENCE_URLS

        for z_val in sorted(rectified):
            arrs, synth = rectified[z_val]
            feh_val = float(np.log10(z_val / SB99_ZSUN))
            fehs.append(feh_val)

            pruned = []
            for m, arr in zip(masses, arrs):
                check_abundances(arr, f"Z = {z_val}, m = {m:.3f}")
                p = prune_repeated_ages(arr)
                if verbose and p.shape[0] != arr.shape[0]:
                    print(f"  Z = {z_val}, m = {m:.3f}: pruned "
                          f"{arr.shape[0] - p.shape[0]} rows with "
                          "non-increasing age")
                pruned.append(p)

            grp = h5file.create_group(
                f"feh_{feh_val:.2f}_afe_0.0_vvcrit_0.00")
            grp.attrs['feh']                = feh_val
            grp.attrs['afe']                = 0.0
            grp.attrs['vvcrit']             = 0.0
            grp.attrs['y_init']             = float(
                pruned[0][0, FIELDS.index('he_surf')])
            grp.attrs['z_init']             = z_val
            grp.attrs['ntime']              = max(p.shape[0] for p in pruned)
            grp.attrs['nmass']              = len(masses)
            grp.attrs['m_min']              = min(masses)
            grp.attrs['m_max']              = max(masses)
            grp.attrs['field_names']        = FIELDS
            grp.attrs['description']        = descriptions[z_val]
            grp.attrs['synthesized_masses'] = np.array(synth, dtype=float)

            grp.create_dataset('masses', data=masses)
            for m, p in zip(masses, pruned):
                grp.create_dataset(f"track_m{m:.3f}", data=p,
                                   compression="gzip")

    return sorted(fehs)


def main():
    """Command-line entry point; see the module docstring."""
    parser = argparse.ArgumentParser(
        description="Import legacy starburst99 stellar tracks from slug2")
    parser.add_argument("srcdir",
                        help="Directory containing the starburst99 "
                             "mod*.dat track files (slug2's lib/tracks/sb99)")
    parser.add_argument("--outdir",
                        default=os.path.join("..", "..", "tracks"),
                        help="Directory to which to write output HDF5 files")
    parser.add_argument("--registry",
                        default=os.path.join("..", "..", "tracks", "tracks.toml"),
                        help="Registry TOML file path")
    parser.add_argument("--families", nargs="+", default=list(SB99_FAMILIES),
                        choices=SB99_FAMILIES,
                        help="Model families to import (c, e, p, s); "
                             "default is all")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing output HDF5 files")
    parser.add_argument("--verbose", action="store_true",
                        help="Print verbose output")
    args = parser.parse_args()

    # Find the files to import, grouped by family
    files_by_family = {}
    for f in sorted(os.listdir(args.srcdir)):
        m = FILE_PATTERN.match(f)
        if m and m.group(1) in args.families:
            files_by_family.setdefault(m.group(1), []).append(f)
    if not files_by_family:
        raise RuntimeError(f"No starburst99 track files found in {args.srcdir}")

    # Read the registry
    if os.path.exists(args.registry):
        with open(args.registry, 'r') as fp:
            registry = tomlkit.parse(fp.read())
    else:
        registry = tomlkit.document()
        registry["name"] = "Registry of track sets"
    if "track_sets" not in registry:
        registry["track_sets"] = []

    # Import each family
    for family, fnames in sorted(files_by_family.items()):
        set_name = f"sb99_mod{family}"
        out_path = os.path.join(args.outdir, f"{set_name}.h5")

        if os.path.exists(out_path) and not args.overwrite:
            if args.verbose:
                print(f"{out_path} exists; skipping (use --overwrite "
                      "to regenerate)")
            with h5py.File(out_path, 'r') as h5file:
                fehs = sorted(float(g.attrs['feh'])
                              for g in h5file.values())
        else:
            if args.verbose:
                print(f"Importing {', '.join(fnames)} -> {out_path}")
            descriptions, tracks_by_z = {}, {}
            for fname in fnames:
                z_val = z_from_filename(fname)
                descriptions[z_val], tracks_by_z[z_val] = read_sb99_file(
                    os.path.join(args.srcdir, fname))
            masses, rectified = rectify_mass_grids(tracks_by_z,
                                                   verbose=args.verbose)
            fehs = write_family_h5(out_path, descriptions, masses,
                                   rectified, verbose=args.verbose)

        # Build the registry entry
        tab = tomlkit.table()
        tab["file"]           = os.path.basename(out_path)
        tab["references"]     = SB99_REFERENCES
        tab["reference_urls"] = SB99_REFERENCE_URLS
        tab["Fe_H"]           = fehs
        tab["v_vcrit"]        = [0.0]
        if set_name in registry:
            registry.pop(set_name)
        registry[set_name] = tab
        if set_name not in registry["track_sets"]:
            registry["track_sets"].append(set_name)

    with open(args.registry, 'w') as fp:
        fp.write(tomlkit.dumps(registry))
    if args.verbose:
        print(f"Updated {args.registry}")


if __name__ == "__main__":
    main()
