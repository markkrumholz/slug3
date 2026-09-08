"""
This is a script to fetch Geneva/SYCLIST stellar tracks from the public
SYCLIST GitHub repository (https://github.com/GESEG/SYCLIST), extract
the columns needed by slug, and write them to an HDF5 file in slug's
track format.

The SYCLIST large_grids directory contains one subdirectory per
metallicity, named Z{NNN} where NNN encodes the absolute metallicity
Z by prepending "0." (e.g. Z014 → Z = 0.014).  Within each directory,
files are named M{int}p{frac}Z{code}V{vv}.dat, where {int}p{frac} is
the initial mass (e.g. M001p25 → 1.25 Msun) and V{vv} encodes the
rotation rate v/vcrit (V00 → 0.0, V40 → 0.4).

:copyright: Copyright (c) 2026 Mark Krumholz
"""

import argparse
import h5py
import json
import numpy as np
import re
import shutil
import tomlkit
import urllib3

# Constants / metadata
GENEVA_GITHUB_API = (
    "https://api.github.com/repos/GESEG/SYCLIST/contents/tables/large_grids"
)
GENEVA_ZSUN = 0.014   # Solar metallicity for the Geneva/SYCLIST track set
GENEVA_references = [
    "Ekström, S., Georgy, C., Eggenberger, P., et al. 2012, A&A, 537, A146",
    "Georgy, C., Ekström, S., Meyney, G., et al. 2012, A&A, 542, A29",
    "Georgy, C., Ekström, S., Granada, A., et al. 2013, A&A, 553, A24",
    "Georgy, C., Ekström, S., Meyney, G., et al. 2013, A&A, 558, A103",
    "Georgy, C., Granada, A., Ekström, S., et al. 2014, A&A, 566, A21",
    "Eggenberger, P., Ekström, S., Georgy, C., et al. 2021, A&A, 652, A137",
    "Groh, J., Ekström, S., Georgy, C., et al. 2019, A&A, 627, A24",
]
GENEVA_reference_URLs = [
    "https://ui.adsabs.harvard.edu/abs/2012A%26A...537A.146E/abstract",
    "https://ui.adsabs.harvard.edu/abs/2012A%26A...542A..29G/abstract",
    "https://ui.adsabs.harvard.edu/abs/2013A%26A...553A..24G/abstract",
    "https://ui.adsabs.harvard.edu/abs/2013A%26A...558A.103G/abstract",
    "https://ui.adsabs.harvard.edu/abs/2014A%26A...566A..21G/abstract",
    "https://ui.adsabs.harvard.edu/abs/2021A%26A...652A.137E/abstract",
    "https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..24G/abstract",
]

# Parse command line arguments
parser = argparse.ArgumentParser(
    description="Fetch Geneva/SYCLIST stellar tracks from GitHub")
parser.add_argument("--url", default=GENEVA_GITHUB_API,
                    help="GitHub API URL for the SYCLIST large_grids directory")
parser.add_argument("--output",
                    default=shutil.os.path.join("..", "..", "tracks", "geneva.h5"),
                    help="Output HDF5 file path")
parser.add_argument("--registry",
                    default=shutil.os.path.join("..", "..", "tracks", "tracks.toml"),
                    help="Registry TOML file path")
parser.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing tracks in output file")
parser.add_argument("--feh", type=float, nargs="+", default=[],
                    help="[Fe/H] values to fetch; if unspecified, fetch all")
parser.add_argument("--vvcrit", type=float, nargs="+", default=[],
                    help="v/vcrit values to fetch (0.0 or 0.4); "
                         "if unspecified, fetch all")
parser.add_argument("--verbose", action="store_true",
                    help="Print verbose output")
args = parser.parse_args()

# urllib3 pool manager; GitHub API requests use the JSON accept header.
# Timeouts: 10s to establish a connection, 120s to receive a response
# (track .dat files run to a few hundred KB, so 120s is generous but safe
# on a slow link; without a timeout the script can hang indefinitely on a
# stalled upstream connection).
http = urllib3.PoolManager(
    timeout=urllib3.Timeout(connect=10, read=120),
    headers={"Accept": "application/vnd.github.v3+json",
             "User-Agent": "slug3-fetch-geneva"})


def github_api_get(url):
    """Fetch and return parsed JSON from the GitHub REST API."""
    response = http.request('GET', url)
    if response.status != 200:
        raise RuntimeError(
            f"GitHub API request failed ({url}): HTTP {response.status}")
    return json.loads(response.data.decode('utf-8'))


def z_from_dir(dirname):
    """Parse Z metallicity from a SYCLIST directory name (e.g. Z014 → 0.014)."""
    return float("0." + dirname[1:])


def parse_filename(filename):
    """Return (initial_mass, vvcrit) from a SYCLIST track filename.

    Filename convention: M{int}p{frac}Z{z_code}V{vv}.dat
    e.g. M001p25Z14V40.dat → mass=1.25, vvcrit=0.4
    """
    m = re.match(r'M(\d+)p(\d+)Z\w+V(\d+)\.dat$', filename)
    if not m:
        raise ValueError(f"Cannot parse SYCLIST filename: {filename}")
    frac_str = m.group(2)
    mass = int(m.group(1)) + int(frac_str) / 10**len(frac_str)
    vvcrit = int(m.group(3)) / 100.0   # "00" → 0.0, "40" → 0.4
    return mass, vvcrit


# Discover metallicity subdirectories
top_listing = github_api_get(args.url)
z_dirs = sorted(item['name'] for item in top_listing if item['type'] == 'dir')
if args.verbose:
    print(f"Found {len(z_dirs)} metallicity directories: {z_dirs}")

# If the output file already exists and --overwrite is not set, record
# which (feh, vvcrit) combinations are already present so we can skip them
existing = set()
if shutil.os.path.exists(args.output) and not args.overwrite:
    with h5py.File(args.output, 'r') as h5file:
        for grp in h5file.keys():
            existing.add((round(float(h5file[grp].attrs['feh']),   4),
                          round(float(h5file[grp].attrs['vvcrit']), 2)))

# Create the HDF5 output file if it does not already exist
if not shutil.os.path.exists(args.output):
    with h5py.File(args.output, 'w') as h5file:
        h5file.attrs['references']     = GENEVA_references
        h5file.attrs['reference_urls'] = GENEVA_reference_URLs

# Process each metallicity directory
for zdir in z_dirs:
    z_val   = z_from_dir(zdir)
    feh_val = float(np.log10(z_val / GENEVA_ZSUN))

    # Apply --feh filter
    if args.feh and not any(abs(feh_val - f) < 1e-3 for f in args.feh):
        continue

    # List the .dat files in this subdirectory
    dir_listing = github_api_get(f"{args.url}/{zdir}")
    dat_items   = sorted(
        (item for item in dir_listing if item['name'].endswith('.dat')),
        key=lambda x: x['name'])
    if args.verbose:
        print(f"Directory {zdir} (Z={z_val}, [Fe/H]={feh_val:.3f}): "
              f"{len(dat_items)} files")

    # Group files by vvcrit value
    files_by_vvcrit = {}
    for item in dat_items:
        try:
            mass, vvcrit = parse_filename(item['name'])
        except ValueError as err:
            print(f"Warning: skipping {item['name']}: {err}")
            continue
        if args.vvcrit and not any(abs(vvcrit - v) < 1e-3 for v in args.vvcrit):
            continue
        files_by_vvcrit.setdefault(vvcrit, []).append(
            (mass, item['download_url']))

    # Download and pack each (feh, vvcrit) group
    for vvcrit, mass_url_pairs in sorted(files_by_vvcrit.items()):
        feh_key    = round(feh_val, 4)
        vvcrit_key = round(vvcrit, 2)
        if (feh_key, vvcrit_key) in existing:
            if args.verbose:
                print(f"  Skipping feh={feh_val:.3f} vvcrit={vvcrit:.2f}: "
                      "already in file")
            continue

        if args.verbose:
            print(f"  Fetching feh={feh_val:.3f} vvcrit={vvcrit:.2f} "
                  f"({len(mass_url_pairs)} tracks)...")

        track_data     = []
        track_metadata = []

        for mass, download_url in sorted(mass_url_pairs):
            # Fetch the individual track file
            resp = http.request('GET', download_url,
                                headers={"Accept": "*/*",
                                         "User-Agent": "slug3-fetch-geneva"})
            if resp.status != 200:
                raise RuntimeError(
                    f"Failed to download {download_url}: HTTP {resp.status}")

            lines = resp.data.decode('utf-8').splitlines()

            # Line 0: column names; line 1: units (skip); rest: data
            cols       = lines[0].split()
            data_lines = [l for l in lines[2:] if l.strip()]
            if not data_lines:
                print(f"  Warning: no data rows in {download_url}; skipping")
                continue

            # Some Fortran output drops the 'E' before a 3-digit exponent
            # (e.g. "2.83-289" instead of "2.83E-289"); fix before parsing
            data_lines = [re.sub(r'(\d)([+-]\d{3})(?!\d)', r'\1E\2', l)
                          for l in data_lines]

            fdat = np.loadtxt(data_lines)
            if fdat.ndim == 1:       # single-row track
                fdat = fdat.reshape(1, -1)
            n_pts = fdat.shape[0]

            # Extract the columns slug needs
            t        = fdat[:, cols.index('time')]
            m        = fdat[:, cols.index('mass')]
            log_L    = fdat[:, cols.index('lg(L)')]
            log_Teff = fdat[:, cols.index('lg(Teff)')]   # first occurrence
            h_surf   = fdat[:, cols.index('1H_surf')]
            he_surf  = fdat[:, cols.index('4He_surf')]
            c_surf   = (fdat[:, cols.index('12C_surf')] +
                        fdat[:, cols.index('13C_surf')])
            n_surf   = fdat[:, cols.index('14N_surf')]
            o_surf   = (fdat[:, cols.index('16O_surf')] +
                        fdat[:, cols.index('17O_surf')] +
                        fdat[:, cols.index('18O_surf')])

            # lg(Md) == 0.0 is a sentinel meaning no mass loss is recorded;
            # for all other values it is log10(mdot / (Msun yr^-1))
            lg_mdot = fdat[:, cols.index('lg(Md)')]
            mdot    = np.where(lg_mdot == 0.0, 0.0, 10.0**lg_mdot)

            track_metadata.append({
                'm_init':  mass,
                'y_init':  float(he_surf[0]),
                'z_init':  z_val,
                'v_vcrit': vvcrit,
                'fe_h':    feh_val,
                'n_pts':   n_pts,
            })
            track_data.append({
                'age':      t,
                'mass':     m,
                'mdot':     mdot,
                'log_L':    log_L,
                'log_Teff': log_Teff,
                'h_surf':   h_surf,
                'he_surf':  he_surf,
                'c_surf':   c_surf,
                'n_surf':   n_surf,
                'o_surf':   o_surf,
            })

        if not track_data:
            continue

        # Write this (feh, vvcrit) group to the HDF5 file
        flds = ['age', 'mass', 'mdot', 'log_L', 'log_Teff',
                'h_surf', 'he_surf', 'c_surf', 'n_surf', 'o_surf']
        with h5py.File(args.output, 'a') as h5file:
            grp_name = f"feh_{feh_val:.2f}_afe_0.0_vvcrit_{vvcrit:.2f}"
            if grp_name in h5file:
                del h5file[grp_name]
            grp = h5file.create_group(grp_name)

            grp.attrs['feh']         = feh_val
            grp.attrs['afe']         = 0.0
            grp.attrs['vvcrit']      = vvcrit
            grp.attrs['y_init']      = track_metadata[0]['y_init']
            grp.attrs['z_init']      = z_val
            grp.attrs['ntime']       = max(md['n_pts'] for md in track_metadata)
            grp.attrs['nmass']       = len(track_data)
            grp.attrs['m_min']       = min(md['m_init'] for md in track_metadata)
            grp.attrs['m_max']       = max(md['m_init'] for md in track_metadata)
            grp.attrs['field_names'] = flds

            grp.create_dataset('masses',
                               data=[md['m_init'] for md in track_metadata])

            for md, td in zip(track_metadata, track_data):
                arr = np.column_stack([td[f] for f in flds])
                grp.create_dataset(f"track_m{md['m_init']:.3f}",
                                   data=arr, compression="gzip")

        if args.verbose:
            print(f"    Wrote {len(track_data)} tracks → {grp_name}")
        existing.add((feh_key, vvcrit_key))

# Update the tracks.toml registry
if shutil.os.path.exists(args.registry):
    with open(args.registry, 'r') as fp:
        registry = tomlkit.parse(fp.read())
else:
    registry = {"name": "Registry of track sets"}

if "track_sets" in registry:
    if "Geneva" not in registry["track_sets"]:
        registry["track_sets"].append("Geneva")
else:
    registry["track_sets"] = ["Geneva"]

if "Geneva" in registry:
    registry.pop("Geneva")
geneva_tab = tomlkit.table()
geneva_tab["file"]           = shutil.os.path.basename(args.output)
geneva_tab["references"]     = GENEVA_references
geneva_tab["reference_urls"] = GENEVA_reference_URLs

for qty, attr in zip(["Fe_H", "v_vcrit"], ["feh", "vvcrit"]):
    seen = []
    with h5py.File(args.output, 'r') as h5file:
        for grp in h5file.keys():
            v = float(h5file[grp].attrs[attr])
            if not any(abs(v - s) < 1e-6 for s in seen):
                seen.append(v)
    seen.sort()
    geneva_tab[qty] = seen

registry["Geneva"] = geneva_tab

with open(args.registry, 'w') as fp:
    fp.write(tomlkit.dumps(registry))

if args.verbose:
    print("Registry updated.")
