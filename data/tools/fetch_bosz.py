"""
This is a script to fetch the BOSZ spectra (https://archive.stsci.edu/hlsp/bosz)
from the offical STScI website, extract data from them, and write them into a
gzip'ed HDF5 file that slug can read.
"""

# Imports
import argparse
from collections.abc import Iterator
import h5py
import numpy as np
import re
import shutil
import tomlkit
import urllib3

# Magic strings
BOSZ_version = "2024"
BOSZ_URL = "https://archive.stsci.edu/hlsps/bosz/bosz2024/"
BOSZ_references = ["Mészáros, Sz., Allende Prieto, C., Edvardsson, B., et al. 2012, AJ, 144, 120",
                   "Bohlin, R., Mészáros, Sz., Fleming, S., et al. 2017, AJ, 153, 234",
                   "Mészáros, Sz., Bohlin, R., Allende Prieto, C., et al. 2024, A&A, 688, A197"]
BOSZ_refernce_URLS = ["https://ui.adsabs.harvard.edu/abs/2012AJ....144..120M/abstract",
                      "https://ui.adsabs.harvard.edu/abs/2017AJ....153..234B/abstract",
                      "https://ui.adsabs.harvard.edu/abs/2024A%26A...688A.197M/abstract",
                      "https://dx.doi.org/10.17909/T95G68"]

# Parse command line arguments
parser = argparse.ArgumentParser(description="Fetch BOSZ spectral library")
parser.add_argument("--version", default=BOSZ_version, 
                    help="BOSZ version to fetch")
parser.add_argument("--url", default=BOSZ_URL, 
                    help="URL of the BOSZ data")
parser.add_argument("--output", 
                    default=shutil.os.path.join("..", "spectra", "bosz.h5"), 
                    help="Output file for the HDF5 spectra")
parser.add_argument("--registry", 
                    default=shutil.os.path.join("..", "spectra", "spectra.toml"), 
                    help="Output file for the registry")
parser.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing output file")
parser.add_argument("--feh", type=float, nargs="+", default=[],
                    help="List of [Fe/H] values to fetch; if unspecified, fetch all")
parser.add_argument("--afe", type=float, nargs="+", default=[],
                    help="List of [alpha/Fe] values to fetch; if unspecified, fetch all")
parser.add_argument("--cfe", type=float, nargs="+", default=[],
                    help="List of [C/Fe] values to fetch; if unspecified, fetch all")
parser.add_argument("--r", type=int, nargs="+", default=[],
                    help="List of r values to fetch; if unspecified, fetch all")
parser.add_argument("--micro", type=int, nargs="+", default=[],
                    help="List of microturbulent vleocity values to fetch; if unspecified, fetch all")
parser.add_argument("--verbose", action="store_true",
                    help="Print verbose output")
args = parser.parse_args()

# Regex to parse a BOSZ spectrum filename into its component fields; see
# https://archive.stsci.edu/hlsp/bosz for the naming convention, e.g.
# bosz2024_mp_t5000_g+5.0_m+0.00_a+0.00_c+0.00_v0_r500_resam.txt.gz
name_re = re.compile(
    r'bosz' + re.escape(args.version) +
    r'_(?P<atmos>[a-z]+)_t(?P<teff>[0-9]+)_g(?P<logg>[+-][0-9.]+)'
    r'_m(?P<feh>[+-][0-9.]+)_a(?P<afe>[+-][0-9.]+)_c(?P<cfe>[+-][0-9.]+)'
    r'_v(?P<micro>[0-9]+)_r(?P<r>[0-9]+)_(?P<prod>[a-z]+)\.txt\.gz')

# BOSZ archive coverage (https://archive.stsci.edu/hlsp/bosz). Rather
# than querying the server for directory listings -- which is painfully
# slow because the archive keeps thousands of files in a single flat
# directory -- generate candidate paths algorithmically from the
# documented grid. Any combination the archive doesn't actually carry
# will return a 404, which the download loop skips gracefully.

# All supported values for the filterable parameters
_ALL_R_VALS    = [500, 1000, 2000, 5000, 10000, 20000, 50000]
_ALL_FEH_VALS  = [round(-2.50 + 0.25 * i, 2) for i in range(14)]  # -2.50 to +0.75
_ALL_AFE_VALS  = [round(-0.25 + 0.25 * i, 2) for i in range(4)]   # -0.25 to +0.50
_ALL_CFE_VALS  = [round(-0.75 + 0.25 * i, 2) for i in range(6)]   # -0.75 to +0.50
_ALL_MICRO_VALS = [0, 1, 2, 4]

# MARCS (Teff, log g) sub-grids; each entry is (teff_values, logg_values)
_MARCS_RANGES = [
    (list(range(2800, 4001, 100)), [round(-0.5 + 0.5*k, 1) for k in range(13)]),
    (list(range(4250, 4751, 250)), [round(-0.5 + 0.5*k, 1) for k in range(12)]),
    (list(range(5000, 5751, 250)), [round( 0.0 + 0.5*k, 1) for k in range(12)]),
    (list(range(6000, 7001, 250)), [round( 1.0 + 0.5*k, 1) for k in range(10)]),
    (list(range(7250, 8001, 250)), [round( 2.0 + 0.5*k, 1) for k in range(9)]),
]
# ATLAS9 (Teff, log g) sub-grids
_ATLAS9_RANGES = [
    (list(range( 7500, 12001, 250)), [round(2.0 + 0.5*k, 1) for k in range(7)]),
    (list(range(12500, 16001, 500)), [round(3.0 + 0.5*k, 1) for k in range(5)]),
]

def _teff_logg_candidates() -> Iterator[tuple[str, int, str]]:
    """Yield (atmos, teff, logg_str) for every documented BOSZ (Teff, log g) pair.

    ATLAS9 ('ap') covers Teff 7500-16000 K; MARCS covers 2800-8000 K. In
    the 7500-8000 K overlap, ATLAS9 is preferred and the MARCS entry is
    skipped. Within the MARCS range, 'ms' (spherical) is used for
    log g <= 3.0 and 'mp' (plane-parallel) for log g > 3.0.
    """
    atlas9_covered = set()
    for teff_list, logg_list in _ATLAS9_RANGES:
        for teff in teff_list:
            for logg in logg_list:
                atlas9_covered.add((teff, logg))
                yield 'ap', teff, f"{logg:+.1f}"
    for teff_list, logg_list in _MARCS_RANGES:
        for teff in teff_list:
            for logg in logg_list:
                if (teff, logg) in atlas9_covered:
                    continue
                atmos = 'ms' if logg <= 3.0 else 'mp'
                yield atmos, teff, f"{logg:+.1f}"

# Apply --r / --feh / --afe / --cfe / --micro filters (no filter = use all)
r_vals     = args.r     if args.r     else _ALL_R_VALS
feh_vals   = args.feh   if args.feh   else _ALL_FEH_VALS
afe_vals   = args.afe   if args.afe   else _ALL_AFE_VALS
cfe_vals   = args.cfe   if args.cfe   else _ALL_CFE_VALS
micro_vals = args.micro if args.micro else _ALL_MICRO_VALS

files_avail = []
feh = []
afe = []
cfe = []
r = []
micro = []
for r_val in r_vals:
    r_dir = f"r{r_val}"
    for feh_val in feh_vals:
        m_dir = f"m{feh_val:+.2f}"
        for afe_val in afe_vals:
            for cfe_val in cfe_vals:
                for micro_val in micro_vals:
                    for atmos, teff, logg_str in _teff_logg_candidates():
                        fname = (
                            f"bosz{args.version}_{atmos}_t{teff}_g{logg_str}"
                            f"_m{feh_val:+.2f}_a{afe_val:+.2f}_c{cfe_val:+.2f}"
                            f"_v{micro_val}_r{r_val}_resam.txt.gz"
                        )
                        files_avail.append(f"{r_dir}/{m_dir}/{fname}")
                        feh.append(feh_val)
                        afe.append(afe_val)
                        cfe.append(cfe_val)
                        r.append(r_val)
                        micro.append(micro_val)
if args.verbose:
    print(f"Generated {len(files_avail)} candidate file paths.")

# If target file exists and overwrite is not specified, check if any of the
# requested spectra already exist in the file -- each (feh, afe, cfe, r,
# micro) combination lives in its own group, named
# spectra_feh<feh>_afe<afe>_cfe<cfe>_r<r>_micro<micro>, with those five
# values stored as group attributes -- and drop any match from the
# filtered list so we don't re-fetch it.
if shutil.os.path.exists(args.output) and not args.overwrite:
    existing = set()
    with h5py.File(args.output, 'r') as h5file:
        for grp in h5file.keys():
            attrs = h5file[grp].attrs
            existing.add((float(attrs['feh']), float(attrs['afe']), float(attrs['cfe']),
                          int(attrs['r']), int(attrs['micro'])))

    keep = [i for i in range(len(files_avail))
            if (feh[i], afe[i], cfe[i], r[i], micro[i]) not in existing]
    nduplicates = len(files_avail) - len(keep)
    if nduplicates > 0:
        files_avail = [files_avail[i] for i in keep]
        feh = [feh[i] for i in keep]
        afe = [afe[i] for i in keep]
        cfe = [cfe[i] for i in keep]
        r = [r[i] for i in keep]
        micro = [micro[i] for i in keep]
        if args.verbose:
            print(f"Found {nduplicates} existing spectra in {args.output}; skipping them.")

# If target output file does not exist, create it and write the references
if not shutil.os.path.exists(args.output):
    h5file = h5py.File(args.output, 'w')
    h5file.attrs['references'] = BOSZ_references
    h5file.attrs['reference_urls'] = BOSZ_refernce_URLS
    h5file.close()

# Group the filtered candidates by (feh, afe, cfe, r, micro): each
# combination becomes its own HDF5 group, holding every (teff, logg)
# spectrum for that combination.
groups = {}
for i in range(len(files_avail)):
    key = (feh[i], afe[i], cfe[i], r[i], micro[i])
    groups.setdefault(key, []).append(i)
if args.verbose:
    print(f"Grouped {len(files_avail)} files into {len(groups)} HDF5 groups.")

# When more than one atmosphere code covers the same (teff, logg) within
# a group, prefer "ap" (ATLAS9) over "ms"/"mp" (MARCS); lower number
# wins ties in the selection loop below
ATMOS_PREFERENCE = {'ap': 0, 'mp': 1, 'ms': 1}

# Create a temporary directory to store the downloaded files
temp_dir = "bosz_temp"
shutil.rmtree(temp_dir, ignore_errors=True)
shutil.os.makedirs(temp_dir, exist_ok=True)

# Loop over groups, downloading and processing the spectra in each
for key, idxs in groups.items():
    feh_val, afe_val, cfe_val, r_val, micro_val = key
    grp_name = (f"spectra_feh{feh_val:.2f}_afe{afe_val:.2f}_cfe{cfe_val:.2f}"
                f"_r{r_val}_micro{micro_val}")
    if args.verbose:
        print(f"Processing group {grp_name} ({len(idxs)} candidate files)...")

    # For each (teff, logg) covered by this group, keep only the file
    # from the preferred atmosphere code, in case more than one is
    # available
    best = {}
    for i in idxs:
        fname = shutil.os.path.basename(files_avail[i])
        match = name_re.fullmatch(fname)
        tl = (match.group('teff'), match.group('logg'))
        pref = ATMOS_PREFERENCE.get(match.group('atmos'), len(ATMOS_PREFERENCE))
        if tl not in best or pref < best[tl][0]:
            best[tl] = (pref, i, fname)

    # Download each selected file and extract the surface flux (4 pi
    # times the surface Eddington H values in its first column)
    spectra = {}
    for tl, (_, i, fname) in best.items():
        file_url = args.url + files_avail[i]
        if args.verbose:
            print(f"Fetching {file_url}...")
        outname = shutil.os.path.join(temp_dir, fname)
        http_status = 0
        with urllib3.PoolManager().request("GET", file_url, preload_content=False) as response, \
             open(outname, "wb") as out_file:
            http_status = response.status
            if http_status == 200:
                shutil.copyfileobj(response, out_file)
            elif http_status != 404:
                raise RuntimeError(f"Failed to fetch {file_url}: HTTP {http_status}")
        if http_status == 404:
            if args.verbose:
                print(f"  Not found (404), skipping.")
            shutil.os.remove(outname)
            continue

        # np.loadtxt transparently gunzips a .gz-named file
        eddington_h = np.loadtxt(outname, usecols=0)
        spectra[tl] = 4 * np.pi * eddington_h

        # Clean up the downloaded file
        shutil.os.remove(outname)

    # Write this group's spectra to the HDF5 file
    with h5py.File(args.output, 'a') as h5file:
        # Delete an existing group of this name first; if we're here it
        # means we want to overwrite it
        if grp_name in h5file:
            del h5file[grp_name]
        grp = h5file.create_group(grp_name)
        grp.attrs['feh'] = feh_val
        grp.attrs['afe'] = afe_val
        grp.attrs['cfe'] = cfe_val
        grp.attrs['r'] = r_val
        grp.attrs['micro'] = micro_val

        # Name each dataset by its Teff and log(g), using the same
        # naming convention as the original files (e.g. "t8000_g+3.0")
        for (teff, logg), flux in spectra.items():
            grp.create_dataset(f"t{teff}_g{logg}", data=flux, compression="gzip")

        if args.verbose:
            print(f"Wrote {len(spectra)} spectra to group {grp_name} in {args.output}.")

# The set of (Teff, log g) combinations available is the same for every
# spectra group, so rather than recovering it later by parsing dataset
# names, extract it once from whichever spectra group we have on hand
# and record it in a top-level "logg_Teff_grid" group, as two parallel
# datasets "Teff" and "logg" (pairs, not a full outer product -- the
# grid can be irregular at its edges).
ds_name_re = re.compile(r't(?P<teff>[0-9]+)_g(?P<logg>[+-][0-9.]+)')
with h5py.File(args.output, 'a') as h5file:
    spectra_groups = [g for g in h5file.keys() if g.startswith('spectra_')]
    if not spectra_groups:
        if args.verbose:
            print(f"No spectra groups found in {args.output}; skipping logg_Teff_grid.")
    else:
        sample_grp = h5file[spectra_groups[0]]
        pairs = sorted(
            (int(match.group('teff')), float(match.group('logg')))
            for match in (ds_name_re.fullmatch(ds_name) for ds_name in sample_grp.keys())
            if match is not None
        )

        if 'logg_Teff_grid' in h5file:
            del h5file['logg_Teff_grid']
        grid_grp = h5file.create_group('logg_Teff_grid')
        grid_grp.create_dataset('Teff', data=[p[0] for p in pairs])
        grid_grp.create_dataset('logg', data=[p[1] for p in pairs])

        if args.verbose:
            print(f"Wrote logg_Teff_grid ({len(pairs)} combinations, from group "
                  f"{spectra_groups[0]}) to {args.output}.")

# Fetch the wavelength grid for each resolution ("r") value we ended up
# with spectra for, and write it into a top-level "wavelengths" group as
# a dataset named r<value> -- one shared grid per resolution, rather
# than storing wavelengths with every individual spectrum. Skips a grid
# already present unless --overwrite is set, same convention as the
# spectra themselves.
with h5py.File(args.output, 'a') as h5file:
    wave_grp = h5file.require_group('wavelengths')
    for r_val in sorted(set(r)):
        ds_name = f"r{r_val}"
        if ds_name in wave_grp and not args.overwrite:
            if args.verbose:
                print(f"Wavelength grid {ds_name} already present in {args.output}; skipping.")
            continue

        wave_filename = f"bosz{args.version}_wave_r{r_val}.txt"
        wave_url = f"{args.url}wavelength_grids/{wave_filename}"
        if args.verbose:
            print(f"Fetching {wave_url}...")
        outname = shutil.os.path.join(temp_dir, wave_filename)
        with urllib3.PoolManager().request("GET", wave_url, preload_content=False) as response, open(outname, "wb") as out_file:
            if response.status != 200:
                raise RuntimeError(f"Failed to fetch {wave_url}: HTTP {response.status}")
            shutil.copyfileobj(response, out_file)

        wave = np.loadtxt(outname)
        if ds_name in wave_grp:
            del wave_grp[ds_name]
        wave_grp.create_dataset(ds_name, data=wave, compression="gzip")
        shutil.os.remove(outname)

        if args.verbose:
            print(f"Wrote wavelength grid {ds_name} ({len(wave)} points) to {args.output}.")

# Clean up all downloads
shutil.rmtree(temp_dir, ignore_errors=True)

# Read existing registry file if it exists, otherwise create a new one
if shutil.os.path.exists(args.registry):
    with open(args.registry, 'r') as f:
        registry = tomlkit.parse(f.read())
else:
    registry = { "name" : "Registry of spectra sets" }

# Add BOSZ to list of spectra sets
if "spectra_sets" in registry.keys():
    if "BOSZ" not in registry["spectra_sets"]:
        registry["spectra_sets"].append("BOSZ")
else:
    registry["spectra_sets"] = [ "BOSZ" ]

# Generate registry entry for BOSZ
if "BOSZ" in registry.keys():
    registry.pop("BOSZ")
bosz_tab = tomlkit.table()
bosz_tab["file"] = args.output
bosz_tab["version"] = args.version
bosz_tab["references"] = BOSZ_references
bosz_tab["reference_urls"] = BOSZ_refernce_URLS
# r and micro are integer-valued; the rest are floats. tomlkit accepts
# numpy.float64 (which happens to subclass Python's float) but not
# numpy.int64 (which does not subclass int), so cast explicitly to the
# right native Python type rather than handing it numpy scalars.
int_qtys = {"r", "micro"}
for qty, attr in zip(["Fe_H", "alpha_Fe", "C_Fe", "r", "micro"],
                     ["feh", "afe", "cfe", "r", "micro"]):
    val_in_file = []
    with h5py.File(args.output, 'r') as h5file:
        for grp in h5file.keys():
            if not grp.startswith('spectra_'):
                continue
            if h5file[grp].attrs[attr] not in val_in_file:
                val_in_file.append(h5file[grp].attrs[attr])
    val_in_file = np.array(val_in_file)
    val_in_file.sort()
    cast = int if qty in int_qtys else float
    bosz_tab[qty] = [ cast(v) for v in val_in_file ]
registry["BOSZ"] = bosz_tab

# Write registry back to file
with open(args.registry, 'w') as fp:
    fp.write(tomlkit.dumps(registry))
