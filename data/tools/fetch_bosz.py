"""
This is a script to fetch the BOSZ spectra (https://archive.stsci.edu/hlsp/bosz)
from the offical STScI website, extract data from them, and write them into a
gzip'ed HDF5 file that slug can read.
"""

# Imports
import argparse
import h5py
import numpy as np
import re
import shutil
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

# Little helper function to fetch a directory-listing page and return the
# list of href values it links to -- subdirectory names ending in "/", or
# plain file names
def list_dir_entries(url) -> list:
    response = urllib3.PoolManager().request('GET', url)
    if response.status != 200:
        raise RuntimeError(f"Failed to fetch BOSZ directory listing from {url}: HTTP {response.status}")
    return re.findall('<a href="([^"]+)">', str(response.data))

# Regex to parse a BOSZ spectrum filename into its component fields; see
# https://archive.stsci.edu/hlsp/bosz for the naming convention, e.g.
# bosz2024_mp_t5000_g+5.0_m+0.00_a+0.00_c+0.00_v0_r500_resam.txt.gz
name_re = re.compile(
    r'bosz' + re.escape(args.version) +
    r'_(?P<atmos>[a-z]+)_t(?P<teff>[0-9]+)_g(?P<logg>[+-][0-9.]+)'
    r'_m(?P<feh>[+-][0-9.]+)_a(?P<afe>[+-][0-9.]+)_c(?P<cfe>[+-][0-9.]+)'
    r'_v(?P<micro>[0-9]+)_r(?P<r>[0-9]+)_(?P<prod>[a-z]+)\.txt\.gz')

# BOSZ organizes spectra in a two-level directory hierarchy under args.url:
# r<value>/ (instrumental broadening, our --r argument) then m<value>/
# (metallicity, our --feh argument), with the actual spectrum files
# living flat inside each m<value>/ directory. Rather than crawling the
# whole tree and filtering afterward (as fetch_mist.py does for its
# single flat directory), descend directly into the requested r/feh
# subdirectories when given, since crawling every combination would mean
# hundreds of HTTP requests just to build the candidate list; an omitted
# --r or --feh falls back to discovering every subdirectory, matching
# fetch_mist.py's "no filter means match everything" convention.
if args.r:
    r_dirs = [f"r{rv}" for rv in args.r]
else:
    r_dirs = sorted({e.rstrip('/') for e in list_dir_entries(args.url)
                     if re.fullmatch(r'r[0-9]+/', e)})
if args.verbose:
    print(f"Searching resolution directories: {r_dirs}")

# Walk the requested (or discovered) r/feh subdirectories, parse every
# filename found in each, and keep only the "resam" products (resampled
# onto a logarithmically-uniform wavelength grid, the ones we want)
# matching any --afe/--cfe/--micro filters given; as with --r/--feh
# above, an omitted filter matches everything.
files_avail = []
feh = []
afe = []
cfe = []
r = []
micro = []
for r_dir in r_dirs:
    r_url = f"{args.url}{r_dir}/"
    if args.feh:
        m_dirs = [f"m{fv:+.2f}" for fv in args.feh]
    else:
        m_dirs = sorted({e.rstrip('/') for e in list_dir_entries(r_url)
                         if re.fullmatch(r'm[+-][0-9.]+/', e)})

    for m_dir in m_dirs:
        m_url = f"{r_url}{m_dir}/"
        for fname in list_dir_entries(m_url):
            match = name_re.fullmatch(fname)
            if match is None or match.group('prod') != 'resam':
                continue
            afe_val = float(match.group('afe'))
            cfe_val = float(match.group('cfe'))
            micro_val = int(match.group('micro'))
            if args.afe and afe_val not in args.afe:
                continue
            if args.cfe and cfe_val not in args.cfe:
                continue
            if args.micro and micro_val not in args.micro:
                continue
            files_avail.append(f"{r_dir}/{m_dir}/{fname}")
            feh.append(float(match.group('feh')))
            afe.append(afe_val)
            cfe.append(cfe_val)
            r.append(int(match.group('r')))
            micro.append(micro_val)
if args.verbose:
    print(f"Filtered file list: {len(files_avail)} files to fetch.")

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
        with urllib3.PoolManager().request("GET", file_url, preload_content=False) as response, open(outname, "wb") as out_file:
            if response.status != 200:
                raise RuntimeError(f"Failed to fetch {file_url}: HTTP {response.status}")
            shutil.copyfileobj(response, out_file)

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

# Clean up all downloads
shutil.rmtree(temp_dir, ignore_errors=True)
