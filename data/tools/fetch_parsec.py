"""
This is a script to fetch PARSEC (PAdova and TRieste Stellar Evolution Code)
tracks from the official PARSEC database, extract the data needed by
slug, and write it out in slug's HDF5 track format.
"""

# Imports
import argparse
import h5py
import numpy as np
import re
import shutil
import tomlkit
import urllib3
import zipfile

# Magic strings
PARSEC_version = "v2.0"
PARSEC_URLS = ["https://stev.oapd.inaf.it/PARSEC/Database/PARSECv2.0_ROT_2025/",
              "https://stev.oapd.inaf.it/PARSEC/Database/PARSECv2.0_VMS/"]
# Regexes used to pick out the downloadable track archives from the
# directory listing at each of the URLs above; order must match PARSEC_URLS
PARSEC_FILE_PATTERNS = [r'<a href="(VAR_ROT[^"]*?\.zip)">',
                       r'<a href="(Z[^"]*?_tracks\.zip)">']
PARSEC_references = [
    "Nguyen, C. T., Costa, G., Girardi, L., et al. 2022, A&A, 665, A126",
    "Nguyen, C. T., Costa, G., Bressan, A., et al. 2025, A&A, 701, A258",
    "Costa, G., Shepherd, K. G., Bressan, A., et al. 2025, arXiv:2501.12917",
    "Costa, G., Girardi, L., Bressan, A., et al. 2019, MNRAS, 485, 4641",
    "Costa, G., Girardi, L., Bressan, A., et al. 2019, A&A, 631, A128",
]
PARSEC_reference_URLs = [
    "https://ui.adsabs.harvard.edu/abs/2022A%26A...665A.126N/abstract",
    "https://ui.adsabs.harvard.edu/abs/2025A%26A...701A.258N/abstract",
    "https://ui.adsabs.harvard.edu/abs/2025arXiv250112917C/abstract",
    "https://ui.adsabs.harvard.edu/abs/2019MNRAS.485.4641C/abstract",
    "https://ui.adsabs.harvard.edu/abs/2019A%26A...631A.128C/abstract",
]
PARSEC_ZSUN = 0.01524    # PARSEC's preferred "Solar" metallicity

# Parse command line arguments
parser = argparse.ArgumentParser(description="Fetch PARSEC tracks")
parser.add_argument("--version", default=PARSEC_version,
                    help="PARSEC version to fetch")
parser.add_argument("--url", type=str, nargs=2, default=PARSEC_URLS,
                    metavar=("ROT_URL", "VMS_URL"),
                    help="URLs of the PARSEC rotating and VMS track databases")
parser.add_argument("--output",
                    default=shutil.os.path.join("..", "tracks", "parsec.h5"),
                    help="Output file for the HDF5 tracks")
parser.add_argument("--registry",
                    default=shutil.os.path.join("..", "tracks", "tracks.toml"),
                    help="Output file for the registry")
parser.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing output file")
parser.add_argument("--feh", type=float, nargs="+", default=[],
                    help="List of [Fe/H] values to fetch; if unspecified, fetch all")
parser.add_argument("--vvcrit", type=float, nargs="+", default=[],
                    help="List of v/vcrit values to fetch; if unspecified, fetch all")
parser.add_argument("--verbose", action="store_true",
                    help="Print verbose output")
args = parser.parse_args()

# The PARSEC web server does not send its intermediate certificate during
# the TLS handshake, so Python's certificate verification cannot build a
# full chain to a trusted root (unlike e.g. curl / macOS, which fetch the
# missing intermediate automatically). Since we are only fetching public,
# non-sensitive data, we disable certificate verification for these
# requests rather than failing.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
http = urllib3.PoolManager(cert_reqs="CERT_NONE")

# If target output file does not exist, create it and write the references
if not shutil.os.path.exists(args.output):
    h5file = h5py.File(args.output, 'w')
    h5file.attrs['references'] = PARSEC_references
    h5file.attrs['reference_urls'] = PARSEC_reference_URLs
    h5file.close()

# Create a temporary directory to store the downloaded files
temp_dir = "parsec_temp"
shutil.rmtree(temp_dir, ignore_errors=True)
shutil.os.makedirs(temp_dir, exist_ok=True)

# Loop over the two source URLs, along with the regex used to pick out the
# downloadable archives at each one
for url, pattern in zip(args.url, PARSEC_FILE_PATTERNS):

    # Fetch the directory listing for this URL
    response = http.request('GET', url)
    if response.status != 200:
        raise RuntimeError(f"Failed to fetch PARSEC data from {url}: HTTP {response.status}")
    files_avail = re.findall(pattern, str(response.data))
    if args.verbose:
        print(f"Fetched track list from {url}: {len(files_avail)} files found.")

    # From the file names, extract the value of Fe/H for each file, encoded
    # as Z<value>_ in the filename; value may be in plain decimal or
    # scientific notation (e.g. Z0.014_ or Z1E-6_)
    feh = [ float(re.findall(r'Z([\d.]+(?:[eE][+-]?\d+)?)_', f)[0]) for f in files_avail ]

    # If we were given a list of Fe/H values to fetch, filter the list of
    # files accordingly
    if args.feh:
        files_avail = [ f for f, feh_ in zip(files_avail, feh)
                       if feh_ in args.feh ]
        feh = [ feh_ for feh_ in feh if feh_ in args.feh ]

    # From the file names, extract the value of v/vcrit for each file,
    # encoded as VAR_ROT<value>_ in the filename; files that do not match
    # the VAR_ROT pattern (i.e. the VMS tracks) are non-rotating, so
    # vvcrit = 0 for them
    vvcrit = []
    has_vvcrit = []
    for f in files_avail:
        match = re.findall(r'VAR_ROT([\d.]+)_', f)
        vvcrit.append(float(match[0]) if match else 0.0)

    # If we were given a list of v/vcrit values to fetch, filter the list of
    # files accordingly, but keep any files (like the VMS tracks) that do
    # not carry v/vcrit information regardless of the requested values
    if args.vvcrit:
        keep = [ (not h) or (vvcrit_ in args.vvcrit)
                for vvcrit_, h in zip(vvcrit, has_vvcrit) ]
        files_avail = [ f for f, k in zip(files_avail, keep) if k ]
        feh = [ feh_ for feh_, k in zip(feh, keep) if k ]
        vvcrit = [ vvcrit_ for vvcrit_, k in zip(vvcrit, keep) if k ]

    # Print effects of filtering
    if args.verbose and (args.vvcrit or args.feh):
        print(f"Filtered track list: {len(files_avail)} files to fetch.")

    # If target file exists and overwrite is not specified, check if any of
    # the requested tracks already exist in the file. If so, skip them. Note
    # that we have to be a bit careful with the metallicity, because the
    # metallicity stored in the PARSEC database is the absolute value of Z,
    # but what we store in the slug database is log10(Z/ZSun), truncated to
    # a finite number of decimal places
    if shutil.os.path.exists(args.output) and not args.overwrite:
        h5file = h5py.File(args.output, 'r')
        nduplicates = 0
        feh_converted = np.log10(np.array(feh)/PARSEC_ZSUN)
        for grp in h5file.keys():
            existing_feh = h5file[grp].attrs['feh']
            existing_vvcrit = h5file[grp].attrs['vvcrit']
            for i, feh_, vvcrit_ in zip(np.arange(len(feh)), feh, vvcrit):
                if (np.abs(np.log10(feh_/PARSEC_ZSUN) - existing_feh) < 1e-4) \
                    and (vvcrit_ == existing_vvcrit):
                    nduplicates += 1
                    files_avail.pop(i)
                    feh.pop(i)
                    vvcrit.pop(i)
                    break
        if args.verbose and nduplicates > 0:
            print(f"Found {nduplicates} existing tracks in {args.output}; skipping them.")
        h5file.close()

    # Loop over files to fetch
    for filename, feh_, vvcrit_ in zip(files_avail, feh, vvcrit):

        # Construct the full URL for the file
        file_url = url + filename
        if args.verbose:
            print(f"Fetching {file_url}...")

        # Fetch the file and redirect the output to the temporary directory
        outname = shutil.os.path.join(temp_dir, filename)
        with http.request("GET", file_url, preload_content=False) as response, open(outname, "wb") as out_file:
            if response.status != 200:
                raise RuntimeError(f"Failed to fetch {file_url}: HTTP {response.status}")
            shutil.copyfileobj(response, out_file)

        # Unpack the zip archive, keeping track of the names of the
        # members it contains so we can find them (and clean them up)
        # without picking up leftover files from a previously processed
        # archive
        with zipfile.ZipFile(outname) as zf:
            member_names = zf.namelist()
            zf.extractall(temp_dir)

        # Remove the zip archive after unpacking
        shutil.os.remove(outname)

        # Select the individual track files: those whose names end in
        # M<mass>.TAB, where <mass> is the initial stellar mass
        track_files = [ m for m in member_names if re.search(r'M[\d.]+\.TAB$', m) ]
        track_files.sort()

        # Loop through the available initial masses and extract the data needed by slug
        track_data = []
        track_metadata = []
        for track_file in track_files:

            # Extract initial mass, metallicity, and helium fraction from
            # the file name; both the VMS (Z<z>_Y<y>_...) and ROT
            # (Z<z>Y<y>...) naming conventions are handled by making the
            # separating underscore optional
            m_init = float(re.search(r'M([\d.]+)\.TAB$', track_file).group(1))
            z_init, y_init = ( float(v) for v in
                               re.search(r'Z([\d.]+(?:[eE][+-]?\d+)?)_?Y([\d.]+)',
                                        track_file).groups() )

            # Read the track data from the file
            track_file_path = shutil.os.path.join(temp_dir, track_file)
            with open(track_file_path, 'r') as fp:

                # Skip any comment lines (VMS files have two, ROT files
                # have none), then read the column header line
                line = fp.readline()
                while line.startswith('#'):
                    line = fp.readline()
                cols = line.split()

                # Read remainder of file
                fdat = np.loadtxt(fp)

            # Helper to extract surface abundances; this function is needed
            # because there are three different formats of PARSEC files, which
            # have different column names
            def surf_abund(colformat) -> np.ndarray:
                for cf in colformat:
                    if all(c in cols for c in cf):
                        return sum(fdat[:, cols.index(c)] for c in cf)
                raise ValueError("no matching column format")

            # Extract parts of data that we need to save
            t = fdat[:, cols.index('AGE')]
            m = fdat[:, cols.index('MASS')]
            if 'RATE' in cols:
                mdot = fdat[:, cols.index('RATE')]
            else:
                mdot = np.zeros(fdat.shape[0])
            log_L = fdat[:, cols.index('LOG_L')]
            log_Teff = fdat[:, cols.index('LOG_TE')]
            h_surf = surf_abund([['XH1_SURF', 'XD_SURF'], ['H_SUP'], ['Xsup']])
            he_surf = surf_abund([['XHE3_SURF', 'XHE4_SURF'], ['HE_SUP'], ['Ysup']])
            c_surf = surf_abund([['XC12_SURF', 'XC13_SURF'], ['C_SUP'], ['XCsup', 'XC13sup']])
            n_surf = surf_abund([['XN14_SURF', 'XN15_SURF'], ['N_SUP'], ['XNsup']])
            o_surf = surf_abund([['XO16_SURF', 'XO17_SURF', 'XO18_SURF'], ['O_SUP'], ['XOsup', 'XO18sup']])

            # Store file data
            file_metadata = {
                'm_init' : m_init,
                'y_init' : y_init,
                'z_init' : z_init,
                'v_vcrit' : vvcrit_,
                'fe_h' : feh_,
                'n_pts' : fdat.shape[0]
            }
            file_data = {
                'age' : t,
                'mass' : m,
                'mdot' : mdot,
                'log_L' : log_L,
                'log_Teff' : log_Teff,
                'h_surf' : h_surf,
                'he_surf' : he_surf,
                'c_surf' : c_surf,
                'n_surf' : n_surf,
                'o_surf' : o_surf
            }
            track_data.append(file_data)
            track_metadata.append(file_metadata)

        # Write the data to the HDF5 file
        flds = ['age', 'mass', 'mdot', 'log_L', 'log_Teff',
                'h_surf', 'he_surf', 'c_surf', 'n_surf', 'o_surf']
        with h5py.File(args.output, 'a') as h5file:

            # Get metallicity on a Solar-normalized log scale
            feh_solar = np.log10(feh_/PARSEC_ZSUN)

            # Create a group for this set of tracks; delete existing one if
            # found, since if we're here it means we want to overwrite it
            grp_name = f"feh_{feh_solar:.6g}_vvcrit_{vvcrit_:.2f}"
            if grp_name in h5file:
                del h5file[grp_name]
            grp = h5file.create_group(grp_name)

            # Write metadata for this set of tracks
            grp.attrs['feh'] = feh_solar
            grp.attrs['vvcrit'] = track_metadata[0]['v_vcrit']
            grp.attrs['y_init'] = track_metadata[0]['y_init']
            grp.attrs['z_init'] = track_metadata[0]['z_init']
            grp.attrs['ntime'] = track_metadata[0]['n_pts']
            grp.attrs['nmass'] = len(track_data)
            grp.attrs['m_min'] = min(md['m_init'] for md in track_metadata)
            grp.attrs['m_max'] = max(md['m_init'] for md in track_metadata)

            # Write the list of initial masses as a dataset in the group
            grp.create_dataset('masses', data=[md['m_init'] for md in track_metadata])

            # Write track for each initial mass to a separate group
            for md, td in zip(track_metadata, track_data):
                track_array = np.zeros((md["n_pts"], len(flds)))
                for j, f in enumerate(flds):
                    track_array[:,j] = td[f]
                grp.create_dataset(f"track_m{md['m_init']:.3f}",
                                   data=track_array, compression="gzip")

            # Store the mapping from field names to indices in the last dimension of the track array
            grp.attrs['field_names'] = flds

            # Print verbose output if requested
            if args.verbose:
                print(f"Wrote {len(track_data)} tracks to group {grp_name} in {args.output}.")

        # Clean up the unpacked track files for this archive
        for member in member_names:
            member_path = shutil.os.path.join(temp_dir, member)
            if shutil.os.path.exists(member_path):
                shutil.os.remove(member_path)

# Clean up all downloads
shutil.rmtree(temp_dir, ignore_errors=True)

# Read existing registry file if it exists, otherwise create a new one
if shutil.os.path.exists(args.registry):
    with open(args.registry, 'r') as f:
        registry = tomlkit.parse(f.read())
else:
    registry = { "name" : "Registry of track sets" }

# Add PARSEC to list of track sets
if "track_sets" in registry.keys():
    if "PARSEC" not in registry["track_sets"]:
        registry["track_sets"].append("PARSEC")
else:
    registry["track_sets"] = [ "PARSEC" ]

# Generate registry entry for PARSEC
if "PARSEC" in registry.keys():
    registry.pop("PARSEC")
parsec_tab = tomlkit.table()
parsec_tab["file"] = args.output
parsec_tab["version"] = args.version
parsec_tab["references"] = PARSEC_references
parsec_tab["reference_urls"] = PARSEC_reference_URLs
for qty, attr in zip(["Fe_H", "v_vcrit"],
                     ["feh", "vvcrit"]):
    val_in_file = []
    with h5py.File(args.output, 'r') as h5file:
        for grp in h5file.keys():
            if h5file[grp].attrs[attr] not in val_in_file:
                val_in_file.append(h5file[grp].attrs[attr])
    val_in_file = np.array(val_in_file)
    val_in_file.sort()
    parsec_tab[qty] = [ v for v in val_in_file ]
registry["PARSEC"] = parsec_tab

# Write registry back to file
with open(args.registry, 'w') as fp:
    fp.write(tomlkit.dumps(registry))
