"""
This is a script to fetch MIST (MESA Isochrones and Stellar Tracks)
tracks from the official MIST website, extract the data needed by
slug, and write it out in slug's HDF5 track format.
"""

# Imports
import argparse
import h5py
import http
import numpy as np
import re
import shutil
import tomlkit
import urllib3

# Magic strings
MIST_version = "v2.5"
MIST_URL = "https://mist.science/data/tarballs_v2.5/eeps/"
MIST_references = ["Dotter, A., Choi, J., Conroy, C., et al. 2016, ApJS, 222, 8",
                   "Choi, J., Dotter, A., Conroy, C., et al. 2016, ApJ, 823, 102",
                   "Dotter, A., Bauer, E., Park, M., et al. 2026, ApJS, 283, 64",
                   "Bauer, E., Dotter, A., Conroy, C., et al. 2026, ApJS, 283, 41"]
MIST_reference_URLs = ["https://ui.adsabs.harvard.edu/abs/2016ApJS..222....8D/abstract",
                       "https://ui.adsabs.harvard.edu/abs/2016ApJ...823..102C/abstract",
                       "https://ui.adsabs.harvard.edu/abs/2026ApJS..283...64D/abstract",
                       "https://ui.adsabs.harvard.edu/abs/2026ApJS..283...41B/abstract"]

# Parse command line arguments
parser = argparse.ArgumentParser(description="Fetch MIST tracks")
parser.add_argument("--version", default=MIST_version, 
                    help="MIST version to fetch")
parser.add_argument("--url", default=MIST_URL, 
                    help="URL of the MIST data")
parser.add_argument("--output", 
                    default=shutil.os.path.join("..", "tracks", "mist.h5"), 
                    help="Output file for the HDF5 tracks")
parser.add_argument("--registry", 
                    default=shutil.os.path.join("..", "tracks", "tracks.toml"), 
                    help="Output file for the registry")
parser.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing output file")
parser.add_argument("--feh", type=float, nargs="+", default=[],
                    help="List of [Fe/H] values to fetch; if unspecified, fetch all")
parser.add_argument("--afe", type=float, nargs="+", default=[],
                    help="List of [alpha/Fe] values to fetch; if unspecified, fetch all")
parser.add_argument("--vvcrit", type=float, nargs="+", default=[],
                    help="List of v/vcrit values to fetch; if unspecified, fetch all")
parser.add_argument("--verbose", action="store_true",
                    help="Print verbose output")
args = parser.parse_args()

# Start by reading the URL and fetching the list of available files
response = urllib3.PoolManager().request('GET', args.url)
if response.status != 200:
    raise RuntimeError(f"Failed to fetch MIST data from {args.url}: HTTP {response.status}")
files_avail = re.findall('<a href="MIST_'+args.version+'(.*?)">', str(response.data))
if args.verbose:
    print(f"Fetched track list from {args.url}: {len(files_avail)} files found.")

# Little helper function to translate the MIST convention of writing
# numbers in filenames as m050 or p025 into float values
def mist_to_float(s, digits=2) -> float:
    if s[0] == 'm':
        return -float(s[1:])/10**digits
    elif s[0] == 'p':
        return float(s[1:])/10**digits
    else:
        raise ValueError(f"Invalid MIST filename component: {s}")

# From the file names, extract the list of available Fe/H and alpha/Fe values
feh = [ mist_to_float(re.findall('_feh_(.*?)_', f)[0], 2) for f in files_avail ]
afe = [ mist_to_float(re.findall('_afe_(.*?)_', f)[0], 1) for f in files_avail ]
vvcrit = [ float(re.findall('_vvcrit(.*?)_', f)[0]) for f in files_avail ]

# If we were given a list of Fe/H or alpha/Fe values to fetch, filter the list of files accordingly
if args.feh:
    files_avail = [ f for f, feh_ in zip(files_avail, feh) 
                   if feh_ in args.feh ]
    feh = [ feh_ for feh_ in feh if feh_ in args.feh ]
if args.afe:
    files_avail = [ f for f, afe_ in zip(files_avail, afe) 
                   if afe_ in args.afe ]
    afe = [ afe_ for afe_ in afe if afe_ in args.afe ]
if args.vvcrit:
    files_avail = [ f for f, vvcrit_ in zip(files_avail, vvcrit) 
                   if vvcrit_ in args.vvcrit ]
    vvcrit = [ vvcrit_ for vvcrit_ in vvcrit if vvcrit_ in args.vvcrit ]
if args.verbose and (args.feh or args.afe or args.vvcrit):
    print(f"Filtered track list: {len(files_avail)} files to fetch.")

# If target file exists and overwrite is not specified, check if any of the
# requested tracks already exist in the file. If so, skip them.
if shutil.os.path.exists(args.output) and not args.overwrite:
    h5file = h5py.File(args.output, 'r')
    nduplicates = 0
    for grp in h5file.keys():
        existing_feh = h5file[grp].attrs['feh']
        existing_afe = h5file[grp].attrs['afe']
        existing_vvcrit = h5file[grp].attrs['vvcrit']
        for i, feh_, afe_, vvcrit_ in zip(np.arange(len(feh)), feh, afe, vvcrit):
            if (feh_ == existing_feh) and (afe_ == existing_afe) and \
                (vvcrit_ == existing_vvcrit):
                nduplicates += 1
                files_avail.pop(i)
                feh.pop(i)
                afe.pop(i)
                vvcrit.pop(i)
                break
    if args.verbose and nduplicates > 0:
        print(f"Found {nduplicates} existing tracks in {args.output}; skipping them.")
    h5file.close()

# If target output file does not exist, create it and write the references
if not shutil.os.path.exists(args.output):
    h5file = h5py.File(args.output, 'w')
    h5file.attrs['references'] = MIST_references
    h5file.attrs['reference_urls'] = MIST_reference_URLs
    h5file.close()

# Create a temporary directory to store the downloaded files
temp_dir = "mist_temp"
shutil.rmtree(temp_dir, ignore_errors=True)
shutil.os.makedirs(temp_dir, exist_ok=True)

# Loop over files to fetch
for f in files_avail:

    # Construct the full URL for the file
    filename = 'MIST_' + args.version + f
    file_url = args.url + filename
    if args.verbose:
        print(f"Fetching {file_url}...")
    
    # Fetch the file and redirect the output to the temporary directory
    outname = shutil.os.path.join(temp_dir, filename)
    with urllib3.PoolManager().request("GET", file_url, preload_content=False) as response, open(outname, "wb") as out_file:
        if response.status != 200:
            raise RuntimeError(f"Failed to fetch {file_url}: HTTP {response.status}")
        shutil.copyfileobj(response, out_file)

    # Unpack the tarball
    shutil.unpack_archive(outname, temp_dir)

    # Remove the tarball after unpacking
    shutil.os.remove(outname)
    
    # Read the list of unpacked files and extract the list of available initial masses
    unpack_dir_name = shutil.os.path.join(
        temp_dir, 
        re.findall('MIST_'+args.version+'_(.*?)_EEP', outname)[0],
        'eeps')
    track_files = shutil.os.listdir(unpack_dir_name)
    track_files.sort()
    track_masses = [ float(re.findall('(.*?)M', f)[0])/100. for f in track_files ]

    # Loop through the available initial masses and extract the data needed by slug
    track_data = []
    track_metadata = []
    for mass, track_file in zip(track_masses, track_files):
        # Construct the full path to the track file
        track_file_path = shutil.os.path.join(unpack_dir_name, track_file)
        
        # Read the track data from the file
        with open(track_file_path, 'r') as fp:
            
            # Ingest the file header
            for i in range(5):
                line = fp.readline()
            splt = line[1:].split()
            y_init = float(splt[0])
            z_init = float(splt[1])
            fe_h = float(splt[2])
            a_fe = float(splt[3])
            v_vcrit = float(splt[4])
            for i in range(3):
                line = fp.readline()
            splt = line[1:].split()
            m_init = float(splt[0])
            n_pts = int(splt[1])
            n_eep = int(splt[2])
            line = fp.readline()
            splt = line[1:].split()
            eeps = np.array([int(s) for s in splt[1:]])
            for i in range(3):
                line = fp.readline()
            cols = line[1:].split()

            # Read remainder of file
            fdat = np.loadtxt(fp)

        # Extract parts of data that we need to save
        t = fdat[:,cols.index('star_age')]
        m = fdat[:,cols.index('star_mass')]
        mdot = -fdat[:,cols.index('star_mdot')]  # Note sign convention change
        log_L = fdat[:,cols.index('log_L')]
        log_Teff = fdat[:,cols.index('log_Teff')]
        h_surf = fdat[:,cols.index('surface_h1')]
        he_surf = fdat[:,cols.index('surface_he3')] + \
            fdat[:,cols.index('surface_he4')]
        c_surf = fdat[:,cols.index('surface_c12')] + \
            fdat[:,cols.index('surface_c13')]
        n_surf = fdat[:,cols.index('surface_n14')]
        o_surf = fdat[:,cols.index('surface_o16')]
        phase = fdat[:,cols.index('phase')]

        # Store file data
        file_metadata = {
            'm_init' : m_init,
            'y_init' : y_init,
            'z_init' : z_init,
            'v_vcrit' : v_vcrit,
            'fe_h' : fe_h,
            'a_fe' : a_fe,
            'n_pts' : n_pts
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
            'o_surf' : o_surf,
            'phase' : phase
        }
        track_data.append(file_data)
        track_metadata.append(file_metadata)

    # Write the data to the HDF5 file
    flds = ['age', 'mass', 'mdot', 'log_L', 'log_Teff',
            'h_surf', 'he_surf', 'c_surf', 'n_surf', 'o_surf', 'phase']
    with h5py.File(args.output, 'a') as h5file:

        # Create a group for this set of tracks; delete existing one if
        # found, since if we're here it means we want to overwrite it
        grp_name = f"feh_{fe_h:.2f}_afe_{a_fe:.1f}_vvcrit_{v_vcrit:.2f}"
        if grp_name in h5file:
            del h5file[grp_name]
        grp = h5file.create_group(grp_name)

        # Write metadata for this set of tracks
        grp.attrs['feh'] = track_metadata[0]['fe_h']
        grp.attrs['afe'] = track_metadata[0]['a_fe']
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

    # Clean up the unpacked files
    cleanup_name = shutil.os.path.join(
        temp_dir, 
        re.findall('MIST_'+args.version+'_(.*?)_EEP', outname)[0])
    shutil.rmtree(cleanup_name, ignore_errors=True)

# Clean up all downloads
shutil.rmtree(temp_dir, ignore_errors=True)

# Read existing registry file if it exists, otherwise create a new one
if shutil.os.path.exists(args.registry):
    with open(args.registry, 'r') as f:
        registry = tomlkit.parse(f.read())
else:
    registry = { "name" : "Registry of track sets" }

# Add MIST to list of track sets
if "track_sets" in registry.keys():
    if "MIST" not in registry["track_sets"]:
        registry["track_sets"].append("MIST")
else:
    registry["track_sets"] = [ "MIST" ]
    
# Generate registry entry for MIST
if "MIST" in registry.keys():
    registry.pop("MIST")
mist_tab = tomlkit.table()
mist_tab["file"] = args.output
mist_tab["version"] = args.version
mist_tab["references"] = MIST_references
mist_tab["reference_urls"] = MIST_reference_URLs
for qty, attr in zip(["Fe_H", "alpha_Fe", "v_vcrit"],
                     ["feh", "afe", "vvcrit"]):
    val_in_file = []
    with h5py.File(args.output, 'r') as h5file:
        for grp in h5file.keys():
            if h5file[grp].attrs[attr] not in val_in_file:
                val_in_file.append(h5file[grp].attrs[attr])
    val_in_file = np.array(val_in_file)
    val_in_file.sort()
    mist_tab[qty] = [ v for v in val_in_file ]
registry["MIST"] = mist_tab

# Write registry back to file
with open(args.registry, 'w') as fp:
    fp.write(tomlkit.dumps(registry))