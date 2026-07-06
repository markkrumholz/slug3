"""
This is a script to fetch the Stromlo Stellar Tracks (Grasha et al. 2021)
from their public Google Drive repository, extract the data needed by
slug, and write it out in slug's HDF5 track format.

Note: this script requires the third-party "gdown" package (pip install
gdown), since the tracks are hosted as Google Drive folders rather than a
plain HTTP directory listing.
"""

# Imports
import argparse
import gdown
import h5py
import numpy as np
import re
import shutil
import tarfile
import tomlkit

# Magic strings
STROMLO_URLS = [
    "https://drive.google.com/drive/folders/1nyHm_kHpRdWhr-4adrRZFtMl4MmYxDtm?usp=sharing",
    "https://drive.google.com/drive/folders/1ZFBqZfcXpS1Tuo1nVAtx4B8bRvQV4ovX?usp=sharing",
    "https://drive.google.com/drive/folders/1dulkZ8ffycTMCt6ECodLtqK1I-oMlATa?usp=sharing",
]
STROMLO_references = [
    "Grasha, K., Roy, A., Sutherland, R. S., Kewley, L. J. 2021, ApJ, 908, 241",
]
STROMLO_reference_URLs = [
    "https://ui.adsabs.harvard.edu/abs/2021ApJ...908..241G/abstract",
]

# Parse command line arguments
parser = argparse.ArgumentParser(description="Fetch Stromlo tracks")
parser.add_argument("--url", type=str, nargs=3, default=STROMLO_URLS,
                    metavar=("VVCRIT0_URL", "VVCRIT02_URL", "VVCRIT04_URL"),
                    help="URLs of the Google Drive folders holding the "
                    "v/vcrit = 0.0, 0.2, and 0.4 track grids")
parser.add_argument("--output",
                    default=shutil.os.path.join("..", "tracks", "stromlo.h5"),
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

# Little helper function to translate the Stromlo convention of writing
# signed values in filenames as e.g. m0.1 or p0.5 into float values; unlike
# MIST, these are literal decimal values, not integer digits to be scaled
def stromlo_to_float(s) -> float:
    if s[0] == 'm':
        return -float(s[1:])
    elif s[0] == 'p':
        return float(s[1:])
    else:
        return float(s)

# If target output file does not exist, create it and write the references
if not shutil.os.path.exists(args.output):
    h5file = h5py.File(args.output, 'w')
    h5file.attrs['references'] = STROMLO_references
    h5file.attrs['reference_urls'] = STROMLO_reference_URLs
    h5file.close()

# Create a temporary directory to store the downloaded files
temp_dir = "stromlo_temp"
shutil.rmtree(temp_dir, ignore_errors=True)
shutil.os.makedirs(temp_dir, exist_ok=True)

# Loop over the three source Google Drive folders (one per v/vcrit grid)
for url in args.url:

    # Fetch the folder listing for this URL; skip_download=True just
    # retrieves the file names and ids without fetching their contents
    if args.verbose:
        print(f"Fetching file list from {url}...")
    drive_files = gdown.download_folder(url, skip_download=True,
                                        quiet=not args.verbose)
    if drive_files is None:
        raise RuntimeError(f"Failed to fetch Stromlo file list from {url}")

    # Keep only the track archives; this also excludes the README.pdf that
    # each folder contains
    files_avail = [ f for f in drive_files
                   if re.search(r'^stromlo_feh_.*_gc\.tar\.gz$', f.path) ]
    if args.verbose:
        print(f"Fetched track list from {url}: {len(files_avail)} files found.")

    # From the file names, extract the values of Fe/H, alpha/Fe, and
    # v/vcrit for each file; note that we get v/vcrit from the filename
    # rather than the per-track file header (see below) because at least
    # one of the released archives has an incorrect v/vcrit value baked
    # into every one of its files' headers
    feh = [ stromlo_to_float(re.findall(r'_feh_(.*?)_afe_', f.path)[0])
           for f in files_avail ]
    afe = [ stromlo_to_float(re.findall(r'_afe_(.*?)_vvcrit_', f.path)[0])
           for f in files_avail ]
    vvcrit = [ stromlo_to_float(re.findall(r'_vvcrit_(.*?)_gc', f.path)[0])
              for f in files_avail ]

    # If we were given lists of Fe/H, alpha/Fe, or v/vcrit values to fetch,
    # filter the list of files accordingly
    if args.feh:
        files_avail = [ f for f, feh_ in zip(files_avail, feh) if feh_ in args.feh ]
        afe = [ afe_ for afe_, feh_ in zip(afe, feh) if feh_ in args.feh ]
        vvcrit = [ vvcrit_ for vvcrit_, feh_ in zip(vvcrit, feh) if feh_ in args.feh ]
        feh = [ feh_ for feh_ in feh if feh_ in args.feh ]
    if args.afe:
        files_avail = [ f for f, afe_ in zip(files_avail, afe) if afe_ in args.afe ]
        feh = [ feh_ for feh_, afe_ in zip(feh, afe) if afe_ in args.afe ]
        vvcrit = [ vvcrit_ for vvcrit_, afe_ in zip(vvcrit, afe) if afe_ in args.afe ]
        afe = [ afe_ for afe_ in afe if afe_ in args.afe ]
    if args.vvcrit:
        files_avail = [ f for f, vvcrit_ in zip(files_avail, vvcrit) if vvcrit_ in args.vvcrit ]
        feh = [ feh_ for feh_, vvcrit_ in zip(feh, vvcrit) if vvcrit_ in args.vvcrit ]
        afe = [ afe_ for afe_, vvcrit_ in zip(afe, vvcrit) if vvcrit_ in args.vvcrit ]
        vvcrit = [ vvcrit_ for vvcrit_ in vvcrit if vvcrit_ in args.vvcrit ]
    if args.verbose and (args.feh or args.afe or args.vvcrit):
        print(f"Filtered track list: {len(files_avail)} files to fetch.")

    # If target file exists and overwrite is not specified, check if any of
    # the requested tracks already exist in the file. If so, skip them.
    if shutil.os.path.exists(args.output) and not args.overwrite:
        h5file = h5py.File(args.output, 'r')
        nduplicates = 0
        for grp in h5file.keys():
            existing_feh = h5file[grp].attrs['feh']
            existing_afe = h5file[grp].attrs['afe']
            existing_vvcrit = h5file[grp].attrs['vvcrit']
            if (existing_feh in feh) and (existing_afe in afe) and \
                (existing_vvcrit in vvcrit):
                idx = feh.index(existing_feh)
                if (idx == afe.index(existing_afe)) and \
                    (idx == vvcrit.index(existing_vvcrit)):
                    nduplicates += 1
                    files_avail.pop(idx)
                    feh.pop(idx)
                    afe.pop(idx)
                    vvcrit.pop(idx)
        if args.verbose and nduplicates > 0:
            print(f"Found {nduplicates} existing tracks in {args.output}; skipping them.")
        h5file.close()

    # Loop over files to fetch
    for drive_file, feh_, afe_, vvcrit_ in zip(files_avail, feh, afe, vvcrit):

        # Download the tarball into the temporary directory
        if args.verbose:
            print(f"Fetching {drive_file.path}...")
        outname = shutil.os.path.join(temp_dir, drive_file.path)
        gdown.download(id=drive_file.id, output=outname, quiet=not args.verbose)

        # Unpack the tarball, keeping track of the names of the members it
        # contains so we can find them (and clean them up) without picking
        # up leftover files from a previously processed archive
        with tarfile.open(outname) as tf:
            member_names = tf.getnames()
            tf.extractall(temp_dir, filter='data')

        # Remove the tarball after unpacking
        shutil.os.remove(outname)

        # Select the individual track files: those ending in .track.eep,
        # excluding the macOS "AppleDouble" resource-fork files (._*) that
        # some of these tarballs were archived with
        track_files = [ m for m in member_names
                       if m.endswith('.track.eep')
                       and not shutil.os.path.basename(m).startswith('._') ]
        track_files.sort()

        # Loop through the available initial masses and extract the data needed by slug
        track_data = []
        track_metadata = []
        for track_file in track_files:
            track_file_path = shutil.os.path.join(temp_dir, track_file)

            # Read the track data from the file
            with open(track_file_path, 'r') as fp:

                # Ingest the file header; y_init/z_init come from the
                # header, but fe_h/a_fe/v_vcrit are taken from the
                # filename instead (see note above)
                for i in range(5):
                    line = fp.readline()
                splt = line[1:].split()
                y_init = float(splt[0])
                z_init = float(splt[1])
                for i in range(3):
                    line = fp.readline()
                splt = line[1:].split()
                m_init = float(splt[0])
                n_pts = int(splt[1])
                for i in range(4):
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
                'v_vcrit' : vvcrit_,
                'fe_h' : feh_,
                'a_fe' : afe_,
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
            grp_name = f"feh_{feh_:.2f}_afe_{afe_:.1f}_vvcrit_{vvcrit_:.2f}"
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
                for j, fld in enumerate(flds):
                    track_array[:,j] = td[fld]
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
                if shutil.os.path.isdir(member_path):
                    shutil.rmtree(member_path, ignore_errors=True)
                else:
                    shutil.os.remove(member_path)

# Clean up all downloads
shutil.rmtree(temp_dir, ignore_errors=True)

# Read existing registry file if it exists, otherwise create a new one
if shutil.os.path.exists(args.registry):
    with open(args.registry, 'r') as f:
        registry = tomlkit.parse(f.read())
else:
    registry = { "name" : "Registry of track sets" }

# Add Stromlo to list of track sets
if "track_sets" in registry.keys():
    if "Stromlo" not in registry["track_sets"]:
        registry["track_sets"].append("Stromlo")
else:
    registry["track_sets"] = [ "Stromlo" ]

# Generate registry entry for Stromlo
if "Stromlo" in registry.keys():
    registry.pop("Stromlo")
stromlo_tab = tomlkit.table()
stromlo_tab["file"] = args.output
stromlo_tab["references"] = STROMLO_references
stromlo_tab["reference_urls"] = STROMLO_reference_URLs
for qty, attr in zip(["Fe_H", "alpha_Fe", "v_vcrit"],
                     ["feh", "afe", "vvcrit"]):
    val_in_file = []
    with h5py.File(args.output, 'r') as h5file:
        for grp in h5file.keys():
            if h5file[grp].attrs[attr] not in val_in_file:
                val_in_file.append(h5file[grp].attrs[attr])
    val_in_file = np.array(val_in_file)
    val_in_file.sort()
    stromlo_tab[qty] = [ v for v in val_in_file ]
registry["Stromlo"] = stromlo_tab

# Write registry back to file
with open(args.registry, 'w') as fp:
    fp.write(tomlkit.dumps(registry))
