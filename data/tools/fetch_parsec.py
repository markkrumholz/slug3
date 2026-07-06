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

# Magic strings
PARSEC_version = "v2.0"
PARSEC_URLS = ["https://stev.oapd.inaf.it/PARSEC/Database/PARSECv2.0_ROT_2025/",
              "https://stev.oapd.inaf.it/PARSEC/Database/PARSECv2.0_VMS/"]
# Regexes used to pick out the downloadable track archives from the
# directory listing at each of the URLs above; order must match PARSEC_URLS
PARSEC_FILE_PATTERNS = [r'<a href="(ALL_ROT_.*?\.zip)">',
                       r'<a href="(.*?_tracks\.zip)">']
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

# Create a temporary directory to store the downloaded files
temp_dir = "parsec_temp"
shutil.rmtree(temp_dir, ignore_errors=True)
shutil.os.makedirs(temp_dir, exist_ok=True)

# Loop over the two source URLs, along with the regex used to pick out the
# downloadable archives at each one
for url, pattern in zip(args.url, PARSEC_FILE_PATTERNS):

    # Fetch the directory listing for this URL
    response = urllib3.PoolManager().request('GET', url)
    if response.status != 200:
        raise RuntimeError(f"Failed to fetch PARSEC data from {url}: HTTP {response.status}")
    files_avail = re.findall(pattern, str(response.data))
    if args.verbose:
        print(f"Fetched track list from {url}: {len(files_avail)} files found.")

    # Loop over files to fetch
    for filename in files_avail:

        # Construct the full URL for the file
        file_url = url + filename
        if args.verbose:
            print(f"Fetching {file_url}...")

        # Fetch the file and redirect the output to the temporary directory
        outname = shutil.os.path.join(temp_dir, filename)
        with urllib3.PoolManager().request("GET", file_url, preload_content=False) as response, open(outname, "wb") as out_file:
            if response.status != 200:
                raise RuntimeError(f"Failed to fetch {file_url}: HTTP {response.status}")
            shutil.copyfileobj(response, out_file)

        # Unpack the zip archive
        shutil.unpack_archive(outname, temp_dir)

        # Remove the zip archive after unpacking
        shutil.os.remove(outname)

        # TODO: Loop through the unpacked track files, extract the data
        # needed by slug (following the pattern used in fetch_mist.py),
        # filter using args.feh / args.vvcrit / args.overwrite, and write
        # the results out to the HDF5 file at args.output, including the
        # PARSEC_references / PARSEC_reference_URLs metadata.

# Clean up all downloads
shutil.rmtree(temp_dir, ignore_errors=True)

# TODO: Update the registry file at args.registry with an entry for PARSEC,
# following the pattern used in fetch_mist.py: read the existing registry
# with tomlkit (or start a new one), add "PARSEC" to the "track_sets" list,
# build a table with file/version/references/reference_urls and the
# available Fe_H / v_vcrit grid values pulled from the HDF5 file, then
# write the registry back out.
