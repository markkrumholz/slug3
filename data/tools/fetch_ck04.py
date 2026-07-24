"""
Script to fetch the Castelli & Kurucz (2004) ATLAS9 atmosphere model
grid ("ck04models") from the STScI archive, extract data from them,
and write them into a gzip'ed HDF5 file that slug can read. See
https://www.stsci.edu/hst/instrumentation/reference-data-for-calibration-and-tools/astronomical-catalogs/castelli-and-kurucz-atlas
for background on the grid itself.

Unlike BOSZ, this grid has no alpha/Fe, C/Fe, or microturbulence axes
-- each (Teff, log g, [Fe/H]) point has exactly one model -- so this
script (and the registry entry it writes) has no --afe/--cfe/--micro
filters or micro_default to go with them.
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
CK04_version = "2004"
CK04_URL = "https://archive.stsci.edu/hlsps/reference-atlases/cdbs/grid/ck04models/"
CK04_references = [
    "Castelli, F., & Kurucz, R. 2002, Proc. IAU Symposium 210, "
    "Astronomical Society of the Pacific, A20",
]
CK04_reference_urls = [
    "https://ui.adsabs.harvard.edu/abs/2003IAUS..210P.A20C/abstract",
    "https://www.stsci.edu/hst/instrumentation/reference-data-for-calibration-and-tools/astronomical-catalogs/castelli-and-kurucz-atlas",
]

# Parse command line arguments
parser = argparse.ArgumentParser(
    description="Fetch Castelli & Kurucz (2004) atmosphere model grid")
parser.add_argument("--version", default=CK04_version,
                    help="CK04 version string (default: %(default)s)")
parser.add_argument("--url", default=CK04_URL,
                    help="URL of the CK04 data (default: %(default)s)")
parser.add_argument("--output",
                    default=shutil.os.path.join("..", "spectra", "ck04.h5"),
                    help="Output file for the HDF5 spectra (default: %(default)s)")
parser.add_argument("--registry",
                    default=shutil.os.path.join("..", "spectra", "spectra.toml"),
                    help="Spectra registry TOML file (default: %(default)s)")
parser.add_argument("--overwrite", action="store_true",
                    help="Overwrite spectra groups already in the output file")
parser.add_argument("--feh", type=float, nargs="+", default=[],
                    help="List of [Fe/H] values to fetch; if unspecified, fetch all")
parser.add_argument("--verbose", action="store_true",
                    help="Print verbose output")
args = parser.parse_args()
