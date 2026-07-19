"""
This is a script to fetch the BOSZ spectra (https://archive.stsci.edu/hlsp/bosz)
from the offical STScI website, extract data from them, and write them into a
gzip'ed HDF5 file that slug can read.
"""

# Imports
import argparse
import shutil

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
