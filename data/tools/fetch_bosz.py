"""
This is a script to fetch the BOSZ spectra (https://archive.stsci.edu/hlsp/bosz)
from the offical STScI website, extract data from them, and write them into a
gzip'ed HDF5 file that slug can read.
"""

# Imports
import argparse

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
