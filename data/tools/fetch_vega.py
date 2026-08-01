"""
Script to fetch the CALSPEC standard Vega reference spectrum (Bohlin
et al. 2020, 2025) from the STScI archive and write it into a
gzip'ed HDF5 file (data/spectra/vega.h5 by default) that slug can
read, for use in converting photometry to Vega magnitudes.

Unlike the other fetch_*.py scripts in this directory, this is not a
stellar atmosphere model grid: it is a single reference spectrum, so
there is no spectra.toml registry entry to update, and no [Fe/H]/Teff/
log g axes to iterate over.
"""

# Imports
import argparse
import h5py
import numpy as np
import shutil
import urllib3
from astropy.io import fits

# Magic strings
VEGA_URL = ("https://archive.stsci.edu/hlsps/reference-atlases/cdbs/"
           "current_calspec/alpha_lyr_mod_005.fits")
VEGA_references = [
    "Bohlin, R., Hubeny, I., & Rauch, T., 2020, AJ, 160, 21",
    "Bohlin, R., Deustua, S., Narayanan, G., et al. 2025, AJ, 169, 40",
]
VEGA_reference_urls = [
    "https://ui.adsabs.harvard.edu/abs/2020AJ....160...21B/abstract",
    "https://ui.adsabs.harvard.edu/abs/2025AJ....169...40B/abstract",
]

# Parse command line arguments
parser = argparse.ArgumentParser(
    description="Fetch the CALSPEC standard Vega reference spectrum")
parser.add_argument("--url", default=VEGA_URL,
                    help="URL of the CALSPEC Vega FITS file (default: %(default)s)")
parser.add_argument("--output",
                    default=shutil.os.path.join("..", "spectra", "vega.h5"),
                    help="Output file for the HDF5 Vega spectrum (default: %(default)s)")
parser.add_argument("--overwrite", action="store_true",
                    help="Overwrite --output if it already exists")
parser.add_argument("--verbose", action="store_true",
                    help="Print verbose output")
args = parser.parse_args()

if shutil.os.path.exists(args.output) and not args.overwrite:
    print(f"{args.output} already exists; use --overwrite to refetch. Nothing to do.")
    raise SystemExit(0)

shutil.os.makedirs(shutil.os.path.dirname(args.output) or ".", exist_ok=True)

# ---------------------------------------------------------------------------
# Download the FITS file to a local temporary copy
# ---------------------------------------------------------------------------

fits_path = shutil.os.path.join(
    shutil.os.path.dirname(args.output) or ".", shutil.os.path.basename(args.url))

if args.verbose:
    print(f"Downloading {args.url} ...")
http = urllib3.PoolManager()
resp = http.request("GET", args.url, preload_content=True)
if resp.status != 200:
    raise RuntimeError(f"Failed to fetch {args.url}: HTTP {resp.status}")
with open(fits_path, "wb") as f:
    f.write(resp.data)
if args.verbose:
    print(f"Wrote {fits_path} ({len(resp.data)} bytes).")

# ---------------------------------------------------------------------------
# Extract the spectrum and write it to the output HDF5 file, cleaning
# up the downloaded FITS file afterward regardless of outcome
# ---------------------------------------------------------------------------

try:
    # hdul[0] is an empty primary HDU; hdul[1] (the second HDU) is the
    # binary table holding the 'wavelength' (Angstrom) and 'flux'
    # (erg/s/cm^2/Angstrom) columns
    with fits.open(fits_path) as hdul:
        table = hdul[1].data
        wavelength = np.asarray(table["wavelength"], dtype=float)
        flux = np.asarray(table["flux"], dtype=float)

    if args.verbose:
        print(f"Read {len(wavelength)} points from {fits_path}.")

    with h5py.File(args.output, "w") as h5file:
        h5file.attrs["source"] = args.url
        h5file.attrs["references"] = VEGA_references
        h5file.attrs["reference_urls"] = VEGA_reference_urls

        wl_ds = h5file.create_dataset("wl", data=wavelength, compression="gzip")
        wl_ds.attrs["units"] = "Angstrom"
        flux_ds = h5file.create_dataset("flux", data=flux, compression="gzip")
        flux_ds.attrs["units"] = "erg/s/cm^2/Angstrom"

    print(f"Wrote {args.output}.")
finally:
    if shutil.os.path.exists(fits_path):
        shutil.os.remove(fits_path)
        if args.verbose:
            print(f"Removed {fits_path}.")
