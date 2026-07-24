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
import io
import numpy as np
import re
import shutil
import tomlkit
import urllib3
from astropy.io import fits

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

# ---------------------------------------------------------------------------
# Directory/filename conventions
# ---------------------------------------------------------------------------

# Column name for a single log(g) value within a file's binary table:
# gXX, XX two digits meaning log(g) = XX / 10 (e.g. g50 -> 5.0, g45 -> 4.5)
logg_col_re = re.compile(r'g(?P<gg>\d{2})')


def feh_to_dirname(feh: float) -> str:
    """[Fe/H] -> its ck<X><YY> directory name (X = 'm'/'p' for minus/plus,
    YY two digits meaning Y.Y -- e.g. -0.5 -> ckm05, +0.2 -> ckp02)."""
    sign = 'p' if feh >= 0.0 else 'm'
    yy = round(abs(feh) * 10.0)
    return f"ck{sign}{yy:02d}"


# This archive's actual [Fe/H] grid (verified directly against its own
# top-level directory listing -- some [Fe/H] values quoted for "the"
# CK04 grid elsewhere, e.g. -0.3/-0.2/-0.1 and +1.0, are not present
# in this particular STScI mirror), and Teff from 3500-13000 K in
# 250 K steps, then 14000-50000 K in 1000 K steps (log g is not
# filtered here at all -- every log g present in a given (feh, Teff)
# file's own binary table is read directly from its column names,
# rather than enumerated as a separate candidate axis the way BOSZ's
# file-per-point layout requires)
_ALL_FEH_VALS = [-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.2, 0.5]
_ALL_TEFF_VALS = list(range(3500, 13001, 250)) + list(range(14000, 50001, 1000))

feh_vals = args.feh if args.feh else _ALL_FEH_VALS

# ---------------------------------------------------------------------------
# Check for already-present [Fe/H] groups in the output file
# ---------------------------------------------------------------------------

existing_feh = set()
if shutil.os.path.exists(args.output) and not args.overwrite:
    with h5py.File(args.output, "r") as h5file:
        for grp in h5file.keys():
            if not grp.startswith("spectra_"):
                continue
            existing_feh.add(float(h5file[grp].attrs["feh"]))
    keep = [feh for feh in feh_vals if feh not in existing_feh]
    nskipped = len(feh_vals) - len(keep)
    if nskipped and args.verbose:
        print(f"Skipping {nskipped} [Fe/H] groups already in {args.output}.")
    feh_vals = keep

if not feh_vals:
    print("All requested [Fe/H] groups already present; nothing to do.")
    raise SystemExit(0)

# ---------------------------------------------------------------------------
# Create output file if it doesn't exist yet
# ---------------------------------------------------------------------------

if not shutil.os.path.exists(args.output):
    with h5py.File(args.output, "w") as h5file:
        h5file.attrs["references"] = CK04_references
        h5file.attrs["reference_urls"] = CK04_reference_urls

# ---------------------------------------------------------------------------
# Download and process each [Fe/H] directory
# ---------------------------------------------------------------------------

http = urllib3.PoolManager()
wave_ds: np.ndarray | None = None  # shared wavelength grid (set once)

for feh_val in sorted(feh_vals):
    dirname = feh_to_dirname(feh_val)
    grp_name = f"spectra_feh{feh_val:+.2f}"
    if args.verbose:
        print(f"Processing {grp_name} ({dirname}) ...")

    spectra: dict[tuple[int, float], np.ndarray] = {}  # (teff, logg) -> flux
    for teff in _ALL_TEFF_VALS:
        fname = f"{dirname}_{teff}.fits"
        file_url = f"{args.url}{dirname}/{fname}"
        resp = http.request("GET", file_url, preload_content=True)
        if resp.status == 404:
            continue
        if resp.status != 200:
            raise RuntimeError(f"Failed to fetch {file_url}: HTTP {resp.status}")

        if args.verbose:
            print(f"  Fetched {fname}")

        with fits.open(io.BytesIO(resp.data)) as hdul:
            # hdul[0] is a dummy, empty primary HDU; hdul[1] is the
            # binary table actually holding the wavelength grid and
            # one flux column per log(g) present for this (feh, Teff)
            table = hdul[1].data
            wave = np.asarray(table["wavelength"], dtype=float)
            if wave_ds is None:
                wave_ds = wave

            for col in table.columns.names:
                m = logg_col_re.fullmatch(col)
                if m is None:
                    continue
                logg = int(m.group("gg")) / 10.0
                # Surface flux (erg/s/cm^2/Angstrom) is stored directly,
                # unlike TLUSTY/BOSZ's Eddington H values, so no 4 pi
                # factor is needed here
                spectra[(teff, logg)] = np.asarray(table[col], dtype=float)

    if not spectra:
        if args.verbose:
            print(f"  No files found for {grp_name}; skipping.")
        continue

    with h5py.File(args.output, "a") as h5file:
        if grp_name in h5file:
            del h5file[grp_name]
        grp = h5file.create_group(grp_name)
        grp.attrs["feh"] = feh_val
        for (teff, logg), flux in sorted(spectra.items()):
            dset = grp.create_dataset(
                f"t{teff}_g{logg:+.1f}", data=flux, compression="gzip")
            dset.attrs["teff"] = teff
            dset.attrs["logg"] = logg
    if args.verbose:
        print(f"  Wrote {len(spectra)} spectra to {grp_name}.")

# ---------------------------------------------------------------------------
# Write the shared wavelength grid
# ---------------------------------------------------------------------------

if wave_ds is not None:
    with h5py.File(args.output, "a") as h5file:
        wgrp = h5file.require_group("wavelengths")
        # Not named after a resolution value (e.g. "r500", as BOSZ's
        # wavelength grids are): this library has no resolution axis
        # at all, and every file shares the one native grid -- see
        # SpecsynLibNoWind's own single-entry wavelength-grid fallback
        ds_name = "native"
        if ds_name in wgrp:
            del wgrp[ds_name]
        wgrp.create_dataset(ds_name, data=wave_ds, compression="gzip")
        if args.verbose:
            print(f"Wrote wavelength grid '{ds_name}' ({len(wave_ds)} pts).")

# ---------------------------------------------------------------------------
# Update the TOML registry
# ---------------------------------------------------------------------------

if shutil.os.path.exists(args.registry):
    with open(args.registry) as f:
        registry = tomlkit.parse(f.read())
else:
    registry = {"name": "Registry of spectra sets"}

if "spectra_sets" in registry:
    if "CK04" not in registry["spectra_sets"]:
        registry["spectra_sets"].append("CK04")
else:
    registry["spectra_sets"] = ["CK04"]

if "CK04" in registry:
    registry.pop("CK04")
tab = tomlkit.table()
tab["file"] = args.output
tab["version"] = args.version
tab["references"] = CK04_references
tab["reference_urls"] = CK04_reference_urls

with h5py.File(args.output, "r") as h5file:
    fehs_in_file = sorted({
        float(h5file[grp].attrs["feh"])
        for grp in h5file.keys() if grp.startswith("spectra_")
    })
tab["Fe_H"] = fehs_in_file

registry["CK04"] = tab

with open(args.registry, "w") as f:
    f.write(tomlkit.dumps(registry))
if args.verbose:
    print(f"Updated registry at {args.registry}.")
