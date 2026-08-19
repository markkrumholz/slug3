"""
Script to fetch the Tremblay et al. pure-hydrogen (DA) white dwarf
atmosphere grids from the University of Warwick model grid page, and
pack them into an HDF5 file as four flat datasets -- wl, logg,
log_Teff, and a (n_logg, n_logTeff, n_wl) flux tensor -- rather than
the usual one-group-per-[Fe/H] layout other spectral libraries use,
since these grids have no [Fe/H]/[alpha/Fe]/[C/Fe] axis at all and are
already a single, uniform (log g, Teff, wavelength) tensor as
distributed.

Two separate grids are available, fetched independently via
--grid-type:

  "ir"  -- the standard grid: log(g) = 6.5-9.5 in steps of 0.5 dex
           (7 values), Teff = 1500-140000 K (60 values, non-uniformly
           spaced), 2711 wavelength points.
  "elm" -- the extremely low-mass grid: log(g) = 4.0-9.5 in steps of
           0.25 dex (23 values), Teff = 4000-40000 K (36 values),
           3747 wavelength points.

Each grid is a single tar(.gz) archive containing one text file per
log(g) (verified, for both grids, to all share one common wavelength
grid and one common Teff list across every log(g) file in the
archive, and for every Teff block to hold exactly as many flux values
as the wavelength grid has points -- so no interpolation or per-file
bookkeeping is needed, just a per-(log g, Teff) dict keyed as each
file is read). Each file holds a header line giving that wavelength
grid's point count, the wavelength grid itself (in Angstrom), and
then, for every Teff in the grid, a header line ("Effective
temperature = ... gravity = ... y = ...") followed by that model's
Eddington flux (in erg/s/cm^2/Hz) on the shared wavelength grid. Some
of the smallest flux values (typically deep in the UV for cool
models) are written in Fortran's fixed-width notation with the "E"
dropped before a 3-digit exponent (e.g. "0.25697-100" instead of
"0.25697E-100"); see _fix_missing_exponent().

Unlike slug's other spectral-library fetch scripts, these models have
no [Fe/H], [alpha/Fe], or [C/Fe] axis at all -- they are pure-hydrogen
(DA) models parameterized by log(g) and Teff alone -- so this script
takes no --feh/--afe/--cfe arguments.

References
----------
Main references (grid_ir):
  Tremblay, P.-E., Bergeron, P. & Gianninas, A. (2011) ApJ, 730, 128.
  Kowalski P. M., Saumon D., 2006, ApJL, 651, L137.

Main reference (grid_elm_new):
  Claret A., Cukanovaite E., Burdge K., Tremblay P.-E. et al. (2020)
  A&A, 634, A93.

:copyright: Copyright (c) 2026 Mark Krumholz
"""

# Imports
import argparse
import astropy.units as u
import h5py
import math
import numpy as np
import re
import shutil
import tarfile
import tomlkit
import urllib3

# Source URLs: each grid is a single tar(.gz) archive, plus a
# plain-text readme describing its format and giving its own
# reference list (reproduced in TREMBLAY_GRID_TYPES below rather than
# parsed at run time)
TREMBLAY_BASE_URL = ("https://warwick.ac.uk/fac/sci/physics/research/"
                      "astro/people/tremblay/modelgrids")
TREMBLAY_IR_URL   = f"{TREMBLAY_BASE_URL}/grid_ir.tar"
TREMBLAY_ELM_URL  = f"{TREMBLAY_BASE_URL}/grid_elm_new.tar.gz"
TREMBLAY_IR_README_URL  = f"{TREMBLAY_BASE_URL}/readme.txt"
TREMBLAY_ELM_README_URL = f"{TREMBLAY_BASE_URL}/readme_elm.txt"

# References shared by both grids (line-profile physics) plus each
# grid's own main reference(s), as given by its own readme's "Main
# references" section
TREMBLAY_2011 = "Tremblay, P.-E., Bergeron, P. & Gianninas, A. (2011) ApJ, 730, 128"
KOWALSKI_2006 = "Kowalski P. M., Saumon D., 2006, ApJL, 651, L137"
CLARET_2020   = "Claret A., Cukanovaite E., Burdge K., Tremblay P.-E. et al. (2020) A&A, 634, A93"

TREMBLAY_2011_URL = "https://ui.adsabs.harvard.edu/abs/2011ApJ...730..128T/abstract"
KOWALSKI_2006_URL = "https://ui.adsabs.harvard.edu/abs/2006ApJ...651L.137K/abstract"
CLARET_2020_URL   = "https://ui.adsabs.harvard.edu/abs/2020A%26A...634A..93C/abstract"

# Per-grid-type settings: source URL, output HDF5 file, registry entry
# name, this grid's own main reference(s), and a version string (the
# publication year of that grid's own main reference) -- verified
# against both archives' actual member files: grid_ir.tar extracts
# (under a grid_IR/ subdirectory) into one "<100*logg>_LyA_IR" file
# per log(g), and grid_elm_new.tar.gz extracts (flat, no
# subdirectory) into one "<100*logg>_ELM_new" file per log(g); this
# script does not rely on either naming scheme, since it just treats
# every regular file found in the extracted archive as a grid file to
# parse (see main loop below).
TREMBLAY_GRID_TYPES = {
    "ir": {
        "url": TREMBLAY_IR_URL,
        "output": shutil.os.path.join("..", "..", "spectra", "tremblay_da.h5"),
        "registry_name": "TREMBLAY_DA",
        "references": [TREMBLAY_2011, KOWALSKI_2006],
        "reference_urls": [TREMBLAY_2011_URL, KOWALSKI_2006_URL],
        "version": "2011",
    },
    "elm": {
        "url": TREMBLAY_ELM_URL,
        "output": shutil.os.path.join("..", "..", "spectra", "tremblay_elm.h5"),
        "registry_name": "TREMBLAY_ELM",
        "references": [CLARET_2020],
        "reference_urls": [CLARET_2020_URL],
        "version": "2020",
    },
}

# ---------------------------------------------------------------------------
# Parse command line arguments
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Fetch and process a Tremblay et al. white dwarf atmosphere grid")
parser.add_argument("--grid-type", choices=sorted(TREMBLAY_GRID_TYPES), required=True,
                    help="Which grid to fetch: 'ir' for the standard grid "
                         "(-> tremblay_da.h5/TREMBLAY_DA) or 'elm' for the "
                         "extremely low-mass grid (-> tremblay_elm.h5/TREMBLAY_ELM)")
parser.add_argument("--url", default="",
                    help="URL of the grid's tar(.gz) archive (default: chosen by --grid-type)")
parser.add_argument("--version", default="",
                    help="Version string for the registry entry (default: chosen by --grid-type)")
parser.add_argument("--output", default="",
                    help="Output HDF5 file (default: chosen by --grid-type)")
parser.add_argument("--registry",
                    default=shutil.os.path.join("..", "..", "spectra", "spectra.toml"),
                    help="Spectra registry TOML file (default: %(default)s)")
parser.add_argument("--overwrite", action="store_true",
                    help="Overwrite the output HDF5 file and registry entry if already present")
parser.add_argument("--local-file", default="",
                    help="Path to an already-downloaded grid_ir.tar or "
                         "grid_elm_new.tar.gz (matching --grid-type); skips "
                         "the HTTP download")
parser.add_argument("--verbose", action="store_true",
                    help="Print progress messages")
args = parser.parse_args()

grid_cfg = TREMBLAY_GRID_TYPES[args.grid_type]
if not args.url:
    args.url = grid_cfg["url"]
if not args.version:
    args.version = grid_cfg["version"]
if not args.output:
    args.output = grid_cfg["output"]

if shutil.os.path.exists(args.output) and not args.overwrite:
    raise SystemExit(
        f"{args.output} already exists; pass --overwrite to regenerate it.")

# ---------------------------------------------------------------------------
# Text-file parsing
# ---------------------------------------------------------------------------

# Fortran's fixed-width output drops the "E" before a 3-digit exponent
# to keep the field width constant (e.g. "0.25697-100" instead of
# "0.25697E-100"), which happens for the smallest fluxes (deep UV,
# cool models). The "E" is always still present when the exponent
# fits in 2 digits (e.g. "0.29155E-97"), so requiring a digit --
# rather than the "E" a well-formed token would have -- immediately
# before the sign is enough to target only the broken tokens.
_MISSING_EXPONENT_RE = re.compile(r'(\d)([+-]\d{2,})(?!\d)')


def _fix_missing_exponent(line: str) -> str:
    return _MISSING_EXPONENT_RE.sub(r'\1E\2', line)


_HEADER_RE = re.compile(
    r'Effective temperature\s*=\s*([\d.]+)\s*gravity\s*=\s*([\d.eE+-]+)\s*y\s*=\s*([\d.eE+-]+)')


def parse_grid_file(path: str, verbose: bool = False):
    """Parse one Tremblay grid text file (one log(g), every Teff).

    Returns (wl, logg, {teff: flux}), where wl is the shared
    wavelength grid (Angstrom, length n_wl), logg is this file's
    single log(g) value (derived from the header's "gravity" in
    cgs units, not from the filename), and the dict maps each Teff
    (K) in this file to its raw Eddington flux (erg/s/cm^2/Hz, length
    n_wl, still needing the 4*pi and Fnu->Flambda conversions applied
    by the caller).
    """
    with open(path) as f:
        lines = f.readlines()

    n_wl = int(lines[0].strip())
    ptr = 1
    buf: list[str] = []
    while len(buf) < n_wl:
        buf.extend(_fix_missing_exponent(lines[ptr]).split())
        ptr += 1
    if len(buf) != n_wl:
        raise RuntimeError(
            f"{path}: wavelength grid has {len(buf)} values, header said {n_wl}")
    wl = np.array([float(x) for x in buf])

    logg = None
    fluxes: dict[float, np.ndarray] = {}
    while ptr < len(lines):
        line = lines[ptr]
        if not line.strip():
            ptr += 1
            continue
        m = _HEADER_RE.search(line)
        if m is None:
            raise RuntimeError(f"{path}: expected a model header at line {ptr + 1}, got: {line!r}")
        teff = float(m.group(1))
        gravity = float(m.group(2))
        this_logg = math.log10(gravity)
        if logg is None:
            logg = this_logg
        elif not math.isclose(logg, this_logg, abs_tol=1e-6):
            raise RuntimeError(
                f"{path}: log(g) changes within one file ({logg} vs {this_logg})")
        ptr += 1

        buf = []
        while ptr < len(lines) and 'Effective temperature' not in lines[ptr]:
            buf.extend(_fix_missing_exponent(lines[ptr]).split())
            ptr += 1
        if len(buf) != n_wl:
            raise RuntimeError(
                f"{path}: Teff={teff} block has {len(buf)} flux values, "
                f"expected {n_wl}")
        fluxes[teff] = np.array([float(x) for x in buf])
        if verbose:
            print(f"    Read Teff={teff} log(g)={logg:.2f} ({len(buf)} pts)")

    return wl, logg, fluxes


def fnu_to_flambda(flux_nu: np.ndarray, wl_angstrom: np.ndarray) -> np.ndarray:
    """Convert flux per unit frequency (erg/s/cm^2/Hz) to flux per unit
    wavelength in Angstrom (erg/s/cm^2/Angstrom), broadcasting
    wl_angstrom (shape (n_wl,)) against flux_nu's own last axis."""
    flux_nu_q = flux_nu * u.erg / u.s / u.cm**2 / u.Hz
    wl_q = wl_angstrom * u.Angstrom
    flux_lambda_q = flux_nu_q.to(
        u.erg / u.s / u.cm**2 / u.Angstrom,
        equivalencies=u.spectral_density(wl_q))
    return flux_lambda_q.value


# ---------------------------------------------------------------------------
# Obtain the tar(.gz) archive -- either a pre-downloaded --local-file, or a
# fresh download to a temp file
# ---------------------------------------------------------------------------

temp_dir = "tremblay_temp"
shutil.rmtree(temp_dir, ignore_errors=True)
shutil.os.makedirs(temp_dir, exist_ok=True)

if args.local_file:
    tar_path = args.local_file
else:
    tar_path = shutil.os.path.join(temp_dir, shutil.os.path.basename(args.url))
    if args.verbose:
        print(f"Downloading {args.url} ...")
    http = urllib3.PoolManager(timeout=urllib3.util.Timeout(connect=30, read=600))
    with http.request("GET", args.url, preload_content=False, retries=3) as resp, \
         open(tar_path, "wb") as out_file:
        if resp.status != 200:
            raise RuntimeError(f"Failed to fetch {args.url}: HTTP {resp.status}")
        shutil.copyfileobj(resp, out_file)

# ---------------------------------------------------------------------------
# Extract and parse every member file: tarfile.open's "r:*" mode
# auto-detects plain vs gzip'ed tar, and every regular file the
# archive contains is one of this grid's own per-log(g) grid files
# (neither archive bundles anything else), regardless of whether it
# extracts flat or under a subdirectory
# ---------------------------------------------------------------------------

extract_dir = shutil.os.path.join(temp_dir, "extracted")
if args.verbose:
    print(f"Extracting {tar_path} ...")
with tarfile.open(tar_path, "r:*") as tf:
    tf.extractall(extract_dir, filter="data")

member_paths = [
    shutil.os.path.join(dirpath, fname)
    for dirpath, _, fnames in shutil.os.walk(extract_dir)
    for fname in fnames
]
if not member_paths:
    raise SystemExit(f"No files found in {tar_path}.")

wl_ref: np.ndarray | None = None
teff_set: set[float] = set()
by_logg: dict[float, dict[float, np.ndarray]] = {}
for path in sorted(member_paths):
    if args.verbose:
        print(f"  Parsing {shutil.os.path.basename(path)} ...")
    wl, logg, fluxes = parse_grid_file(path, verbose=args.verbose)
    if wl_ref is None:
        wl_ref = wl
        teff_set = set(fluxes)
    else:
        if wl.shape != wl_ref.shape or not np.allclose(wl, wl_ref):
            raise RuntimeError(
                f"{path}: wavelength grid differs from the first file parsed")
        if set(fluxes) != teff_set:
            raise RuntimeError(f"{path}: Teff list differs from the first file parsed")
    if logg in by_logg:
        raise RuntimeError(f"{path}: log(g)={logg} already seen in another file")
    by_logg[logg] = fluxes

shutil.rmtree(temp_dir, ignore_errors=True)

# ---------------------------------------------------------------------------
# Assemble the (n_logg, n_logTeff, n_wl) flux tensor
# ---------------------------------------------------------------------------

logg_keys = sorted(by_logg)
teff_vals = np.array(sorted(teff_set))
log_teff_vals = np.log10(teff_vals)

flux_nu = np.empty((len(logg_keys), len(teff_vals), len(wl_ref)))
for i, logg in enumerate(logg_keys):
    for j, teff in enumerate(teff_vals):
        flux_nu[i, j, :] = by_logg[logg][teff]

# log10(gravity) carries a little floating-point noise from the
# header's own limited-precision "gravity" field (e.g. "3.162E+06"
# for what is meant to be exactly log(g) = 6.5, or "1.778E+04" for
# 4.25 -- off by up to ~7e-5 dex); round it away, to 2 decimals (finer
# than either grid's own 0.25/0.5 dex step), now that logg_keys has
# served its purpose for indexing into by_logg above, rather than
# storing e.g. 6.49996186559619 in the output file.
logg_vals = np.round(np.array(logg_keys), 2)

if args.verbose:
    print("Converting Eddington flux to surface F_lambda ...")
flux_lambda = fnu_to_flambda(4.0 * np.pi * flux_nu, wl_ref)

# ---------------------------------------------------------------------------
# Write the HDF5 file: four flat datasets, no per-[Fe/H] groups
# ---------------------------------------------------------------------------

with h5py.File(args.output, "w") as h5file:
    h5file.attrs["references"] = grid_cfg["references"]
    h5file.attrs["reference_urls"] = grid_cfg["reference_urls"]
    h5file.create_dataset("wl", data=wl_ref, compression="gzip")
    h5file.create_dataset("logg", data=logg_vals, compression="gzip")
    h5file.create_dataset("log_Teff", data=log_teff_vals, compression="gzip")
    h5file.create_dataset("flux", data=flux_lambda, compression="gzip")
if args.verbose:
    print(f"Wrote {args.output}: {flux_lambda.shape} "
          f"(n_logg, n_logTeff, n_wl) flux tensor.")

# ---------------------------------------------------------------------------
# Update the TOML registry
# ---------------------------------------------------------------------------

if shutil.os.path.exists(args.registry):
    with open(args.registry) as f:
        registry = tomlkit.parse(f.read())
else:
    registry = {"name": "Registry of spectra sets"}

registry_name = grid_cfg["registry_name"]

if "spectra_sets" in registry:
    if registry_name not in registry["spectra_sets"]:
        registry["spectra_sets"].append(registry_name)
else:
    registry["spectra_sets"] = [registry_name]

if registry_name in registry:
    registry.pop(registry_name)
tab = tomlkit.table()
tab["file"] = shutil.os.path.basename(args.output)
tab["version"] = args.version
tab["references"] = grid_cfg["references"]
tab["reference_urls"] = grid_cfg["reference_urls"]
# Tells SpecsynLibChained to build this entry as a SpecsynLibWD (the
# flat, 4-dataset (logg, Teff)-only schema this script writes), rather
# than the default SpecsynLibNoWind -- analogous to WR_grid, which
# selects SpecsynLibWR instead
tab["WD_grid"] = True
tab["logg"] = [float(x) for x in logg_vals]
tab["log_Teff"] = [float(x) for x in log_teff_vals]
registry[registry_name] = tab

with open(args.registry, "w") as f:
    f.write(tomlkit.dumps(registry))
if args.verbose:
    print(f"Updated registry at {args.registry}.")
