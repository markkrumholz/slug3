"""
Script to download Wolf-Rayet model spectra from the Potsdam Wolf-Rayet (PoWR)
group (https://www.astro.physik.uni-potsdam.de/PoWR/), and repack them into
HDF5 files that slug can read.

Downloads spectral energy distributions (SEDs) for all WNE, WNL, and WC model
grids at Galactic, LMC, SMC, and sub-SMC (Z=0.07 solar) metallicities.

The download mechanism uses two steps per grid:
  1. POST to ajax.php to trigger server-side compilation of a tar.gz containing
     all SED files for that grid.
  2. GET download.php to retrieve the compiled archive.

Each archive unpacks to a directory 'griddl-{gridname}-sed/' containing:
  - modelparameters.txt — table of T*, R_t, log L, log Mdot, v_inf, etc. for
    every model in the grid
  - {gridname}_{model-id}_sed.txt — two-column ASCII SED for each model:
      col 1: log10(wavelength / Angstrom)
      col 2: log10(F_lambda / erg cm^-2 s^-1 Angstrom^-1)  at d = 10 pc
    Values of -100.00 indicate flux too small to represent (effectively zero).
    Some models listed in modelparameters.txt have no corresponding SED (an
    empty sed.txt member); these are skipped.

Once downloaded, each archive is unpacked in memory and repacked into one of
three HDF5 files -- powr_wne.h5, powr_wnl.h5, powr_wc.h5 -- grouped by
[Fe/H] (and, for WNL, also by surface hydrogen mass fraction), following the
same conventions as fetch_bosz.py / fetch_tlusty.py. [Fe/H] is derived from
each grid's iron mass fraction X_Fe (see POWR_XFE below) relative to the
Milky Way grid of the same star type. SED fluxes, given at d = 10 pc, are
converted to luminosities (multiplied by 4 pi (10 pc)^2) before storage.
Each spectrum's own wavelength grid is stored alongside it, since (verified
empirically) no two models -- not even within the same grid -- share a
wavelength grid.
"""

import argparse
import math
import re
import shutil
import tarfile
import h5py
import numpy as np
import tomlkit
import urllib3

# Magic strings
POWR_version = "2026"
POWR_URL = "https://www.astro.physik.uni-potsdam.de/PoWR/"
POWR_references = [
    "Gräfener, G., Koesterke, L., & Hamann, W.-R. 2002, A&A, 387, 244",
    "Hamann, W.-R., & Gräfener, G. 2003, A&A, 410, 993",
    "Sander, A., Shenar, T., Hainich, R., et al. 2015, A&A, 577, A13",
    "Hamann, W.-R., & Gräfener, G. 2004, A&A, 427, 697",
    "Todt, H., Sander, A., Hainich, R., et al. 2015, A&A, 579, A75",
    "Sander, A., Hamann, W.-R., & Todt, H. 2012, A&A, 540, A144",
]
POWR_reference_urls = [
    "http://adsabs.harvard.edu/abs/2002A%26A...387..244G",
    "http://adsabs.harvard.edu/abs/2003A%26A...410..993H",
    "http://adsabs.harvard.edu/abs/2015A%26A...577A..13S",
    "http://adsabs.harvard.edu/abs/2004A%26A...427..697H",
    "http://adsabs.harvard.edu/abs/2015A%26A...579A..75T",
    "http://adsabs.harvard.edu/abs/2012A%26A...540A.144S",
]

# Grid definitions: (server_name, star_type, h_fraction, metallicity)
# star_type: 'wne' | 'wnl' | 'wc'
# h_fraction: surface hydrogen mass fraction (0.0 for WNE and WC)
# metallicity: 'mw' | 'lmc' | 'smc' | 'z007'
POWR_GRIDS: list[tuple[str, str, float, str]] = [
    # Galactic (MW) metallicity
    ("wne",              "wne", 0.00, "mw"),
    ("wnl",              "wnl", 0.20, "mw"),
    ("wnl-h50",          "wnl", 0.50, "mw"),
    ("wc",               "wc",  0.00, "mw"),
    # LMC metallicity
    ("lmc-wne",          "wne", 0.00, "lmc"),
    ("lmc-wnl-h20",      "wnl", 0.20, "lmc"),
    ("lmc-wnl-h40",      "wnl", 0.40, "lmc"),
    ("lmc-wc",           "wc",  0.00, "lmc"),
    # SMC metallicity
    ("smc-wne",          "wne", 0.00, "smc"),
    ("smc-wnl-h20",      "wnl", 0.20, "smc"),
    ("smc-wnl-h40",      "wnl", 0.40, "smc"),
    ("smc-wnl-h60",      "wnl", 0.60, "smc"),
    ("smc-wc-2021",      "wc",  0.00, "smc"),
    # sub-SMC (0.07 solar) metallicity
    ("007-wne-2015",     "wne", 0.00, "z007"),
    ("007-wnl-h20-2015", "wnl", 0.20, "z007"),
    ("007-wnl-h40-2015", "wnl", 0.40, "z007"),
    ("007-wnl-h60-2015", "wnl", 0.60, "z007"),
    ("007-wc-2021",      "wc",  0.00, "z007"),
]

# Iron mass fraction X_Fe of each (star_type, metallicity) grid, taken from
# the chemical-composition table on the PoWR Wolf-Rayet grid overview page
# (https://www.astro.physik.uni-potsdam.de/PoWR/). [Fe/H] for a grid is
# log10(X_Fe / X_Fe_MW), where X_Fe_MW is the Milky Way value *for the same
# star type* -- WC's Milky Way X_Fe (1.6e-3) differs slightly from
# WNE/WNL's (1.4e-3) because the WC grids come from a separate paper
# (Sander et al. 2012) that assumed a different solar Fe abundance, so
# using a single shared MW reference would make WC's [Fe/H] scale
# inconsistent with its own grid-to-grid Fe abundance ratios.
POWR_XFE: dict[tuple[str, str], float] = {
    ("wne", "mw"): 1.4e-3, ("wne", "lmc"): 7.0e-4,
    ("wne", "smc"): 3.0e-4, ("wne", "z007"): 9.2e-5,
    ("wnl", "mw"): 1.4e-3, ("wnl", "lmc"): 7.0e-4,
    ("wnl", "smc"): 3.0e-4, ("wnl", "z007"): 9.2e-5,
    ("wc",  "mw"): 1.6e-3, ("wc",  "lmc"): 7.0e-4,
    ("wc",  "smc"): 3.0e-4, ("wc",  "z007"): 9.2e-5,
}


def powr_feh(star_type: str, metallicity: str) -> float:
    """[Fe/H] = log10(X_Fe / X_Fe_MW) for a given (star_type, metallicity)."""
    return math.log10(POWR_XFE[(star_type, metallicity)] /
                       POWR_XFE[(star_type, "mw")])


# 1 pc in cm (IAU definition), and the resulting factor of 4 pi (10 pc)^2
# that converts a flux quoted at d = 10 pc into a luminosity
POWR_PC_CM = 3.0856775814913673e18
POWR_LUM_FACTOR = 4.0 * math.pi * (10.0 * POWR_PC_CM) ** 2

# Output HDF5 file, and spectra-registry entry name, for each star type
POWR_H5_FILES = {"wne": "powr_wne.h5", "wnl": "powr_wnl.h5", "wc": "powr_wc.h5"}
POWR_REGISTRY_NAMES = {"wne": "POWR_WNE", "wnl": "POWR_WNL", "wc": "POWR_WC"}

# Parse command line arguments
_all_grid_names = [g[0] for g in POWR_GRIDS]
parser = argparse.ArgumentParser(
    description="Download and repack PoWR Wolf-Rayet model SED archives")
parser.add_argument("--version", default=POWR_version,
                    help="PoWR version label (default: %(default)s)")
parser.add_argument("--url", default=POWR_URL,
                    help="Base URL of the PoWR website (default: %(default)s)")
parser.add_argument("--output-dir",
                    default="powr_temp",
                    help="Directory in which downloaded archives are saved, "
                         "and from which they are read for repacking "
                         "(default: %(default)s)")
parser.add_argument("--spectra-dir",
                    default=shutil.os.path.join("..", "spectra"),
                    help="Directory in which to write the output HDF5 files "
                         "powr_wne.h5, powr_wnl.h5, powr_wc.h5 "
                         "(default: %(default)s)")
parser.add_argument("--registry",
                    default=shutil.os.path.join("..", "spectra", "spectra.toml"),
                    help="Spectra registry TOML file (default: %(default)s)")
parser.add_argument("--grids", nargs="+", default=[], metavar="GRIDNAME",
                    help="Grid names to download/process (default: all). "
                         "Choices: " + ", ".join(_all_grid_names))
parser.add_argument("--overwrite", action="store_true",
                    help="Re-download archives that already exist on disk, "
                         "and overwrite spectra groups already present in "
                         "the output HDF5 files")
parser.add_argument("--keep-temp", action="store_true",
                    help="Keep each downloaded archive in --output-dir "
                         "after it has been processed, instead of the "
                         "default behavior of deleting it once its grid "
                         "has been written to the output HDF5 file")
parser.add_argument("--verbose", action="store_true",
                    help="Print progress messages")
args = parser.parse_args()

# Validate grid selection
_valid = set(_all_grid_names)
selected = set(args.grids) if args.grids else _valid
unknown = selected - _valid
if unknown:
    parser.error(
        f"Unknown grid name(s): {', '.join(sorted(unknown))}. "
        f"Available: {', '.join(_all_grid_names)}")

# Prepare output directory
shutil.os.makedirs(args.output_dir, exist_ok=True)

# Download loop
base = args.url.rstrip("/")
ajax_url = f"{base}/ajax.php"
dl_url   = f"{base}/download.php"
http = urllib3.PoolManager(
    timeout=urllib3.util.Timeout(connect=30, read=600))

for grid_name, star_type, h_frac, metallicity in POWR_GRIDS:
    if grid_name not in selected:
        continue

    out_path = shutil.os.path.join(args.output_dir, f"{grid_name}_sed_all.gz")
    if shutil.os.path.exists(out_path) and not args.overwrite:
        if args.verbose:
            print(f"Skipping {grid_name} (already exists)")
        continue

    if args.verbose:
        print(f"Requesting compilation for {grid_name} ...")

    # Step 1: ask the server to compile all SEDs for this grid into a tar.gz.
    # The server returns synchronously when done.
    resp = http.request(
        "POST", ajax_url,
        body=(f"content=griddl&grid={grid_name}"
              f"&datatype=sed&dlformat=gz").encode(),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    if resp.status != 200:
        raise RuntimeError(
            f"Compilation request failed for {grid_name}: HTTP {resp.status}")

    # The response contains: gridDlComplete('filename.gz')
    html = resp.data.decode("utf-8", errors="replace")
    m = re.search(r"gridDlComplete\('([^']+)'\)", html)
    if m is None:
        raise RuntimeError(
            f"Could not find compiled filename in ajax.php response "
            f"for {grid_name}:\n{html[:400]}")
    archive_name = m.group(1)

    if args.verbose:
        print(f"  Downloading {archive_name} ...")

    # Step 2: fetch the compiled archive
    with http.request(
        "GET", f"{dl_url}?content=file&file={archive_name}",
        preload_content=False,
    ) as dl_resp:
        if dl_resp.status != 200:
            raise RuntimeError(
                f"Archive download failed for {archive_name}: "
                f"HTTP {dl_resp.status}")
        with open(out_path, "wb") as f:
            shutil.copyfileobj(dl_resp, f)

    if args.verbose:
        size_kb = shutil.os.path.getsize(out_path) // 1024
        print(f"  Saved to {out_path} ({size_kb} KB)")

# ---------------------------------------------------------------------------
# Parsing helpers for the contents of a downloaded archive
# ---------------------------------------------------------------------------

def parse_modelparameters(text: str) -> dict[str, dict[str, float]]:
    """Parse a griddl modelparameters.txt file into a dict mapping each
    model's name (e.g. "12-12") to a dict of its parameters: log_teff,
    log_rt, mass, logg, logl, logmdot, vinf, r23, t23 -- corresponding to
    the file's T_EFF, R_TRANS, MASS, LOG_G, LOG_L, LOG_MDOT, V_INF, R_23,
    T_23 columns, respectively. T_EFF and R_TRANS are converted to log10
    here, since PoWR's grid -- confirmed by the uniform 0.05 dex spacing
    of T_EFF and 0.1 dex spacing of R_TRANS within a grid -- is a regular
    tensor grid in (log_teff, log_rt), not in (T_eff, R_trans)
    themselves, and interpolation wants the coordinates that are
    actually regular.
    """
    lines = text.splitlines()
    hdr = next(i for i, line in enumerate(lines)
               if line.strip().startswith("MODEL"))
    keys = ["log_teff", "log_rt", "mass", "logg", "logl",
            "logmdot", "vinf", "r23", "t23"]
    models = {}
    for line in lines[hdr + 2:]:
        parts = line.split()
        if len(parts) != len(keys) + 1:
            continue
        vals = [float(v) for v in parts[1:]]
        vals[0] = math.log10(vals[0])    # T_EFF [K] -> log10(T_*)
        vals[1] = math.log10(vals[1])    # R_TRANS [R_sun] -> log10(R_t)
        models[parts[0]] = dict(zip(keys, vals))
    return models


def parse_sed(text: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse the contents of a griddl *_sed.txt file into
    (log10(wave/Angstrom), log10(F_lambda)) arrays.

    Columns are fixed-width Fortran fields (log wavelength in characters
    0:14, log flux in characters 14:28) rather than whitespace-separated:
    when the wavelength is small enough to need scientific notation, its
    field fills all 14 characters with no separating space before the
    (always-present, always-negative) flux field, so a naive
    whitespace/np.loadtxt split misreads the two glued numbers as one.
    """
    log_wave = []
    log_flux = []
    for line in text.splitlines():
        if not line.strip():
            continue
        log_wave.append(float(line[:14]))
        log_flux.append(float(line[14:28]))
    return np.array(log_wave), np.array(log_flux)


def process_grid(archive_path: str, grid_name: str,
                  h5group: "h5py.Group") -> int:
    """Unpack one downloaded griddl archive and write each of its models'
    spectra as a dataset in h5group, converting fluxes (quoted at
    d = 10 pc in the source files) to luminosities and storing each
    spectrum's own wavelength grid alongside it. Returns the number of
    spectra written.
    """
    prefix = f"griddl-{grid_name}-sed/"
    with tarfile.open(archive_path, "r:gz") as tf:
        members = {m.name: m for m in tf.getmembers()}
        params_text = tf.extractfile(
            members[prefix + "modelparameters.txt"]).read().decode(
            "utf-8", errors="replace")
        models = parse_modelparameters(params_text)

        n_written = 0
        for model_id, params in models.items():
            member = members.get(f"{prefix}{grid_name}_{model_id}_sed.txt")
            if member is None or member.size == 0:
                continue    # no SED computed for this model; skip it

            raw = tf.extractfile(member).read().decode(
                "utf-8", errors="replace")
            log_wave, log_flux = parse_sed(raw)
            if log_wave.size == 0:
                continue

            wave = 10.0 ** log_wave
            lum  = 10.0 ** log_flux * POWR_LUM_FACTOR

            ds_name = f"logt{params['log_teff']:.4g}_logrt{params['log_rt']:.4g}"
            ds = h5group.create_dataset(
                ds_name, data=lum, compression="gzip")
            for key, val in params.items():
                ds.attrs[key] = val
            h5group.create_dataset(
                f"{ds_name}_wave", data=wave, compression="gzip")
            n_written += 1
    return n_written


# ---------------------------------------------------------------------------
# Repack every downloaded archive into the three PoWR HDF5 files
# ---------------------------------------------------------------------------

shutil.os.makedirs(args.spectra_dir, exist_ok=True)

for fname in POWR_H5_FILES.values():
    h5_path = shutil.os.path.join(args.spectra_dir, fname)
    if not shutil.os.path.exists(h5_path):
        with h5py.File(h5_path, "w") as h5:
            h5.attrs["references"] = POWR_references
            h5.attrs["reference_urls"] = POWR_reference_urls

for grid_name, star_type, h_frac, metallicity in POWR_GRIDS:
    if grid_name not in selected:
        continue

    archive_path = shutil.os.path.join(
        args.output_dir, f"{grid_name}_sed_all.gz")
    if not shutil.os.path.exists(archive_path):
        if args.verbose:
            print(f"Archive for {grid_name} not found at {archive_path}; "
                  f"skipping processing.")
        continue

    feh = powr_feh(star_type, metallicity)
    if star_type == "wnl":
        grp_name = f"spectra_feh{feh:.4g}_xh{h_frac:.2f}"
    else:
        grp_name = f"spectra_feh{feh:.4g}"

    h5_path = shutil.os.path.join(args.spectra_dir, POWR_H5_FILES[star_type])
    with h5py.File(h5_path, "a") as h5:
        already_done = grp_name in h5
        if already_done and not args.overwrite:
            if args.verbose:
                print(f"Group {grp_name} already in {h5_path}; "
                      f"skipping {grid_name}.")
        else:
            if already_done:
                del h5[grp_name]

            grp = h5.create_group(grp_name)
            grp.attrs["feh"] = feh
            if star_type == "wnl":
                grp.attrs["xh"] = h_frac

            if args.verbose:
                print(f"Processing {grid_name} -> {h5_path}:{grp_name} ...")
            n_written = process_grid(archive_path, grid_name, grp)
            if args.verbose:
                print(f"  Wrote {n_written} spectra to {grp_name}.")

    # This grid's spectra are now safely written to the output HDF5 file
    # (whether just now or in an earlier run), so the downloaded archive
    # is no longer needed -- delete it unless the user asked to keep it.
    if not args.keep_temp:
        shutil.os.remove(archive_path)
        if args.verbose:
            print(f"  Removed {archive_path}.")

# ---------------------------------------------------------------------------
# Update the TOML registry
# ---------------------------------------------------------------------------

if shutil.os.path.exists(args.registry):
    with open(args.registry) as f:
        registry = tomlkit.parse(f.read())
else:
    registry = {"name": "Registry of spectra sets"}

if "spectra_sets" in registry:
    for reg_name in POWR_REGISTRY_NAMES.values():
        if reg_name not in registry["spectra_sets"]:
            registry["spectra_sets"].append(reg_name)
else:
    registry["spectra_sets"] = list(POWR_REGISTRY_NAMES.values())

# Unlike BOSZ/TLUSTY, the WR grids have no afe/cfe/r/micro axes, so those
# keys are omitted here. WR_grid marks these entries as needing the
# forthcoming WR-specific Specsyn parser, since WR atmospheres --
# parameterized by (log T_*, log R_t) rather than (Teff, logg) -- are
# interpolated in fundamentally different variables than stars without
# optically thick winds.
for star_type, reg_name in POWR_REGISTRY_NAMES.items():
    h5_path = shutil.os.path.join(args.spectra_dir, POWR_H5_FILES[star_type])
    if not shutil.os.path.exists(h5_path):
        continue

    if reg_name in registry:
        registry.pop(reg_name)
    tab = tomlkit.table()
    tab["file"] = h5_path
    tab["version"] = args.version
    tab["references"] = POWR_references
    tab["reference_urls"] = POWR_reference_urls
    with h5py.File(h5_path, "r") as h5:
        feh_vals = sorted({float(h5[grp].attrs["feh"])
                            for grp in h5.keys() if grp.startswith("spectra_")})
    tab["Fe_H"] = feh_vals
    tab["WR_grid"] = True
    registry[reg_name] = tab

with open(args.registry, "w") as f:
    f.write(tomlkit.dumps(registry))
if args.verbose:
    print(f"Updated registry at {args.registry}.")

if args.verbose:
    print("Done.")
