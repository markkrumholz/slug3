"""
Script to download Wolf-Rayet model spectra from the Potsdam Wolf-Rayet (PoWR)
group (https://www.astro.physik.uni-potsdam.de/PoWR/).

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

Run a separate processing script to convert the raw archives to HDF5.
"""

import argparse
import re
import shutil
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

# Parse command line arguments
_all_grid_names = [g[0] for g in POWR_GRIDS]
parser = argparse.ArgumentParser(
    description="Download PoWR Wolf-Rayet model SED archives")
parser.add_argument("--version", default=POWR_version,
                    help="PoWR version label (default: %(default)s)")
parser.add_argument("--url", default=POWR_URL,
                    help="Base URL of the PoWR website (default: %(default)s)")
parser.add_argument("--output-dir",
                    default="powr_temp",
                    help="Directory in which to save downloaded archives "
                         "(default: %(default)s)")
parser.add_argument("--grids", nargs="+", default=[], metavar="GRIDNAME",
                    help="Grid names to download (default: all). "
                         "Choices: " + ", ".join(_all_grid_names))
parser.add_argument("--overwrite", action="store_true",
                    help="Re-download archives that already exist on disk")
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

if args.verbose:
    print("Done.")
