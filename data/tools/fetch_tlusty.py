"""
Script to fetch the TLUSTY OB star spectra (Hubeny et al. 2025) from the
STScI Box repository, downsample them, and write them into an HDF5 file
in the same format slug uses for BOSZ.

Box.com requires OAuth for its API -- even for publicly shared folders.
Provide a Box developer token via --box-token, or pre-download the tar
files yourself and point the script at them with --local-dir.

To get a Box developer token:
  1. Log in at https://developer.box.com/
  2. Create or open an app -> Configuration -> Developer Token -> Generate
  3. Tokens expire after 60 min; regenerate as needed.
"""

# Imports
import argparse
from collections.abc import Iterator
import gzip
import h5py
import io
import json
import math
import numpy as np
import re
import shutil
import tarfile
import tomlkit
import urllib3

# Magic strings
TLUSTY_version = "2025"
TLUSTY_URL = "https://stsci.app.box.com/v/tlustyOB2025"
TLUSTY_BOX_API = "https://api.box.com/2.0"
TLUSTY_references = [
    "Hubeny, I., Bohlin, R., Gordon, G., et al. 2025, AJ, 169, 178"
]
TLUSTY_reference_urls = [
    "https://ui.adsabs.harvard.edu/abs/2025AJ....169..178H/abstract"
]

# Default microturbulent velocity (km/s) for TLUSTY, used by SpecsynLib
# when no explicit value is requested; TLUSTY's hot, massive OB stars
# are conventionally modeled with substantial microturbulence
TLUSTY_MICRO_DEFAULT = 10

# Number of original wavelength points per spectrum, and the downsample factor.
# ceil(739791 / 100) = 7398 output points.
TLUSTY_N_ORIG     = 739_791
TLUSTY_DOWNSAMPLE = 100
TLUSTY_N_OUT      = math.ceil(TLUSTY_N_ORIG / TLUSTY_DOWNSAMPLE)   # 7398

# Parse command line arguments
parser = argparse.ArgumentParser(
    description="Fetch and process TLUSTY OB star spectral library")
parser.add_argument("--version", default=TLUSTY_version,
                    help="TLUSTY version string (default: %(default)s)")
parser.add_argument("--url", default=TLUSTY_URL,
                    help="Box shared-folder URL for TLUSTY data")
parser.add_argument("--output",
                    default=shutil.os.path.join("..", "spectra", "tlusty.h5"),
                    help="Output HDF5 file (default: %(default)s)")
parser.add_argument("--registry",
                    default=shutil.os.path.join("..", "spectra", "spectra.toml"),
                    help="Spectra registry TOML file (default: %(default)s)")
parser.add_argument("--overwrite", action="store_true",
                    help="Overwrite spectra groups already in the output file")
parser.add_argument("--feh", type=float, nargs="+", default=[],
                    help="[Fe/H] = log10(Z) values to fetch (e.g. 0.0 -0.3); "
                         "default: all available")
parser.add_argument("--micro", type=int, nargs="+", default=[],
                    help="Microturbulence values in km/s to fetch; "
                         "default: all available")
parser.add_argument("--box-token", default="",
                    help="Box OAuth developer token for downloading from Box")
parser.add_argument("--local-dir", default="",
                    help="Directory containing pre-downloaded OSTAR_z*v*.tar "
                         "files; skips Box download")
parser.add_argument("--verbose", action="store_true",
                    help="Print progress messages")
args = parser.parse_args()

if not args.box_token and not args.local_dir:
    parser.error(
        "Provide either --box-token (Box OAuth token for download) or "
        "--local-dir (directory of pre-downloaded tar files).")

# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

# Tar filename: OSTAR_z<zval>v<micro>.tar
# zval is 3 or 4 digits; micro is the microturbulence in km/s.
tar_re = re.compile(r'OSTAR_z(?P<zval>[0-9]{3,4})v(?P<micro>[0-9]+)\.tar')

# Spec filename inside each tar: z<zval>t<teff>g<gval>v<micro>.spec.gz
# gval is log(g_cgs) × 100 (e.g. 400 → log g = 4.00).
spec_re = re.compile(
    r'z(?P<zval>[0-9]{3,4})t(?P<teff>[0-9]+)g(?P<gval>[0-9]+)'
    r'v(?P<micro>[0-9]+)\.spec\.gz$'
)

def zval_to_z(zval_str: str) -> float:
    """Convert the z-string from a TLUSTY filename to a linear, solar-scaled
    metallicity Z value.

    Three-digit strings (e.g. '100') are divided by 100 to give Z.
    Four-digit strings (e.g. '0033') are divided by 1000, covering the
    Z = 1/30 ≈ 0.033 special case.
    """
    zval = int(zval_str)
    return zval / 1000.0 if len(zval_str) == 4 else zval / 100.0


def zval_to_feh(zval_str: str) -> float:
    """Convert the z-string from a TLUSTY filename to [Fe/H] = log10(Z),
    so that TLUSTY's metallicity is on the same dex scale as the [Fe/H]
    values used by other spectral libraries (e.g. BOSZ).
    """
    return math.log10(zval_to_z(zval_str))


def feh_to_zval_str(feh: float) -> str:
    """Inverse of zval_to_feh: reconstruct the filename z-string from
    [Fe/H] = log10(Z)."""
    z = 10 ** feh
    if abs(z - round(z * 100) / 100) < 1e-6:
        return f"{round(z * 100):03d}"
    return f"{round(z * 1000):04d}"


# ---------------------------------------------------------------------------
# Box API helpers (used only when --box-token is provided)
# ---------------------------------------------------------------------------

def _box_get(path: str, token: str) -> dict:
    """GET from the Box API with Bearer auth and the shared-link header."""
    http = urllib3.PoolManager()
    url = f"{TLUSTY_BOX_API}{path}"
    resp = http.request(
        "GET", url,
        headers={
            "Authorization": f"Bearer {token}",
            "BoxApi": f"shared_link={TLUSTY_URL}",
        }
    )
    if resp.status != 200:
        raise RuntimeError(
            f"Box API error {resp.status} for {url}: {resp.data[:200]}")
    return json.loads(resp.data)


def box_list_shared_folder(token: str) -> list[dict]:
    """Return all file entries in the TLUSTY shared Box folder."""
    folder = _box_get("/shared_items", token)
    folder_id = folder["id"]
    items: list[dict] = []
    offset = 0
    limit = 1000
    while True:
        page = _box_get(
            f"/folders/{folder_id}/items?limit={limit}&offset={offset}", token)
        items.extend(page["entries"])
        if offset + len(page["entries"]) >= page["total_count"]:
            break
        offset += limit
    return items


def box_download_file(file_id: str, token: str, dest_path: str) -> None:
    """Download a Box file by ID to dest_path (follows redirect to CDN)."""
    http = urllib3.PoolManager()
    url = f"{TLUSTY_BOX_API}/files/{file_id}/content"
    with http.request(
        "GET", url,
        headers={
            "Authorization": f"Bearer {token}",
            "BoxApi": f"shared_link={TLUSTY_URL}",
        },
        preload_content=False,
        redirect=True,
    ) as resp, open(dest_path, "wb") as out:
        if resp.status != 200:
            raise RuntimeError(
                f"Box download failed for file {file_id}: HTTP {resp.status}")
        shutil.copyfileobj(resp, out)


# ---------------------------------------------------------------------------
# Downsample helper
# ---------------------------------------------------------------------------

def downsample(wave: np.ndarray, flux: np.ndarray,
               n: int = TLUSTY_DOWNSAMPLE) -> tuple[np.ndarray, np.ndarray]:
    """Downsample wavelength and flux arrays by factor n.

    For each block of n values: output wavelength = geometric mean (i.e.
    arithmetic mean of log wavelengths, then exponentiated); output flux =
    arithmetic mean.  The final block may be shorter than n if the array
    length is not divisible by n -- its mean is computed over however many
    points it contains.
    """
    n_out = math.ceil(len(wave) / n)
    pad = n_out * n - len(wave)
    log_wave = np.log(wave)
    if pad:
        log_wave = np.concatenate([log_wave, np.full(pad, np.nan)])
        flux     = np.concatenate([flux,     np.full(pad, np.nan)])
    log_wave = log_wave.reshape(n_out, n)
    flux     = flux.reshape(n_out, n)
    return np.exp(np.nanmean(log_wave, axis=1)), np.nanmean(flux, axis=1)


# ---------------------------------------------------------------------------
# Discover which tar files to process
# ---------------------------------------------------------------------------

# Build a dict: (feh, micro) -> local_path_or_box_file_id
tar_sources: dict[tuple[float, int], str] = {}

if args.local_dir:
    # Scan the local directory for OSTAR_z*v*.tar files
    for fname in shutil.os.listdir(args.local_dir):
        m = tar_re.fullmatch(fname)
        if m is None:
            continue
        feh_val  = zval_to_feh(m.group("zval"))
        micro_val = int(m.group("micro"))
        if args.feh   and feh_val   not in args.feh:
            continue
        if args.micro and micro_val not in args.micro:
            continue
        tar_sources[(feh_val, micro_val)] = shutil.os.path.join(
            args.local_dir, fname)
    if args.verbose:
        print(f"Found {len(tar_sources)} tar files in {args.local_dir}: "
              f"{sorted(tar_sources.keys())}")
else:
    # Query Box API for folder listing
    if args.verbose:
        print(f"Querying Box API for folder listing...")
    for item in box_list_shared_folder(args.box_token):
        if item["type"] != "file":
            continue
        m = tar_re.fullmatch(item["name"])
        if m is None:
            continue
        feh_val  = zval_to_feh(m.group("zval"))
        micro_val = int(m.group("micro"))
        if args.feh   and feh_val   not in args.feh:
            continue
        if args.micro and micro_val not in args.micro:
            continue
        tar_sources[(feh_val, micro_val)] = item["id"]   # Box file ID
    if args.verbose:
        print(f"Found {len(tar_sources)} tar files in Box folder: "
              f"{sorted(tar_sources.keys())}")

if not tar_sources:
    raise SystemExit("No matching tar files found; check --feh / --micro filters.")

# ---------------------------------------------------------------------------
# Check for already-present (feh, micro) groups in the output file
# ---------------------------------------------------------------------------

existing: set[tuple[float, int]] = set()
if shutil.os.path.exists(args.output) and not args.overwrite:
    with h5py.File(args.output, "r") as h5:
        for grp in h5.keys():
            if not grp.startswith("spectra_"):
                continue
            attrs = h5[grp].attrs
            existing.add((float(attrs["feh"]), int(attrs["micro"])))
    keep = {k: v for k, v in tar_sources.items() if k not in existing}
    skipped = len(tar_sources) - len(keep)
    if skipped and args.verbose:
        print(f"Skipping {skipped} groups already in {args.output}.")
    tar_sources = keep

if not tar_sources:
    print("All requested spectra already present; nothing to do.")
    raise SystemExit(0)

# ---------------------------------------------------------------------------
# Create output file if it doesn't exist yet
# ---------------------------------------------------------------------------

if not shutil.os.path.exists(args.output):
    with h5py.File(args.output, "w") as h5:
        h5.attrs["references"]     = TLUSTY_references
        h5.attrs["reference_urls"] = TLUSTY_reference_urls

# ---------------------------------------------------------------------------
# Download (or locate) and process each tar file
# ---------------------------------------------------------------------------

temp_dir = "tlusty_temp"
shutil.rmtree(temp_dir, ignore_errors=True)
shutil.os.makedirs(temp_dir, exist_ok=True)

wave_ds: np.ndarray | None = None   # downsampled wavelength grid (set once)

for (feh_val, micro_val), source in sorted(tar_sources.items()):

    grp_name = f"spectra_feh{feh_val:.4g}_micro{micro_val}"
    if args.verbose:
        print(f"Processing {grp_name} ...")

    # Obtain the tar file -- either download from Box or use local path
    if args.local_dir:
        tar_path = source
    else:
        tar_fname = (f"OSTAR_z{feh_to_zval_str(feh_val)}v{micro_val}.tar")
        tar_path  = shutil.os.path.join(temp_dir, tar_fname)
        if args.verbose:
            print(f"  Downloading {tar_fname} from Box...")
        box_download_file(source, args.box_token, tar_path)

    # Extract and process every .spec.gz in the tar
    spectra: dict[tuple[int, float], np.ndarray] = {}   # (teff, logg) -> flux
    with tarfile.open(tar_path, "r") as tf:
        for member in tf.getmembers():
            bname = shutil.os.path.basename(member.name)
            m = spec_re.fullmatch(bname)
            if m is None:
                continue
            micro_in_file = int(m.group("micro"))
            if micro_in_file != micro_val:
                continue    # guard: should always match within tar
            teff  = int(m.group("teff"))
            logg  = int(m.group("gval")) / 100.0

            raw = tf.extractfile(member)
            if raw is None:
                continue
            data = np.loadtxt(gzip.open(io.BytesIO(raw.read())))
            wavelengths_orig = data[:, 0]
            eddington_h      = data[:, 1]
            flux_orig        = 4.0 * np.pi * eddington_h

            wave_block, flux_block = downsample(wavelengths_orig, flux_orig)

            # Capture the shared wavelength grid on the first spectrum seen
            if wave_ds is None:
                wave_ds = wave_block

            spectra[(teff, logg)] = flux_block
            if args.verbose:
                print(f"    Read t{teff} g{logg:+.2f} ({len(flux_block)} pts)")

    if not spectra:
        if args.verbose:
            print(f"  No spectra found in {shutil.os.path.basename(tar_path)}; skipping.")
        if not args.local_dir:
            shutil.os.remove(tar_path)
        continue

    # Write this group to the HDF5 file
    with h5py.File(args.output, "a") as h5:
        if grp_name in h5:
            del h5[grp_name]
        grp = h5.create_group(grp_name)
        grp.attrs["feh"]   = feh_val
        grp.attrs["micro"] = micro_val
        for (teff, logg), flux in sorted(spectra.items()):
            ds = grp.create_dataset(
                f"t{teff}_g{logg:+.2f}", data=flux, compression="gzip")
            ds.attrs["teff"] = teff
            ds.attrs["logg"] = logg
    if args.verbose:
        print(f"  Wrote {len(spectra)} spectra to {grp_name}.")

    # Clean up downloaded tar (keep local-dir files as-is)
    if not args.local_dir:
        shutil.os.remove(tar_path)

# ---------------------------------------------------------------------------
# Write the shared wavelength grid
# ---------------------------------------------------------------------------

if wave_ds is not None:
    with h5py.File(args.output, "a") as h5:
        wgrp = h5.require_group("wavelengths")
        # Not named after a resolution value (e.g. "r500", as BOSZ's
        # wavelength grids are): downsampling by TLUSTY_DOWNSAMPLE means
        # this grid isn't actually at the original spectra's resolution
        # any more, and there's no meaningful single "r" value to
        # attach to it, the same way there's no afe/cfe/r group
        # attribute on these spectra -- treat this grid as matching
        # any r a caller asks for, rather than claiming an r it isn't.
        ds_name = "native"
        if ds_name in wgrp:
            del wgrp[ds_name]
        wgrp.create_dataset(ds_name, data=wave_ds, compression="gzip")
        if args.verbose:
            print(f"Wrote wavelength grid '{ds_name}' ({len(wave_ds)} pts).")

# ---------------------------------------------------------------------------
# Clean up temp directory
# ---------------------------------------------------------------------------

shutil.rmtree(temp_dir, ignore_errors=True)

# ---------------------------------------------------------------------------
# Update the TOML registry
# ---------------------------------------------------------------------------

if shutil.os.path.exists(args.registry):
    with open(args.registry) as f:
        registry = tomlkit.parse(f.read())
else:
    registry = {"name": "Registry of spectra sets"}

if "spectra_sets" in registry:
    if "TLUSTY" not in registry["spectra_sets"]:
        registry["spectra_sets"].append("TLUSTY")
else:
    registry["spectra_sets"] = ["TLUSTY"]

if "TLUSTY" in registry:
    registry.pop("TLUSTY")
tab = tomlkit.table()
tab["file"]           = args.output
tab["version"]        = args.version
tab["references"]     = TLUSTY_references
tab["reference_urls"] = TLUSTY_reference_urls

with h5py.File(args.output, "r") as h5:
    grps = [g for g in h5.keys() if g.startswith("spectra_")]
    for qty, attr, cast in [("Fe_H", "feh", float), ("micro", "micro", int)]:
        vals: list = []
        for g in grps:
            v = cast(h5[g].attrs[attr])
            if v not in vals:
                vals.append(v)
        vals.sort()
        tab[qty] = vals
tab["micro_default"] = TLUSTY_MICRO_DEFAULT

registry["TLUSTY"] = tab

with open(args.registry, "w") as f:
    f.write(tomlkit.dumps(registry))
if args.verbose:
    print(f"Updated registry at {args.registry}.")
