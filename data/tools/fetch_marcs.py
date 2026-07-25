"""
Script to fetch the MARCS (Gustafsson et al. 2008) stellar atmosphere
model grid, extract spectra from it, and write them into a gzip'ed
HDF5 file that slug can read. See https://marcs.astro.uu.se/data.html
for background on the grid and its available downloads.

Unlike BOSZ/CK04/PoWR, MARCS is not organized so that individual
(Teff, log g, [Fe/H], ...) models can be downloaded on their own --
the whole grid is distributed as a handful of large tarballs, one per
composition variant (see MARCS_COMPOSITIONS below). This script
therefore has no --feh/--afe/--cfe-style per-model filters, just a
choice of which composition tarball(s) to fetch.
"""

# Imports
import argparse
import glob
import gzip
import h5py
import io
import math
import numpy as np
import re
import shutil
import tarfile
import tomlkit
import urllib3

# Magic strings
MARCS_version = "2008"
MARCS_URL = "https://marcs.astro.uu.se/data/"
# Unlike BOSZ/CK04/PoWR, MARCS keeps its wavelength grid in a single
# file entirely separate from the flux tarballs above (and not even
# under MARCS_URL's own data/ directory), shared by every spectrum in
# the grid -- so it needs its own URL, fetched once alongside them.
MARCS_WAVELENGTH_URL = "https://marcs.astro.uu.se/flx_wavelengths.vac.gz"
MARCS_references = [
    "Gustafsson, E., Edvardsson, B., Eriksson, K., et al. 2008, A&A, 486, 951",
]
MARCS_reference_urls = [
    "https://ui.adsabs.harvard.edu/abs/2008A%26A...486..951G/abstract",
]

# The MARCS grid's composition variants, each its own tarball at
# MARCS_URL (e.g. "standard" -> marcs_st_flx.tar). Maps each variant's
# descriptive --composition choice to the two-letter code MARCS uses
# in its own "marcs_<code>_flx.tar" filenames. Only the "_flx.tar"
# archives are used here -- unlike the "_opa.tar" (opacity),
# "_mod.tar"/"_krz.tar" (model structure), and "marcs_sun*"
# (solar-specific) archives also offered at MARCS_URL, these are the
# ones that actually hold flux spectra.
MARCS_COMPOSITIONS = {
    "standard": "st",
    "alpha-enhanced": "ae",
    "alpha-poor": "ap",
    "alpha-negative": "an",
    "mildly-cn-cycled": "mc",
    "heavily-cn-cycled": "hc",
}

# Default downsample factor, as with TLUSTY: MARCS's native wavelength
# grid has ~10^5 points, far more than slug needs, so spectra are
# downsampled by this factor (output has ceil(N_orig / factor) points)
# before being written out.
MARCS_DOWNSAMPLE = 20

# MARCS's own grid documentation suggests 1 km/s for dwarfs and 2 km/s
# for giants, rather than one value for the whole grid -- but in
# practice this library is mainly a backup for BOSZ, which already
# covers dwarfs well, so MARCS mostly ends up covering giants instead,
# making 2 km/s the more representative default here.
MARCS_MICRO_DEFAULT = 2

# Parse command line arguments
parser = argparse.ArgumentParser(
    description="Fetch the MARCS stellar atmosphere model grid")
parser.add_argument("--version", default=MARCS_version,
                    help="MARCS version string (default: %(default)s)")
parser.add_argument("--url", default=MARCS_URL,
                    help="URL of the MARCS data (default: %(default)s)")
parser.add_argument("--wavelength-url", default=MARCS_WAVELENGTH_URL,
                    help="URL of the shared MARCS wavelength grid "
                         "(default: %(default)s)")
parser.add_argument("--output",
                    default=shutil.os.path.join("..", "spectra", "marcs.h5"),
                    help="Output file for the HDF5 spectra (default: %(default)s)")
parser.add_argument("--output-dir",
                    default="marcs_temp",
                    help="Directory in which downloaded tarballs are saved "
                         "and extracted, and from which they are read for "
                         "processing (default: %(default)s). Left in place "
                         "afterwards rather than cleaned up, since these "
                         "tarballs are multi-GB and expensive to re-fetch.")
parser.add_argument("--local-dir",
                    default="",
                    help="Directory containing already-extracted composition "
                         "subdirectories (each named marcs_<code>_flx, e.g. "
                         "marcs_st_flx for --composition standard), to use "
                         "instead of downloading; falls back to downloading "
                         "into --output-dir for any composition not found "
                         "here")
parser.add_argument("--registry",
                    default=shutil.os.path.join("..", "spectra", "spectra.toml"),
                    help="Spectra registry TOML file (default: %(default)s)")
parser.add_argument("--overwrite", action="store_true",
                    help="Overwrite spectra groups already in the output file")
parser.add_argument("--composition", nargs="+",
                    choices=list(MARCS_COMPOSITIONS.keys()), default=[],
                    metavar="COMPOSITION",
                    help="Which composition tarball(s) to fetch; if "
                         "unspecified, fetch all. Choices: " +
                         ", ".join(MARCS_COMPOSITIONS.keys()))
parser.add_argument("--downsample", type=int, default=MARCS_DOWNSAMPLE,
                    help="Downsample factor (integer >= 1, default %(default)s); "
                         "output has ceil(N_orig / factor) wavelength points")
parser.add_argument("--registry-only", action="store_true",
                    help="Skip fetching the wavelength grid and processing "
                         "any composition -- just rebuild the MARCS registry "
                         "entry from whatever is already in --output. Useful "
                         "for regenerating a lost/edited registry file "
                         "without re-deriving --output itself, which "
                         "otherwise silently falls back to a full "
                         "(multi-GB) re-download for any --composition whose "
                         "--local-dir/--output-dir source can't be found.")
parser.add_argument("--verbose", action="store_true",
                    help="Print verbose output")
args = parser.parse_args()

if args.downsample < 1:
    parser.error("--downsample must be >= 1")

if args.registry_only and not shutil.os.path.exists(args.output):
    parser.error(f"--registry-only requires --output ({args.output}) to "
                 "already exist")

compositions = args.composition if args.composition else list(MARCS_COMPOSITIONS.keys())

# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

# Regex for a MARCS flux filename, e.g.
# s8000_g+3.5_m1.0_t05_st_z-2.50_a+0.40_c+0.00_n+0.00_o+0.40_r+0.00_s+0.00.flx.gz
# atmos: 's' (spherical) or 'p' (plane-parallel); mass is always 0.0 for
# plane-parallel models. r/s are always 0 for every model found so far and
# are undocumented anywhere on the MARCS site, so they're parsed (in case
# that ever changes) but otherwise ignored.
name_re = re.compile(
    r'(?P<atmos>[sp])(?P<teff>[0-9]+)_g(?P<logg>[+-][0-9.]+)'
    r'_m(?P<mass>[0-9]+\.[0-9]*)_t(?P<micro>[0-9]+)_(?P<comp>[a-z]{2})'
    r'_z(?P<feh>[+-][0-9.]+)_a(?P<afe>[+-][0-9.]+)_c(?P<cfe>[+-][0-9.]+)'
    r'_n(?P<nfe>[+-][0-9.]+)_o(?P<ofe>[+-][0-9.]+)'
    r'_r(?P<rval>[+-][0-9.]+)_s(?P<sval>[+-][0-9.]+)\.flx\.gz')

# Prefer plane-parallel ('p') over spherical ('s') when both are present
# for the same (feh, afe, cfe, micro, teff, logg) point; lower number wins
ATMOS_PREFERENCE = {'p': 0, 's': 1}

# ---------------------------------------------------------------------------
# Downsampling helper
# ---------------------------------------------------------------------------


def downsample_1d(arr: np.ndarray, n: int, log: bool = False) -> np.ndarray:
    """Downsample arr by averaging each block of n consecutive values
    (geometric mean, i.e. arithmetic mean in log space then exponentiated,
    if log is True; arithmetic mean otherwise). The final block may be
    shorter than n if len(arr) is not divisible by n -- its mean is
    computed over however many points it contains. Used identically for
    MARCS's shared wavelength grid (log=True, computed once) and for each
    spectrum's own flux (log=False, computed per spectrum) -- since both
    always have the same native length, calling this with the same n for
    each keeps their blocks aligned without needing to compute them
    together the way fetch_tlusty.py's own downsample() does.
    """
    n_out = math.ceil(len(arr) / n)
    pad = n_out * n - len(arr)
    a = np.log(arr) if log else arr.astype(float)
    if pad:
        a = np.concatenate([a, np.full(pad, np.nan)])
    a = a.reshape(n_out, n)
    out = np.nanmean(a, axis=1)
    return np.exp(out) if log else out


# ---------------------------------------------------------------------------
# Composition tarball download/extraction
# ---------------------------------------------------------------------------


def get_composition_dir(comp_key: str) -> str:
    """Return a local directory of comp_key's .flx.gz files, downloading
    and extracting marcs_<code>_flx.tar into --output-dir first if
    necessary. Checks --local-dir (if given) for an already-extracted
    marcs_<code>_flx/ directory first, so a pre-downloaded tarball never
    needs to be re-fetched.
    """
    code = MARCS_COMPOSITIONS[comp_key]
    dirname = f"marcs_{code}_flx"

    if args.local_dir:
        local_path = shutil.os.path.join(args.local_dir, dirname)
        if shutil.os.path.isdir(local_path):
            if args.verbose:
                print(f"Using already-extracted {local_path} for "
                      f"--composition {comp_key}.")
            return local_path

    extract_dir = shutil.os.path.join(args.output_dir, dirname)
    if shutil.os.path.isdir(extract_dir):
        if args.verbose:
            print(f"Using already-extracted {extract_dir} for "
                  f"--composition {comp_key}.")
        return extract_dir

    shutil.os.makedirs(args.output_dir, exist_ok=True)
    tar_name = f"{dirname}.tar"
    tar_url = args.url + tar_name
    tar_path = shutil.os.path.join(args.output_dir, tar_name)
    if args.verbose:
        print(f"Fetching {tar_url} ...")
    http = urllib3.PoolManager(timeout=urllib3.util.Timeout(connect=30, read=600))
    with http.request("GET", tar_url, preload_content=False) as response, \
         open(tar_path, "wb") as out_file:
        if response.status != 200:
            raise RuntimeError(f"Failed to fetch {tar_url}: HTTP {response.status}")
        shutil.copyfileobj(response, out_file)

    if args.verbose:
        print(f"Extracting {tar_path} ...")
    with tarfile.open(tar_path) as tf:
        tf.extractall(extract_dir)

    return extract_dir


if not args.registry_only:
    # ---------------------------------------------------------------------------
    # Create output file if it doesn't exist yet
    # ---------------------------------------------------------------------------

    if not shutil.os.path.exists(args.output):
        with h5py.File(args.output, "w") as h5file:
            h5file.attrs["references"] = MARCS_references
            h5file.attrs["reference_urls"] = MARCS_reference_urls

    # ---------------------------------------------------------------------------
    # Shared wavelength grid
    # ---------------------------------------------------------------------------

    with h5py.File(args.output, "a") as h5file:
        have_wave = "wavelengths" in h5file and "native" in h5file["wavelengths"]
        if have_wave and not args.overwrite:
            if args.verbose:
                print(f"Wavelength grid already present in {args.output}; skipping.")
            wave_ds = h5file["wavelengths"]["native"][:]
        else:
            if args.verbose:
                print(f"Fetching {args.wavelength_url} ...")
            http = urllib3.PoolManager()
            resp = http.request("GET", args.wavelength_url, preload_content=True)
            if resp.status != 200:
                raise RuntimeError(
                    f"Failed to fetch {args.wavelength_url}: HTTP {resp.status}")
            wave_native = np.loadtxt(gzip.open(io.BytesIO(resp.data)))
            wave_ds = downsample_1d(wave_native, args.downsample, log=True)

            wgrp = h5file.require_group("wavelengths")
            if "native" in wgrp:
                del wgrp["native"]
            wgrp.create_dataset("native", data=wave_ds, compression="gzip")
            if args.verbose:
                print(f"Wrote wavelength grid 'native' ({len(wave_ds)} pts, "
                      f"downsampled from {len(wave_native)}).")

    # ---------------------------------------------------------------------------
    # Process each requested composition
    # ---------------------------------------------------------------------------

    for comp_key in compositions:
        comp_dir = get_composition_dir(comp_key)

        # Every .flx.gz file's full path, recursively -- tolerates either a
        # flat directory of files or one further nested inside a subdirectory,
        # since different composition tarballs use different internal
        # layouts (confirmed against real downloads: e.g. alpha-poor's
        # members are flat, prefixed "./", while get_composition_dir's own
        # extract_dir wrapper adds a subdirectory level either way)
        flx_paths = sorted(glob.glob(
            shutil.os.path.join(comp_dir, "**", "*.flx.gz"), recursive=True))
        if not flx_paths:
            if args.verbose:
                print(f"No .flx.gz files found under {comp_dir}; skipping "
                      f"--composition {comp_key}.")
            continue
        if args.verbose:
            print(f"Processing --composition {comp_key} ({len(flx_paths)} files "
                  f"in {comp_dir}) ...")

        # Parse every filename, and group records first by (feh, afe, cfe,
        # micro) -- the eventual HDF5 group -- then by (teff, logg) -- the
        # eventual spectrum within that group -- so that, per (feh, afe, cfe,
        # micro, teff, logg) point, exactly one winning file can be chosen
        # among however many atmosphere-type/mass variants exist for it
        # (see pick_winner below).
        grid_groups: dict[tuple[float, float, float, int],
                           dict[tuple[int, float], list[dict]]] = {}
        for path in flx_paths:
            fname = shutil.os.path.basename(path)
            match = name_re.fullmatch(fname)
            if match is None:
                if args.verbose:
                    print(f"  Skipping unrecognized filename {fname}.")
                continue
            rec = {
                "path": path,
                "atmos": match.group("atmos"),
                "teff": int(match.group("teff")),
                "logg": float(match.group("logg")),
                "mass": float(match.group("mass")),
                "micro": int(match.group("micro")),
                "feh": float(match.group("feh")),
                "afe": float(match.group("afe")),
                "cfe": float(match.group("cfe")),
                "nfe": float(match.group("nfe")),
                "ofe": float(match.group("ofe")),
            }
            grp_key = (rec["feh"], rec["afe"], rec["cfe"], rec["micro"])
            tl_key = (rec["teff"], rec["logg"])
            grid_groups.setdefault(grp_key, {}).setdefault(tl_key, []).append(rec)

        def pick_winner(recs: list[dict]) -> dict:
            """Choose the single record to keep for one (feh, afe, cfe, micro,
            teff, logg) point: plane-parallel ('p') if any is present (mass is
            always 0 for those, so there is at most one); otherwise, among the
            spherical ('s') records, whichever mass is closest to 1.0 -- this
            is exactly mass == 1.0 when that mass is available, and a
            reasonable fallback (rather than an error) for the minority of
            points where it isn't.
            """
            preferred_atmos = min((r["atmos"] for r in recs),
                                  key=lambda a: ATMOS_PREFERENCE[a])
            candidates = [r for r in recs if r["atmos"] == preferred_atmos]
            if preferred_atmos == 'p':
                return candidates[0]
            return min(candidates, key=lambda r: abs(r["mass"] - 1.0))

        with h5py.File(args.output, "a") as h5file:
            # Skip (feh, afe, cfe, micro) groups already present, unless
            # --overwrite, same convention as fetch_bosz.py/fetch_ck04.py
            existing = set()
            if not args.overwrite:
                for grp_name in h5file.keys():
                    if not grp_name.startswith("spectra_"):
                        continue
                    a = h5file[grp_name].attrs
                    existing.add((float(a["feh"]), float(a["afe"]),
                                  float(a["cfe"]), int(a["micro"])))

            for grp_key, tl_dict in sorted(grid_groups.items()):
                if grp_key in existing:
                    if args.verbose:
                        print(f"  Skipping {grp_key}: already in {args.output}.")
                    continue
                feh_val, afe_val, cfe_val, micro_val = grp_key
                grp_name = (f"spectra_feh{feh_val:.2f}_afe{afe_val:.2f}"
                            f"_cfe{cfe_val:.2f}_micro{micro_val}")

                winners = {tl_key: pick_winner(recs)
                           for tl_key, recs in tl_dict.items()}

                if grp_name in h5file:
                    del h5file[grp_name]
                grp = h5file.create_group(grp_name)
                grp.attrs["feh"] = feh_val
                grp.attrs["afe"] = afe_val
                grp.attrs["cfe"] = cfe_val
                grp.attrs["micro"] = micro_val
                # [N/Fe] and [O/Fe] don't vary independently of (feh, afe,
                # cfe) -- every winner in this group shares the same values --
                # so they're recorded once, informationally, rather than as
                # filterable axes of their own
                any_winner = next(iter(winners.values()))
                grp.attrs["nfe"] = any_winner["nfe"]
                grp.attrs["ofe"] = any_winner["ofe"]

                for (teff, logg), rec in sorted(winners.items()):
                    with gzip.open(rec["path"], "rt") as f:
                        flux_native = np.loadtxt(f)
                    flux = downsample_1d(flux_native, args.downsample, log=False)
                    dset_name = f"t{teff}_g{logg:+.1f}"
                    dset = grp.create_dataset(dset_name, data=flux, compression="gzip")
                    dset.attrs["teff"] = teff
                    dset.attrs["logg"] = logg

                if args.verbose:
                    print(f"  Wrote {len(winners)} spectra to group {grp_name}.")

# ---------------------------------------------------------------------------
# Update the TOML registry
# ---------------------------------------------------------------------------

if shutil.os.path.exists(args.registry):
    with open(args.registry) as f:
        registry = tomlkit.parse(f.read())
else:
    registry = {"name": "Registry of spectra sets"}

if "spectra_sets" in registry:
    if "MARCS" not in registry["spectra_sets"]:
        registry["spectra_sets"].append("MARCS")
else:
    registry["spectra_sets"] = ["MARCS"]

if "MARCS" in registry:
    registry.pop("MARCS")
tab = tomlkit.table()
tab["file"] = args.output
tab["version"] = args.version
tab["references"] = MARCS_references
tab["reference_urls"] = MARCS_reference_urls

int_qtys = {"micro"}
with h5py.File(args.output, "r") as h5file:
    for qty, attr in zip(["Fe_H", "alpha_Fe", "C_Fe", "micro"],
                         ["feh", "afe", "cfe", "micro"]):
        val_in_file = sorted({
            h5file[grp].attrs[attr]
            for grp in h5file.keys() if grp.startswith("spectra_")
        })
        cast = int if qty in int_qtys else float
        tab[qty] = [cast(v) for v in val_in_file]
tab["micro_default"] = MARCS_MICRO_DEFAULT

registry["MARCS"] = tab

with open(args.registry, "w") as f:
    f.write(tomlkit.dumps(registry))
if args.verbose:
    print(f"Updated registry at {args.registry}.")
