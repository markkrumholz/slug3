"""
Script to fetch a Rauch et al. NLTE hot star (central-star-of-
planetary-nebula / hot pre-white-dwarf) model atmosphere grid from the
TheoSSA VO service at the German Astrophysical Virtual Observatory
(GAVO), and pack it into an HDF5 file as four flat datasets -- wl,
logg, log_Teff, and a (n_models, n_wl) flux array. This differs from
fetch_tremblay.py's (n_logg, n_logTeff, n_wl) tensor layout because
these grids' (log g, Teff) coverage is only partially filled, so there
is no way to lay the models out as a filled rectangular grid.

Two grids are available, fetched independently via --grid-type:

  "solar" -- the solar-abundance subset of Rauch (2002)'s NLTE hot
             star models (RAUCH_QUERY_PARAMS_SOLAR below): 51 (log g,
             Teff) combinations spanning Teff = 50000-190000 K and
             log g = 5-8 (log g = 5 is missing for Teff >= 110000 K).
  "h07"   -- a pure H/He (surface H mass fraction ~0.70, no metals at
             all, so no metal line blanketing) subset (RAUCH_QUERY_
             PARAMS_H07 below), filtered down (see RAUCH_SPEC_RANGE_
             H07) to the models spanning the same 5-2000 Angstrom
             range as the solar grid: 223 (log g, Teff) combinations
             spanning Teff = 50000-1000000 K and log g = 5-9. This
             grid's lack of metal line blanketing makes it a less
             physically faithful model than the solar grid, but it
             reaches higher log(g) (up to 9, vs. the solar grid's 8)
             at the hottest temperatures the solar grid also covers,
             plugging a real gap between the solar Rauch grid's own
             log(g) <= 8 ceiling and the Tremblay grids' own Teff
             ceilings (140000 K for TREMBLAY_DA) -- see the WD/hot-
             atmosphere-coverage project notes for the gap this was
             found to close. Intended to be chained after the solar
             grid, not in place of it (see registry.toml's own
             "default" chain), so the more physically faithful solar
             grid is always preferred wherever it actually covers a
             star.

Queried via pyvo's Simple Spectral Access (SSA) protocol client. Every
model in a given grid shares one common wavelength grid (verified
below rather than assumed), so a single "wl" dataset (rather than one
per model) is enough for each.

Unlike slug's other spectral-library fetch scripts, these models have
no [Fe/H], [alpha/Fe], [C/Fe], or microturbulence axis -- parameterized
by log(g) and Teff alone -- so this script takes no --feh/--afe/--cfe
arguments. Fluxes returned by the service are already true (surface)
fluxes rather than Eddington fluxes, so, unlike fetch_tremblay.py, no
4*pi factor is applied here.

References
----------
  Rauch, T. 2003, A&A, 403, 709.
  Rauch, T., Demleitner, M., Hoyer, D., et al. 2018, MNRAS, 475, 3896.
"""

# Imports
import argparse
import astropy.units as u
import h5py
import numpy as np
import pyvo
import shutil
import time
import tomlkit
import urllib.error
from astropy.table import Table
from astropy.utils.data import conf as astropy_data_conf

# TheoSSA's per-model VOTable downloads are large (tens of thousands of
# flux points each) and the service is occasionally slow to generate
# them on demand, so astropy's own default remote-data timeout (10s)
# is too short here -- individual downloads below are retried a few
# times, with this longer per-attempt timeout, rather than failing the
# whole fetch over a single slow response.
astropy_data_conf.remote_timeout = 60.0
DOWNLOAD_RETRIES = 5
DOWNLOAD_RETRY_DELAY = 5.0  # seconds


def read_table_with_retries(url: str) -> Table:
    """Read a VOTable URL, retrying a few times on a transient network error."""
    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            return Table.read(url)
        except (TimeoutError, urllib.error.URLError) as e:
            if attempt == DOWNLOAD_RETRIES:
                raise
            print(f"    (attempt {attempt}/{DOWNLOAD_RETRIES} failed: {e}; retrying)")
            time.sleep(DOWNLOAD_RETRY_DELAY)
    raise AssertionError("unreachable")  # pragma: no cover


# TheoSSA's browse page (for reference/documentation only) and its
# actual SSA protocol endpoint (used to construct the query below)
RAUCH_BROWSE_URL = "https://dc.g-vo.org/browse/theossa/q"
RAUCH_SSA_URL = "http://dc.g-vo.org/theossa/q/ssa/ssap.xml"

# Restricts the "solar" query to the solar-abundance subset of Rauch
# (2002)'s NLTE hot star models, via H/He surface mass fraction ranges
# bracketing the solar values (0.7037/0.2793) -- see this file's own
# docstring
RAUCH_QUERY_PARAMS_SOLAR = {
    "format": "votable",
    "w_H": "0.7036/0.7038",
    "w_He": "0.2793/0.2794",
}

# Restricts the "h07" query to a pure H/He (~0.70 surface H mass
# fraction, no metals) subset -- see this file's own docstring
RAUCH_QUERY_PARAMS_H07 = {
    "format": "votable",
    "w_H": "0.6999/0.7001",
}

# The "h07" query above also returns many very-high-resolution spectra
# covering only a narrow wavelength window (e.g. a single line list),
# which is not what we want here; ssa_specstart/ssa_specend (in
# meters) identify the models that instead span the same 5-2000
# Angstrom range as the solar grid, matching this file's own flat,
# single-shared-wavelength-grid schema
RAUCH_SPEC_RANGE_H07 = (5.1e-10, 2.0e-7)

RAUCH_2003 = "Rauch, T. 2003, A&A, 403, 709"
RAUCH_2018 = "Rauch, T., Demleitner, M., Hoyer, D., et al. 2018, MNRAS, 475, 3896"
RAUCH_REFERENCES = [RAUCH_2003, RAUCH_2018]

RAUCH_2003_URL = "https://ui.adsabs.harvard.edu/abs/2003A%26A...403..709R/abstract"
RAUCH_2018_URL = "https://ui.adsabs.harvard.edu/abs/2018MNRAS.475.3896R/abstract"
RAUCH_REFERENCE_URLS = [RAUCH_2003_URL, RAUCH_2018_URL]

# Per-grid-type settings: query parameters, an optional (specstart,
# specend) filter, output HDF5 file, and registry entry name
RAUCH_GRID_TYPES = {
    "solar": {
        "query_params": RAUCH_QUERY_PARAMS_SOLAR,
        "spec_range": None,
        "output": shutil.os.path.join("..", "spectra", "rauch.h5"),
        "registry_name": "RAUCH",
    },
    "h07": {
        "query_params": RAUCH_QUERY_PARAMS_H07,
        "spec_range": RAUCH_SPEC_RANGE_H07,
        "output": shutil.os.path.join("..", "spectra", "rauch_h07.h5"),
        "registry_name": "RAUCH_H07",
    },
}

RAUCH_VERSION = "2003"

# ---------------------------------------------------------------------------
# Parse command line arguments
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Fetch a Rauch et al. NLTE hot star atmosphere grid via TheoSSA's VO service")
parser.add_argument("--grid-type", choices=sorted(RAUCH_GRID_TYPES), required=True,
                    help="Which grid to fetch: 'solar' for the solar-abundance grid "
                         "(-> rauch.h5/RAUCH) or 'h07' for the pure H/He, no-metals grid "
                         "(-> rauch_h07.h5/RAUCH_H07)")
parser.add_argument("--url", default=RAUCH_SSA_URL,
                    help="SSA service URL to query (default: %(default)s)")
parser.add_argument("--output", default="",
                    help="Output HDF5 file (default: chosen by --grid-type)")
parser.add_argument("--registry",
                    default=shutil.os.path.join("..", "spectra", "spectra.toml"),
                    help="Spectra registry TOML file (default: %(default)s)")
parser.add_argument("--overwrite", action="store_true",
                    help="Overwrite the output HDF5 file and registry entry if already present")
parser.add_argument("--verbose", action="store_true",
                    help="Print progress messages")
args = parser.parse_args()

grid_cfg = RAUCH_GRID_TYPES[args.grid_type]
if not args.output:
    args.output = grid_cfg["output"]

if shutil.os.path.exists(args.output) and not args.overwrite:
    raise SystemExit(
        f"{args.output} already exists; pass --overwrite to regenerate it.")

# ---------------------------------------------------------------------------
# Query TheoSSA for the list of matching models, then download each one
# ---------------------------------------------------------------------------

if args.verbose:
    print(f"Querying {args.url} ...")
svc = pyvo.dal.SSAService(args.url)
recs = svc.create_query(**grid_cfg["query_params"]).execute()
if grid_cfg["spec_range"] is not None:
    specstart, specend = grid_cfg["spec_range"]
    recs = [r for r in recs if r["ssa_specstart"] == specstart and r["ssa_specend"] == specend]
if len(recs) == 0:
    raise SystemExit("Query returned no matching models.")

wl_ref: np.ndarray | None = None
flux_rows: list[np.ndarray] = []
logg_list: list[float] = []
log_teff_list: list[float] = []
for i, rec in enumerate(recs):
    if args.verbose:
        print(f"  [{i + 1}/{len(recs)}] log(g)={rec['log_g']} Teff={rec['t_eff']} ...")
    t = read_table_with_retries(rec.acref)
    wl = t["spectral"].quantity.to(u.Angstrom).value
    if wl_ref is None:
        wl_ref = wl
    elif wl.shape != wl_ref.shape or not np.allclose(wl, wl_ref):
        # A handful of TheoSSA records' actual wavelength coverage
        # doesn't match what their own ssa_specstart/ssa_specend
        # metadata (already filtered on above, for the "h07" grid)
        # claims -- e.g. one "h07" record declares 5-2000 Angstrom but
        # actually extends to ~5993 Angstrom. Rather than aborting the
        # whole fetch over what is a data-quality quirk in a small
        # minority of records, skip just this one (with a warning) and
        # keep the rest -- unlike fetch_tremblay.py's own identical-
        # looking check, which can safely be a hard error there, since
        # every one of its own archive's member files is expected to
        # share a common grid with no exceptions.
        print(f"  WARNING: {rec.acref}: wavelength grid ({len(wl)} points, "
              f"{wl.min()}-{wl.max()} Angstrom) differs from the first model "
              f"fetched ({len(wl_ref)} points, {wl_ref.min()}-{wl_ref.max()} "
              "Angstrom); skipping this record")
        continue
    flux_rows.append(t["flux"].to(u.erg / u.cm**2 / u.s / u.Angstrom).value)
    logg_list.append(float(rec["log_g"]))
    log_teff_list.append(np.log10(float(rec["t_eff"])))

if not flux_rows:
    raise SystemExit("No records had a wavelength grid matching the first one fetched.")
flux = np.array(flux_rows)
logg_vals = np.array(logg_list)
log_teff_vals = np.array(log_teff_list)
if args.verbose and len(flux_rows) != len(recs):
    print(f"Kept {len(flux_rows)} of {len(recs)} records "
          f"({len(recs) - len(flux_rows)} skipped for a mismatched wavelength grid).")

# ---------------------------------------------------------------------------
# Write the HDF5 file: wl, logg, log_Teff, and a (n_models, n_wl) flux
# array -- not a (n_logg, n_logTeff, n_wl) tensor, since this grid's
# (log g, Teff) coverage is only partially filled (see this file's own
# docstring)
# ---------------------------------------------------------------------------

with h5py.File(args.output, "w") as h5file:
    h5file.attrs["references"] = RAUCH_REFERENCES
    h5file.attrs["reference_urls"] = RAUCH_REFERENCE_URLS
    h5file.create_dataset("wl", data=wl_ref, compression="gzip")
    h5file.create_dataset("logg", data=logg_vals, compression="gzip")
    h5file.create_dataset("log_Teff", data=log_teff_vals, compression="gzip")
    h5file.create_dataset("flux", data=flux, compression="gzip")
if args.verbose:
    print(f"Wrote {args.output}: {flux.shape} (n_models, n_wl) flux array.")

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
tab["file"] = args.output
tab["version"] = RAUCH_VERSION
tab["references"] = RAUCH_REFERENCES
tab["reference_urls"] = RAUCH_REFERENCE_URLS
# Tells SpecsynLibChained to build this entry as a SpecsynLibWD --
# same flag fetch_tremblay.py sets, since this grid is read by the
# same (logg, Teff)-parameterized reader despite not being white
# dwarfs themselves, and despite its (logg, Teff) coverage being only
# partially filled rather than a full tensor (see this file's own
# docstring) -- SpecsynLib2D handles that case
tab["WD_grid"] = True
tab["logg"] = [float(x) for x in logg_vals]
tab["log_Teff"] = [float(x) for x in log_teff_vals]
registry[registry_name] = tab

with open(args.registry, "w") as f:
    f.write(tomlkit.dumps(registry))
if args.verbose:
    print(f"Updated registry at {args.registry}.")
