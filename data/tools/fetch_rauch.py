"""
Script to fetch the Rauch et al. NLTE hot star (central-star-of-
planetary-nebula / hot pre-white-dwarf) model atmosphere grid from the
TheoSSA VO service at the German Astrophysical Virtual Observatory
(GAVO), and pack it into an HDF5 file as four flat datasets -- wl,
logg, log_Teff, and a (n_models, n_wl) flux array. This differs from
fetch_tremblay.py's (n_logg, n_logTeff, n_wl) tensor layout because
this grid's (log g, Teff) coverage is only partially filled: RAUCH_
QUERY_PARAMS below (restricted to the solar-abundance subset of Rauch
(2002)'s NLTE hot star models) returns 51 (log g, Teff) combinations
spanning Teff = 50000-190000 K and log g = 5-8, but log g = 5 is
missing for Teff >= 110000 K, so there is no way to lay the models out
as a filled rectangular grid.

Queried via pyvo's Simple Spectral Access (SSA) protocol client. Every
one of the 51 models returned shares one common wavelength grid (5-
2000 Angstrom, 19951 points, verified below rather than assumed), so a
single "wl" dataset (rather than one per model) is enough.

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
import tomlkit
from astropy.table import Table

# TheoSSA's browse page (for reference/documentation only) and its
# actual SSA protocol endpoint (used to construct the query below)
RAUCH_BROWSE_URL = "https://dc.g-vo.org/browse/theossa/q"
RAUCH_SSA_URL = "http://dc.g-vo.org/theossa/q/ssa/ssap.xml"

# Restricts the query to the solar-abundance subset of Rauch (2002)'s
# NLTE hot star models, via H/He surface mass fraction ranges
# bracketing the solar values (0.7037/0.2793) -- see this file's own
# docstring
RAUCH_QUERY_PARAMS = {
    "format": "votable",
    "w_H": "0.7036/0.7038",
    "w_He": "0.2793/0.2794",
}

RAUCH_2003 = "Rauch, T. 2003, A&A, 403, 709"
RAUCH_2018 = "Rauch, T., Demleitner, M., Hoyer, D., et al. 2018, MNRAS, 475, 3896"
RAUCH_REFERENCES = [RAUCH_2003, RAUCH_2018]

RAUCH_2003_URL = "https://ui.adsabs.harvard.edu/abs/2003A%26A...403..709R/abstract"
RAUCH_2018_URL = "https://ui.adsabs.harvard.edu/abs/2018MNRAS.475.3896R/abstract"
RAUCH_REFERENCE_URLS = [RAUCH_2003_URL, RAUCH_2018_URL]

RAUCH_OUTPUT = shutil.os.path.join("..", "spectra", "rauch.h5")
RAUCH_REGISTRY_NAME = "RAUCH"
RAUCH_VERSION = "2003"

# ---------------------------------------------------------------------------
# Parse command line arguments
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Fetch the Rauch et al. NLTE hot star atmosphere grid via TheoSSA's VO service")
parser.add_argument("--url", default=RAUCH_SSA_URL,
                    help="SSA service URL to query (default: %(default)s)")
parser.add_argument("--output", default=RAUCH_OUTPUT,
                    help="Output HDF5 file (default: %(default)s)")
parser.add_argument("--registry",
                    default=shutil.os.path.join("..", "spectra", "spectra.toml"),
                    help="Spectra registry TOML file (default: %(default)s)")
parser.add_argument("--overwrite", action="store_true",
                    help="Overwrite the output HDF5 file and registry entry if already present")
parser.add_argument("--verbose", action="store_true",
                    help="Print progress messages")
args = parser.parse_args()

if shutil.os.path.exists(args.output) and not args.overwrite:
    raise SystemExit(
        f"{args.output} already exists; pass --overwrite to regenerate it.")

# ---------------------------------------------------------------------------
# Query TheoSSA for the list of matching models, then download each one
# ---------------------------------------------------------------------------

if args.verbose:
    print(f"Querying {args.url} ...")
svc = pyvo.dal.SSAService(args.url)
recs = svc.create_query(**RAUCH_QUERY_PARAMS).execute()
if len(recs) == 0:
    raise SystemExit("Query returned no matching models.")

wl_ref: np.ndarray | None = None
flux: np.ndarray | None = None
logg_vals = np.empty(len(recs))
log_teff_vals = np.empty(len(recs))
for i, rec in enumerate(recs):
    if args.verbose:
        print(f"  [{i + 1}/{len(recs)}] log(g)={rec['log_g']} Teff={rec['t_eff']} ...")
    t = Table.read(rec.acref)
    wl = t["spectral"].quantity.to(u.Angstrom).value
    if wl_ref is None:
        wl_ref = wl
        flux = np.empty((len(recs), len(wl_ref)))
    elif wl.shape != wl_ref.shape or not np.allclose(wl, wl_ref):
        raise RuntimeError(
            f"{rec.acref}: wavelength grid differs from the first model fetched")
    flux[i, :] = t["flux"].to(u.erg / u.cm**2 / u.s / u.Angstrom).value
    logg_vals[i] = float(rec["log_g"])
    log_teff_vals[i] = np.log10(float(rec["t_eff"]))

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

if "spectra_sets" in registry:
    if RAUCH_REGISTRY_NAME not in registry["spectra_sets"]:
        registry["spectra_sets"].append(RAUCH_REGISTRY_NAME)
else:
    registry["spectra_sets"] = [RAUCH_REGISTRY_NAME]

if RAUCH_REGISTRY_NAME in registry:
    registry.pop(RAUCH_REGISTRY_NAME)
tab = tomlkit.table()
tab["file"] = args.output
tab["version"] = RAUCH_VERSION
tab["references"] = RAUCH_REFERENCES
tab["reference_urls"] = RAUCH_REFERENCE_URLS
tab["logg"] = [float(x) for x in logg_vals]
tab["log_Teff"] = [float(x) for x in log_teff_vals]
registry[RAUCH_REGISTRY_NAME] = tab

with open(args.registry, "w") as f:
    f.write(tomlkit.dumps(registry))
if args.verbose:
    print(f"Updated registry at {args.registry}.")
