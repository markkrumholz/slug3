#!/usr/bin/env python3
"""Fetch and import Gil-Pons et al. (2022) very-low-metallicity AGB yield
tables from the CDS into data/yields/gilpons22.h5 and yields.toml.

Source: CDS catalogue J/A+A/668/A100
  https://cdsarc.cds.unistra.fr/viz-bin/cat/J/A+A/668/A100

Four data files are downloaded:

  z1em06.dat  Z = 1e-6
  z1em07.dat  Z = 1e-7
  z1em08.dat  Z = 1e-8
  z1em10.dat  Z = 1e-10

Each file is a whitespace-delimited ASCII table with one row per
(metallicity, stellar-mass, isotope) combination.  The columns are:

  1  Z           metallicity
  2  Mini        initial stellar mass, Msun
  3  Species     isotope symbol (e.g. he4, c12, al-6)
  4  A(i)        mass number
  5  Yield(i)    net yield, Msun
  6  Meject(i)   total mass of isotope ejected (gross yield) — we read this
  7  Mini(i)     initial mass of isotope in the star
  8  <X(i)>      mean mass fraction in ejecta
  9  X0(i)       initial mass fraction
  10 log(...)    log ratio

Special species handling:
  ``g``   (A=1) non-isotope catch-all — skipped.
  ``n``   (A=1) neutron — skipped.
  ``p``   (A=1) proton — mapped to H-1.
  ``d``   (A=2) deuterium — mapped to H-2.
  ``al-6``       Al-26 with radioactive-decay-aware treatment — kept as Al-26
                 (Z=13, A=26).
  ``al*6``       Al-26 without decay treatment — skipped (use al-6 instead).

Solar metallicity: Z_SOLAR = 0.014 (Asplund et al. 2009).
[Fe/H] = log10(Z / Z_SOLAR).

Mass grid:
  Because the four metallicities do not share the same mass coverage, the
  union of all stellar masses is used as the shared grid.  For a given [Fe/H]:
    - masses outside [min_mass, max_mass] at that metallicity receive zeros;
    - masses within range that lack a direct model are linearly interpolated
      between the two bracketing mass points, isotope by isotope;
    - isotopes absent from a given model are set to zero.

Downloaded files are cached in --download-dir (default: a temp directory) to
avoid re-fetching on repeated runs.  Pass --no-cache to force re-download.

Run from the repository root:

    python3 data/tools/yields/import_gilpons22.py

:copyright: Copyright (c) 2026 Mark Krumholz
"""

import argparse
import math
import os
import pathlib
import tempfile
import urllib.request
import warnings
from dataclasses import dataclass, field

import h5py
import numpy as np
import tomlkit

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

H5_DIR = "data/yields"
REGISTRY_PATH = "data/yields/yields.toml"
CHANNEL = "agb"
MODEL_NAME = "gilpons22"
Z_SOLAR = 0.014  # Asplund et al. 2009

REFERENCE = (
    "Gil-Pons, P., Doherty, C. L., Campbell, S. W., et al. 2022, A&A, 668, 100"
)
REFERENCE_URL = (
    "https://ui.adsabs.harvard.edu/abs/2022A%26A...668A.100G/abstract"
)

_CDS_BASE = (
    "https://cdsarc.cds.unistra.fr/viz-bin/nph-Cat/txt?J/A+A/668/A100/"
)

# (filename, Z_value)
SOURCE_FILES = [
    ("z1em06.dat", 1e-6),
    ("z1em07.dat", 1e-7),
    ("z1em08.dat", 1e-8),
    ("z1em10.dat", 1e-10),
]

# ---------------------------------------------------------------------------
# Element table and isotope utilities (shared with other import scripts)
# ---------------------------------------------------------------------------

ELEMENT_SYMBOLS = [
    "H", "He",
    "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba",
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
    "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn",
    "Fr", "Ra",
    "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf",
    "Es", "Fm", "Md", "No", "Lr",
    "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn",
    "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
]
SYMBOL_TO_Z: dict[str, int] = {
    sym.lower(): z for z, sym in enumerate(ELEMENT_SYMBOLS, start=1)
}

import re as _re
_SPECIES_PREFIX = _re.compile(r"^([a-z]+)")

# Species requiring special handling: lowercase key → atomic Z or None=skip
_SPECIAL: dict[str, int | None] = {
    "g":    None,   # non-isotope catch-all — skip
    "n":    None,   # neutron — skip
    "al*6": None,   # Al-26 without decay treatment — skip (prefer al-6)
    "p":    1,      # proton  → H
    "d":    1,      # deuterium → H
    "t":    1,      # tritium → H
    "al-6": 13,     # Al-26 with decay — keep as (Z=13, A=26)
}


def parse_species(species: str, a: int) -> tuple[int, int] | None:
    """Return (Z, A) or None to skip.

    Handles:
    - explicit special cases in _SPECIAL (neutron, al-6/al*6, p/d/t)
    - isomer suffixes like ``al-6`` → Z=13, A from file
    - standard labels like ``he4``, ``fe56`` → Z from element symbol, A from file
    """
    s = species.lower().strip()
    if s in _SPECIAL:
        z_val = _SPECIAL[s]
        if z_val is None:
            return None
        return z_val, a

    m = _SPECIES_PREFIX.match(s)
    if m is None:
        return None  # unrecognised token, skip silently
    sym = m.group(1)
    z_val = SYMBOL_TO_Z.get(sym)
    if z_val is None:
        return None
    return z_val, a


# ---------------------------------------------------------------------------
# Data record
# ---------------------------------------------------------------------------

@dataclass
class YieldRecord:
    mass: float                                    # stellar mass, Msun
    feh: float                                     # [Fe/H]
    yields: dict[tuple[int, int], float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Download and parsing
# ---------------------------------------------------------------------------

MEJECT_COL = 5  # 1-indexed column (pipe-delimited): Z|Mini|Species|A|Yield|Meject|...
                # → 0-indexed after split: fields[5]


def download_file(filename: str, cache_dir: pathlib.Path, no_cache: bool) -> str:
    """Download (or return cached) CDS file content as a string."""
    cache_path = cache_dir / filename
    if not no_cache and cache_path.exists():
        return cache_path.read_text()

    url = _CDS_BASE + filename
    print(f"  Downloading {url} ...")
    with urllib.request.urlopen(url, timeout=60) as resp:
        content = resp.read().decode("utf-8", errors="replace")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(content)
    return content


def parse_content(content: str, z_val: float) -> list[YieldRecord]:
    """Parse a CDS pipe-delimited data file; return one YieldRecord per mass.

    The CDS nph-Cat/txt format uses ``|`` as the column separator.  Header
    and separator lines (starting with ``#`` or ``-``, or whose first field
    is not a float) are skipped automatically.

    Columns (1-indexed in CDS docs, 0-indexed in code):
      0 Z, 1 Mini, 2 Species, 3 A, 4 Yield, 5 Meject (we read this)
    """
    feh = math.log10(z_val / Z_SOLAR)
    by_mass: dict[float, YieldRecord] = {}

    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        if "|" not in line:
            continue  # skip any non-pipe lines

        fields = [f.strip() for f in line.split("|")]
        if len(fields) <= MEJECT_COL:
            continue

        # First field must parse as a float (the Z value)
        try:
            float(fields[0])
            mass = float(fields[1])
            a = int(fields[3])
            meject = float(fields[MEJECT_COL])
        except (ValueError, IndexError):
            continue

        species = fields[2].lower().strip()
        za = parse_species(species, a)
        if za is None:
            continue

        if mass not in by_mass:
            by_mass[mass] = YieldRecord(mass=mass, feh=feh)
        rec = by_mass[mass]
        rec.yields[za] = rec.yields.get(za, 0.0) + meject

    return list(by_mass.values())


# ---------------------------------------------------------------------------
# Array building with interpolation (same rules as import_karakas16.py)
# ---------------------------------------------------------------------------

def feh_group_name(feh: float) -> str:
    return f"feh_{feh:g}"


def build_arrays(records: list[YieldRecord]) -> dict:
    """Build yield arrays across all [Fe/H] values.

    Returns
    -------
    dict with keys:
      masses      : sorted 1-D ndarray, Msun  (union of all masses)
      isotope_z   : 1-D ndarray float64
      isotope_a   : 1-D ndarray float64
      feh_yields  : dict mapping float feh -> 2-D ndarray (n_iso, n_mass)
    """
    all_masses: set[float] = set()
    all_za: set[tuple[int, int]] = set()
    by_feh: dict[float, list[YieldRecord]] = {}

    for rec in records:
        all_masses.add(rec.mass)
        all_za.update(rec.yields.keys())
        by_feh.setdefault(rec.feh, []).append(rec)

    masses = sorted(all_masses)
    isotopes = sorted(all_za)  # sort by (Z, A)
    fehs_sorted = sorted(by_feh.keys())

    isotope_z = np.array([za[0] for za in isotopes], dtype=np.float64)
    isotope_a = np.array([za[1] for za in isotopes], dtype=np.float64)
    iso_idx: dict[tuple[int, int], int] = {za: i for i, za in enumerate(isotopes)}
    mass_idx: dict[float, int] = {m: j for j, m in enumerate(masses)}

    feh_yields: dict[float, np.ndarray] = {}
    for feh in fehs_sorted:
        feh_records = by_feh[feh]
        data_at_mass: dict[float, dict[tuple[int, int], float]] = {
            rec.mass: rec.yields for rec in feh_records
        }
        data_masses_sorted = sorted(data_at_mass.keys())
        min_mass = data_masses_sorted[0]
        max_mass = data_masses_sorted[-1]

        arr = np.zeros((len(isotopes), len(masses)), dtype=np.float64)

        for j, m in enumerate(masses):
            if m < min_mass or m > max_mass:
                continue  # outside range — leave zeros

            if m in data_at_mass:
                for za, val in data_at_mass[m].items():
                    if za in iso_idx:
                        arr[iso_idx[za], j] = val
            else:
                # Linear interpolation between bracketing masses
                lo = max(dm for dm in data_masses_sorted if dm < m)
                hi = min(dm for dm in data_masses_sorted if dm > m)
                t = (m - lo) / (hi - lo)
                lo_y = data_at_mass[lo]
                hi_y = data_at_mass[hi]
                for za in set(lo_y) | set(hi_y):
                    if za not in iso_idx:
                        continue
                    v_lo = lo_y.get(za, 0.0)
                    v_hi = hi_y.get(za, 0.0)
                    arr[iso_idx[za], j] = v_lo + t * (v_hi - v_lo)

        feh_yields[feh] = arr

    return {
        "masses": np.array(masses, dtype=np.float64),
        "isotope_z": isotope_z,
        "isotope_a": isotope_a,
        "feh_yields": feh_yields,
    }


# ---------------------------------------------------------------------------
# HDF5 output
# ---------------------------------------------------------------------------

def write_h5(h5_path: pathlib.Path, arrays: dict) -> None:
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, "w") as h5:
        h5.attrs["reference"] = REFERENCE
        h5.attrs["reference_url"] = REFERENCE_URL

        grp = h5.require_group(CHANNEL)

        for dset_name, data in (
            ("masses", arrays["masses"]),
            ("isotope_z", arrays["isotope_z"]),
            ("isotope_a", arrays["isotope_a"]),
        ):
            if dset_name in grp:
                del grp[dset_name]
            grp.create_dataset(dset_name, data=data, compression="gzip")

        for feh, yield_arr in sorted(arrays["feh_yields"].items()):
            gname = feh_group_name(feh)
            if gname in grp:
                del grp[gname]
            feh_grp = grp.create_group(gname)
            feh_grp.attrs["Fe_H"] = feh
            feh_grp.create_dataset("yield", data=yield_arr, compression="gzip")


# ---------------------------------------------------------------------------
# TOML registry
# ---------------------------------------------------------------------------

def update_registry(
    registry_path: pathlib.Path,
    h5_filename: str,
    arrays: dict,
) -> None:
    doc = (
        tomlkit.parse(registry_path.read_text())
        if registry_path.exists()
        else tomlkit.document()
    )

    if "name" not in doc:
        doc["name"] = "Registry of nucleosynthetic yield models"
    if "channels" not in doc:
        doc["channels"] = []

    channel_list = list(doc["channels"])
    if CHANNEL not in channel_list:
        channel_list.append(CHANNEL)
    doc["channels"] = channel_list

    channel_table = doc[CHANNEL] if CHANNEL in doc else tomlkit.table()
    models = list(channel_table.get("models", []))
    if MODEL_NAME not in models:
        models.append(MODEL_NAME)
        models.sort()
    channel_table["models"] = models

    model_table = tomlkit.table()
    model_table["reference"] = REFERENCE
    model_table["reference_url"] = REFERENCE_URL
    model_table["file"] = h5_filename
    model_table["Fe_H"] = sorted(arrays["feh_yields"].keys())
    model_table["masses"] = [float(m) for m in arrays["masses"]]
    channel_table[MODEL_NAME] = model_table
    doc[CHANNEL] = channel_table

    registry_path.write_text(tomlkit.dumps(doc))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--h5-dir",
        default=H5_DIR,
        help=f"Directory to write {MODEL_NAME}.h5 into (default: {H5_DIR})",
    )
    p.add_argument(
        "--registry",
        default=REGISTRY_PATH,
        help=f"Path to yields registry TOML (default: {REGISTRY_PATH})",
    )
    p.add_argument(
        "--download-dir",
        default=None,
        metavar="DIR",
        help="Directory for caching downloaded CDS files "
             "(default: system temp directory; files are reused on re-runs)",
    )
    p.add_argument(
        "--no-cache",
        action="store_true",
        help="Force re-download of CDS files even if cached copies exist",
    )
    args = p.parse_args()

    if args.download_dir is None:
        cache_dir = pathlib.Path(tempfile.gettempdir()) / "gilpons22_cache"
    else:
        cache_dir = pathlib.Path(args.download_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading / reading CDS files into {cache_dir} ...")
    all_records: list[YieldRecord] = []
    for filename, z_val in SOURCE_FILES:
        content = download_file(filename, cache_dir, args.no_cache)
        recs = parse_content(content, z_val)
        feh = math.log10(z_val / Z_SOLAR)
        masses_here = sorted(rec.mass for rec in recs)
        print(
            f"  {filename}: Z={z_val:.0e}, [Fe/H]={feh:.3f}, "
            f"{len(recs)} masses {masses_here}"
        )
        all_records.extend(recs)

    print(f"\nBuilding arrays from {len(all_records)} (mass, [Fe/H]) records ...")
    arrays = build_arrays(all_records)
    n_iso = len(arrays["isotope_z"])
    n_mass = len(arrays["masses"])
    fehs = sorted(arrays["feh_yields"])
    print(f"  union mass grid ({n_mass} masses): {list(arrays['masses'])}")
    print(f"  isotopes: {n_iso}")
    print(f"  [Fe/H] values ({len(fehs)}): {[round(f, 3) for f in fehs]}")
    for feh in fehs:
        y = arrays["feh_yields"][feh]
        n_data = int(np.any(y != 0, axis=0).sum())
        print(f"    [Fe/H]={feh:.5f}: {n_data}/{n_mass} masses with data")

    h5_filename = f"{MODEL_NAME}.h5"
    h5_path = pathlib.Path(args.h5_dir) / h5_filename
    write_h5(h5_path, arrays)
    print(f"\nWrote {h5_path}")

    registry_path = pathlib.Path(args.registry)
    # Store the HDF5 path relative to the registry's own directory, not just
    # its basename, so a custom --h5-dir still resolves correctly when
    # YieldChannel reads it back relative to registry_path.parent
    registry_h5_path = os.path.relpath(h5_path, registry_path.parent)
    update_registry(registry_path, registry_h5_path, arrays)
    print(f"Updated {registry_path}")


if __name__ == "__main__":
    main()
