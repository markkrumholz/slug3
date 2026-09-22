#!/usr/bin/env python3
"""Import Cinquegrana & Karakas (2022) super-solar AGB yield tables into
data/yields/cinquegrana22.h5 and data/yields/yields.toml.

Source data: ~/Downloads/CK22_super_solar (or the path given with --src-dir).
Directory layout:
    z04/yields_m<M>z04.dat   Z = 0.04
    z05/yields_m<M>z05.dat   Z = 0.05
    ...
    z10/yields_m<M>z10.dat   Z = 0.10

File format (same as Karakas & Lugaro 2016/2018):
    Line 1 (col header part 1):  species  A  yield  mass(i)_lost  mass(i)_0  ...
    Line 2 (col header part 2):  X0(i)  log10(<X(i)>/X0(i))
    Remaining lines:              one isotope per row

The ``mass(i)_lost`` column (total isotope mass ejected) is read.
Species ``g`` (grain tracer) and ``n`` (neutron) are skipped.
``p`` maps to H-1, ``d`` to H-2.  Al-26 isomer notation (``al-6``,
``al*6``) is handled by stripping the suffix and summing yields into
the single (Z=13, A=26) bin.

Solar metallicity: Z_SOLAR = 0.014 (Asplund et al. 2009).
[Fe/H] = log10(Z / Z_SOLAR).

Mass grid:
  The union of all masses across all metallicities is used as the
  shared grid.  For each [Fe/H]:
    - masses outside [min_mass, max_mass] at that [Fe/H]: zero yields;
    - interior masses absent from that [Fe/H]'s data: linearly
      interpolated from the two bracketing data masses.

Run from the repository root:

    python3 data/tools/yields/import_cinquegrana22.py \\
        --src-dir ~/Downloads/CK22_super_solar

:copyright: Copyright (c) 2026 Mark Krumholz
"""

import argparse
import math
import os
import pathlib
import re
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
MODEL_NAME = "cinquegrana22"
CHANNEL = "agb"
Z_SOLAR = 0.014  # Asplund et al. 2009

REFERENCE = "Cinquegrana, G. C. & Karakas, A. I. 2022, MNRAS, 510, 1557"
REFERENCE_URL = "https://ui.adsabs.harvard.edu/abs/2022MNRAS.510.1557C/abstract"

# Subdirectory name → nominal Z value
SUBDIRS: list[tuple[str, float]] = [
    ("z04", 0.04),
    ("z05", 0.05),
    ("z06", 0.06),
    ("z07", 0.07),
    ("z08", 0.08),
    ("z09", 0.09),
    ("z10", 0.10),
]

# ---------------------------------------------------------------------------
# Element table
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

# Species that need special treatment: name → atomic Z, or None to skip
_SPECIAL: dict[str, int | None] = {
    "g": None,   # grain tracer — skip
    "n": None,   # neutron — skip
    "p": 1,      # proton → H
    "d": 1,      # deuterium → H
    "t": 1,      # tritium → H
}

_SPECIES_RE = re.compile(r"^([a-z]+)")
_MASS_I_LOST = "mass(i)_lost"


def parse_species(species: str, a: int) -> tuple[int, int] | None:
    """Return (Z, A) for a species token, or None to skip it."""
    s = species.lower()
    if s in _SPECIAL:
        z = _SPECIAL[s]
        return None if z is None else (z, a)
    m = _SPECIES_RE.match(s)
    if m is None:
        return None
    sym = m.group(1)
    z = SYMBOL_TO_Z.get(sym)
    return None if z is None else (z, a)


# ---------------------------------------------------------------------------
# Data record
# ---------------------------------------------------------------------------

@dataclass
class YieldRecord:
    mass: float
    feh: float
    yields: dict[tuple[int, int], float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# File parsing
# ---------------------------------------------------------------------------

# Regex to extract the stellar mass from filenames like yields_m1.5z04.dat
_FNAME_RE = re.compile(r"yields_m([\d.]+)z\d+\.dat$")


def parse_file(path: pathlib.Path, feh: float) -> YieldRecord | None:
    """Parse one yields_m*z*.dat file and return a YieldRecord."""
    m = _FNAME_RE.search(path.name)
    if m is None:
        warnings.warn(f"Cannot parse mass from filename: {path.name!r}; skipping")
        return None
    mass = float(m.group(1))

    lines = path.read_text().splitlines()

    # Find the column-header line (first token == "species")
    col_idx = None
    mass_lost_col = None
    for i, line in enumerate(lines):
        tokens = line.split()
        if tokens and tokens[0] == "species":
            try:
                mass_lost_col = tokens.index(_MASS_I_LOST)
            except ValueError:
                warnings.warn(f"No '{_MASS_I_LOST}' column in {path}")
                return None
            col_idx = i
            break
    if col_idx is None:
        warnings.warn(f"No 'species' column header in {path}")
        return None

    yields: dict[tuple[int, int], float] = {}
    for line in lines[col_idx + 1:]:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tokens = line.split()
        if len(tokens) <= mass_lost_col:
            continue
        try:
            a = int(tokens[1])
        except (ValueError, IndexError):
            continue
        za = parse_species(tokens[0], a)
        if za is None:
            continue
        try:
            yields[za] = yields.get(za, 0.0) + float(tokens[mass_lost_col])
        except ValueError:
            continue

    return YieldRecord(mass=mass, feh=feh, yields=yields)


# ---------------------------------------------------------------------------
# Record collection
# ---------------------------------------------------------------------------

def collect_records(src_dir: pathlib.Path) -> list[YieldRecord]:
    records: list[YieldRecord] = []
    for subdir, z_val in SUBDIRS:
        d = src_dir / subdir
        if not d.is_dir():
            raise FileNotFoundError(f"Required source directory not found: {d}")
        feh = math.log10(z_val / Z_SOLAR)
        for path in sorted(d.glob("yields_m*.dat")):
            rec = parse_file(path, feh)
            if rec is not None:
                records.append(rec)
    return records


# ---------------------------------------------------------------------------
# Array building with interpolation
# ---------------------------------------------------------------------------

def feh_group_name(feh: float) -> str:
    return f"feh_{feh:g}"


def build_arrays(records: list[YieldRecord]) -> dict:
    """Build yield arrays on the union mass grid with interpolation."""
    all_masses: set[float] = set()
    all_za: set[tuple[int, int]] = set()
    by_feh: dict[float, list[YieldRecord]] = {}

    for rec in records:
        all_masses.add(rec.mass)
        all_za.update(rec.yields.keys())
        by_feh.setdefault(rec.feh, []).append(rec)

    masses = sorted(all_masses)
    isotopes = sorted(all_za)
    fehs_sorted = sorted(by_feh.keys())

    isotope_z = np.array([za[0] for za in isotopes], dtype=np.float64)
    isotope_a = np.array([za[1] for za in isotopes], dtype=np.float64)
    iso_idx = {za: i for i, za in enumerate(isotopes)}

    feh_yields: dict[float, np.ndarray] = {}
    for feh in fehs_sorted:
        data_at_mass = {rec.mass: rec.yields for rec in by_feh[feh]}
        data_masses = sorted(data_at_mass)
        min_mass = data_masses[0]
        max_mass = data_masses[-1]

        arr = np.zeros((len(isotopes), len(masses)), dtype=np.float64)
        for j, m in enumerate(masses):
            if m < min_mass or m > max_mass:
                continue
            if m in data_at_mass:
                for za, val in data_at_mass[m].items():
                    if za in iso_idx:
                        arr[iso_idx[za], j] = val
            else:
                lo = max(dm for dm in data_masses if dm < m)
                hi = min(dm for dm in data_masses if dm > m)
                t = (m - lo) / (hi - lo)
                lo_y = data_at_mass[lo]
                hi_y = data_at_mass[hi]
                for za in set(lo_y) | set(hi_y):
                    if za not in iso_idx:
                        continue
                    arr[iso_idx[za], j] = (
                        lo_y.get(za, 0.0) + t * (hi_y.get(za, 0.0) - lo_y.get(za, 0.0))
                    )
        feh_yields[feh] = arr

    return {
        "masses":    np.array(masses, dtype=np.float64),
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

        grp = h5.create_group(CHANNEL)
        grp.create_dataset("masses",    data=arrays["masses"],    compression="gzip")
        grp.create_dataset("isotope_z", data=arrays["isotope_z"], compression="gzip")
        grp.create_dataset("isotope_a", data=arrays["isotope_a"], compression="gzip")

        for feh, yield_arr in sorted(arrays["feh_yields"].items()):
            feh_grp = grp.create_group(feh_group_name(feh))
            feh_grp.attrs["Fe_H"] = feh
            feh_grp.create_dataset("yield", data=yield_arr, compression="gzip")


# ---------------------------------------------------------------------------
# TOML registry
# ---------------------------------------------------------------------------

def update_registry(registry_path: pathlib.Path, h5_filename: str, arrays: dict) -> None:
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
        "--src-dir",
        required=True,
        metavar="DIR",
        help="Root of the CK22_super_solar directory tree",
    )
    p.add_argument(
        "--h5-dir",
        default=H5_DIR,
        help=f"Directory to write {MODEL_NAME}.h5 into (default: {H5_DIR})",
    )
    p.add_argument(
        "--registry",
        default=REGISTRY_PATH,
        help=f"Path to the yields registry TOML (default: {REGISTRY_PATH})",
    )
    args = p.parse_args()

    src_dir = pathlib.Path(args.src_dir).expanduser()
    h5_dir = pathlib.Path(args.h5_dir)
    registry_path = pathlib.Path(args.registry)

    print(f"Collecting records from {src_dir} ...")
    records = collect_records(src_dir)
    print(f"  {len(records)} (mass, [Fe/H]) records parsed")

    by_feh: dict[float, list] = {}
    for rec in records:
        by_feh.setdefault(rec.feh, []).append(rec)
    fehs = sorted(by_feh)
    print(f"  [Fe/H] values ({len(fehs)}):")
    for feh in fehs:
        masses = sorted(r.mass for r in by_feh[feh])
        print(f"    {feh:.5f}  ({len(masses)} masses: {masses[0]}–{masses[-1]} Msun)")

    print("Building yield arrays ...")
    arrays = build_arrays(records)
    n_iso = len(arrays["isotope_z"])
    n_mass = len(arrays["masses"])
    print(f"  isotopes: {n_iso}")
    print(f"  union mass grid ({n_mass}): {list(arrays['masses'])}")
    for feh in fehs:
        y = arrays["feh_yields"][feh]
        n_data = int(np.any(y != 0, axis=0).sum())
        print(f"  [Fe/H]={feh:.5f}: {n_data}/{n_mass} masses with data")

    h5_filename = f"{MODEL_NAME}.h5"
    h5_path = h5_dir / h5_filename
    write_h5(h5_path, arrays)
    print(f"Wrote {h5_path}")

    # Store the HDF5 path relative to the registry's own directory, not just
    # its basename, so a custom --h5-dir still resolves correctly when
    # YieldChannel reads it back relative to registry_path.parent
    registry_h5_path = os.path.relpath(h5_path, registry_path.parent)
    update_registry(registry_path, registry_h5_path, arrays)
    print(f"Updated {registry_path}")


if __name__ == "__main__":
    main()
