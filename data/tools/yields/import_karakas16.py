#!/usr/bin/env python3
"""Import Karakas & Lugaro (2016, 2018) AGB yield tables into per-pmz HDF5
files and register each model in data/yields/yields.toml.

Source data come from two directories:

  --ext-dir   "files_for_external" root, containing z0002models, z0006models,
              z001models, z0028models subdirectories.  Files here use the
              *standard* format: the first line is

                  # Initial mass = M, Z = Z, Y = Y, M_mix = PMZ

              and the second line is

                  # Final mass = ..., Mass expelled = ...

              followed by the column-header line

                  species  A  yield  mass(i)_lost  mass(i)_0  ...

              We read the ``mass(i)_lost`` column (total mass of each isotope
              ejected from the star).

  --slug2-dir root of slug2/lib/yields/AGB_Karakas16, containing z007models,
              z014models, z03models.  Files here use the *headerless* format:
              the column-header is on line 1, with no preceding mass/Z header.
              Mass and pmz are extracted from the filename.

Sources used (other subdirectories are skipped):

  files_for_external/z0002models/scaled_solar/stellar_yields/   Z=0.0002
  files_for_external/z0006models/yields/y0.24/                  Z=0.0006
  files_for_external/z001models/2019-new-Z=0.001models/yields/  Z=0.001
  files_for_external/z0028models/isotopic_yields/               Z=0.0028
  slug2/z007models/                                              Z=0.007
  slug2/z014models/                                             Z=0.014
  slug2/z03models/                                              Z=0.03

Species conventions:
  ``n``  (A=1) neutron — skipped.
  ``p``  (A=1) proton — mapped to H-1.
  ``d``  (A=2) deuterium — mapped to H-2.
  ``t``  (A=3) tritium — mapped to H-3.
  All other labels are parsed as element_symbol + mass_number (e.g. he4, c12).

Solar metallicity:  Z_SOLAR = 0.014  (Asplund et al. 2009, used by KL2016).
[Fe/H] = log10(Z / Z_SOLAR).

Partial mixing zone (pmz) values:
  Each unique pmz value produces a separate HDF5 file named karakas-pmzN.h5
  (N = 0, 1e-4, 2e-4, ... expressed as a compact string).

Mass grid:
  Within each HDF5 file (i.e. for each pmz value), all [Fe/H] subgroups share
  the same ``masses`` dataset — the union of every mass that appears across all
  metallicities at that pmz.  For a given [Fe/H]:
    - masses outside [min_mass, max_mass] at that [Fe/H] receive zero yields;
    - masses within range that are not exact data points are linearly
      interpolated isotope-by-isotope from the two bracketing data masses;
    - isotopes absent from a given model are assigned zero.

When overshoot and no-overshoot variants exist for the same (mass, Z, pmz),
only the no-overshoot variant is kept (it is processed first in sorted order).

For z0002models and z0028models, some masses have both a b95massloss and a
vw93massloss file.  Where both exist for the same (mass, pmz), the b95massloss
file is used (per advice from A. Karakas); the vw93massloss file is skipped.

Run from the repository root:

    python3 data/tools/yields/import_karakas16.py \\
        --ext-dir /path/to/files_for_external \\
        --slug2-dir /path/to/slug2/lib/yields/AGB_Karakas16

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
CHANNEL = "agb"
Z_SOLAR = 0.014  # Asplund et al. 2009

REFERENCES = [
    "Karakas, A. I. & Lugaro, M. 2016, ApJ, 825, 26",
    "Karakas, A. I., Lugaro, M., Carlos, M., et al. 2018, MNRAS, 477, 421",
]
REFERENCE_URLS = [
    "https://ui.adsabs.harvard.edu/abs/2016ApJ...825...26K/abstract",
    "https://ui.adsabs.harvard.edu/abs/2018MNRAS.477..421K/abstract",
]

# Subdirectories to read from ext-dir, with their known Z values and whether
# b95massloss is preferred over vw93massloss when both variants are present.
EXT_SUBDIRS = [
    ("z0002models/scaled_solar/stellar_yields", 0.0002, True),
    ("z0006models/yields/y0.24", 0.0006, False),
    ("z001models/2019-new-Z=0.001models/yields", 0.001, False),
    ("z0028models/isotopic_yields", 0.0028, True),
]

# Subdirectories to read from slug2-dir, with their known Z values.
SLUG2_SUBDIRS = [
    ("z007models", 0.007),
    ("z014models", 0.014),
    ("z03models", 0.030),
]

# ---------------------------------------------------------------------------
# Element table and isotope utilities
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

# Species that need special handling (lowercase key → (atomic_z, or None=skip))
_SPECIAL: dict[str, int | None] = {
    "n": None,   # neutron — skip
    "p": 1,      # proton  → H
    "d": 1,      # deuterium → H
    "t": 1,      # tritium → H
}

_SPECIES_RE = re.compile(r"^([a-z]+)")


def parse_species(species: str, a: int) -> tuple[int, int] | None:
    """Return (Z, A) for a given species string and mass-number column.

    Returns None for species that should be skipped (neutron or unrecognised
    non-species header tokens).  Isomer notation like ``al-6`` / ``al*6``
    (meaning Al-26 in ground/excited state) is handled by stripping suffixes
    after the alphabetic element symbol and using the A column for the mass
    number; yields for both states are summed into the same (Z, A) bin.
    """
    s = species.lower()
    if s in _SPECIAL:
        z_val = _SPECIAL[s]
        if z_val is None:
            return None
        return z_val, a

    m = _SPECIES_RE.match(s)
    if m is None:
        return None  # e.g. header continuation tokens like "X0(i)"
    sym = m.group(1)
    z_val = SYMBOL_TO_Z.get(sym)
    if z_val is None:
        return None  # unrecognised — skip silently
    return z_val, a


def isotope_sort_key(za: tuple[int, int]) -> tuple[int, int]:
    return za


# ---------------------------------------------------------------------------
# Data record
# ---------------------------------------------------------------------------

@dataclass
class YieldRecord:
    mass: float          # initial stellar mass, Msun
    feh: float           # [Fe/H] = log10(Z/Z_SOLAR)
    pmz: float           # partial mixing zone parameter, Msun
    # mapping (Z, A) -> mass_i_lost (Msun)
    yields: dict[tuple[int, int], float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Parsing utilities
# ---------------------------------------------------------------------------

_MASS_I_LOST_HEADER = "mass(i)_lost"

def _locate_col(col_headers: list[str], name: str) -> int:
    try:
        return col_headers.index(name)
    except ValueError:
        raise ValueError(
            f"Column {name!r} not found in header: {col_headers!r}"
        )


def _parse_data_lines(lines: list[str], mass_lost_col: int) -> dict[tuple[int, int], float]:
    """Parse data rows (after the column-header line) into (Z, A) -> mass_lost.

    Isomers (e.g. al-6 and al*6 both at A=26) are summed into the same bin.
    Unrecognised tokens (header continuations, blank lines, comments) are
    silently skipped.
    """
    yields: dict[tuple[int, int], float] = {}
    for line in lines:
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
        yields[za] = yields.get(za, 0.0) + float(tokens[mass_lost_col])
    return yields


def parse_standard_file(path: pathlib.Path) -> YieldRecord | None:
    """Parse a files_for_external style file (has # Initial mass = header).

    Returns None if the file cannot be parsed (wrong format).
    """
    lines = path.read_text().splitlines()
    if not lines:
        return None

    # Line 0: # Initial mass = M, Z = Z, Y = Y, M_mix = PMZ
    hdr = lines[0]
    if not hdr.startswith("# Initial mass"):
        return None

    mass_m = re.search(r"Initial mass\s*=\s*([\d.]+)", hdr)
    z_m = re.search(r"Z\s*=\s*([\d.E+\-]+)", hdr)
    mmix_m = re.search(r"M_mix\s*=\s*([\d.E+\-]+)", hdr)
    if not (mass_m and z_m and mmix_m):
        warnings.warn(f"Cannot parse header in {path}: {hdr!r}")
        return None

    mass = float(mass_m.group(1))
    z_val = float(z_m.group(1))
    pmz = float(mmix_m.group(1))
    feh = math.log10(z_val / Z_SOLAR)

    # Find the column-header line (contains "species")
    col_idx = None
    mass_lost_col = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("species") or (
            stripped.startswith("#") is False and "species" in stripped
        ):
            col_headers = stripped.split()
            if col_headers[0] == "species":
                col_idx = i
                mass_lost_col = _locate_col(col_headers, _MASS_I_LOST_HEADER)
                break

    if col_idx is None:
        warnings.warn(f"No 'species' column header found in {path}")
        return None

    yields = _parse_data_lines(lines[col_idx + 1:], mass_lost_col)
    return YieldRecord(mass=mass, feh=feh, pmz=pmz, yields=yields)


# Regex for slug2 filename: m<mass>z<Z-digits>[.-](nopmz or pmz=?<val>)
_SLUG2_FNAME = re.compile(
    r"^m([\d.]+)z[\d]+"           # m<mass>z<Z-digits>
    r"[-.]"                        # separator: - or .
    r"(nopmz|pmz[=]?(\d+e-?\d+))" # nopmz or pmz=<val> (e.g. 2e-3)
)


def parse_slug2_filename(path: pathlib.Path) -> tuple[float, float]:
    """Return (mass_msun, pmz) by parsing a slug2-style filename.

    Raises ValueError if the filename does not match the expected pattern.
    """
    stem = path.stem  # e.g. "m1.5z007.pmz=2e-3.noovershoot"
    m = _SLUG2_FNAME.match(stem)
    if m is None:
        raise ValueError(f"Cannot parse slug2 filename: {path.name!r}")
    mass = float(m.group(1))
    pmz_part = m.group(2)
    if pmz_part == "nopmz":
        pmz = 0.0
    else:
        pmz_str = m.group(3)
        pmz = float(pmz_str)
    return mass, pmz


def parse_headerless_file(
    path: pathlib.Path, z_val: float
) -> YieldRecord | None:
    """Parse a slug2-style file (no # Initial mass header; data starts on line 1)."""
    try:
        mass, pmz = parse_slug2_filename(path)
    except ValueError as exc:
        warnings.warn(str(exc))
        return None

    feh = math.log10(z_val / Z_SOLAR)
    lines = path.read_text().splitlines()
    if not lines:
        return None

    # First non-blank, non-comment line with "species" is the column header
    col_idx = None
    mass_lost_col = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        tokens = stripped.split()
        if tokens[0] == "species":
            col_idx = i
            mass_lost_col = _locate_col(tokens, _MASS_I_LOST_HEADER)
            break
    if col_idx is None:
        warnings.warn(f"No 'species' column header found in {path}")
        return None

    yields = _parse_data_lines(lines[col_idx + 1:], mass_lost_col)
    return YieldRecord(mass=mass, feh=feh, pmz=pmz, yields=yields)


# ---------------------------------------------------------------------------
# Record collection
# ---------------------------------------------------------------------------

def _filter_prefer_b95(src: pathlib.Path) -> list[pathlib.Path]:
    """Return sorted files from *src*, skipping vw93 files where a b95 file
    exists for the same (mass, pmz).  Files with neither label are kept as-is.
    """
    all_files = sorted(src.glob("m*.dat"))
    b95_files = [f for f in all_files if "b95" in f.name]
    if not b95_files:
        return all_files

    # Parse headers of b95 files to build (mass, pmz) exclusion set
    b95_keys: set[tuple[float, float]] = set()
    for path in b95_files:
        rec = parse_standard_file(path)
        if rec is not None:
            b95_keys.add((rec.mass, rec.pmz))

    # Keep all non-vw93 files; keep vw93 only when no b95 covers that (mass, pmz)
    result: list[pathlib.Path] = []
    for path in all_files:
        if "vw93" not in path.name:
            result.append(path)
            continue
        rec = parse_standard_file(path)
        if rec is not None and (rec.mass, rec.pmz) in b95_keys:
            warnings.warn(
                f"Skipping {path.name}: b95massloss variant preferred "
                f"for mass={rec.mass}, pmz={rec.pmz}"
            )
        else:
            result.append(path)
    return result


def collect_records(
    ext_dir: pathlib.Path,
    slug2_dir: pathlib.Path,
) -> list[YieldRecord]:
    """Walk both source trees and return all successfully parsed YieldRecords."""
    records: list[YieldRecord] = []

    # files_for_external sources (standard format)
    for subdir, _z, prefer_b95 in EXT_SUBDIRS:
        src = ext_dir / subdir
        if not src.is_dir():
            raise FileNotFoundError(f"Required source directory not found: {src}")
        files = _filter_prefer_b95(src) if prefer_b95 else sorted(src.glob("m*.dat"))
        for path in files:
            rec = parse_standard_file(path)
            if rec is not None:
                records.append(rec)

    # slug2 sources (headerless format)
    # Seen set prevents overshoot duplicates: key = (mass, feh_rounded, pmz)
    seen: set[tuple[float, float, float]] = set()
    for subdir, z_val in SLUG2_SUBDIRS:
        src = slug2_dir / subdir
        if not src.is_dir():
            raise FileNotFoundError(f"Required source directory not found: {src}")
        feh = math.log10(z_val / Z_SOLAR)
        for path in sorted(src.glob("m*.dat")):  # sorted → noovershoot first
            rec = parse_headerless_file(path, z_val)
            if rec is None:
                continue
            key = (rec.mass, round(feh, 8), rec.pmz)
            if key in seen:
                warnings.warn(
                    f"Duplicate (mass={rec.mass}, Z={z_val}, pmz={rec.pmz}) "
                    f"from {path.name}; skipping (keeping first/noovershoot variant)"
                )
                continue
            seen.add(key)
            records.append(rec)

    return records


# ---------------------------------------------------------------------------
# Array building with interpolation
# ---------------------------------------------------------------------------

def pmz_label(pmz: float) -> str:
    """Compact string label for a pmz value used in filenames and TOML keys."""
    if pmz == 0.0:
        return "0"
    exp = int(math.floor(math.log10(pmz) + 1e-9))
    coef = round(pmz / 10.0**exp)
    return f"{coef}e{exp}"


def feh_group_name(feh: float) -> str:
    return f"feh_{feh:g}"


def build_arrays(records: list[YieldRecord]) -> dict:
    """Build yield arrays for one pmz group.

    Returns
    -------
    dict with keys:
      masses      : sorted 1-D ndarray, Msun  (union of all masses)
      isotope_z   : 1-D ndarray float64
      isotope_a   : 1-D ndarray float64
      feh_yields  : dict mapping float feh -> 2-D ndarray (n_iso, n_mass)
                    values are mass_i_lost (Msun), zero for out-of-range or
                    missing entries
    """
    # Collect global union of masses and isotopes; group records by feh
    all_masses: set[float] = set()
    all_za: set[tuple[int, int]] = set()
    by_feh: dict[float, list[YieldRecord]] = {}

    for rec in records:
        all_masses.add(rec.mass)
        all_za.update(rec.yields.keys())
        by_feh.setdefault(rec.feh, []).append(rec)

    masses = sorted(all_masses)
    isotopes = sorted(all_za, key=isotope_sort_key)
    fehs_sorted = sorted(by_feh.keys())

    isotope_z = np.array([za[0] for za in isotopes], dtype=np.float64)
    isotope_a = np.array([za[1] for za in isotopes], dtype=np.float64)
    iso_idx: dict[tuple[int, int], int] = {za: i for i, za in enumerate(isotopes)}
    mass_idx: dict[float, int] = {m: j for j, m in enumerate(masses)}

    feh_yields: dict[float, np.ndarray] = {}
    for feh in fehs_sorted:
        feh_records = by_feh[feh]
        # Build a lookup: mass -> {(Z,A): mass_lost}
        data_at_mass: dict[float, dict[tuple[int, int], float]] = {
            rec.mass: rec.yields for rec in feh_records
        }
        data_masses_sorted = sorted(data_at_mass.keys())
        min_mass = data_masses_sorted[0]
        max_mass = data_masses_sorted[-1]

        arr = np.zeros((len(isotopes), len(masses)), dtype=np.float64)

        for j, m in enumerate(masses):
            if m < min_mass or m > max_mass:
                # Outside the modelled range: leave zeros
                continue

            if m in data_at_mass:
                # Exact match
                iso_yields = data_at_mass[m]
                for za, val in iso_yields.items():
                    if za in iso_idx:
                        arr[iso_idx[za], j] = val
            else:
                # Linear interpolation between the two bracketing masses
                lo = max(dm for dm in data_masses_sorted if dm < m)
                hi = min(dm for dm in data_masses_sorted if dm > m)
                t = (m - lo) / (hi - lo)
                lo_yields = data_at_mass[lo]
                hi_yields = data_at_mass[hi]
                all_iso_here = set(lo_yields) | set(hi_yields)
                for za in all_iso_here:
                    if za not in iso_idx:
                        continue
                    v_lo = lo_yields.get(za, 0.0)
                    v_hi = hi_yields.get(za, 0.0)
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
        h5.attrs["reference"] = REFERENCES
        h5.attrs["reference_url"] = REFERENCE_URLS

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
    model_name: str,
    h5_filename: str,
    arrays: dict,
) -> None:
    doc = tomlkit.parse(registry_path.read_text()) if registry_path.exists() else tomlkit.document()

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
    if model_name not in models:
        models.append(model_name)
        models.sort()
    channel_table["models"] = models

    model_table = tomlkit.table()
    model_table["reference"] = REFERENCES
    model_table["reference_url"] = REFERENCE_URLS
    model_table["file"] = h5_filename
    model_table["Fe_H"] = sorted(arrays["feh_yields"].keys())
    model_table["masses"] = [float(m) for m in arrays["masses"]]
    channel_table[model_name] = model_table
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
        "--ext-dir",
        required=True,
        metavar="DIR",
        help="Root of the files_for_external directory tree",
    )
    p.add_argument(
        "--slug2-dir",
        required=True,
        metavar="DIR",
        help="Root of slug2/lib/yields/AGB_Karakas16 directory",
    )
    p.add_argument(
        "--h5-dir",
        default=H5_DIR,
        help=f"Directory to write karakas-pmzN.h5 files into (default: {H5_DIR})",
    )
    p.add_argument(
        "--registry",
        default=REGISTRY_PATH,
        help=f"Path to yields registry TOML (default: {REGISTRY_PATH})",
    )
    args = p.parse_args()

    ext_dir = pathlib.Path(args.ext_dir)
    slug2_dir = pathlib.Path(args.slug2_dir)
    h5_dir = pathlib.Path(args.h5_dir)
    registry_path = pathlib.Path(args.registry)

    print("Collecting yield records ...")
    records = collect_records(ext_dir, slug2_dir)
    print(f"  {len(records)} records parsed")

    # Group by pmz
    by_pmz: dict[float, list[YieldRecord]] = {}
    for rec in records:
        by_pmz.setdefault(rec.pmz, []).append(rec)

    print(f"  {len(by_pmz)} distinct pmz values: "
          f"{sorted(by_pmz.keys())}")

    for pmz, pmz_records in sorted(by_pmz.items()):
        label = pmz_label(pmz)
        model_name = f"karakas-pmz{label}"
        h5_filename = f"{model_name}.h5"
        h5_path = h5_dir / h5_filename

        print(f"\n--- pmz={pmz} ({label}): {len(pmz_records)} records ---")

        arrays = build_arrays(pmz_records)
        n_iso = len(arrays["isotope_z"])
        n_mass = len(arrays["masses"])
        fehs = sorted(arrays["feh_yields"])
        print(f"  union mass grid ({n_mass} masses): "
              f"{list(arrays['masses'])}")
        print(f"  isotopes: {n_iso}")
        print(f"  [Fe/H] values ({len(fehs)}): "
              f"{[round(f, 4) for f in fehs]}")
        for feh in fehs:
            y = arrays["feh_yields"][feh]
            n_data = int(np.any(y != 0, axis=0).sum())
            print(f"    [Fe/H]={feh:.5f}: {n_data}/{n_mass} masses with data")

        write_h5(h5_path, arrays)
        print(f"  Wrote {h5_path}")

        # Store the HDF5 path relative to the registry's own directory, not
        # just its basename, so a custom --h5-dir still resolves correctly
        # when YieldChannel reads it back relative to registry_path.parent
        registry_h5_path = os.path.relpath(h5_path, registry_path.parent)
        update_registry(registry_path, model_name, registry_h5_path, arrays)
        print(f"  Updated {registry_path}")


if __name__ == "__main__":
    main()
