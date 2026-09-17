#!/usr/bin/env python3
"""Import Doherty et al. (2014) super-AGB yield tables into
data/yields/doherty14.h5 and data/yields/yields.toml.

The source is a single multi-table ASCII file whose structure differs
from the s<mass>.yield_table layout that import_yield_tables.py handles:
each table begins with a header line of the form

    <mass>M Z=<Z> VW93

followed by a column-header row (Species, Yield, MassExp, ProdFact, ...)
and then one row per isotope. This script reads the ``MassExp`` column
(the mass expelled, i.e. the yield in Msun) from each table.

Species notation in the source file also differs slightly:

    - ``p`` denotes H-1 (a proton); this script maps it to ``h1``.
    - ``g`` appears at the end of some tables and does not correspond to
      a real isotope (it likely represents grain ejecta or a catch-all
      total); it is skipped entirely.
    - All other species use the same lowercase-symbol + mass-number form
      (e.g. ``he4``, ``fe56``) as import_yield_tables.py already parses.

Metallicity [Fe/H] is derived from the table's own Z value as
log10(Z / 0.02), with Z_solar = 0.02 (the value Doherty et al. adopt).

The three metallicities present (Z = 0.02, 0.008, 0.004, corresponding
to [Fe/H] = 0.0, ~-0.40, ~-0.70) do not share the same mass grid:

    Z = 0.02  : 7.0, 7.5, 8.0, 8.5, 9.0 Msun
    Z = 0.008 : 6.5, 7.0, 7.5, 8.0, 8.5 Msun
    Z = 0.004 : 6.5, 7.0, 7.5, 8.0 Msun

Because the C++ YieldChannel reader requires a single shared mass grid
at the channel-group level (see YieldChannel.cpp's readGroup()), this
script pads the missing entries with zero yields, giving a rectangular
union grid [6.5, 7.0, 7.5, 8.0, 8.5, 9.0] across all three
metallicities. A zero entry means the source simply has no model for
that mass at that metallicity, not that the yield is physically zero.

Output HDF5 structure (one channel ``agb``):

    doherty14.h5
    ├── attrs: reference, reference_url
    └── agb/
        ├── masses          (6,)  float64, Msun
        ├── isotope_z       (N,)  float64
        ├── isotope_a       (N,)  float64
        ├── feh_0/
        │   ├── attrs: Fe_H = 0.0
        │   └── yield       (N, 6)  float64, Msun
        ├── feh_-0.39794/
        │   ├── attrs: Fe_H = log10(0.008/0.02)
        │   └── yield       (N, 6)  float64, Msun
        └── feh_-0.69897/
            ├── attrs: Fe_H = log10(0.004/0.02)
            └── yield       (N, 6)  float64, Msun

Run from the repository root:

    python3 data/tools/yields/import_doherty14.py \\
        --input /path/to/doherty14a_table1.txt

:copyright: Copyright (c) 2026 Mark Krumholz
"""

import argparse
import math
import pathlib
import re

import h5py
import numpy as np
import tomlkit

H5_DIR = "data/yields"
REGISTRY_PATH = "data/yields/yields.toml"
MODEL_NAME = "doherty14"
CHANNEL = "agb"
REFERENCE = "Doherty, Gil-Pons, Lau, et al. 2014, MNRAS, 437, 195"
REFERENCE_URL = "https://ui.adsabs.harvard.edu/abs/2014MNRAS.437..195D/abstract"
Z_SOLAR = 0.02  # solar metallicity adopted by Doherty et al.

# Atomic symbols for all 118 elements, ordered by atomic number -- mirrors
# import_yield_tables.py and src/elem/ElemCommons.hpp's own Symbols enum.
ELEMENT_SYMBOLS = [
    "H", "He",
    "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba",
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn",
    "Fr", "Ra",
    "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",
    "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
]
SYMBOL_TO_Z = {sym.lower(): z for z, sym in enumerate(ELEMENT_SYMBOLS, start=1)}

# Matches the table-header lines like "7.0M Z=0.02 VW93" (with arbitrary whitespace)
_HEADER_RE = re.compile(r"(\d+\.?\d*)M\s+Z=([\d.]+)\s+VW93")
# Matches isotope symbols like "he4", "fe56", "h1"
_ISOTOPE_RE = re.compile(r"^([a-zA-Z]+)(\d+)$")


def isotope_z_a(label: str) -> tuple[int, int]:
    """Resolve an isotope label (e.g. 'fe56') to (Z, A).

    Returns None for labels that should be skipped (currently 'g').
    Raises ValueError for unrecognized or unknown labels.
    """
    m = _ISOTOPE_RE.match(label)
    if m is None:
        raise ValueError(f"unrecognized isotope label: {label!r}")
    symbol, mass_num = m.group(1), int(m.group(2))
    z = SYMBOL_TO_Z.get(symbol.lower())
    if z is None:
        raise ValueError(f"unknown element symbol {symbol!r} in label {label!r}")
    return z, mass_num


def parse_file(path: str) -> dict[tuple[float, float], dict[str, float]]:
    """Parse the Doherty14 multi-table file.

    Returns
    -------
    dict mapping (mass_msun, feh) to {isotope_label: mass_expelled_msun},
    where isotope_label is the normalised species key (e.g. 'h1', 'he4').
    The special 'p' species is mapped to 'h1'; the 'g' entry is dropped.
    """
    tables: dict[tuple[float, float], dict[str, float]] = {}
    current_mass: float | None = None
    current_feh: float | None = None
    in_data = False

    with open(path) as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue

            # Table header: "7.0M Z=0.02 VW93"
            hdr = _HEADER_RE.search(line)
            if hdr:
                current_mass = float(hdr.group(1))
                z_val = float(hdr.group(2))
                current_feh = math.log10(z_val / Z_SOLAR)
                in_data = False
                tables[(current_mass, current_feh)] = {}
                continue

            # Column-header row (starts with "Species")
            if line.startswith("Species"):
                in_data = True
                # locate the MassExp column index (0-based, after splitting)
                col_headers = line.split()
                try:
                    mass_exp_col = col_headers.index("MassExp")
                except ValueError:
                    raise ValueError(f"'MassExp' column not found in header: {line!r}")
                continue

            if not in_data or current_mass is None:
                continue

            tokens = line.split()
            if not tokens:
                continue

            species = tokens[0].lower()

            # Skip the non-isotope 'g' entry (grain ejecta or catch-all total)
            if species == "g":
                continue

            # Map 'p' (proton) to 'h1'
            if species == "p":
                species = "h1"

            mass_exp = float(tokens[mass_exp_col])
            tables[(current_mass, current_feh)][species] = mass_exp

    return tables


def build_arrays(tables: dict[tuple[float, float], dict[str, float]]) -> dict:
    """Build the rectangular (isotopes × masses) yield arrays for each [Fe/H].

    Because the source's mass grid varies by metallicity, the union of all
    masses is used; entries with no data are filled with zero.

    Returns a dict with keys:
        masses      : 1D ndarray, Msun, sorted ascending (union grid)
        isotope_z   : 1D ndarray, float64
        isotope_a   : 1D ndarray, float64
        feh_yields  : dict mapping float feh -> 2D ndarray (n_iso, n_mass)
    """
    all_masses: set[float] = set()
    all_isotopes: set[str] = set()
    fehs: set[float] = set()

    for (mass, feh), iso_table in tables.items():
        all_masses.add(mass)
        all_isotopes.update(iso_table.keys())
        fehs.add(feh)

    masses = sorted(all_masses)
    isotopes = sorted(all_isotopes, key=isotope_z_a)
    fehs_sorted = sorted(fehs)

    isotope_z = np.array([isotope_z_a(iso)[0] for iso in isotopes], dtype=np.float64)
    isotope_a = np.array([isotope_z_a(iso)[1] for iso in isotopes], dtype=np.float64)
    iso_index = {iso: i for i, iso in enumerate(isotopes)}
    mass_index = {m: j for j, m in enumerate(masses)}

    feh_yields: dict[float, np.ndarray] = {}
    for feh in fehs_sorted:
        arr = np.zeros((len(isotopes), len(masses)), dtype=np.float64)
        for (mass, f), iso_table in tables.items():
            if f != feh:
                continue
            j = mass_index[mass]
            for iso, val in iso_table.items():
                arr[iso_index[iso], j] = val
        feh_yields[feh] = arr

    return {
        "masses": np.array(masses, dtype=np.float64),
        "isotope_z": isotope_z,
        "isotope_a": isotope_a,
        "feh_yields": feh_yields,
    }


def feh_group_name(feh: float) -> str:
    """Canonical HDF5 group name for a given [Fe/H] value, matching the
    convention in import_yield_tables.py (``feh_{feh:g}``)."""
    return f"feh_{feh:g}"


def write_h5(h5_path: str, arrays: dict) -> None:
    pathlib.Path(h5_path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, "a") as h5:
        h5.attrs["reference"] = REFERENCE
        h5.attrs["reference_url"] = REFERENCE_URL

        grp = h5.require_group(CHANNEL)

        # masses/isotope_z/isotope_a: shared across all [Fe/H] subgroups
        for dset_name, data in (
            ("masses", arrays["masses"]),
            ("isotope_z", arrays["isotope_z"]),
            ("isotope_a", arrays["isotope_a"]),
        ):
            if dset_name in grp:
                del grp[dset_name]
            grp.create_dataset(dset_name, data=data, compression="gzip")

        # One subgroup per [Fe/H]
        for feh, yield_arr in sorted(arrays["feh_yields"].items()):
            gname = feh_group_name(feh)
            if gname in grp:
                del grp[gname]
            feh_grp = grp.create_group(gname)
            feh_grp.attrs["Fe_H"] = feh
            feh_grp.create_dataset("yield", data=yield_arr, compression="gzip")


def update_registry(registry_path: str, h5_filename: str, arrays: dict) -> None:
    reg = pathlib.Path(registry_path)
    doc = tomlkit.parse(reg.read_text()) if reg.exists() else tomlkit.document()

    if "name" not in doc:
        doc["name"] = "Registry of nucleosynthetic yield models"
    if "channels" not in doc:
        doc["channels"] = []

    channel_list = list(doc["channels"])
    if CHANNEL not in channel_list:
        channel_list.append(CHANNEL)
    doc["channels"] = channel_list

    channel_table = doc[CHANNEL] if CHANNEL in doc else tomlkit.table()
    models = list(channel_table["models"]) if "models" in channel_table else []
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

    reg.write_text(tomlkit.dumps(doc))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True,
        help="Path to doherty14a_table1.txt (or the equivalent file)")
    p.add_argument("--h5-dir", default=H5_DIR,
        help=f"Directory to write {MODEL_NAME}.h5 into (default: {H5_DIR})")
    p.add_argument("--registry", default=REGISTRY_PATH,
        help=f"Path to the yields registry TOML file (default: {REGISTRY_PATH})")
    args = p.parse_args()

    print(f"Parsing {args.input} ...")
    tables = parse_file(args.input)
    print(f"  found {len(tables)} (mass, [Fe/H]) tables")

    arrays = build_arrays(tables)
    n_iso = len(arrays["isotope_z"])
    n_mass = len(arrays["masses"])
    fehs = sorted(arrays["feh_yields"])
    print(f"  union mass grid ({n_mass} masses): {list(arrays['masses'])}")
    print(f"  isotopes: {n_iso}")
    print(f"  [Fe/H] values: {[round(f, 5) for f in fehs]}")
    for feh in fehs:
        # Report how many mass entries are non-zero for this metallicity
        nonzero_cols = int(np.any(arrays["feh_yields"][feh] != 0, axis=0).sum())
        print(f"    [Fe/H]={feh:.5f}: {nonzero_cols}/{n_mass} masses with data")

    h5_filename = f"{MODEL_NAME}.h5"
    h5_path = str(pathlib.Path(args.h5_dir) / h5_filename)
    write_h5(h5_path, arrays)
    print(f"Wrote {h5_path}")

    update_registry(args.registry, h5_filename, arrays)
    print(f"Updated {args.registry}")


if __name__ == "__main__":
    main()
