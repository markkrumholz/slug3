#!/usr/bin/env python3
"""Construct the composite AGB yield table (agb-composite.h5) by combining:

  - Karakas & Lugaro (2016, 2018): 7 metallicities, pmz selected per recipe
  - Doherty et al. (2014): adds higher-mass models and its own metallicities
  - Gil-Pons et al. (2022): very low metallicities (Z = 1e-6 to 1e-10)

Source files read from --h5-dir (default: data/yields):
    karakas-pmz{N}.h5  (8 files, one per pmz value)
    doherty14.h5
    gilpons22.h5

Priority for each (feh, mass) cell:
    Karakas > Doherty > Gil-Pons

For cells inside a source's mass range but absent from its grid, yields are
linearly interpolated from the two bracketing data masses.  For cells outside
a source's mass range, yields are zero.

PMZ selection rules for Karakas:
  When a single pmz value is available for a (feh, mass), it is used.
  When multiple pmz values are available, the following rules apply:

  Z ≈ 0.03  (feh > 0.2):
    M < 3.25                → pmz = 2e-3
    3.25 ≤ M < 4.25         → pmz = 1e-3
    4.25 ≤ M < 5            → pmz = 1e-4
    M ≥ 5                   → pmz = 0

  Z ≈ 0.014  (−0.15 < feh ≤ 0.05):
    M < 1.5                 → pmz = 0
    1.5 ≤ M ≤ 3             → pmz = 2e-3
    3 < M ≤ 4               → pmz = 1e-3
    4 < M ≤ 4.75            → pmz = 1e-4
    M > 4.75                → pmz = 0

  Z ≈ 0.007  (−0.45 < feh ≤ −0.15):
    M < 3                   → pmz = 0
    3 ≤ M < 4               → pmz = 1e-3
    4 ≤ M < 5               → pmz = 1e-4
    M ≥ 5                   → pmz = 0

  Z ≤ 0.0028  (feh ≤ −0.45):
    M > 4                   → pmz = 0
    2.5 < M ≤ 4             → pmz = 1e-3
    M ≤ 2.5                 → pmz = 2e-3

Run from the repository root:

    python3 data/tools/yields/import_agb_composite.py

:copyright: Copyright (c) 2026 Mark Krumholz
"""

import argparse
import pathlib
import warnings
from collections import defaultdict

import h5py
import numpy as np
import tomlkit

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHANNEL = "agb"
MODEL_NAME = "agb-composite"
H5_DIR = "data/yields"
REGISTRY_PATH = "data/yields/yields.toml"

REFERENCES = [
    "Karakas, A. I. & Lugaro, M. 2016, ApJ, 825, 26",
    "Karakas, A. I., Lugaro, M., Carlos, M., et al. 2018, MNRAS, 477, 421",
    "Doherty, C. L., Gil-Pons, P., Lau, H. H. B., et al. 2014, MNRAS, 437, 195",
    "Gil-Pons, P., Doherty, C. L., Campbell, S. W., et al. 2022, A&A, 668, A100",
]
REFERENCE_URLS = [
    "https://ui.adsabs.harvard.edu/abs/2016ApJ...825...26K/abstract",
    "https://ui.adsabs.harvard.edu/abs/2018MNRAS.477..421K/abstract",
    "https://ui.adsabs.harvard.edu/abs/2014MNRAS.437..195D/abstract",
    "https://ui.adsabs.harvard.edu/abs/2022A%26A...668A.100G/abstract",
]

# Karakas pmz files with their pmz values
KARAKAS_PMZ_FILES = [
    ("karakas-pmz0.h5",    0.0),
    ("karakas-pmz1e-4.h5", 1e-4),
    ("karakas-pmz2e-4.h5", 2e-4),
    ("karakas-pmz5e-4.h5", 5e-4),
    ("karakas-pmz1e-3.h5", 1e-3),
    ("karakas-pmz2e-3.h5", 2e-3),
    ("karakas-pmz4e-3.h5", 4e-3),
    ("karakas-pmz6e-3.h5", 6e-3),
]
DOHERTY_FILE = "doherty14.h5"
GILPONS_FILE = "gilpons22.h5"

# Rounding precision for [Fe/H] keys (avoids float-comparison mismatches)
_FEH_ROUND = 6


# ---------------------------------------------------------------------------
# PMZ selection
# ---------------------------------------------------------------------------

def get_preferred_pmz(feh: float, mass: float) -> float:
    """Return the preferred pmz for a given (feh, mass) cell."""
    if feh > 0.2:        # Z ≈ 0.03
        if mass < 3.25:
            return 2e-3
        elif mass < 4.25:
            return 1e-3
        elif mass < 5.0:
            return 1e-4
        else:
            return 0.0
    elif feh > -0.15:    # Z ≈ 0.014
        if mass < 1.5:
            return 0.0
        elif mass <= 3.0:
            return 2e-3
        elif mass <= 4.0:
            return 1e-3
        elif mass <= 4.75:
            return 1e-4
        else:
            return 0.0
    elif feh > -0.45:    # Z ≈ 0.007
        if mass < 3.0:
            return 0.0
        elif mass < 4.0:
            return 1e-3
        elif mass < 5.0:
            return 1e-4
        else:
            return 0.0
    else:                # Z ≤ 0.0028
        if mass > 4.0:
            return 0.0
        elif mass > 2.5:
            return 1e-3
        else:
            return 2e-3


# ---------------------------------------------------------------------------
# Isotope index
# ---------------------------------------------------------------------------

def build_isotope_index(
    h5_dir: pathlib.Path,
    all_files: list[str],
) -> tuple[list[tuple[int, int]], dict[tuple[int, int], int]]:
    """Return sorted (Z, A) pairs and an index dict from the union of all files."""
    all_za: set[tuple[int, int]] = set()
    for fname in all_files:
        path = h5_dir / fname
        if not path.exists():
            warnings.warn(f"Source file not found: {path}")
            continue
        with h5py.File(path) as h5:
            iz = h5["agb/isotope_z"][:].astype(int)
            ia = h5["agb/isotope_a"][:].astype(int)
            all_za.update(zip(iz.tolist(), ia.tolist()))
    isotopes = sorted(all_za)
    return isotopes, {za: i for i, za in enumerate(isotopes)}


# ---------------------------------------------------------------------------
# Source reading
# ---------------------------------------------------------------------------

def _reindex_map(
    src_iz: np.ndarray, src_ia: np.ndarray,
    iso_idx: dict[tuple[int, int], int],
) -> dict[int, int]:
    """Map source isotope positions to common isotope positions."""
    result: dict[int, int] = {}
    for i, (z, a) in enumerate(zip(src_iz.tolist(), src_ia.tolist())):
        j = iso_idx.get((int(z), int(a)))
        if j is not None:
            result[i] = j
    return result


def read_source(
    path: pathlib.Path,
    iso_idx: dict[tuple[int, int], int],
    n_iso: int,
) -> dict[float, dict[float, np.ndarray]]:
    """Read one HDF5 yield file into common-isotope-basis vectors.

    Returns {feh: {mass: yield_vec}} including only masses with non-zero data.
    """
    result: dict[float, dict[float, np.ndarray]] = {}
    with h5py.File(path) as h5:
        grp = h5["agb"]
        masses = grp["masses"][:].tolist()
        remap = _reindex_map(grp["isotope_z"][:], grp["isotope_a"][:], iso_idx)
        for k in grp:
            if not k.startswith("feh_"):
                continue
            feh = round(float(grp[k].attrs["Fe_H"]), _FEH_ROUND)
            y = grp[k]["yield"][:]
            mass_dict: dict[float, np.ndarray] = {}
            for j, m in enumerate(masses):
                col = y[:, j]
                if not np.any(col != 0):
                    continue
                common = np.zeros(n_iso, dtype=np.float64)
                for src_i, dst_j in remap.items():
                    common[dst_j] = col[src_i]
                mass_dict[float(m)] = common
            if mass_dict:
                result[feh] = mass_dict
    return result


# ---------------------------------------------------------------------------
# Karakas pmz selection
# ---------------------------------------------------------------------------

def load_karakas(
    h5_dir: pathlib.Path,
    iso_idx: dict[tuple[int, int], int],
    n_iso: int,
) -> dict[float, dict[float, np.ndarray]]:
    """Load all Karakas pmz files and apply the pmz selection rules.

    Returns {feh: {mass: yield_vec}} on the common isotope basis.
    """
    # Collect all options: {feh: {mass: {pmz: yield_vec}}}
    all_opts: dict[float, dict[float, dict[float, np.ndarray]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for fname, pmz in KARAKAS_PMZ_FILES:
        path = h5_dir / fname
        if not path.exists():
            warnings.warn(f"Karakas file not found, skipping: {path}")
            continue
        for feh, mass_dict in read_source(path, iso_idx, n_iso).items():
            for mass, vec in mass_dict.items():
                all_opts[feh][mass][pmz] = vec

    # Apply selection rules
    selected: dict[float, dict[float, np.ndarray]] = {}
    for feh, mass_dict in all_opts.items():
        selected[feh] = {}
        for mass, pmz_dict in mass_dict.items():
            preferred = get_preferred_pmz(feh, mass)
            if preferred in pmz_dict:
                chosen = preferred
            elif len(pmz_dict) == 1:
                chosen = next(iter(pmz_dict))
            else:
                available = sorted(pmz_dict)
                chosen = min(available, key=lambda p: abs(p - preferred))
                warnings.warn(
                    f"Preferred pmz={preferred} not available at "
                    f"feh={feh:.4f}, mass={mass}; using pmz={chosen}"
                )
            selected[feh][mass] = pmz_dict[chosen]
    return selected


# ---------------------------------------------------------------------------
# Source merging
# ---------------------------------------------------------------------------

def merge_sources(
    karakas: dict[float, dict[float, np.ndarray]],
    doherty: dict[float, dict[float, np.ndarray]],
    gilpons: dict[float, dict[float, np.ndarray]],
) -> dict[float, dict[float, np.ndarray]]:
    """Merge the three sources with priority: Karakas > Doherty > Gil-Pons.

    The merge is mass-by-mass within each feh: a higher-priority source's
    yield vector for a given mass overwrites any lower-priority source's.
    """
    merged: dict[float, dict[float, np.ndarray]] = {}

    # Gil-Pons first (lowest priority)
    for feh, mass_dict in gilpons.items():
        merged.setdefault(feh, {}).update(mass_dict)

    # Doherty (overrides Gil-Pons; adds Doherty-only metallicities)
    for feh, mass_dict in doherty.items():
        merged.setdefault(feh, {}).update(mass_dict)

    # Karakas (highest priority; overrides both at shared feh/mass cells)
    for feh, mass_dict in karakas.items():
        merged.setdefault(feh, {}).update(mass_dict)

    return merged


# ---------------------------------------------------------------------------
# Grid building with interpolation
# ---------------------------------------------------------------------------

def build_arrays(
    merged: dict[float, dict[float, np.ndarray]],
    isotopes: list[tuple[int, int]],
) -> dict:
    """Build the composite yield arrays on the union (feh, mass) grid.

    Masses outside a metallicity's data range receive zero yields.
    Interior masses not in the data receive linearly interpolated yields.
    """
    all_masses = sorted({m for md in merged.values() for m in md})
    fehs_sorted = sorted(merged.keys())
    n_iso = len(isotopes)

    feh_yields: dict[float, np.ndarray] = {}
    for feh in fehs_sorted:
        mass_data = merged[feh]
        data_masses = sorted(mass_data)
        if not data_masses:
            continue
        min_mass = data_masses[0]
        max_mass = data_masses[-1]

        arr = np.zeros((n_iso, len(all_masses)), dtype=np.float64)
        for j, m in enumerate(all_masses):
            if m < min_mass or m > max_mass:
                continue
            if m in mass_data:
                arr[:, j] = mass_data[m]
            else:
                lo = max(dm for dm in data_masses if dm < m)
                hi = min(dm for dm in data_masses if dm > m)
                t = (m - lo) / (hi - lo)
                arr[:, j] = mass_data[lo] + t * (mass_data[hi] - mass_data[lo])
        feh_yields[feh] = arr

    return {
        "masses":    np.array(all_masses, dtype=np.float64),
        "isotope_z": np.array([za[0] for za in isotopes], dtype=np.float64),
        "isotope_a": np.array([za[1] for za in isotopes], dtype=np.float64),
        "feh_yields": feh_yields,
    }


# ---------------------------------------------------------------------------
# HDF5 output
# ---------------------------------------------------------------------------

def feh_group_name(feh: float) -> str:
    return f"feh_{feh:g}"


def write_h5(h5_path: pathlib.Path, arrays: dict) -> None:
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, "w") as h5:
        h5.attrs["reference"] = REFERENCES
        h5.attrs["reference_url"] = REFERENCE_URLS

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
    model_table["reference"] = REFERENCES
    model_table["reference_url"] = REFERENCE_URLS
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
        help=f"Directory containing source HDF5 files and where the output is written (default: {H5_DIR})",
    )
    p.add_argument(
        "--registry",
        default=REGISTRY_PATH,
        help=f"Path to the yields registry TOML (default: {REGISTRY_PATH})",
    )
    args = p.parse_args()

    h5_dir = pathlib.Path(args.h5_dir)
    registry_path = pathlib.Path(args.registry)

    all_source_files = (
        [fname for fname, _ in KARAKAS_PMZ_FILES]
        + [DOHERTY_FILE, GILPONS_FILE]
    )

    # Build common isotope index
    print("Building common isotope index ...")
    isotopes, iso_idx = build_isotope_index(h5_dir, all_source_files)
    n_iso = len(isotopes)
    print(f"  {n_iso} isotopes in union")

    # Load sources
    print("Loading Karakas yields (applying pmz selection rules) ...")
    karakas = load_karakas(h5_dir, iso_idx, n_iso)
    karakas_fehs = sorted(karakas)
    karakas_masses = sorted({m for md in karakas.values() for m in md})
    print(f"  {sum(len(v) for v in karakas.values())} (feh, mass) cells")
    print(f"  fehs: {[round(f, 4) for f in karakas_fehs]}")
    print(f"  mass range: {karakas_masses[0]} – {karakas_masses[-1]} Msun")

    print("Loading Doherty yields ...")
    doherty = read_source(h5_dir / DOHERTY_FILE, iso_idx, n_iso)
    doherty_fehs = sorted(doherty)
    doherty_masses = sorted({m for md in doherty.values() for m in md})
    print(f"  {sum(len(v) for v in doherty.values())} (feh, mass) cells")
    print(f"  fehs: {[round(f, 4) for f in doherty_fehs]}")
    print(f"  mass range: {doherty_masses[0]} – {doherty_masses[-1]} Msun")

    print("Loading Gil-Pons yields ...")
    gilpons = read_source(h5_dir / GILPONS_FILE, iso_idx, n_iso)
    gilpons_fehs = sorted(gilpons)
    gilpons_masses = sorted({m for md in gilpons.values() for m in md})
    print(f"  {sum(len(v) for v in gilpons.values())} (feh, mass) cells")
    print(f"  fehs: {[round(f, 4) for f in gilpons_fehs]}")
    print(f"  mass range: {gilpons_masses[0]} – {gilpons_masses[-1]} Msun")

    # Merge
    print("Merging sources (priority: Karakas > Doherty > Gil-Pons) ...")
    merged = merge_sources(karakas, doherty, gilpons)
    all_fehs = sorted(merged)
    all_masses = sorted({m for md in merged.values() for m in md})
    print(f"  {len(all_fehs)} [Fe/H] values: {[round(f, 4) for f in all_fehs]}")
    print(f"  {len(all_masses)} masses: {all_masses[0]} – {all_masses[-1]} Msun")

    # Report Doherty-only fehs (informational)
    karakas_feh_set = set(karakas_fehs)
    doherty_only = [f for f in doherty_fehs if f not in karakas_feh_set]
    if doherty_only:
        print(f"  Doherty-only fehs added: {[round(f, 4) for f in doherty_only]}")

    # Build arrays
    print("Building composite grid (interpolating/zero-filling) ...")
    arrays = build_arrays(merged, isotopes)
    n_mass = len(arrays["masses"])
    for feh in all_fehs:
        y = arrays["feh_yields"][feh]
        n_data = int(np.any(y != 0, axis=0).sum())
        lo = arrays["masses"][np.any(y != 0, axis=0)][0] if n_data else float("nan")
        hi = arrays["masses"][np.any(y != 0, axis=0)][-1] if n_data else float("nan")
        print(f"  [Fe/H]={feh:8.4f}: {n_data}/{n_mass} masses with data  ({lo}–{hi} Msun)")

    # Write HDF5
    h5_filename = f"{MODEL_NAME}.h5"
    h5_path = h5_dir / h5_filename
    write_h5(h5_path, arrays)
    print(f"Wrote {h5_path}")

    # Update registry
    update_registry(registry_path, h5_filename, arrays)
    print(f"Updated {registry_path}")


if __name__ == "__main__":
    main()
