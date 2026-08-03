#!/usr/bin/env python3
"""Add (or update) a single extinction curve in data/extinct/extinct.h5
and data/extinct/extinct.toml.

Unlike the fetch_*.py scripts, most extinction curve data slug uses
does not exist in a publicly-fetchable digital form -- it is added by
hand from files supplied via private communication, each documented
with its own literature reference. This script takes one such curve,
given as a plain two-column (wavelength in Angstrom, kappa in
arbitrary units -- only the curve's shape matters, since a user scales
it by a supplied A_V) whitespace-delimited text file, and writes it
into extinct.h5 as a new (or replaced) HDF5 group named --name, with
"wavelength"/"kappa" datasets and "reference"/"reference_url"
attributes, then updates extinct.toml to match -- adding --name to the
top-level "curves" list (if not already present) and writing/replacing
the [--name] table with the same reference/reference_url, plus
--description. Like spectra.toml (see fetch_filter_vo.py's own
update_registry() docstring for the contrast with filters.toml), only
this one curve's own entry is touched -- every other curve already in
extinct.toml/extinct.h5 is left exactly as it was.

Run from the repository root, e.g.:
    python3 data/tools/add_extinction_curve.py \\
        --input /path/to/SB_ATT_SLUG.dat \\
        --name Calzetti_starburst \\
        --reference "Calzetti, D., Armus, L., Bohlin, R., et al. 2000, ApJ, 533, 682" \\
        --reference-url "https://ui.adsabs.harvard.edu/abs/2000ApJ...533..682C/abstract" \\
        --description "Calzetti starburst attenuation curve"
"""
import argparse
import pathlib

import h5py
import numpy as np
import tomlkit

H5_PATH = "data/extinct/extinct.h5"
REGISTRY_PATH = "data/extinct/extinct.toml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True,
        help="Path to a two-column (wavelength, kappa) text file")
    p.add_argument("--name", required=True,
        help="Name of this curve, e.g. Calzetti_starburst")
    p.add_argument("--reference", required=True, nargs="+",
        help="One or more literature references, e.g. "
             "'Calzetti, D., et al. 2000, ApJ, 533, 682'; written as a "
             "plain string if exactly one is given, or a TOML array of "
             "strings otherwise")
    p.add_argument("--reference-url", nargs="*", default=[],
        help="URL(s) for --reference, e.g. ADS abstract link(s); same "
             "scalar-or-array rule as --reference, and independently "
             "sized -- a reference with no URL (e.g. a personal "
             "communication) is simply omitted here")
    p.add_argument("--description", required=True,
        help="Short human-readable description of this curve")
    p.add_argument("--h5", default=H5_PATH,
        help=f"Path to the extinction HDF5 file (default: {H5_PATH})")
    p.add_argument("--registry", default=REGISTRY_PATH,
        help=f"Path to the extinction registry TOML file (default: {REGISTRY_PATH})")
    return p.parse_args()


def scalar_or_list(values: list[str]) -> str | list[str]:
    """A single value collapses to a plain string; more than one stays
    a list, written as a TOML/HDF5-attribute array."""
    return values[0] if len(values) == 1 else list(values)


def write_h5(h5_path: str, name: str, wavelength: np.ndarray, kappa: np.ndarray,
             reference: list[str], reference_url: list[str]) -> None:
    pathlib.Path(h5_path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, "a") as h5file:
        if name in h5file:
            del h5file[name]
        grp = h5file.create_group(name)
        grp.create_dataset("wavelength", data=wavelength, compression="gzip")
        grp.create_dataset("kappa", data=kappa, compression="gzip")
        grp.attrs["reference"] = scalar_or_list(reference)
        if reference_url:
            grp.attrs["reference_url"] = scalar_or_list(reference_url)


def update_registry(registry_path: str, h5_path: str, name: str,
                     reference: list[str], reference_url: list[str],
                     description: str) -> None:
    reg_file = pathlib.Path(registry_path)
    if reg_file.exists():
        doc = tomlkit.parse(reg_file.read_text())
    else:
        doc = tomlkit.document()
        doc["name"] = "Registry of extinction curves"
        doc["file"] = pathlib.Path(h5_path).name
        doc["curves"] = []

    curves = list(doc["curves"])
    if name not in curves:
        curves.append(name)
        curves.sort()
    doc["curves"] = curves

    curve_table = tomlkit.table()
    curve_table["reference"] = scalar_or_list(reference)
    if reference_url:
        curve_table["reference_url"] = scalar_or_list(reference_url)
    curve_table["description"] = description
    doc[name] = curve_table

    reg_file.write_text(tomlkit.dumps(doc))


def main() -> None:
    args = parse_args()

    data = np.loadtxt(args.input)
    wavelength, kappa = data[:, 0], data[:, 1]
    if not np.all(np.diff(wavelength) > 0):
        raise ValueError(f"{args.input}: wavelength column must be strictly increasing")

    write_h5(args.h5, args.name, wavelength, kappa, args.reference, args.reference_url)
    print(f"wrote curve '{args.name}' to {args.h5}")

    update_registry(args.registry, args.h5, args.name,
        args.reference, args.reference_url, args.description)
    print(f"updated {args.registry}")


if __name__ == "__main__":
    main()
