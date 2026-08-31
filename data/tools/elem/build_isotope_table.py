#!/usr/bin/env python3
"""Build data/elem/isotopes.h5 from slug2's plain-text isotope decay
table (lib/yields/isotope_data.txt in the slug2 repository).

Each line of the input file lists an isotope's atomic number Z, mass
number A, and lifetime (mean lifetime in seconds; -1 marks a stable
isotope). An unstable line (lifetime >= 0) is optionally followed by n
branching ratios and then n (Z, A) daughter pairs -- n is inferred
from the line's own token count, and can be 0: some isotopes are known
to decay but have no tabulated decay channel. This exactly mirrors
slug2's own parser (see src/yields/slug_isotopes.cpp's isotope_table
constructor) -- same file format, same token layout.

The output is a flat (CSR-style) layout rather than one HDF5 group per
isotope, since ~3000 isotopes would mean ~3000 group opens to read the
whole table back: parallel "Z"/"A"/"lifetime" datasets (one entry per
isotope; lifetime is 0 for stable isotopes rather than slug2's -1
sentinel, matching elem::IsotopeData's own convention), plus a
"daughterOffset" dataset (one entry per isotope, plus a final entry
giving the total daughter-entry count, so isotope i's daughters are
the half-open range [daughterOffset[i], daughterOffset[i+1]) into the
flat "daughterZ"/"daughterA"/"branchingRatio" datasets).

Run from the repository root:
    python3 data/tools/elem/build_isotope_table.py \\
        --input /path/to/slug2/lib/yields/isotope_data.txt

:copyright: Copyright (c) 2026 Mark Krumholz
"""
import argparse
import pathlib

import h5py
import numpy as np

H5_PATH = "data/elem/isotopes.h5"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True,
        help="Path to slug2's isotope_data.txt")
    p.add_argument("--h5", default=H5_PATH,
        help=f"Path to the output HDF5 file (default: {H5_PATH})")
    return p.parse_args()


def parse_isotope_file(path: str):
    """Parse slug2's isotope_data.txt, returning the flat CSR arrays
    described in this module's own docstring, as a tuple
    (Z, A, lifetime, daughterOffset, daughterZ, daughterA, branchingRatio)."""
    z_list: list[int] = []
    a_list: list[int] = []
    lifetime_list: list[float] = []
    offset_list: list[int] = [0]
    daughter_z: list[int] = []
    daughter_a: list[int] = []
    branch: list[float] = []

    with open(path, encoding="utf-8") as f:
        for lineno, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            tokens = line.split()
            if len(tokens) < 3:
                raise ValueError(f"{path}:{lineno}: badly formatted line {line!r}")

            z = int(tokens[0])
            a = int(tokens[1])
            lifetime = float(tokens[2])

            z_list.append(z)
            a_list.append(a)
            lifetime_list.append(0.0 if lifetime < 0 else lifetime)

            if lifetime >= 0:
                ndaughter = (len(tokens) - 3) // 3
                if len(tokens) != 3 + 3 * ndaughter:
                    raise ValueError(f"{path}:{lineno}: badly formatted line {line!r}")
                for i in range(ndaughter):
                    branch.append(float(tokens[i + 3]))
                    daughter_z.append(int(tokens[2 * i + 3 + ndaughter]))
                    daughter_a.append(int(tokens[2 * i + 4 + ndaughter]))

            offset_list.append(len(daughter_z))

    return (
        np.array(z_list, dtype=np.float64),
        np.array(a_list, dtype=np.float64),
        np.array(lifetime_list, dtype=np.float64),
        np.array(offset_list, dtype=np.float64),
        np.array(daughter_z, dtype=np.float64),
        np.array(daughter_a, dtype=np.float64),
        np.array(branch, dtype=np.float64),
    )


def write_h5(h5_path: str, z: np.ndarray, a: np.ndarray, lifetime: np.ndarray,
             offset: np.ndarray, daughter_z: np.ndarray, daughter_a: np.ndarray,
             branch: np.ndarray) -> None:
    pathlib.Path(h5_path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, "w") as h5file:
        h5file.create_dataset("Z", data=z, compression="gzip")
        h5file.create_dataset("A", data=a, compression="gzip")
        h5file.create_dataset("lifetime", data=lifetime, compression="gzip")
        h5file.create_dataset("daughterOffset", data=offset, compression="gzip")
        h5file.create_dataset("daughterZ", data=daughter_z, compression="gzip")
        h5file.create_dataset("daughterA", data=daughter_a, compression="gzip")
        h5file.create_dataset("branchingRatio", data=branch, compression="gzip")
        h5file.attrs["source"] = "slug2 lib/yields/isotope_data.txt"
        h5file.attrs["lifetime_units"] = (
            "seconds (mean lifetime = half-life / ln 2); 0 means stable")


def main() -> None:
    args = parse_args()
    z, a, lifetime, offset, daughter_z, daughter_a, branch = parse_isotope_file(args.input)
    write_h5(args.h5, z, a, lifetime, offset, daughter_z, daughter_a, branch)
    print(f"wrote {len(z)} isotopes ({len(daughter_z)} daughter entries) to {args.h5}")


if __name__ == "__main__":
    main()
