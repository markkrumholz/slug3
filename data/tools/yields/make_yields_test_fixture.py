#!/usr/bin/env python3
"""Generate tests/yields/assets/yields_test.h5 and
tests/yields/assets/yields.toml, a tiny fixture for exercising
src/yields/YieldChannel.hpp against real Sukhbold et al. (2016) yield
values, without loading the full data/yields/sukhbold16.h5 (200
masses, 302 isotopes) in a unit test.

Extracts exactly 2 progenitor masses (18.2 Msun, an ordinary case with
both ejecta and wind yields; 100.0 Msun, a failed supernova with an
all-zero ccsn yield -- see import_yield_tables.py's own comment) and 3
isotopes (h1, fe56, ni56) from the real, already-imported
data/yields/sukhbold16.h5, for both the ccsn and massive_star_winds
channels, at their one shared Fe_H = 0.0. Reading real values (rather
than making up small round numbers by hand, the way most other
tests/*/assets fixtures in this project do) means the resulting unit
test can assert against numbers already independently verified by hand
against the original s18.2.yield_table/s100.yield_table text files.

Run from the repository root, after data/yields/sukhbold16.h5 already
exists (see import_yield_tables.py):
    python3 data/tools/yields/make_yields_test_fixture.py

:copyright: Copyright (c) 2026 Mark Krumholz
"""
import pathlib

import h5py
import numpy as np
import tomlkit

SOURCE_H5 = "data/yields/sukhbold16.h5"
DEST_H5 = "tests/yields/assets/yields_test.h5"
DEST_REGISTRY = "tests/yields/assets/yields.toml"

MASSES = [18.2, 100.0]
ISOTOPES = [(1, 1), (26, 56), (28, 56)]  # (Z, A) for h1, fe56, ni56
CHANNELS = ["ccsn", "massive_star_winds"]
MODEL_NAME = "sukhbold_test"


def extract_channel(src: h5py.File, channel: str) -> dict:
    grp = src[channel]
    masses = grp["masses"][:]
    isotope_z = grp["isotope_z"][:]
    isotope_a = grp["isotope_a"][:]
    yield_full = grp["feh_0"]["yield"][:]  # shape (n_isotopes, n_masses)

    mass_idx = [int(np.nonzero(masses == m)[0][0]) for m in MASSES]
    iso_idx = [int(np.nonzero((isotope_z == z) & (isotope_a == a))[0][0])
        for z, a in ISOTOPES]

    return {
        "masses": np.array(MASSES),
        "isotope_z": np.array([z for z, _ in ISOTOPES], dtype=np.float64),
        "isotope_a": np.array([a for _, a in ISOTOPES], dtype=np.float64),
        "yield": yield_full[np.ix_(iso_idx, mass_idx)],
    }


def write_h5(extracted: dict[str, dict]) -> None:
    pathlib.Path(DEST_H5).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(DEST_H5, "w") as dest:
        dest.attrs["reference"] = "Sukhbold, T., Ertl, T., Woosley, S. E., Brown, J. M., Janka, H.-T. 2016, ApJ, 821, 38"
        dest.attrs["reference_url"] = "https://ui.adsabs.harvard.edu/abs/2016ApJ...821...38S/abstract"
        for channel, data in extracted.items():
            grp = dest.create_group(channel)
            grp.create_dataset("masses", data=data["masses"])
            grp.create_dataset("isotope_z", data=data["isotope_z"])
            grp.create_dataset("isotope_a", data=data["isotope_a"])
            feh_grp = grp.create_group("feh_0")
            feh_grp.attrs["Fe_H"] = 0.0
            feh_grp.create_dataset("yield", data=data["yield"])


def write_registry() -> None:
    doc = tomlkit.document()
    doc["name"] = "Registry of nucleosynthetic yield models (test fixture)"
    doc["channels"] = CHANNELS
    for channel in CHANNELS:
        channel_table = tomlkit.table()
        channel_table["models"] = [MODEL_NAME]
        model_table = tomlkit.table()
        model_table["reference"] = "Sukhbold, T., Ertl, T., Woosley, S. E., Brown, J. M., Janka, H.-T. 2016, ApJ, 821, 38"
        model_table["reference_url"] = "https://ui.adsabs.harvard.edu/abs/2016ApJ...821...38S/abstract"
        model_table["file"] = "yields_test.h5"
        model_table["Fe_H"] = [0.0]
        model_table["masses"] = MASSES
        channel_table[MODEL_NAME] = model_table
        doc[channel] = channel_table

    pathlib.Path(DEST_REGISTRY).write_text(tomlkit.dumps(doc))


def main() -> None:
    with h5py.File(SOURCE_H5, "r") as src:
        extracted = {channel: extract_channel(src, channel) for channel in CHANNELS}

    write_h5(extracted)
    print(f"wrote {DEST_H5}")
    write_registry()
    print(f"wrote {DEST_REGISTRY}")


if __name__ == "__main__":
    main()
