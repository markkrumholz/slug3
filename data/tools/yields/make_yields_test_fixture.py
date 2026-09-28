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
channels, at their one real Fe_H = 0.0. Reading real values (rather
than making up small round numbers by hand, the way most other
tests/*/assets fixtures in this project do) means the resulting unit
test can assert against numbers already independently verified by hand
against the original s18.2.yield_table/s100.yield_table text files.

The real Sukhbold et al. (2016) data is itself singular in [Fe/H]
(Solar-only), so it cannot supply a second, real metallicity to
exercise YieldChannel::yield()'s interpolation *across* Fe_H groups --
only across masses. A second, synthetic Fe_H = -1.0 group is added
here purely for that purpose: its yield array is exactly 2x the real
Fe_H = 0.0 array, elementwise, so any yield() call blending the two is
trivial to verify by hand (e.g. the Fe_H = -0.5 midpoint is exactly 1.5x
the Fe_H = 0.0 value). This does not represent a real [Fe/H] = -1
Sukhbold model and must never be read as one.

Also generates tests/yields/assets/yields_test_kobayashi.h5, a second,
independent fixture (registered as "kobayashi_test", ccsn only -- see
YieldChannel's own registry-per-model-per-file layout) extracted from
the real, already-imported data/yields/kobayashi06_11.h5: 3 progenitor
masses (13.0, 15.0, 18.0 Msun, its 3 lowest) and 3 isotopes (h1, fe56,
ni58 -- picked to actually exist in this source, unlike sukhbold_test's
own choice of ni56, which Kobayashi et al. 2006/2011 doesn't tabulate)
at Fe_H = 0.0. This is the fixture YieldChannel::rebuildMassGrid()'s own
tests use: real Sukhbold data only ever needs 2 masses to exercise
interpolation/extrapolation (a single pair brackets or is extrapolated
from), but exercising "some requested masses fall between native grid
points, some don't" at once needs a grid with more than 2 points, which
only Kobayashi's own (7-mass) data provides among what this project has
already imported.

Also generates tests/yields/assets/yields_test_gap.h5 (registered as
"gap_test", ccsn only), a fully synthetic fixture -- not extracted from
any real model -- for exercising YieldChannel::hasYield(mass, feh)'s
detection of failed-supernova gaps: masses where a yield table has an
entry, but every isotope's yield is zero (as in the real Sukhbold et
al. 2016 tables, which have several such gaps within their mass range).
It has 4 masses (10, 20, 30, 40 Msun) and 3 [Fe/H] groups (-2, -1, 0),
with the same 3 isotopes as sukhbold_test. The 20 and 30 Msun entries
are all zero at [Fe/H] = -1 and 0, and the 20 Msun entry is also
non-zero at [Fe/H] = -2, so the gap between 20 and 30 Msun is closed
for [Fe/H] between -2 and -1 but open for [Fe/H] between -1 and 0.
Every other entry is non-zero. The non-zero values themselves (a fixed
set of per-isotope values scaled by mass/10) are arbitrary.

Run from the repository root, after data/yields/sukhbold16.h5 and
data/yields/kobayashi06_11.h5 already exist (see import_yield_tables.py):
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

# Synthetic second Fe/H group -- see the module docstring's own
# explanation of why this can't be extracted from the real (singular
# in Fe/H) source data
SYNTHETIC_FEH = -1.0
SYNTHETIC_SCALE = 2.0

KOBAYASHI_SOURCE_H5 = "data/yields/kobayashi06_11.h5"
KOBAYASHI_DEST_H5 = "tests/yields/assets/yields_test_kobayashi.h5"
KOBAYASHI_MASSES = [13.0, 15.0, 18.0]
KOBAYASHI_ISOTOPES = [(1, 1), (26, 56), (28, 58)]  # (Z, A) for h1, fe56, ni58
KOBAYASHI_CHANNEL = "ccsn"
KOBAYASHI_MODEL_NAME = "kobayashi_test"
KOBAYASHI_REFERENCE = [
    "Kobayashi, Umeda, Nomoto, et al. 2006, ApJ, 653, 1145",
    "Kobayashi, Karakas, & Umeda, 2011, MNRAS, 414, 3231",
]
KOBAYASHI_REFERENCE_URL = [
    "https://ui.adsabs.harvard.edu/abs/2006ApJ...653.1145K/abstract",
    "https://ui.adsabs.harvard.edu/abs/2011MNRAS.414.3231K/abstract",
]


GAP_DEST_H5 = "tests/yields/assets/yields_test_gap.h5"
GAP_MODEL_NAME = "gap_test"
GAP_CHANNEL = "ccsn"
GAP_MASSES = [10.0, 20.0, 30.0, 40.0]
GAP_FEH = {"feh_neg2": -2.0, "feh_neg1": -1.0, "feh_0": 0.0}
GAP_ISOTOPES = ISOTOPES
GAP_BASE_YIELD = [1.0, 0.1, 0.05]  # per-isotope yield at 10 Msun, scaled by mass/10
# Masses whose yields are all zero, per [Fe/H] group -- see the module docstring
GAP_INACTIVE = {"feh_neg2": [30.0], "feh_neg1": [20.0, 30.0], "feh_0": [20.0, 30.0]}
GAP_REFERENCE = "Synthetic test fixture -- not a real yield model"


def make_gap_yields(feh_group: str) -> np.ndarray:
    """Return the synthetic gap_test yield array for one [Fe/H] group.

    Parameters
    ----------
    feh_group : str
        Key into GAP_FEH / GAP_INACTIVE.

    Returns
    -------
    numpy.ndarray
        Yields of shape (n_isotopes, n_masses), all zero at the masses
        listed in GAP_INACTIVE[feh_group].
    """
    yields = np.array([[y * m / 10.0 for m in GAP_MASSES] for y in GAP_BASE_YIELD])
    for m in GAP_INACTIVE[feh_group]:
        yields[:, GAP_MASSES.index(m)] = 0.0
    return yields


def write_gap_h5() -> None:
    """Write the synthetic gap_test fixture -- see the module docstring."""
    pathlib.Path(GAP_DEST_H5).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(GAP_DEST_H5, "w") as dest:
        dest.attrs["reference"] = GAP_REFERENCE
        dest.attrs["reference_url"] = ""
        grp = dest.create_group(GAP_CHANNEL)
        grp.create_dataset("masses", data=np.array(GAP_MASSES))
        grp.create_dataset("isotope_z", data=np.array([z for z, _ in GAP_ISOTOPES], dtype=np.float64))
        grp.create_dataset("isotope_a", data=np.array([a for _, a in GAP_ISOTOPES], dtype=np.float64))
        for group_name, feh in GAP_FEH.items():
            feh_grp = grp.create_group(group_name)
            feh_grp.attrs["Fe_H"] = feh
            feh_grp.create_dataset("yield", data=make_gap_yields(group_name))


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


def extract_kobayashi_channel(src: h5py.File) -> dict:
    grp = src[KOBAYASHI_CHANNEL]
    masses = grp["masses"][:]
    isotope_z = grp["isotope_z"][:]
    isotope_a = grp["isotope_a"][:]
    yield_full = grp["feh_0"]["yield"][:]  # shape (n_isotopes, n_masses)

    mass_idx = [int(np.nonzero(masses == m)[0][0]) for m in KOBAYASHI_MASSES]
    iso_idx = [int(np.nonzero((isotope_z == z) & (isotope_a == a))[0][0])
        for z, a in KOBAYASHI_ISOTOPES]

    return {
        "masses": np.array(KOBAYASHI_MASSES),
        "isotope_z": np.array([z for z, _ in KOBAYASHI_ISOTOPES], dtype=np.float64),
        "isotope_a": np.array([a for _, a in KOBAYASHI_ISOTOPES], dtype=np.float64),
        "yield": yield_full[np.ix_(iso_idx, mass_idx)],
    }


def write_kobayashi_h5(extracted: dict) -> None:
    pathlib.Path(KOBAYASHI_DEST_H5).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(KOBAYASHI_DEST_H5, "w") as dest:
        dest.attrs["reference"] = KOBAYASHI_REFERENCE
        dest.attrs["reference_url"] = KOBAYASHI_REFERENCE_URL
        grp = dest.create_group(KOBAYASHI_CHANNEL)
        grp.create_dataset("masses", data=extracted["masses"])
        grp.create_dataset("isotope_z", data=extracted["isotope_z"])
        grp.create_dataset("isotope_a", data=extracted["isotope_a"])
        feh_grp = grp.create_group("feh_0")
        feh_grp.attrs["Fe_H"] = 0.0
        feh_grp.create_dataset("yield", data=extracted["yield"])


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
            # Synthetic second Fe/H group -- see the module docstring
            synth_grp = grp.create_group("feh_neg1")
            synth_grp.attrs["Fe_H"] = SYNTHETIC_FEH
            synth_grp.create_dataset("yield", data=data["yield"] * SYNTHETIC_SCALE)


def write_registry() -> None:
    doc = tomlkit.document()
    doc["name"] = "Registry of nucleosynthetic yield models (test fixture)"
    doc["channels"] = CHANNELS
    for channel in CHANNELS:
        channel_table = tomlkit.table()
        models = [MODEL_NAME]
        if channel == KOBAYASHI_CHANNEL:
            models.append(KOBAYASHI_MODEL_NAME)
        if channel == GAP_CHANNEL:
            models.append(GAP_MODEL_NAME)
        channel_table["models"] = models
        model_table = tomlkit.table()
        model_table["reference"] = "Sukhbold, T., Ertl, T., Woosley, S. E., Brown, J. M., Janka, H.-T. 2016, ApJ, 821, 38"
        model_table["reference_url"] = "https://ui.adsabs.harvard.edu/abs/2016ApJ...821...38S/abstract"
        model_table["file"] = "yields_test.h5"
        model_table["Fe_H"] = [SYNTHETIC_FEH, 0.0]
        model_table["masses"] = MASSES
        channel_table[MODEL_NAME] = model_table

        if channel == KOBAYASHI_CHANNEL:
            kobayashi_table = tomlkit.table()
            kobayashi_table["reference"] = KOBAYASHI_REFERENCE
            kobayashi_table["reference_url"] = KOBAYASHI_REFERENCE_URL
            kobayashi_table["file"] = pathlib.Path(KOBAYASHI_DEST_H5).name
            kobayashi_table["Fe_H"] = [0.0]
            kobayashi_table["masses"] = KOBAYASHI_MASSES
            channel_table[KOBAYASHI_MODEL_NAME] = kobayashi_table

        if channel == GAP_CHANNEL:
            gap_table = tomlkit.table()
            gap_table["reference"] = GAP_REFERENCE
            gap_table["reference_url"] = ""
            gap_table["file"] = pathlib.Path(GAP_DEST_H5).name
            gap_table["Fe_H"] = sorted(GAP_FEH.values())
            gap_table["masses"] = GAP_MASSES
            channel_table[GAP_MODEL_NAME] = gap_table

        doc[channel] = channel_table

    pathlib.Path(DEST_REGISTRY).write_text(tomlkit.dumps(doc))


def main() -> None:
    with h5py.File(SOURCE_H5, "r") as src:
        extracted = {channel: extract_channel(src, channel) for channel in CHANNELS}
    write_h5(extracted)
    print(f"wrote {DEST_H5}")

    with h5py.File(KOBAYASHI_SOURCE_H5, "r") as src:
        kobayashi_extracted = extract_kobayashi_channel(src)
    write_kobayashi_h5(kobayashi_extracted)
    print(f"wrote {KOBAYASHI_DEST_H5}")

    write_gap_h5()
    print(f"wrote {GAP_DEST_H5}")

    write_registry()
    print(f"wrote {DEST_REGISTRY}")


if __name__ == "__main__":
    main()
