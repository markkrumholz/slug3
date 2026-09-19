#!/usr/bin/env python3
"""Fetch Limongi & Chieffi (2018) massive-star yields from CDS catalog
J/ApJS/237/13 and import them into data/yields/limongi_chieffi18_v<vel>.h5
and yields.toml -- one model per initial rotation velocity the source
provides (0, 150, 300 km/s), since these don't map onto the v/vcrit
system used elsewhere in this project and are simplest just treated as
three independent yield models.

The source (https://cdsarc.cds.unistra.fr/viz-bin/cat/J/ApJS/237/13)
provides two fixed-width tables this script actually needs (see its
own ReadMe for the full byte-by-byte layout of every table; only the
two below matter here):

    table8.dat ("Isotopic yields - Recommended set"): one row per
        (Vel, [Fe/H], isotope), giving that isotope's total yield
        (Msun) at 9 masses: 13, 15, 20, 25, 30, 40, 60, 80, 120 Msun.
    table9.dat ("Isotopic yields in the wind - Recommended set"): the
        same row layout, but only 4 mass columns (13, 15, 20, 25 Msun)
        and covering wind mass loss alone, not the supernova ejecta.

Limongi & Chieffi's own models assume every star above 25 Msun
collapses directly to a black hole with no explosion at all -- so
table8.dat only ever tabulates masses above 25 Msun at all because it
is *also* the source's only record of those higher-mass stars' own
wind yield (there being no supernova ejecta to add to it). This means:

- The wind yield, over the full 13-120 Msun grid, is table9.dat's own
  value at the 4 masses it covers (13, 15, 20, 25 Msun), and table8.dat's
  own value at the remaining 5 (30, 40, 60, 80, 120 Msun), where it
  already *is* the star's entire yield.
- The core-collapse supernova (ccsn) yield is table8.dat minus
  table9.dat at the first 4 masses (the explosive yield alone, with the
  wind contribution subtracted back out), and exactly zero at the
  remaining 5 -- not small or unmeasured, but genuinely zero, since
  Limongi & Chieffi's own models never explode those masses at all.

Every isotope table8.dat/table9.dat name is already present in this
project's own global isotope table (data/elem/isotopes.h5), verified
against a snapshot of this source's own 333-isotope network at import
time -- see this module's own parse_isotope() for the one quirk in how
the source spells isotope names (H1 written as bare "H").

Run from the repository root: python3 data/tools/yields/fetch_limongi_chieffi18.py

:copyright: Copyright (c) 2026 Mark Krumholz
"""
import pathlib
import sys

import numpy as np
import urllib3

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import import_yield_tables as iyt  # noqa: E402

BASE_URL = "https://cdsarc.cds.unistra.fr/ftp/J/ApJS/237/13/"
TABLE8_URL = BASE_URL + "table8.dat"
TABLE9_URL = BASE_URL + "table9.dat"

REFERENCE = "Limongi, M., & Chieffi, A. 2018, ApJS, 237, 13"
REFERENCE_URL = "https://ui.adsabs.harvard.edu/abs/2018ApJS..237...13L/abstract"

VELOCITIES = [0, 150, 300]  # km/s, this source's own three rotation rates
FEH_VALUES = [0, -1, -2, -3]

# table8.dat's own 9 mass columns, ascending, in the exact order the
# file itself lists them; table9.dat only ever has the first 4 (see
# this module's own docstring for why).
MASSES = [13, 15, 20, 25, 30, 40, 60, 80, 120]
N_TABLE9_MASSES = 4

# Byte ranges (0-indexed [start, end) Python slices, converted from the
# CDS ReadMe's own 1-indexed, inclusive byte-by-byte description) for
# table8.dat/table9.dat, which share an identical Vel/[Fe/H]/Iso layout
# and differ only in how many mass columns follow.
_VEL_COL = (0, 3)
_FEH_COL = (4, 6)
_ISO_COL = (7, 12)
_MASS_COLS = [(13, 23), (24, 34), (35, 45), (46, 56), (57, 67),
    (68, 78), (79, 89), (90, 100), (101, 111)]
_T8_MASS_COLS = _MASS_COLS
_T9_MASS_COLS = _MASS_COLS[:N_TABLE9_MASSES]

TableByKey = dict[tuple[int, int], dict[str, list[float]]]


def fetch_text(url: str) -> str:
    # No custom User-Agent: CDS sits behind an anti-bot layer (Anubis)
    # that allowlists honest non-browser clients by their own default
    # UA string (curl, urllib3, ...) but serves a JS proof-of-work
    # challenge page instead of the file to anything claiming to be a
    # real browser (e.g. a spoofed "Mozilla/5.0") without behaving like
    # one -- verified directly against this exact URL.
    http = urllib3.PoolManager()
    resp = http.request("GET", url)
    if resp.status != 200:
        raise RuntimeError(f"Failed to fetch {url}: HTTP {resp.status}")
    # Not ASCII: like the Kobayashi et al. source files
    # import_yield_tables.py's own parse_float() already works around,
    # this source's exponents occasionally use a Unicode minus sign
    # (U+2212) in place of ASCII '-' -- iyt.parse_float(), used below
    # instead of a bare float(), normalizes it back.
    return resp.data.decode("utf-8")


def parse_isotope(token: str) -> tuple[int, int]:
    """Resolve one table8.dat/table9.dat Iso token to (Z, A).

    Every isotope in this source is an element symbol immediately
    followed by its mass number (e.g. "He4", "Fe56"), except H1, the
    only isotope written with no trailing mass number at all (bare
    "H") -- verified against every one of this source's own 333
    isotopes, not just assumed.
    """
    if token == "H":
        return iyt.isotope_z_a("H1")
    return iyt.isotope_z_a(token)


def parse_table(text: str, mass_cols: list[tuple[int, int]]) -> TableByKey:
    """Parse one of table8.dat/table9.dat's full text.

    Returns
    -------
    table[(vel, feh)][iso] = [yield (Msun) at each of mass_cols' own
    masses, in order] -- one entry per (velocity, integer [Fe/H])
    combination the file holds, so the whole file is parsed once and
    then sliced per model by build_model() below.
    """
    table: TableByKey = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        vel = int(line[slice(*_VEL_COL)])
        feh = int(line[slice(*_FEH_COL)])
        iso = line[slice(*_ISO_COL)].strip()
        values = [iyt.parse_float(line[slice(*cols)]) for cols in mass_cols]
        table.setdefault((vel, feh), {})[iso] = values
    return table


def build_model(t8: TableByKey, t9: TableByKey, vel: int, feh: int) -> dict:
    """Derive one (velocity, [Fe/H]) combination's own wind/ccsn yields.

    Returns a dict in exactly the shape import_yield_tables.py's own
    write_h5()/update_registry() expect (see that module's own
    import_directory() docstring) -- see this module's own docstring
    for the physical reasoning behind the wind/ccsn split itself.
    """
    row8 = t8[(vel, feh)]
    row9 = t9[(vel, feh)]
    isotopes = sorted(row8.keys(), key=parse_isotope)
    isotope_z = np.array([parse_isotope(iso)[0] for iso in isotopes], dtype=np.float64)
    isotope_a = np.array([parse_isotope(iso)[1] for iso in isotopes], dtype=np.float64)

    n_mass = len(MASSES)
    wind_yield = np.zeros((len(isotopes), n_mass))
    ejecta_yield = np.zeros((len(isotopes), n_mass))
    for i, iso in enumerate(isotopes):
        vals8 = row8[iso]
        vals9 = row9[iso]
        for j in range(N_TABLE9_MASSES):
            wind_yield[i, j] = vals9[j]
            ejecta = vals8[j] - vals9[j]
            # Verified across the whole source (every isotope, mass,
            # velocity, and [Fe/H]) that table8 >= table9 always holds
            # here, so this is a real invariant of the data, not just
            # an assumption -- fail loudly rather than silently clip
            # or produce a negative yield if a future re-fetch of this
            # source ever violates it.
            assert ejecta >= -1e-6, (
                f"{iso} at Vel={vel}, [Fe/H]={feh}, mass={MASSES[j]}: "
                f"table8 ({vals8[j]!r}) < table9 ({vals9[j]!r})")
            ejecta_yield[i, j] = max(ejecta, 0.0)
        for j in range(N_TABLE9_MASSES, n_mass):
            wind_yield[i, j] = vals8[j]
            # ejecta_yield[i, j] stays 0.0: direct collapse to a black
            # hole at this mass, no explosion, nothing returned by ccsn.

    return {
        "masses": np.array(MASSES, dtype=np.float64),
        "isotope_z": isotope_z,
        "isotope_a": isotope_a,
        "ejecta_yield": ejecta_yield,
        "wind_yield": wind_yield,
        "has_wind": True,
    }


def main() -> None:
    print(f"fetching {TABLE8_URL}")
    t8 = parse_table(fetch_text(TABLE8_URL), _T8_MASS_COLS)
    print(f"fetching {TABLE9_URL}")
    t9 = parse_table(fetch_text(TABLE9_URL), _T9_MASS_COLS)

    for vel in VELOCITIES:
        name = f"limongi_chieffi18_v{vel}"
        h5_filename = f"{name}.h5"
        h5_path = str(pathlib.Path(iyt.H5_DIR) / h5_filename)
        for feh_int in FEH_VALUES:
            model = build_model(t8, t9, vel, feh_int)
            feh = float(feh_int)
            n_iso, n_mass = model["ejecta_yield"].shape
            print(f"{name}, [Fe/H] = {feh:g}: {n_mass} masses, {n_iso} isotopes")

            iyt.write_h5(h5_path, [REFERENCE], [REFERENCE_URL],
                "ccsn", "massive_star_winds", feh, model)
            iyt.update_registry(iyt.REGISTRY_PATH, h5_filename, name,
                [REFERENCE], [REFERENCE_URL],
                ["ccsn", "massive_star_winds"], feh, model["masses"])
        print(f"wrote {h5_path}")

    print(f"updated {iyt.REGISTRY_PATH}")


if __name__ == "__main__":
    main()
