# Yield pipeline

This directory holds the scripts used to build and populate the nucleosynthetic
yield data that slug ships as `data/yields/*.h5` and the registry
`data/yields/yields.toml`. Each script documents its own purpose, input format,
and algorithm in its module docstring; this file records the *order* the scripts
need to be run in to reproduce every model from scratch.

The registry file `data/yields/yields.toml` is updated in-place by each import
script and does not need to be maintained by hand.

## Core import tool

**`import_yield_tables.py`** is the general-purpose importer for yield data
supplied as a directory of plain-text per-mass files, one file per progenitor
mass, each named `s<mass>.yield_table` (e.g. `s18.2.yield_table`). It handles
files with `[ejecta]`, `[wind]`, or both columns, mapping them to the `ccsn`
and `massive_star_winds` channels respectively. Failed supernovae (files with no
`[ejecta]` column) are written as exactly zero for the `ccsn` channel. It is
called once per `[Fe/H]` grid point, accumulating each metallicity's yields into
the same HDF5 file. This script is used for the Sukhbold (2016) and Kobayashi
(2006/2011) models; it is also called internally by `fetch_limongi_chieffi18.py`
(see below).

## Core-collapse supernova and massive-star wind models

### Sukhbold et al. (2016)

The Sukhbold (2016) data are not publicly downloadable; they are added by hand
from files supplied via private communication. Once those files are available
locally:

1. **`import_yield_tables.py`** -- run once pointing at the directory of
   `s*.yield_table` files for the Solar metallicity grid, passing
   `--ejecta-channel ccsn --wind-channel massive_star_winds --feh 0.0
   --name sukhbold16`. This produces `data/yields/sukhbold16.h5` and writes
   the registry entry.

No further processing is needed; Sukhbold (2016) is Solar-metallicity only.

### Kobayashi et al. (2006, 2011)

Like Sukhbold, the Kobayashi data are added from privately-supplied files.
Run **`import_yield_tables.py`** three times, once per metallicity
([Fe/H] = −1.30, −0.70, 0.0), each time pointing at the corresponding
directory of `s*.yield_table` files and passing `--ejecta-channel ccsn
--name kobayashi06_11 --feh <value>`. The three runs accumulate all three
metallicities into a single `data/yields/kobayashi06_11.h5`.

### Limongi & Chieffi (2018)

1. **`fetch_limongi_chieffi18.py`** -- downloads tables 8 and 9 from the CDS
   catalogue J/ApJS/237/13, converts them to the per-mass plain-text format that
   `import_yield_tables.py` expects, and calls it internally, once per rotation
   velocity (0, 150, 300 km/s). Produces three files,
   `data/yields/limongi_chieffi18_v0.h5`,
   `data/yields/limongi_chieffi18_v150.h5`, and
   `data/yields/limongi_chieffi18_v300.h5`, each with four metallicities
   ([Fe/H] = −3, −2, −1, 0) and both `ccsn` and `massive_star_winds` channels.
   The script handles the source's convention that stars above 25 M☉ collapse
   directly to black holes and are therefore absent from the supernova table
   (their wind yields alone are read from the wind table at those masses).

## AGB models

The four component AGB models can be imported in any order; the composite model
(`import_agb_composite.py`, step 5 below) must be run *last* because it reads
all four component files.

1. **`import_karakas16.py`** -- imports the Karakas & Lugaro (2016) and
   Karakas et al. (2018) AGB yield tables from a local directory tree (passed
   as `--ext-dir`; the data are supplied via private communication from A.
   Karakas). Produces eight HDF5 files, one per partial mixing zone (PMZ) value:
   `karakas-pmz0.h5`, `karakas-pmz1e-4.h5`, `karakas-pmz2e-4.h5`,
   `karakas-pmz5e-4.h5`, `karakas-pmz1e-3.h5`, `karakas-pmz2e-3.h5`,
   `karakas-pmz4e-3.h5`, `karakas-pmz6e-3.h5`. Covers seven metallicities
   (Z = 0.0002 to 0.03, [Fe/H] ≈ −1.85 to +0.33) and masses ≈ 0.9–8 M☉,
   varying by PMZ. Where both a Bloecker (1995) and a Vassiliadis & Wood (1993)
   mass-loss variant exist for the same (mass, PMZ) cell in the Z = 0.0002 and
   Z = 0.0028 grids, the script prefers the Bloecker variant per advice from A.
   Karakas.

2. **`import_doherty14.py`** -- imports the Doherty et al. (2014) super-AGB
   yields from a single multi-table ASCII file (passed as `--input`; available
   on request from C. Doherty). Produces `data/yields/doherty14.h5`. Covers
   three metallicities (Z = 0.004, 0.008, 0.02, [Fe/H] ≈ −0.70 to 0.0) and
   masses 6.5–9.0 M☉.

3. **`import_gilpons22.py`** -- fetches the Gil-Pons et al. (2022) very-low-
   metallicity AGB yields directly from the CDS catalogue J/A+A/668/A100.
   Produces `data/yields/gilpons22.h5`. Covers four metallicities
   (Z = 10⁻⁶ to 10⁻¹⁰, [Fe/H] ≈ −4.15 to −8.15) and masses 3.0–8.5 M☉.
   This script downloads its own data and requires no local files.

4. **`import_cinquegrana22.py`** -- imports the Cinquegrana & Karakas (2022)
   super-solar AGB yields from a local directory tree (passed as `--src-dir`;
   the data are supplied via private communication from G. Cinquegrana). Produces
   `data/yields/cinquegrana22.h5`. Covers seven super-solar metallicities
   (Z = 0.04 to 0.10, [Fe/H] ≈ +0.46 to +0.85) and masses 1.0–8.0 M☉.

5. **`import_agb_composite.py`** -- combines all four component AGB models into
   a single recommended composite grid, `data/yields/agb-composite.h5`. Must be
   run *after* steps 1–4 because it reads their output files. Source priority at
   each (mass, [Fe/H]) cell is Karakas > Doherty > Gil-Pons > Cinquegrana; in
   practice the sources cover largely non-overlapping regions of the
   (mass, [Fe/H]) plane. The Karakas PMZ value for each cell is selected
   automatically by mass- and metallicity-dependent rules described in the
   script's module docstring. Covers 19 metallicities ([Fe/H] ≈ −8.15 to +0.85)
   and masses 0.9–9.0 M☉.

## Test fixture

**`make_yields_test_fixture.py`** -- extracts a tiny subset of the already-
imported `data/yields/sukhbold16.h5` into `tests/yields/assets/yields_test.h5`
and a companion `tests/yields/assets/yields.toml`. The fixture covers 2 masses
(18.2 and 100 M☉), 3 isotopes (H-1, Fe-56, Ni-56), and 2 metallicities (the
real Solar [Fe/H] = 0 plus a synthetic [Fe/H] = −1 whose yields are exactly 2×
the Solar values, for testing interpolation by hand). Requires `sukhbold16.h5`
to be present. Run this script any time `sukhbold16.h5` changes.
