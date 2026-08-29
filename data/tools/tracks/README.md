# Track pipeline

This directory holds the scripts used to fetch and clean up the raw
stellar evolutionary track data slug ships as `data/tracks/*.h5`. Each
script documents its own purpose and algorithm in its module
docstring; this file only records the *order* the scripts for each
track set need to be run in to reproduce the production files. Every
script that reads an existing HDF5 file also accepts that file's path
as an argument, so all steps below can be re-run standalone against an
already-fetched file, not just as part of a fresh fetch.

## MIST

1. **`fetch_mist.py`** -- fetches the raw MIST track grid and writes
   `mist.h5`. Automatically calls `mist_truncate_wd.py` immediately
   afterward (step 2), unless run with `--no_truncate`.
2. **`mist_truncate_wd.py`** -- truncates any track that reaches white
   dwarf cooling (phase 6) down to just its first two phase-6 points,
   since slug has no use for a WD's own negligible light and wildly
   mismatched WD-track lengths break isochrone interpolation between
   neighbors. Runs automatically as part of step 1.
3. **`interpolate_mist.py`** -- fills mass gaps (missing or stalled/
   non-converged tracks) by interpolating between neighboring masses.
   Must run *after* step 2: it explicitly treats a track ending at
   phase 6 as a truncation, not a stall, so it needs the WD tails
   already cut before it can tell the two apart.
4. **`prune_mist_age_regressions.py`** -- cuts out any stretch of a
   track where age plateaus or regresses (typically floating-point
   noise around an already near-constant age during a fast
   transition), which otherwise crashes slug's exact-`[Fe/H]`-match
   fast path outright.

## PARSEC

PARSEC ships as two independently-fetched grids -- a rotating
(v/vcrit) grid for low/intermediate mass and a non-rotating
very-massive-star (VMS) grid -- that get spliced together into one
composite file. Steps 1-4 process each grid independently (VMS and
rotating tracks never interact); step 5 combines the two current
files, so it must be the last thing run, and re-run (with
`--overwrite`) any time either input file changes afterward.

1. **`fetch_parsec_vms.py`** -- fetches the raw VMS grid and writes
   `parsec_vms.h5`. Automatically calls `truncate_parsec_vms.py`
   immediately afterward (step 2), unless run with `--no_truncate`.
2. **`truncate_parsec_vms.py`** -- truncates every `[Fe/H]` group's
   mass grid down to the ~600 Msun range common to all of them, since
   the three most metal-poor groups alone extend to 2000 Msun and
   Tracks3D requires one shared mass grid across every group a query
   brackets. Runs automatically as part of step 1.
3. **`fetch_parsec_rot.py`** -- fetches the raw rotating grid and
   writes `parsec_rot.h5`.
4. **`interpolate_parsec_rot.py`** -- fills mass gaps, independently
   within each v/vcrit slice, by mass interpolation and (failing
   that) donation from a neighboring v/vcrit or `[Fe/H]`. Run
   standalone against `parsec_rot.h5` after step 3.
5. **`downsample_parsec.py`** -- run against *both* `parsec_vms.h5`
   and `parsec_rot.h5` (after steps 1-4). Thins each track's time
   sampling by greedily keeping only points that diverge enough from
   the last kept point, since some tracks carry tens of thousands of
   points sampled far finer than slug needs -- the main driver of
   Tracks3D's memory footprint.
6. **`prune_parsec_age_regressions.py`** -- run against both files
   again (after step 5). Cuts out any stretch of a track where age
   plateaus or genuinely regresses (PARSEC's raw output occasionally
   backtracks by a few percent), which otherwise crashes slug's
   exact-`[Fe/H]`-match fast path outright.
7. **`combine_parsec.py`** -- splices the current `parsec_vms.h5` and
   `parsec_rot.h5` together at a fixed mass (rotating grid below it,
   VMS grid at/above it) into `parsec_composite.h5`, giving one file
   with full mass coverage. Needs `--overwrite` to regenerate an
   existing output file.

## Stromlo

1. **`fetch_stromlo.py`** -- fetches the raw Stromlo track grid and
   writes `stromlo.h5`. Silently excludes any `(feh, v/vcrit)` pair
   listed in `STROMLO_EXCLUDE_FEH_VVCRIT` (currently just
   `feh=+0.6, v/vcrit=0.4`, the sole rotation rate at which that
   metal-rich endpoint was ever computed -- Tracks3D requires a
   uniform `[Fe/H]` grid across every v/vcrit a query brackets, so a
   single-v/vcrit point there is unusable and was dropped rather than
   left to crash slug at that exact `[Fe/H]`).

No further processing is currently needed for Stromlo.
