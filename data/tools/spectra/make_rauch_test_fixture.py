#!/usr/bin/env python3
"""Generate tests/specsyn/assets/RAUCH_test.h5, a tiny synthetic
fixture for exercising SpecsynLibWD/SpecsynLib2D's handling of a
partially-filled (logg, log_Teff) grid -- the schema
data/tools/spectra/fetch_rauch.py writes for the real Rauch et al. NLTE hot
star grid, whose (logg, Teff) coverage is not a full rectangular grid.
Not real Rauch data: a flat "wl"/"logg"/"log_Teff"/"flux" schema on a
3x2 (logg, log_Teff) grid, with only one corner missing -- mirroring
the real grid's own structure, where only the hottest/lowest-log(g)
corner is absent rather than the grid being sparse throughout:

    logg \\ log_Teff   4.00    4.50
    7.0                1.0     2.0
    8.0                3.0     4.0
    9.0                5.0    (missing)

This leaves one bracketing cell -- logg in [7.0, 8.0] -- completely
populated (all 4 corners present), and the other -- logg in
[8.0, 9.0] -- with its (logg=9.0, log_Teff=4.5) corner missing, so a
single fixture can exercise both an ordinary, fully-populated query
(testSparseGridExactPoint) and a missing-corner one
(testSparseGridMissingCorner) without the two interfering: a query
anywhere in the first cell never touches the missing corner at all.

A query point at logg = 8.5, log_Teff = 4.25 (the exact center of the
second cell) sits equidistant from all four of its corners, so under
OOBPolicy::raise or ::silent it falls in a gap (the missing corner),
while under OOBPolicy::coerce it should be interpolated from the three
populated corners alone, renormalized by their combined weight (0.75
of the full cell) -- working out to the plain average of the three
populated values, (3.0 + 4.0 + 5.0) / 3 = 4.0, since all four corners
share equal (0.25) weight at this exact query point.

Every populated model holds a constant flux (no wavelength
dependence), so the interpolated result at every wavelength is exactly
this weighted average, making the expected result trivial to check
exactly rather than only approximately.

Run from the repository root: python3 data/tools/spectra/make_rauch_test_fixture.py

:copyright: Copyright (c) 2026 Mark Krumholz
"""
import h5py
import numpy as np

WL_VALS = [1000.0, 5000.0, 10000.0]  # Angstrom

# (logg, log_Teff) -> constant flux value; (9.0, 4.5) is deliberately absent
FLUX_VALS = {
    (7.0, 4.0): 1.0,
    (7.0, 4.5): 2.0,
    (8.0, 4.0): 3.0,
    (8.0, 4.5): 4.0,
    (9.0, 4.0): 5.0,
}


def main():
    wl = np.array(WL_VALS)
    logg = np.array([g for g, _ in FLUX_VALS])
    log_teff = np.array([t for _, t in FLUX_VALS])
    flux = np.array([[v] * len(wl) for v in FLUX_VALS.values()])

    path = "tests/specsyn/assets/RAUCH_test.h5"
    with h5py.File(path, "w") as h5:
        h5.create_dataset("wl", data=wl)
        h5.create_dataset("logg", data=logg)
        h5.create_dataset("log_Teff", data=log_teff)
        h5.create_dataset("flux", data=flux)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
