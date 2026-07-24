#!/usr/bin/env python3
"""Generate tests/specsyn/assets/COERCE_ZERO_test.h5, a tiny synthetic
fixture for exercising SpecsynLib::spec()'s OOBPolicy::coerce behavior
in the specific edge case where a query lands exactly on a grid line
along one axis. Not real spectral data -- follows the same schema as
COERCE_test.h5 (see make_coerce_test_fixture.py), with a Teff axis of
{1000, 10000} and a logg axis of {4.0, 4.5, 5.0}:

    logg \\ Teff   1000    10000
    4.00           1.0    (missing)
    4.50           2.0    (missing)
    5.00          (missing)  9.0

The lone point at (10000, 5.0) exists only so that Teff = 10000 enters
this library's Teff axis at all (an entirely unpopulated Teff column
would otherwise just be absent from the axis, rather than present but
empty); the query below never brackets logg = 5.0, so this point never
otherwise participates. Teff = 1000 and 10000 K are both exact powers
of ten, chosen so that querying at exactly Teff = 10000 K round-trips
exactly through SpecsynLibNoWind::spec()'s log10()/pow(10, .)
conversion (an arbitrary Teff does not survive that round trip bit-
for-bit -- see the comment on testSpecCoerceZeroWeight in
tests/specsyn/testSpecsynLib.cpp for why that matters here). A query at
exactly Teff = 10000 K and logg strictly between 4.0 and 4.5 sits at
the grid's own upper edge along the Teff axis alone, giving the
Teff = 1000 corners exactly zero interpolation weight regardless of
logg -- so they are skipped as populated-but-irrelevant, while the two
corners that do carry weight (both at Teff = 10000, logg = 4.0 or 4.5)
are both unpopulated. This reproduces a real bug found via slug3's
full-scale end-to-end test (testClusterSpecsynFull): hasValidNeighbor
was true (the Teff = 1000 corners are populated), yet every neighbor
actually used in the weighted sum had zero total weight, so dividing
by that sum produced NaN/Inf instead of a clean out-of-bounds result.

Run from the repository root:
    python3 data/tools/make_coerce_zero_weight_test_fixture.py
"""
import h5py
import numpy as np

FEH = 0.0
MICRO = 0

# (teff, logg) -> constant flux value; (10000, 4.0) and (10000, 4.5)
# are deliberately absent -- (10000, 5.0) exists only to put Teff =
# 10000 on this library's Teff axis at all (see module docstring)
FLUX_VALS = {
    (1000.0, 4.0): 1.0,
    (1000.0, 4.5): 2.0,
    (10000.0, 5.0): 9.0,
}

wave = np.linspace(1000.0, 20000.0, 50)

with h5py.File("tests/specsyn/assets/COERCE_ZERO_test.h5", "w") as h5:
    grp = h5.create_group(f"spectra_feh{FEH:.4g}_micro{MICRO}")
    grp.attrs["feh"] = FEH
    grp.attrs["micro"] = MICRO
    for (teff, logg), flux_val in FLUX_VALS.items():
        name = f"t{teff:.0f}_g{logg:+.2f}"
        flux = np.full_like(wave, flux_val)
        ds = grp.create_dataset(name, data=flux, compression="gzip")
        ds.attrs["teff"] = teff
        ds.attrs["logg"] = logg

    wgrp = h5.create_group("wavelengths")
    wgrp.create_dataset("native", data=wave, compression="gzip")

print("wrote tests/specsyn/assets/COERCE_ZERO_test.h5")
