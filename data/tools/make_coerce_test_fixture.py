#!/usr/bin/env python3
"""Generate tests/specsyn/assets/COERCE_test.h5, a tiny synthetic
fixture for exercising SpecsynLib::spec()'s OOBPolicy::coerce behavior.
Not real spectral data -- follows the same schema as
data/tools/fetch_tlusty.py (a single shared wavelength grid under
wavelengths/<name>, and per-(Teff, logg) datasets carrying "teff"/
"logg" attributes), but deliberately leaves one corner of an otherwise
complete 2x2 (Teff, logg) grid unpopulated:

    logg \\ Teff   5000    6000
    4.00           1.0     3.0
    4.50           2.0    (missing)

A query star at Teff = 5500 K, logg = 4.25 (the exact center of this
cell) sits equidistant from all four corners, so under OOBPolicy::raise
or ::silent it falls in a gap (the missing corner), while under
OOBPolicy::coerce it should be interpolated from the three populated
corners alone, renormalized by their combined weight (0.75 of the full
cell) -- working out to the plain average of the three populated
values, (1.0 + 2.0 + 3.0) / 3 = 2.0, since all four corners share equal
(0.25) weight at this exact query point.

Every populated point holds a constant flux (no wavelength dependence),
so the interpolated result at every wavelength is exactly this
weighted average, making the expected result trivial to check exactly
rather than only approximately.

Run from the repository root: python3 data/tools/make_coerce_test_fixture.py
"""
import h5py
import numpy as np

FEH = 0.0
MICRO = 0
TEFF_VALS = [5000.0, 6000.0]
LOGG_VALS = [4.0, 4.5]

# (teff, logg) -> constant flux value; (6000, 4.5) is deliberately absent
FLUX_VALS = {
    (5000.0, 4.0): 1.0,
    (5000.0, 4.5): 2.0,
    (6000.0, 4.0): 3.0,
}

wave = np.linspace(1000.0, 20000.0, 50)

with h5py.File("tests/specsyn/assets/COERCE_test.h5", "w") as h5:
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

print("wrote tests/specsyn/assets/COERCE_test.h5")
