#!/usr/bin/env python3
"""Generate the synthetic white dwarf test fixture under
tests/specsyn/assets/ used to exercise SpecsynLibWD::spec():
WD_test.h5 (for the WD_test registry entry).

Not real Tremblay et al. data -- follows the flat, four-dataset schema
fetch_tremblay.py writes ("wl", "logg", "log_Teff", and a
(n_logg, n_logTeff, n_wl) "flux" tensor, with no [Fe/H] axis at all),
but with a synthetic flux chosen so that every test check can be exact
rather than approximate: flux(logg, logTeff, wl) = amplitude(logg,
logTeff) * shape(wl), with amplitude an exactly linear function of
(logg, logTeff) -- AMP0 + AMP_LOGG * logg + AMP_LOGTEFF * logTeff --
and shape a fixed per-wavelength-point pattern. Bilinear interpolation
of a function that is linear (not just bilinear) in its two variables
is exact everywhere within the grid, not just at grid points, so a
test can pick an arbitrary off-grid (logg, logTeff) point and compare
SpecsynLibWD's interpolated result against amplitude()*shape()
evaluated directly, rather than only checking exact grid points or
settling for an approximate/sanity check.

Run from the repository root: python3 data/tools/spectra/make_wd_test_fixture.py
"""
import h5py
import numpy as np

LOGG_VALS = [7.0, 8.0, 9.0]
LOG_TEFF_VALS = [4.0, 4.3, 4.6]
WL_VALS = [1000.0, 2000.0, 4000.0, 8000.0, 16000.0]  # Angstrom
SHAPE = [1.0, 2.0, 3.0, 4.0, 5.0]  # a distinct value per wavelength point

AMP0 = 1.0
AMP_LOGG = 2.0
AMP_LOGTEFF = 3.0


def amplitude(logg, log_teff):
    return AMP0 + AMP_LOGG * logg + AMP_LOGTEFF * log_teff


def main():
    wl = np.array(WL_VALS)
    logg = np.array(LOGG_VALS)
    log_teff = np.array(LOG_TEFF_VALS)
    shape = np.array(SHAPE)

    flux = np.empty((len(logg), len(log_teff), len(wl)))
    for i, g in enumerate(logg):
        for j, t in enumerate(log_teff):
            flux[i, j, :] = amplitude(g, t) * shape

    path = "tests/specsyn/assets/WD_test.h5"
    with h5py.File(path, "w") as h5:
        h5.create_dataset("wl", data=wl)
        h5.create_dataset("logg", data=logg)
        h5.create_dataset("log_Teff", data=log_teff)
        h5.create_dataset("flux", data=flux)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
