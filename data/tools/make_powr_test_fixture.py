#!/usr/bin/env python3
"""Generate tests/specsyn/assets/POWR_test.h5, a small synthetic fixture
for exercising SpecsynLibWR::spec(). Not real PoWR data -- follows the
schema fetch_powr.py writes (per-[Fe/H] groups with "feh"/"dinf"
attributes, containing one dataset per (log_teff, log_rt) model with
"log_teff"/"log_rt"/"logl" attributes and a "<name>_wave" companion
dataset), but with a synthetic Gaussian SED shared by every model so
that the interpolated result is exactly the same shape at every grid
point, and only the overall normalization (via logl) changes.

Run from the repository root: python3 data/tools/make_powr_test_fixture.py
"""
import h5py
import numpy as np

SOLAR_LUM = 3.828e33  # erg/s, IAU 2015 nominal value

FEH_VALS = [-1.0, 0.0]
LOG_TEFF_VALS = [4.6, 4.8]
LOG_RT_VALS = [0.5, 1.0]
LOGL = 5.5          # log10(L/Lsun) shared by every model in the fixture
DINF = 1.0          # wind clumping density contrast, shared by every group

WL0 = 5000.0        # Angstrom, Gaussian SED center
SIGMA = 500.0       # Angstrom, Gaussian SED width
PEAK_LUM = (10.0 ** LOGL) * SOLAR_LUM  # erg/s, target peak of wl * F(wl)
AMPLITUDE = PEAK_LUM / WL0              # erg/s/Angstrom, at wl0

wave = np.geomspace(1000.0, 20000.0, 200)
flux = AMPLITUDE * np.exp(-0.5 * ((wave - WL0) / SIGMA) ** 2)

with h5py.File("tests/specsyn/assets/POWR_test.h5", "w") as h5:
    for feh in FEH_VALS:
        grp = h5.create_group(f"feh_{feh:+.2f}")
        grp.attrs["feh"] = feh
        grp.attrs["dinf"] = DINF
        for log_teff in LOG_TEFF_VALS:
            for log_rt in LOG_RT_VALS:
                name = f"logt{log_teff:.2f}_logrt{log_rt:.2f}"
                ds = grp.create_dataset(name, data=flux, compression="gzip")
                ds.attrs["log_teff"] = log_teff
                ds.attrs["log_rt"] = log_rt
                ds.attrs["logl"] = LOGL
                grp.create_dataset(f"{name}_wave", data=wave, compression="gzip")

print("wrote tests/specsyn/assets/POWR_test.h5")
print(f"amplitude = {AMPLITUDE:.6e} erg/s/Angstrom at wl0 = {WL0} Angstrom")
