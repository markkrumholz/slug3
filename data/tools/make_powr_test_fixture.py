#!/usr/bin/env python3
"""Generate the synthetic PoWR test fixtures under tests/specsyn/assets/
used to exercise SpecsynLibWR::spec(): POWR_test.h5 (for the WNE-type
registry entry POWR_WNE_test) and POWR_WNL_test.h5 (for the WNL-type
registry entry POWR_WNL_test). Not real PoWR data -- follows the schema
fetch_powr.py writes (per-[Fe/H] groups with "feh"/"dinf" attributes,
containing one dataset per (log_teff, log_rt) model with
"log_teff"/"log_rt"/"logl" attributes and a "<name>_wave" companion
dataset), but with a synthetic Gaussian SED shared by every model at a
given (feh, xh) so that the interpolated result is exactly the same
shape at every grid point, and only the overall normalization (via
logl) changes.

POWR_WNL_test.h5 additionally carries the "xh" (surface H mass
fraction) group attribute real WNL groups have, and -- unlike real PoWR
data, where every metallicity has exactly one xh = 0.20 ("H20") grid --
gives each of its two [Fe/H] values both an H20 group and a decoy
xh = 0.50 group with a deliberately different SED (peaked at a
different wavelength, WL0_DECOY rather than WL0), so that a test can
confirm SpecsynLibWR's WNL H20-only filtering (see its constructor)
actually excludes the decoy rather than merely happening not to need
it.

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

WL0 = 5000.0         # Angstrom, Gaussian SED center for WNE and WNL's H20 models
WL0_DECOY = 15000.0  # Angstrom, Gaussian SED center for WNL's decoy (xh = 0.50) models
SIGMA = 500.0        # Angstrom, Gaussian SED width


def make_flux(wave, wl0, logl):
    """A Gaussian SED centered at wl0, normalized so that the peak of
    wave * flux equals the luminosity implied by logl.
    """
    peak_lum = (10.0 ** logl) * SOLAR_LUM  # erg/s, target peak of wl * F(wl)
    amplitude = peak_lum / wl0             # erg/s/Angstrom, at wl0
    return amplitude * np.exp(-0.5 * ((wave - wl0) / SIGMA) ** 2)


wave = np.geomspace(1000.0, 20000.0, 200)

# POWR_test.h5: the WNE-type fixture (POWR_WNE_test), unchanged from its
# original form -- one group per [Fe/H], no xh axis
flux = make_flux(wave, WL0, LOGL)
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
print(f"amplitude = {(10.0 ** LOGL) * SOLAR_LUM / WL0:.6e} erg/s/Angstrom at "
      f"wl0 = {WL0} Angstrom")

# POWR_WNL_test.h5: the WNL-type fixture (POWR_WNL_test) -- each [Fe/H]
# gets both an H20 (xh = 0.20) group and a decoy (xh = 0.50) group
flux_h20 = make_flux(wave, WL0, LOGL)
flux_decoy = make_flux(wave, WL0_DECOY, LOGL)
with h5py.File("tests/specsyn/assets/POWR_WNL_test.h5", "w") as h5:
    for feh in FEH_VALS:
        for xh, flux_xh in ((0.20, flux_h20), (0.50, flux_decoy)):
            grp = h5.create_group(f"feh_{feh:+.2f}_xh{xh:.2f}")
            grp.attrs["feh"] = feh
            grp.attrs["dinf"] = DINF
            grp.attrs["xh"] = xh
            for log_teff in LOG_TEFF_VALS:
                for log_rt in LOG_RT_VALS:
                    name = f"logt{log_teff:.2f}_logrt{log_rt:.2f}"
                    ds = grp.create_dataset(name, data=flux_xh, compression="gzip")
                    ds.attrs["log_teff"] = log_teff
                    ds.attrs["log_rt"] = log_rt
                    ds.attrs["logl"] = LOGL
                    grp.create_dataset(f"{name}_wave", data=wave, compression="gzip")

print("wrote tests/specsyn/assets/POWR_WNL_test.h5")
print(f"H20 SED peaked at wl0 = {WL0} Angstrom; decoy SED peaked at "
      f"wl0 = {WL0_DECOY} Angstrom")
