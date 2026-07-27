#!/usr/bin/env python3
"""Generate tests/phot/assets/filters_test.h5 and filters_test.toml, a
tiny synthetic fixture for exercising FilterTabulated's registry
constructor. Not real filter data -- follows the same schema
fetch_filter_vo.py writes (an HDF5 file with a facility/instrument/
filter group hierarchy, each filter group holding "wavelength" and
"transmission" datasets, paired with a TOML registry whose top-level
"file" entry names the HDF5 file and whose nested tables mirror that
same hierarchy) -- but with a single synthetic Gaussian response curve
under facility "SLUGTEST", instrument "CAM1", filter "G500", rather
than any real facility's filters.

The response curve is a Gaussian centered at WL0 = 5000 Angstrom with
width SIGMA = 500 Angstrom, sampled on a 25-point wavelength grid from
3000-7000 Angstrom -- enough points that Interpolator1D actually uses
its default Steffen spline (rather than silently falling back to
linear interpolation for a too-small input) internally, matching the
shape of a real, many-point tabulated response curve.

Run from the repository root: python3 data/tools/make_filter_test_fixture.py
"""
import h5py
import numpy as np
import tomlkit

FACILITY = "SLUGTEST"
INSTRUMENT = "CAM1"
FILTER = "G500"

WL0 = 5000.0    # Angstrom, Gaussian response center
SIGMA = 500.0   # Angstrom, Gaussian response width
WL_MIN = 3000.0
WL_MAX = 7000.0
N_WL = 25

H5_PATH = "tests/phot/assets/filters_test.h5"
REGISTRY_PATH = "tests/phot/assets/filters_test.toml"

wave = np.linspace(WL_MIN, WL_MAX, N_WL)
response = np.exp(-0.5 * ((wave - WL0) / SIGMA) ** 2)

with h5py.File(H5_PATH, "w") as h5file:
    grp = h5file.require_group(FACILITY).require_group(INSTRUMENT).create_group(FILTER)
    grp.create_dataset("wavelength", data=wave, compression="gzip")
    grp.create_dataset("transmission", data=response, compression="gzip")
print(f"wrote {H5_PATH}")

doc = tomlkit.document()
doc["name"] = "Registry of filters (test fixture)"
doc["file"] = "filters_test.h5"
doc["Facilities"] = [FACILITY]

fac_table = tomlkit.table()
fac_table["instruments"] = [INSTRUMENT]
instr_table = tomlkit.table()
instr_table["filters"] = [FILTER]
filt_table = tomlkit.table()
filt_table["description"] = "Synthetic Gaussian test filter"
filt_table["wl_ref"] = WL0
instr_table[FILTER] = filt_table
fac_table[INSTRUMENT] = instr_table
doc[FACILITY] = fac_table

with open(REGISTRY_PATH, "w") as f:
    f.write(tomlkit.dumps(doc))
print(f"wrote {REGISTRY_PATH}")
