#!/usr/bin/env python3
"""Generate the tiny tests/specsyn/assets/TIEDFEH_test.h5 fixture used by
tests/specsyn/testSpecsynUtils.cpp's testFindMatchingSpectraTiedFeh: two
groups, both with feh = 0.0 and no afe/cfe/micro/r attributes (so both
always match regardless of query parameters, mirroring a WR-type
registry entry, which has no such axes at all -- see fetch_powr.py),
exercising findMatchingSpectra's handling of a spectral library whose
groups are not guaranteed to have distinct feh values (real PoWR WNL
grids used to be the only real-world example of this, before
fetch_powr.py split them by surface-hydrogen bucket into separate,
tie-free files -- see its own module docstring). No datasets are needed
inside either group: findMatchingSpectra only ever reads a group's own
attributes, never its contents.

Run from the repository root: python3 data/tools/make_tied_feh_test_fixture.py
"""
import h5py

with h5py.File("tests/specsyn/assets/TIEDFEH_test.h5", "w") as h5:
    for name in ("group_a", "group_b"):
        grp = h5.create_group(name)
        grp.attrs["feh"] = 0.0

print("wrote tests/specsyn/assets/TIEDFEH_test.h5")
