#!/usr/bin/env python3
"""Generate data/filters/V_filter.h5 and V_filter.toml, a standalone,
single-filter copy of Generic.Johnson.V extracted from the full
data/filters/filters.h5 (itself too large to store in the repo -- see
.gitignore). Extinction is conventionally quantified in V-band
magnitudes, so this gives slug a small, repo-committed filter it can
use for that purpose without needing the full filter set fetched.

The extracted HDF5 group (attributes -- including the SVO Filter
Profile Service's own description/references/source metadata -- and
the "wavelength"/"transmission" datasets) is copied verbatim via
h5py's own Group.copy(), so V_filter.h5 is byte-for-byte the same
filter data as filters.h5's own Generic/Johnson/V group, just without
every other filter alongside it. V_filter.toml is a registry in the
same format fetch_filter_vo.py's own update_registry() writes,
containing only this one facility/instrument/filter.

Run from the repository root: python3 data/tools/filters/make_v_filter_fixture.py

:copyright: Copyright (c) 2026 Mark Krumholz
"""
import h5py
import tomlkit

SOURCE_H5 = "data/filters/filters.h5"
FACILITY = "Generic"
INSTRUMENT = "Johnson"
FILTER = "V"

H5_PATH = "data/filters/V_filter.h5"
REGISTRY_PATH = "data/filters/V_filter.toml"
FILTER_VO_URL = "http://svo2.cab.inta-csic.es/svo/theory/fps3/index.php"

with h5py.File(SOURCE_H5, "r") as src, h5py.File(H5_PATH, "w") as dst:
    src_grp = src[f"{FACILITY}/{INSTRUMENT}/{FILTER}"]
    dst_grp = dst.require_group(FACILITY).require_group(INSTRUMENT)
    src.copy(src_grp, dst_grp, name=FILTER)
print(f"wrote {H5_PATH}")

with h5py.File(H5_PATH, "r") as h5file:
    filt_attrs = h5file[f"{FACILITY}/{INSTRUMENT}/{FILTER}"].attrs
    description = filt_attrs.get("Description", "")
    wl_ref = float(filt_attrs.get("WavelengthRef", float("nan")))

doc = tomlkit.document()
doc["name"] = "Registry of filters (V-band only)"
doc["file"] = "V_filter.h5"
doc["Facilities"] = [FACILITY]

fac_table = tomlkit.table()
fac_table["instruments"] = [INSTRUMENT]
instr_table = tomlkit.table()
instr_table["filters"] = [FILTER]
filt_table = tomlkit.table()
filt_table["description"] = description
filt_table["wl_ref"] = wl_ref
filt_table["source"] = FILTER_VO_URL
instr_table[FILTER] = filt_table
fac_table[INSTRUMENT] = instr_table
doc[FACILITY] = fac_table

with open(REGISTRY_PATH, "w") as f:
    f.write(tomlkit.dumps(doc))
print(f"wrote {REGISTRY_PATH}")
