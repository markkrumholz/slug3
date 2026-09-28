#!/usr/bin/env python3
"""Generate the MIST_test_lowmass stellar track test fixture at
tests/tracks/assets/MIST_test_lowmass.h5, a copy of the MIST_test
fixture (tests/tracks/assets/MIST_test.h5) truncated to initial masses
<= 100 Msun and to the [Fe/H] groups -0.5, 0.0, and 0.5. Its maximum
mass is therefore below MIST_test's own (300 Msun), which lets a test
check that SimControls::setTracks() rejects tracks that do not extend
up to the current IMF's maximum mass, without needing any of the
large, gitignored track sets under data/tracks.

The output file is registered as track set "MIST_test_lowmass" in
tests/tracks/assets/tracks.toml; if the [Fe/H] groups kept here
change, update that registry entry's Fe_H list to match.

Run from the repository root:
python3 data/tools/tracks/make_lowmass_track_fixture.py

:copyright: Copyright (c) 2026 Mark Krumholz
"""

import argparse

import h5py
import numpy as np

KEEP_FEH = (-0.5, 0.0, 0.5)
M_MAX = 100.0


def truncate(src_path, dst_path, m_max=M_MAX, keep_feh=KEEP_FEH):
    """
    Copy a track file, keeping only tracks with initial mass <= m_max
    and groups whose [Fe/H] is in keep_feh.

    Parameters
    ----------
    src_path : str
        Path to the source track file
    dst_path : str
        Path to the output track file
    m_max : float
        Maximum initial mass to keep
    keep_feh : tuple of float
        [Fe/H] values of the groups to keep

    Returns
    -------
    None
    """
    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        for key, val in src.attrs.items():
            dst.attrs[key] = val
        for name, grp in src.items():
            if not np.any(np.isclose(grp.attrs["feh"], keep_feh)):
                continue
            out = dst.create_group(name)
            masses = grp["masses"][:]
            kept = masses[masses <= m_max]
            for key, val in grp.attrs.items():
                out.attrs[key] = val
            out.attrs["m_max"] = kept.max()
            out.attrs["nmass"] = kept.size
            out.create_dataset("masses", data=kept)
            for track_name, track in grp.items():
                if track_name == "masses":
                    continue
                if float(track_name.removeprefix("track_m")) > m_max:
                    continue
                src.copy(track, out, name=track_name)


def main():
    """Parse the command line and write the fixture."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", default="tests/tracks/assets/MIST_test.h5",
                        help="source track file")
    parser.add_argument("--out", default="tests/tracks/assets/MIST_test_lowmass.h5",
                        help="output track file")
    args = parser.parse_args()
    truncate(args.src, args.out)


if __name__ == "__main__":
    main()
