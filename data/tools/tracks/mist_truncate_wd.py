"""
Script to truncate MIST evolutionary tracks once they enter phase 6
(post-AGB / white dwarf cooling), keeping only the first two phase-6
points on each such track and discarding the rest.

Some MIST tracks run all the way through white dwarf cooling to a
Hubble-time-scale age, while their immediate neighbors in mass and/or
[Fe/H] terminate in the tens-of-Myr range (having not reached phase 6
at all, or having reached it only right at the very end of their own
tabulated range). This multi-order-of-magnitude disparity in track
lifetime between neighboring grid points can cause slug's isochrone
interpolation to produce nonphysical intermediate stars for ages
between a short track's own end and a long track's own end (see the
WD/hot-atmosphere-coverage project notes for how this was found). For
slug's own purposes -- unresolved integrated spectra and photometry,
where individual white dwarfs contribute negligible light -- this is
better addressed by truncating every track before its own white dwarf
cooling tail, rather than by teaching the interpolator to handle
wildly mismatched track lengths.

Truncating right at the first phase-6 point is not quite enough,
however: the transition from phase 5 to phase 6 in MIST tracks
corresponds to a rapid (envelope-ejection) mass loss episode that
continues for one more point beyond the first phase-6 row -- typically
completing within a few tens to hundreds of years, utterly negligible
next to the tens-of-Myr ages involved, but dropping the star's present-
day mass by another ~15-20%. The point immediately after that (the
second phase-6 point) lands within ~1-2% of the star's true final
(post-AGB) remnant mass, and so is kept as well, both to record a
useful approximation of remnant mass for possible future use (e.g.
chemical yields) and to give the truncated track a well-defined final
mass rather than one still mid-transition.

Every other track -- one that never reaches phase 6 at all -- is left
untouched.

Run standalone against an existing MIST HDF5 track file:
    python3 mist_truncate_wd.py /path/to/mist.h5
or import mist_truncate_wd() to call it as part of another script
(see fetch_mist.py, which calls this automatically after fetching new
tracks, unless run with --no_truncate).

:copyright: Copyright (c) 2026 Mark Krumholz
"""

import argparse
import h5py
import numpy as np


def mist_truncate_wd(h5_filename):
    """Truncate every track in h5_filename that reaches phase 6, in place.

    Loops over every group (one per Fe/H x alpha/Fe x v/vcrit
    combination) and every mass track within it. A track that never
    reaches phase 6 is left unchanged. A track that does reach phase 6
    is truncated to keep only its first two phase-6 points (or fewer,
    if the track itself ends before a second phase-6 point exists),
    discarding everything after that.
    """
    with h5py.File(h5_filename, 'r+') as h5file:
        for grp_name in h5file:
            grp = h5file[grp_name]
            field_names = list(grp.attrs['field_names'])
            phase_idx = field_names.index('phase')
            for m in grp['masses'][:]:
                ds_name = f"track_m{m:.3f}"
                track = grp[ds_name][:]
                phase = track[:, phase_idx]
                phase6 = np.where(phase == 6.0)[0]
                if len(phase6) == 0:
                    continue  # never reaches phase 6: leave untouched

                keep_through = min(phase6[0] + 1, len(track) - 1)
                truncated = track[:keep_through + 1]
                if truncated.shape[0] == track.shape[0]:
                    continue  # already this short: nothing to do

                del grp[ds_name]
                grp.create_dataset(ds_name, data=truncated, compression="gzip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Truncate MIST tracks after their first two points "
                    "in phase 6 (post-AGB/white dwarf cooling)")
    parser.add_argument("h5file", help="MIST HDF5 track file to truncate in place")
    args = parser.parse_args()
    mist_truncate_wd(args.h5file)
