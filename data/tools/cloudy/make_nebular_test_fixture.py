#!/usr/bin/env python3
"""Generate the synthetic nebular emission test fixture at
tests/nebular/assets/nebular_test.h5, used to exercise Nebular's
constructor and getGalaxy()/getCluster(). Not real cloudy grid data --
follows the schema process_cloudy_grid.py writes (see its own module
docstring), but with simple analytic continuum/line values, linear in
[Fe/H] and cluster age, so a test can compute the exact expected result
of any getGalaxy()/getCluster() call by hand rather than needing real
cloudy physics to check against.

Track group "MIST_test" (matching stars.tracks in tests/*/assets/*.in),
three [Fe/H] points (-0.5, 0.0, 0.5, bracketing/hitting
tests/tracks/assets/tracks.toml's own MIST_test grid), one v/vcrit
(0.0, the default), and a single log(U) point exactly at
nebular::defaultLogU (-2.5) -- so no log(U) interpolation ever actually
happens; that machinery is already exercised by [Fe/H] and age
interpolation instead, via the identical bracket/blend code path (see
Nebular.cpp's own blendByLogU()/bracketGrid()), so a second log(U)
point would only add fixture complexity without adding coverage.

Every continuum/line value below is exactly linear in [Fe/H] and (for
the cluster subtype) age -- both grids are read as the true VALUE
(not just index) an interpolation is meant to land on, so an exact
expected value for any in-range (feH, age) request is just the
corresponding formula, no matter where the request falls relative to
the tabulated grid points.

Run from the repository root: python3 data/tools/cloudy/make_nebular_test_fixture.py

:copyright: Copyright (c) 2026 Mark Krumholz
"""
import h5py
import numpy as np

TRACK_NAME = "MIST_test"
FEH_VALS = [-0.5, 0.0, 0.5]
VVCRIT = 0.0
LOG_U = -2.5  # exactly nebular::defaultLogU -- see module docstring

WL_NATIVE = np.geomspace(500.0, 20000.0, 30)  # Angstrom
TIME = np.array([1.0e6, 3.0e6, 1.0e7, 3.0e7, 1.0e8])  # yr

LINE_WL = np.array([4000.0, 6000.0, 9000.0])  # Angstrom
LINE_LABEL = ["LINE1", "LINE2", "LINE3"]
MAX_LABEL_LEN = 16

# Galaxy continuum/line coefficients: value(feH) = COEFF * (1 + feH),
# constant across wl (so resampling from WL_NATIVE onto any target
# wl grid reproduces this same constant exactly at every in-range
# point, per Steffen-spline interpolation of a constant function)
CTM_GAL0 = 1.0e-20    # erg/s/Angstrom per ionizing photon, at feH = 0
LINE_GAL0 = 1.0e-18   # erg/s per ionizing photon, at feH = 0, per line index (1-based)

# Cluster continuum/line coefficients: value(feH, age) = COEFF *
# (1 + feH) * (age / 1e6 yr), likewise constant across wl
CTM_CLUS0 = 1.0e-21   # erg/s/Angstrom per ionizing photon, at feH = 0, age = 1e6 yr
LINE_CLUS0 = 1.0e-19  # erg/s per ionizing photon, at feH = 0, age = 1e6 yr, per line index (1-based)


def galaxy_ctm(feh: float) -> np.ndarray:
    """Galaxy continuum per ionizing photon, on WL_NATIVE, at feH."""
    return np.full_like(WL_NATIVE, CTM_GAL0 * (1.0 + feh))


def galaxy_line(feh: float) -> np.ndarray:
    """Galaxy per-line luminosity per ionizing photon, at feH."""
    idx1 = np.arange(1, len(LINE_WL) + 1)
    return LINE_GAL0 * idx1 * (1.0 + feh)


def cluster_ctm(feh: float) -> np.ndarray:
    """Cluster continuum per ionizing photon, on (TIME, WL_NATIVE), at feH."""
    per_time = CTM_CLUS0 * (1.0 + feh) * (TIME / 1.0e6)  # (ntime,)
    return np.outer(per_time, np.ones_like(WL_NATIVE))   # (ntime, nwl)


def cluster_line(feh: float) -> np.ndarray:
    """Cluster per-line luminosity per ionizing photon, on (TIME, line), at feH."""
    idx1 = np.arange(1, len(LINE_WL) + 1)
    per_time = LINE_CLUS0 * (1.0 + feh) * (TIME / 1.0e6)  # (ntime,)
    return np.outer(per_time, idx1)                       # (ntime, nline)


def main() -> None:
    path = "tests/nebular/assets/nebular_test.h5"
    with h5py.File(path, "w") as fout:
        fout.create_dataset("wl", data=WL_NATIVE)
        fout.create_dataset("line_wl", data=LINE_WL)
        label_dset = fout.create_dataset(
            "line_label", shape=(len(LINE_LABEL),),
            dtype=h5py.string_dtype(encoding="ascii", length=MAX_LABEL_LEN))
        label_dset[:] = LINE_LABEL
        fout.create_dataset("time", data=TIME)

        track_grp = fout.create_group(TRACK_NAME)
        track_grp.attrs["track"] = TRACK_NAME
        for feh in FEH_VALS:
            feh_grp = track_grp.create_group(f"FeH{feh:+.4f}")
            feh_grp.attrs["FeH"] = feh
            vvcrit_grp = feh_grp.create_group(f"vvcrit{VVCRIT:.2f}")
            vvcrit_grp.attrs["v_vcrit"] = VVCRIT
            logu_grp = vvcrit_grp.create_group(f"logU{LOG_U:+.2f}")
            logu_grp.attrs["logU"] = LOG_U

            galaxy_grp = logu_grp.create_group("galaxy")
            galaxy_grp.create_dataset("spec", data=galaxy_ctm(feh))
            galaxy_grp.create_dataset("line_lum", data=galaxy_line(feh))

            cluster_grp = logu_grp.create_group("cluster")
            cluster_grp.create_dataset("spec", data=cluster_ctm(feh))
            cluster_grp.create_dataset("line_lum", data=cluster_line(feh))

    print(f"wrote {path}")


if __name__ == "__main__":
    main()
