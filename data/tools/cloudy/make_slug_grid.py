"""
make_slug_grid.py

First stage of a three-stage pipeline to build a "quick and dirty"
nebular emission lookup table -- a table slug can consult instead of
running the (expensive) cloudy photoionization code on every cluster,
for users willing to trade some accuracy for speed. The three stages,
mirroring slug2's own make_grid.py/process_grid.py two-stage design
(extended here to three, since cloudy is a separate step in slug3):

1. make_slug_grid.py (this script): for each (track set, [Fe/H],
   v/vcrit) combination in the grid, write and run a cluster-type and
   a galaxy-type slug input deck, producing the stellar spectra (and
   Q(HI) photometry) cloudy needs as its own input.
2. run_cloudy_grid.py: runs cloudy on every spectrum this script
   produces.
3. process_cloudy_grid.py: post-processes cloudy's own output into the
   final nebular emission lookup table, cloudy_table.h5.

Unlike slug2, where a single nebular model existed per track set,
slug3 supports far more track-set/rotation/[alpha/Fe] combinations
than is practical to build a full grid for, so this deliberately
starts from a narrow slice: the [alpha/Fe] = 0 subset of the MIST,
Stromlo, and PARSEC_comp track sets, at every [Fe/H] value and (for
the track sets that have a rotation axis at all) every v/vcrit value
each one's own registry entry lists, optionally narrowed further by
--feh-min/--feh-max and --v-vcrit-min/--v-vcrit-max (see TRACK_SETS,
get_feh_grid, get_vvcrit_grid, and data/tracks/tracks.toml).

For each (track set, [Fe/H], v/vcrit) triple, this writes two decks:

- A cluster-type deck: a single fixed-mass (1e4 Msun) Chabrier-IMF
  cluster at that [Fe/H], spectral synthesis via the "default" chained
  model over a 200-2e5 Angstrom, 1024-point grid, Q(HI) as the only
  photometric filter (run_cloudy_grid.py's own way of picking which
  output times are still worth running cloudy on), no extinction,
  output every 0.25 Myr from 0.25 to 10 Myr, one trial with stars
  above 120 Msun treated non-stochastically (min_stoch_mass; keeps the
  rare, ionizing-dominant high-mass end of the IMF from being
  all-or-nothing across a single trial). The grid deliberately omits
  t = 0: at t = 0 every star is still at its (purely artificial, since
  it depends on accretion history the tracks don't model) pre-main-
  sequence contraction radius, and massive stars reach the main
  sequence in far less than the 0.25 Myr step used here, so a t = 0
  output would just be linearly interpolated against a physically
  meaningless endpoint. Consumers of this grid should instead treat
  every t < 0.25 Myr as equal to the t = 0.25 Myr values.
- A galaxy-type deck: identical stellar/spectral/photometric settings,
  but a galaxy forming stars at a fixed 1 Msun/yr with f_cluster = 0
  (so every star forms as part of the smooth field population, not in
  a resolved cluster) and a single output at t = 100 Myr, once the
  galaxy's own stellar population has reached a quasi-equilibrium
  luminosity.

Every deck runs with slug itself (not slugpy.run_sim): with n_trial =
1, there is no per-simulation internal parallelism to exploit, so
concurrency instead comes from running many single-trial slug
subprocesses at once via a thread pool -- exactly mirroring
slugpy.cloudy.cloudy_process.run_cloudy_decks's own reasoning for
running cloudy itself as external subprocesses rather than in-process.

A (track, [Fe/H], v/vcrit, sim_type) combination whose own output
HDF5 file already exists in --work-dir is skipped entirely -- no deck is
written and slug is not run for it -- mirroring run_cloudy_grid.py's
own skip-existing-results behavior (there, via slug_reader.run_cloudy's
own duplicate detection). This makes a partially-completed or
re-run invocation cheap to resume; delete an output file to force
that combination to be regenerated.

--output-table takes this further: pointed at an existing
process_cloudy_grid.py output table (e.g. cloudy_table.h5), a (track,
[Fe/H], v/vcrit) combination already fully present there (every
--check-log-u value, each with both a cluster and a galaxy group) is
skipped entirely, without even checking whether its own slug output
files exist in --work-dir. This is for the workflow of building the
grid one track set (or [Fe/H]/v/vcrit slice) at a time on a shared
cluster, cleaning up each slice's own intermediate slug output once
process_cloudy_grid.py has folded it into the table -- rerunning this
script later (e.g. to add a newly-available track set) then only
regenerates what the table doesn't already have, instead of redoing
everything already-cleaned-up slices once produced.

This whole pipeline is far too computationally expensive to run in
full on a laptop -- the intended workflow is to verify each stage
against a deliberately narrow slice locally (see --feh-min/--feh-max
below), then run the full grid on a shared cluster.

:copyright: Copyright (c) 2026 Mark Krumholz
"""

# Imports
import argparse
import concurrent.futures
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import h5py

# Track sets this grid covers -- the [alpha/Fe] = 0 slice of each
# one's own [Fe/H] and v/vcrit grid, read from data/tracks/tracks.toml
# at run time (see get_feh_grid, get_vvcrit_grid). alphaFe = 0 is
# slug's own library default, and is also valid regardless of whether
# a given track set even has that axis at all: TrackUtils treats a
# track group with no vvcrit/afe attribute of its own as matching any
# requested value (see src/tracks/TrackUtils.cpp's own comment) --
# true of PARSEC_comp, which has neither axis (get_vvcrit_grid treats
# it as a single v/vcrit = 0.0 entry accordingly).
TRACK_SETS = ("MIST", "Stromlo", "PARSEC_comp")

# This script's own directory, and the repo's data/tracks/tracks.toml
# registry relative to it (data/tools/cloudy -> data/tools -> data)
_SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TRACK_REGISTRY = _SCRIPT_DIR.parents[1] / "tracks" / "tracks.toml"

# run_cloudy_grid.py is a sibling script (not a package), reached via
# sys.path directly, so its own LOG_U_VALUES stays the single source of
# truth for the default log10(U) grid (see --check-log-u below).
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from run_cloudy_grid import LOG_U_VALUES  # noqa: E402 -- see sys.path setup above

DEFAULT_WORK_DIR = _SCRIPT_DIR / "slug_grid_work"

# Fixed physical parameters shared by every deck this script writes
# (see this module's own docstring for the reasoning behind each)
CLUSTER_MASS = 1e4        # Msun
CLUSTER_LIFETIME = 1e6    # yr; a fixed clusters.CLF value for the galaxy
                          # deck's own required-but-unused cluster lifetime
                          # function (f_cluster = 0 there, so no cluster
                          # ever actually forms, and this value is never used)
SFR = 1.0                 # Msun/yr
MIN_STOCH_MASS = 120.0    # Msun
WL_MIN = 200.0            # Angstrom
WL_MAX = 2e5              # Angstrom
NWL = 1024
CLUSTER_START_TIME = 2.5e5  # yr (0.25 Myr); see this module's own docstring
                             # for why the grid starts here rather than at 0
CLUSTER_END_TIME = 1e7    # yr (10 Myr)
CLUSTER_NTIME = 40        # 0.25 to 10 Myr in 0.25 Myr steps
GALAXY_TIME = 1e8         # yr (100 Myr)


def get_feh_grid(track_registry: dict, track: str,
    feh_min: float | None, feh_max: float | None) -> list[float]:
    """
    Get one track set's own [Fe/H] grid, optionally restricted to a range.

    Parameters
    ----------
    track_registry : dict
        Parsed contents of a data/tracks/tracks.toml-format registry
        (see DEFAULT_TRACK_REGISTRY).
    track : str
        Name of the track set to look up, e.g. "MIST".
    feh_min, feh_max : float, optional
        If given, only [Fe/H] values >= feh_min and/or <= feh_max
        (inclusive) are returned; otherwise the full grid is used.

    Returns
    -------
    list of float
        track's own Fe_H grid, in registry order, filtered to
        [feh_min, feh_max] if either bound was given.
    """
    feh_values = track_registry[track]["Fe_H"]
    if feh_min is not None:
        feh_values = [feh for feh in feh_values if feh >= feh_min]
    if feh_max is not None:
        feh_values = [feh for feh in feh_values if feh <= feh_max]
    return feh_values


def get_vvcrit_grid(track_registry: dict, track: str,
    vvcrit_min: float | None, vvcrit_max: float | None) -> list[float]:
    """
    Get one track set's own v/vcrit grid, optionally restricted to a range.

    Parameters
    ----------
    track_registry : dict
        Parsed contents of a data/tracks/tracks.toml-format registry
        (see DEFAULT_TRACK_REGISTRY).
    track : str
        Name of the track set to look up, e.g. "MIST".
    vvcrit_min, vvcrit_max : float, optional
        If given, only v/vcrit values >= vvcrit_min and/or <= vvcrit_max
        (inclusive) are returned; otherwise the full grid is used.

    Returns
    -------
    list of float
        track's own v_vcrit grid, in registry order, filtered to
        [vvcrit_min, vvcrit_max] if either bound was given -- [0.0] if
        track's own registry entry has no v_vcrit axis at all (see
        TRACK_SETS's own comment), unless that single value is itself
        filtered out by vvcrit_min/vvcrit_max.
    """
    vvcrit_values = track_registry[track].get("v_vcrit", [0.0])
    if vvcrit_min is not None:
        vvcrit_values = [v for v in vvcrit_values if v >= vvcrit_min]
    if vvcrit_max is not None:
        vvcrit_values = [v for v in vvcrit_values if v <= vvcrit_max]
    return vvcrit_values


def combo_complete_in_table(output_table: Path | None, track: str, feh: float, vvcrit: float,
    log_u_values: list[float]) -> bool:
    """
    Check whether a (track, feh, vvcrit) combination is already fully present in an output table.

    Parameters
    ----------
    output_table : pathlib.Path, optional
        Path to an existing process_cloudy_grid.py output table (e.g.
        cloudy_table.h5) to check. If None, or the file doesn't exist
        yet, this always returns False.
    track : str
        Name of the track set, e.g. "MIST".
    feh : float
        [Fe/H] value.
    vvcrit : float
        v/vcrit value.
    log_u_values : list of float
        The log10(U) grid this combination must have every value of
        (each with both a cluster and a galaxy group) to count as
        complete; should match process_cloudy_grid.py's own --log-u
        (see --check-log-u).

    Returns
    -------
    bool
        True iff output_table has a track/FeH{feh}/vvcrit{vvcrit} group
        containing a logU{log_u}/cluster and a logU{log_u}/galaxy group
        for every entry in log_u_values -- the exact group layout
        process_cloudy_grid.py's own build_table writes.
    """
    if output_table is None or not output_table.exists():
        return False

    with h5py.File(output_table, "r") as f:
        vvcrit_group = f.get(f"{track}/FeH{feh:+.4f}/vvcrit{vvcrit:.2f}")
        if vvcrit_group is None:
            return False
        return all(
            f"logU{log_u:+.2f}/cluster" in vvcrit_group and f"logU{log_u:+.2f}/galaxy" in vvcrit_group
            for log_u in log_u_values
        )


def deck_name(track: str, feh: float, vvcrit: float, sim_type: str) -> str:
    """
    Build the base name (no extension) shared by a (track, feh,
    vvcrit, sim_type) combination's own deck file and output
    model_name.

    Parameters
    ----------
    track : str
        Name of the track set, e.g. "MIST".
    feh : float
        [Fe/H] value.
    vvcrit : float
        v/vcrit value.
    sim_type : {"cluster", "galaxy"}
        Which kind of deck this is.

    Returns
    -------
    str
        e.g. "MIST_FeH+0.0000_vvcrit0.00_cluster". [Fe/H] is formatted
        to 4 decimal places, sign-prefixed, and v/vcrit to 2 decimal
        places -- enough to keep every value in every track set's own
        grid distinct (checked directly against data/tracks/
        tracks.toml), while keeping the name readable.
    """
    return f"{track}_FeH{feh:+.4f}_vvcrit{vvcrit:.2f}_{sim_type}"


def build_cluster_deck(track: str, feh: float, vvcrit: float, work_dir: Path) -> str:
    """
    Build the text of a cluster-type slug input deck.

    Parameters
    ----------
    track : str
        Name of the track set to use, e.g. "MIST".
    feh : float
        [Fe/H] to run at; must be one of track's own registry [Fe/H]
        grid values (see get_feh_grid) for the deck to describe an
        on-grid, non-interpolated point.
    vvcrit : float
        v/vcrit to run at; must be one of track's own registry
        v_vcrit grid values (see get_vvcrit_grid) for the deck to
        describe an on-grid, non-interpolated point.
    work_dir : pathlib.Path
        Directory this deck's own HDF5 output should be written into
        (written as an absolute outputs.out_dir, so the deck resolves
        the same regardless of the working directory slug itself is
        launched from).

    Returns
    -------
    str
        The deck's own TOML text, ready to write to a file.
    """
    model_name = deck_name(track, feh, vvcrit, "cluster")
    return f'''# Auto-generated by data/tools/cloudy/make_slug_grid.py -- do not edit by hand
# Cluster-type deck: track = {track}, [Fe/H] = {feh}, v/vcrit = {vvcrit}

sim_type = "cluster"
n_trial = 1

[stars]
IMF = "chabrier.toml"
tracks = "{track}"
v_vcrit = {vvcrit}
alphaFe = 0.0
FeH = {feh!r}
min_stoch_mass = {MIN_STOCH_MASS}

[spectra]
model = "default"
wl_min = {WL_MIN}
wl_max = {WL_MAX}
nwl = {NWL}

[clusters]
CMF = {CLUSTER_MASS}

[phot]
filters = ["Q(HI)"]

[output]
model_name = "{model_name}"
start_time = {CLUSTER_START_TIME}
end_time = {CLUSTER_END_TIME}
ntime = {CLUSTER_NTIME}

[outputs]
out_dir = "{work_dir.resolve()}"
'''


def build_galaxy_deck(track: str, feh: float, vvcrit: float, work_dir: Path) -> str:
    """
    Build the text of a galaxy-type slug input deck.

    Parameters
    ----------
    track : str
        Name of the track set to use, e.g. "MIST".
    feh : float
        [Fe/H] to run at; must be one of track's own registry [Fe/H]
        grid values (see get_feh_grid) for the deck to describe an
        on-grid, non-interpolated point.
    vvcrit : float
        v/vcrit to run at; must be one of track's own registry
        v_vcrit grid values (see get_vvcrit_grid) for the deck to
        describe an on-grid, non-interpolated point.
    work_dir : pathlib.Path
        Directory this deck's own HDF5 output should be written into
        (written as an absolute outputs.out_dir, so the deck resolves
        the same regardless of the working directory slug itself is
        launched from).

    Returns
    -------
    str
        The deck's own TOML text, ready to write to a file.
    """
    model_name = deck_name(track, feh, vvcrit, "galaxy")
    return f'''# Auto-generated by data/tools/cloudy/make_slug_grid.py -- do not edit by hand
# Galaxy-type deck: track = {track}, [Fe/H] = {feh}, v/vcrit = {vvcrit}

sim_type = "galaxy"
n_trial = 1

[stars]
IMF = "chabrier.toml"
tracks = "{track}"
v_vcrit = {vvcrit}
alphaFe = 0.0
FeH = {feh!r}
min_stoch_mass = {MIN_STOCH_MASS}

[spectra]
model = "default"
wl_min = {WL_MIN}
wl_max = {WL_MAX}
nwl = {NWL}

[clusters]
CMF = {CLUSTER_MASS}
CLF = {CLUSTER_LIFETIME}
f_cluster = 0.0

[phot]
filters = ["Q(HI)"]

[galaxy]
sfr = {SFR}

[output]
model_name = "{model_name}"
output_times = {GALAXY_TIME}

[outputs]
out_dir = "{work_dir.resolve()}"
'''


def find_slug_executable(slug_path: str | Path | None) -> Path:
    """
    Locate the slug executable.

    Parameters
    ----------
    slug_path : str or pathlib.Path, optional
        Explicit path to the slug executable. If omitted, the
        SLUG_EXE environment variable is tried next, then a small set
        of common local CMake build directories relative to the repo
        root (build-release/slug, build/slug).

    Returns
    -------
    pathlib.Path
        Absolute path to the slug executable.

    Raises
    ------
    RuntimeError
        If no candidate resolves to an executable file.
    """
    candidates = []
    if slug_path is not None:
        candidates.append(Path(slug_path))
    env_path = os.environ.get("SLUG_EXE")
    if env_path is not None:
        candidates.append(Path(env_path))
    repo_root = _SCRIPT_DIR.parents[2]
    for build_dir in ("build-release", "build"):
        candidates.append(repo_root / build_dir / "slug")

    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()

    raise RuntimeError(
        "find_slug_executable: no executable slug binary found; tried "
        + ", ".join(str(c) for c in candidates)
        + ". Pass --slug-path, or set the SLUG_EXE environment variable.")


def run_slug_deck(deck_path: Path, slug_exe: Path) -> bool:
    """
    Run slug on a single input deck.

    Parameters
    ----------
    deck_path : pathlib.Path
        Path to the slug input deck to run (as written by
        build_cluster_deck/build_galaxy_deck).
    slug_exe : pathlib.Path
        Path to the slug executable to run (see find_slug_executable).

    Returns
    -------
    bool
        True if slug exited with status 0, False otherwise (including
        if it crashed or raised). Either way, slug's own captured
        stdout/stderr is written to deck_path with its suffix replaced
        by ".log".
    """
    log_path = deck_path.with_suffix(".log")
    with open(log_path, "wb") as fout:
        result = subprocess.run([str(slug_exe), str(deck_path)],
            stdout=fout, stderr=subprocess.STDOUT, check=False)
    return result.returncode == 0


def run_slug_decks(deck_paths: list[Path], slug_exe: Path,
    max_workers: int | None = None) -> list[bool]:
    """
    Run slug concurrently on multiple input decks.

    Parameters
    ----------
    deck_paths : list of pathlib.Path
        Paths to the slug input decks to run.
    slug_exe : pathlib.Path
        Path to the slug executable to run (see find_slug_executable).
    max_workers : int, optional
        Maximum number of slug processes to run at once. Defaults to
        the number of available CPUs.

    Returns
    -------
    list of bool
        Whether each deck's own slug run succeeded (see
        run_slug_deck), in the same order as deck_paths.

    Details
    -------
    Every deck requests n_trial = 1, so there is no per-deck internal
    (OpenMP) parallelism for slug to exploit; concurrency instead
    comes from running many single-trial slug subprocesses at once,
    via a thread pool (each worker just blocks on its own external
    slug subprocess and releases the GIL while doing so, so threads
    are enough -- mirrors
    slugpy.cloudy.cloudy_process.run_cloudy_decks's own identical
    reasoning for cloudy itself).
    """
    if max_workers is None:
        max_workers = os.cpu_count() or 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_slug_deck, deck_path, slug_exe) for deck_path in deck_paths]
        return [future.result() for future in futures]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Write and run the cluster- and galaxy-type slug input decks "
            "needed as input to a cloudy-based nebular emission grid.")
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR,
        help="Directory to write decks and slug's own HDF5 output into "
            f"(default: {DEFAULT_WORK_DIR}). Point this at a scratch "
            "filesystem when running the full grid on a cluster.")
    parser.add_argument("--track-registry", type=Path, default=DEFAULT_TRACK_REGISTRY,
        help=f"Path to the track registry to read TRACK_SETS's own [Fe/H] and "
            f"v/vcrit grids from (default: {DEFAULT_TRACK_REGISTRY}).")
    parser.add_argument("--slug-path", type=Path, default=None,
        help="Path to the slug executable (see find_slug_executable for "
            "the fallback lookup order if this is omitted).")
    parser.add_argument("--feh-min", type=float, default=None,
        help="If given, only run [Fe/H] values >= this (inclusive). "
            "Combine with --feh-max to narrow the grid down to a single "
            "value per track set, for a fast local test run.")
    parser.add_argument("--feh-max", type=float, default=None,
        help="If given, only run [Fe/H] values <= this (inclusive).")
    parser.add_argument("--v-vcrit-min", type=float, default=None,
        help="If given, only run v/vcrit values >= this (inclusive). "
            "Combine with --v-vcrit-max to narrow the grid down to a single "
            "value per track set, for a fast local test run.")
    parser.add_argument("--v-vcrit-max", type=float, default=None,
        help="If given, only run v/vcrit values <= this (inclusive).")
    parser.add_argument("--output-table", type=Path, default=None,
        help="Path to an existing process_cloudy_grid.py output table (e.g. cloudy_table.h5) to "
            "check: a (track, [Fe/H], v/vcrit) combination already fully present there for every "
            "--check-log-u value is skipped entirely, without even checking whether its own slug "
            "output files exist in --work-dir. Omit to always (re)run every matching combination.")
    parser.add_argument("--check-log-u", type=float, nargs="+", default=list(LOG_U_VALUES),
        help="log10(U) grid a combination must have every value of (each with both a cluster and "
            f"a galaxy group) to count as already done in --output-table (default: {list(LOG_U_VALUES)}); "
            "should match process_cloudy_grid.py's own --log-u.")
    parser.add_argument("--max-workers", type=int, default=None,
        help="Maximum number of slug processes to run at once "
            "(default: os.cpu_count()).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.track_registry, "rb") as f:
        track_registry = tomllib.load(f)

    work_dir = args.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    slug_exe = find_slug_executable(args.slug_path)

    deck_paths: list[Path] = []
    n_skipped = 0
    n_skipped_table = 0
    for track in TRACK_SETS:
        feh_grid = get_feh_grid(track_registry, track, args.feh_min, args.feh_max)
        vvcrit_grid = get_vvcrit_grid(track_registry, track, args.v_vcrit_min, args.v_vcrit_max)
        for feh in feh_grid:
            for vvcrit in vvcrit_grid:
                if combo_complete_in_table(args.output_table, track, feh, vvcrit, args.check_log_u):
                    n_skipped_table += 1
                    continue
                for sim_type, build_deck in (("cluster", build_cluster_deck), ("galaxy", build_galaxy_deck)):
                    output_path = work_dir / f"{deck_name(track, feh, vvcrit, sim_type)}.h5"
                    if output_path.exists():
                        n_skipped += 1
                        continue
                    deck_text = build_deck(track, feh, vvcrit, work_dir)
                    deck_path = work_dir / f"{deck_name(track, feh, vvcrit, sim_type)}.toml"
                    deck_path.write_text(deck_text)
                    deck_paths.append(deck_path)

    if n_skipped_table:
        print(f"Skipping {n_skipped_table} (track, [Fe/H], v/vcrit) combination(s) already fully "
            f"present in {args.output_table}.")
    if n_skipped:
        print(f"Skipping {n_skipped} deck(s) whose own output file already exists in {work_dir}.")

    if not deck_paths:
        print("Nothing left to run: every matching (track, [Fe/H], v/vcrit) combination's own "
            "output already exists (in --work-dir or --output-table), or none matched "
            "--feh-min/--feh-max/--v-vcrit-min/--v-vcrit-max.")
        return

    max_workers = args.max_workers or (os.cpu_count() or 1)
    print(f"Wrote {len(deck_paths)} decks to {work_dir}; "
        f"running slug on each (max_workers={max_workers})...")

    successes = run_slug_decks(deck_paths, slug_exe, max_workers=max_workers)

    n_ok = sum(successes)
    n_fail = len(successes) - n_ok
    print(f"{n_ok} succeeded, {n_fail} failed.")
    if n_fail:
        print(f"Failed decks (see their own .log files in {work_dir}):")
        for deck_path, ok in zip(deck_paths, successes, strict=True):
            if not ok:
                print(f"  {deck_path.name}")


if __name__ == "__main__":
    main()
