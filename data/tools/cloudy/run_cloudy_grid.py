"""
run_cloudy_grid.py

Second stage of the three-stage pipeline to build a "quick and dirty"
nebular emission lookup table -- see make_slug_grid.py's own module
docstring for the full pipeline description. This script's own piece:
runs cloudy, via slugpy.slug_reader.run_cloudy, on every slug output
file make_slug_grid.py produced in a given working directory.

Cluster- and galaxy-type output files need different treatment, since
they end up with very different numbers of cloudy runs to do:

- A galaxy-type file has exactly one (trial, output time) pair -- one
  cloudy run per --log-u value.
- A cluster-type file has one cluster but many output times (see
  make_slug_grid.py's own 0.25-10 Myr, 0.25 Myr cadence), most of
  which are not worth running cloudy on: once the cluster's own
  ionizing luminosity Q(HI) has dropped well below its own peak, the
  resulting nebular emission is negligible. This script reads each
  cluster file's own Q(HI) photometry (see make_slug_grid.py's own
  Q(HI)-only [phot] stanza) and only requests cloudy runs for output
  times where Q(HI) exceeds --qhi-fraction (default 1%) of Q(HI) at
  the earliest output time.

Every qualifying spectrum (every qualifying output time for a cluster
file, the single output time for a galaxy file) is run through cloudy
once per ionization parameter in --log-u (default log U = -3, -2.5,
-2), at the fixed density nII = 100 cm^-3 run_cloudy itself otherwise
defaults to -- i.e. this script always passes both nII and U
explicitly, rather than relying on run_cloudy's own no-arguments
default, so that only U varies across runs. These per-U runs are
separate run_cloudy() calls (duplicate detection is keyed on the full
set of six hiiregparam values, so they can't collide).

All of this work -- every (cluster file, qualifying time, log U)
triple and every (galaxy file, log U) pair, across every file in
--work-dir -- shares one single ThreadPoolExecutor bounded at
--max-workers cloudy processes at once (passed to slug_reader.run_cloudy
as its own executor argument). Each file gets its own lightweight
driver thread that writes/queues that file's own decks into the shared
pool and blocks on their results; driver threads themselves are cheap
(they do nothing but wait), so there are as many of them as there are
files, regardless of --max-workers. This matters because cluster and
galaxy files, and different cluster files' own qualifying-time counts,
are wildly uneven in size: processing cluster files to completion
before starting galaxy files (or bounding how many files' own decks
can be queued at once) would leave --max-workers cloudy processes idle
whenever the currently-active file(s) run out of qualifying work
before the others even start -- exactly what was observed on an
8-CPU/104-core cluster smoke test, where a single slow cluster file
monopolized every worker while five other cluster files and all six
galaxy files sat completely unstarted. Sharing one pool across every
file's own driver thread means an idle worker always immediately picks
up whatever's next, regardless of which file it came from.

Every file's own run_cloudy() calls also use an explicit, file-specific
temp_dir (cloudy_tmp_<file's own stem>) rather than run_cloudy's own
"cloudy_tmp" default: every file here lives in the same --work-dir, so
that default would resolve to the exact same directory for all of
them, and run_cloudy deletes it wholesale once its own decks are done
(unless --save-temp) -- fine for a single caller at a time, but with
multiple files' own calls genuinely running concurrently, one file
finishing (and deleting the shared directory) could delete another
file's still-in-flight decks out from under it.

Like make_slug_grid.py, this is far too expensive to run in full on a
laptop -- verify locally against the same narrow --feh-min/--feh-max
slice make_slug_grid.py was run with, then run the full grid on a
shared cluster.

:copyright: Copyright (c) 2026 Mark Krumholz
"""

# Imports
import argparse
import concurrent.futures
import os
import sys
from pathlib import Path
from typing import cast

import astropy.units as u

# slugpy is a sibling package three directories up from this script
# (data/tools/cloudy -> data/tools -> data -> repo root), and is not
# installed, so it has to be reached via sys.path directly.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from slugpy.slug_reader import slug_reader  # noqa: E402 -- see sys.path setup above

QHI_FRACTION = 0.01

# Matches run_cloudy's own no-arguments default density, so passing it
# explicitly alongside a varying U doesn't change the density grid.
NII_DEFAULT = 100.0 / u.cm ** 3

LOG_U_VALUES = (-3.0, -2.5, -2.0)

DEFAULT_WORK_DIR = Path(__file__).resolve().parent / "slug_grid_work"


def find_h5_files(work_dir: Path, files: list[str] | None = None) -> list[Path]:
    """
    Find every slug HDF5 output file make_slug_grid.py wrote.

    Parameters
    ----------
    work_dir : pathlib.Path
        Directory to search (non-recursively).
    files : list of str, optional
        If given, restrict the result to just these files (matched by
        name, with or without a ".h5" suffix -- see --files) instead
        of every "*.h5" file in work_dir. Matched in the order given
        in files, not sorted.

    Returns
    -------
    list of pathlib.Path
        Every "*.h5" file directly in work_dir, sorted by name -- or,
        if files was given, just those files, in the order given.

    Raises
    ------
    ValueError
        If files was given and any of its entries has no matching
        "*.h5" file directly in work_dir.
    """
    if files is None:
        return sorted(work_dir.glob("*.h5"))

    result = []
    for name in files:
        path = work_dir / (name if name.endswith(".h5") else f"{name}.h5")
        if not path.is_file():
            raise ValueError(f"--files: no such file: {path}")
        result.append(path)
    return result


def qualifying_cluster_times(reader: slug_reader, qhi_fraction: float) -> list[float]:
    """
    Find which of a cluster file's own output times are still worth running cloudy on.

    Parameters
    ----------
    reader : slugpy.slug_reader.slug_reader
        An open reader on a cluster-type slug output file, with a
        Q(HI)-only cluster_phot group (see make_slug_grid.py).
    qhi_fraction : float
        An output time qualifies if its own Q(HI) exceeds this
        fraction of Q(HI) at the earliest output time.

    Returns
    -------
    list of float
        The qualifying output times, in yr, in ascending order --
        possibly empty, if even the earliest output time's own Q(HI)
        is 0 (that time never qualifies against its own value, since
        the comparison is strict).

    Raises
    ------
    ValueError
        If reader has no cluster_phot group, or its cluster_phot has
        no Q(HI) filter.
    """
    phot = reader.cluster_phot
    if phot is None or "Q(HI)" not in phot.filters:
        raise ValueError(
            "qualifying_cluster_times: this file has no cluster_phot Q(HI) filter -- "
            "was it generated by an up-to-date make_slug_grid.py?")

    times = cast(u.Quantity, phot["time"]).to_value("yr")
    qhi = cast(u.Quantity, phot["Q(HI)"]).value

    order = times.argsort()
    times_sorted = times[order]
    qhi_sorted = qhi[order]

    threshold = qhi_fraction * qhi_sorted[0]
    return times_sorted[qhi_sorted > threshold].tolist()


def process_cluster_file(path: Path, qhi_fraction: float, log_u_values: list[float],
    cloudy_path: str | Path | None, executor: concurrent.futures.Executor,
    overwrite: bool, save_temp: bool) -> bool:
    """
    Run cloudy on every qualifying output time in one cluster-type slug output file.

    Parameters
    ----------
    path : pathlib.Path
        Path to the cluster-type slug HDF5 output file.
    qhi_fraction : float
        See qualifying_cluster_times.
    log_u_values : list of float
        log10 of the ionization parameter U to run cloudy at; the
        qualifying output times are run once per value, each at the
        fixed density NII_DEFAULT.
    cloudy_path : str or pathlib.Path, optional
        Passed through to slug_reader.run_cloudy.
    executor : concurrent.futures.Executor
        Shared cloudy-process pool passed through to
        slug_reader.run_cloudy's own executor argument -- this
        function's own decks are queued into it alongside every other
        file's, rather than run against a private pool of their own.
        This function blocks (on its own decks only) until they've all
        completed; it is meant to be called from its own driver thread
        so multiple files' own calls can be in flight against executor
        at once (see this module's own docstring).
    overwrite : bool
        Passed through to slug_reader.run_cloudy.
    save_temp : bool
        Passed through to slug_reader.run_cloudy.

    Returns
    -------
    bool
        True if every qualifying output time's own cloudy run
        succeeded at every log U value (or there were no qualifying
        output times), False if any failed.

    Details
    -------
    Each call passes its own file-specific temp_dir (see this module's
    own docstring for why that matters once multiple files' calls can
    be genuinely concurrent).
    """
    reader = slug_reader(str(path))
    times = qualifying_cluster_times(reader, qhi_fraction)
    if not times:
        print(f"  {path.name}: no output times with Q(HI) > {qhi_fraction:.0%} of the earliest time's own value; skipping")
        return True

    all_ok = True
    for log_u in log_u_values:
        print(f"  {path.name}: running cloudy on {len(times)} qualifying output time(s) at log U = {log_u}")
        result = reader.run_cloudy("cluster", time=times, nII=NII_DEFAULT, U=10.0 ** log_u,
            cloudy_path=cloudy_path, executor=executor, temp_dir=f"cloudy_tmp_{path.stem}",
            overwrite=overwrite, save_temp=save_temp, progress=False)
        successes = result if isinstance(result, list) else [result]
        all_ok &= all(successes)
    return all_ok


def process_galaxy_file(path: Path, log_u_values: list[float], cloudy_path: str | Path | None,
    executor: concurrent.futures.Executor, overwrite: bool, save_temp: bool) -> bool:
    """
    Run cloudy on one galaxy-type slug output file's own single (trial, output time).

    Parameters
    ----------
    path : pathlib.Path
        Path to the galaxy-type slug HDF5 output file.
    log_u_values : list of float
        log10 of the ionization parameter U to run cloudy at; the
        file's single spectrum is run once per value, each at the
        fixed density NII_DEFAULT.
    cloudy_path : str or pathlib.Path, optional
        Passed through to slug_reader.run_cloudy.
    executor : concurrent.futures.Executor
        Shared cloudy-process pool -- see process_cluster_file's own
        identical parameter.
    overwrite : bool
        Passed through to slug_reader.run_cloudy.
    save_temp : bool
        Passed through to slug_reader.run_cloudy.

    Returns
    -------
    bool
        True if every log U value's own run succeeded, False if any failed.

    Details
    -------
    Each call passes its own file-specific temp_dir -- see
    process_cluster_file's own identical detail.
    """
    reader = slug_reader(str(path))
    all_ok = True
    for log_u in log_u_values:
        result = reader.run_cloudy("galaxy", nII=NII_DEFAULT, U=10.0 ** log_u, cloudy_path=cloudy_path,
            executor=executor, temp_dir=f"cloudy_tmp_{path.stem}",
            overwrite=overwrite, save_temp=save_temp, progress=False)
        successes = result if isinstance(result, list) else [result]
        all_ok &= all(successes)
    return all_ok


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run cloudy, via slugpy's own run_cloudy, on every slug output "
            "file make_slug_grid.py wrote into a working directory.")
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR,
        help=f"Directory to look for slug *.h5 output files in (default: {DEFAULT_WORK_DIR}); "
            "should be the same --work-dir make_slug_grid.py was run with.")
    parser.add_argument("--files", nargs="+", default=None,
        help="Restrict to just these files instead of every *.h5 file in --work-dir, e.g. "
            "to rerun a single file that failed (with --save-temp, to inspect why). Matched "
            "by name against --work-dir, with or without a trailing \".h5\".")
    parser.add_argument("--cloudy-path", type=Path, default=None,
        help="Path to the cloudy executable. If omitted, run_cloudy's own default lookup "
            "($CLOUDY_DIR/cloudy.exe) applies.")
    parser.add_argument("--qhi-fraction", type=float, default=QHI_FRACTION,
        help=f"A cluster output time is only run through cloudy if its own Q(HI) exceeds "
            f"this fraction of Q(HI) at the earliest output time (default: {QHI_FRACTION}).")
    parser.add_argument("--log-u", type=float, nargs="+", default=list(LOG_U_VALUES),
        help="log10 of the ionization parameter(s) U to run cloudy at, at the fixed density "
            f"{NII_DEFAULT.value:g} cm^-3 (default: {list(LOG_U_VALUES)}).")
    parser.add_argument("--max-workers", type=int, default=None,
        help="Maximum number of cloudy processes to run at once, shared across every file "
            "in --work-dir at once (default: os.cpu_count()).")
    parser.add_argument("--overwrite", action="store_true",
        help="Passed through to run_cloudy: rerun and replace any already-stored cloudy "
            "results, rather than skipping them (default: skip).")
    parser.add_argument("--save-temp", action="store_true",
        help="Passed through to run_cloudy: keep each run's own cloudy_tmp directory "
            "instead of cleaning it up (default: clean up).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    h5_files = find_h5_files(args.work_dir, files=args.files)
    if not h5_files:
        print(f"No .h5 files found in {args.work_dir}")
        return

    cluster_files: list[Path] = []
    galaxy_files: list[Path] = []
    for path in h5_files:
        sim_type = slug_reader(str(path)).input_deck.get("sim_type")
        if sim_type == "cluster":
            cluster_files.append(path)
        elif sim_type == "galaxy":
            galaxy_files.append(path)
        else:
            raise ValueError(f"run_cloudy_grid: {path} has an unrecognized sim_type: {sim_type!r}")

    print(f"Found {len(cluster_files)} cluster file(s) and {len(galaxy_files)} galaxy file(s) in {args.work_dir}")

    max_workers = args.max_workers or (os.cpu_count() or 1)
    n_files = len(cluster_files) + len(galaxy_files)
    print(f"Running cloudy across all files at once, up to {max_workers} cloudy process(es) total...")

    all_ok = True
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as cloudy_executor:
        # One driver thread per file: each just writes/queues that
        # file's own decks into cloudy_executor and blocks on their
        # results, so an idle cloudy_executor worker always immediately
        # picks up whatever file's work is next (see this module's own
        # docstring). Driver threads do nothing but wait, so there's no
        # reason to cap how many run at once beyond one per file.
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_files) as driver_executor:
            futures: dict[concurrent.futures.Future[bool], Path] = {
                driver_executor.submit(process_cluster_file, path, args.qhi_fraction, args.log_u,
                    args.cloudy_path, cloudy_executor, args.overwrite, args.save_temp): path
                for path in cluster_files
            }
            futures.update({
                driver_executor.submit(process_galaxy_file, path, args.log_u,
                    args.cloudy_path, cloudy_executor, args.overwrite, args.save_temp): path
                for path in galaxy_files
            })
            for future in concurrent.futures.as_completed(futures):
                path = futures[future]
                try:
                    ok = future.result()
                except Exception as exc:  # noqa: BLE001 -- report every file's outcome, however it failed
                    all_ok = False
                    print(f"  {path.name}: FAILED ({exc!r})")
                else:
                    all_ok &= ok
                    print(f"  {path.name}: {'OK' if ok else 'FAILED (see cloudy_tmp/*.log if --save-temp was used)'}")

    print("All cloudy runs succeeded." if all_ok else "Some cloudy runs failed -- see above.")
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
