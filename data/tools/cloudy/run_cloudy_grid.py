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
  cloudy run -- so there is no internal parallelism for a single
  run_cloudy() call to exploit. Concurrency instead comes from running
  several galaxy files' own run_cloudy() calls at once, via a thread
  pool (each one blocks on its own external cloudy subprocess and
  releases the GIL while doing so, so threads are enough).
- A cluster-type file has one cluster but many output times (see
  make_slug_grid.py's own 0.25-10 Myr, 0.25 Myr cadence), most of
  which are not worth running cloudy on: once the cluster's own
  ionizing luminosity Q(HI) has dropped well below its own peak, the
  resulting nebular emission is negligible. This script reads each
  cluster file's own Q(HI) photometry (see make_slug_grid.py's own
  Q(HI)-only [phot] stanza) and only requests cloudy runs for output
  times where Q(HI) exceeds --qhi-fraction (default 1%) of Q(HI) at
  the earliest output time -- passed to a single run_cloudy() call as
  a list of times, so slugpy's own run_cloudy_decks thread pool
  parallelizes across them internally, rather than this script
  managing that concurrency itself.

Every qualifying spectrum (every qualifying output time for a cluster
file, the single output time for a galaxy file) is run through cloudy
once per ionization parameter in --log-u (default log U = -3, -2.5,
-2), at the fixed density nII = 100 cm^-3 run_cloudy itself otherwise
defaults to -- i.e. this script always passes both nII and U
explicitly, rather than relying on run_cloudy's own no-arguments
default, so that only U varies across runs. These per-U runs are
separate run_cloudy() calls (duplicate detection is keyed on the full
set of six hiiregparam values, so they can't collide), issued one log
U value at a time so cluster runs stay parallelized over qualifying
output times rather than over log U.

Cluster files are processed one at a time (each one's own internal
concurrency already uses up to --max-workers cloudy processes at
once); galaxy files are then processed together, up to --max-workers
of them at once -- the two groups are not run concurrently with each
other, to avoid oversubscribing the machine beyond --max-workers
cloudy processes running at any one time.

Like make_slug_grid.py, this is far too expensive to run in full on a
laptop -- verify locally against the same narrow --feh-min/--feh-max
slice make_slug_grid.py was run with, then run the full grid on a
shared cluster.

:copyright: Copyright (c) 2026 Mark Krumholz
"""

# Imports
import argparse
import concurrent.futures
import sys
from pathlib import Path
from typing import cast

import astropy.units as u

# slugpy is a sibling package three directories up from this script
# (data/tools/cloudy -> data/tools -> data -> repo root), and is not
# installed, so it has to be reached via sys.path directly.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from slugpy.slug_reader import slug_reader  # noqa: E402 -- see sys.path setup above

QHI_FRACTION = 0.01

# Matches run_cloudy's own no-arguments default density, so passing it
# explicitly alongside a varying U doesn't change the density grid.
NII_DEFAULT = 100.0 / u.cm ** 3

LOG_U_VALUES = (-3.0, -2.5, -2.0)

DEFAULT_WORK_DIR = Path(__file__).resolve().parent / "slug_grid_work"


def find_h5_files(work_dir: Path) -> list[Path]:
    """
    Find every slug HDF5 output file make_slug_grid.py wrote.

    Parameters
    ----------
    work_dir : pathlib.Path
        Directory to search (non-recursively).

    Returns
    -------
    list of pathlib.Path
        Every "*.h5" file directly in work_dir, sorted by name.
    """
    return sorted(work_dir.glob("*.h5"))


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
    cloudy_path: str | Path | None, max_workers: int | None, overwrite: bool, save_temp: bool) -> bool:
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
    max_workers : int, optional
        Passed through to slug_reader.run_cloudy -- the qualifying
        output times found here are run concurrently, up to this many
        cloudy processes at once, for each log U value in turn.
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
            cloudy_path=cloudy_path, max_workers=max_workers, overwrite=overwrite,
            save_temp=save_temp, progress=False)
        successes = result if isinstance(result, list) else [result]
        all_ok &= all(successes)
    return all_ok


def process_galaxy_file(path: Path, log_u_values: list[float], cloudy_path: str | Path | None,
    max_workers: int | None, overwrite: bool, save_temp: bool) -> bool:
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
    cloudy_path, max_workers, overwrite, save_temp
        Passed through to slug_reader.run_cloudy.

    Returns
    -------
    bool
        True if every log U value's own run succeeded, False if any failed.
    """
    reader = slug_reader(str(path))
    all_ok = True
    for log_u in log_u_values:
        result = reader.run_cloudy("galaxy", nII=NII_DEFAULT, U=10.0 ** log_u, cloudy_path=cloudy_path,
            max_workers=max_workers, overwrite=overwrite, save_temp=save_temp, progress=False)
        successes = result if isinstance(result, list) else [result]
        all_ok &= all(successes)
    return all_ok


def process_galaxy_files(paths: list[Path], log_u_values: list[float], cloudy_path: str | Path | None,
    max_workers: int | None, overwrite: bool, save_temp: bool) -> dict[Path, bool]:
    """
    Run cloudy on several galaxy-type slug output files concurrently.

    Parameters
    ----------
    paths : list of pathlib.Path
        Paths to the galaxy-type slug HDF5 output files to process.
    log_u_values : list of float
        Passed through to process_galaxy_file.
    cloudy_path, max_workers, overwrite, save_temp
        Passed through to process_galaxy_file.

    Returns
    -------
    dict mapping pathlib.Path to bool
        Whether each file's own cloudy run succeeded, keyed by path.

    Details
    -------
    Each galaxy file has exactly one (trial, output time) pair to run
    cloudy on per log U value, so there is no internal parallelism for
    any single file's own run_cloudy() calls to exploit -- concurrency
    instead comes from running up to max_workers files at once, via a
    thread pool (each worker just blocks on process_galaxy_file's own
    external cloudy subprocesses and releases the GIL while doing so,
    so threads are enough).
    """
    results: dict[Path, bool] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_galaxy_file, path, log_u_values, cloudy_path,
                max_workers, overwrite, save_temp): path
            for path in paths
        }
        for future in concurrent.futures.as_completed(futures):
            path = futures[future]
            results[path] = future.result()
    return results


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run cloudy, via slugpy's own run_cloudy, on every slug output "
            "file make_slug_grid.py wrote into a working directory.")
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR,
        help=f"Directory to look for slug *.h5 output files in (default: {DEFAULT_WORK_DIR}); "
            "should be the same --work-dir make_slug_grid.py was run with.")
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
        help="Maximum number of cloudy processes to run at once, whether across a single "
            "cluster file's own qualifying times or across galaxy files (default: os.cpu_count()).")
    parser.add_argument("--overwrite", action="store_true",
        help="Passed through to run_cloudy: rerun and replace any already-stored cloudy "
            "results, rather than skipping them (default: skip).")
    parser.add_argument("--save-temp", action="store_true",
        help="Passed through to run_cloudy: keep each run's own cloudy_tmp directory "
            "instead of cleaning it up (default: clean up).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    h5_files = find_h5_files(args.work_dir)
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

    all_ok = True

    if cluster_files:
        print("Running cloudy on cluster files (one file at a time; each file's own "
            "qualifying output times are parallelized internally)...")
        for path in cluster_files:
            ok = process_cluster_file(path, args.qhi_fraction, args.log_u, args.cloudy_path,
                args.max_workers, args.overwrite, args.save_temp)
            all_ok &= ok
            print(f"  {path.name}: {'OK' if ok else 'FAILED (see cloudy_tmp/*.log if --save-temp was used)'}")

    if galaxy_files:
        print(f"Running cloudy on {len(galaxy_files)} galaxy file(s), "
            f"up to {args.max_workers or 'os.cpu_count()'} at once...")
        results = process_galaxy_files(galaxy_files, args.log_u, args.cloudy_path,
            args.max_workers, args.overwrite, args.save_temp)
        for path in galaxy_files:
            ok = results[path]
            all_ok &= ok
            print(f"  {path.name}: {'OK' if ok else 'FAILED (see cloudy_tmp/*.log if --save-temp was used)'}")

    print("All cloudy runs succeeded." if all_ok else "Some cloudy runs failed -- see above.")


if __name__ == "__main__":
    main()
