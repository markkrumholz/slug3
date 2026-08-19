"""
cloudy_process.py

Implements the machinery to locate the cloudy executable and launch it
on one or more written cloudy input decks, running multiple decks
concurrently via concurrent.futures.

:copyright: Copyright (c) 2026 Mark Krumholz
"""

import concurrent.futures
import os
import subprocess
from collections.abc import Sequence
from pathlib import Path


def find_cloudy_executable(cloudy_path: str | Path | None = None) -> Path:
    """
    Locate the cloudy executable.

    Parameters
    ----------
    cloudy_path : str or pathlib.Path, optional
        Explicit path to the cloudy executable. If omitted, the
        executable is looked up as $CLOUDY_DIR/cloudy.exe.

    Returns
    -------
    pathlib.Path
        Absolute path to the cloudy executable.

    Raises
    ------
    RuntimeError
        If cloudy_path is omitted and the CLOUDY_DIR environment
        variable is not set, or if the resolved path is not an
        executable file.
    """
    if cloudy_path is not None:
        exe = Path(cloudy_path)
    else:
        cloudy_dir = os.environ.get("CLOUDY_DIR")
        if cloudy_dir is None:
            raise RuntimeError(
                "find_cloudy_executable: no cloudy_path given, and "
                "CLOUDY_DIR is not set in the environment")
        exe = Path(cloudy_dir) / "cloudy.exe"
    if not exe.is_file() or not os.access(exe, os.X_OK):
        raise RuntimeError(
            f"find_cloudy_executable: no executable cloudy binary found at {exe}")
    return exe.resolve()


def run_cloudy_deck(input_path: str | Path, cloudy_exe: str | Path) -> Path:
    """
    Run cloudy on a single input deck.

    Parameters
    ----------
    input_path : str or pathlib.Path
        Path to the cloudy input deck to run (as written by
        write_cloudy_input).
    cloudy_exe : str or pathlib.Path
        Path to the cloudy executable to run (see
        find_cloudy_executable).

    Returns
    -------
    pathlib.Path
        Path to the file that cloudy's own stdout/stderr were
        captured into: input_path with its suffix replaced by ".out".
        Cloudy's own output files (.con/.lines/.linearr/.hcon, as
        requested by the input deck's own "save" commands) are written
        alongside input_path, since cloudy is run with input_path's
        own parent directory as its working directory.

    Details
    -------
    cloudy is invoked with the input deck piped in on stdin
    (cloudy.exe < deck.in), not as a command-line argument -- passing
    it as an argument causes cloudy to double-prefix its own output
    filenames.
    """
    input_path = Path(input_path)
    output_path = input_path.with_suffix(".out")
    with open(input_path, "rb") as fin, open(output_path, "wb") as fout:
        subprocess.run([str(cloudy_exe)], stdin=fin, stdout=fout,
            stderr=subprocess.STDOUT, cwd=input_path.parent, check=False)
    return output_path


def run_cloudy_decks(input_paths: Sequence[str | Path], cloudy_exe: str | Path,
    max_workers: int | None = None) -> list[Path]:
    """
    Run cloudy concurrently on multiple input decks.

    Parameters
    ----------
    input_paths : sequence of str or pathlib.Path
        Paths to the cloudy input decks to run.
    cloudy_exe : str or pathlib.Path
        Path to the cloudy executable to run (see
        find_cloudy_executable).
    max_workers : int, optional
        Maximum number of cloudy processes to run at once. Defaults to
        the number of available CPUs.

    Returns
    -------
    list of pathlib.Path
        The path each deck's own captured stdout/stderr was written
        to (see run_cloudy_deck), in the same order as input_paths.

    Details
    -------
    Runs are dispatched onto a concurrent.futures.ThreadPoolExecutor
    rather than a ProcessPoolExecutor: each worker just blocks on an
    external cloudy subprocess and releases the GIL while doing so, so
    threads are enough, and avoid the pickling overhead of processes.
    At most max_workers decks run at once; as each finishes, the next
    queued one is launched, until all of input_paths have been run.
    """
    if max_workers is None:
        max_workers = os.cpu_count() or 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_cloudy_deck, path, cloudy_exe) for path in input_paths]
        return [future.result() for future in futures]


def cloudy_run_succeeded(out_path: str | Path) -> bool:
    """
    Determine whether a cloudy run completed successfully.

    Parameters
    ----------
    out_path : str or pathlib.Path
        Path to the file cloudy's own stdout/stderr were captured
        into (see run_cloudy_deck).

    Returns
    -------
    bool
        True if the last non-blank line of out_path contains "Cloudy
        exited OK" (cloudy's own success message, e.g. "[Stop in
        cdMain at maincl.cpp:590, Cloudy exited OK]"), False
        otherwise -- including if out_path doesn't exist or is empty.
    """
    out_path = Path(out_path)
    if not out_path.is_file():
        return False
    for line in reversed(out_path.read_text().splitlines()):
        if line.strip():
            return "Cloudy exited OK" in line
    return False
