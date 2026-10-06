"""
_cli.py

The ``slug`` console script for a pip-installed copy of slug.

A pip-installed slug places its compiled command-line executable
inside the slugpy package (``slugpy/_bin/slug``), next to the data
files bundled with it, rather than directly on the user's PATH. The
``slug`` command that pip does put on PATH is this small launcher,
which first points the executable at those data files (see
_paths.ensure_search_path) and then replaces itself with the real
executable, passing all of its command-line arguments through
unchanged.

:copyright: Copyright (c) 2026 Mark Krumholz
"""

import os
import sys
from pathlib import Path

from ._paths import ensure_search_path


def slug_executable() -> Path:
    """
    Return the path to the slug executable bundled with this package.

    Returns
    -------
    pathlib.Path
        ``slugpy/_bin/slug`` (``slug.exe`` on Windows). It exists only
        in a pip-installed copy of slugpy; it need not exist when this
        is called.
    """
    name = "slug.exe" if sys.platform == "win32" else "slug"
    return Path(__file__).resolve().parent / "_bin" / name


def main() -> None:
    """
    Run the bundled slug executable with this process's command-line arguments.

    Sets up SLUG_DATA_PATH (see _paths.ensure_search_path), then
    replaces the current process with the slug executable, so that its
    exit status, output, and signal handling are exactly those of
    running it directly.

    Raises
    ------
    SystemExit
        If the bundled executable does not exist -- e.g. when slugpy is
        being used from a source checkout rather than pip-installed, in
        which case the slug executable is in the CMake build directory
        instead.
    """
    exe = slug_executable()
    if not exe.is_file():
        raise SystemExit(
            f"slug: the slug executable was not found at {exe}. This launcher "
            "only works for a pip-installed copy of slugpy; in a source "
            "checkout, run the slug executable from your CMake build "
            "directory instead.")
    ensure_search_path()
    os.execv(exe, [str(exe), *sys.argv[1:]])
