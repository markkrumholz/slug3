"""
_paths.py

Locations slug searches for its data files when it is pip-installed,
and the SLUG_DATA_PATH environment variable that tells the C++ side
(utils::getFilePath, in src/utils/MiscUtils.hpp) about them.

A plain, in-tree build of slug (see AGENTS.md) finds its data via the
SLUG_DIR environment variable or the source tree's own location,
baked in at compile time. A pip-installed copy has no source tree, so
it instead relies on two directories listed in SLUG_DATA_PATH:

- the per-user data directory (see user_data_dir), where large data
  files (stellar tracks, spectral libraries, nebular tables) are
  downloaded by default (see slugpy.download_data); and
- the package's own share directory (see package_share_dir), holding
  the small data files that are bundled into the package itself.

Both directories mirror the repository's own layout (data/imfs/,
data/tracks/, etc.), so every file name slug resolves relative to
SLUG_DIR resolves the same way relative to them.

:copyright: Copyright (c) 2026 Mark Krumholz
"""

import os
from pathlib import Path

import platformdirs

SLUG_DATA_PATH_VAR: str = "SLUG_DATA_PATH"
"""Name of the environment variable listing extra data search directories."""


def package_share_dir() -> Path:
    """
    Return the directory holding the data files bundled into slugpy.

    Returns
    -------
    pathlib.Path
        ``slugpy/_share`` inside the installed package. It exists only
        in a pip-installed copy of slugpy; in an in-tree checkout,
        those same files are instead found in the source tree itself.
    """
    return Path(__file__).resolve().parent / "_share"


def user_data_dir() -> Path:
    """
    Return the per-user directory where slug's large data is stored by default.

    Returns
    -------
    pathlib.Path
        The platform's standard per-user data directory for slug, e.g.
        ``~/.local/share/slug`` on Linux or
        ``~/Library/Application Support/slug`` on macOS. It need not
        exist yet.
    """
    return Path(platformdirs.user_data_dir("slug", appauthor=False))


def default_data_root() -> Path:
    """
    Return the directory that large data is downloaded into by default.

    Returns
    -------
    pathlib.Path
        The directory named by the SLUG_DIR environment variable, if it
        is set and non-empty, and otherwise user_data_dir(). Data files
        are installed in the data/ subdirectory of this directory (e.g.
        data/tracks/), mirroring the repository's own layout.
    """
    slug_dir = os.environ.get("SLUG_DIR", "")
    if slug_dir:
        return Path(slug_dir)
    return user_data_dir()


def ensure_search_path() -> None:
    """
    Add slugpy's data directories to the SLUG_DATA_PATH environment variable.

    Prepends user_data_dir() and then, if it exists,
    package_share_dir() to SLUG_DATA_PATH, so that the C++ side looks
    in the per-user directory first and then in the bundled data, after
    the working directory and SLUG_DIR but before anything already
    listed in SLUG_DATA_PATH. Directories already present are not added
    again, so calling this more than once has no further effect.
    Setting os.environ also sets the process's own environment, so this
    takes effect both for the _slug extension module and for any
    subprocess (e.g. the slug executable) started afterward.
    """
    existing = [d for d in os.environ.get(SLUG_DATA_PATH_VAR, "").split(os.pathsep) if d]
    ours = [str(user_data_dir())]
    share = package_share_dir()
    if share.is_dir():
        ours.append(str(share))
    new_dirs = [d for d in ours if d not in existing]
    if new_dirs:
        os.environ[SLUG_DATA_PATH_VAR] = os.pathsep.join(new_dirs + existing)
