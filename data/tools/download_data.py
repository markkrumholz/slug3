"""
Download slug's large data files (stellar tracks, spectral libraries,
filters, etc.) from the public mirror on the ANU Data Commons, and
install them into the matching subdirectories of this repository's own
data/ directory.

This is a thin wrapper around slugpy/download_data.py (which a
pip-installed slugpy runs as `python -m slugpy.download_data`), that
just changes its default destination from the per-user data directory
to this repository's own data/. That module is loaded directly from
its source file rather than via `import slugpy`, so that this works in
a fresh checkout before the compiled _slug extension has been built.
See --help for the full detail (merge rules, etc.); see also AGENTS.md's
own Data files section.

Run from anywhere, e.g. from the repository root:
    python3 data/tools/download_data.py
    python3 data/tools/download_data.py --subdir filters
    python3 data/tools/download_data.py --overwrite --subdir tracks

:copyright: Copyright (c) 2026 Mark Krumholz
"""

import importlib.util
import pathlib
import sys

# data/tools/download_data.py -> data/ is one directory up, and the
# repository root two
DEFAULT_DEST = pathlib.Path(__file__).resolve().parent.parent
MODULE_PATH = DEFAULT_DEST.parent / "slugpy" / "download_data.py"


def main() -> None:
    spec = importlib.util.spec_from_file_location("slug_download_data", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not load {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    argv = sys.argv[1:]
    if not any(a == "--dest" or a.startswith("--dest=") for a in argv):
        argv = ["--dest", str(DEFAULT_DEST), *argv]
    module.main(argv)


if __name__ == "__main__":
    main()
