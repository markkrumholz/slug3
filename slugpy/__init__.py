"""
slugpy: the Python frontend for slug.

:copyright: Copyright (c) 2026 Mark Krumholz
"""

# The classes and functions that wrap slug's C++ core (Cluster,
# SimControls, Tracks2D/Tracks3D, Specsyn and its subclasses, the Filter
# family, PDF, etc.) are implemented in the compiled extension module
# _slug (see src/pybind/Bindings.cpp) and re-exported here with a
# wildcard import, so that adding a new binding there does not also
# require updating this list by hand, and so that users can write
# `from slugpy import Cluster` rather than reaching into the private
# _slug submodule directly.

# Pure-Python additions -- reading and post-processing slug output files
# -- live in their own modules under this package and are imported
# below explicitly.

from ._paths import ensure_search_path as _ensure_search_path
from ._slug import *
from .compute_isochrones import compute_isochrones as compute_isochrones
from .compute_tracks import compute_tracks as compute_tracks
from .phot_convert import phot_convert as phot_convert
from .read import read as read
from .run_sim import run_sim as run_sim

# Tell the C++ side where to find slug's data files when slugpy is
# pip-installed (see _paths.py): the bundled data inside this package,
# and the per-user directory large data is downloaded into. The C++
# side reads SLUG_DATA_PATH only when it actually looks for a file, not
# when _slug is imported, so this need not precede the imports above.
_ensure_search_path()
