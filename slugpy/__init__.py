"""
slugpy: the Python frontend for slug.

The classes and functions that wrap slug's C++ core (Cluster,
SimControls, Tracks2D/Tracks3D, Specsyn and its subclasses, the Filter
family, PDF, etc.) are implemented in the compiled extension module
_slug (see src/pybind/Bindings.cpp) and re-exported here with a
wildcard import, so that adding a new binding there does not also
require updating this list by hand, and so that users can write
`from slugpy import Cluster` rather than reaching into the private
_slug submodule directly.

Pure-Python additions -- reading and post-processing slug output files
-- live in their own modules under this package and are imported
below explicitly.

:copyright: Copyright (c) 2026 Mark Krumholz
"""

from ._slug import *  # noqa: F401,F403
from .compute_isochrones import compute_isochrones
from .phot_convert import phot_convert
from .read import read
from .run_sim import run_sim
