"""
slugpy.cloudy: interface between slug and the cloudy photoionization code.

Pure-Python additions related to running cloudy on slug spectra and
ingesting its output live in their own modules under this subpackage.
"""

from .cloudy_input import DEFAULT_TEMPLATE, write_cloudy_input
from .hiiregparam import hiiregparam

__all__ = ["DEFAULT_TEMPLATE", "hiiregparam", "write_cloudy_input"]
