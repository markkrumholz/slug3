"""
slugpy.cloudy: interface between slug and the cloudy photoionization code.

Pure-Python additions related to running cloudy on slug spectra and
ingesting its output live in their own modules under this subpackage.
"""

from .hiiregparam import hiiregparam

__all__ = ["hiiregparam"]
