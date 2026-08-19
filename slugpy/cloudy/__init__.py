"""
slugpy.cloudy: interface between slug and the cloudy photoionization code.

Pure-Python additions related to running cloudy on slug spectra and
ingesting its output live in their own modules under this subpackage.

:copyright: Copyright (c) 2026 Mark Krumholz
"""

from .hiiregparam import hiiregparam

__all__ = ["hiiregparam"]
