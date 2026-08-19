"""
slugpy.cloudy: interface between slug and the cloudy photoionization code.

Pure-Python additions related to running cloudy on slug spectra and
ingesting its output live in their own modules under this subpackage.

:copyright: Copyright (c) 2026 Mark Krumholz
"""

from .cloudy_continuum import read_cloudy_continuum
from .cloudy_input import DEFAULT_TEMPLATE, write_cloudy_input
from .cloudy_output import CloudyRunResult, write_cloudy_h5_results
from .cloudy_process import (
    cloudy_run_succeeded,
    find_cloudy_executable,
    run_cloudy_deck,
    run_cloudy_decks,
)
from .hiiregparam import hiiregparam

__all__ = [
    "DEFAULT_TEMPLATE",
    "CloudyRunResult",
    "cloudy_run_succeeded",
    "find_cloudy_executable",
    "hiiregparam",
    "read_cloudy_continuum",
    "run_cloudy_deck",
    "run_cloudy_decks",
    "write_cloudy_h5_results",
    "write_cloudy_input",
]
