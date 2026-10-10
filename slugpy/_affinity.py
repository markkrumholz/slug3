"""
_affinity.py

Preserve the process's CPU affinity across slugpy's own import.

The compiled _slug extension links the OpenMP runtime (libgomp), and when
any OpenMP thread-binding variable is set (OMP_PROC_BIND, OMP_PLACES,
GOMP_CPU_AFFINITY), libgomp pins the importing thread to the first CPU of
its place list as soon as it loads. Inside a Python process that pinning
is harmless in itself, but it is inherited by everything the pinned
thread starts afterward -- most importantly the real slug executable,
which the ``slug`` console script (see _cli.py) replaces itself with via
os.execv, and cloudy subprocesses run by slugpy.cloudy. Those then
compute their own OpenMP places from a one-CPU mask, so every OpenMP
thread they start runs on that single CPU.

This module is imported first by slugpy/__init__.py, before _slug, so
that it records the affinity the process started with; __init__.py then
calls restore_initial_affinity() right after importing _slug. Under an
MPI launcher that binds each rank to a set of cores, the recorded mask
is that rank's own set, so intentional binding is preserved. Platforms
without CPU affinity support (e.g. macOS) are left untouched.

:copyright: Copyright (c) 2026 Mark Krumholz
"""

import os
from collections.abc import Callable, Iterable

# Looked up dynamically, since these functions only exist on some
# platforms (e.g. Linux, not macOS)
_getaffinity: Callable[[int], set[int]] | None = getattr(os, "sched_getaffinity", None)
_setaffinity: Callable[[int, Iterable[int]], None] | None = getattr(os, "sched_setaffinity", None)

_initial_affinity: frozenset[int] | None = (
    frozenset(_getaffinity(0)) if _getaffinity is not None else None)


def initial_affinity() -> frozenset[int] | None:
    """
    Return the CPU affinity recorded when slugpy started being imported.

    Returns
    -------
    frozenset of int or None
        The set of CPUs the importing thread was allowed to run on before
        _slug (and with it the OpenMP runtime) was loaded, or None on a
        platform without CPU affinity support.
    """
    return _initial_affinity


def restore_initial_affinity() -> None:
    """
    Restore the calling thread's CPU affinity to initial_affinity().

    A no-op if the affinity has not changed, or on a platform without CPU
    affinity support. See this module's own docstring for why this is
    needed.
    """
    if _initial_affinity is None or _getaffinity is None or _setaffinity is None:
        return
    if frozenset(_getaffinity(0)) != _initial_affinity:
        _setaffinity(0, _initial_affinity)
