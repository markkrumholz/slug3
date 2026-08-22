"""
progress.py

A thin progress-bar factory that picks the right implementation for
the current environment: marimo's own native progress bar when
running inside a marimo notebook (tqdm's widgets don't render there),
tqdm otherwise (which auto-detects a Jupyter notebook vs. a plain
terminal on its own, via tqdm.auto). Also implements ProgressWatcher,
which drives a progress bar from a background thread polling a
counter callback -- for tracking a call (like SimCluster.run()) whose
own progress can only be observed by querying some other object while
that call is still executing.

:copyright: Copyright (c) 2026 Mark Krumholz
"""

import threading
from collections.abc import Callable
from typing import Any

# marimo is an optional dependency: slugpy must work the same from a
# plain command-line Python session, a Jupyter notebook, or a script
# regardless of whether marimo happens to be installed at all --
# marimo's own presence is only relevant when actually running inside
# one of its notebooks (see running_in_marimo() below).
try:
    import marimo as _marimo
except ImportError:
    _marimo = None


def running_in_marimo() -> bool:
    """
    Returns
    -------
    bool
        True if currently executing inside a live marimo notebook
        kernel, False otherwise -- including if marimo isn't installed
        at all.
    """
    return _marimo is not None and _marimo.running_in_notebook()


class _MarimoProgressBar:
    """
    Adapts marimo's own mo.status.progress_bar -- a context manager
    whose update() takes a keyword-only increment -- to the tqdm-style
    update(n)/close() interface make_progress_bar's own callers use.
    """

    def __init__(self, total: int, desc: str) -> None:
        assert _marimo is not None
        self._cm = _marimo.status.progress_bar(total=total, title=desc)
        self._bar = self._cm.__enter__()

    def update(self, n: int = 1) -> None:
        self._bar.update(increment=n)

    def close(self) -> None:
        self._cm.__exit__(None, None, None)


def make_progress_bar(total: int, desc: str) -> Any:
    """
    Construct a progress bar appropriate to the current environment.

    Parameters
    ----------
    total : int
        Total number of steps the returned progress bar represents.
    desc : str
        Short label describing what's being tracked.

    Returns
    -------
    A progress bar object with update(n=1) and close() methods:
    marimo's own native progress bar (see running_in_marimo()) if
    running inside a live marimo notebook kernel -- tqdm's own widgets
    don't render there -- otherwise a tqdm.auto.tqdm instance, which
    itself auto-detects a Jupyter notebook vs. a plain terminal and
    renders accordingly.
    """
    if running_in_marimo():
        return _MarimoProgressBar(total, desc)
    from tqdm.auto import tqdm
    return tqdm(total=total, desc=desc)


class ProgressWatcher:
    """
    Context manager that drives a progress bar from a background
    thread polling a counter callback, for wrapping a call that blocks
    (typically because it releases the GIL) while some other thread or
    native code advances that counter -- e.g. SimCluster/SimGalaxy's
    own run(), whose progress can only be observed via
    trialsCompleted() while it executes.

    Parameters
    ----------
    get_current : callable
        Zero-argument callable returning the current count so far.
    total : int
        Total count expected once the wrapped call completes.
    desc : str
        Short label describing what's being tracked.
    poll_interval : float, default 0.2
        Seconds between successive polls of get_current.

    Examples
    --------
    >>> with ProgressWatcher(sim.trialsCompleted, n_trial, "Running trials"):
    ...     sim.run()
    """

    def __init__(self, get_current: Callable[[], int], total: int, desc: str,
        poll_interval: float = 0.2) -> None:
        self._get_current = get_current
        self._bar = make_progress_bar(total=total, desc=desc)
        self._poll_interval = poll_interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._watch, daemon=True)

    def _watch(self) -> None:
        last = 0
        while True:
            stopped = self._stop.wait(self._poll_interval)
            current = self._get_current()
            if current > last:
                self._bar.update(current - last)
                last = current
            if stopped:
                return

    def __enter__(self) -> "ProgressWatcher":
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self._stop.set()
        self._thread.join()
        self._bar.close()
