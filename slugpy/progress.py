"""
progress.py

A thin progress-bar factory that picks the right implementation for
the current environment: marimo's own native progress bar when
running inside a marimo notebook (tqdm's widgets don't render there),
tqdm otherwise (which auto-detects a Jupyter notebook vs. a plain
terminal on its own, via tqdm.auto). Also implements
run_with_progress, which drives a progress bar while running a call
(like SimCluster.run()) whose own progress can only be observed by
polling some other object while that call is still executing.

:copyright: Copyright (c) 2026 Mark Krumholz
"""

import threading
from collections.abc import Callable
from typing import Any

# marimo is an optional dependency: slugpy must work the same from a
# plain command-line Python session, a Jupyter notebook, or a script
# regardless of whether marimo happens to be installed at all --
# marimo's own presence is only relevant when actually running inside
# one of its notebooks (see running_in_marimo() below). It's
# deliberately not installed in CI's own pyright environment, so
# pyright can't resolve this import there either -- suppress that
# specific diagnostic rather than installing an optional dependency
# just to satisfy the type checker.
try:
    import marimo as _marimo  # pyright: ignore[reportMissingImports]
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


def run_with_progress[T](fn: Callable[[], T], get_current: Callable[[], int], total: int,
    desc: str, poll_interval: float = 0.2) -> T:
    """
    Run fn to completion while showing a progress bar tracking
    get_current() against total.

    Parameters
    ----------
    fn : callable
        Zero-argument callable to run to completion -- typically one
        that blocks (releasing the GIL) while some other thread or
        native code advances the count get_current() reports, e.g.
        SimCluster.run().
    get_current : callable
        Zero-argument callable returning the current count so far.
    total : int
        Total count expected once fn completes.
    desc : str
        Short label describing what's being tracked.
    poll_interval : float, default 0.2
        Seconds between successive polls of get_current.

    Returns
    -------
    Whatever fn() itself returned.

    Raises
    ------
    BaseException
        Whatever fn itself raised, re-raised here once fn's own
        background thread has finished.

    Details
    -------
    fn runs on a background thread; the progress bar is created and
    updated entirely on the calling thread, which instead just polls
    get_current() while periodically joining that background thread.
    This -- not the reverse -- is required for marimo's own progress
    bar: marimo tracks, in genuine thread-local state, which thread
    owns a given notebook cell's execution context, and silently drops
    any UI update made from another thread, so every bar.update() call
    must happen on the same thread that called run_with_progress in
    the first place, not a separate watcher thread polling while fn
    itself runs on the calling thread.
    """
    bar = make_progress_bar(total=total, desc=desc)
    error: list[BaseException] = []
    result: list[T] = []

    def _target() -> None:
        try:
            result.append(fn())
        except BaseException as exc:  # noqa: BLE001 -- re-raised on the calling thread below
            error.append(exc)

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()

    last = 0
    while True:
        thread.join(poll_interval)
        current = get_current()
        if current > last:
            bar.update(current - last)
            last = current
        if not thread.is_alive():
            break

    bar.close()
    if error:
        raise error[0]
    return result[0]
