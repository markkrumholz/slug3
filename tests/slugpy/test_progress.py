"""
Unit tests for slugpy.progress (running_in_marimo, make_progress_bar,
run_with_progress).

Uses small hand-built fakes for both tqdm and marimo's own APIs,
rather than depending on marimo actually being installed (it's an
optional dependency -- see progress.py's own comment) or on tqdm's own
real rendering, so these tests are self-contained and independent of
the environment they happen to run in.

:copyright: Copyright (c) 2026 Mark Krumholz
"""

import threading
import time

import pytest

from slugpy import progress


class _FakeMarimoBar:
    def __init__(self) -> None:
        self.updates: list[int] = []
        self.closed = False

    def update(self, increment: int = 1) -> None:
        self.updates.append(increment)


class _FakeMarimoBarCM:
    def __init__(self, bar: _FakeMarimoBar) -> None:
        self._bar = bar

    def __enter__(self) -> _FakeMarimoBar:
        return self._bar

    def __exit__(self, *args: object) -> None:
        self._bar.closed = True


class _FakeMarimoStatus:
    def __init__(self) -> None:
        self.calls: list[tuple[int, str]] = []
        self.bars: list[_FakeMarimoBar] = []

    def progress_bar(self, total: int, title: str) -> _FakeMarimoBarCM:
        self.calls.append((total, title))
        bar = _FakeMarimoBar()
        self.bars.append(bar)
        return _FakeMarimoBarCM(bar)


class _FakeMarimoModule:
    def __init__(self, running: bool) -> None:
        self.status = _FakeMarimoStatus()
        self._running = running

    def running_in_notebook(self) -> bool:
        return self._running


# ---------------------------------------------------------------------
# running_in_marimo
# ---------------------------------------------------------------------

def test_running_in_marimo_false_when_marimo_not_installed(monkeypatch):
    monkeypatch.setattr(progress, "_marimo", None)
    assert progress.running_in_marimo() is False


def test_running_in_marimo_reflects_marimo_own_check(monkeypatch):
    monkeypatch.setattr(progress, "_marimo", _FakeMarimoModule(running=True))
    assert progress.running_in_marimo() is True

    monkeypatch.setattr(progress, "_marimo", _FakeMarimoModule(running=False))
    assert progress.running_in_marimo() is False


# ---------------------------------------------------------------------
# make_progress_bar
# ---------------------------------------------------------------------

def test_make_progress_bar_uses_tqdm_outside_marimo(monkeypatch):
    monkeypatch.setattr(progress, "_marimo", None)
    bar = progress.make_progress_bar(total=5, desc="widgets")
    try:
        from tqdm.auto import tqdm
        assert isinstance(bar, tqdm)
        assert bar.total == 5
    finally:
        bar.close()


def test_make_progress_bar_uses_marimo_when_running_in_notebook(monkeypatch):
    fake = _FakeMarimoModule(running=True)
    monkeypatch.setattr(progress, "_marimo", fake)

    bar = progress.make_progress_bar(total=5, desc="widgets")
    assert isinstance(bar, progress._MarimoProgressBar)
    assert fake.status.calls == [(5, "widgets")]
    fake_bar = fake.status.bars[0]

    bar.update(3)
    assert fake_bar.updates == [3]

    bar.close()
    assert fake_bar.closed is True


# ---------------------------------------------------------------------
# run_with_progress
# ---------------------------------------------------------------------

class _FakeBar:
    def __init__(self) -> None:
        self.updates: list[int] = []
        self.closed = False

    def update(self, n: int = 1) -> None:
        self.updates.append(n)

    def close(self) -> None:
        self.closed = True


def test_run_with_progress_polls_and_reports_final_count(monkeypatch):
    """A counter advanced by fn (on its own background thread) while the calling thread polls ends up fully reflected once fn completes, even between polls."""
    fake_bar = _FakeBar()
    monkeypatch.setattr(progress, "make_progress_bar", lambda total, desc: fake_bar)

    state = {"n": 0}

    def fn():
        state["n"] = 1
        time.sleep(0.05)
        state["n"] = 3

    progress.run_with_progress(fn, lambda: state["n"], total=3, desc="trials", poll_interval=0.01)

    assert sum(fake_bar.updates) == 3
    assert fake_bar.closed is True


def test_run_with_progress_no_updates_when_counter_never_advances(monkeypatch):
    fake_bar = _FakeBar()
    monkeypatch.setattr(progress, "make_progress_bar", lambda total, desc: fake_bar)

    progress.run_with_progress(lambda: time.sleep(0.03), lambda: 0, total=3, desc="trials",
        poll_interval=0.01)

    assert fake_bar.updates == []
    assert fake_bar.closed is True


def test_run_with_progress_propagates_exceptions_from_fn(monkeypatch):
    """An exception raised inside fn (on its own background thread) still stops and closes the bar cleanly, and is re-raised on the calling thread."""
    fake_bar = _FakeBar()
    monkeypatch.setattr(progress, "make_progress_bar", lambda total, desc: fake_bar)

    def fn():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        progress.run_with_progress(fn, lambda: 0, total=1, desc="trials", poll_interval=0.01)

    assert fake_bar.closed is True


def test_run_with_progress_updates_happen_on_the_calling_thread(monkeypatch):
    """Every bar.update() call happens on the thread that called run_with_progress, not on fn's own background thread -- this is what makes the progress bar actually work under marimo, whose runtime context is genuine thread-local state (see progress.py's own comment)."""
    update_threads: list[threading.Thread] = []

    class _TrackingBar(_FakeBar):
        def update(self, n: int = 1) -> None:
            update_threads.append(threading.current_thread())
            super().update(n)

    tracking_bar = _TrackingBar()
    monkeypatch.setattr(progress, "make_progress_bar", lambda total, desc: tracking_bar)

    state = {"n": 0}

    def fn():
        state["n"] = 1
        time.sleep(0.03)
        state["n"] = 2

    calling_thread = threading.current_thread()
    progress.run_with_progress(fn, lambda: state["n"], total=2, desc="trials", poll_interval=0.01)

    assert update_threads
    assert all(t is calling_thread for t in update_threads)
