"""
Unit tests for slugpy.progress (running_in_marimo, make_progress_bar,
ProgressWatcher).

Uses small hand-built fakes for both tqdm and marimo's own APIs,
rather than depending on marimo actually being installed (it's an
optional dependency -- see progress.py's own comment) or on tqdm's own
real rendering, so these tests are self-contained and independent of
the environment they happen to run in.

:copyright: Copyright (c) 2026 Mark Krumholz
"""

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
# ProgressWatcher
# ---------------------------------------------------------------------

class _FakeBar:
    def __init__(self) -> None:
        self.updates: list[int] = []
        self.closed = False

    def update(self, n: int = 1) -> None:
        self.updates.append(n)

    def close(self) -> None:
        self.closed = True


def test_progress_watcher_polls_and_reports_final_count(monkeypatch):
    """A counter advanced from the main thread while the watcher's own background thread is polling ends up fully reflected once the context manager exits, even between polls."""
    fake_bar = _FakeBar()
    monkeypatch.setattr(progress, "make_progress_bar", lambda total, desc: fake_bar)

    state = {"n": 0}
    with progress.ProgressWatcher(lambda: state["n"], total=3, desc="trials", poll_interval=0.01):
        state["n"] = 1
        time.sleep(0.05)
        state["n"] = 3

    assert sum(fake_bar.updates) == 3
    assert fake_bar.closed is True


def test_progress_watcher_no_updates_when_counter_never_advances(monkeypatch):
    fake_bar = _FakeBar()
    monkeypatch.setattr(progress, "make_progress_bar", lambda total, desc: fake_bar)

    with progress.ProgressWatcher(lambda: 0, total=3, desc="trials", poll_interval=0.01):
        time.sleep(0.03)

    assert fake_bar.updates == []
    assert fake_bar.closed is True


def test_progress_watcher_propagates_exceptions_from_the_wrapped_block(monkeypatch):
    """An exception raised inside the with-block still stops and closes the watcher cleanly, and propagates as normal."""
    fake_bar = _FakeBar()
    monkeypatch.setattr(progress, "make_progress_bar", lambda total, desc: fake_bar)

    with pytest.raises(RuntimeError, match="boom"):
        with progress.ProgressWatcher(lambda: 0, total=1, desc="trials", poll_interval=0.01):
            raise RuntimeError("boom")

    assert fake_bar.closed is True
