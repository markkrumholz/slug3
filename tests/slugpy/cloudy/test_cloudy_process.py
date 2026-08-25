"""
Unit tests for slugpy.cloudy.cloudy_process: find_cloudy_executable,
run_cloudy_deck, and run_cloudy_decks.

Uses a tiny hand-written shell script as a stand-in for the real
cloudy executable (which takes minutes per run and isn't available in
CI), so the process-launching machinery itself -- executable lookup,
stdin/stdout redirection, working directory, and concurrency -- can be
exercised quickly and deterministically.
"""

import concurrent.futures
import os
import time
from pathlib import Path

import pytest

import slugpy.cloudy.cloudy_process as cloudy_process_module
from slugpy.cloudy.cloudy_process import (
    find_cloudy_executable,
    run_cloudy_deck,
    run_cloudy_decks,
)


@pytest.fixture
def fake_cloudy_exe(tmp_path) -> Path:
    """A stand-in for cloudy.exe: prints a marker and its own cwd, sleeps if told to, then copies stdin to stdout."""
    script = tmp_path / "cloudy.exe"
    script.write_text(
        '#!/bin/sh\n'
        'echo "FAKE_CLOUDY_RAN in $(pwd)"\n'
        'sleep "${FAKE_CLOUDY_SLEEP:-0}"\n'
        'cat\n'
    )
    script.chmod(0o755)
    return script


# ---------------------------------------------------------------------
# find_cloudy_executable
# ---------------------------------------------------------------------

def test_explicit_path_used(fake_cloudy_exe):
    """An explicit cloudy_path is returned (resolved to an absolute path), regardless of CLOUDY_DIR."""
    result = find_cloudy_executable(fake_cloudy_exe)
    assert result == fake_cloudy_exe.resolve()


def test_cloudy_dir_env_used(fake_cloudy_exe, monkeypatch):
    """With no explicit path, $CLOUDY_DIR/cloudy.exe is used."""
    monkeypatch.setenv("CLOUDY_DIR", str(fake_cloudy_exe.parent))
    result = find_cloudy_executable()
    assert result == fake_cloudy_exe.resolve()


def test_neither_given_raises(monkeypatch):
    """No cloudy_path and no CLOUDY_DIR raises RuntimeError."""
    monkeypatch.delenv("CLOUDY_DIR", raising=False)
    with pytest.raises(RuntimeError, match="CLOUDY_DIR"):
        find_cloudy_executable()


def test_nonexistent_path_raises(tmp_path):
    """A cloudy_path that doesn't exist raises RuntimeError."""
    with pytest.raises(RuntimeError, match="no executable cloudy binary"):
        find_cloudy_executable(tmp_path / "does_not_exist.exe")


def test_non_executable_file_raises(tmp_path):
    """A cloudy_path that exists but isn't executable raises RuntimeError."""
    not_exe = tmp_path / "cloudy.exe"
    not_exe.write_text("not actually executable\n")
    not_exe.chmod(0o644)
    with pytest.raises(RuntimeError, match="no executable cloudy binary"):
        find_cloudy_executable(not_exe)


def test_cloudy_dir_missing_binary_raises(tmp_path, monkeypatch):
    """CLOUDY_DIR pointing at a directory with no cloudy.exe in it raises RuntimeError."""
    monkeypatch.setenv("CLOUDY_DIR", str(tmp_path))
    with pytest.raises(RuntimeError, match="no executable cloudy binary"):
        find_cloudy_executable()


# ---------------------------------------------------------------------
# run_cloudy_deck
# ---------------------------------------------------------------------

def test_run_cloudy_deck_captures_stdout(fake_cloudy_exe, tmp_path):
    """The fake cloudy's own stdout is captured into a sibling .out file."""
    deck = tmp_path / "mydeck.in"
    deck.write_text("hden 2.0\n")
    out_path = run_cloudy_deck(deck, fake_cloudy_exe)
    assert out_path == deck.with_suffix(".out")
    assert "FAKE_CLOUDY_RAN" in out_path.read_text()


def test_run_cloudy_deck_pipes_stdin(fake_cloudy_exe, tmp_path):
    """The deck's own content is piped in on stdin, and the fake cloudy echoes it back after the marker."""
    deck = tmp_path / "mydeck.in"
    deck.write_text("hden 2.0\nradius 18.0\n")
    out_path = run_cloudy_deck(deck, fake_cloudy_exe)
    assert deck.read_text() in out_path.read_text()


def test_run_cloudy_deck_uses_decks_own_directory_as_cwd(fake_cloudy_exe, tmp_path):
    """cloudy is launched with the deck's own parent directory as its cwd, so relative $OUTPUT_FILENAME-style outputs land next to the deck."""
    subdir = tmp_path / "nested"
    subdir.mkdir()
    deck = subdir / "mydeck.in"
    deck.write_text("hden 2.0\n")
    out_path = run_cloudy_deck(deck, fake_cloudy_exe)
    assert f"FAKE_CLOUDY_RAN in {subdir.resolve()}" in out_path.read_text()


# ---------------------------------------------------------------------
# run_cloudy_decks
# ---------------------------------------------------------------------

def test_run_cloudy_decks_runs_all_and_preserves_order(fake_cloudy_exe, tmp_path):
    """Every deck gets run, and the returned .out paths are in the same order as the inputs."""
    decks = []
    for i in range(4):
        deck = tmp_path / f"deck{i}.in"
        deck.write_text(f"hden {i}.0\n")
        decks.append(deck)

    out_paths = run_cloudy_decks(decks, fake_cloudy_exe, max_workers=2, progress=False)
    assert out_paths == [d.with_suffix(".out") for d in decks]
    for out_path in out_paths:
        assert "FAKE_CLOUDY_RAN" in out_path.read_text()


def test_run_cloudy_decks_defaults_max_workers_to_cpu_count(fake_cloudy_exe, tmp_path, monkeypatch):
    """With max_workers omitted, os.cpu_count() worth of concurrency is available (checked indirectly via timing below); here just confirm the default doesn't error with cpu_count patched to a known value."""
    monkeypatch.setattr(os, "cpu_count", lambda: 2)
    decks = []
    for i in range(3):
        deck = tmp_path / f"deck{i}.in"
        deck.write_text(f"hden {i}.0\n")
        decks.append(deck)
    out_paths = run_cloudy_decks(decks, fake_cloudy_exe, progress=False)
    assert len(out_paths) == 3


def test_run_cloudy_decks_runs_concurrently(fake_cloudy_exe, tmp_path, monkeypatch):
    """max_workers >= number of decks runs them essentially in parallel, not sequentially -- wall-clock time should be much closer to one sleep than to n_decks sleeps."""
    monkeypatch.setenv("FAKE_CLOUDY_SLEEP", "0.5")
    decks = []
    for i in range(4):
        deck = tmp_path / f"deck{i}.in"
        deck.write_text(f"hden {i}.0\n")
        decks.append(deck)

    start = time.perf_counter()
    run_cloudy_decks(decks, fake_cloudy_exe, max_workers=4, progress=False)
    elapsed = time.perf_counter() - start

    # Sequential execution would take >= 4 * 0.5 = 2.0s; concurrent
    # execution should take well under that (allow generous slack for
    # process-launch overhead and a loaded CI machine)
    assert elapsed < 1.5


# ---------------------------------------------------------------------
# executor
# ---------------------------------------------------------------------

def test_given_executor_is_used_and_left_running(fake_cloudy_exe, tmp_path):
    """A caller-supplied executor is used instead of a private pool, and is left running afterward, not shut down."""
    deck = tmp_path / "deck0.in"
    deck.write_text("hden 0.0\n")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        out_paths = run_cloudy_decks([deck], fake_cloudy_exe, max_workers=99, progress=False, executor=executor)
        assert out_paths == [deck.with_suffix(".out")]
        assert "FAKE_CLOUDY_RAN" in out_paths[0].read_text()
        # Still usable: submitting more work after run_cloudy_decks
        # returns doesn't raise (a shut-down executor raises
        # RuntimeError on submit), confirming run_cloudy_decks did not
        # call executor.shutdown() -- max_workers=99 above is also
        # ignored without error, since executor's own size governs
        # concurrency once given.
        assert executor.submit(lambda: 42).result() == 42


def test_shared_executor_bounds_concurrency_across_calls(fake_cloudy_exe, tmp_path, monkeypatch):
    """Two run_cloudy_decks calls sharing one executor are bounded by that executor's own size together, not each getting an independent max_workers-worth of concurrency of its own."""
    monkeypatch.setenv("FAKE_CLOUDY_SLEEP", "0.3")

    def _make_decks(prefix, n):
        decks = []
        for i in range(n):
            deck = tmp_path / f"{prefix}{i}.in"
            deck.write_text(f"hden {i}.0\n")
            decks.append(deck)
        return decks

    decks_a = _make_decks("a", 4)
    decks_b = _make_decks("b", 4)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as shared:
        start = time.perf_counter()
        # Two "driver" threads, mirroring how run_cloudy_grid.py calls
        # slug_reader.run_cloudy for different files concurrently,
        # each passing the same shared executor through.
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as drivers:
            fut_a = drivers.submit(run_cloudy_decks, decks_a, fake_cloudy_exe, progress=False, executor=shared)
            fut_b = drivers.submit(run_cloudy_decks, decks_b, fake_cloudy_exe, progress=False, executor=shared)
            results_a = fut_a.result()
            results_b = fut_b.result()
        elapsed = time.perf_counter() - start

    assert len(results_a) == 4
    assert len(results_b) == 4

    # 8 decks total sharing 2 workers, 0.3s each, run in 4 sequential
    # rounds -> ~1.2s. If each call instead got its own private
    # 2-worker pool (the behavior this executor argument replaces),
    # both calls' own 4 decks would run in 2 rounds each -- ~0.6s --
    # fully in parallel with each other, since they'd be on
    # independent pools. The threshold below sits well above that
    # 0.6s alternative and comfortably below the ~1.2s ideal, so this
    # only passes if concurrency is really bounded by the one shared
    # pool.
    assert elapsed > 0.9


# ---------------------------------------------------------------------
# progress
# ---------------------------------------------------------------------

def test_progress_true_shows_a_progress_bar(fake_cloudy_exe, tmp_path, monkeypatch):
    """progress=True (the default) drives a progress bar, updated once per completed deck."""
    calls = []

    class _FakeBar:
        def __init__(self):
            self.updates = []

        def update(self, n=1):
            self.updates.append(n)

        def close(self):
            calls.append("closed")

    def _fake_make_progress_bar(total, desc):
        calls.append(("made", total, desc))
        return _FakeBar()

    monkeypatch.setattr(cloudy_process_module, "make_progress_bar", _fake_make_progress_bar)

    decks = []
    for i in range(3):
        deck = tmp_path / f"deck{i}.in"
        deck.write_text(f"hden {i}.0\n")
        decks.append(deck)
    run_cloudy_decks(decks, fake_cloudy_exe, max_workers=2, progress=True)

    assert calls[0] == ("made", 3, "Running cloudy")
    assert calls[-1] == "closed"


def test_progress_false_never_builds_a_progress_bar(fake_cloudy_exe, tmp_path, monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("make_progress_bar should not be called when progress=False")

    monkeypatch.setattr(cloudy_process_module, "make_progress_bar", _boom)

    deck = tmp_path / "deck0.in"
    deck.write_text("hden 0.0\n")
    out_paths = run_cloudy_decks([deck], fake_cloudy_exe, progress=False)
    assert len(out_paths) == 1


def test_progress_is_a_noop_for_empty_input(fake_cloudy_exe, monkeypatch):
    """progress=True with no decks to run never builds a progress bar (nothing to track)."""
    def _boom(*args, **kwargs):
        raise AssertionError("make_progress_bar should not be called for an empty deck list")

    monkeypatch.setattr(cloudy_process_module, "make_progress_bar", _boom)
    assert run_cloudy_decks([], fake_cloudy_exe, progress=True) == []
