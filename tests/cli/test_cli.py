"""
Tests of the slug executable's own command-line handling (src/main.cpp):
--help/-h and --version/-V print and exit 0, and any other option is an
error rather than the name of an input deck.

Like tests/mpi, these launch the slug executable itself, whose path
CMakeLists.txt's test_CLI passes in as SLUG_EXECUTABLE; without it, they
are skipped.

:copyright: Copyright (c) 2026 Mark Krumholz
"""

import os
import re
import subprocess
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

SLUG_EXECUTABLE = os.environ.get("SLUG_EXECUTABLE")

pytestmark = pytest.mark.skipif(
    not SLUG_EXECUTABLE or not Path(SLUG_EXECUTABLE).is_file(),
    reason="SLUG_EXECUTABLE is not set to the slug executable (see CMakeLists.txt's test_CLI)",
)


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    """
    Run slug with the given arguments.

    Parameters
    ----------
    *args : str
        The command-line arguments.

    Returns
    -------
    subprocess.CompletedProcess
        The completed process, with its stdout and stderr as text.
    """
    assert SLUG_EXECUTABLE is not None
    return subprocess.run([SLUG_EXECUTABLE, *args], capture_output=True, text=True, timeout=60,
                          check=False)


@pytest.mark.parametrize("flag", ["--help", "-h"])
def test_help(flag: str) -> None:
    """--help/-h prints the usage and where the deck is documented, and exits 0."""
    result = _run(flag)
    assert result.returncode == 0
    assert result.stdout.startswith("Usage: slug [--restart|-R] slug.in")
    assert "parameters.html" in result.stdout
    assert result.stderr == ""


@pytest.mark.parametrize("flag", ["--version", "-V"])
def test_version(flag: str) -> None:
    """--version/-V prints pyproject.toml's version and a git hash, and exits 0."""
    with open(REPO_ROOT / "pyproject.toml", "rb") as f:
        version = tomllib.load(f)["project"]["version"]
    result = _run(flag)
    assert result.returncode == 0
    assert re.fullmatch(rf"slug {re.escape(version)} \(git \S+\)\n", result.stdout)


@pytest.mark.parametrize(("args", "message"), [
    (("--bogus",), "unrecognized option '--bogus'"),
    (("-x", "deck.toml"), "unrecognized option '-x'"),
    (("--restart",), "--restart needs an input deck"),
    (("-R",), "-R needs an input deck"),
    ((), ""),
])
def test_bad_arguments(args: tuple[str, ...], message: str) -> None:
    """Unrecognized options, a restart flag without a deck, or no arguments print the usage and exit 1."""
    result = _run(*args)
    assert result.returncode == 1
    assert message in result.stderr
    assert "Usage: slug" in result.stderr
    assert "Failed to parse input file" not in result.stderr
