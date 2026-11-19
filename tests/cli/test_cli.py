"""
Tests of the slug executable's own command-line handling (src/main.cpp):
--help/-h and --version/-V print and exit 0, any other option is an
error rather than the name of an input deck, and --restart resumes a
checkpointed run the way a batch job invokes it.

Like tests/mpi, these launch the slug executable itself, whose path
CMakeLists.txt's test_CLI passes in as SLUG_EXECUTABLE; without it, they
are skipped, and with a path that is not the executable, they fail.

:copyright: Copyright (c) 2026 Mark Krumholz
"""

import os
import re
import subprocess
import tomllib
from pathlib import Path

import h5py
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

SLUG_EXECUTABLE = os.environ.get("SLUG_EXECUTABLE")

# Skipped only when SLUG_EXECUTABLE is unset (e.g. a direct pytest run);
# if it is set but wrong, the tests fail rather than quietly skip
pytestmark = pytest.mark.skipif(
    not SLUG_EXECUTABLE,
    reason="SLUG_EXECUTABLE is not set to the slug executable (see CMakeLists.txt's test_CLI)",
)


def _run(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    """
    Run slug with the given arguments.

    Parameters
    ----------
    *args : str
        The command-line arguments.
    cwd : Path or None, optional
        Working directory to run slug in; the current one if None.

    Returns
    -------
    subprocess.CompletedProcess
        The completed process, with its stdout and stderr as text.
    """
    assert SLUG_EXECUTABLE is not None
    return subprocess.run([SLUG_EXECUTABLE, *args], capture_output=True, text=True, timeout=60,
                          check=False, cwd=cwd)


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
    # CMakeLists.txt embeds only the version's leading X.Y.Z (e.g. 3.1.0
    # for 3.1.0.dev1), so compare with that
    with open(REPO_ROOT / "pyproject.toml", "rb") as f:
        full_version = tomllib.load(f)["project"]["version"]
    match = re.match(r"\d+\.\d+\.\d+", full_version)
    assert match is not None
    version = match.group(0)
    result = _run(flag)
    assert result.returncode == 0
    assert re.fullmatch(rf"slug {re.escape(version)} \(git \S+\)\n", result.stdout)


@pytest.mark.parametrize(("args", "message"), [
    (("--bogus",), "unrecognized option '--bogus'"),
    (("-x", "deck.toml"), "unrecognized option '-x'"),
    (("--restart",), "--restart needs an input deck"),
    (("-R",), "-R needs an input deck"),
    (("-R", "--help"), "-R needs an input deck, not '--help'"),
    (("--restart", "-x"), "--restart needs an input deck, not '-x'"),
    ((), ""),
])
def test_bad_arguments(args: tuple[str, ...], message: str) -> None:
    """Unrecognized options, a restart flag without a deck, or no arguments print the usage and exit 1."""
    result = _run(*args)
    assert result.returncode == 1
    assert message in result.stderr
    assert "Usage: slug" in result.stderr
    assert "Failed to parse input file" not in result.stderr


def _restart_deck(n_trial: int) -> str:
    """
    A small checkpointed cluster deck with no out_dir, so output goes to the working directory.

    Parameters
    ----------
    n_trial : int
        Number of trials.

    Returns
    -------
    str
        The deck's TOML text. Its track registry is an absolute path,
        since the deck is run from outside the repository.
    """
    registry = REPO_ROOT / "tests" / "tracks" / "assets" / "tracks.toml"
    return f"""\
sim_type = "cluster"
n_trial = {n_trial}

[stars]
IMF = "chabrier.toml"
track_registry = "{registry}"
tracks = "MIST_test"
FeH = 0.0
alphaFe = -0.2

[spectra]
model = "blackbody"

[clusters]
CMF = 1e3

[output]
model_name = "r"
checkpoint_interval = 2
start_time = 5.0
end_time = 5.0
ntime = 1

[nebular]
compute_neb = false
"""


def test_restart_with_default_out_dir(tmp_path: Path) -> None:
    """
    --restart resumes from the working directory's checkpoints when out_dir is unset.

    out_dir defaults to the working directory, which is where a batch job
    leaves a run's checkpoints. --restart used to look for them in the
    directory named by out_dir itself, which for the default "" does not
    exist, so it found none, started over from trial 0, and failed because
    the first checkpoint it then tried to write already existed.
    """
    deck = tmp_path / "deck.toml"
    deck.write_text(_restart_deck(n_trial=4))
    first = _run(str(deck), cwd=tmp_path)
    assert first.returncode == 0, first.stderr

    deck.write_text(_restart_deck(n_trial=6))
    restarted = _run("--restart", str(deck), cwd=tmp_path)
    assert restarted.returncode == 0, restarted.stderr
    assert "no checkpoint" not in restarted.stdout + restarted.stderr

    trials: list[int] = []
    for path in sorted(tmp_path.glob("r_chk*.h5")):
        with h5py.File(path, "r") as f:
            trials.extend(int(t) for t in f["clusters/trial"][:])
    assert sorted(trials) == list(range(6))
