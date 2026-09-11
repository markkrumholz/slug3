"""
End-to-end regression tests for slug's optional MPI support --
utils::MPIUtils.hpp and the SLUG_MPI-guarded paths in
src/core/SimCluster.cpp/SimGalaxy.cpp and src/io/OutputManagerH5.cpp.

Unlike every other test in this suite (either an in-process gtest
binary, or a pytest module driving the Python bindings/slugpy
directly), these tests actually launch the slug CLI under mpiexec --
the SLUG_MPI-guarded code only exists in the slug target at all (see
CMakeLists.txt's own comment on why MPI is linked only there, not
slugPython or the test suite), and depends on genuinely separate
processes (MPI_Comm_rank/size) to exercise. See CMakeLists.txt's
test_MPI, which only registers this module as a CTest test when the
build actually found and linked MPI (MPI_CXX_FOUND) -- everywhere
else, the skipif below makes it a no-op.
"""

import os
import shutil
import subprocess
from pathlib import Path

import h5py
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

SLUG_EXECUTABLE = os.environ.get("SLUG_EXECUTABLE")
MPIEXEC_EXECUTABLE = os.environ.get("MPIEXEC_EXECUTABLE")
MPIEXEC_NUMPROC_FLAG = os.environ.get("MPIEXEC_NUMPROC_FLAG") or "-n"

pytestmark = pytest.mark.skipif(
    not SLUG_EXECUTABLE
    or not Path(SLUG_EXECUTABLE).is_file()
    or not MPIEXEC_EXECUTABLE
    or not shutil.which(MPIEXEC_EXECUTABLE),
    reason="slug was not built with MPI support, or mpiexec is "
    "unavailable (see CMakeLists.txt's test_MPI, which only registers "
    "this test when MPI_CXX_FOUND)",
)

# A minimal, fast-to-run cluster deck: MIST_test is a small, committed
# track fixture (tests/tracks/assets/, unlike the real, gitignored
# data/tracks/*.h5), and blackbody spectra need no spectral library at
# all -- the same recipe test_Core/test_Io already use to stay a
# "quick" CTest test with no external data dependency. Run from
# REPO_ROOT below, since track_registry is resolved relative to the
# working directory, not this file.
DECK_TEMPLATE = """\
sim_type = "cluster"
n_trial = {n_trial}

[stars]
IMF = "chabrier.toml"
track_registry = "tests/tracks/assets/tracks.toml"
tracks = "MIST_test"
FeH = 0.0
alphaFe = -0.2

[spectra]
model = "blackbody"

[clusters]
CMF = 1e3

[output]
model_name = "{model_name}"
out_dir = "{out_dir}"
output_mode = "{output_mode}"
checkpoint_interval = {checkpoint_interval}
start_time = 5.0
end_time = 5.0
ntime = 1

[nebular]
compute_neb = false
"""


def run_slug(deck_path, nranks, restart=False):
    cmd = [MPIEXEC_EXECUTABLE, MPIEXEC_NUMPROC_FLAG, str(nranks)]
    if restart:
        cmd += [SLUG_EXECUTABLE, "--restart", str(deck_path)]
    else:
        cmd += [SLUG_EXECUTABLE, str(deck_path)]
    result = subprocess.run(
        cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, (
        f"{' '.join(cmd)} exited {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def read_uids_and_trials(h5_path):
    # A rank/thread that happened to be assigned zero trials in its
    # own share of a (possibly small) batch never calls writeCluster(),
    # so "clusters" is never even created as a group in its own file --
    # not a bug, just an empty contribution to this test's own totals.
    with h5py.File(h5_path, "r") as f:
        if "clusters/uid" not in f:
            return [], []
        return list(f["clusters/uid"][:]), list(f["clusters/trial"][:])


def test_mpi_basic_run_has_unique_uids_and_trials(tmp_path):
    """
    A single, non-restarted multi-rank run must produce exactly
    n_trial clusters, each with a globally unique uid (see
    utils::mpiUidStride's per-rank striping) and a distinct trial
    number covering 0..n_trial-1. This is the direct regression test
    for the "trials_completed counted per rank instead of once
    globally on normal completion" bug that SimCluster::run()'s/
    SimGalaxy::run()'s own unconditional notifyEarlyTermination() call
    fixed.
    """
    n_trial = 24
    deck = tmp_path / "basic.in"
    deck.write_text(DECK_TEMPLATE.format(
        n_trial=n_trial, model_name="mpi_basic", out_dir=tmp_path,
        output_mode="h5", checkpoint_interval=0,
    ))
    run_slug(deck, nranks=4)

    out_file = tmp_path / "mpi_basic.h5"
    assert out_file.is_file()
    uids, trials = read_uids_and_trials(out_file)
    assert len(uids) == n_trial
    assert len(set(uids)) == n_trial
    assert sorted(trials) == list(range(n_trial))


def test_mpi_restart_with_different_rank_count(tmp_path):
    """
    Restarting with a different number of ranks than the original run
    must not collide uids or drop/duplicate trials. This is the direct
    regression test for OutputManagerH5::findRestartUidForRank(),
    which looks up each rank's own most recent checkpoint by rank
    number across every past checkpoint (not just the last one) so
    that a rank added between runs starts fresh from its own striped
    offset instead of colliding with an existing rank's IDs -- the bug
    that produced 6 duplicate uids out of 30 before this lookup
    existed. Uses output_mode = "h5divided" specifically because it is
    the one mode that never consolidates per-rank files away (see
    OutputManagerH5::~OutputManagerH5()'s own comment), so every
    rank's own file from both runs is still on disk afterward to
    check directly.
    """
    out_dir = tmp_path
    model_name = "mpi_restart"

    first_n_trial = 12
    deck1 = tmp_path / "restart1.in"
    deck1.write_text(DECK_TEMPLATE.format(
        n_trial=first_n_trial, model_name=model_name, out_dir=out_dir,
        output_mode="h5divided", checkpoint_interval=3,
    ))
    run_slug(deck1, nranks=2)

    total_n_trial = 24
    deck2 = tmp_path / "restart2.in"
    deck2.write_text(DECK_TEMPLATE.format(
        n_trial=total_n_trial, model_name=model_name, out_dir=out_dir,
        output_mode="h5divided", checkpoint_interval=3,
    ))
    run_slug(deck2, nranks=4, restart=True)

    all_uids = []
    all_trials = []
    for h5_path in sorted(out_dir.glob(f"{model_name}_chk*/rank_*.h5")):
        uids, trials = read_uids_and_trials(h5_path)
        all_uids.extend(uids)
        all_trials.extend(trials)

    assert len(all_uids) == total_n_trial
    assert len(set(all_uids)) == total_n_trial
    assert sorted(all_trials) == list(range(total_n_trial))
