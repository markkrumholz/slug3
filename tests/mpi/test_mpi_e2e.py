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
build actually found and linked MPI (MPI_C_FOUND) -- everywhere
else, the skipif below makes it a no-op.
"""

import os
import shutil
import signal
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
    "this test when MPI_C_FOUND)",
)


def _oversubscribe_flag():
    # A CI runner's own core count can be smaller than the rank counts
    # this module deliberately exercises (e.g. 4), and Open MPI's
    # mpiexec refuses to run more ranks than it thinks there are cores
    # ("not enough slots") unless told otherwise -- --oversubscribe is
    # exactly that override. Only add it for Open MPI specifically:
    # the flag doesn't exist for MPICH (whose mpiexec doesn't impose
    # this limit to begin with, so nothing needs overriding there),
    # and passing an unrecognized flag would just be a new failure
    # mode of its own.
    if not MPIEXEC_EXECUTABLE:
        return []
    try:
        version = subprocess.run(
            [MPIEXEC_EXECUTABLE, "--version"], capture_output=True, text=True, timeout=10,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    return ["--oversubscribe"] if "Open MPI" in version or "OpenRTE" in version else []


_OVERSUBSCRIBE_FLAG = _oversubscribe_flag()

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
    cmd = [MPIEXEC_EXECUTABLE, MPIEXEC_NUMPROC_FLAG, str(nranks)] + _OVERSUBSCRIBE_FLAG
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
    return result


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


@pytest.mark.parametrize("nranks", [2, 4])
def test_mpi_run_wide_messages_are_printed_once(tmp_path, nranks):
    """
    Messages that are identical on every rank -- warnings about the
    input deck, and the run's start and completion lines -- must reach
    stdout exactly once however many ranks there are, not once per
    rank (see utils::isIORank()). A stray key under [nebular] gives a
    deck warning that every rank would otherwise repeat.
    """
    deck = tmp_path / "once.in"
    deck.write_text("verbosity = 1\n" + DECK_TEMPLATE.format(
        n_trial=8, model_name="mpi_once", out_dir=tmp_path,
        output_mode="h5", checkpoint_interval=0,
    ) + "stray_key = 1\n")
    result = run_slug(deck, nranks=nranks)

    assert result.stdout.count("input deck key 'nebular.stray_key'") == 1, result.stdout
    assert result.stdout.count("slug: cluster simulation starting") == 1, result.stdout
    assert result.stdout.count("slug: simulation complete") == 1, result.stdout


# How long a run that is meant to fail may take before the test treats
# it as hung. Each of the runs below takes about a second when it
# shuts down correctly; a deadlocked run never finishes.
FAILURE_TIMEOUT = 60


def run_slug_expecting_failure(deck_path, nranks, extra_env):
    """
    Run slug under mpiexec with extra environment variables set (the
    test-only SLUG_TEST_FAIL_* hooks; see SimCluster's own
    testFailTrial_/testFailRank_ members), and require it to exit
    promptly with a nonzero status. A run that never exits is the
    deadlock these tests guard against, so a timeout fails the test
    rather than erroring out.
    """
    cmd = ([MPIEXEC_EXECUTABLE, MPIEXEC_NUMPROC_FLAG, str(nranks)] + _OVERSUBSCRIBE_FLAG +
           [SLUG_EXECUTABLE, str(deck_path)])
    # Start from neither hook set, whatever the parent environment holds,
    # so each test gets only the failure it asks for
    env = {k: v for k, v in os.environ.items() if not k.startswith("SLUG_TEST_FAIL_")}
    env.update(extra_env)
    # Started in its own session so that, on a timeout, every process
    # left in it can be killed at once. Deadlocked ranks would otherwise
    # outlive the test, busy-waiting in MPI. SIGKILL to mpiexec alone is
    # not enough: Open MPI's launcher puts the ranks in a process group
    # of their own, so SIGTERM mpiexec first, which forwards the signal
    # and then SIGKILLs every rank still running.
    proc = subprocess.Popen(
        cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env=env, start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=FAILURE_TIMEOUT)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            stdout, stderr = proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass  # the launcher's group is already gone; just report below
            try:
                stdout, stderr = proc.communicate(timeout=30)
            except subprocess.TimeoutExpired:
                stdout = stderr = "(output unavailable: pipes held by surviving processes)"
        pytest.fail(
            f"{' '.join(cmd)} with {extra_env} did not exit within "
            f"{FAILURE_TIMEOUT} s (deadlocked?)\nstdout:\n{stdout}\nstderr:\n{stderr}"
        )
    assert proc.returncode != 0, (
        f"{' '.join(cmd)} with {extra_env} exited 0, but was expected to fail\n"
        f"stdout:\n{stdout}\nstderr:\n{stderr}"
    )
    return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)


# Trial 2 runs on rank 0 in the first batch; trial 17 runs on rank 2
# in the second, after a checkpoint has already been written. With 4
# ranks, each batch of 10 splits as 3, 3, 2, 2 trials (e.g. [10, 20) as
# 10-12, 13-15, 16-17, 18-19), and a failing rank completes only the
# trials before the failing one, so the failed batch's checkpoint must
# record 2 + 3 + 2 + 2 = 9 trials completed in total for trial 2, and
# 10 + (3 + 3 + 1 + 2) = 19 for trial 17.
@pytest.mark.parametrize(("fail_trial", "expected_trials"), [(2, 9), (17, 19)])
def test_mpi_trial_error_stops_every_rank_cleanly(tmp_path, fail_trial, expected_trials):
    """
    A trial that throws on one rank must end the whole run promptly,
    with a nonzero exit status and the error reported, rather than
    deadlocking it. This is the regression test for a production run
    that hung for hours: the failing rank rethrew its trial's
    exception straight away, unwinding into OutputManagerH5's
    destructor and its MPI_Barrier, while every other rank waited
    forever in the end-of-batch MPI_Allreduce. Now every rank agrees,
    at the end of the batch, to stop on the error (see
    utils::mpiAgreeBatchStop()/rethrowAgreedError()), so every rank
    shuts down in lockstep without MPI_Abort: every output file is
    closed, and so still readable, and the failed batch's checkpoint has
    a file from every rank, each carrying the run-wide trials_completed
    that only the cross-rank checkpoint sync writes.
    """
    deck = tmp_path / "trial_error.in"
    deck.write_text(DECK_TEMPLATE.format(
        n_trial=40, model_name="mpi_trial_error", out_dir=tmp_path,
        output_mode="h5divided", checkpoint_interval=10,
    ))
    result = run_slug_expecting_failure(
        deck, nranks=4, extra_env={"SLUG_TEST_FAIL_TRIAL": str(fail_trial)})

    assert f"slug: trial {fail_trial} failed: injected test failure" in result.stderr, result.stderr
    assert "slug: simulation failed" in result.stderr, result.stderr
    # A clean, agreed shutdown, not the MPI_Abort backstop
    assert "MPI_ABORT" not in result.stdout + result.stderr, result.stdout + result.stderr

    files = sorted(tmp_path.glob("mpi_trial_error_chk*/rank_*.h5"))
    assert files, "no per-rank output files were written"
    for h5_path in files:
        with h5py.File(h5_path, "r") as f:
            assert "trials_completed" in f.attrs, h5_path

    # The failed batch is the last checkpoint written. Its files are
    # rank_NNNN.h5 without OpenMP, rank_NNNN_thread_MMMM.h5 with it.
    failed_checkpoint = max(path.parent for path in files)
    failed_files = sorted(failed_checkpoint.glob("rank_*.h5"))
    ranks = {int(path.name.split("_")[1].split(".")[0]) for path in failed_files}
    assert ranks == {0, 1, 2, 3}, failed_files
    for h5_path in failed_files:
        with h5py.File(h5_path, "r") as f:
            assert f.attrs["trials_completed"] == expected_trials, h5_path


@pytest.mark.parametrize("fail_rank", [0, 1])
def test_mpi_unagreed_error_aborts_every_rank(tmp_path, fail_rank):
    """
    An error that only one rank knows about, thrown outside the
    end-of-batch agreement (injected via SLUG_TEST_FAIL_RANK, after
    that agreement and before the checkpoint rollover every other rank
    goes on to), must still end the whole run promptly with a nonzero
    exit status and the error reported. The failing rank must not
    attempt OutputManagerH5's collective shutdown (the other ranks
    would never match it -- see utils::unwindingFromLocalError()), and
    main() must then abort every rank (utils::mpiAbortIfMultiRank()).
    """
    deck = tmp_path / "rank_error.in"
    deck.write_text(DECK_TEMPLATE.format(
        n_trial=40, model_name="mpi_rank_error", out_dir=tmp_path,
        output_mode="h5", checkpoint_interval=10,
    ))
    result = run_slug_expecting_failure(
        deck, nranks=4, extra_env={"SLUG_TEST_FAIL_RANK": str(fail_rank)})

    assert (f"slug: simulation failed on MPI rank {fail_rank}: injected test failure"
            in result.stderr), result.stderr
