"""
run_sim.py

Implements run_sim, a convenience function that runs an entire slug
simulation end to end from Python, exactly as the slug command-line
executable does.

:copyright: Copyright (c) 2026 Mark Krumholz
"""

import gc
import pathlib
import warnings

from ._slug import (
    OutputManagerAscii,
    OutputManagerH5,
    SimCluster,
    SimControls,
    SimGalaxy,
)
from .progress import run_with_progress
from .read import read
from .slug_reader import slug_reader


def _run(sim: SimCluster | SimGalaxy, sim_controls: SimControls, progress: bool) -> int:
    """
    Run sim (a SimCluster or SimGalaxy), optionally tracking
    sim.trialsCompleted() against sim_controls.nTrial() via a progress
    bar.

    Returns
    -------
    int
        Whatever sim.run() itself returned: 0 if every trial completed
        normally, or its own sigtermExitCode (143) if a SIGTERM was
        caught and it stopped early instead (see SimCluster.run/
        SimGalaxy.run's own docstrings).
    """
    if progress:
        return run_with_progress(sim.run, sim.trialsCompleted, sim_controls.nTrial(), "Running trials")
    return sim.run()


def run_sim(controls: SimControls | str, progress: bool = True, restart: bool = False) -> slug_reader | None:
    """
    Run a slug simulation end to end, exactly as the slug command-line
    executable does.

    Parameters
    ----------
    controls : SimControls or str
        Either an already-built SimControls object, used directly, or
        the text of a slug TOML input deck, or a path to one on disk
        (see SimControls's own constructor for the exact string-
        parsing rule), used to construct one. Passing a SimControls
        object directly lets a simulation be driven entirely from
        Python -- via SimControls's own constructor keyword arguments
        and setters -- without ever writing an input deck to text.
    progress : bool, default True
        Whether to show a progress bar (see slugpy.progress) tracking
        trials completed while the simulation runs.
    restart : bool, default False
        Whether this run should resume a previous, interrupted run of
        the same model_name/out_dir from its most recent checkpoint,
        rather than starting a new one from trial 0 (see
        OutputManagerH5's own docstring for the full detail, and
        SimCluster.run/SimGalaxy.run's own docstrings for how the
        starting trial is chosen). Only valid with HDF5 output and a
        non-zero output.checkpoint_interval in controls -- passing
        True with ASCII output raises RuntimeError, mirroring the
        slug command-line executable's own --restart/-R check.

    Returns
    -------
    slug_reader or None
        A lazy reader for the HDF5 output file this simulation
        produced, or None if this simulation's own output mode is
        ASCII rather than HDF5 -- ASCII output cannot currently be
        read back this way (a UserWarning is issued in that case; see
        Details below).

    Details
    -------
    Mirrors main.cpp's own control flow: obtains a SimControls (either
    controls itself, or one built from it), an OutputManagerH5 or
    OutputManagerAscii from it (matching sim_controls.outputMode()),
    and -- depending on sim_controls.simType() -- a SimCluster or
    SimGalaxy, which it then runs.

    The progress bar (when enabled) is driven by run_with_progress:
    run() itself executes on a background thread, while this calling
    thread polls the running SimCluster/SimGalaxy's own
    trialsCompleted() against nTrial() every 0.2s (see
    run_with_progress's own docstring for why the polling has to
    happen on this thread rather than a separate watcher thread). This
    is safe because run() releases the GIL for its own entire duration
    (see SimCluster.run/SimGalaxy.run's own pybind docstrings), so
    polling this thread is never blocked by it, and the OpenMP worker
    threads run() uses internally never touch Python or the GIL at
    all. Note that when restart is True, trialsCompleted() only counts
    trials run by this call, not ones a previous, resumed run already
    completed -- so the progress bar's own denominator (nTrial()) can
    look larger than what this call alone will actually run.

    ASCII output is meant for small, human-readable output rather than
    batch processing, so there is no ASCII counterpart to slug_reader
    (yet -- this may change in the future). A deck requesting it still
    runs to completion, and its output files are still written, but
    this function has nothing to return for them, so it warns and
    returns None rather than trying to read back a file that was never
    written.
    """
    if isinstance(controls, SimControls):
        sim_controls = controls
    else:
        sim_controls = SimControls(controls)

    if restart and sim_controls.outputMode() == SimControls.OutputMode.ascii:
        raise RuntimeError(
            "run_sim: restart is True, but sim_controls.outputMode() is "
            "ascii -- restarting is only supported with HDF5 output")

    if sim_controls.outputMode() in (SimControls.OutputMode.h5, SimControls.OutputMode.h5divided):
        output_manager = OutputManagerH5(sim_controls, restart)
    else:
        output_manager = OutputManagerAscii(sim_controls)

    if sim_controls.simType() == SimControls.SimType.cluster:
        sim = SimCluster(sim_controls, output_manager, restart)
        exit_code = _run(sim, sim_controls, progress)
        del sim
    elif sim_controls.simType() == SimControls.SimType.galaxy:
        sim = SimGalaxy(sim_controls, output_manager, restart)
        exit_code = _run(sim, sim_controls, progress)
        del sim
    else:
        # Unreachable for a SimControls built from a real input deck --
        # sim_type is a required key, so simType() can only ever be
        # SimType.none (a dummy, pre-parse default) here if SimControls
        # was somehow constructed without going through that parsing at
        # all. Raised explicitly rather than left for exit_code to be
        # silently unbound below.
        raise RuntimeError(
            f"run_sim: sim_controls.simType() is {sim_controls.simType()!r}, "
            "which is neither cluster nor galaxy")

    # exit_code is sim.run()'s own return value: 0 if every trial
    # completed normally, or its sigtermExitCode (143) if a SIGTERM was
    # caught partway through (see SimCluster.run/SimGalaxy.run's own
    # docstrings) -- not exposed to Python as a named constant, so
    # compared against 0 directly rather than by name
    if exit_code != 0:
        warnings.warn(
            f"run_sim: the simulation was interrupted (exit code {exit_code}) "
            "before every trial completed, most likely by a caught SIGTERM -- "
            "the output written so far is still valid, and this run can be "
            "resumed later by calling run_sim again with restart=True.",
            stacklevel=2)

    # Drop the only Python-side reference to output_manager (whether or
    # not it was actually consumed by a SimCluster/SimGalaxy above) so
    # its destructor -- which closes the underlying output file(s) --
    # runs now, before the read() below tries to reopen the same file
    # for reading; HDF5 otherwise refuses a second open while the
    # first (write) handle is still live.
    del output_manager
    gc.collect()

    if sim_controls.outputMode() == SimControls.OutputMode.ascii:
        warnings.warn(
            "run_sim: ASCII output cannot be returned as a slug_reader; "
            "returning None. Use HDF5 output (the default) to get a "
            "slug_reader back.",
            stacklevel=2)
        return None

    # An extension-less model-name path, not a hardcoded modelName + ".h5"
    # -- slug_reader's own _find_output_files() resolves this to
    # whichever shape the run's output actually ended up in (a single
    # consolidated file, per-checkpoint files, or h5divided's own
    # per-thread files), which a hardcoded ".h5" path would get wrong
    # whenever checkpointing was enabled.
    output_path = pathlib.Path(sim_controls.outDir()) / sim_controls.modelName()
    return read(str(output_path))
