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

from ._slug import OutputManagerAscii, OutputManagerH5, SimCluster, SimControls, SimGalaxy
from .progress import run_with_progress
from .read import read
from .slug_reader import slug_reader


def _run(sim: SimCluster | SimGalaxy, sim_controls: SimControls, progress: bool) -> None:
    """Run sim (a SimCluster or SimGalaxy), optionally tracking sim.trialsCompleted() against sim_controls.nTrial() via a progress bar."""
    if progress:
        run_with_progress(sim.run, sim.trialsCompleted, sim_controls.nTrial(), "Running trials")
    else:
        sim.run()


def run_sim(input_deck: str, progress: bool = True, restart: bool = False) -> slug_reader | None:
    """
    Run a slug simulation end to end, exactly as the slug command-line
    executable does.

    Parameters
    ----------
    input_deck : str
        Either the text of a slug TOML input deck, or a path to one on
        disk (see SimControls's own constructor for the exact rule).
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
        non-zero outputs.checkpoint_interval in input_deck -- passing
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
    Mirrors main.cpp's own control flow: builds a SimControls from
    input_deck, an OutputManagerH5 or OutputManagerAscii from it
    (matching sim_controls.outputMode()), and -- depending on
    sim_controls.simType() -- a SimCluster or SimGalaxy, which it then
    runs.

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
    sim_controls = SimControls(input_deck)

    if restart and sim_controls.outputMode() == SimControls.OutputMode.ascii:
        raise RuntimeError(
            "run_sim: restart is True, but sim_controls.outputMode() is "
            "ascii -- restarting is only supported with HDF5 output")

    if sim_controls.outputMode() in (SimControls.OutputMode.h5, SimControls.OutputMode.h5divided):
        output_manager = OutputManagerH5(sim_controls, input_deck, restart)
    else:
        output_manager = OutputManagerAscii(sim_controls, input_deck)

    if sim_controls.simType() == SimControls.SimType.cluster:
        sim = SimCluster(sim_controls, output_manager, restart)
        _run(sim, sim_controls, progress)
        del sim
    elif sim_controls.simType() == SimControls.SimType.galaxy:
        sim = SimGalaxy(sim_controls, output_manager, restart)
        _run(sim, sim_controls, progress)
        del sim

    # Drop the only Python-side reference to output_manager (whether or
    # not it was actually consumed by a SimCluster/SimGalaxy above) so
    # its destructor -- which closes the underlying output file(s) --
    # runs now, before the read() below tries to reopen the same file
    # for reading; HDF5 otherwise refuses a second open while the
    # first (write) handle is still live.
    del output_manager
    gc.collect()

    if sim_controls.outputMode() != SimControls.OutputMode.h5:
        warnings.warn(
            "run_sim: ASCII output cannot be returned as a slug_reader; "
            "returning None. Use HDF5 output (the default) to get a "
            "slug_reader back.",
            stacklevel=2)
        return None

    output_path = pathlib.Path(sim_controls.outDir()) / (sim_controls.modelName() + ".h5")
    return read(str(output_path))
