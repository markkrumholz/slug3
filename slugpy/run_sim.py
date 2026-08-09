"""
run_sim.py

Implements run_sim, a convenience function that runs an entire slug
simulation end to end from Python, exactly as the slug command-line
executable does.
"""

import gc
import pathlib
import warnings

from ._slug import OutputManagerAscii, OutputManagerH5, SimCluster, SimControls
from .read import read
from .slug_reader import slug_reader


def run_sim(input_deck: str) -> slug_reader | None:
    """
    Run a slug simulation end to end, exactly as the slug command-line
    executable does.

    Parameters
    ----------
    input_deck : str
        Either the text of a slug TOML input deck, or a path to one on
        disk (see SimControls's own constructor for the exact rule).

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
    (matching sim_controls.outputMode()), and -- if
    sim_controls.simType() is SimControls.SimType.cluster -- a
    SimCluster, which it then runs. Galaxy-type simulations are not
    yet supported, matching main.cpp's own current state (that branch
    is intentionally left empty here too).

    ASCII output is meant for small, human-readable output rather than
    batch processing, so there is no ASCII counterpart to slug_reader
    (yet -- this may change in the future). A deck requesting it still
    runs to completion, and its output files are still written, but
    this function has nothing to return for them, so it warns and
    returns None rather than trying to read back a file that was never
    written.
    """
    sim_controls = SimControls(input_deck)

    if sim_controls.outputMode() == SimControls.OutputMode.h5:
        output_manager = OutputManagerH5(sim_controls, input_deck)
    else:
        output_manager = OutputManagerAscii(sim_controls, input_deck)

    if sim_controls.simType() == SimControls.SimType.cluster:
        sim = SimCluster(sim_controls, output_manager)
        sim.run()
        del sim
    elif sim_controls.simType() == SimControls.SimType.galaxy:
        pass  # Galaxy simulation support will be added in a future PR

    # Drop the only Python-side reference to output_manager (whether or
    # not it was actually consumed by a SimCluster above) so its
    # destructor -- which closes the underlying output file(s) -- runs
    # now, before the read() below tries to reopen the same file for
    # reading; HDF5 otherwise refuses a second open while the first
    # (write) handle is still live.
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
