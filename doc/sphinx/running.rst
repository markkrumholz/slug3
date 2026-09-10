.. highlight:: rest

.. _sec-running:

Running a SLUG Simulation
==========================

SLUG v3 can be run in two ways: either as a command-line program, or as a Python
module. See :ref:`sec-quickstart` for a quick introduction to both methods.

A first step for either way of running SLUG, which is recommended but not required,
is to set the environment variable ``SLUG_DIR`` to the repository root directory.
This simplifies finding the data files that SLUG needs to run by providing an
automatic search path that will be used when parsing the inputs -- see
:ref:`sec-parameters` for details. For example, if you cloned the repository
into ``/home/user/slug3``, you can do:

    .. code-block:: bash

        export SLUG_DIR=/home/user/slug3

Running SLUG From the Command Line
----------------------------------

To run SLUG from the command line, you need to write a parameter file describing the
simulation you want to run. See :ref:`sec-parameters` for a full description of
the parameter file format, and the ``examples/`` directory in the repository for
example parameter files. Once you have a parameter file, you can run SLUG by
doing

    .. code-block:: bash

        build/slug path/to/parameter_file.toml

The run will automatically use OpenMP to parallelize if SLUG was built with OpenMP
support, and will use all available threads by default. You can control the number
of threads used by setting the ``OMP_NUM_THREADS`` environment variable, e.g.

    .. code-block:: bash

        export OMP_NUM_THREADS=4

The outputs of the simulation will be written to a file in the current working directory
(or in the directory specified by the ``out_dir`` keyword in the parameter file)
whose name matches the ``model_name`` provided in the parameter file.
See :ref:`sec-output` for details on the outputs and how to read them.


.. _ssec-running-python:

Running SLUG From Python
------------------------

To run a SLUG simulation from Python, you first need to import the ``slugpy`` module;
see :ref:`sec-slugpy` for details. The module contains a function ``run_sim`` that can
run a simulation end-to-end. This function takes a single argument, which can be either
the path to a parameter file, or a ``SimControls`` object that contains all the information
that a parameter file would contain; see :ref:`sec-slugpy` for details on ``SimControls``
and ``run_sim``. Once you have your input, you can run a simulation by doing

    .. code-block:: python
        
        import slugpy
        sim_result = slugpy.run_sim("path/to/parameter_file.toml")

or:

    .. code-block:: python

        import slugpy
        sim_controls = slugpy.SimControls(...)
        sim_result = slugpy.run_sim(sim_controls)

If the simulation's own output mode is HDF5 (the default; see
:ref:`ssec-parameters-output`), the result will be returned in ``sim_result``,
which is a ``slugpy.slug_reader`` object that can be used to inspect the
outputs; for ASCII output, ``sim_result`` is ``None`` instead, since ASCII
output cannot currently be read back this way. The output is written to disk
either way, exactly as with runs from the command line. See
:ref:`sec-output` for details on the outputs and how to read them.

Simulations run from Python will also automatically be parallelized with OpenMP if
SLUG was built with OpenMP support, and will use all available threads by default.
You can control the number of threads used by setting the ``OMP_NUM_THREADS``
environment variable, e.g.,

    .. code-block:: python

        import os
        os.environ["OMP_NUM_THREADS"] = "4"


Checkpointing and Batch Runs
----------------------------

In order to support long runs involving many Monte Carlo trials, particularly in
queued HPC environments, SLUG can write checkpoints partway through runs; see
:ref:`ssec-parameters-output` for how to turn on this capability. Checkpoints will
contain a certain number of trials, and SLUG runs can pick back up from the most
recent checkpoint. To restart from a checkpoint when running from the command line
do

    .. code-block:: bash

        build/slug --restart path/to/parameter_file.toml

or

    .. code-block:: bash

        build/slug -R path/to/parameter_file.toml

To restart from a checkpoint when running in Python, do

    .. code-block:: python

        sim_result = slugpy.run_sim(sim_controls, restart=True)

For either the command line or Python interface, when the restart flag is present SLUG
will automatically find the most recent checkpoint file, determine the number of trials
already completed, and resume to reach the target number. The :ref:`sec-slugpy` output
reader also automatically handles runs whose outputs are divided across checkpoints.

When checkpointing is enabled, SLUG will also safely complete the current trial and
finalize its ouptut files if it receives ``SIGTERM``. This capability can be used to
force-write a final checkpoint shortly before hitting walltime limits when running
in a queued environment. 