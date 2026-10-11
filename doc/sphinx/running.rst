.. highlight:: rest

.. _sec-running:

Running a SLUG Simulation
==========================

SLUG v3 can be run in three ways: as a :ref:`command-line program <ssec-running-command-line>`,
as a :ref:`Python module <ssec-running-python>`, or through a
:ref:`container <ssec-running-container>`. See :ref:`sec-quickstart` for a quick introduction
to the first two methods.

.. _ssec-running-command-line:

Running SLUG From the Command Line
----------------------------------

To run SLUG from the command line, you need to write a parameter file describing the
simulation you want to run. See :ref:`sec-parameters` for a full description of
the parameter file format, and the ``examples/`` directory in the repository for
example parameter files. Once you have a parameter file, you can run SLUG by
doing

    .. code-block:: bash

        slug path/to/parameter_file.toml

if you installed from PyPI or built using :ref:`pip <sec-pip-install-source>`, or by
doing

    .. code-block:: bash

        build/slug path/to/parameter_file.toml

if you built :ref:`using CMake <sec-building>`. The run will automatically use OpenMP
to parallelize if SLUG was built with OpenMP support, and will use all available threads
by default. You can control the number of threads used by setting the ``OMP_NUM_THREADS``
environment variable, e.g.

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
environment variable, either in the shell before starting Python, or from Python
itself before importing slugpy (the OpenMP runtime reads it only once, when slugpy is
first imported), e.g.,

    .. code-block:: python

        import os
        os.environ["OMP_NUM_THREADS"] = "4"
        import slugpy

.. _ssec-running-container:

Running SLUG from a Container
-----------------------------

If you installed SLUG using the :ref:`container <ssec-getting-container>`, you can run
SLUG from the command line inside the container using the normal way of launching
containers on your system, followed by the normal command-line invocation of SLUG. When
running from a container, you must bind the directory on the host system that contains
the SLUG data files to the container's ``/data`` directory. For Apptainer, the command
to run a simulation is:

    .. code-block:: bash

        apptainer exec --bind /path/to/slug-data:/data slug3.sif slug path/to/parameter_file.toml

while the equivalent command for Docker is:

    .. code-block:: bash

        docker run --rm -v /path/to/slug-data:/data -v "$PWD":"$PWD" -w "$PWD" \
            ghcr.io/markkrumholz/slug3:latest slug path/to/parameter_file.toml

where ``/path/to/slug-data`` is the directory on the host system that contains the
``data/`` subdirectory with the SLUG data files. See :ref:`ssec-getting-container` for
details on how to install the container and download the data files.

SLUG reads its parameter file (and any other input files it names by relative path)
from, and writes its output to, its working directory inside the container, so the
host directory you run from must be visible inside the container and be its working
directory. Apptainer usually arranges this automatically, but if your output does not
appear, add ``--bind "$PWD" --pwd "$PWD"`` to the ``apptainer exec`` command. Docker
never does this automatically, which is why the Docker command above includes
``-v "$PWD":"$PWD" -w "$PWD"``; note also that Docker runs as the root user by default,
so the output files it writes will be owned by root unless you also add
``--user "$(id -u):$(id -g)"``. See :ref:`ssec-parameters-output` for how to write
the output to a different directory instead.

The container also includes Python with slugpy installed, so you can likewise run
Python scripts that use slugpy (see :ref:`ssec-running-python`) inside it, e.g.,
``apptainer exec --bind /path/to/slug-data:/data slug3.sif python my_script.py``.

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

(or ``slug --restart``/``slug -R`` if you installed SLUG from PyPI or with ``pip``, or
inside the container). To restart from a checkpoint when running in Python, do

    .. code-block:: python

        sim_result = slugpy.run_sim(sim_controls, restart=True)

For either the command line or Python interface, when the restart flag is present SLUG
will automatically find the most recent checkpoint file, determine the number of trials
already completed, and resume to reach the target number. The :ref:`sec-slugpy` output
reader also automatically handles runs whose outputs are divided across checkpoints.

When checkpointing is enabled, SLUG will also safely complete the current trial and
finalize its output files if it receives ``SIGTERM``. This capability can be used to
force-write a final checkpoint shortly before hitting walltime limits when running
in a queued environment.

.. _sec-running-mpi:

Running SLUG With MPI
----------------------

If SLUG was built with MPI support (see :ref:`sec-dependencies`), the trials of a
single run can also be divided across multiple processes -- potentially spanning
multiple machines -- in addition to (or instead of) OpenMP's own multi-threading
within a process. This is only available from the command line, not from Python.
Note that installations from PyPI, and builds made with ``pip`` (unless MPI is
enabled explicitly; see :ref:`sec-pip-install-source`), do not include MPI support,
so to use MPI either build SLUG from source :ref:`using CMake <sec-building>`, or use
the :ref:`container <ssec-getting-container>`. Launch ``slug`` through your MPI
implementation's usual launcher, e.g.

    .. code-block:: bash

        # CMake build
        mpirun -n 4 build/slug path/to/parameter_file.toml
        # pip build with MPI support
        mpirun -n 4 slug path/to/parameter_file.toml
        # container
        mpirun -n 4 apptainer exec --bind /path/to/slug-data:/data slug3.sif slug path/to/parameter_file.toml

which divides the run's trials evenly across the 4 processes ("ranks"). When the
container is launched by the host system's ``mpirun``, as in the last example, the
host's MPI implementation must be compatible with the one inside the container (Open
MPI 4.1). For a run on a single machine, you can avoid this requirement by using the
container's own ``mpirun`` instead, e.g.,
``apptainer exec --bind /path/to/slug-data:/data slug3.sif mpirun -n 4 slug path/to/parameter_file.toml``. Each rank
still uses OpenMP to further parallelize across its own share of trials if SLUG was
also built with OpenMP support, so a run started this way is dividing work two
levels deep: first across ranks, then across each rank's own threads. Set
``OMP_NUM_THREADS`` as usual (see above) to control the second level; the first is
controlled the normal way for your MPI implementation, e.g. ``mpirun -n``'s own
argument above. It is generally advisable to then set additional environment
variables to control thread binding and affinity, e.g. ``OMP_PROC_BIND=close`` and
``OMP_PLACES=cores``, but the exact settings depend on the MPI implementation and
machine architecture, so consult the recommended settings for the system on which
you are running.

Restarting an MPI run works the same way as a single-process run (see
"Checkpointing and Batch Runs" above), with one added benefit: the number of ranks
used to restart a run need not match the number used originally -- a run started
on 4 ranks can be restarted on 8, or on just 1, and SLUG will pick up correctly
either way. This flexibility does not extend to a run whose output has already
been fully consolidated into a single file (``output_mode = "h5"``, once the run
completes; see :ref:`sec-output`): restarting from a consolidated checkpoint is
only supported with exactly one rank, since consolidation discards the
per-rank bookkeeping a multi-rank restart needs. A run using
``output_mode = "h5divided"`` never consolidates its output at all, so it can
always be restarted with a different number of ranks.
