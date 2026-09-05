.. highlight:: rest

Running a SLUG Simulation
==========================

SLUG v3 can be run in two ways: either as a command-line program, or as a Python
module. See :ref:`sec-quickstart` for a quick introduction to both methods.

A first step for either way of running SLUG, which is recommended but not required,
is to set the environment variable ``SLUG_DIR`` to the repository root directory.
This simplifies finding the data files that SLUG needs to run by providing an
automatic search path that will be used when parsing the inputs -- see
:ref:`sec-parameters` for details. For example, if you cloned the repository
into ``/home/user/slug3``, you can do::

    export SLUG_DIR=/home/user/slug3

Running SLUG From the Command Line
----------------------------------

To run SLUG from the command line, you need to write a parameter file describing the
simulation you want to run. See :ref:`sec-parameters` for a full description of
the parameter file format, and the ``examples/`` directory in the repository for
example parameter files. Once you have a parameter file, you can run SLUG by
doing::

    build/slug path/to/parameter_file.toml

The run will automatically use OpenMP to parallelize if SLUG was built with OpenMP
support, and will use all available threads by default. You can control the number
of threads used by setting the ``OMP_NUM_THREADS`` environment variable, e.g.::

    export OMP_NUM_THREADS=4

The outputs of the simulation will be written to a file in the current working directory
(or in the directory specified by the ``out_dir`` keyword in the parameter file)
whose name matches the ``model_name`` provided in the parameter file.
See :ref:`sec-output` for details on the outputs and how to read them.

Running SLUG From Python
------------------------

To run a SLUG simulation from Python, you first need to import the ``slugpy`` module;
see :ref:`sec-slugpy` for details. The module contains a function ``run_slug`` that can
run a simulation end-to-end. This function takes a single argument, which can be either
the path to a parameter file, or a ``SimControls`` object that contains all the information
that a parameter file would contain; see :ref:`sec-slugpy` for details on ``SimControls``
and ``run_sim``. Once you have your input, you can run a simulation by doing::

    import slugpy
    sim_result = slugpy.run_sim("path/to/parameter_file.toml")

or::

    import slugpy
    sim_controls = slugpy.SimControls(...)
    sim_result = slugpy.run_sim(sim_controls)

The result of the simulation will be returned in ``sim_result``, which is a 
``slugpy.slug_reader`` object that can be used to inspect the outputs. The output
will also be written to disk as with runs from the command line. See
:ref:`sec-output` for details on the outputs and how to read them.

Simulations run from Python will also automatically be parallelized with OpenMP if
SLUG was built with OpenMP support, and will use all available threads by default.
You can control the number of threads used by setting the ``OMP_NUM_THREADS``
environment variable, e.g.::

    import os
    os.environ["OMP_NUM_THREADS"] = "4"