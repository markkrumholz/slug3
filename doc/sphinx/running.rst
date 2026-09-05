.. highlight:: rest

Running a SLUG Simulation
==========================

SLUG v3 can be run in two ways: either as a command-line program, or as a Python
module. The command-line program is the easiest way to get started, but the
Python interface is more flexible and allows for more complex workflows.
See :ref:`sec-quickstart` for a quick introduction to both methods.

Running SLUG From the Command Line
----------------------------------

A first step, which is recommended but not required, is to set the environment
variable ``SLUG_DIR`` to the repository root directory. This simplifies finding
the data files that SLUG needs to run. For example, if you cloned the repository
into ``/home/user/slug3``, you can do::

    export SLUG_DIR=/home/user/slug3

The next step, once that is done, is to write a parameter file describing the
simulation you want to run. See :ref:`sec-parameters` for a full description of
the parameter file format, and the ``examples/`` directory in the repository for
example parameter files. Once you have a parameter file, you can run SLUG by
doing::

    build/slug path/to/parameter_file.toml

The run will automatically use OpenMP to parallelize if SLUG was built with OpenMP
support, and will use all available threads by default. You can control the number
of threads used by setting the ``OMP_NUM_THREADS`` environment variable, e.g.::

    export OMP_NUM_THREADS=4

Running SLUG From Python
------------------------

