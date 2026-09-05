.. highlight:: rest

Quickstart
==========

To get started with SLUG, carry out the following steps:

1. Download SLUG and its required data by doing:

    .. code-block:: bash
    
        git clone https://github.com/markkrumholz/slug3.git
        git submodule update --init --recursive
        python data/tools/download_data.py --all

    See :ref:`sec-getting` for full details.

2. Compile SLUG by doing:

    .. code-block:: bash

        cmake -S . -B build -G Ninja
        cmake --build build

    from the repository root. See :ref:`sec-building` for full details. (If this
    doesn't work, check the list of dependencies and supported compilers in
    :ref:`sec-building`.)

3. Run the ``quick`` test suite to make sure everything is working:

    .. code-block:: bash

        cd build
        ctest -L quick

    See :ref:`sec-tests` for the full details on the test suite.

4. Write a parameter file describing the simulation you want to run. See 
:ref:`sec-parameters` for details, and the ``examples/`` directory in the
repository for example parameter files.

5. Run SLUG either from the command line or from Python. To run from the command
line, do::

    build/slug path/to/parameter_file.toml

and to run from Python, do::

    import slugpy
    sim_result = slugpy.run_sim("path/to/parameter_file.toml")

See :ref:`sec-running` for full details on running SLUG from the command line or
from Python.
