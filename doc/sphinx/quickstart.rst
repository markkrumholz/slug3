.. highlight:: rest

.. _sec-quickstart:

Quickstart
==========

To get started with SLUG, carry out the following steps:

#. Download SLUG and its required data by doing:

    .. code-block:: bash
    
        git clone https://github.com/markkrumholz/slug3.git
        git submodule update --init --recursive
        python data/tools/download_data.py

    See :ref:`sec-getting` for full details.

#. Compile SLUG by doing:

    .. code-block:: bash

        cmake -S . -B build -G Ninja
        cmake --build build

    from the repository root. See :ref:`sec-building` for full details. (If this
    doesn't work, check the list of dependencies and supported compilers in
    :ref:`sec-building`.)

#. Run the ``quick`` test suite to make sure everything is working:

    .. code-block:: bash

        cd build
        ctest -L quick

    See :ref:`sec-tests` for the full details on the test suite.

#. Write a parameter file describing the simulation you want to run. See :ref:`sec-parameters` for details, and the ``examples/`` directory in the repository for example parameter files (see :ref:`sec-examples`).

#. Run SLUG either from the command line or from Python. To run from the command line, do:

    .. code-block:: bash

        build/slug path/to/parameter_file.toml

    and to run from Python, do:

    .. code-block:: python

        import slugpy
        sim_result = slugpy.run_sim("path/to/parameter_file.toml")

    See :ref:`sec-running` for full details on running SLUG from the command line or 
    from Python.

#. If SLUG was run from the command line, use the Python lazy-reader to examine the output:

    .. code-block:: python

        import slugpy
        sim_result = slugpy.read("path/to/model_name")

    This step is not needed if SLUG was run from Python, as the Python runner automatically
    returns a Python lazy-reader pointing to the simulation output. See :ref:`sec-slugpy`
    for details on how to use the lazy-reader, and :ref:`sec-output` for full documentation
    of the contents of the output of a simulation.