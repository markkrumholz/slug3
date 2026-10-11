.. highlight:: rest

.. _sec-quickstart:

Quickstart
==========

To get started with SLUG, you can either :ref:`install it from PyPI <ssec-quickstart-pypi>`,
use :ref:`a container <ssec-quickstart-container>`, or
:ref:`build from source <ssec-quickstart-source>`. Quickstart directions for each path are
below, with full details in :ref:`sec-getting`.

.. _ssec-quickstart-pypi:

Installing from PyPI
--------------------

#. Install SLUG and its required data by doing:

    .. code-block:: bash

        pip install slugpy
        python -m slugpy.download_data

#. Write a parameter file describing the simulation you want to run. See :ref:`sec-parameters` for details, and the `examples/ <https://github.com/markkrumholz/slug3/tree/main/examples>`_ directory in the SLUG repository for example parameter files (see :ref:`sec-examples`).

#. Run SLUG either from the command line or from Python. To run from the command line, do:

    .. code-block:: bash

        slug path/to/parameter_file.toml

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

    This step is not needed if SLUG was run from Python with HDF5 output (the default), as
    the Python runner automatically returns a Python lazy-reader pointing to the simulation
    output in that case (for ASCII output, it returns ``None`` instead, and this step doesn't
    apply). See :ref:`sec-slugpy` for details on how to use the lazy-reader, and
    :ref:`sec-output` for full documentation of the contents of the output of a simulation.

.. _ssec-quickstart-container:

Using a Container
-----------------

#. Install SLUG and its required data by doing:

    .. code-block:: bash

        # If using Docker:
        docker pull ghcr.io/markkrumholz/slug3:latest
        docker run --rm -v /path/to/slug-data:/data ghcr.io/markkrumholz/slug3:latest \
            python -m slugpy.download_data

        # If using Apptainer/Singularity:
        apptainer pull slug3.sif docker://ghcr.io/markkrumholz/slug3:latest
        apptainer exec --bind /path/to/slug-data:/data slug3.sif python -m slugpy.download_data

    depending on whether you are using `Docker <https://www.docker.com/>`_ or
    `Apptainer <https://apptainer.org/>`_.

#. Write a parameter file describing the simulation you want to run. See :ref:`sec-parameters` for details, and the `examples/ <https://github.com/markkrumholz/slug3/tree/main/examples>`_ directory in the SLUG repository for example parameter files (see :ref:`sec-examples`). With Docker, the parameter file must be in or below the directory from which you run the container, since that is the only directory of the host system (other than the data directory) visible inside it.

#. Run SLUG either from the command line or from Python. To run from the command line, do:

    .. code-block:: bash

        # If using Docker:
        docker run --rm -v /path/to/slug-data:/data -v "$PWD":"$PWD" -w "$PWD" \
            ghcr.io/markkrumholz/slug3:latest slug path/to/parameter_file.toml

        # If using Apptainer/Singularity:
        apptainer exec --bind /path/to/slug-data:/data slug3.sif slug path/to/parameter_file.toml

    and to run from Python, start up a Python interpreter inside the container by doing:

    .. code-block:: bash

        # If using Docker:
        docker run --rm -it -v /path/to/slug-data:/data -v "$PWD":"$PWD" -w "$PWD" \
            ghcr.io/markkrumholz/slug3:latest python

        # If using Apptainer/Singularity:
        apptainer exec --bind /path/to/slug-data:/data slug3.sif python

    and then do:

    .. code-block:: python

        import slugpy
        sim_result = slugpy.run_sim("path/to/parameter_file.toml")

#. If SLUG was run from the command line, use the Python lazy-reader to examine the output. First start up a Python interpreter inside the container as per the previous step, then do:

    .. code-block:: python

        import slugpy
        sim_result = slugpy.read("path/to/model_name")

    This step is not needed if SLUG was run from Python with HDF5 output (the default), as
    the Python runner automatically returns a Python lazy-reader pointing to the simulation
    output in that case (for ASCII output, it returns ``None`` instead, and this step doesn't
    apply). See :ref:`sec-slugpy` for details on how to use the lazy-reader, and
    :ref:`sec-output` for full documentation of the contents of the output of a simulation.

.. _ssec-quickstart-source:

Building from Source
--------------------

#. Download SLUG and its required data by doing:

    .. code-block:: bash

        git clone https://github.com/markkrumholz/slug3.git
        cd slug3
        git submodule update --init --recursive
        python data/tools/download_data.py

    See :ref:`sec-getting` for full details.

#. Compile SLUG by doing:

    .. code-block:: bash

        cmake -S . -B build -G Ninja
        cmake --build build

    from the repository root. See :ref:`sec-building` for full details. (If this
    doesn't work, check the list of dependencies and supported compilers in
    :ref:`sec-dependencies`.)

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

    This step is not needed if SLUG was run from Python with HDF5 output (the default), as
    the Python runner automatically returns a Python lazy-reader pointing to the simulation
    output in that case (for ASCII output, it returns ``None`` instead, and this step doesn't
    apply). See :ref:`sec-slugpy` for details on how to use the lazy-reader, and
    :ref:`sec-output` for full documentation of the contents of the output of a simulation.