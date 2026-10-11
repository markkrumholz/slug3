.. highlight:: rest

.. _sec-getting:

Getting SLUG
============

You can get SLUG v3 in three ways: :ref:`installing from PyPI <ssec-getting-pypi>`,
:ref:`using a container <ssec-getting-container>`, or
:ref:`installing from source <ssec-getting-source>`. The first two options are the easiest
and are recommended for most users, while the third option is the most flexible and is
likely to give the best performance on HPC systems.

.. _ssec-getting-pypi:

Installing from PyPI
----------------------

You can install SLUG v3 from PyPI using pip, by doing:

    .. code-block:: bash

        pip install slugpy

in a terminal. This will install the latest release of both the slugpy Python package and
the ``slug`` executable, which can be run from the command line. SLUG requires a number of
data files that are not included in the PyPI distribution due to their size. To download
these, run:

    .. code-block:: bash

        python -m slugpy.download_data

By default the data will be downloaded into the ``data`` subdirectory of a per-user directory
(e.g., ``~/.local/share/slug/data`` on Linux, or
``~/Library/Application Support/slug/data`` on macOS), or into ``$SLUG_DIR/data`` if the
environment variable ``SLUG_DIR`` is set. Warning: the data files are large (several GB),
so make sure you have enough disk space before downloading them. You can download only some
of the data by doing:

    .. code-block:: bash

        python -m slugpy.download_data --subdir SUBDIR

where ``SUBDIR`` is the name of one subdirectory of the data directory; run the command once
for each subdirectory you want. The full list of subdirectories is:

* ``filters``: photometric filter transmission curves
* ``nebular``: nebular emission model data
* ``spectra``: stellar atmosphere model spectra
* ``tracks``: stellar evolutionary tracks

Prebuilt wheels are available for Linux on x86_64 and aarch64 processors, and for macOS 14
or later on Apple Silicon (arm64), for Python 3.12 through 3.14. On other platforms or
Python versions, pip builds SLUG from source, which requires the compiler and libraries
listed in :ref:`sec-dependencies`.

.. _ssec-getting-container:

Using a container
-----------------

SLUG v3 is also distributed as a container image, hosted on the GitHub Container
Registry as ``ghcr.io/markkrumholz/slug3``. Each release is tagged with its version
number (e.g., ``ghcr.io/markkrumholz/slug3:3.0.1``), and the tag ``latest`` always
points to the most recent release. The image contains the ``slug`` command-line
executable, built with support for both OpenMP and MPI parallelism, and the slugpy
package, installed in a Python environment whose ``python`` and ``slug`` are on the
default ``PATH``. The image is built for Linux on x86_64 (``linux/amd64``) processors; it
will run on other architectures (e.g., Apple Silicon Macs) only under emulation, which
is much slower, so on those platforms installing from PyPI is a better choice.

You can use the image with `Docker <https://www.docker.com/>`_, by doing:

    .. code-block:: bash

        docker pull ghcr.io/markkrumholz/slug3:latest

or, on HPC systems, which generally do not provide Docker, with
`Apptainer <https://apptainer.org/>`_ (formerly called Singularity), which converts
the image into a single file, here called ``slug3.sif``:

    .. code-block:: bash

        apptainer pull slug3.sif docker://ghcr.io/markkrumholz/slug3:latest

The large data files SLUG needs are not included in the image. Instead, the image
expects to find them in a directory ``/data`` inside the container, which you provide
by binding (mounting) a directory on the host system there. The data files themselves
go in a ``data/`` subdirectory of that directory, and you can download them into it by
running the same download script as for PyPI installations inside the container -- be
warned that the data files are large (several GB), so make sure you have enough disk
space to hold them. For example, with Apptainer, to download the data into
``/path/to/slug-data/data``:

    .. code-block:: bash

        apptainer exec --bind /path/to/slug-data:/data slug3.sif python -m slugpy.download_data

The equivalent with Docker is:

    .. code-block:: bash

        docker run --rm -v /path/to/slug-data:/data ghcr.io/markkrumholz/slug3:latest \
            python -m slugpy.download_data

You only need to do this once; every later run of the container just needs the same
directory bound to ``/data`` again. Similarly, slug writes its output files into its
working directory (unless the input deck specifies otherwise), so the directory you
run it from must also be visible inside the container, and must be the container's
working directory. Apptainer usually does this automatically, but if your output does
not appear, bind it explicitly by adding ``--bind "$PWD" --pwd "$PWD"`` to the
``apptainer exec`` command. With Docker, do the same with
``-v "$PWD":"$PWD" -w "$PWD"``. See :ref:`sec-running` for how to run simulations
with the container.

.. _ssec-getting-source:

Installing from source
----------------------

SLUG v3 is available for download from the GitHub repository at
`https://github.com/markkrumholz/slug3 <https://github.com/markkrumholz/slug3>`_.
To get it, use git to clone the repository by doing:

    .. code-block:: bash

        git clone https://github.com/markkrumholz/slug3.git
        cd slug3
        git submodule update --init --recursive

In addition to the SLUG source code and its submodules, SLUG requires a number of
data files that are not included in the GitHub repository due to their size. These data
files can be downloaded using the script ``data/tools/download_data.py`` in the
repository. For example, to download all of the data files, run:

    .. code-block:: bash

        python data/tools/download_data.py

Warning: the data files are large (several GB), so make sure you have enough disk space
before downloading them.

If you prefer to download the data files manually, they can be accessed from the
Australian National University
`Data Commons <https://datacommons.anu.edu.au/DataCommons/rest/display/anudc:6518>`_
service.

Once you have downloaded the source code and data files, the next step is to
:ref:`compile SLUG <sec-compiling>`.