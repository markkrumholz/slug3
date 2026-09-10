.. highlight:: rest

.. _sec-tests:

Tests
=====

SLUG has an extensive unit test suite covering both the C++ core and the
Python ``slugpy`` package. Tests are built and run through CTest, CMake's
test driver, and are organized into subdirectories that largely mirror the
source tree. This page describes how to run the tests, the major pieces of
the testing architecture, and the automated checks that run against every
pull request.

Running the tests
------------------

Building the project (see :ref:`sec-building`) also builds one test
executable per subdirectory of ``tests``. Running ``ctest`` from the build
directory executes all of them, along with two Python-based entries
(``test_PythonBindings`` and ``test_Slugpy``, described below):

.. code-block:: bash

  cd build
  ctest

The Python-based tests require ``pytest``, ``numpy``, ``h5py``,
``tomlkit``, ``astropy``, and ``scipy`` to be installed for whichever
Python interpreter CMake picked up at configure time.

To run just the tests for a single area of the code, use CTest's ``-R``
(regex) filter, or invoke the corresponding test executable directly, e.g.:

.. code-block:: bash

  ctest -R test_PDFs
  # or, equivalently:
  ./slugTestPDFs

This is almost always preferable to running the whole suite while
iterating on a change -- see the note below on quick vs. slow tests.

Quick vs. slow tests
~~~~~~~~~~~~~~~~~~~~~

Every test is labelled in ``CMakeLists.txt`` as either ``quick`` or
``slow``. The ``quick`` tests each run in a few seconds; the ``slow`` ones
(currently just ``test_CoreFull``, the full-scale end-to-end counterpart to
``test_Core``) drive real simulations against large data files (full MIST
tracks, POWR/TLUSTY/BOSZ/CK04 spectral libraries) that are gitignored
rather than committed to the repository, and can take upwards of a minute
to run even when that data is present. When it isn't, the slow tests
self-skip rather than fail.

Use CTest's label filter to select one group or the other:

.. code-block:: bash

  ctest -L quick    # fast tests only
  ctest -LE slow    # equivalently, everything except the slow tests
  ctest -L slow     # just the slow, full-scale tests

Plain ``ctest`` with no label filter still runs everything, quick and slow
alike. As a rule of thumb: run the quick tests for the areas you touched
routinely while developing, and reserve the slow tests for cases where
you've changed something likely to affect a full end-to-end run (e.g. the
simulation drivers in ``src/core``, or the data files a real run depends
on).

Test suite layout
-------------------

The ``tests`` directory's subdirectory structure mirrors the first-party
C++ modules under ``src``, with one test executable per subdirectory (see
AGENTS.md for the full convention, including the one exception:
``src/extern``, vendored third-party code, has no corresponding
``tests/extern``). Subdirectories may contain their own ``assets``
directory holding data files the tests need; some of these are generated
by scripts in ``data/tools`` (e.g. ``make_powr_test_fixture.py``).

The C++ test executables are:

* ``tests/interpolation`` -- the grid interpolation classes (``src/interpolation``).
* ``tests/elem`` -- element/isotope data handling (``src/elem``).
* ``tests/extinct`` -- extinction curves and their application (``src/extinct``).
* ``tests/phot`` -- photometric filters and magnitude conversions (``src/phot``).
* ``tests/pdfs`` -- the probability distribution classes used for the IMF and
  other sampled quantities (``src/pdfs``).
* ``tests/utils`` -- assorted utility functions (``src/utils``).
* ``tests/tracks`` -- stellar evolutionary track classes (``src/tracks``).
* ``tests/specsyn`` -- spectral synthesis (``src/specsyn``).
* ``tests/nebular`` -- nebular emission processing (``src/nebular``).
* ``tests/io`` -- output and simulation-control I/O (``src/io``).
* ``tests/core`` -- the simulation drivers (clusters, galaxies, and their
  combination with spectral synthesis) in ``src/core``. This directory holds
  two executables: ``slugTestCore``, fast tests that run on every CI build,
  and ``slugTestCoreFull``, the slow, full-scale end-to-end tests described
  above.

Two further entries run Python tests via ``pytest``, rather than being
their own compiled executables:

* ``tests/pybind`` (``test_PythonBindings``) -- exercises the compiled
  ``_slug`` extension module directly, against the same interpreter
  pybind11 built it for.
* ``tests/slugpy`` (``test_Slugpy``) -- pure-Python tests of the ``slugpy``
  package itself (readers, post-processing utilities, and its
  ``cloudy`` subpackage for building/parsing Cloudy input and output).
  Since these are pure Python, they need no compiled extension, and since
  they exercise ``slugpy`` against a small real output file committed to
  the repository (``examples/clusterlib/clusterlib.h5``) rather than a
  full simulation run, they stay in the ``quick`` group.

Continuous integration
------------------------

Every pull request triggers a GitHub Actions workflow (``.github/workflows/ci.yml``)
with four jobs:

* **test** -- builds SLUG and runs ``ctest --output-on-failure -L quick``
  across a matrix of four configurations: GCC and Clang, each on Ubuntu and
  macOS. Only the quick tests run in CI, for the same reason they self-skip
  locally without the necessary data: the large track/spectral-library
  files the slow tests need don't fit in the repository.
* **lint** -- runs ``clang-tidy`` on the C++ source and header files changed
  relative to the PR's base branch (excluding vendored code under
  ``src/extern``), using the exact ``clang-tidy`` version CI uses.
* **pyright** -- type-checks the ``slugpy`` and ``tests/slugpy`` Python files
  changed relative to the base branch. This requires the compiled ``_slug``
  extension to be built first, so that ``slugpy``'s ``from ._slug import
  ...`` imports resolve.
* **ruff** -- runs ``ruff check`` against the same changed ``slugpy``/
  ``tests/slugpy`` files.

The ``clang-tidy``, ``pyright``, and ``ruff`` jobs all operate only on files
that changed in the PR, not the whole tree, so it's worth running the
equivalent check locally on your own changed files before pushing (see
AGENTS.md's Coding style section for the exact commands). ``data/tools``
and ``tests/pybind`` are not held to the ``pyright``/``ruff`` standard and
are excluded from those two jobs.

A pull request is expected to have all four CI jobs passing, plus a
completed CodeRabbit review, before it is merged.
