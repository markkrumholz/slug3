.. highlight:: rest

.. _sec-compiling:

Compiling and Installing SLUG
==============================

SLUG has no separate installation step: building it with CMake produces the
``slug`` command-line executable directly in the build directory, and the
compiled Python extension module lands directly inside the ``slugpy``
package in the source tree, alongside its pure-Python code. There is
nothing to run afterwards to "install" it -- see :ref:`sec-building` below.

.. _sec-dependencies:

Dependencies
-------------

Required
~~~~~~~~~

* `CMake <https://cmake.org/>`_, version 3.15 or later.
* A C++23-capable compiler. This is a firmer requirement than it sounds: two
  particular C++23 features are used throughout the code with no fallback
  if the compiler/standard-library combination doesn't have them --

  * The multidimensional ``operator[]`` (e.g. ``view[i, j, k]``), used
    wherever the code indexes a ``std::mdspan``.
  * ``std::views::zip``, from ``<ranges>``.

  In practice, the two configurations SLUG's own continuous integration
  builds and tests against are:

  * GCC 16 (paired with a libstdc++ new enough to have these features --
    see the note on ``<mdspan>`` under "Bundled" below, since that specific
    header can lag behind the rest of a compiler's own C++23 support even
    at the same nominal GCC version).
  * Clang 18, linked against LLVM's own libc++ 18 (*not* whatever
    system-provided libc++/libstdc++ the compiler defaults to -- see below).

  Older GCC/Clang releases, and other C++23-capable compilers such as
  Intel's LLVM-based ``icpx``/``icx`` (part of the oneAPI DPC++/C++
  Compiler; the older, classic ``icc`` is discontinued and should not be
  used), have not been verified and may or may not work -- if you have
  success or failure with a configuration not listed here, please let us
  know so this list can be extended.
* The `GNU Scientific Library <https://www.gnu.org/software/gsl/>`_ (GSL).
* `HDF5 <https://www.hdfgroup.org/solutions/hdf5/>`_, built with its C++
  component enabled.

Bundled (no action needed)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These are vendored as git submodules under ``src/extern`` and built
automatically as part of the normal build -- just make sure they were
fetched (see :ref:`sec-getting` on cloning with submodules; if you forgot,
``git submodule update --init --recursive`` from the repository root
fetches them after the fact too):

* `pybind11 <https://pybind11.readthedocs.io/>`_, used to build the Python
  bindings (see below).
* A reference implementation of C++23's ``<mdspan>`` (from the `Kokkos
  mdspan <https://github.com/kokkos/mdspan>`_ project), used as a last
  resort if neither the compiler's own standard library nor an externally
  installed ``mdspan`` package provides one.

  Whether a given compiler provides ``<mdspan>`` natively depends on the
  *exact* standard-library snapshot it was built against, not just its
  nominal major version -- e.g. Homebrew's own ``gcc@16`` build on macOS
  bundles a libstdc++ new enough to have it, while the ``gcc-16`` package
  from Ubuntu's ``ppa:ubuntu-toolchain-r/test`` (as used in CI) does not,
  as of when this was last checked, and Apple's own system Clang/libc++
  has no native ``<mdspan>`` at all. None of this matters in practice:
  ``cmake/Mdspan.cmake`` detects native support with a real compile check
  (not a version guess) and falls back automatically, in the order listed
  above, to an externally-installed ``mdspan`` package and then this
  bundled submodule -- see that file's own comments, or AGENTS.md, for the
  full detail. Application code always just does ``#include <mdspan>``
  regardless of which provider actually supplied it.

Optional
~~~~~~~~~

* `OpenMP <https://www.openmp.org/>`_, for multi-threaded execution. If no
  OpenMP implementation is found, the build falls back to a single-threaded
  binary automatically, with a warning at configure time.
* `Ninja <https://ninja-build.org/>`_, as the CMake generator. Not required
  -- CMake falls back to its own platform default (Unix Makefiles on
  Linux/macOS) -- but Ninja parallelizes the build automatically and
  matches what SLUG's own CI uses, so local builds behave the same way and
  incremental rebuilds are noticeably faster.
* `Doxygen <https://www.doxygen.nl/>`_ (install separately, e.g. via your
  system package manager or Homebrew), plus the Python packages `Sphinx
  <https://www.sphinx-doc.org/>`_, `Breathe
  <https://breathe.readthedocs.io/>`_, and `Exhale
  <https://exhale.readthedocs.io/>`_::

      pip install sphinx breathe exhale sphinx_rtd_theme

  to build this documentation site itself as part of the normal build --
  see :ref:`sec-building-docs` below. Entirely optional: every other part
  of the build is unaffected if these aren't installed.

Python
~~~~~~~

SLUG's Python frontend, ``slugpy`` (see :ref:`sec-slugpy`), requires Python
3.12 or later, plus `astropy <https://www.astropy.org/>`_, `h5py
<https://www.h5py.org/>`_, `NumPy <https://numpy.org/>`_, `SciPy
<https://scipy.org/>`_, `tomlkit <https://tomlkit.readthedocs.io/>`_, and
`tqdm <https://tqdm.github.io/>`_::

    pip install astropy h5py numpy scipy tomlkit tqdm

CMake builds the compiled ``_slug`` extension module against whichever
Python interpreter it finds (or the one given explicitly via
``-DPython_EXECUTABLE``, see below), and that same interpreter needs the
packages above installed for ``import slugpy`` to work afterwards.
``pyproject.toml`` lists these same packages (and the optional ones just
below) under ``[project.dependencies]``/``[project.optional-dependencies]``
for tooling to read, but that file cannot itself be installed with ``pip
install .`` -- there is deliberately no ``[build-system]`` table, since
``_slug`` is built by CMake directly into the ``slugpy`` package rather
than through a separate Python packaging step (see the file's own header
comment).

A further three tools are needed only for specific development tasks, not
for using ``slugpy`` itself: `pytest <https://pytest.org/>`_ to run the
test suite, `pyright <https://microsoft.github.io/pyright/>`_ to type-check
``slugpy`` locally, and `ruff <https://docs.astral.sh/ruff/>`_ to lint it::

    pip install pytest pyright ruff

.. _sec-building:

Configuring and Building
--------------------------

SLUG uses an out-of-source CMake build. From the repository root::

    cmake -S . -B build -G Ninja
    cmake --build build

(omit ``-G Ninja`` to use CMake's own default generator instead, if Ninja
isn't installed -- see :ref:`sec-dependencies` above). This produces the
``slug`` command-line executable at ``build/slug``, and builds the compiled
Python extension module directly into the ``slugpy`` package in the source
tree (so ``import slugpy`` works from the repository root immediately
afterward, with no separate install step).

A few configure-time options worth knowing about:

``-DPython_EXECUTABLE=/path/to/python3``
    Build the Python bindings against a specific Python interpreter, rather
    than whichever one CMake finds automatically. Needed if you use a
    virtual environment, conda environment, or other non-default
    interpreter for ``slugpy`` -- see :ref:`sec-dependencies` above for
    what that interpreter needs installed.

``-DSLUG_MDSPAN_PROVIDER=AUTO|NATIVE|PACKAGE|SUBMODULE``
    Override the automatic ``<mdspan>`` provider selection described above
    -- e.g. to require a specific provider and get a hard error if it
    isn't available, rather than a silent fallback. Defaults to ``AUTO``.

``-DCMAKE_BUILD_TYPE=Debug|Release|RelWithDebInfo``
    The usual CMake build-type switch. Some of SLUG's own performance-
    sensitive targets (e.g. the Python extension module) are always built
    with optimizations enabled regardless of this setting, since they drive
    the same interpolation-heavy code any analysis using ``slugpy``
    depends on.

Verifying the Build
---------------------

Once the build finishes, ``ctest`` (run from the build directory) executes
SLUG's own test suite::

    cd build
    ctest -L quick

The ``quick`` label selects tests that run in a few seconds each and need
no large data files beyond what's already committed to the repository;
some further, slower end-to-end tests exist but require data files too
large to distribute this way (see :ref:`sec-tests` for the full detail).

.. _sec-building-docs:

Building this Documentation
------------------------------

If Doxygen and the Python packages Sphinx, Breathe, and Exhale (see
:ref:`sec-dependencies` above) are all found at configure time, ``cmake
--build build`` (with no explicit ``--target``) builds this documentation
site too, into ``doc/html``, alongside everything else -- if they aren't
found, that step is simply skipped, with a note printed at configure time,
and every other target is unaffected. To build just the documentation
without touching anything else::

    cmake --build build --target docs

or, from ``doc/sphinx`` directly, without going through CMake at all::

    cd doc/sphinx
    make html
