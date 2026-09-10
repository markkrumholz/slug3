.. highlight:: rest

.. _sec-troubleshooting:

Troubleshooting
================

This page collects fixes for problems users commonly run into, organized by
where they show up: building SLUG, locating its data files, writing an input
deck, running a simulation, and post-processing with Cloudy. If your problem
isn't covered here, the individual pages linked from each section below go
into much more detail.

Building SLUG
--------------

SLUG requires a C++23-capable compiler with two specific features
(multidimensional ``operator[]`` and ``std::views::zip``) that not every
nominally-C++23 compiler provides -- see :ref:`sec-dependencies` for exactly
which compiler/standard-library combinations are verified to work, and
:ref:`sec-building` for the ``-DSLUG_MDSPAN_PROVIDER`` option if the build
fails specifically on ``<mdspan>``. If ``cmake --build build`` doesn't also
build this documentation site, that's expected unless Doxygen, Sphinx,
Breathe, and Exhale are all installed -- see :ref:`sec-building-docs`.

Data Files Not Found
----------------------

SLUG's own git repository does not include the large stellar track,
atmosphere, and filter data files -- these are downloaded separately (see
:ref:`sec-getting`). If they haven't been downloaded, or SLUG can't find
them, you'll typically see an error like

::

    findMatchingTracks: unable to open HDF5 file .../data/tracks/mist.h5

or

::

    findMatchingTracks: no trackset named MIST

The first indicates the registry entry points to a file that isn't actually
there (usually because the data files were never downloaded); the second
indicates the *registry* itself doesn't know about the track set you asked
for (a typo in ``stars.tracks``, or a custom registry file that doesn't list
it). The same pattern applies to spectral synthesis models
(``findMatchingSpectra: ...``).

Things to check, in order:

#. Did you run ``python data/tools/download_data.py`` (see :ref:`sec-getting`)?
   Without it, ``data/tracks``, ``data/spectra``, etc. only contain registry
   files, not the HDF5 data they point to.
#. Is the ``SLUG_DIR`` environment variable set to the repository root (see
   :ref:`sec-running`)? File names in the input deck are resolved relative to
   the current working directory first, then ``SLUG_DIR``, then the
   repository root determined at build time -- if you run ``slug`` from some
   other directory with ``SLUG_DIR`` unset, and that directory doesn't happen
   to contain its own copy of ``data/``, resolution can fail even though the
   data files exist elsewhere on disk.
#. If you're using a custom registry file (see e.g. :ref:`ssec-tracks-standard`
   for the track-registry format), remember that each entry's ``file`` key is
   resolved relative to *the registry file's own directory*, not the current
   working directory -- a registry moved without the data files it describes
   (or vice versa) will fail to resolve even though both files exist
   somewhere.

Input Deck Errors
-------------------

SLUG validates the input deck up front, before running any simulation, and
reports a specific missing or malformed keyword rather than a generic
parse failure, e.g.:

::

    SimControls: sim_type not found
    SimControls: sim_type must be 'galaxy' or 'cluster'
    SimControls: unknown output_mode <value>
    SimControls: output.start_time not found

These all mean the same thing: a required keyword is missing, misspelled, or
has an invalid value. See :ref:`sec-parameters` for the full, authoritative
list of keywords, their exact spelling, and which section of the input deck
each belongs in -- a keyword given at the wrong nesting level (e.g. outside
its ``[section]``) is reported the same way as if it were missing entirely.

One combination is rejected outright rather than merely defaulting:
requesting checkpointing (``output.checkpoint_interval`` non-zero) together
with ``output.output_mode = "ascii"``:

::

    SimControls: output.checkpoint_interval is non-zero, but checkpointing
    is only supported with HDF5 output (output.output_mode = "h5" or
    "h5divided"), not ascii

Switch to ``"h5"`` or ``"h5divided"`` output if you need checkpointing (see
:ref:`ssec-parameters-output`).

Runtime Warnings and Errors
------------------------------

**"stars ... will be treated as having zero luminosity"**: printed at
startup if the IMF's minimum mass is lower than the lowest mass covered by
the selected track set, e.g.:

::

    slug: warning: minimum mass in selected tracks is 0.1 but IMF minimum
    mass is 0.08; stars with masses from 0.08 to 0.1 will be treated as
    having zero luminosity

This is a warning, not an error -- the run continues, but any star drawn in
that mass gap contributes no light, yield, or feedback to the output. If
this matters for your science case, either pick a track set with wider mass
coverage or restrict the IMF's own range; see :ref:`sec-tracks` for the
mass range of each standard track set.

**SLUG exits with an error partway through spectral synthesis**: this means
a star's properties (temperature, gravity, composition) fell outside the
coverage of every atmosphere model in the chain given by ``spectra.model``.
The default chain (``model = "default"``) is assembled to cover essentially
the full range produced by the standard tracks, so this is most likely if
you've selected a single non-default atmosphere library, or a non-standard
track set with more extreme properties than the defaults expect -- see
:ref:`ssec-atmospheres-alternative` for how the model chain is searched and
what to do about gaps in coverage.

**A run appears to hang or produce no output**: increase ``verbosity`` in
the input deck (see :ref:`ssec-parameters-toplevel`). At ``verbosity = 1``
SLUG prints a line when the run starts and finishes; at ``verbosity = 2`` it
also prints a line every time an individual Monte Carlo trial starts, which
is usually enough to tell whether the run is actually progressing or has
stalled (e.g. waiting on a slow filesystem for output).

Restarting and Checkpointed Runs
-----------------------------------

``build/slug --restart``/``-R`` (or ``restart=True`` from Python) only works
with HDF5 output that was actually checkpointed -- see the "Checkpointing
and Batch Runs" section of :ref:`sec-running` for how to enable
checkpointing and restart from it. Common mistakes:

* Passing ``--restart`` to an ``ascii``-output run: rejected immediately
  with ``slug: --restart/-R is not supported with ascii output``, since
  ASCII output has no checkpoint to resume from.
* Restarting into a fresh output directory, or after moving/renaming the
  original output file: SLUG resumes by finding the most recent checkpoint
  *in the output file named by the current input deck's own*
  ``model_name``/``out_dir``, so a restart only works if those still point
  at the original run's own output.

Cloudy Post-Processing
--------------------------

Running ``slug_reader.run_cloudy`` (see :ref:`sec-cloudy-slug`) needs a
working Cloudy installation, located either via an explicit path or the
``CLOUDY_DIR`` environment variable. If neither is set correctly, you'll see
one of:

::

    find_cloudy_executable: no cloudy_path given, and CLOUDY_DIR is not set
    in the environment

::

    find_cloudy_executable: no executable cloudy binary found at <path>

The first means ``CLOUDY_DIR`` was never set (and no explicit path was
passed); the second means it *was* set, but ``$CLOUDY_DIR/cloudy.exe``
either doesn't exist or isn't executable -- double-check that ``CLOUDY_DIR``
points at the directory containing the compiled ``cloudy.exe`` (Cloudy's
``source`` directory after building it), not Cloudy's top-level install
directory.

Getting Further Help
-----------------------

If a problem doesn't match anything above, :ref:`sec-tests` describes how to
run SLUG's own test suite, which can help distinguish a local build/data
problem (tests fail) from a problem specific to your own input deck (tests
pass). Include the exact error message, your input deck, and the output of
``ctest -L quick`` when reporting an issue.
