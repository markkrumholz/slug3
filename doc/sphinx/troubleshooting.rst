.. highlight:: rest

.. _sec-troubleshooting:

Troubleshooting
================

This page collects fixes for problems users commonly run into, organized by
where they show up: building SLUG, locating its data files, writing an input
deck, setting up nucleosynthetic yields, running a simulation, and
post-processing with Cloudy. If your problem
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

.. _ssec-troubleshooting-yields:

Yield Errors
----------------

The ``[yields]`` section of the input deck (see :ref:`ssec-parameters-yields`) is
checked at startup, so a mistake in it stops SLUG before any simulation runs, with a
message that names the problem. The common ones are below; see :ref:`sec-yields` for
the yield models available and what each covers.

**The requested [Fe/H] is outside the range of the yield model**:

::

    slug: simulation failed: YieldChannel: requested [Fe/H] range [-0.500000,
    -0.500000] extends outside the range available for channel 'ccsn', model
    'sukhbold16' ([0.000000, 0.000000])

The [Fe/H] of the simulated population (``stars.FeH``) must lie within the range
of [Fe/H] that the yield model tabulates, and the message reports both the range
you asked for and the range the model provides. A model tabulated at a single [Fe/H]
can only be used with exactly that value: ``sukhbold16``, for example, is available
only at [Fe/H] = 0. Either change ``stars.FeH`` or choose a model whose range covers
it (see :ref:`ssec-yield-models` for the range of each standard model).

**The registry has no such model**:

::

    slug: simulation failed: YieldChannel: registry .../data/yields/yields.toml has
    no model 'kobayashi06_11' for channel 'massive_star_winds'

Either the model name is misspelled, or the model is not available for the channel
you paired it with -- the registry lists the models for each channel separately, and
some models (like ``kobayashi06_11``, which has supernova yields only) provide only one.
The registry named in the message is the one that was actually searched, which is
useful if you have set ``yields.registry`` to a custom one. If the registry file itself
cannot be found (``YieldChannel: registry ... not found``), the same file-resolution
rules apply as for the data files described under "Data Files Not Found" above.

**A channel or model is missing or not recognized**:

::

    SimControls: yields.channel1.channel = 'supernova' is not a recognized yield channel
    getTOMLKeyWithError: required key yields.channel1.model not found

Every ``[yields.channelN]`` table needs both a ``channel`` and a ``model``, and
``channel`` must be one of the channels described in :ref:`sec-yields`. Note also that
channels must be numbered consecutively from 1: SLUG stops at the first missing
number and *silently ignores* any later table, so a ``[yields.channel3]`` with no
``[yields.channel2]`` produces no error, but also contributes nothing to the output.
If a channel you expected is missing from the output, check the numbering.

**Problems with the isotope list**:

::

    Yields::rebuildYieldGrid: the given isotopes list does not intersect any isotope
    tabulated by any loaded channel

    SimControls: yields.isotopes entry 'H' is not a valid isotope specifier (expected
    an element symbol followed by a mass number, e.g. 'H1', 'Na22', or 'fe56')

    SimControls: yields.isotopes entry 'Xx12' names an unrecognized element symbol 'Xx'

    SimControls: yields.isotopes entry 'Fe999' does not correspond to a known isotope
    in the isotope table

The first means that none of the isotopes you listed is tabulated by the yield models
you selected, or is a decay product of an isotope they tabulate -- a typo is
the most likely cause, but the isotope may simply not be one of those the model provides.
The others mean that an entry in ``yields.isotopes`` is not a valid isotope name: each
must be an element symbol followed by a mass number, with no space between them. A valid
isotope that the models do not tabulate is not an error as long as at least one other entry
matches. See the description of ``isotopes`` in :ref:`ssec-parameters-yields`.

**The mass range of a channel is invalid**:

::

    YieldChannel::rebuildYieldGrid: mMin (30.000000) must be strictly less than mMax
    (20.000000)

``m_min`` and ``m_max`` must both be positive, and ``m_min`` must be less than
``m_max``. Remember that an omitted ``m_max`` defaults to the highest mass tabulated by
the model, so setting ``m_min`` above that mass produces this error unless you also give a
larger ``m_max``.

**Yields are requested but would never be written**:

::

    SimControls: yield channels were given, but output.write_cluster_yields and
    output.write_galaxy_yields are both false (or this is a cluster-type simulation
    and output.write_cluster_yields is false), so the computed yields would never be
    written

You have defined at least one yield channel but turned off every output that would
contain the result. For a cluster simulation ``output.write_cluster_yields`` must be
true; for a galaxy simulation at least one of ``output.write_cluster_yields`` and
``output.write_galaxy_yields`` must be. Remove the yield channels if you do not actually
want yields, or see :ref:`ssec-parameters-output`.

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
