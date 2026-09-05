.. highlight:: rest

.. _sec-output:

Output Files and Format
=======================

Controlling Outputs
-------------------

A SLUG simulation produces output files whose names, locations, and types are
controlled by the ``model_name``, ``out_dir``, and ``output_format`` keywords in
the parameter file.

The outputs are written in either HDF5 or ASCII format, depending on whether
``output_format`` is set to ``"h5"`` (or ``"h5divided"``) or ``ascii`` -- see
:ref:`sec-parameters` for details. ASCII format is intended for small, quick
simulations with small outputs, and is not recommended for large, production-scale
runs. HDF5 format is recommended for all but the smallest runs, and is the only
format that the Python interface can read, as detailed below. The ``"h5divided"``
option produces outputs that are identical to ``"h5"`` except that the outputs
from different threads are are left in separate HDF5 files rather than consolidated
into a single one, which is preferable for very large runs where a monolithic
HDF5 file would be unwieldy; the ``slugpy`` interface can read both formats
transparently.

Outputs are placed in the directory specified by ``out_dir`` in the parameter
file, which defaults to the current working directory if not specified. 
The output file name is constructed from the ``model_name`` keyword in the parameter
file. For ``h5`` output, the output file is named ``<model_name>.h5``, while for
``h5divided`` output the output files are placed in a directory named ``<model_name>``,
with each thread's output stored in a separate file in that directory called
``thread_<NNNNN>.h5`` where ``<NNNNN>`` is the thread number. For ASCII output,
there will be multiple file outputs, whose names follow the pattern
``<model_name>_<output_type>.txt``. Which output types are produced is controlled
by the keywords in the ``[output]`` section of the parameter file; 
see :ref:`sec-parameters` for details.

A full description of the format of the outputs is provided below.

Reading Outputs
---------------

Outputs in ASCII mode are simple text files that can be read with any text editor,
and should be self-explanatory (though full details are provided below). To read
outputs in HDF5 format, you can use any HDF5 reader, but the recommended approach
is to use ``slugpy``. A simulation output in HDF5 format can be read into Python
simply by doing::

    import slugpy
    sim_result = slugpy.read("path/model_name")

The reader will automatically detect whether the output format is ``h5`` or ``h5divided``
and read the outputs appropriately. The result returned by ``slugpy.read`` is a
``slugpy.slug_reader`` object, which is a lazy-reader that can be used safely even for
very large outputs that will not fit in memory at once. See :ref:`sec-slugpy` for
details on how to use the ``slug_reader`` object to inspect the outputs of a simulation.

Full Documentation of the SLUG Output Format
--------------------------------------------

Below is a full listing of every dataset and attribute that SLUG's HDF5 and ASCII
output can contain. Which of these actually appear in a given run's own output
depends on the parameter file used to produce it -- e.g. ``cluster_phot`` (or
``<model_name>_cluster_phot.txt``) only exists if photometry was requested, and a
cluster-type simulation never produces any of the ``galaxy*`` outputs at all,
since it has no ``Galaxy`` object -- see :ref:`sec-parameters` for the keywords
that control this.

HDF5 Format
~~~~~~~~~~~

Every dataset described below also has its own ``units`` attribute (a string,
e.g. ``"Msun"``, or an empty string for a dimensionless quantity); rather than
list this attribute separately under every single dataset, its value is simply
given alongside each dataset below.

Top-level Attributes
^^^^^^^^^^^^^^^^^^^^^

These attributes are set directly on the output file itself, rather than on any
group within it.

* ``slug-hash``: The git commit hash of the SLUG build that produced this file.
* ``date``: The local date (``YYYY-MM-DD``) on which this file was created.
* ``time``: The local time (``HH:MM:SS``) at which this file was created.
* ``rng_state``: The serialized state of the random number generator at the
  moment this file was opened.
* ``trials_completed``: The number of trials actually completed and written to
  this file (or, for one checkpoint of a checkpointed run, to that checkpoint)
  by the time it was closed.
* ``restart_uid``: An internal identifier used to keep cluster/star ID
  numbering consistent across a run resumed with ``--restart``; not generally
  meaningful on its own.
* ``max_trial``: The largest trial number actually written to this file. Used
  internally when resuming a run from a checkpoint, so that trial numbering in
  the resumed session never collides with a trial number a previous session
  already used.

The ``input_deck`` Group
^^^^^^^^^^^^^^^^^^^^^^^^^

Holds a single dataset recording the parameter file that produced this
simulation.

* ``toml`` (string scalar): The full text of the TOML input deck used for this
  run, re-serialized -- lets the exact parameters behind a given output file be
  recovered from the file itself. Empty if the ``SimControls`` driving this run
  was built directly from Python with no backing input deck at all.

The ``clusters`` Group
^^^^^^^^^^^^^^^^^^^^^^^

Holds one row per individually-tracked (stochastic) star cluster ever formed
across every trial, whether or not that cluster survived to be observed at any
output time. For a cluster-type simulation this is exactly one row per trial
(each trial forms exactly one cluster); for a galaxy-type simulation, the
number of rows varies from trial to trial, depending on the star formation
history and the cluster mass function. Every dataset below has shape
``(n_clusters,)``, where ``n_clusters`` is this total row count.

* ``trial`` (unitless integer): The trial number this cluster belongs to.
* ``uid`` (unitless integer): This cluster's unique identifier.
* ``target_mass`` (Msun): The cluster mass drawn from the cluster mass
  function.
* ``birth_mass`` (Msun): The cluster's actual initial stellar mass, as realized
  by stochastically sampling individual stars from the IMF -- generally close
  to, but not exactly equal to, ``target_mass``, since a discrete set of
  stellar masses cannot sum to an arbitrary target exactly.
* ``form_time`` (yr): The time at which this cluster formed. Always 0 for a
  cluster-type simulation, which forms its single population instantaneously.
* ``feh`` (unitless): This cluster's [Fe/H] metallicity.
* ``rng`` (string): The serialized state of the random number generator at the
  moment this cluster was drawn, so that its own stochastic realization can be
  reproduced independently of the rest of the run.
* ``A_V`` (mag; only present if ``extinct.model`` was set): The V-band
  extinction applied to this cluster.

The ``cluster_spectra`` Group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Holds the spectra of individually-tracked clusters. There is one row (in every
extensible dataset below) per (cluster, output time) pair at which a spectrum
was actually written; call this common row count ``n_rows``.

Fixed wavelength/line grids, shared by every row:

* ``wl`` (Angstrom, shape ``(n_wl,)``): The rest-frame wavelength grid that
  every row's ``spec``/``spec_neb`` value is tabulated on.
* ``wl_extinct`` (Angstrom, shape ``(n_wl_extinct,)``; only if ``extinct.model``
  was set): The wavelength grid for ``spec_extinct``/``spec_neb_extinct`` --
  may cover a narrower range than ``wl``, since an extinction curve need not
  extend across the full spectral grid.
* ``line_wl`` (Angstrom, shape ``(n_lines,)``; only if a nebular emission grid
  was requested): The rest-frame wavelength of each nebular emission line
  tabulated in ``neb_lines``/``neb_lines_extinct``.
* ``line_label`` (string, shape ``(n_lines,)``; only if a nebular emission grid
  was requested): A human-readable label (species/transition) identifying each
  entry of ``line_wl``, in the same order.

Per-row datasets:

* ``trial`` (unitless integer, shape ``(n_rows,)``): The trial number this row
  belongs to.
* ``time`` (yr, shape ``(n_rows,)``): The output time this spectrum was
  computed at.
* ``uid`` (unitless integer, shape ``(n_rows,)``): The unique identifier of the
  cluster this spectrum belongs to.
* ``spec`` (erg/(s Angstrom), shape ``(n_rows, n_wl)``): The cluster's
  un-extincted spectrum.
* ``spec_extinct`` (erg/(s Angstrom), shape ``(n_rows, n_wl_extinct)``; only if
  ``extinct.model`` was set): The same spectrum after applying dust extinction.
* ``spec_neb`` (erg/(s Angstrom), shape ``(n_rows, n_wl)``; only if a nebular
  emission grid was requested): The un-extincted spectrum including the
  nebular continuum.
* ``spec_neb_extinct`` (erg/(s Angstrom), shape ``(n_rows, n_wl_extinct)``;
  only if both a nebular emission grid and ``extinct.model`` were requested):
  The nebular-inclusive spectrum after applying dust extinction.
* ``neb_lines`` (erg/s, shape ``(n_rows, n_lines)``; only if a nebular emission
  grid was requested): The luminosity of each nebular emission line listed in
  ``line_wl``/``line_label``, without extinction applied.
* ``neb_lines_extinct`` (erg/s, shape ``(n_rows, n_lines)``; only if both a
  nebular emission grid and ``extinct.model`` were requested): The same line
  luminosities after applying dust extinction.

The ``cluster_phot`` Group
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Holds the photometry of individually-tracked clusters, present only if
photometry (``phot.filters``) or the bolometric luminosity was requested.
There is one row per (cluster, output time) pair at which photometry was
actually written; call this row count ``n_rows``.

* ``filters`` (attribute, array of strings): The ordered list of filter names
  corresponding to the columns of ``phot`` (and, where present,
  ``phot_extinct``/``phot_neb``/``phot_neb_extinct``, excluding their own final
  entry -- see below). Includes ``"Lbol"`` as its own final entry if the
  bolometric luminosity was requested.
* ``trial`` (unitless integer, shape ``(n_rows,)``): The trial number.
* ``time`` (yr, shape ``(n_rows,)``): The output time this photometry was
  computed at.
* ``uid`` (unitless integer, shape ``(n_rows,)``): The unique identifier of the
  cluster this photometry belongs to.
* ``phot`` (shape ``(n_rows, n_filters)``; units given per-column by this
  dataset's own ``units`` attribute, an array of strings matching ``filters``,
  since different filters can report in different units): The un-extincted
  photometry in every requested filter, including a final ``"Lbol"`` column if
  the bolometric luminosity was requested.
* ``phot_extinct`` (shape ``(n_rows, n_real_filters)``; only if ``extinct.model``
  was set and at least one real filter was requested): The same photometry
  after applying dust extinction. Never includes an ``"Lbol"`` column: the
  bolometric luminosity is computed directly from the stellar tracks, not
  from the (extincted or nebular-inclusive) spectrum, so it has no extincted
  counterpart -- ``n_real_filters`` excludes it even when ``phot`` itself
  includes it.
* ``phot_neb`` (shape ``(n_rows, n_real_filters)``; only if a nebular emission
  grid was requested and at least one real filter was requested): The
  un-extincted photometry including the nebular continuum's contribution.
* ``phot_neb_extinct`` (shape ``(n_rows, n_real_filters)``; only if a nebular
  emission grid, ``extinct.model``, and at least one real filter were all
  requested): The nebular-inclusive photometry after applying dust extinction.

The ``galaxy`` Group
^^^^^^^^^^^^^^^^^^^^^

Present only for a galaxy-type simulation. Holds one row per (trial, output
time) pair actually written; call this row count ``n_rows`` (ordinarily
``n_trial`` times the number of output times per trial, though a stochastic
output-time distribution can make different trials write different numbers of
rows -- see :ref:`sec-parameters`). Every dataset below has shape
``(n_rows,)``.

* ``trial`` (unitless integer): The trial number.
* ``time`` (yr): The output time this row was recorded at.
* ``target_mass`` (Msun): The total stellar mass the galaxy's own star
  formation history says should have formed by this time.
* ``actual_mass`` (Msun): The total stellar mass actually realized (summed
  over every individually-tracked cluster and the continuous stellar
  population, if any) by this time.

The ``galaxy_spectra`` Group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Present only for a galaxy-type simulation with a spectral synthesizer
requested. Identical in structure and meaning to the ``cluster_spectra`` group
above -- including its ``wl``/``wl_extinct``/``line_wl``/``line_label``,
``spec``/``spec_extinct``/``spec_neb``/``spec_neb_extinct``, and
``neb_lines``/``neb_lines_extinct`` datasets, all with the same meaning -- but
with no ``uid`` dataset, since a galaxy has no individual identity the way a
cluster does: each row is one (trial, output time) pair (the whole galaxy's
own integrated spectrum), not one (cluster, output time) pair.

The ``galaxy_phot`` Group
^^^^^^^^^^^^^^^^^^^^^^^^^^

Present only for a galaxy-type simulation with photometry or the bolometric
luminosity requested. Identical in structure and meaning to the
``cluster_phot`` group above -- including its ``filters`` attribute and
``phot``/``phot_extinct``/``phot_neb``/``phot_neb_extinct`` datasets -- but
with no ``uid`` dataset: each row is one (trial, output time) pair (the whole
galaxy's own integrated photometry), not one (cluster, output time) pair.

ASCII Format
~~~~~~~~~~~~

Every ASCII output file begins with two header rows (column names, then column
units) and a row of dashes, followed by one data row per line, with columns
laid out as fixed-width fields. Where a group above splits one physical
quantity across several HDF5 datasets of different shapes (e.g. a 2D ``spec``
array alongside 1D ``trial``/``time``/``uid`` arrays), the corresponding ASCII
file instead lays out one row per (cluster or trial, output time, wavelength or
line) combination, repeating the ``trial``/``time``/``uid`` columns on every
such row -- there is no way to represent a ragged, multi-dimensional structure
in a flat text table. Because of this, spectra and nebular-line files are
generally far longer, row for row, than their HDF5 counterparts' own row
counts would suggest.

The ``<model_name>_summary.txt`` File
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Always written, exactly once, regardless of what other output was requested.
Holds the same top-level information the HDF5 format stores as file
attributes, as plain ``key  value`` lines, followed by the input deck's own
text in full:

* ``slug-hash``: The git commit hash of the SLUG build that produced this run.
* ``date``: The local date (``YYYY-MM-DD``) this run started.
* ``time``: The local time (``HH:MM:SS``) this run started.
* ``rng_state``: The serialized state of the random number generator at the
  start of the run.
* ``input_deck``: A label line, followed by the full text of the TOML input
  deck used for this run. Unlike HDF5's ``trials_completed``/``restart_uid``/
  ``max_trial`` attributes, ASCII output has no corresponding fields, since
  checkpointing (and, with it, ``--restart``) is only supported with HDF5
  output.

The ``<model_name>_clusters.txt`` File
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

One row per individually-tracked cluster -- see the ``clusters`` HDF5 group
above for exactly which clusters that means and what each column means.
Columns, in order: ``trial``, ``uid``, ``target_mass`` (Msun), ``birth_mass``
(Msun), ``form_time`` (yr), ``feh``, ``A_V`` (mag; only if ``extinct.model``
was set), ``rng``.

The ``<model_name>_cluster_spectra.txt`` File
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

One row per (cluster, output time, wavelength) combination -- i.e. every
wavelength point of every spectrum gets its own row. Columns, in order:
``trial``, ``time`` (yr), ``uid``, ``wl`` (Angstrom), ``spec``
(erg/(s Angstrom)), ``spec_ex`` (erg/(s Angstrom); only if ``extinct.model``
was set), ``spec_neb`` (erg/(s Angstrom); only if a nebular emission grid was
requested), ``spec_neb_ex`` (erg/(s Angstrom); only if both were requested).
See the ``cluster_spectra`` HDF5 group above for what each column means --
``spec_ex``/``spec_neb``/``spec_neb_ex`` here correspond to that group's own
``spec_extinct``/``spec_neb``/``spec_neb_extinct`` datasets.

The ``<model_name>_cluster_neb_lines.txt`` File
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Only written if a nebular emission grid was requested. One row per (cluster,
output time, emission line) combination. Columns, in order: ``trial``,
``time`` (yr), ``uid``, ``line_label`` (identifying which emission line this
row is), ``wl`` (Angstrom, that line's rest-frame wavelength), ``line_lum``
(erg/s, that line's luminosity without extinction), ``line_lum_ex`` (erg/s;
only if ``extinct.model`` was set, that line's luminosity with extinction
applied). Corresponds to the ``cluster_spectra`` HDF5 group's own
``line_wl``/``line_label``/``neb_lines``/``neb_lines_extinct`` datasets.

The ``<model_name>_cluster_phot.txt`` File
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Only written if photometry or the bolometric luminosity was requested. One row
per (cluster, output time) pair. Columns, in order: ``trial``, ``time`` (yr),
``uid``, then one column per requested filter (named after the filter itself,
e.g. ``HST.WFC3_UVIS1.F555W``, or ``Lbol`` if the bolometric luminosity was
requested), giving the un-extincted photometry. If ``extinct.model`` was set,
one further ``<filter>_ex`` column follows per real filter (excluding
``Lbol``, which has no extincted counterpart); if a nebular emission grid was
requested, one further ``<filter>_neb`` column follows per real filter; if
both were requested, one further ``<filter>_neb_ex`` column follows per real
filter. See the ``cluster_phot`` HDF5 group above for what each of these
means -- ``<filter>_ex``/``<filter>_neb``/``<filter>_neb_ex`` here correspond
to that group's own ``phot_extinct``/``phot_neb``/``phot_neb_extinct``
datasets.

The ``<model_name>_galaxy.txt`` File
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Only written for a galaxy-type simulation. One row per (trial, output time)
pair. Columns, in order: ``trial``, ``time`` (yr), ``target_mass`` (Msun),
``actual_mass`` (Msun) -- see the ``galaxy`` HDF5 group above for what each
means.

The ``<model_name>_galaxy_spectra.txt`` File
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Only written for a galaxy-type simulation with a spectral synthesizer
requested. Identical to ``<model_name>_cluster_spectra.txt`` above -- same
columns and meanings -- except with no ``uid`` column, and one row per
(trial, output time, wavelength) combination rather than (cluster, output
time, wavelength), since each row describes the whole galaxy's own integrated
spectrum.

The ``<model_name>_galaxy_neb_lines.txt`` File
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Only written for a galaxy-type simulation with a nebular emission grid
requested. Identical to ``<model_name>_cluster_neb_lines.txt`` above, except
with no ``uid`` column, and one row per (trial, output time, emission line)
combination rather than (cluster, output time, emission line).

The ``<model_name>_galaxy_phot.txt`` File
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Only written for a galaxy-type simulation with photometry or the bolometric
luminosity requested. Identical to ``<model_name>_cluster_phot.txt`` above,
except with no ``uid`` column, and one row per (trial, output time) pair
describing the whole galaxy's own integrated photometry rather than one
individual cluster's.


