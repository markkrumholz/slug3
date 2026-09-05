.. highlight:: rest

.. _sec-parameters:

The SLUG Input Deck
====================

SLUG command-line simulations are controlled by an input deck, and input decks can
also be used to control SLUG when it is run as a Python module. The input deck is a
`TOML <https://toml.io/en/>`_ file which contains human-readable key-value pairs
that control various aspects of the simulation. The input deck is read by SLUG at
runtime when SLUG is run in command-line mode, and the :ref:`slugpy` Python module
provides a class to read input decks which can then be used to control the simulation
in Python -- see :ref:`sec-running` for details.

The input deck contains some top-level key-value pairs that control basic operations
of the simulation, followed by a series of sections that control either some aspect
of simulation physics or some part of the simulation operation. Some of these keywords
and sections are required, while others are optional and have default values. Some
example input decks can be found in the ``examples`` directory of the repository.

Below is a full section-by-section listing of the input deck keywords and their meanings.

Numerical Values vs. Distributions vs. Files
---------------------------------------------

Many of the keywords below (the IMF, CMF, CLF, [Fe/H], star formation rate, and
V-band extinction) describe a quantity that SLUG treats as a probability
distribution function (PDF) rather than a fixed value -- see :ref:`sec-pdfs` for
the full set of distributions SLUG understands and how to describe them in a file
of their own. For all such keywords, the corresponding input deck value may be
given in either of two forms:

* A bare number, which is interpreted as a delta function at that value (i.e., the
  quantity is fixed, not stochastic).
* A string naming a PDF descriptor file, which is parsed to construct the
  corresponding distribution.

Unless noted otherwise, a file name given for one of these keywords (or for
``stars.tracks``, ``spectra.model``, ``phot.filters``, ``extinct.model``, and the
various ``*_registry``/``registry`` overrides below) is resolved by first checking
for a file of that name in the current working directory; if none is found and the
name is not an absolute path, SLUG next checks for the file under the directory
named by the ``SLUG_DIR`` environment variable (if set), and finally under the
repository's own top-level directory. Most of these keywords resolve directly
under that search root, but a few (``stars.IMF``, and the internal defaults for
track/spectral/filter/extinction registries) are resolved under a fixed
subdirectory of ``data/`` instead -- this is noted individually below where it
applies.

Top-level Keywords
------------------

These keywords must appear at the top level of the input deck, outside of any section.
They control basic aspects of the simulation, such as the type of simulation to run
and the number of trials to perform.

* ``sim_type`` (required): The type of simulation to run. Must be either ``"cluster"``
  (a single simple stellar population, formed all at once) or ``"galaxy"`` (a composite
  population, built up over time from a star formation history) -- see
  :ref:`sec-running` for the distinction.
* ``n_trial`` (optional, default=1): The number of independent Monte Carlo trials to run.
* ``verbosity`` (optional, default=0): The level of diagnostic output SLUG prints while running.
* ``rng_seed`` (optional): An integer seed for SLUG's random number generator. If not
  given, SLUG seeds itself from the operating system's own entropy source, so
  repeated runs of the same input deck will not produce identical results unless a
  fixed seed is given.


Output Control Keywords
-----------------------

These keywords, in the ``[output]`` section, control what outputs are produced by
the simulation, how they are formatted, and where they are written.

* ``model_name`` (optional, default="slug_sim"): The base name used for this
  model's output file(s).
* ``output_times``: Specifies the simulation time(s), in yr, at which output should
  be written. This may be given either as a number/PDF (see above) -- in which case
  every trial draws its own single output time from it -- or as an explicit array of
  times, in which case every trial produces output at each of the listed times.
  Exactly one of ``output_times`` or the ``start_time`` / ``end_time`` / ``ntime`` trio
  below must be given.
* ``start_time``, ``end_time``, ``ntime``: An alternative to ``output_times``, these
  three keywords (which must all be given together) specify a grid of ``ntime``
  output times running from ``start_time`` to ``end_time`` (in yr), shared by every
  trial. ``start_time`` and ``end_time`` may only be equal if ``ntime`` = 1, and vice versa.
* ``log_time`` (optional, default=false): If true, the ``start_time`` / ``end_time`` / ``ntime``
  grid above is spaced logarithmically rather than linearly; requires ``start_time`` > 0.
* ``write_cluster`` (optional, default=true): Whether to write basic properties for
  each individual star cluster.
* ``write_cluster_spec`` (optional, default=true): Whether to write the spectrum of
  each individual star cluster; only meaningful if spectral synthesis has been requested
  (see ``spectra.model`` below).
* ``write_cluster_phot`` (optional, default=true): Whether to write the photometry
  of each individual star cluster; only meaningful if photometry has been requested
  (see ``phot.filters`` below).
* ``write_galaxy`` (optional, default=true; galaxy simulations only): Whether to
  write basic properties for the galaxy as a whole.
* ``write_galaxy_spec`` (optional, default=true; galaxy simulations only): Whether
  to write the integrated spectrum of the whole galaxy.
* ``write_galaxy_phot`` (optional, default=true; galaxy simulations only): Whether
  to write the integrated photometry of the whole galaxy.
* ``output_mode`` (optional, default="h5"): The output file format. Must be one of
  ``h5`` / ``hdf5`` (a single, consolidated HDF5 file), ``h5divided`` / ``hdf5divided``
  (HDF5 output, but skipping consolidation when SLUG is built with OpenMP, leaving
  one file per thread), or ``ascii`` / ``txt`` (plain-text output).
* ``out_dir`` (optional, default=""): The directory into which output should be
  written. An empty string (the default) writes into the current working directory.
* ``checkpoint_interval`` (optional, default=0): The number of trials between
  checkpoints; 0 disables checkpointing. Only supported when ``output_mode`` is
  ``h5`` or ``h5divided`` -- a non-zero value combined with ``ascii`` output is an error.

At least one of the ``write_*`` keywords relevant to the simulation's own
``sim_type`` must be left true, and if ``phot.filters`` was given, at least one of
``write_cluster_phot`` / ``write_galaxy_phot`` must be true -- otherwise SLUG raises
an error at startup, since the corresponding output(s) would otherwise never be written.


Integrator Control Keywords
----------------------------

These optional keywords, in the ``[integrator]`` section, control the numerical
tolerances SLUG's adaptive PDF integrator uses wherever it must integrate over a
continuous (non-stochastic) population. The defaults are chosen to be reasonable
for typical SLUG simulations and rarely need to be changed.

* ``rel_tol`` (optional, default=1e-2): The integrator's relative error tolerance.
* ``abs_tol`` (optional, default=1e-3): The integrator's absolute error tolerance.
* ``max_iter`` (optional, default=524288, i.e. 2^19): The maximum number of
  integrand evaluations the integrator is allowed before giving up; 0 means unlimited.


Stellar Population Keywords
-----------------------------

These keywords, in the ``[stars]`` section, describe the stellar population itself
-- the initial mass function, the stellar evolutionary tracks, and related
composition/rotation settings.

* ``IMF`` (required): The stellar initial mass function, as a number or PDF (see
  above). A file name is resolved under ``data/imfs`` (in addition to the usual
  current-working-directory/``SLUG_DIR``/repository search).
* ``tracks`` (required): The name of the stellar evolutionary track set to use
  (e.g. ``"MIST"``), as listed in the track registry (see ``track_registry`` below).
* ``track_registry`` (optional, default="data/tracks/tracks.toml"): Overrides the
  default registry file used to look up ``tracks``.
* ``FeH`` (required): The stellar metallicity, [Fe/H], as a number or PDF (see above).
* ``v_vcrit`` (optional, default=0.0): The stellar rotation rate, as a fraction of
  the critical (breakup) velocity; shared by the tracks and, if used, the nebular
  emission grid.
* ``alphaFe`` (optional, default=0.0): The stellar [alpha/Fe] abundance ratio;
  shared by the tracks and the spectral synthesis model.
* ``CFe`` (optional, default=0.0): The stellar [C/Fe] abundance ratio; used only by
  the spectral synthesis model (the tracks have no corresponding axis).
* ``min_stoch_mass`` (optional, default=0.0): The minimum stellar mass treated
  fully stochastically; stars below this mass are instead drawn from the
  continuous part of the IMF.


Cluster Keywords
-------------------

These keywords, in the ``[clusters]`` section, describe the population of star
clusters formed by the simulation.

* ``CMF`` (required): The cluster mass function, as a number or PDF (see above).
* ``CLF`` (required for galaxy simulations; unused for cluster simulations): The
  cluster lifetime function, as a number or PDF (see above).
* ``f_cluster`` (optional, default=1.0; galaxy simulations only): The fraction of a
  galaxy simulation's stellar mass formed in individually-tracked, stochastic
  clusters; the remainder is treated as continuously-distributed in time.


Galaxy Keywords
-------------------

These keywords, in the ``[galaxy]`` section, describe the star formation history of
a galaxy simulation, and are only used (and only permitted) when ``sim_type`` =
``"galaxy"``. Exactly one of ``sfr`` / ``sfr_dist`` must be given.

* ``sfr``: The star formation rate as a function of time. A bare number gives a
  constant rate, in Msun/yr, shared by every trial; a string names a PDF descriptor
  file giving the (possibly time-varying) rate directly -- unlike every other
  file-valued keyword described above, this file name is resolved relative to the
  current working directory only, not via the ``SLUG_DIR``/repository search.
* ``sfr_dist``: A distribution (number or PDF, as above) from which each trial
  independently draws its own single, constant star formation rate, in Msun/yr.
  Unlike ``sfr``, a bare number here is treated as an ordinary delta function, as
  for every other PDF-valued keyword.


Spectral Synthesis Keywords
------------------------------

These keywords, in the ``[spectra]`` section, control spectral synthesis. The
entire section is optional: if ``model`` is not given, no spectra (and, since
photometry and extinction both require a spectral synthesizer, no photometry or
extinction) are computed for this simulation.

* ``model`` (required to enable spectral synthesis): The spectral synthesis
  model(s) to use. May be ``"blackbody"`` (approximate every star as a blackbody,
  needing no stellar atmosphere library), ``"default"`` (a standard, physically
  comprehensive chain of atmosphere libraries covering the whole HR diagram, from
  Wolf-Rayet winds through white dwarf atmospheres), the name of a single atmosphere
  library from the spectral registry (see ``registry`` below), or an array of such
  library names, in which case the libraries are chained together, each covering
  whichever stars fall outside the coverage of the ones before it.
* ``registry`` (optional, default=``data/spectra/spectra.toml``): Overrides the
  default registry file used to look up ``model``.
* ``wl_min``, ``wl_max``, ``nwl`` (optional; must be given together): Override the
  default output wavelength grid (roughly 91 to 1e5 Angstrom at 2048 points) with a
  grid of ``nwl`` points spanning ``wl_min`` to ``wl_max`` Angstrom.
* ``z`` (optional, default=0.0): A redshift applied to every computed spectrum.


Photometry Keywords
-----------------------

These keywords, in the ``[phot]`` section, control the computation of photometry
from the synthesized spectra. The entire section is optional: if ``filters`` is not
given (or names no filters), no photometry is computed. Using this section requires
spectral synthesis to also be enabled (see ``spectra.model`` above).

* ``filters`` (required to enable photometry): The filter(s) to compute photometry
  in, as a string (for a single filter) or array of strings. Most filter names take
  the form ``facility.instrument.filter`` (or ``facility.filter`` where a facility
  has only one instrument), as listed in the filter registry (see ``registry``
  below); idealized filters (e.g. ionizing photon rates) are also available, named
  ``Q(...)``. The special name ``"Lbol"`` requests the bolometric luminosity, which
  is computed directly from the spectrum rather than through the ordinary filter
  machinery.
* ``system`` (optional, default="Flambda"): The photometric system to report
  magnitudes/fluxes in. Must be one of ``Flambda``, ``Fnu``, ``ST``, ``AB``, or ``Vega``.
* ``registry`` (optional, default="data/filters/filters.toml"): Overrides the
  default registry file used to look up ``filters``.
* ``vega`` (optional): Overrides the default reference Vega spectrum file used by
  the ``Vega`` photometric system.


Extinction Keywords
-----------------------

These keywords, in the ``[extinct]`` section, control dust extinction applied to
the synthesized spectra and photometry. The entire section is optional: if neither
``AV`` nor ``AV_field`` is given, no extinction is applied. Using this section
requires spectral synthesis to also be enabled (see ``spectra.model`` above).

* ``AV`` (optional): The V-band extinction, in mag, applied to stars in
  individually-tracked clusters, as a number or PDF (see above).
* ``AV_field`` (optional; galaxy simulations only): The V-band extinction applied
  to the continuous ("field") stellar population, as a number or PDF. If ``AV`` is
  given but ``AV_field`` is not (or vice versa), the missing one defaults to a
  fixed extinction of zero, rather than being left unset.
* ``model`` (required if ``AV`` or ``AV_field`` is given): The name of the
  extinction curve to apply, as listed in the extinction registry (see ``registry`` below).
* ``registry`` (optional, default="data/extinct/extinct.toml"): Overrides the
  default registry file used to look up ``model``.


Nebular Emission Keywords
-----------------------------

These keywords, in the ``[nebular]`` section, control the contribution of nebular
(ionized gas) emission to the computed spectra and photometry. Unlike most of the
sections above, every keyword here is independently optional and simply falls back
to its own default if omitted.

* ``compute_neb`` (optional, default=true): Whether nebular emission is computed at
  all. If false, every other keyword in this section is ignored.
* ``log_U`` (optional, default=-2.5): The base-10 logarithm of the ionization parameter.
* ``cov_fac`` (optional, default=0.5): The nebular covering factor (the fraction of
  ionizing photons that actually ionize gas within the covering material, as
  opposed to escaping).
* ``line_width`` (optional, default=20.0): The assumed width of nebular emission
  lines, in km/s.
* ``table`` (optional, default=``data/nebular/nebular.h5``): Overrides the default
  nebular emission grid used to compute nebular contributions for this simulation's
  own track set.
