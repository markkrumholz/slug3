.. highlight:: rest

.. _sec-tracks:

Stellar Tracks
==============

The first ingredient in any SLUG simulation is a set of stellar tracks that describe
the properties of stars with a specified initial mass and other properties 
(e.g., metallicity, rotation rate) as a function of age. 

SLUG refers to a collection of evolutionary tracks computed by a particular group or with
a particular code as a track set; generally a single track set will include models
generated with a range of parameters; the ones that SLUG knows about and can use are:

* :math:`v/v_\mathrm{crit}`: initial rotation speed relative to the critical speed
* [Fe/H]: :math:`\log_{10}` of the iron to hydrogen ratio normalized to Solar
* [alpha/Fe]: :math:`\log_{10}` of the :math:`\alpha` to iron ratio normalized to Solar

The choice of track set to use, and the values of each of these parameters,
is controlled by the :ref:`ssec-parameters-stars` in the input deck. The
set of available tracks is listed in a registry file specified by the ``registry``
keyword in the ``[stars]`` section of :ref:`sec-parameters`. The values of
the parameters ``v_vcrit``, ``FeH``, and ``alphaFe``, which control :math:`v/v_\mathrm{crit}`,
[Fe/H], and [alpha/Fe], are permitted to lie anywhere within the range covered by
the track set, and need not exactly match one of the values provided; for values
that are intermediate between grid points, SLUG will automatically interpolate.
Stellar masses need not fall within the range covered by a given set of tracks
(i.e., the IMF can include masses that are outside the track range), but such stars
will be treated as having zero luminosity, yield, or feedback power, and SLUG will
print a warning when running if it detects that the IMF extends outside the range
for which the tracks provide coverage.

.. _ssec-tracks-standard:

Standard Track Sets
-------------------

The list of track sets available for use in SLUG are listed in a registry file. 
The default registry, which is provided as part of the standard data download
(see :ref:`sec-getting`), is ``data/tracks/tracks.toml``. The registry is a
human-readable description of the availalble track sets, with the set of available
track sets listed in the ``track_sets`` keyword at the top-level of the file. 

The SLUG standard data files provide the following track sets:

* ``MIST``: version 2.5 of the `Mesa Isochrones and Stellar Tracks <https://mist.science/>`_ tracks. These tracks are available for stars with masses of :math:`0.1 - 300\,\mathrm{M}_\odot` at :math:`v/v_\mathrm{crit} = 0` and 0.4, [Fe/H] = -4.0 to 0.5, and [alpha/Fe] = -0.2 to 0.6.
* ``PARSEC_vms``: version 2.0 of the `Padova Trieste Stellar Evolution Code <https://stev.oapd.inaf.it/PARSEC/>`_ intermediate-to-massive star tracks. These tracks are available for stars with masses of :math:`2 - 600\,\mathrm{M}_\odot` at :math:`v/v_\mathrm{crit} = 0`, [Fe/H] = -9.18 to 0.29, and [alpha/Fe] = 0. Since these tracks do not include rotation, the ``v_vcrit`` parameter is ignored if this track set is selected.
* ``PARSEC_rot``: version 2.0 of the `Padova Trieste Stellar Evolution Code <https://stev.oapd.inaf.it/PARSEC/>`_ low-to-intermediate mass rotating tracks. These tracks are available for stars with masses of :math:`0.09 - 14\,\mathrm{M}_\odot` (though the coverage at lower masses is not the same for all combinations of :math:`v/v_\mathrm{crit}` and [Fe/H]) at :math:`v/v_\mathrm{crit} = 0 - 0.99`, [Fe/H] = -2.18 to 0.29, and [alpha/Fe] = 0.
* ``PARSEC_comp``: a composite set of tracks made by combining the low-mass, :math:`v/v_\mathrm{crit} = 0` tracks from ``PARSEC_rot`` with the higher-mass tracks from ``PARSEC_vms``, in order to provide more complete mass coverage. These tracks span :math:`0.09 - 600\,\mathrm{M}_\odot` at :math:`v/v_\mathrm{crit} = 0`, [Fe/H] = -2.18 to 0.29, and [alpha/Fe] = 0.
* ``Stromlo``: the `Stromlo Stellar Tracks <https://sites.google.com/view/stromlotracks>`_. These tracks are available for stars with masses of :math:`10-300\,\mathrm{M}_\odot` at :math:`v/v_\mathrm{crit} = 0 - 0.4` and [Fe/H] = -2.5 to 0.5. The value of [alpha/Fe] is not meaningful for these tracks because their abundances are set using an empirically-determined scaling that produces a complex, non-linear dependence of alpha on Fe. The ``alphaFe`` parameter is therefore ignored for this track set.

The full list of references for each set of tracks, which should be cited in any
publication using that track set, is provided in the ``data/tracks/tracks.toml``
registry.

Browsing the Tracks
-------------------

The :ref:`sec-slugpy` module provides two convenient tools to browse the tracks: the
``compute_tracks`` and ``compute_isochrones`` routines. These provide a high-level
interface to compute sets of stellar properties as a function of time for stars of
specified initial mass (tracks) or as a function of mass for stars at a specified
age (isochrones). See :ref:`ssec-slugpy-full` for full documentation of these tools. 

Adding New Track Sets
---------------------

Users are free to add additional track sets to SLUG, and do so without modifying the
source code. All that is required is to create an HDF5 file containing the track
information that follows the layout specification described below, and then to
register that set of tracks in a `TOML <https://toml.io/en/>`_ registry, either by
adding a new entry to the standard registry or by making a new registry file -- the
choice of registry file can be controlled by the :ref:`ssec-parameters-stars` in
the input deck.

The required format for the HDF5 track set file is as follows.

A track set file contains one HDF5 group per combination of [Fe/H],
:math:`v/v_\mathrm{crit}`, and [alpha/Fe] at which tracks are available; the name
given to each such group is arbitrary and is never interpreted by SLUG, except
that the name ``masses`` is reserved and must not be used for a track group (a
top-level link with that name is always skipped when SLUG searches the file for
track groups). Each track group must have:

* An attribute ``feh`` (a single float): the [Fe/H] value of this group's tracks.
  This attribute is required.
* An attribute ``vvcrit`` (a single float): the :math:`v/v_\mathrm{crit}` value of
  this group's tracks. This attribute is optional; if it is absent, this group is
  treated as matching *any* requested :math:`v/v_\mathrm{crit}` value, which is
  appropriate for a track set (like ``PARSEC_vms``) that provides only
  non-rotating tracks and so has no need to distinguish between rotation rates.
* An attribute ``afe`` (a single float): the [alpha/Fe] value of this group's
  tracks. This attribute is optional, with the same "matches anything" behavior
  as ``vvcrit`` if it is absent -- appropriate for a track set (like ``Stromlo``)
  whose abundances are not parameterized by a single [alpha/Fe] value at all.
* A dataset ``masses`` (a 1D array of length :math:`n_\mathrm{mass}`): the
  zero-age (initial) mass, in :math:`\mathrm{M}_\odot`, of every track in this
  group. The masses need not be listed in sorted order -- SLUG sorts them
  internally -- but if tracks at more than one [Fe/H] (i.e. more than one group)
  are to be used together in a single simulation, every one of those groups must
  provide the exact same set of masses, since SLUG interpolates over a single,
  shared mass grid across the whole [Fe/H] range it loads.
* An attribute ``field_names`` (a 1D array of strings, of the same length as the
  number of columns in every ``track_m*`` dataset described below): the name of
  the physical quantity stored in each column, in column order. This list must
  include an entry named ``age``, plus one entry each named ``mass``, ``mdot``,
  ``log_L``, ``log_Teff``, ``h_surf``, ``he_surf``, ``c_surf``, ``n_surf``, and
  ``o_surf`` -- in any order, not necessarily the order just given -- with the
  meanings described in the table below. Additional columns with other names may
  also be present; SLUG reads them (to confirm the dataset's column count matches
  ``field_names``) but otherwise ignores them.
* One dataset ``track_m<mass>`` for every entry in ``masses``, where ``<mass>``
  is that entry's own mass value formatted with *exactly* three digits after the
  decimal point (e.g. the track for a :math:`1.5\,\mathrm{M}_\odot` star is named
  ``track_m1.500``, and the track for a :math:`12\,\mathrm{M}_\odot` star is
  named ``track_m12.000``). Each such dataset is a 2D array with one row per
  time point (which should be sorted in order of increasing age, though this is
  not checked) and one column per entry of ``field_names``, in the same column
  order. Different masses -- and, if multiple [Fe/H] groups are used together,
  different [Fe/H] values -- are not required to have the same number of time
  points; SLUG pads any track shorter than the longest one actually loaded by
  holding its final row's values fixed for all later times.

The physical quantities named in ``field_names`` have the following required
meanings and units:

* ``age``: Stellar age, in yr. Used as the time coordinate; not itself stored as
  one of the interpolated quantities described below.
* ``mass``: The star's current mass at this age, in :math:`\mathrm{M}_\odot`
  (as opposed to the ``masses`` dataset's own zero-age mass for this track).
* ``mdot``: The star's mass-loss rate at this age, in
  :math:`\mathrm{M}_\odot\,\mathrm{yr}^{-1}`.
* ``log_L``: :math:`\log_{10}` of the star's bolometric luminosity, in
  :math:`\mathrm{L}_\odot`.
* ``log_Teff``: :math:`\log_{10}` of the star's effective temperature, in K.
* ``h_surf``, ``he_surf``, ``c_surf``, ``n_surf``, ``o_surf``: The surface mass
  fraction (dimensionless) of H, He, C, N, and O respectively.

The required format for a registry entry is as follows.

A track registry is a TOML file with a top-level ``track_sets`` key giving an
array of strings, one per track set the registry describes; the default
registry, ``data/tracks/tracks.toml``, lists the standard track sets described
above. For every name appearing in ``track_sets``, the registry must also
contain a top-level table of that same name (e.g. ``[MIST]``) with at least the
following two keys:

* ``file`` (a string): The path to this track set's HDF5 file, resolved
  relative to the directory containing the registry file itself (not the
  current working directory), so that a registry and the track files it
  describes can be moved together as a unit. This may be a relative path
  containing ``..`` components (e.g. ``"../tracks/mist.h5"``), not just a bare
  file name.
* ``Fe_H`` (an array of numbers): The list of [Fe/H] values this track set
  actually provides. SLUG requires this key to be present -- registering a
  track set without it raises an error before the corresponding HDF5 file is
  even opened -- but does not otherwise use its contents itself: the [Fe/H]
  value SLUG actually selects a track from is always read directly from each
  HDF5 group's own ``feh`` attribute at runtime, so this key exists purely to
  let a user (or slugpy) discover a track set's own metallicity coverage
  without opening the HDF5 file at all.

A registry entry may also include any of the following optional keys, which
SLUG itself never reads but which are included in the standard registry for
documentation and citation purposes, and are recommended for any new track set
added in the same way:

* ``version``: A version string for this track set.
* ``references``: An array of strings, one full citation per publication that
  should be cited by anyone using this track set.
* ``reference_urls``: An array of URL strings, one per entry of ``references``,
  in the same order.
* ``alpha_Fe``: An array of numbers, the [alpha/Fe] values this track set
  provides (omitted for a track set, like ``Stromlo``, whose abundances are not
  parameterized by [alpha/Fe] at all).
* ``v_vcrit``: An array of numbers, the :math:`v/v_\mathrm{crit}` values this
  track set provides.
* ``download_urls``: An array of URL strings pointing to the original,
  unprocessed data this track set was built from.

Finally, a registry file may include a top-level ``name`` key (a string), a
human-readable title for the registry as a whole; like the per-entry optional
keys above, this is purely descriptive and is never read by SLUG itself.

Examples of how to add tracks are provided by the scripts in the ``data/tools/tracks``
directory; the scripts there were the ones used to populate the standard tracks set.

Notes on the Alterations to the Standard Tracks
-----------------------------------------------

The standard track sets as provided on the URLs referenced above do not fully
conform to the data format structure required by SLUG. Some of these problems are
due to data defects, for example models that failed to run or crashed, leaving
tracks that are incomplete or missing, or where the times in a single track are
not strictly increasing due to data processing errors. Others are cases
where published track sets include a single model at one set of parameters rather
than a regular grid suitable for interpolation. Finally, some alterations have
been made to the published track sets for computational efficiency, for example
reducing the sampling density for published tracks with very large numbers of
time points, or removing extended sections of tracks covering the white dwarf
phase of stellar evolution where the light output is negligible on a stellar
population level. The full set of steps taken to process the standard track
sets and produce SLUG's version of them is documented in ``data/tools/tracks/README.md``
in the repository.
