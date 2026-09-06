.. highlight:: rest

.. _sec-atmospheres:

Stellar Atmospheres and Spectral Synthesis
==========================================

A second step is most SLUG simulations is, after using the :ref:`sec-tracks` to
generate the properties of every star, to use those properties as inputs to a
set of stellar atmosphere models to predict the stars' specific luminosities
per unit wavelength :math:`L_\lambda`.

The behavior of stellar atmosphere models in SLUG is controlled by the ``model``
keyword in the :ref:`ssec-parameters-specsyn` section of :ref:`sec-parameters`,
which specifies which stellar atmosphere model or set of models to use. The
most common choice will be ``model = "default"``, which selects a combination of
atmosphere models from the standard data set (see :ref:`sec-getting`) with
broad coverage of the stellar properties that matter for spectral synthesis:
effective temperature :math:`T_\mathrm{eff}`, log surface gravity :math:`\log g`,
iron metallicity [Fe/H], and, for some evolved stars, surface composition and
stellar wind mass flux. See :ref:`ssec-atmospheres-default` for more details on
the default atmosphere model, :ref:`ssec-atmospheres-alternative` for details
on how to select other models from the available list, and
:ref:`ssec-atmospheres-adding` for details on how to add new atmosphere models.

Spectral synthesis is also affected by the properties of the stars, as controlled
by the ``alphaFe`` and ``CFe`` keywords in the :ref:`ssec-parameters-stars` section
of :ref:`sec-parameters`. These parameters set the [alpha/Fe] and [C/Fe] abundances
of the stellar atmopsheres, for the subset of model atmosphere libraries that have
models available for a range of [alpha/Fe] and [C/Fe] values.

In addition to setting the model atmosphere, users can control the wavelength grid
over which spectra are computed using the ``wl_min``, ``wl_max``, and ``nwl`` keywords
in the :ref:`ssec-parameters-specsyn` section of :ref:`sec-parameters`, and can set
a redshift ``z``. If the redshift is non-zero, the wavelengths specfied by the
``wl_min``, ``wl_max``, and ``nwl`` keywords are interpreted as describing the
wavelength grid in the *comoving* frame of the emitting stellar population, and
the output spectrum will be written in this comoving frame -- the ``z`` does not
affect the output spectra. However, it does affect :ref:`sec-photometry`, which
is computed using the wavelength grid redshifted to the *observed* frame.

.. _ssec-atmospheres-default:

The Default Atmosphere Model
----------------------------

The default atmosphere model in SLUG categorizes stars into three types: Wolf-Rayet
stars, white dwarf stars, and all other stars ("normal" stars). The precise criteria
used for these categories are as follows.

A star is first tested for being a Wolf-Rayet star. Only stars with a current mass
above :math:`5\,\mathrm{M}_\odot` are ever considered as candidates. Among these, a
star whose surface helium mass fraction lies between 0.4 and 0.9 is a candidate for
one of three hydrogen-rich (WNL) subtypes, selected by its surface hydrogen mass
fraction (below 0.3, between 0.3 and 0.5, or above 0.5, corresponding to the WNL-H20,
WNL-H40, and WNL-H60 subtypes respectively); it is actually classified into that
subtype only if its :math:`\log T_\mathrm{eff}` also falls within the range actually
covered by the chained atmosphere models available for that subtype (see
:ref:`ssec-atmospheres-alternative`), so that a star whose composition merely
resembles a WNL star, but whose temperature no available WNL grid covers, falls
through to the next test rather than being forced into an ill-fitting classification.
A star that is not classified as WNL by this test, but whose :math:`T_\mathrm{eff}`
exceeds 50000 K, is instead classified as a hydrogen-poor Wolf-Rayet star, of subtype
WNE if its surface carbon mass fraction is less than its surface nitrogen mass
fraction, or WC otherwise. A star failing every one of these tests is not a
Wolf-Rayet star.

A star that is not a Wolf-Rayet star is next tested for being a white dwarf. This
classification applies only if the star's (:math:`T_\mathrm{eff}`, :math:`\log g`)
falls above the coverage provided by the available normal-star atmosphere models
along at least one of the two axes, while at the same time remaining within reach of
the floor of the coverage provided by the available white-dwarf atmosphere models
along *both* axes. This two-sided test ensures that a star is only routed to the
white dwarf atmosphere models when there is a genuine gap in the normal-star
atmosphere models' own coverage that the white dwarf models can plausibly fill,
rather than routing every hot, compact star there regardless of whether the normal
atmosphere models already cover it.

Every other star -- the vast majority of them -- is treated as a normal star.

Once a star has been catagorized, it is routed to the stellar atmosphere models that
cover that type of star. These are applied in order, with stars that are not covered
by one set of atmospheres being passed on to the next and so forth; the code is
structured this way because published model atmosphere libraries generally specialize
to one particular type of star (e.g., O and B stars, evolved cool giants, main sequence
stars, etc.). Thus a star is tried against atmospheres until one that fits it is found.

The atmosphere models in the default set for WR stars are:

* ``POWR_WC``: the carbon-sequence (WC, with WO and WNC folded in) grid from the
  `Potsdam Wolf-Rayet (PoWR) <https://www.astro.physik.uni-potsdam.de/PoWR/>`_ model
  atmosphere group. Covers :math:`T_\mathrm{eff} \approx 39800-199500\,\mathrm{K}`,
  parameterized by transformed radius :math:`\log R_t = -0.80` to :math:`1.60` rather
  than :math:`\log g` (Wolf-Rayet winds are optically thick, so :math:`\log g` is not
  a meaningful atmosphere parameter), and [Fe/H] :math:`= -1.24` to :math:`0` (4
  values, spanning roughly Galactic through sub-SMC metallicity).
* ``POWR_WNE``: the hydrogen-poor nitrogen-sequence (WNE) grid from PoWR. Covers
  :math:`T_\mathrm{eff} \approx 31600-199500\,\mathrm{K}`,
  :math:`\log R_t = -0.20` to :math:`1.90`, and [Fe/H] :math:`= -1.18` to :math:`0`
  (4 values).
* ``POWR_WNL_H20``, ``POWR_WNL_H40``, ``POWR_WNL_H60``: the hydrogen-rich
  nitrogen-sequence (WNL) grids from PoWR, split into three separate files by surface
  hydrogen mass fraction bucket (approximately 0.20, 0.40, and 0.60 respectively),
  following the Roy et al. (2020) WNL classification. Each covers
  :math:`T_\mathrm{eff} \approx 25100-112200\,\mathrm{K}`,
  :math:`\log R_t = 0.0` to :math:`1.90`, and [Fe/H] :math:`= -1.18` to :math:`0`.

The atmosphere models in the default set for normal stars are:

* ``TLUSTY_O``: the non-LTE `TLUSTY <https://stsci.app.box.com/v/tlustyOB2025>`_
  OSTAR grid (Hubeny et al. 2025) for O stars. Covers
  :math:`T_\mathrm{eff} = 27500-55000\,\mathrm{K}`, :math:`\log g = 3.0-4.75`, and
  [Fe/H] :math:`= -2.0` to :math:`0.30` (8 values), at a fixed microturbulent
  velocity of 10 km/s.
* ``TLUSTY_B``: the companion non-LTE `TLUSTY
  <https://stsci.app.box.com/v/tlustyOB2025>`_ BSTAR grid (Hubeny et al. 2025) for B
  stars. Covers :math:`T_\mathrm{eff} = 15000-30000\,\mathrm{K}`,
  :math:`\log g = 1.75-4.75`, and [Fe/H] :math:`= -2.0` to :math:`0.30` (8 values),
  at a choice of 2, 5, or 10 km/s microturbulent velocity (2 km/s by default).
* ``BOSZ``: the `BOSZ <https://archive.stsci.edu/hlsps/bosz/bosz2024/>`_ synthetic
  spectral library (Mészáros et al. 2012, 2024; Bohlin et al. 2017), built from a
  combination of ATLAS9 and MARCS model atmospheres. Covers
  :math:`T_\mathrm{eff} = 2800-16000\,\mathrm{K}`, :math:`\log g = -0.5` to
  :math:`5.5`, [Fe/H] :math:`= -2.5` to :math:`0.75`, and [alpha/Fe] :math:`= -0.25`
  to :math:`0.50`, at fixed [C/Fe] :math:`= 0`, spectral resolution :math:`R = 500`,
  and microturbulent velocity 0 km/s.
* ``CK04``: the Castelli & Kurucz (2004) ATLAS9 grid, as distributed via the `STScI
  CDBS archive
  <https://archive.stsci.edu/hlsps/reference-atlases/cdbs/grid/ck04models/>`_.
  Covers :math:`T_\mathrm{eff} = 3500-50000\,\mathrm{K}`, :math:`\log g = 0.0-5.0`,
  and [Fe/H] :math:`= -2.5` to :math:`0.5` (8 values); this grid has no
  [alpha/Fe], [C/Fe], or microturbulence axis.
* ``MARCS``: the `MARCS <https://marcs.astro.uu.se/data.html>`_ grid of spherical
  and plane-parallel LTE model atmospheres (Gustafsson et al. 2008). Covers
  :math:`T_\mathrm{eff} = 2500-8000\,\mathrm{K}`, :math:`\log g = -0.5` to
  :math:`5.5`, [Fe/H] :math:`= -5.0` to :math:`1.0` (15 values), and [alpha/Fe]
  :math:`= -0.4` to :math:`0.4` (6 values), at fixed [C/Fe] :math:`= 0` and a choice
  of 0, 1, 2, or 5 km/s microturbulent velocity (2 km/s by default).

The atmosphere models in the default set for white dwarf stars are:

* ``TREMBLAY_DA``: the pure-hydrogen (DA) white dwarf atmosphere grid of Tremblay
  et al. (2011), including the Kowalski & Saumon (2006) high-density correction, from
  the `Warwick model grid page
  <https://warwick.ac.uk/fac/sci/physics/research/astro/people/tremblay/modelgrids/grid_ir.tar>`_.
  Covers :math:`T_\mathrm{eff} = 1500-140000\,\mathrm{K}` and
  :math:`\log g = 6.5-9.5`; white dwarf atmospheres have no [Fe/H], [alpha/Fe], or
  [C/Fe] axis at all, since they are parameterized by :math:`T_\mathrm{eff}` and
  :math:`\log g` alone.
* ``TREMBLAY_ELM``: the companion extremely-low-mass (ELM) pure-hydrogen white dwarf
  grid of Claret et al. (2020), from the `same Warwick model grid page
  <https://warwick.ac.uk/fac/sci/physics/research/astro/people/tremblay/modelgrids/grid_elm_new.tar.gz>`_.
  Covers :math:`T_\mathrm{eff} = 4000-40000\,\mathrm{K}` and
  :math:`\log g = 4.0-9.5`.
* ``RAUCH``: the solar-abundance NLTE hot subdwarf / pre-white-dwarf (central-star-
  of-planetary-nebula) atmosphere grid of Rauch (2003), queried from the `TheoSSA
  <https://dc.g-vo.org/browse/theossa/q>`_ virtual observatory service operated by
  the German Astrophysical Virtual Observatory. Covers (non-rectangularly)
  :math:`T_\mathrm{eff} = 50000-190000\,\mathrm{K}` and :math:`\log g = 5-8`.
* ``RAUCH_H07``: a companion pure H/He grid (no metal line blanketing) from the same
  `TheoSSA <https://dc.g-vo.org/browse/theossa/q>`_ source, filtered to the same
  5-2000 Angstrom wavelength range as ``RAUCH``. Covers (non-rectangularly) the same
  :math:`T_\mathrm{eff} = 50000-190000\,\mathrm{K}` range as ``RAUCH``, but extends to
  higher :math:`\log g = 5-9`, filling a gap between ``RAUCH``'s own
  :math:`\log g \le 8` ceiling and the Tremblay grids' own :math:`T_\mathrm{eff}`
  ceilings. Because it lacks metal line blanketing, it is less physically faithful
  than ``RAUCH``, so it is listed after ``RAUCH`` in the default chain and is only
  ever used for a star ``RAUCH`` cannot cover.

The full list of references for each set of atmospheres, which should be cited in any
publication using that track set, is provided in the ``data/spectra/spectra.toml``
registry.

.. _ssec-atmospheres-alternative:

Alternative Atmosphere Models
-----------------------------

Users are free to use either individual stellar atmosphere libraries
or arbitrary combinations of them setting the ``model`` keyword to one or more
names of stellar atmosphere models. The list of available models is provided
in a registry file, the name of which is provided by the ``registry`` keyword
in the ``[specsyn]`` section of :ref:`sec-parameters`; the default registry
used by the standard data set is ``data/spectra/spectra.toml``. If a single model name
is provided in ``model``, only that atmosphere set will be used, while if
multiple names are provided SLUG will try the specified models in order, by
checking for each star if its properties are within the range covered by the
first set of models, if not trying the second set, and so forth.

However, be warned that the SLUG will exit with an error if it encounters a
star whose properties are not within the coverage of any of the provided models; the
"default" model set has been assembled carefully to ensure that it covers 
almost all of the possible stellar properties that can be produced by
:ref:`ssec-tracks-standard`, with clamping applied to some extreme, rare cases
that fall outside the coverage grid (e.g., very compact, high :math:`\log g`
O and B stars that can be produced at the lowest [Fe/H], and which are outside
the range covered by any of the available atmosphere models). However, this
will in general not be the case for any other model grid. Therefore it is
recommended to use options other than ``model = "default"`` only for simulations
exploring a limited range of stellar masses or ages, or for users who are
prepared to do a considerable amount of hand-tuning to ensure adequate coverage.

Finally, users have the option to set ``model = "blackbody"``, in which case stars
will be treated as blackbodies at the effective temperature provided by the
tracks. This option is intended for code testing *only*, and should not be used in
production simulations.

.. _ssec-atmospheres-adding:

Adding New Stellar Atmosphere Models
------------------------------------

As with :ref:`sec-tracks`, it is possible to add new atmosphere models to SLUG
without altering the source code, just by adding an HDF5 file containing the
atmosphere data and a registry entry in a `TOML <https://toml.io/en/>`_ registry
file -- either a new entry in the standard registry file ``data/spectra/spectra.toml``,
or a new registry file specified by the ``registry`` keyword in the
:ref:`ssec-parameters-specsyn` section of :ref:`sec-parameters`.

The required format described below is the one used by the ordinary ("normal star")
atmosphere models -- e.g. ``BOSZ``, ``CK04``, ``MARCS``, ``TLUSTY_O``, ``TLUSTY_B`` --
and is the format to follow when adding a new model of this kind. The Wolf-Rayet and
white dwarf atmosphere models use their own, more specialized HDF5 layouts (selected
by the ``WR_grid`` and ``WD_grid`` registry keys described below), which are not
documented here; the scripts referenced at the end of this section can be used as a
guide to those formats as well.

The required format for the HDF5 file is as follows.

A spectral library file contains one top-level group named ``wavelengths``, which
holds one or more 1D datasets giving the wavelength grid(s), in Angstrom and sorted
ascending, on which the library's spectra are tabulated. If the library provides
spectra at more than one internal spectral resolution ``r``, each such dataset
should be named ``r<r>``, where ``<r>`` is that resolution rounded to the nearest
integer (e.g. ``r500``); if the library provides only a single wavelength grid, that
one dataset may be given any name, and it will be used automatically regardless of
``r``. SLUG does not currently expose any input deck keyword to select among
multiple resolutions -- every spectral synthesizer is built with a single, fixed
internal ``r`` (see ``defaultR`` in ``SpecsynCommons.hpp``) -- so this axis exists
in the file format for future extensibility, but is not presently a way for a user
to distinguish between resolutions at run time.

Beyond ``wavelengths``, the file contains one top-level group per available
combination of [Fe/H] and, optionally, [alpha/Fe], [C/Fe], microturbulent velocity,
and spectral resolution; the name given to each such group is arbitrary and is never
interpreted by SLUG. Each such group must have:

* An attribute ``feh`` (a single float): the [Fe/H] value of this group's models.
  This attribute is required.
* An attribute ``afe`` (a single float): the [alpha/Fe] value of this group's
  models. This attribute is optional; if it is absent, this group is treated as
  matching *any* requested [alpha/Fe] value, appropriate for a library (like
  ``TLUSTY_O`` or ``CK04``) that does not provide multiple [alpha/Fe] choices.
* An attribute ``cfe`` (a single float): the [C/Fe] value of this group's models.
  This attribute is optional, with the same "matches anything" behavior as ``afe``
  if it is absent.
* An attribute ``micro`` (a single float, in km/s): the microturbulent velocity of
  this group's models. This attribute is optional, with the same "matches anything"
  behavior as ``afe`` if it is absent -- appropriate for a library (like ``CK04``)
  with no microturbulence axis at all.
* An attribute ``r`` (a single float): the spectral resolution of this group's
  models. This attribute is optional, with the same "matches anything" behavior as
  ``afe`` if it is absent.

Within each such group, there is one dataset per (:math:`T_\mathrm{eff}`,
:math:`\log g`) point actually available at that group's own [Fe/H]/[alpha/Fe]/
[C/Fe]/microturbulence/resolution combination; the name given to each such dataset
is likewise arbitrary. These points need not form a complete rectangular grid --
gaps are permitted, and SLUG simply treats an unpopulated point as unavailable when
interpolating (routing a star that needs it to the next atmosphere model in the
chain, if one is available; see :ref:`ssec-atmospheres-alternative`). Each such
dataset must have:

* An attribute ``teff`` (a single float, in K -- *not* :math:`\log_{10} T_\mathrm{eff}`):
  the effective temperature of this model. This attribute is required.
* An attribute ``logg`` (a single float): :math:`\log_{10}` of this model's surface
  gravity, with :math:`g` in cgs units (:math:`\mathrm{cm\,s^{-2}}`). This attribute
  is required.

and must hold a 1D array, of the same length as the group's own wavelength grid
(``wavelengths/r<r>``, or the file's sole wavelength dataset if there is only one),
giving the model's surface flux density :math:`F_\lambda`, in
:math:`\mathrm{erg\,s}^{-1}\,\mathrm{cm}^{-2}\,\mathrm{Angstrom}^{-1}`, at each
wavelength -- that is, the flux emitted per unit area of the stellar surface, *not*
the flux already scaled by any particular star's actual surface area or by distance.
SLUG multiplies this by the surface area of the star actually being synthesized
(computed from its luminosity and effective temperature) to obtain its specific
luminosity.

The required format for the registry entry is as follows.

A spectra registry is a TOML file with a top-level ``spectra_sets`` key giving an
array of strings, one per atmosphere model the registry describes; the default
registry, ``data/spectra/spectra.toml``, lists the standard atmosphere models
described above. For every name appearing in ``spectra_sets``, the registry must
also contain a top-level table of that same name (e.g. ``[BOSZ]``) with at least the
following key:

* ``file`` (a string): The path to this model's HDF5 file, resolved relative to the
  directory containing the registry file itself (not the current working
  directory), exactly as for a track registry's own ``file`` key (see
  :ref:`ssec-tracks-standard`); this may be a relative path containing ``..``
  components, not just a bare file name.

Unlike a track registry, a spectra registry entry has no other strictly required
key -- in particular, an [Fe/H] value is always read directly from each HDF5 group's
own ``feh`` attribute at runtime, so nothing here needs to advertise it up front
(though the optional ``Fe_H`` key below is still recommended, for documentation
purposes, exactly as for a track registry).

A registry entry may also include any of the following optional keys:

* ``version``: A version string for this atmosphere model.
* ``references``: An array of strings, one full citation per publication that
  should be cited by anyone using this atmosphere model.
* ``reference_urls``: An array of URL strings, one per entry of ``references``, in
  the same order.
* ``Fe_H``: An array of numbers, the [Fe/H] values this model provides. SLUG itself
  never reads this key (unlike the corresponding key in a track registry); it exists
  purely to let a user discover a model's own metallicity coverage without opening
  its HDF5 file.
* ``alpha_Fe``, ``C_Fe``: Arrays of numbers, the [alpha/Fe] and [C/Fe] values this
  model provides (omitted for a model, like ``CK04``, that does not vary along
  either axis).
* ``micro``: An array of numbers, the microturbulent velocities (in km/s) this
  model provides (omitted for a model with no microturbulence axis at all).
* ``micro_default``: A single number, the microturbulent velocity (in km/s) SLUG
  uses for this model -- there is currently no input deck keyword to override it
  with a different value. Meaningful only for a model that has a microturbulence
  axis in the first place.
* ``r``: An array of numbers, the spectral resolutions this model provides.
* ``download_urls``: An array of URL strings pointing to the original, unprocessed
  data this model was built from.
* ``WR_grid`` (a boolean): If ``true``, this entry describes a Wolf-Rayet
  atmosphere grid, to be read using the specialized format used internally by the
  ``SpecsynLibWR`` class rather than the general format described above. Mutually
  exclusive with ``WD_grid``.
* ``WD_grid`` (a boolean): If ``true``, this entry describes a white dwarf
  atmosphere grid, to be read using the specialized format used internally by the
  ``SpecsynLibWD`` class rather than the general format described above. Mutually
  exclusive with ``WR_grid``.

The scripts used to generate the HDF5 files for all the atmosphere models in the
standard data set are provided in the ``data/tools/spectra`` directory of the
repository, and can be used as guides.