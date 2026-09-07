.. highlight:: rest

.. _sec-phot:

Photometry and Filters
======================

Once it has generated a spectrum for a stellar population (see :ref:`sec-atmospheres`),
SLUG can convolve that spectrum with filter response curves to produce predictions for
photometry. The fundamental calculation that SLUG carries out to compute photometry is
to evaluate the integral

.. math:: \left\langle L_\lambda \right\rangle_R = \frac{\int_{-\infty}^\infty L_\lambda R_\lambda \,\mathrm{d}\ln\lambda}{\int_{-\infty}^\infty R_\lambda \,\mathrm{d}\ln\lambda}

where :math:`L_\lambda` is the specific luminosity and :math:`R_\lambda` is the response
function of the filter, which describes the expected number of counts produced in the
detector per incident photon of wavelength :math:`\lambda`. Which filter response
function(s) are evaluated is determined by the ``filters`` keyword
in the :ref:`ssec-parameters-phot` section of :ref:`sec-parameters`. 
See :ref:`ssec-phot-filterdata` and :ref:`ssec-phot-idealfilters` for a list of
available filters in the standard data set, and :ref:`ssec-phot-adding-filters`
for instructions on how to add additional filter data.

Photometric Systems
-------------------

The fundamental integral of the spectrum :math:`L_\lambda` over a filter response
function :math:`R_\lambda` returns a mean specific luminosity over the filter in
the same units as :math:`L_\lambda`: energy per unit time per unit wavelength. SLUG
can automatically convert this output into a variety of other photometric systems,
as specified by the ``system`` keyword in the :ref:`ssec-parameters-phot` section
of :ref:`sec-parameters`. The allowed photometric systems are:

* ``Flambda``: the default, returns :math:`\left\langle L_\lambda \right\rangle_R` directly
* ``Fnu``: returns a specific luminosity :math:`\left\langle L_\nu \right\rangle_R` with units of energy per time per frequency, which is related to :math:`\left\langle L_\lambda \right\rangle_R` by :math:`\left\langle L_\nu \right\rangle_R = (\lambda_R^2/c) \left\langle L_\lambda \right\rangle_R`, where :math:`\lambda_R` is the pivot wavelength of the filter (see :ref:`ssec-phot-filterdata`)
* ``ST``: returns a magnitude in the ST system, which is related to :math:`\left\langle L_\lambda \right\rangle_R` by :math:`\mathrm{mag}_\mathrm{ST} = -2.5 \log_{10}[\left\langle L_\lambda \right\rangle_R / (4\pi d^2) / F_\mathrm{ST,0}]`, where :math:`d = 10` pc is the standard distance and :math:`F_\mathrm{ST,0} = 3.631\times 10^{-9}\,\mathrm{erg}/\mathrm{cm}^2/\mathrm{s}/\mathrm{Å}` is the zero point of the ST magnitude system 
* ``AB``: returns a magnitude in the AB system, which is related to :math:`\left\langle L_\nu \right\rangle_R` by :math:`\mathrm{mag}_\mathrm{AB} = -2.5 \log_{10}[\left\langle L_\nu \right\rangle_R / (4 \pi d^2) / F_\mathrm{AB,0}]`, where :math:`d = 10` pc is the standard distance and :math:`F_\mathrm{AB,0} = 3631.0\,\mathrm{Jy}` is the zero point of the AB magnitude system
* ``Vega``: returns a magnitude in the Vega system, which is computed from :math:`\left\langle L_\lambda \right\rangle_R` as :math:`\mathrm{mag}_\mathrm{Vega} = -2.5 \log_{10}[\left\langle L_\lambda \right\rangle_R / (4\pi d^2) / \langle F_\mathrm{Vega} \rangle_R]`, where :math:`\langle F_\mathrm{Vega}\rangle_R` is the flux of Vega convolved with the response function :math:`R_\lambda`. The Vega flux is computed on-the-fly from a reference Vega spectrum stored in the respository as ``data/spectra/vega.h5``.

:ref:`sec-slugpy` also contains a tool, ``phot_convert``, to convert between
photometric systems after a simulation has been run.

.. _ssec-phot-filterdata:

Filter Data
-----------

SLUG's default data includes response curves and pivot wavelengths for a standard set of
filters, which are taken from the
`SVO Filter Profile Service <https://svo2.cab.inta-csic.es/svo/theory/fps3/>`_. The
list of filters available is stored in a registry file, ``data/filters/filters.toml``,
and the data themselves are stored in ``data/filters/filters.h5``. SLUG follows the SVO
naming convention where a filter name takes the form `FACILITY.INSTRUMENT.FILTER`, where
`FACILITY` is the name of the facility for that filter (e.g., HST), `INSTRUMENT` is the
name of the instrument on that facility (e.g., ACS_WFC), and `FILTER` is the name of the
filter on that instrument (e.g., F435W). Thus to generate photometry in this filter, one
would include ``HST.ACS_WFC.F435W`` in the list of filters supplied by the ``filters``
keyword in the :ref:`ssec-parameters-phot` section of :ref:`sec-parameters`.

.. _ssec-phot-idealfilters:

Ideal and Special Filters
-------------------------

In addition to real astronomical filters, SLUG also provides "ideal" filters that can
be used to compute photometric or photometric-like quantities from spectra. The following
entries can be included in the filter list to produce these outputs:

* ``ideal_energy_X_Y``: a filter where the response function :math:`R_\lambda` is a perfect tophat function that is unity for wavelengths between X and Y (in Angstrom) and 0 elsewhere
* ``ideal_phot_X_Y``: a filter that measures the total number of photons per unit time evaluated over the wavelength range from X to Y, by evaluating the integral :math:`\int_X^Y L_\lambda / (hc/\lambda) \, \mathrm{d}\lambda`. For filters of this type, Y can be "inf", in which case the upper limit on the integral is taken to be infinity.
* ``Q(<Elem><IonStage>)``, where ``<Elem>`` is a the standard elemental symbol for some element (e.g., H, He) and ``<IonStage>`` is a Roman numeral giving the ionization stage of that element; this filter returns the number of photons emitter per unit time with energies high enough to ionize the given ion to one higher ionization state, e.g., "Q(HI)" gives the number of photons per second emitted capable of ionizing neutral hydrogen, "Q(CII)" gives the number of photons per second emitted capable of ionizing C+ to C++. This is equivalent to ``ideal_phot_0_Y`` with Y set to the wavelength that corresponds to the ionization potential of the ion in question.
* ``Lbol``: a "filter" that returnst the total bolometric luminosity of the stellar population; this is not computed by integrating a spectrum at all, and instead is computed directly from the output of the :ref:`sec-tracks`.

Outputs for ``ideal_energy_X_Y``-type filters use the same photometric system as real
filters, while outputs for all the other ideal filter types use units of photons per
time regardless of which photometric system is selected.

.. _ssec-phot-adding-filters:

Adding Additional Filters
-------------------------

SLUG's default data set contains all the filters hosted by SVO for the following facilities:
2MASS, CFHT, GALEX, HST, JWST, Paranal, SLOAN, SkyMapper, and Spitzer. It also provides the
"Generic" facility filters (which includes the standard Bessell, Cousins, Johnson, and Stromgren
filters). If you want to add fitlers from any other facility whose filter data are
hosted on SVO, the repository contains a script ``data/tools/filters/fetch_filter_vo.py``
that can be used to add additional filters to the ``data/filters/filters.h5`` filter
catalogue and add corresponding registry entries to ``data/filters/filters.toml`` so
that filters can be access by SLUG.

If you want to add filters from another source, you can do so by writing your own HDF5
file (or appending to the existing one) with a corresponding `TOML <https://toml.io/en/>`_
registry file (or entries in the existing registry file).

The required format for HDF5 files containing filter data is as follows.

A filter data file is organized as a three-level nested group hierarchy that
mirrors the facility/instrument/filter naming convention described above: one
top-level group per facility (e.g. ``HST``), containing one subgroup per
instrument on that facility (e.g. ``ACS_WFC``), containing one subgroup per
filter on that instrument (e.g. ``F435W``) -- so a given filter's data live at
the HDF5 path ``<FACILITY>/<INSTRUMENT>/<FILTER>``. The names given to these
groups are exactly the ``FACILITY``, ``INSTRUMENT``, and ``FILTER`` names used
to refer to the filter elsewhere (e.g. in the ``filters`` keyword of
:ref:`ssec-parameters-phot`), so they must match the corresponding names used
in the registry entry described below. Each filter's own group must contain
two 1D datasets of equal length:

* ``wavelength``: the wavelength grid, in Angstrom, on which the response
  function is tabulated. This need not be evenly spaced, but should be sorted
  in order of increasing wavelength.
* ``transmission``: the filter's response function :math:`R_\lambda` at each
  wavelength in ``wavelength`` -- the expected number of counts produced in
  the detector per incident photon, as described at the start of this section.

A filter group may also carry additional HDF5 attributes (e.g. the
``Description``, ``ZeroPoint``, and ``DetectorType`` attributes written by
``data/tools/filters/fetch_filter_vo.py`` for filters fetched from SVO); SLUG
itself never reads any of these, so they are optional and are included purely
for reference.

The required format for registry entries for filters is as follows.

Unlike a track or spectra registry, which lists one entry per named
track/atmosphere set, a single filter registry describes every facility,
instrument, and filter in one file, and only ever points to a single HDF5
file. The registry must have a top-level ``file`` key (a string): the path to
the HDF5 file described above, resolved relative to the directory containing
the registry file itself, exactly as for a track or spectra registry's own
``file`` key (see :ref:`ssec-tracks-standard`).

For every facility to be made available, the registry must also contain a
top-level table of that facility's name (e.g. ``[HST]``) with an
``instruments`` key (an array of strings, the names of that facility's
instruments). For every one of those instruments, the registry must in turn
contain a subtable (e.g. ``[HST.ACS_WFC]``) with a ``filters`` key (an array
of strings, the names of that instrument's filters). Finally, for every one
of those filters, the registry must contain a further subtable (e.g.
``[HST.ACS_WFC.F435W]``) with the following required key:

* ``wl_ref`` (a number): the filter's pivot wavelength, in Angstrom -- the
  :math:`\lambda_R` used to convert between :math:`\left\langle L_\lambda
  \right\rangle_R` and :math:`\left\langle L_\nu \right\rangle_R` for the
  ``Fnu`` photometric system (see the ``Fnu`` entry under `Photometric
  Systems`_ above). Unlike a filter's wavelength grid and response function,
  its pivot wavelength is read from this registry entry alone, not from the
  HDF5 file.

A filter's subtable may also include any of the following optional keys,
which SLUG itself never reads but which are included in the standard registry
(as written by ``fetch_filter_vo.py``) for documentation purposes:

* ``description``: A short, human-readable description of the filter.
* ``source``: A URL identifying where this filter's data came from.

A registry may likewise include top-level ``name`` and ``Facilities`` keys
(the latter listing every facility the registry describes); SLUG never reads
either of these either, since it always looks a facility up by name directly,
but both are included in the standard registry for documentation purposes.