.. highlight:: rest

.. _sec-nebular:

Nebular Emission
================

SLUG can also compute the additional emission and absorption produced when the light from
a stellar population passes through a nebula. A detailed calculation of this process is
provided by the :ref:`sec-cloudy-slug` capability, which automates the process of
passing spectra calculated by SLUG through the 
`Cloudy <https://gitlab.nublado.org/cloudy/cloudy>`_ photoionization code. However,
because Cloudy simulations are very expensive (typically a few CPU minutes per spectrum
processed), SLUG also provides a "quick-and-dirty" estimate of nebular emission that
can be computed much faster.

SLUG's fast, approximate nebular emission calculation uses a grid of models computed
by Cloudy from the spectra of fully sampled stellar populations (i.e., computed with 
``min_stoch_mass`` set to a large value -- see :ref:`ssec-parameters-stars`) drawn from
the Chabrier (2005) IMF, for a nebula with a fixed density of
:math:`n = 100\,\mathrm{cm}^{-3}` and Solar-scaled abundances matching the [Fe/H] values
of the stellar tracks. This calculation is performed for both simple stellar populations
at a range of ages up to 10 Myr, and for a composite stellar population with a constant
star formation rate, for each of the track sets provided by the standard data files (see
:ref:`sec-tracks`). The script that generates the grid is included in the repository
as ``data/tools/cloudy/run_grid_pipeline.pbs``, which in turn calls other scripts in
the same directory.

The script computes the emission per unit ionizing photon in both the continuum and in
the hundred brightest emission lines and stores the results in an HDF5 table; the
default table in SLUG is ``data/nebular/nebular.h5``, but this choice can
be overridden if you wish to produce your own table (see :ref:`ssec-parameters-nebular`).
To produce its quick approximate stellar plus nebular emission spectrum, SLUG assumes
the HI-ionizing portion of the stellar spectrum is fully absorbed by the nebula, and
it scales the HI-ionizing photon flux by the recorded emission per ionizing photon
to produce a nebular emission spectrum that is added to the sub-ionizing stellar
spectrum.

The nebular emission calculation is controlled by three parameters in the 
:ref:`ssec-parameters-nebular` section of :ref:`sec-parameters`:

* ``compute_neb``: defaults to true; setting to false disables nebular computation entirely
* ``log_U``: the volume-averaged ionization parameter of the nebula; see :ref:`sec-cloudy-slug` for the precise definition of this parameter. Valid values are in the range -3 to -2.
* ``cov_fac``: the covering factor of the nebula, which in practice means the fraction of output ionizing photons that are reprocessed into nebular emission. Photons that are not reprocessed are assumed to be absorbed by dust grains, to be reprocessed outside the observational aperture, or to form part of the photon background responsible for producing diffuse ionized gas.
* ``line_width``: the width of the nebular emission lines that are inserted into the spectrum.

When nebular emission is enabled, SLUG writes both the intrinsic stellar spectrum and
the stellar plus nebular reprocessed spectrum to the output. If photometry is enabled
(see :ref:`sec-phot`), it also computes photometry on the stellar plus nebular spectrum.
See :ref:`sec-output` for details on how these data are stored in the output file, and
:ref:`sec-slugpy` for a description of how to access them using the slugpy reader.