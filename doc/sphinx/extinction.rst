.. highlight:: rest

.. _sec-extinction:

Extinction
==========

SLUG can apply extinction to a calculated spectrum. The extincted spectrum is computed
as

.. math:: L_{\lambda,\mathrm{ext}} = L_\lambda \exp\left(-A_V C_{\lambda,\mathrm{ext}}\right),

where :math:`L_{\lambda,\mathrm{ext}}` is the extincted spectrum, :math:`L_\lambda` is
the intrinsic spectrum, :math:`A_V` is the V-band extinction, and
:math:`C_{\lambda,\mathrm{ext}}` is a dimensionless extinction curve. The extinction
curve is normalized so that when :math:`A_V = 1`

.. math:: -2.5 \log_{10}\frac{\langle L_{\lambda,\mathrm{ext}} \rangle_V}{\langle L_{\lambda} \rangle_V} = 1\,\mathrm{mag},

where :math:`\langle\cdot\rangle_V` is the spectrum averaged over a Johnson V filter;
see :ref:`sec-phot` for the precise definition of the filter average. 

When extinction is enabled (by setting values of :math:`A_V` and the extinction curve)
as described below, the output will contain an extincted spectrum as well as the the
intrinsic spectrum. Note that the extincted spectrum may be truncated to a wavelength
grid that is narrower than the one used for the intrinsic spectrum if the extinction
curve selected does not cover the full wavelength range of the intrinsic spectrum.

If nebular emission is enabled (see :ref:`sec-nebular`) it will also contain spectra
and photometry computed on the nebular-reprocessed spectrum. Thus in the most general
case when both nebular reprocessing and extinction are enabled, SLUG will output four
versions of each spectrum: the intrinsic stellar spectrum, the spectrum after nebular
processing, the stellar spectrum with extinction applied but not nebular processing,
and the spectrum processed by the nebula and then with extinction applied to it. 
If photometry is enabled (see :ref:`sec-phot`), each output spectrum will also have a
corresponding set of photometric outputs computed. See :ref:`sec-output` for an
explanation of how these outputs are stored in the output file, and :ref:`sec-slugpy`
for an explanation of how to access them using the slugpy reader.

Setting the Extinction
----------------------

The values of :math:`A_V` used in the simulations are controlled by keywords in the
:ref:`ssec-parameters-extinct` section of :ref:`sec-parameters`. The keyword ``AV``
gives the extinction (or distribution of extinctions) for stars that are part of
clusters, while in galaxy simulations ``AV_field`` provides a separate distribution
of extinctions that is applied only to "field" stars that are not part of clusters.

Extinction Curves
-----------------

The ``model`` keyword in the :ref:`ssec-parameters-extinct` specifies the shape of the
extinction curve to apply. Available extinction curves are registered in a
`TOML <https://toml.io/en/>`_ registry file (defaults to ``data/extinct/extinct.toml``,
but this can be overridden by setting the ``registry`` keyword). The curves included
in the default data set are:

* ``Bouchet_SMC``: the Bouchet SMC extinction curve (Bouchet et al. 1985).
* ``Calzetti_LMC``: the Calzetti LMC extinction curve (Landini et al. 1984; Fitzpatrick
  1999; D. Calzetti, personal communication).
* ``Calzetti_MW``: the Calzetti Milky Way extinction curve (Landini et al. 1984;
  Fitzpatrick 1999; D. Calzetti, personal communication).
* ``Calzetti_starburst``: the Calzetti starburst attenuation curve (Calzetti et al. 2000).
* ``Draine_MW_RV3.1``: the Draine (2003) Milky Way extinction curve, for a total-to-selective
  extinction ratio :math:`R_V = 3.1` (Weingartner & Draine 2001; Li & Draine 2001; Draine
  2003a,b,c).
* ``Draine_MW_RV4.0``: the same Draine (2003) Milky Way extinction curve, for :math:`R_V = 4.0`.
* ``Draine_MW_RV5.5``: the same Draine (2003) Milky Way extinction curve, for :math:`R_V = 5.5`.

The full list of references for each curve, which should be cited in any publication
using it, is provided in the ``data/extinct/extinct.toml`` registry.

Adding New Extinction Curves
----------------------------

Users can add new extinction curves by adding to the default HDF5 file
(``data/extinct/extinct.h5``) and adding a corresponding registry entry in the
registry file, or by creating a new HDF5 and a corresponding registry.

The required format for the HDF5 file is as follows.

An extinction curve file contains one top-level HDF5 group per curve, named exactly
as the curve is named in the ``curves`` list of the registry entry described below
(e.g. a group named ``Calzetti_starburst``). Each such group must contain two 1D
datasets of equal length:

* ``wavelength``: the wavelength grid, in Angstrom, on which the curve is tabulated,
  sorted in order of increasing wavelength. A star (or, for a continuously-distributed
  population, a wavelength point) that falls outside this grid's own coverage is
  simply dropped from the extincted spectrum, rather than extrapolated -- see the
  note on truncation above.
* ``kappa``: the extinction curve itself, :math:`C_{\lambda,\mathrm{ext}}`, at each
  wavelength in ``wavelength``. Only the curve's overall *shape* matters: SLUG
  automatically rescales whatever is provided here so that it corresponds to
  :math:`A_V = 1` mag, as described above, so ``kappa`` may be tabulated in any
  convenient, even arbitrary, units or overall normalization.

The required format for the registry entry is as follows.

An extinction registry is a TOML file with a top-level ``file`` key (a string): the
path to the HDF5 file described above, resolved relative to the directory containing
the registry file itself, exactly as for a track, spectra, or filter registry's own
``file`` key (see :ref:`ssec-tracks-standard`). It must also have a top-level
``curves`` key (an array of strings), listing every curve name the registry
describes; the default registry, ``data/extinct/extinct.toml``, lists the standard
curves described above.

For every name appearing in ``curves``, the registry may (but is not required to)
also contain a top-level table of that same name (e.g. ``[Calzetti_starburst]``).
SLUG itself never reads anything from this table -- a curve's actual data always
come from the HDF5 file -- but the following keys are included in the standard
registry, and recommended for any new curve added in the same way, for
documentation and citation purposes:

* ``reference``: One literature reference (as a string) or several (as an array of
  strings) that should be cited by anyone using this curve.
* ``reference_url``: One URL string, or an array of them matching ``reference``,
  giving a link (e.g. an ADS abstract page) for each reference. A reference with no
  readily citable URL (e.g. a personal communication) may simply be omitted here.
* ``description``: A short, human-readable description of the curve.

The scripts used to generate the existing extinction curve data set are provided
in ``data/tools/extinct``, and these can serve as a guide.
